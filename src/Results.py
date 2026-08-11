"""Result-format aggregation — turns the per-device reports the server collected
into the exact lines of the shared result format (guide/01-result-format.md), plus
the end-of-run archiver (guide/05-archiving.md).

Naming scheme: this project uses the **cluster** set of filenames
(`fps_cluster_ns.log`, `fps_cluster.log`, `utilization_cluster.log`,
`latency_cluster.log`) — the domain already calls a group a cluster
(`intermediate_queue_k`, src/Clustering.py). One scheme per project, never mixed
(01 §2), so validate with `--names cluster`.

Everything here is a pure function of its arguments: the server hands over what it
collected and gets back a list of strings to write. That keeps the line formats
exercisable without a broker, which is the only way to check them cheaply.
"""
import math
import os
import shutil
import time

# Emission order inside latency_cluster.log. Alphabetical sorting would put
# pipeline before service; the spec's example lists service first and the two are
# read as a pair, so the order is pinned explicitly.
LATENCY_KINDS = ("service", "pipeline")


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def pct(sorted_samples, q):
    """Nearest-rank percentile, no interpolation (04 §1) — every number printed
    is a latency some unit actually experienced. An interpolated p95 is a value
    that never happened."""
    if not sorted_samples:
        return None
    k = max(1, math.ceil(q / 100.0 * len(sorted_samples)))
    return sorted_samples[k - 1]


def _stats(samples_ms):
    """Reduce a POOLED sample list. Callers must concatenate raw samples across
    devices before calling this: averaging per-device percentiles is not a valid
    operation (04 §1)."""
    s = sorted(float(x) for x in samples_ms)
    if not s:
        return None
    return {
        "n": len(s),
        "mean_ms": sum(s) / len(s),
        "p50_ms": pct(s, 50),
        "p95_ms": pct(s, 95),
        "max_ms": s[-1],
    }


def _fmt_stats(st):
    return (f"n={st['n']} mean_ms={st['mean_ms']:.3f} p50_ms={st['p50_ms']:.3f} "
            f"p95_ms={st['p95_ms']:.3f} max_ms={st['max_ms']:.3f}")


def _group_by(rows, key):
    out = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def _pooled(reports):
    """Σbusy / Σtotal — weights each device by how long it actually ran."""
    busy = sum(r.get("busy_ns", 0) for r in reports)
    total = sum(r.get("total_ns", 0) for r in reports)
    return (busy / total if total else 0.0), busy, total


def _mean_ratio(reports):
    """Plain mean of the per-device ratios. Emitted beside the pooled figure
    because a pooled number can hide one idle device inside a busy cluster —
    when the two diverge, the cluster is imbalanced (03 §6)."""
    vals = [r["busy_ns"] / r["total_ns"] for r in reports if r.get("total_ns")]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------
# batch_done_ns.log / fps_cluster_ns.log  (01 §3.1-3.2)
# --------------------------------------------------------------------------

# Charts assume this width (01 §3.1). It lives here rather than in the server so
# the two live series and the console line can never disagree about it.
WINDOW = 16


def window_rate(times_s, batch_size, window=WINDOW):
    """Smoothed rate over the last `window` completions, or None before that many
    exist. None rather than 0.0 on purpose: a zero would be indistinguishable
    from a genuine stall, so absence means 'no window yet' (01 §3.1)."""
    if len(times_s) < window:
        return None
    span = times_s[-1] - times_s[-window]
    if span <= 0:
        return None
    return (window - 1) * batch_size / span


def batch_done_line(ts_ns, times_s, batch_size, window=WINDOW):
    """One line of batch_done_ns.log — the authoritative system-wide series.

    TWO arities: bare timestamp during warm-up, timestamp + rate afterwards. That
    is the single most common parsing bug in a port, so it is emitted from one
    place and exercised by tools/selftest_format.py."""
    rate = window_rate(times_s, batch_size, window)
    return f"{ts_ns}" if rate is None else f"{ts_ns} {rate:.2f}"


def rate_series_line(ts_ns, cluster, times_s, batch_size, window=WINDOW):
    """One line of fps_cluster_ns.log — the same arrival, bucketed by cluster.

    Each cluster runs its OWN window counter, so it reaches its first full window
    later than the system does; that is correct, not a bug. Exactly one of these
    per batch_done_ns.log line, which is the conformance check the validator
    runs."""
    line = f"{ts_ns} cluster={cluster} done={len(times_s)}"
    rate = window_rate(times_s, batch_size, window)
    if rate is not None:
        line += f" window_fps={rate:.2f}"
    return line


# --------------------------------------------------------------------------
# utilization.log  (01 §3.4)
# --------------------------------------------------------------------------

def utilization_device_line(ts_ns, report):
    """One line per device, stamped with the report's ARRIVAL on the server's
    clock — never a device timestamp. `busy_s` and `total_s` both come from that
    one device's own clock, which is what keeps clock skew out of the ratio."""
    return (f"{ts_ns} client={report.get('client_id')} "
            f"role={report.get('role')} packages={report.get('packages')} "
            f"busy_s={report.get('busy_ns', 0) / 1e9:.3f} "
            f"total_s={report.get('total_ns', 0) / 1e9:.3f} "
            f"utilization={report.get('utilization', 0) * 100:.2f}%")


# --------------------------------------------------------------------------
# fps_cluster.log  (01 §3.3)
# --------------------------------------------------------------------------

def rate_summary_lines(ts_ns, cluster_times, start_t, batch_size):
    """One line per cluster + one SYSTEM line.

    `fps` divides by the SHARED start and that scope's OWN last completion, which
    is what makes the SYSTEM span exactly the max of the cluster spans (the
    validator checks that identity rather than a sum — cluster fps values are
    deliberately NOT additive, see 01 §3.3).

    `steady_fps` divides by the cluster's own first->last completion instead, so
    a cluster that started late is not penalised for the time it did not exist.
    That is the fair number for comparing clusters."""
    lines = []
    live = {c: t for c, t in cluster_times.items() if t}
    total_done = sum(len(t) for t in live.values())
    if not total_done or start_t is None:
        return lines

    for cluster in sorted(live):
        t = live[cluster]
        n = len(t)
        frames = n * batch_size
        span = t[-1] - start_t
        fps = frames / span if span > 0 else 0.0
        own = t[-1] - t[0]
        steady = (n - 1) * batch_size / own if n >= 2 and own > 0 else 0.0
        lines.append(f"{ts_ns} cluster={cluster} fps={fps:.3f} "
                     f"steady_fps={steady:.3f} done={n} frames={frames} "
                     f"share={n / total_done * 100.0:.2f}%")

    sys_span = max(t[-1] for t in live.values()) - start_t
    sys_fps = total_done * batch_size / sys_span if sys_span > 0 else 0.0
    # The SYSTEM line carries neither steady_fps nor share (01 §3.3).
    lines.append(f"{ts_ns} SYSTEM fps={sys_fps:.3f} done={total_done} "
                 f"frames={total_done * batch_size} clusters={len(live)}")
    return lines


# --------------------------------------------------------------------------
# utilization_cluster.log  (01 §3.5)
# --------------------------------------------------------------------------

def utilization_lines(ts_ns, reports):
    """Three line kinds in one file: cluster total (`ALL`), cluster x role, and
    `SYSTEM`. This rolls up the same per-device reports that utilization.log
    holds one-per-line; it does not replace that file."""
    lines = []
    if not reports:
        return lines

    by_cluster = _group_by(reports, lambda r: r.get("cluster", "unknown"))
    for cluster in sorted(by_cluster):
        devs = by_cluster[cluster]
        pooled, busy, total = _pooled(devs)
        lines.append(
            f"{ts_ns} cluster={cluster} ALL devices={len(devs)} "
            f"utilization={pooled * 100:.2f}% "
            f"utilization_mean={_mean_ratio(devs) * 100:.2f}% "
            f"busy_s={busy / 1e9:.3f} total_s={total / 1e9:.3f} "
            f"packages={sum(d.get('packages', 0) for d in devs)}")
        by_role = _group_by(devs, lambda r: r.get("role", "unknown"))
        for role in sorted(by_role):
            rdevs = by_role[role]
            pooled, busy, total = _pooled(rdevs)
            lines.append(
                f"{ts_ns} cluster={cluster} role={role} devices={len(rdevs)} "
                f"utilization={pooled * 100:.2f}% "
                f"busy_s={busy / 1e9:.3f} total_s={total / 1e9:.3f} "
                f"packages={sum(d.get('packages', 0) for d in rdevs)}")

    pooled, busy, total = _pooled(reports)
    lines.append(f"{ts_ns} SYSTEM devices={len(reports)} clusters={len(by_cluster)} "
                 f"utilization={pooled * 100:.2f}% "
                 f"utilization_mean={_mean_ratio(reports) * 100:.2f}% "
                 f"busy_s={busy / 1e9:.3f} total_s={total / 1e9:.3f}")
    return lines


# --------------------------------------------------------------------------
# latency_cluster.log  (01 §3.6)
# --------------------------------------------------------------------------

def latency_lines(ts_ns, reports):
    """Pool the raw per-unit samples the devices shipped, THEN reduce.

    `service`/`pipeline` are per cluster x role; `e2e` is per cluster plus a
    pooled SYSTEM line and never carries `role=` — it is not a property of one
    role, and tagging it would double-count it in any chart that groups by role."""
    lines = []
    by_role, by_cluster_e2e, all_e2e = {}, {}, []

    for r in reports:
        cluster = r.get("cluster", "unknown")
        role = r.get("role", "unknown")
        for kind in LATENCY_KINDS:
            samples = r.get(f"{kind}_ms") or []
            if samples:
                by_role.setdefault((cluster, role, kind), []).extend(samples)
        e2e = r.get("e2e_ms") or []
        if e2e:
            by_cluster_e2e.setdefault(cluster, []).extend(e2e)
            all_e2e.extend(e2e)

    for cluster, role, kind in sorted(by_role, key=lambda k: (k[0], k[1],
                                                             LATENCY_KINDS.index(k[2]))):
        st = _stats(by_role[(cluster, role, kind)])
        lines.append(f"{ts_ns} cluster={cluster} role={role} kind={kind} {_fmt_stats(st)}")

    for cluster in sorted(by_cluster_e2e):
        st = _stats(by_cluster_e2e[cluster])
        lines.append(f"{ts_ns} cluster={cluster} kind=e2e {_fmt_stats(st)}")

    if all_e2e:
        lines.append(f"{ts_ns} SYSTEM kind=e2e {_fmt_stats(_stats(all_e2e))}")
    return lines


# --------------------------------------------------------------------------
# free_time.log / free_time_cluster.log / free_time_series.log  (01 §3.8-3.10, 10)
# --------------------------------------------------------------------------

def _fmt_pct(x):
    return f"{x * 100:.2f}%"


def free_time_lines(ts_ns, reports):
    """One line per device. `busy_s` is the MERGED busy time across every lane,
    never a sum of per-stage timers — a sum can exceed the span outright on a
    pipelined device, and it also misses the gaps between stages, which is
    precisely where free time lives (10 §1).

    `busy_s + free_s == span_s` exactly, and `free` and `utilization` measure
    different things and must NOT be expected to sum to 100%."""
    lines = []
    for r in sorted(reports or [], key=lambda x: (str(x.get("machine")), str(x.get("role")))):
        span_ns = float(r.get("span_ns") or 0)
        if span_ns <= 0:
            continue
        free_ns = float(r.get("free_ns") or 0)
        line = (f"{ts_ns} client={r.get('client_name')} role={r.get('role')} "
                f"machine={r.get('machine')} cluster={r.get('cluster')} "
                f"device={r.get('device') or 'unknown'} "
                f"span_s={span_ns / 1e9:.3f} busy_s={float(r.get('busy_ns') or 0) / 1e9:.3f} "
                f"free_s={free_ns / 1e9:.3f} free={_fmt_pct(free_ns / span_ns)} "
                f"gaps={r.get('gaps', 0)} "
                f"longest_free_ms={float(r.get('longest_free_ns') or 0) / 1e6:.3f}")
        if r.get("host_idle") is not None:
            line += f" host_idle={_fmt_pct(float(r['host_idle']))}"
        lines.append(line)
    return lines


def _free_pooled(reports):
    free = sum(float(r.get("free_ns") or 0) for r in reports)
    span = sum(float(r.get("span_ns") or 0) for r in reports)
    return (free / span if span else 0.0), free, span


def _free_mean(reports):
    vals = [float(r["free_ns"]) / float(r["span_ns"])
            for r in reports if float(r.get("span_ns") or 0) > 0]
    return sum(vals) / len(vals) if vals else 0.0


def _reason_lines(ts_ns, scope_prefix, reports):
    """`FREE reason=` shares MUST sum to 100% of that scope's free time. Each
    device already attributed its own free time in the published priority order
    (src/FreeTime.FREE_REASON_PRIORITY), so nothing is claimed twice and whatever
    no reason covered arrived as `unaccounted` rather than being dropped —
    summing them here therefore cannot leak."""
    totals = {}
    for r in reports:
        for reason, ns in (r.get("reasons") or {}).items():
            totals[reason] = totals.get(reason, 0.0) + float(ns)
    grand = sum(totals.values())
    if grand <= 0:
        return []
    return [f"{ts_ns} {scope_prefix}FREE reason={reason} free_s={ns / 1e9:.3f} "
            f"share={ns / grand * 100:.2f}%"
            for reason, ns in sorted(totals.items(), key=lambda kv: -kv[1])]


def _kind_lines(ts_ns, scope_prefix, reports):
    """`KIND` shares MAY sum to more than 100%: per-kind sums overlap across
    lanes by construction. Only the merged `busy_s` is exclusive, and that is the
    point — if these agree with it on a pipelined worker, the merge is silently a
    sum (10 §7)."""
    totals, span = {}, sum(float(r.get("span_ns") or 0) for r in reports)
    for r in reports:
        for kind, ns in (r.get("kinds") or {}).items():
            totals[kind] = totals.get(kind, 0.0) + float(ns)
    if not totals or span <= 0:
        return []
    return [f"{ts_ns} {scope_prefix}KIND kind={kind} busy_s={ns / 1e9:.3f} "
            f"share={ns / span * 100:.2f}%"
            for kind, ns in sorted(totals.items(), key=lambda kv: -kv[1])]


def _machine_lines(ts_ns, reports):
    """One line per host, from the UNION of the busy intervals of the device
    processes on it — never from their ratios. Two devices that are each 50% free
    can keep a machine 100% busy by interleaving, so averaging their percentages
    answers a different question than the one asked.

    This is the ONE place intervals from different processes are compared, and it
    is valid for exactly one reason: they share a clock. Intervals are never
    unioned across machines."""
    from src.FreeTime import merge_intervals

    by_machine = _group_by(reports, lambda r: r.get("machine") or "unknown")
    lines = []
    for machine in sorted(by_machine):
        devs = by_machine[machine]
        starts = [int(d["start_epoch_ns"]) for d in devs if d.get("start_epoch_ns")]
        ends = [int(d["end_epoch_ns"]) for d in devs if d.get("end_epoch_ns")]
        if not starts or not ends:
            continue
        lo, hi = min(starts), max(ends)
        span_ns = hi - lo
        if span_ns <= 0:
            continue
        busy = merge_intervals([list(iv) for d in devs
                                for iv in (d.get("intervals_epoch_ns") or [])])
        busy_ns = sum(min(e, hi) - max(s, lo) for s, e in busy
                      if min(e, hi) > max(s, lo))
        free_ns = max(0, span_ns - busy_ns)
        idles = [float(d["host_idle"]) for d in devs if d.get("host_idle") is not None]
        line = (f"{ts_ns} MACHINE machine={machine} devices={len(devs)} "
                f"free={_fmt_pct(free_ns / span_ns)} free_s={free_ns / 1e9:.3f} "
                f"span_s={span_ns / 1e9:.3f} "
                f"merge_slop_s={sum(float(d.get('merge_slop_ns') or 0) for d in devs) / 1e9:.3f}")
        if idles:
            line += f" host_idle={_fmt_pct(sum(idles) / len(idles))}"
        lines.append(line)
    return lines


def free_time_rollup_lines(ts_ns, reports):
    """free_time_cluster.log — six line kinds in one file: cluster total (`ALL`),
    cluster x role, free breakdown (`FREE`), busy breakdown (`KIND`), per-machine
    (`MACHINE`), and `SYSTEM`.

    Read them in this order: SYSTEM free (how much of the fleet was idle at all),
    then MACHINE (whether the idleness is concentrated on specific hosts), then
    FREE reason (starvation vs congestion vs overhead), and only then the
    per-device lines (10 §6)."""
    lines = []
    reports = [r for r in (reports or []) if float(r.get("span_ns") or 0) > 0]
    if not reports:
        return lines

    by_cluster = _group_by(reports, lambda r: r.get("cluster", "unknown"))
    for cluster in sorted(by_cluster, key=str):
        devs = by_cluster[cluster]
        pooled, free, span = _free_pooled(devs)
        lines.append(f"{ts_ns} cluster={cluster} ALL devices={len(devs)} "
                     f"free={_fmt_pct(pooled)} free_mean={_fmt_pct(_free_mean(devs))} "
                     f"free_s={free / 1e9:.3f} span_s={span / 1e9:.3f}")
        by_role = _group_by(devs, lambda r: r.get("role", "unknown"))
        for role in sorted(by_role, key=str):
            rdevs = by_role[role]
            pooled, free, span = _free_pooled(rdevs)
            lines.append(f"{ts_ns} cluster={cluster} role={role} devices={len(rdevs)} "
                         f"free={_fmt_pct(pooled)} "
                         f"free_mean={_fmt_pct(_free_mean(rdevs))} "
                         f"free_s={free / 1e9:.3f} span_s={span / 1e9:.3f}")
        lines += _reason_lines(ts_ns, f"cluster={cluster} ", devs)
        lines += _kind_lines(ts_ns, f"cluster={cluster} ", devs)

    lines += _machine_lines(ts_ns, reports)

    pooled, free, span = _free_pooled(reports)
    machines = len(set(r.get("machine") or "unknown" for r in reports))
    lines.append(f"{ts_ns} SYSTEM devices={len(reports)} clusters={len(by_cluster)} "
                 f"machines={machines} free={_fmt_pct(pooled)} "
                 f"free_mean={_fmt_pct(_free_mean(reports))} "
                 f"free_s={free / 1e9:.3f} span_s={span / 1e9:.3f}")
    return lines


def free_time_series_lines(ts_ns, reports):
    """One line per device per time bucket — the plottable 'when was each device
    idle' series. Read it against batch_done_ns.log on the same axis: a band of
    free time on one device that lines up with a throughput dip names the stage
    that stalled.

    The leading timestamp is the report's server-clock ARRIVAL; the position in
    the run is `t_offset_s`, on the DEVICE's clock. Devices start at different
    moments, so their offsets are not directly comparable — do not conflate the
    two (01 §3.10). `bucket_s` rides on every line rather than being assumed, so a
    long run may widen its buckets without breaking a reader."""
    lines = []
    for r in sorted(reports or [], key=lambda x: (str(x.get("machine")), str(x.get("role")))):
        bucket_s = float(r.get("bucket_s") or 1.0)
        for i, free in enumerate(r.get("series") or []):
            lines.append(
                f"{ts_ns} client={r.get('client_name')} role={r.get('role')} "
                f"machine={r.get('machine')} cluster={r.get('cluster')} "
                f"i={i} t_offset_s={i * bucket_s:.3f} bucket_s={bucket_s:.3f} "
                f"free={_fmt_pct(max(0.0, min(1.0, float(free))))}")
    return lines


# --------------------------------------------------------------------------
# message_size.log / message_size_series.log  (01 §3.11-3.12, 12)
# --------------------------------------------------------------------------

# MB = 10^6, matching broker_ram.log, so a payload size and the broker host's
# memory growth compare without a unit conversion in between (12 §4).
_SIZE_MB = 1e6


def message_size_lines(ts_ns, reports, batch_size):
    """(summary_lines, series_lines) for the two message-size files.

    Normally exactly one report: the payload shape is fixed by the configuration,
    so every edge in a cluster publishes the same size and measuring all nine
    produces one number nine times at nine times the cost (12 §1).

    Both `bytes` and `mb` go on every series line on purpose — `bytes` is the
    authoritative integer, `mb` keeps the file readable, and a reader that rounds
    its own MB from `bytes` still agrees with the summary."""
    summary, series = [], []
    for r in reports or []:
        n = int(r.get("n") or 0)
        if not n:
            continue
        ctx = r.get("context") or {}
        span_s = float(r.get("span_s") or 0.0)
        total_mb = float(r.get("total_bytes", 0)) / _SIZE_MB
        mean_mb = float(r.get("mean_bytes", 0)) / _SIZE_MB
        # The context keys are not decoration: a size without the compression
        # setting, the split point, the mode and the unit size cannot be
        # reproduced, and the same run re-read a month later reads as noise.
        ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
        summary.append(
            f"{ts_ns} client={r.get('client_name')} role={r.get('role')} "
            f"machine={r.get('machine')} cluster={r.get('cluster')} "
            + (ctx_str + " " if ctx_str else "") +
            f"n={n} total_mb={total_mb:.3f} mean_mb={mean_mb:.3f} "
            f"p50_mb={float(r.get('p50_bytes', 0)) / _SIZE_MB:.3f} "
            f"p95_mb={float(r.get('p95_bytes', 0)) / _SIZE_MB:.3f} "
            f"max_mb={float(r.get('max_bytes', 0)) / _SIZE_MB:.3f} "
            f"min_mb={float(r.get('min_bytes', 0)) / _SIZE_MB:.3f} "
            f"span_s={span_s:.3f} "
            f"rate_mb_s={(total_mb / span_s if span_s > 0 else 0.0):.3f} "
            f"per_frame_mb={(mean_mb / batch_size if batch_size else 0.0):.4f}")

        for i, offset_s, batch_id, n_bytes in r.get("series") or []:
            # The leading timestamp is the SERVER's clock (the report's arrival);
            # the position in the run is t_offset_s on the WORKER's clock. Never
            # conflate them — that split is what keeps a device timestamp out of
            # a shared file (01 §3.12).
            series.append(
                f"{ts_ns} client={r.get('client_name')} cluster={r.get('cluster')} "
                f"i={i} t_offset_s={float(offset_s):.3f} batch_id={batch_id} "
                f"bytes={int(n_bytes)} mb={int(n_bytes) / _SIZE_MB:.3f}")
    return summary, series


# --------------------------------------------------------------------------
# archiving  (05)
# --------------------------------------------------------------------------

def run_tag(config):
    """Name WHICH CONFIGURATION produced the run, from a closed vocabulary
    (05 §2). Anything finer belongs in the archived config.yaml — which is
    exactly why that file is archived beside the numbers."""
    exp = config.get("experiment") or {}
    mode = exp.get("mode", "split") if exp.get("enable", True) else "split"
    if mode in ("only_cloud", "only_edge", "adaptive"):
        return mode
    return "dynamic" if (config.get("clustering") or {}).get("enable") else "split"


def archive_run(log_path, filenames, tag, config_path="config.yaml", now=None):
    """Copy this run's result files plus the config that produced them into
    `<log_path>/results/results_<MMDD>_<HHMM>_<tag>/`.

    Copies rather than moves, so every existing reader keeps finding the live
    logs where they are (the next run truncates them itself). Empty files are
    skipped: a zero-length log must never be archived as a misleading result.
    Since the server truncates all seven files unconditionally at startup, "not
    empty" here really does mean "written by this run" — no stale file can leak
    into the archive (05 §4).

    Returns (destination, files_copied), or (None, 0) if archiving failed.
    Failure is non-fatal by design; the run still shuts down cleanly."""
    try:
        stamp = time.strftime("%m%d_%H%M", time.localtime(now))
        base = os.path.join(log_path, "results", f"results_{stamp}_{tag}")
        dest, n = base, 2
        while os.path.exists(dest):          # two runs can finish in one minute
            dest, n = f"{base}-{n}", n + 1
        os.makedirs(dest)

        copied = 0
        for src_file in list(filenames) + [config_path]:
            try:
                if os.path.getsize(src_file) == 0:
                    continue
            except OSError:
                continue
            shutil.copy2(src_file, os.path.join(dest, os.path.basename(src_file)))
            copied += 1
        return dest, copied
    except Exception:
        return None, 0
