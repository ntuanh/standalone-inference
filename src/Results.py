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
