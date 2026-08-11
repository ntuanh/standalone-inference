"""selftest_format.py — prove the result format conforms, without a broker.

guide/09 Phase 6 asks for two things a distributed run cannot give you on one
machine: `validate_results.py` exiting 0 on a run directory, and a NEGATIVE test
showing the validator would actually have caught a corrupted one. A validator
nobody has ever seen fail is not evidence.

So this builds a run directory out of the REAL line builders — src.Results for
every server-side file, src.FreeTime.FreeTimeTracker for the free-time reports,
src.MessageSize for the payload-size report, src.BrokerRam for the memory
summary — feeds it synthetic device reports, and runs the validator over the
result. What it checks is the FORMAT, which is the part that drifts silently: a
percentile printed from pre-averaged data or a second stage publishing per unit
looks fine in code review and is caught here in a second.

    python tools/selftest_format.py                # validate + negative tests
    python tools/selftest_format.py --keep results # also leave two run dirs
                                                   # behind, for the notebook

**The numbers it emits are synthetic.** They exist to exercise the format and to
give tools/build_nb.py something to render while the charts are being checked.
Never archive one of these directories as a result, and never quote a number out
of one — the fixture runs are named `synthetic-*` for exactly that reason.
"""
import argparse
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.Results as R                                       # noqa: E402
from src.BrokerRam import BrokerRamSampler                    # noqa: E402
from src.FreeTime import FreeTimeTracker                      # noqa: E402
from src.MessageSize import MessageSizeRecorder               # noqa: E402

# A fixed epoch base keeps every run byte-identical between invocations, so a
# diff of two selftest outputs shows a code change and nothing else. 19 digits,
# as the grammar requires (01 §1).
BASE_NS = 1_785_000_000_000_000_000
BATCH_SIZE = 32
CLUSTERS = ("intermediate_queue_0", "intermediate_queue_1")

# The six required files plus the eight optional ones (01 §2). Written in full
# even by the fixture: a missing file is a hard error for the reader, and the
# notebook's coverage check reads all fourteen.
FILES = {
    "batch":   "batch_done_ns.log",
    "rate_ns": "fps_cluster_ns.log",
    "rate":    "fps_cluster.log",
    "util":    "utilization.log",
    "util_c":  "utilization_cluster.log",
    "lat":     "latency_cluster.log",
    "events":  "events_ns.log",
    "free":    "free_time.log",
    "free_c":  "free_time_cluster.log",
    "free_s":  "free_time_series.log",
    "ram_ns":  "broker_ram_ns.log",
    "ram":     "broker_ram.log",
    "msize":   "message_size.log",
    "msize_s": "message_size_series.log",
}


# --------------------------------------------------------------------------
# the synthetic run
# --------------------------------------------------------------------------

def _arrivals(rng, n, period_s, start_s, jitter=0.25):
    """Completion times for one cluster. Jittered, because a perfectly regular
    series makes the window rate a constant and hides every ordering bug in the
    charts that read it."""
    t, out = start_s, []
    for _ in range(n):
        t += period_s * (1.0 + rng.uniform(-jitter, jitter))
        out.append(t)
    return out


def _device_reports(rng, cluster_arrivals, speed):
    """Per-device utilization + raw latency samples, exactly as the devices ship
    them over utilization_queue.

    `busy_ns` is derived from the service samples rather than drawn separately,
    so `Σ service == busy_s` holds by construction — the cross-check in 04 §2.1,
    and the one that makes service the only latency comparable with utilization.
    """
    reports, uid = [], 0
    for cluster, arrivals in cluster_arrivals.items():
        n_units = len(arrivals)
        span_s = arrivals[-1] - arrivals[0]
        # Three edges feed one cloud; the cloud is the completing tier, so only
        # it carries e2e (01 §3.6).
        for role, count in (("edge", 3), ("cloud", 1)):
            for k in range(count):
                uid += 1
                share = n_units // count if role == "edge" else n_units
                units = max(1, share // (3 if role == "edge" else 1))
                base_ms = (140.0 if role == "edge" else 900.0) * speed
                service = [base_ms * (1.0 + rng.uniform(-0.18, 0.30))
                           for _ in range(units)]
                busy_ns = int(round(sum(service) * 1e6))
                total_ns = int(span_s * 1e9 * (1.02 if role == "cloud" else 1.0))
                # A device is never busier than the wall clock it ran for; a
                # ratio above 100% is a measurement bug, and the validator
                # treats it as a hard error (03 §2.1).
                total_ns = max(total_ns, busy_ns + int(0.5e9))
                report = {
                    "action": "UTILIZATION",
                    "client_id": f"{uid:08x}-0000-4000-8000-{uid:012x}",
                    "role": role,
                    "cluster": cluster,
                    "packages": units,
                    "busy_ns": busy_ns,
                    "total_ns": total_ns,
                    "utilization": busy_ns / total_ns,
                    "service_ms": service,
                    # pipeline adds the in-process hand-off wait, so it is
                    # service plus queueing and never below it (04 §2.2).
                    "pipeline_ms": [s * (1.0 + rng.uniform(0.02, 0.45))
                                    for s in service],
                    "e2e_ms": ([sum(service[:1]) + base_ms * 6 * (1 + rng.uniform(0, 1.4))
                                for _ in range(units)] if role == "cloud" else []),
                }
                reports.append(report)
    return reports


def _free_reports(rng, util_reports, speed):
    """Drive the REAL FreeTimeTracker with synthetic spans.

    Feeding the tracker rather than hand-writing a report is the point: the
    merge, the priority-ordered reason attribution and the series bucketing are
    the parts with invariants (`busy + free == span`, reasons summing to exactly
    free), and hand-writing the answer would test nothing.
    """
    out = []
    for i, u in enumerate(util_reports):
        # Two device processes per host, so free_time_cluster.log gets MACHINE
        # lines built from a real union of two processes' intervals.
        machine = f"machine-{i // 2 + 1}"
        tracker = FreeTimeTracker(
            enabled=True, client_id=u["client_id"], machine=machine,
            role=u["role"], cluster=u["cluster"], bucket_s=1.0,
            log_path=None, device="cuda" if u["role"] == "cloud" else "cpu",
            client_label=str(u["client_id"])[:12])
        t0 = tracker.start_ns
        span_ns = u["total_ns"]
        cursor = 0
        # Startup is work done before the first unit exists; left out it would
        # land in `unaccounted` and the device would look idle exactly when it
        # was busiest (10 §1).
        tracker.add_work("startup", t0, t0 + int(0.4e9))
        cursor = int(0.4e9)
        unit_ns = max(1, span_ns // max(1, u["packages"]))
        for _ in range(u["packages"]):
            if cursor >= span_ns:
                break
            wait = int(unit_ns * rng.uniform(0.05, 0.45))
            work = max(1, int(unit_ns * rng.uniform(0.30, 0.70)))
            reason = "input" if u["role"] == "cloud" else "backpressure"
            tracker.add_wait(reason, t0 + cursor, t0 + cursor + wait)
            cursor += wait
            if cursor + work > span_ns:
                break
            # Two overlapping lanes: the per-kind sums therefore total MORE than
            # the merged busy time, which is the check that the merge is a union
            # and not a sum (10 §7).
            tracker.add_work("inference", t0 + cursor, t0 + cursor + work)
            tracker.add_work("metrics", t0 + cursor + work // 2,
                             t0 + cursor + work)
            cursor += work
        tracker.finish(end_ns=t0 + span_ns)
        report = tracker.report()
        if report is not None:
            out.append(report)
    return out


def _msize_report(rng, n_units, cluster, speed):
    """Payload-size report for the ONE worker the server picked (12 §1). Built
    by the real recorder from a synthetic sample list, so the percentiles,
    decimation and per-frame arithmetic are the shipped ones."""
    rec = MessageSizeRecorder(
        enabled=True, client_id="00000001-0000-4000-8000-000000000001",
        machine="machine-1", role="edge", cluster=cluster,
        context={"mode": "adaptive", "splits": None, "compress": "off",
                 "num_bit": 8, "batch_size": BATCH_SIZE, "chunks": 4},
        log_path=None, max_series=2000, client_label="00000001-000")
    mean = 4_915_200 * BATCH_SIZE / 4.0          # one chunk of a 32-frame batch
    rec._samples = [(i * 1.4 * speed, i, int(mean * (1.0 + rng.uniform(-0.03, 0.03))))
                    for i in range(n_units)]
    return rec.report()


def _ram_lines(rng, dest, span_s, speed):
    """Broker-host memory: a live series plus the summary, from the real
    sampler. The window opens before the run and closes after it, so the file
    holds the host at rest as well as under load — which is what makes the
    COMPARE line ('running the system costs this host N MB') possible at all
    (11 §6)."""
    sampler = BrokerRamSampler(
        {"broker-ram": {"enable": True, "host": "192.168.101.91"},
         "rabbit": {"address": "192.168.101.91"}},
        series_path=str(dest / FILES["ram_ns"]),
        summary_path=str(dest / FILES["ram"]))
    sampler._source = "ssh"
    total_mb, idle_mb = 7822.0, 1180.0
    t = BASE_NS
    # idle -> run -> tail, the three phases that PARTITION the series. Sampling
    # never pauses at a boundary, so there is no gap between them.
    plan = [("idle", 12, lambda k: idle_mb + rng.uniform(-6, 6)),
            ("run", int(span_s), lambda k: idle_mb + 240 * min(1.0, k / 25.0)
             + rng.uniform(-18, 18)),
            ("tail", 3, lambda k: idle_mb + 90 - 25 * k + rng.uniform(-6, 6))]
    for phase, count, level in plan:
        sampler._phase = phase
        for k in range(count):
            used = max(0.0, level(k))
            sampler._add({"ts_ns": t, "source": "ssh", "total_mb": total_mb,
                          "used_mb": used, "avail_mb": total_mb - used,
                          "free_mb": max(0.0, total_mb - used - 300),
                          "cached_mb": 300.0, "swap_used_mb": 0.0,
                          "rabbit_rss_mb": max(0.0, used - idle_mb + 120)})
            t += 1_000_000_000
    (dest / FILES["ram"]).write_text("\n".join(sampler.summary_lines()) + "\n",
                                     encoding="utf-8")


def write_run(dest, seed=7, n0=140, n1=70, speed=1.0):
    """Write one complete synthetic run directory and return it."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    start_s = BASE_NS / 1e9
    cluster_arrivals = {
        CLUSTERS[0]: _arrivals(rng, n0, 1.05 * speed, start_s),
        CLUSTERS[1]: _arrivals(rng, n1, 2.20 * speed, start_s + 3.0),
    }

    # ---- the two live series (01 §3.1-3.2) -------------------------------
    # One line in each per completion, from the same arrival — that identity is
    # the line-count check the validator runs, and writing them together here is
    # what makes it structurally true rather than hopefully true.
    merged = sorted((t, c) for c, ts in cluster_arrivals.items() for t in ts)
    sys_times, per_cluster = [], {c: [] for c in CLUSTERS}
    batch_lines, series_lines = [], []
    for t_s, cluster in merged:
        ts_ns = int(t_s * 1e9)
        sys_times.append(t_s)
        per_cluster[cluster].append(t_s)
        batch_lines.append(R.batch_done_line(ts_ns, sys_times, BATCH_SIZE))
        series_lines.append(R.rate_series_line(ts_ns, cluster,
                                               per_cluster[cluster], BATCH_SIZE))

    end_ns = int(merged[-1][0] * 1e9)
    util_reports = _device_reports(rng, cluster_arrivals, speed)
    free_reports = _free_reports(rng, util_reports, speed)
    msize = _msize_report(rng, n0, CLUSTERS[0], speed)

    out = {
        "batch":   batch_lines,
        "rate_ns": series_lines,
        "rate":    R.rate_summary_lines(end_ns, per_cluster, start_s, BATCH_SIZE),
        "util":    [R.utilization_device_line(end_ns, r) for r in util_reports],
        "util_c":  R.utilization_lines(end_ns, util_reports),
        "lat":     R.latency_lines(end_ns, util_reports),
        "events":  [f"{int((merged[0][0] + 30) * 1e9)} {CLUSTERS[0]}: "
                    f"route start->edge_only (cloud backlogged)",
                    f"{int((merged[0][0] + 95) * 1e9)} {CLUSTERS[0]}: "
                    f"route edge_only->split (cloud has capacity)"],
        "free":    R.free_time_lines(end_ns, free_reports),
        "free_c":  R.free_time_rollup_lines(end_ns, free_reports),
        "free_s":  R.free_time_series_lines(end_ns, free_reports),
    }
    summary, series = R.message_size_lines(end_ns, [msize], BATCH_SIZE)
    out["msize"], out["msize_s"] = summary, series

    for key, lines in out.items():
        (dest / FILES[key]).write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    (dest / FILES["ram_ns"]).write_text("", encoding="utf-8")
    _ram_lines(rng, dest, merged[-1][0] - merged[0][0], speed)
    return dest


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate(run_dir):
    """Run the guide's validator exactly as a porter would. Returns
    (returncode, output)."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "guide" / "validate_results.py"),
         str(run_dir), "--names", "cluster"],
        capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _replace(path, old, new, count=1):
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"mutation target not found in {path.name}: {old!r}")
    path.write_text(text.replace(old, new, count), encoding="utf-8")


# Each mutation is one row of the "common porting failures" table in guide/09 —
# the defects that are invisible in code review and visible in the output. The
# expected substring pins WHICH check fired: a mutation caught by the wrong rule
# is a validator bug hiding behind a passing test.
def _mut_double_count(d):
    """Two stages publishing per unit — group `done` sums past SYSTEM."""
    p = d / FILES["rate"]
    line = [l for l in p.read_text(encoding="utf-8").splitlines() if "SYSTEM" in l][0]
    done = int(line.split("done=")[1].split()[0])
    _replace(p, f"done={done} ", f"done={done - 20} ")


def _mut_line_count(d):
    """A tier that stopped reporting early — the two live series disagree."""
    p = d / FILES["rate_ns"]
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-3]) + "\n", encoding="utf-8")


def _mut_util_over_100(d):
    """Overlapping busy intervals summed instead of merged."""
    p = d / FILES["util"]
    first = p.read_text(encoding="utf-8").splitlines()[0]
    old = first.split("utilization=")[1]
    _replace(p, f"utilization={old}", "utilization=118.40%")


def _mut_percentiles(d):
    """Per-device percentiles averaged instead of pooled raw samples."""
    p = d / FILES["lat"]
    first = p.read_text(encoding="utf-8").splitlines()[0]
    p50 = first.split("p50_ms=")[1].split()[0]
    p95 = first.split("p95_ms=")[1].split()[0]
    _replace(p, f"p50_ms={p50} p95_ms={p95}", f"p50_ms={p95} p95_ms={p50}")


def _mut_e2e_role(d):
    """e2e tagged with a role — it spans two stages and would double-count in
    any chart that groups by role."""
    _replace(d / FILES["lat"], "kind=e2e", "role=cloud kind=e2e")


def _mut_system_steady(d):
    """steady_fps on the SYSTEM line, which has no single first completion."""
    _replace(d / FILES["rate"], "SYSTEM fps=", "SYSTEM steady_fps=1.0 fps=")


def _mut_missing_mean(d):
    """A pooled ratio with no plain mean beside it — the divergence between the
    two is the imbalance signal, so dropping one hides it."""
    p = d / FILES["util_c"]
    first = [l for l in p.read_text(encoding="utf-8").splitlines() if " ALL " in l][0]
    mean = first.split("utilization_mean=")[1].split()[0]
    _replace(p, f" utilization_mean={mean}", "")


def _mut_bad_timestamp(d):
    """A timestamp that is not ns-epoch — the realistic version of this defect
    is a unit slip (seconds or microseconds logged as if they were nanoseconds),
    not a garbled string, and the grammar check is what catches it."""
    p = d / FILES["batch"]
    lines = p.read_text(encoding="utf-8").splitlines()
    parts = lines[5].split()
    parts[0] = parts[0][:13]                 # ns -> microseconds, 13 digits
    lines[5] = " ".join(parts)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_three_columns(d):
    """A third column in the two-arity file."""
    p = d / FILES["batch"]
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1] + " 3.00"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_missing_file(d):
    """A required file absent — a hard error for the reader (01 §2)."""
    (d / FILES["util_c"]).unlink()


def _mut_per_group_start(d):
    """Each group divided by its OWN first completion instead of the shared
    START. Every group span shrinks, so the longest one no longer matches the
    SYSTEM span — which is the identity the validator checks, precisely because
    group fps values are not additive and a sum cannot detect this."""
    p = d / FILES["rate"]
    groups = [l for l in p.read_text(encoding="utf-8").splitlines() if "cluster=" in l]
    # Mutate the group that currently defines the SYSTEM span; changing any
    # other one leaves max(spans) untouched and the defect invisible.
    def span(line):
        return (float(line.split("frames=")[1].split()[0])
                / float(line.split(" fps=")[1].split()[0]))
    longest = max(groups, key=span)
    fps = longest.split(" fps=")[1].split()[0]
    _replace(p, f" fps={fps} ", f" fps={float(fps) * 1.5:.3f} ")


NEGATIVE = [
    ("group done sums past SYSTEM",     _mut_double_count,   "group done sums to"),
    ("a tier stopped reporting early",  _mut_line_count,     "line-count mismatch"),
    ("utilization above 100%",          _mut_util_over_100,  "exceeds 100%"),
    ("percentiles out of order",        _mut_percentiles,    "percentiles out of order"),
    ("e2e line carries role=",          _mut_e2e_role,       "must not carry role="),
    ("SYSTEM line carries steady_fps",  _mut_system_steady,  "must not carry steady_fps"),
    ("ALL line lost utilization_mean",  _mut_missing_mean,   "must carry"),
    ("line without a ns timestamp",     _mut_bad_timestamp,  "19-digit"),
    ("three columns in batch_done",     _mut_three_columns,  "expected 1 or 2 columns"),
    ("a required file is missing",      _mut_missing_file,   "MISSING (required)"),
    ("START was not shared",            _mut_per_group_start, "SYSTEM span"),
]


def negative_tests(good_dir):
    """Corrupt a COPY per defect and confirm the validator fails on it, with the
    check that fired named. Skipping this step leaves a validator nobody has
    ever seen reject anything (09 Phase 6)."""
    failures = []
    for name, mutate, expected in NEGATIVE:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad"
            shutil.copytree(good_dir, bad)
            mutate(bad)
            rc, out = validate(bad)
            if rc == 0:
                failures.append(f"{name}: validator PASSED a corrupted run")
            elif expected not in out:
                failures.append(f"{name}: caught, but not by the expected check "
                                f"(wanted {expected!r})")
            else:
                print(f"  [caught] {name}")
    return failures


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", metavar="DIR",
                    help="also write synthetic-a/ and synthetic-b/ under DIR "
                         "(two runs of the same workload, for the notebook)")
    args = ap.parse_args()

    print("building a synthetic run from the real line builders...")
    with tempfile.TemporaryDirectory() as tmp:
        run = write_run(Path(tmp) / "run")
        rc, out = validate(run)
        print(out.strip())
        if rc != 0:
            print("\n=== FAIL: the format the code emits is NOT conformant ===")
            return 1

        print("\nnegative tests — each corrupts a copy and must be rejected:")
        failures = negative_tests(run)
        if failures:
            print("\n=== FAIL ===")
            for f in failures:
                print(f"  {f}")
            return 1
        print(f"\n  -> {len(NEGATIVE)}/{len(NEGATIVE)} defects caught")

    if args.keep:
        base = Path(args.keep)
        # Two runs of the SAME workload: identical done/frames, different speed.
        # Comparing runs that did different amounts of work is the mistake in
        # 05 §6, so the fixture cannot demonstrate it.
        for name, speed in (("synthetic-a", 1.0), ("synthetic-b", 1.32)):
            dest = base / name
            if dest.exists():
                shutil.rmtree(dest)
            write_run(dest, speed=speed)
            rc, _ = validate(dest)
            print(f"wrote {dest}  (validator rc={rc})")
        print("\nNOTE: synthetic numbers, for exercising the charts only. "
              "Never archive or quote them.")

    print("\n=== format conformance: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
