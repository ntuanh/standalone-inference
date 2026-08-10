"""Device-side free-time tracker — guide/10-free-time.md.

**Free time is the wall clock in which a device did nothing at all** — no reading
input, no compute, no serialize, no send, no receive, no decode, no postprocess,
no bookkeeping:

    free = (end - start) - | union of every lane's busy intervals |

It is a property of the *device*, not of a stage, and it is neither utilization
nor `1 - utilization`:

    utilization (03)  busy / total over each unit's 'get input -> output' window,
                      on ONE lane. A back-pressure wait INSIDE that window counts
                      as busy; work on another lane counts as nothing.
    free time  (10)   every lane, whole run. That same back-pressure wait counts
                      as FREE; that same other-lane work counts as BUSY.

So `free% + utilization% != 100%` and neither can be derived from the other. Emit
both: a device at 40% utilization and 3% free time is not idle, it is doing work
the unit window never saw, and that gap is the finding.

**Never sum per-stage timers.** Two lanes each 90% busy sum to 180% of the wall
clock, and a sum also misses the gaps *between* stages — which is exactly where
free time lives. Intervals are merged, never added. `busy_ns` here is a UNION;
the per-kind sums beside it deliberately overlap and are expected to total more.
"""
import os
import threading
import time

# Free-time reasons in the order they claim a moment. Attribution walks this list
# subtracting what busy time and earlier reasons already took, so the parts sum to
# EXACTLY free_ns with nothing double counted and nothing dropped. The order is
# part of the definition, not an implementation detail, so it is published here
# and repeated in the file that consumes it (src/Results.free_time_rollup_lines):
#
#   input         starved — nothing to work on
#   backpressure  the broker or the permit pool would not take the next body
#   downstream    another device holds the lock/queue this one needs
#   control       waiting on a control-plane decision (STOP, a reply)
#   unaccounted   free, but no wait span covered it: the overhead BETWEEN
#                 instrumented stages. Useful in itself, so it is reported rather
#                 than folded into a neighbour.
FREE_REASON_PRIORITY = ("input", "backpressure", "downstream", "control")
UNACCOUNTED = "unaccounted"

# A 20 ms poll loop produces ~50 wait intervals per idle second and a 2 ms one
# ~500. Extending the lane's open wait span instead of appending keeps a long
# stall as ONE interval, so memory tracks the number of distinct stalls rather
# than the number of polls. Work spans are never coalesced with a tolerance —
# busy has to stay exact.
_WAIT_COALESCE_NS = 1_000_000        # 1 ms

# Fold pending spans into the merged list this often, so memory is bounded by the
# number of DISJOINT busy periods rather than by the number of recorded operations.
_MERGE_EVERY = 2048

# Cap on the merged busy intervals SHIPPED to the server for the machine-level
# union. Beyond it the SMALLEST gaps are closed first and the swallowed total is
# reported as merge_slop_s — biasing the answer toward less free time, which is
# the safe direction, and stating its own error bar (guide/10 §4).
_MAX_SHIPPED_INTERVALS = 4000

# A long run must not turn the report into a multi-megabyte message either. The
# series widens its buckets instead of dropping any, so it still spans the whole
# run — bucket_s travels on every line precisely so a reader never assumes it
# (01 §3.10).
_MAX_SERIES_BUCKETS = 3000


def merge_intervals(intervals):
    """Union of [start, end) pairs, sorted and coalesced. The whole method rests
    on this: adding durations instead is the error the file exists to prevent."""
    if not intervals:
        return []
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1][1] = end
        else:
            out.append([start, end])
    return out


def _total(intervals):
    return sum(e - s for s, e in intervals)


def _clip(intervals, lo, hi):
    """Keep the report inside the run window. Without this a span opened before
    `start` or closed after `end` makes busy_ns + free_ns != span_ns, which is the
    first invariant the validator checks."""
    out = []
    for s, e in intervals:
        s, e = max(s, lo), min(e, hi)
        if e > s:
            out.append([s, e])
    return out


def _complement(busy, lo, hi):
    """The gaps — what is left of [lo, hi) after the merged busy intervals."""
    out, cursor = [], lo
    for s, e in busy:
        if s > cursor:
            out.append([cursor, s])
        cursor = max(cursor, e)
    if cursor < hi:
        out.append([cursor, hi])
    return out


def _intersect(a, b):
    """Intersection of two sorted, disjoint interval lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract(a, b):
    """a minus b, both sorted and disjoint."""
    out = []
    j = 0
    for s, e in a:
        cursor = s
        while j < len(b) and b[j][1] <= cursor:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cursor:
                out.append([cursor, b[k][0]])
            cursor = max(cursor, b[k][1])
            k += 1
        if cursor < e:
            out.append([cursor, e])
    return out


def _cap_intervals(busy, limit):
    """Close the SMALLEST gaps until the list fits, and return the non-busy time
    that was swallowed. Smallest first because a big gap is a real idle stretch
    and a 3 ms one between two consecutive stages is noise — the cap must not be
    allowed to eat the finding."""
    if len(busy) <= limit or limit <= 0:
        return busy, 0
    gaps = sorted(range(len(busy) - 1), key=lambda i: busy[i + 1][0] - busy[i][1])
    drop = set(gaps[:len(busy) - limit])
    out, slop = [], 0
    for i, (s, e) in enumerate(busy):
        if out and (i - 1) in drop:
            slop += s - out[-1][1]
            out[-1][1] = e
        else:
            out.append([s, e])
    return out, slop


class FreeTimeTracker:
    """One per device process. Lanes are threads, identified automatically —
    a lane must never have to be named by the caller, or the instrumentation rots
    the moment a stage moves to another thread."""

    def __init__(self, enabled, client_id, machine, role, cluster,
                 bucket_s=1.0, log_path=None, device=None, client_label=None):
        self.enabled = bool(enabled)
        self.client_id = str(client_id)
        # `client` and `machine` are deliberately DIFFERENT keys. Several device
        # processes share one host, and the machine rollup exists precisely to
        # union them — so if both keys carried the host name, the two processes on
        # machine-2 would be indistinguishable in free_time.log and their series
        # lines would collide (guide/10 §4).
        self.client_name = client_label or str(client_id).replace("-", "")[:12]
        self.machine = machine or self.client_name
        self.role = role
        self.cluster = cluster
        self.bucket_s = float(bucket_s) if bucket_s else 1.0
        self.device = device
        self._path = log_path

        self._lock = threading.Lock()
        self._work = []                  # (start_ns, end_ns) monotonic, every lane
        self._merged = []                # periodically folded union of the above
        self._pending = 0
        self._kinds = {}                 # kind -> ns. OVERLAPS across lanes by
                                         # construction; never the answer.
        self._lanes = {}                 # thread name -> ns
        self._waits = {}                 # reason -> [(start, end), ...]
        self._open_wait = {}             # (thread, reason) -> [start, end]

        # perf_counter_ns, not time_ns: a clock step mid-run must not be able to
        # produce a negative interval. The epoch reading beside it is only used to
        # translate the merged intervals for the server's machine-level union,
        # where processes on ONE host are compared and nothing crosses a machine.
        self._t0_mono = time.perf_counter_ns()
        self._t0_epoch = time.time_ns()
        self._t_end_mono = None
        self._cpu0 = self._cpu_times()
        self._file_ok = True

    # -- clock -------------------------------------------------------------

    @property
    def start_ns(self):
        """When this device's run window opened. Handed to add_work by the tier
        loops so the model load and the video open — real work, done before the
        first unit exists — are not left sitting in `unaccounted`."""
        return self._t0_mono

    @staticmethod
    def now():
        """Take this BEFORE a call that may be either work or a wait, and classify
        after it returns — a non-blocking receive is work when it yields a unit
        and free when the queue was empty, and guessing beforehand gets it wrong
        every time the queue runs dry (guide/10 §2)."""
        return time.perf_counter_ns()

    # -- recording ---------------------------------------------------------

    def add_work(self, kind, t0_ns, t1_ns=None):
        """A real operation: read input, compute, encode, send, receive, decode,
        postprocess, write metrics. Work spans define the answer."""
        if not self.enabled or t0_ns is None:
            return
        t1_ns = self.now() if t1_ns is None else t1_ns
        if t1_ns <= t0_ns:
            return
        with self._lock:
            self._work.append((t0_ns, t1_ns))
            self._kinds[kind] = self._kinds.get(kind, 0) + (t1_ns - t0_ns)
            lane = threading.current_thread().name
            self._lanes[lane] = self._lanes.get(lane, 0) + (t1_ns - t0_ns)
            self._pending += 1
            if self._pending >= _MERGE_EVERY:
                self._merge_locked()

    def add_wait(self, reason, t0_ns, t1_ns=None):
        """A block: waiting for input, for a permit, for a lock, for a control
        reply. Wait spans do NOT define the answer — they only *explain* it. A
        moment covered by no wait span is still free; it lands in `unaccounted`,
        which is the overhead between instrumented stages and a signal in itself."""
        if not self.enabled or t0_ns is None:
            return
        t1_ns = self.now() if t1_ns is None else t1_ns
        if t1_ns <= t0_ns:
            return
        key = (threading.current_thread().name, reason)
        with self._lock:
            open_span = self._open_wait.get(key)
            if open_span is not None and t0_ns - open_span[1] <= _WAIT_COALESCE_NS:
                # Same stall, next poll — extend rather than append, or a 20 ms
                # loop writes 50 intervals per idle second.
                open_span[1] = max(open_span[1], t1_ns)
                return
            if open_span is not None:
                self._waits.setdefault(key[1], []).append(tuple(open_span))
            self._open_wait[key] = [t0_ns, t1_ns]

    # -- lifecycle ---------------------------------------------------------

    def finish(self, end_ns=None):
        """Close the run window. Idempotent — the shutdown path can be reached
        from more than one place and closing twice must not move the span.

        The end timestamp is taken FIRST and stored LAST: anything recorded while
        this runs still lands inside the window it is being measured against."""
        if not self.enabled or self._t_end_mono is not None:
            return
        end = self.now() if end_ns is None else end_ns
        with self._lock:
            for key, span in self._open_wait.items():
                self._waits.setdefault(key[1], []).append(tuple(span))
            self._open_wait.clear()
            self._merge_locked()
        self._t_end_mono = end

    def report(self):
        """The whole-run report shipped to the server, or None when disabled or
        when the window never opened."""
        if not self.enabled:
            return None
        self.finish()
        lo, hi = self._t0_mono, self._t_end_mono
        span_ns = hi - lo
        if span_ns <= 0:
            return None

        # Fold anything still pending, unconditionally. finish() already does it,
        # but a report built from a window closed by any other route would
        # otherwise read busy_s=0 — a device that did all its work looking
        # perfectly idle, with every other number still self-consistent.
        with self._lock:
            self._merge_locked()
        busy = _clip(self._merged, lo, hi)
        busy_ns = _total(busy)
        free = _complement(busy, lo, hi)
        free_ns = _total(free)

        # busy_ns + free_ns == span_ns EXACTLY, by construction: free is the
        # complement of busy inside the clipped window, so the two partition it.
        # The validator checks this and its failure means intervals escaped the
        # window or the clip step is missing.
        reasons, remaining = {}, free
        for reason in FREE_REASON_PRIORITY:
            spans = _clip(merge_intervals([list(x) for x in self._waits.get(reason, [])]), lo, hi)
            if not spans:
                continue
            claimed = _intersect(spans, remaining)
            if not claimed:
                continue
            reasons[reason] = _total(claimed)
            remaining = _subtract(remaining, claimed)
        leftover = _total(remaining)
        if leftover > 0:
            reasons[UNACCOUNTED] = leftover

        shipped, slop = _cap_intervals([list(x) for x in busy], _MAX_SHIPPED_INTERVALS)

        report = {
            "action": "FREE_TIME",
            "client_id": self.client_id,
            "client_name": self.client_name,
            "machine": self.machine,
            "role": self.role,
            "cluster": self.cluster,
            "device": self.device,
            "span_ns": span_ns,
            # The window's ends on the epoch clock, so the server can lay this
            # device's process beside the others on the SAME host and take the
            # envelope. Meaningless across hosts, and never used that way.
            "start_epoch_ns": self._t0_epoch,
            "end_epoch_ns": self._t0_epoch + span_ns,
            "busy_ns": busy_ns,
            "free_ns": free_ns,
            "gaps": len(free),
            "longest_free_ns": max((e - s for s, e in free), default=0),
            "host_idle": self._host_idle(),
            "kinds": dict(self._kinds),          # overlapping, by construction
            "reasons": reasons,                  # sums to free_ns exactly
            "lanes": dict(self._lanes),
            "merge_slop_ns": slop,
            "bucket_s": self._series_bucket_s(span_ns),
            "series": self._series(busy, lo, hi),
            # Epoch ns, so the server can union the device PROCESSES that share a
            # machine. Safe only because those processes share a clock; intervals
            # are never unioned across machines (guide/10 §4).
            "intervals_epoch_ns": [(self._t0_epoch + (s - lo), self._t0_epoch + (e - lo))
                                   for s, e in shipped],
        }
        self._write_local(report)
        return report

    # -- internals ---------------------------------------------------------

    def _merge_locked(self):
        self._merged = merge_intervals(self._merged + [list(x) for x in self._work])
        self._work = []
        self._pending = 0

    def _series_bucket_s(self, span_ns):
        """Widen the bucket rather than dropping samples, so a long run's series
        still covers the whole run at a coarser resolution."""
        span_s = span_ns / 1e9
        want = self.bucket_s
        if span_s / want > _MAX_SERIES_BUCKETS:
            want = span_s / _MAX_SERIES_BUCKETS
        return want

    def _series(self, busy, lo, hi):
        """Free fraction per fixed-width bucket — the plottable 'when was this
        device idle' curve, read against batch_done_ns.log on one axis."""
        bucket_ns = int(self._series_bucket_s(hi - lo) * 1e9) or 1
        n = max(1, int((hi - lo + bucket_ns - 1) // bucket_ns))
        out, i = [], 0
        for b in range(n):
            b_lo = lo + b * bucket_ns
            b_hi = min(hi, b_lo + bucket_ns)
            if b_hi <= b_lo:
                break
            busy_in = 0
            while i < len(busy) and busy[i][1] <= b_lo:
                i += 1
            j = i
            while j < len(busy) and busy[j][0] < b_hi:
                busy_in += min(busy[j][1], b_hi) - max(busy[j][0], b_lo)
                j += 1
            out.append(1.0 - busy_in / float(b_hi - b_lo))
        return out

    @staticmethod
    def _cpu_times():
        try:
            import psutil
            return psutil.cpu_times()
        except Exception:
            return None

    def _host_idle(self):
        """OS-level idle share of the WHOLE machine over the run, across all
        processes. It answers a different question from the pipeline's own free
        time, and disagreement is informative: high free with low host idle means
        something else on the box is eating the CPU."""
        t1 = self._cpu_times()
        t0 = self._cpu0
        if t0 is None or t1 is None:
            return None
        try:
            fields = [f for f in t0._fields if hasattr(t1, f)]
            total = sum(getattr(t1, f) - getattr(t0, f) for f in fields)
            idle = (t1.idle - t0.idle)
            if hasattr(t1, "iowait") and hasattr(t0, "iowait"):
                idle += t1.iowait - t0.iowait
            return max(0.0, min(1.0, idle / total)) if total > 0 else None
        except Exception:
            return None

    def _write_local(self, report):
        """Keep a copy on the device even though the server has everything: it
        survives a broker or server failure, and it is the artifact you read when
        one device behaves differently from its peers (guide/10 §5)."""
        if not self._path or not self._file_ok:
            return
        try:
            span_s = report["span_ns"] / 1e9
            with open(self._path, "w") as f:
                f.write(f"{self._t0_epoch} client={report['client_name']} "
                        f"role={report['role']} machine={report['machine']} "
                        f"cluster={report['cluster']} span_s={span_s:.3f} "
                        f"busy_s={report['busy_ns'] / 1e9:.3f} "
                        f"free_s={report['free_ns'] / 1e9:.3f} "
                        f"free={report['free_ns'] / report['span_ns'] * 100:.2f}% "
                        f"gaps={report['gaps']}\n")
                for reason, ns in sorted(report["reasons"].items()):
                    f.write(f"{self._t0_epoch} FREE reason={reason} "
                            f"free_s={ns / 1e9:.3f}\n")
                for kind, ns in sorted(report["kinds"].items()):
                    f.write(f"{self._t0_epoch} KIND kind={kind} busy_s={ns / 1e9:.3f}\n")
        except Exception:
            # One failure disables the local copy; the report still ships.
            self._file_ok = False
