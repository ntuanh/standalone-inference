"""Device-side payload-size recorder — guide/12-message-size.md.

Every other measurement in the guide describes *time*. This one describes
*bytes*: how large the body this worker hands to pika is, sampled once per
published message, recorded BEFORE the publish call.

It is the number that makes three other files readable. utilization.log says a
device was busy; this says whether it was busy computing or busy shipping.
broker_ram.log shows the queue filling; message size x queue depth says whether
that is the payload or something else.

Exactly ONE worker runs this, and it does not choose itself: the server picks the
first edge that registered and sets `measure_message_size` in that client's START
message (guide/12 §1, README invariant 9). A worker that read the decision from
its own config.yaml would be one stale file away from a run where all 9 edges
measured, or none did — and the summary line looks identical either way.

Only bodies published to intermediate_queue are sampled. That is the payload that
crosses the network and occupies broker memory, so it is the one that pairs with
broker_ram.log; bbox_queue bodies are ~KB of text on a different route and mixing
the two would make every percentile bimodal and meaningless.
"""
import os
import time

from src.Results import pct

# MB = 10^6 throughout, matching guide/11-broker-ram.md, so a payload size and the
# host's memory growth can be compared without a conversion in between.
_MB = 1e6

# Cap on the series SHIPPED to the server. Beyond this the samples are decimated
# EVENLY (never truncated), so the series still spans the whole run — a 3000-batch
# run must not turn one report into a multi-megabyte message. The summary
# statistics are computed over every sample regardless: decimation may coarsen a
# plot, never a number (guide/12 §3).
DEFAULT_MAX_SERIES = 2000


def _decimate(samples, cap):
    """Even stride, first and last always kept. Truncating instead would leave a
    series that stops in the middle of the run and looks like a device that died."""
    n = len(samples)
    if cap <= 0 or n <= cap:
        return list(samples)
    step = n / float(cap)
    out = [samples[min(n - 1, int(i * step))] for i in range(cap)]
    if out[-1] is not samples[-1]:
        out[-1] = samples[-1]
    return out


class MessageSizeRecorder:
    """One per worker. Inert on every worker the server did not pick: `record`
    returns on the first line, so the cost on the other 8 edges is one attribute
    lookup per message."""

    def __init__(self, enabled, client_id, machine, role, cluster,
                 context=None, log_path=None, max_series=DEFAULT_MAX_SERIES,
                 client_label=None):
        self.enabled = bool(enabled)
        self.client_id = str(client_id)
        # Separate keys for the same reason as free time: a host can run several
        # device processes, and `client` has to identify the process while
        # `machine` identifies the box.
        self.client_name = client_label or str(client_id).replace("-", "")[:12]
        self.machine = machine or self.client_name
        self.role = role
        self.cluster = cluster
        self.context = dict(context or {})
        self.max_series = int(max_series)

        self._samples = []          # (t_offset_s, batch_id, n_bytes) — ALL of them
        self._t0_ns = None          # this worker's first publish, its own clock
        self._path = log_path or f"message_size_{self.client_name}.log"
        # One warning, then the recorder disables itself. A per-message warning on
        # a full disk buries the run's real output (guide/12 §6.7).
        self._file_ok = True

        if self.enabled:
            try:
                open(self._path, "w").close()
            except Exception as e:
                self._file_ok = False
                self._warn(f"cannot open {self._path}: {e}")

    # -- recording ---------------------------------------------------------

    def record(self, n_bytes, batch_id=None):
        """Call with the SERIALIZED byte count, immediately before the publish.

        Before, not after: both orderings look equivalent until the transport
        blocks, and a broker at its high-water mark stopping mid-write is exactly
        the run this measurement exists to explain. Measuring after the call
        writes that sample late, or — if the publish raises — never.
        """
        if not self.enabled:
            return
        try:
            now_ns = time.time_ns()
            if self._t0_ns is None:
                self._t0_ns = now_ns
            offset_s = (now_ns - self._t0_ns) / 1e9
            self._samples.append((offset_s, batch_id, int(n_bytes)))
            if self._file_ok:
                # Live append: the local copy is the only one that survives a
                # broker or server problem, so a run that dies still leaves every
                # sample it took (guide/12 §3).
                with open(self._path, "a") as f:
                    f.write(f"{now_ns} i={len(self._samples) - 1} "
                            f"t_offset_s={offset_s:.3f} batch_id={batch_id} "
                            f"bytes={int(n_bytes)} mb={int(n_bytes) / _MB:.3f}\n")
        except Exception as e:
            self._file_ok = False
            self._warn(f"recording disabled after: {e}")

    # -- reporting ---------------------------------------------------------

    def report(self):
        """The message shipped to the server at finish, or None when this worker
        did not measure. Statistics come from the FULL sample list; only the
        series is decimated."""
        if not self.enabled or not self._samples:
            return None

        sizes = sorted(s[2] for s in self._samples)
        n = len(sizes)
        total_bytes = sum(sizes)
        span_s = self._samples[-1][0] - self._samples[0][0]
        series = _decimate(self._samples, self.max_series)

        return {
            "action": "MESSAGE_SIZE",
            "client_id": self.client_id,
            "client_name": self.client_name,
            "machine": self.machine,
            "role": self.role,
            "cluster": self.cluster,
            "context": self.context,
            "n": n,
            "total_bytes": total_bytes,
            "span_s": span_s,
            # Nearest-rank, no interpolation — same rule as latency, so every
            # number printed is a size that actually occurred (guide/12 §4).
            "mean_bytes": total_bytes / n,
            "p50_bytes": pct(sizes, 50),
            "p95_bytes": pct(sizes, 95),
            "max_bytes": sizes[-1],
            "min_bytes": sizes[0],
            "series": [(i, off, bid, nb)
                       for i, (off, bid, nb) in enumerate(series)],
        }

    # -- misc --------------------------------------------------------------

    def _warn(self, msg):
        try:
            import src.Log as Log
            Log.print_with_color(f"[MsgSize] {msg}", "yellow")
        except Exception:
            print(f"[MsgSize] {msg}")

    @property
    def path(self):
        return self._path

    def __len__(self):
        return len(self._samples)
