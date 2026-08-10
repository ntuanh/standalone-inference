"""Broker-host RAM sampler — guide/11-broker-ram.md.

Every other measurement in this project is reported *by* a process we wrote.
This one is not: between the edges and the clouds sits the RabbitMQ box, which
runs no code of ours and can still be the bottleneck. When it is, **every symptom
shows up somewhere else** — a broker at its high-water mark does not fail, it
blocks publishers, and on the edge that looks like a stall with no local cause.
"The cloud is slow" and "the broker stopped accepting" produce almost identical
worker-side telemetry; this host's memory curve is what separates them.

Two rules, both structural:

**The SERVER samples it, not a worker.** The server is the only component alive
for the whole run, already owns the shutdown and archive steps, and holds the
authoritative clock. A worker would stop when it finished, and workers do not
finish together.

**One long-lived session, not one connection per sample.** The obvious loop —
connect, read, close, sleep, repeat — makes the machine whose load we are
measuring pay a TCP handshake and an authentication per sample: at 1 s over a
20-minute run, 1200 logins added to the thing under test. Instead one SSH session
runs a bounded remote loop that prints one line per sample, and we read its
stdout.
"""
import base64
import threading
import time

# The remote sampler. Every field of a sample is printed on ONE line, so a partial
# read can never interleave two samples. The '^' anchors matter: /Cached:/ without
# one also matches SwapCached:. The loop is BOUNDED ($MAX) so a sampler orphaned
# by a hard kill of the server cannot run on the broker host forever.
_REMOTE_SCRIPT = r"""
MAX=%(max_iter)d
INTERVAL=%(interval)s
i=0
while [ $i -lt $MAX ]; do
  i=$((i+1))
  r=`ps -eo rss=,comm= | awk '$2 ~ /beam|rabbit/ {s+=$1} END{print s+0}'`
  awk -v ts="`date +%%s%%N`" -v r="$r" '
    /^MemTotal:/{t=$2} /^MemFree:/{f=$2} /^MemAvailable:/{a=$2}
    /^Buffers:/{b=$2} /^Cached:/{c=$2}
    /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2}
    END{print "RAM", ts, t, f, a, b, c, st, sf, r}' /proc/meminfo
  sleep $INTERVAL
done
"""

# Marks that partition the series. They never gate it: sampling does not pause at
# a boundary, so a missing mark coarsens the split rather than leaving a gap. A
# run that never dispatched is all `idle`, which is exactly what it was.
PHASES = ("idle", "run", "tail")

_KB = 1024.0


def _pct(sorted_vals, q):
    """Nearest-rank, no interpolation — same rule as latency, so every number
    printed is a value that was actually observed on this host."""
    if not sorted_vals:
        return 0.0
    k = max(1, int(-(-q * len(sorted_vals) // 100)))
    return sorted_vals[k - 1]


class BrokerRamSampler:
    """Start it in the server's constructor, mark the phases as the run moves
    through them, stop it a second or two after the drain.

    Telemetry never kills the run: an unreachable host, refused credentials or a
    missing SSH client all degrade to a warning and a `samples=0` line naming the
    reason. A missing file is indistinguishable from a run where the host was
    fine; `samples=0 (paramiko not installed)` is not.
    """

    def __init__(self, config, series_path, summary_path, log=None):
        cfg = (config.get("broker-ram") or {})
        rabbit = (config.get("rabbit") or {})
        self.enabled = bool(cfg.get("enable", False))
        self.host = cfg.get("host") or rabbit.get("address")
        self.interval_s = float(cfg.get("interval_s", 1.0))
        self.tail_s = float(cfg.get("tail_s", 2.0))
        self.max_run_s = float(cfg.get("max_run_s", 7200.0))
        # HOST credentials, not the broker's application credentials. Confusing
        # the two produces an authentication failure that looks like a network
        # problem (guide/11 §8.5), so they are separate keys and never fall back
        # to rabbit.username / rabbit.password.
        ssh_cfg = cfg.get("ssh") or {}
        self.ssh_user = ssh_cfg.get("username")
        self.ssh_password = ssh_cfg.get("password")
        self.ssh_key = ssh_cfg.get("key_file")
        self.ssh_port = int(ssh_cfg.get("port", 22))
        # The management API can report the BROKER PROCESS's memory over HTTP.
        # It answers a different question from host memory, so it is labelled
        # source=rabbitmq_api and never silently substituted (guide/11 §4).
        self.api_user = rabbit.get("username")
        self.api_password = rabbit.get("password")
        self.api_port = int(cfg.get("api_port", 15672))
        self.allow_api_fallback = bool(cfg.get("api_fallback", True))

        self.series_path = series_path
        self.summary_path = summary_path
        self._log = log or (lambda msg, colour="cyan": None)

        self._samples = []          # dicts, in arrival order
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._phase = "idle"
        self._marks = {}            # phase -> ns at which it began
        self._reason = "not started"
        self._source = None
        self._stderr_tail = []
        self._ssh = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        """Open the window at CONTROLLER START — before any worker registers and
        long before anything is published.

        The instinct is to start at dispatch, so the first sample is 'every queue
        empty'. That is a baseline of the wrong thing: one sample, on a machine
        that has already been connected to and had its queues declared and purged.
        What is actually needed is the counterfactual — this host, measured the
        same way, while the system is not running at all — and that is a stretch
        of samples, obtainable only before the run exists. It costs nothing: the
        controller is already alive and the samples land in the same file. What it
        buys is a denominator, so every later number becomes "running the system
        costs this host N MB" instead of "this host was using N MB", which is
        unfalsifiable on a box we do not otherwise control.
        """
        if not self.enabled:
            self._reason = "disabled in config"
            return
        if not self.host:
            self._reason = "no broker host configured"
            self._log(f"[BrokerRAM] {self._reason}", "yellow")
            return
        self._mark("idle")
        self._thread = threading.Thread(target=self._run, name="broker-ram",
                                        daemon=True)
        self._thread.start()

    def mark_run(self):
        """Dispatch — the host starts being asked to do something."""
        self._mark("run")

    def mark_tail(self):
        """The last collector has finished. The drain is precisely when a
        backed-up host gives memory back, so a curve that does NOT fall here is
        the signal that something is still holding units. The tail is kept short
        on purpose: too short to wait out a runtime's own garbage collection, so a
        positive tail reads as "not back to rest yet", never as proof of a leak. A
        leak is a tail still there at the NEXT run's idle phase."""
        self._mark("tail")

    def stop_and_summarize(self):
        """Close the window a second or two past the drain, then write the
        summary. Never raises."""
        if not self.enabled:
            self._write_summary()
            return
        try:
            if self._thread is not None and self._thread.is_alive():
                time.sleep(max(0.0, self.tail_s))
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
        except Exception as e:
            self._reason = f"stop failed: {e}"
        finally:
            self._close_ssh()
            self._write_summary()

    # -- sampling ----------------------------------------------------------

    def _mark(self, phase):
        with self._lock:
            if phase not in self._marks:
                self._marks[phase] = time.time_ns()
            self._phase = phase

    def _run(self):
        if self._start_ssh():
            return
        if self.allow_api_fallback:
            self._run_api()

    def _start_ssh(self):
        """One session, one remote loop, one sleep. Returns True if it ran."""
        try:
            import paramiko
        except ImportError:
            self._reason = "paramiko not installed (pip install paramiko)"
            self._log(f"[BrokerRAM] SSH unavailable: {self._reason} — "
                      f"falling back to the management API", "yellow")
            return False
        if not self.ssh_user:
            self._reason = "no broker-ram.ssh.username configured"
            self._log(f"[BrokerRAM] {self._reason} — falling back to the "
                      f"management API", "yellow")
            return False

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.host, port=self.ssh_port, username=self.ssh_user,
                           password=self.ssh_password, key_filename=self.ssh_key,
                           timeout=10, banner_timeout=10, auth_timeout=10)
            self._ssh = client
        except Exception as e:
            self._reason = f"ssh connect failed: {e}"
            self._log(f"[BrokerRAM] {self._reason} — falling back to the "
                      f"management API", "yellow")
            return False

        script = _REMOTE_SCRIPT % {
            "max_iter": max(1, int(self.max_run_s / max(0.05, self.interval_s))),
            "interval": f"{self.interval_s:g}",
        }
        # Base64 so the script survives BOTH a local argv and a remote login
        # shell — two quoting layers that otherwise mangle the awk program.
        payload = base64.b64encode(script.encode()).decode()
        try:
            _stdin, stdout, stderr = self._ssh.exec_command(
                f"echo {payload} | base64 -d | sh", get_pty=False)
        except Exception as e:
            self._reason = f"ssh exec failed: {e}"
            self._log(f"[BrokerRAM] {self._reason}", "yellow")
            return False

        # stderr on its own thread, last few lines kept: when authentication or
        # the remote shell fails, that tail is the entire diagnosis.
        threading.Thread(target=self._drain_stderr, args=(stderr,),
                         name="broker-ram-err", daemon=True).start()

        self._source = "ssh"
        self._log(f"[BrokerRAM] sampling {self.host} over ssh every "
                  f"{self.interval_s:g}s", "cyan")
        try:
            for raw in iter(stdout.readline, ""):
                if self._stop.is_set():
                    break
                self._on_ssh_line(raw)
        except Exception as e:
            self._reason = f"ssh stream ended: {e}"
        if not self._samples:
            self._reason = self._reason or ("no samples; stderr: " +
                                            " | ".join(self._stderr_tail[-3:]))
            return False
        return True

    def _drain_stderr(self, stderr):
        try:
            for line in iter(stderr.readline, ""):
                line = line.strip()
                if line:
                    self._stderr_tail.append(line)
                    del self._stderr_tail[:-5]
        except Exception:
            pass

    def _on_ssh_line(self, raw):
        parts = raw.split()
        if len(parts) != 10 or parts[0] != "RAM":
            return
        try:
            _, ts, total, free, avail, buffers, cached, sw_t, sw_f, rss = parts
            total_mb = float(total) / _KB
            avail_mb = float(avail) / _KB
            # MemTotal - MemAvailable, never - MemFree: free counts reclaimable
            # page cache as used and reads ~90% on any machine that has touched a
            # disk — always alarming, never actionable. MemAvailable is the
            # kernel's own estimate of what a new allocation could actually get,
            # so the difference is memory that is really committed (guide/11 §3).
            self._add({
                "ts_ns": int(ts),
                "source": "ssh",
                "total_mb": total_mb,
                "used_mb": total_mb - avail_mb,
                "avail_mb": avail_mb,
                "free_mb": float(free) / _KB,
                "cached_mb": (float(buffers) + float(cached)) / _KB,
                # A host that is swapping is already past the point where its
                # latency contribution is stable, so a non-zero value here
                # invalidates any latency conclusion from the same run.
                "swap_used_mb": (float(sw_t) - float(sw_f)) / _KB,
                "rabbit_rss_mb": float(rss) / _KB,
            })
        except (ValueError, ZeroDivisionError):
            return

    def _run_api(self):
        """Fallback: the management API's node memory. It is the BROKER PROCESS,
        not the host, so `used_mb` means something different here — which is
        exactly why every line carries source= and why this is never substituted
        silently. A fallback that is not labelled is worse than no fallback: it
        produces a plausible number that means something other than what the file
        name says."""
        try:
            import requests
            from requests.auth import HTTPBasicAuth
        except ImportError:
            self._reason = "requests not installed"
            return
        url = f"http://{self.host}:{self.api_port}/api/nodes"
        auth = HTTPBasicAuth(self.api_user, self.api_password)
        self._source = "rabbitmq_api"
        self._log(f"[BrokerRAM] sampling {self.host} over the management API "
                  f"every {self.interval_s:g}s (broker process, not host memory)",
                  "yellow")
        while not self._stop.is_set():
            try:
                r = requests.get(url, auth=auth, timeout=5)
                node = (r.json() or [{}])[0]
                used_mb = float(node.get("mem_used") or 0) / 1e6
                total_mb = float(node.get("mem_limit") or 0) / 1e6
                self._add({
                    "ts_ns": time.time_ns(),
                    "source": "rabbitmq_api",
                    "total_mb": total_mb,
                    "used_mb": used_mb,
                    "avail_mb": max(0.0, total_mb - used_mb),
                    "free_mb": max(0.0, total_mb - used_mb),
                    "cached_mb": 0.0,
                    "swap_used_mb": 0.0,
                    "rabbit_rss_mb": used_mb,
                })
            except Exception as e:
                self._reason = f"management API unreachable: {e}"
            self._stop.wait(self.interval_s)

    def _add(self, sample):
        """Stamp the phase as the line is written. Marks only move forward, so a
        line written before dispatch is `idle` and stays `idle`. Carrying it per
        sample makes the series self-describing: a plot can shade the run window
        without cross-referencing the summary's mark timestamps."""
        with self._lock:
            sample["phase"] = self._phase
            self._samples.append(sample)
        # Written LIVE, not buffered to the end, for the same reason
        # batch_done_ns.log is: a run that dies still leaves the series behind,
        # and the series is the part that cannot be reconstructed.
        try:
            with open(self.series_path, "a") as f:
                f.write(
                    f"{sample['ts_ns']} host={self.host} source={sample['source']} "
                    f"phase={sample['phase']} total_mb={sample['total_mb']:.1f} "
                    f"used_mb={sample['used_mb']:.1f} "
                    f"used={self._ratio(sample) * 100:.2f}% "
                    f"avail_mb={sample['avail_mb']:.1f} "
                    f"free_mb={sample['free_mb']:.1f} "
                    f"cached_mb={sample['cached_mb']:.1f} "
                    f"swap_used_mb={sample['swap_used_mb']:.1f} "
                    f"rabbit_rss_mb={sample['rabbit_rss_mb']:.1f}\n")
        except Exception:
            pass

    @staticmethod
    def _ratio(sample):
        total = sample.get("total_mb") or 0.0
        return (sample["used_mb"] / total) if total > 0 else 0.0

    def _close_ssh(self):
        try:
            if self._ssh is not None:
                self._ssh.close()
        except Exception:
            pass
        self._ssh = None

    # -- summary -----------------------------------------------------------

    def _write_summary(self):
        try:
            with open(self.summary_path, "w") as f:
                f.write("\n".join(self.summary_lines()) + "\n")
        except Exception as e:
            self._log(f"[BrokerRAM] summary write failed: {e}", "yellow")

    def summary_lines(self):
        """Four whole-window lines, then one per phase, then the comparison the
        phases exist for."""
        ts = time.time_ns()
        with self._lock:
            samples = list(self._samples)

        if not samples:
            # Still write a line. A missing file is indistinguishable from a run
            # where the host was fine (guide/11 §5).
            return [f"{ts} BROKER host={self.host} source={self._source or 'none'} "
                    f"samples=0 interval_s={self.interval_s:.3f} "
                    f"({self._reason})"]

        used = sorted(s["used_mb"] for s in samples)
        totals = [s["total_mb"] for s in samples]
        span_s = (samples[-1]["ts_ns"] - samples[0]["ts_ns"]) / 1e9
        total_mb = max(totals) if totals else 0.0

        def pctify(v):
            return (v / total_mb * 100.0) if total_mb > 0 else 0.0

        lines = [
            f"{ts} BROKER host={self.host} source={self._source} "
            f"samples={len(samples)} interval_s={self.interval_s:.3f} "
            f"span_s={span_s:.1f} total_mb={total_mb:.1f} "
            f"t_start_ns={samples[0]['ts_ns']} t_end_ns={samples[-1]['ts_ns']}",

            f"{ts} USED min_mb={used[0]:.1f} "
            f"mean_mb={sum(used) / len(used):.1f} p50_mb={_pct(used, 50):.1f} "
            f"p95_mb={_pct(used, 95):.1f} max_mb={used[-1]:.1f} "
            f"min={pctify(used[0]):.2f}% mean={pctify(sum(used) / len(used)):.2f}% "
            f"p95={pctify(_pct(used, 95)):.2f}% max={pctify(used[-1]):.2f}%",

            # DELTA measures the window's own ends. With the window opening at
            # controller start, start_mb IS the at-rest figure and growth_mb is
            # what the whole session added — a positive growth across a run that
            # drained completely means units are still buffered somewhere.
            f"{ts} DELTA start_mb={samples[0]['used_mb']:.1f} "
            f"end_mb={samples[-1]['used_mb']:.1f} "
            f"growth_mb={samples[-1]['used_mb'] - samples[0]['used_mb']:.1f} "
            f"peak_over_start_mb={used[-1] - samples[0]['used_mb']:.1f}",

            f"{ts} RABBIT "
            f"mean_rss_mb={sum(s['rabbit_rss_mb'] for s in samples) / len(samples):.1f} "
            f"max_rss_mb={max(s['rabbit_rss_mb'] for s in samples):.1f} "
            f"swap_max_mb={max(s['swap_used_mb'] for s in samples):.1f}",
        ]

        by_phase = {}
        for s in samples:
            by_phase.setdefault(s["phase"], []).append(s)
        for phase in PHASES:
            group = by_phase.get(phase)
            if not group:
                continue        # a phase with no samples is OMITTED, never zeros
            g_used = sorted(s["used_mb"] for s in group)
            g_span = (group[-1]["ts_ns"] - group[0]["ts_ns"]) / 1e9
            lines.append(
                f"{ts} PHASE phase={phase} samples={len(group)} "
                f"span_s={g_span:.3f} min_mb={g_used[0]:.1f} "
                f"mean_mb={sum(g_used) / len(g_used):.1f} "
                f"p50_mb={_pct(g_used, 50):.1f} p95_mb={_pct(g_used, 95):.1f} "
                f"max_mb={g_used[-1]:.1f} "
                f"mean={pctify(sum(g_used) / len(g_used)):.2f}% "
                f"max={pctify(g_used[-1]):.2f}% "
                f"mean_rss_mb={sum(s['rabbit_rss_mb'] for s in group) / len(group):.1f} "
                f"max_rss_mb={max(s['rabbit_rss_mb'] for s in group):.1f} "
                f"t_start_ns={group[0]['ts_ns']} t_end_ns={group[-1]['ts_ns']}")

        idle, run, tail = by_phase.get("idle"), by_phase.get("run"), by_phase.get("tail")
        if idle and run:
            # The answer the whole file was built to give: running the system
            # costs this host +X MB on average and +Y MB at peak, measured against
            # the SAME host at rest. COMPARE beats DELTA because it puts a stretch
            # of idle samples against a stretch of running ones, not one endpoint
            # against another.
            i_mean = sum(s["used_mb"] for s in idle) / len(idle)
            r_mean = sum(s["used_mb"] for s in run) / len(run)
            r_peak = max(s["used_mb"] for s in run)
            i_rss = sum(s["rabbit_rss_mb"] for s in idle) / len(idle)
            r_rss = sum(s["rabbit_rss_mb"] for s in run) / len(run)
            line = (f"{ts} COMPARE idle_mean_mb={i_mean:.1f} run_mean_mb={r_mean:.1f} "
                    f"run_minus_idle_mb={r_mean - i_mean:.1f} "
                    f"run_peak_over_idle_mb={r_peak - i_mean:.1f} "
                    f"idle_rss_mb={i_rss:.1f} run_rss_mb={r_rss:.1f} "
                    f"run_rss_over_idle_mb={r_rss - i_rss:.1f}")
            if tail:
                t_mean = sum(s["used_mb"] for s in tail) / len(tail)
                t_span = (tail[-1]["ts_ns"] - tail[0]["ts_ns"]) / 1e9
                line += (f" tail_mean_mb={t_mean:.1f} "
                         f"tail_minus_idle_mb={t_mean - i_mean:.1f} "
                         f"tail_span_s={t_span:.3f}")
            lines.append(line)
        return lines
