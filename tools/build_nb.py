"""build_nb.py — emit the result-visualization notebook (guide/08-build-pipeline.md).

The notebook is the deliverable; this script is how it stays regenerable. Never
edit the .ipynb directly — the next build silently reverts it. Fix the chart here,
then run both scripts:

    python tools/build_nb.py && python tools/run_nb.py

One markdown heading + one code cell per chart, never two charts in a cell: an
error in the first would hide the second. Code cells use r'''...''' so the Windows
path backslashes survive.
"""
import sys
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:
    sys.exit("nbformat is required: pip install nbformat nbclient ipykernel")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "visual" / "Result Visualization.ipynb"

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- cell 0
md(r"""
# Split-inference run results

Every chart here is built from the shared result format
(`guide/01-result-format.md`), so this notebook renders any conforming run
directory with **no chart-code changes** — that is the entire point of fixing the
format.

Runs are discovered under `results/`, **at any depth**: every directory holding a
`batch_done_ns.log` is one run, named after its folder. That covers both layouts
`guide/05 §1` allows — `results/<run-id>/` as the archiver writes it, and
`results/<date>/<variant>/` for a compared set. Charts are written to
`results/imgs/`.

| Input file | Feeds |
|---|---|
| `batch_done_ns.log` | C3, C9 |
| `fps_cluster_ns.log` | C2, C4 |
| `fps_cluster.log` | C1, C10, C11 |
| `utilization.log` | C8 |
| `utilization_cluster.log` | C7, C10 |
| `latency_cluster.log` | C5, C6, C10 |
| `events_ns.log` | C9 (skipped when absent) |
| `free_time.log`, `free_time_cluster.log` | C12 |
| `free_time_series.log` | C13 |
| `broker_ram_ns.log` | C14 |
| `message_size_series.log`, `message_size.log` | C15 |

Naming scheme: **cluster** (`fps_cluster*.log`, `utilization_cluster.log`,
`latency_cluster.log`, `free_time_cluster.log`) — one scheme per project, never
mixed. Validate a run before charting it:

```
python guide/validate_results.py results/<run> --names cluster
```
""")

# ---------------------------------------------------------------- setup
md("## 0 · Setup — paths, palette, chart style")
code(r'''
import re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT    = Path.cwd()
while not (ROOT / "guide").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
RESULTS = ROOT / "results"
IMG_DIR = RESULTS / "imgs"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# A run is any directory under results/ that holds the required system series.
# The search is RECURSIVE: guide/05 §1 allows both `results/<run-id>/` (what the
# archiver writes) and `results/<date>/<variant>/` for a compared set, and a
# one-level scan silently finds zero runs in the second layout.
_SKIP = {"visual", "imgs"}
_found = sorted(p.parent for p in RESULTS.rglob("batch_done_ns.log")
                if not _SKIP & set(p.relative_to(RESULTS).parts))
# Label by folder name, which is what the archiver makes unique; fall back to the
# path relative to results/ only where two nested folders share a leaf name.
_leaf = [d.name for d in _found]
RUNS = {(d.name if _leaf.count(d.name) == 1 else d.relative_to(RESULTS).as_posix()): d
        for d in _found}
RUN_ORDER = list(RUNS)
assert RUN_ORDER, f"no run directories found under {RESULTS}"
print(f"ROOT    {ROOT}\nresults {RESULTS}\nimgs    {IMG_DIR}")
print(f"runs    {RUN_ORDER}")

# ---- tokens (guide/06 §3, §5) -------------------------------------------
SURFACE = "#fcfcfb"; PAGE  = "#f9f9f7"
INK     = "#0b0b0b"; INK_2 = "#52514e"
MUTED   = "#898781"; GRID  = "#e1e0d9"; AXIS = "#c3c2b7"

# Categorical slots, taken IN ORDER — the ordering is the colourblind-safety
# mechanism, not cosmetics. Validated with:
#   python guide/validate_palette.py "#2a78d6,#eb6834,#1baf7a" light all
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
# Status colours, and they do NOT pass the CVD check as a pair (green/red are
# ~4 dE apart under deuteranopia) — no red/green pair can. That is why the one
# chart using them, C10, prints the verdict as a WORD beside every bar: colour
# alone never carries the good/bad reading. Do not "fix" this by re-tinting.
GOOD, BAD, NEUTRAL = "#0ca30c", "#d03b3b", MUTED
# Sequential blue, light -> dark. For a discrete ordered ramp on a light surface,
# start no lighter than #86b6ef.
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,      # else the PNG is transparent -> black in dark mode
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlepad": 12,
    "axes.labelsize": 10.5, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linestyle": "-", "grid.linewidth": 0.8,   # solid hairline
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "legend.fontsize": 9.5, "legend.labelcolor": INK_2,
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# The surface-coloured edge IS the gap between adjacent fills. Never a black border.
BAR_KW  = dict(edgecolor=SURFACE, linewidth=1.2)
LINE_KW = dict(linewidth=2.0, solid_capstyle="round")

SAVED = []

def finish(fig, filename, hide_spines=("top", "right")):
    for ax in fig.get_axes():
        for side in hide_spines:
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
    out = IMG_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    SAVED.append(filename)
    print(f"saved -> {out}")
    plt.show()

def label_bars(ax, bars, fmt="{:.2f}", dy=3, fontsize=9, color=INK_2):
    """Direct value labels — mandatory relief for the sub-3:1 fills (aqua,
    yellow, magenta) and useful everywhere else."""
    for bar in bars:
        h = bar.get_height()
        if h is None or np.isnan(h):
            continue
        ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=color)

def shorten(text, limit=46):
    """Trim free text for an annotation without leaving it mid-word. A blunt
    `text[:46]` cut an events_ns.log description to '...edge_only (' in the
    render — a dangling bracket reads as a rendering fault rather than as a
    label that was too long."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    if " " in cut:
        cut = cut[:cut.rindex(" ")]
    return cut.rstrip(" ([{-") + "…"

# Colour follows the ENTITY, never its rank: dicts, never palette[i] over a
# filtered list. A reader who learned "edge is orange" must not be misled by a
# later chart that happened to filter the clouds out.
RUN_COLOR  = {run: c for run, c in zip(RUN_ORDER, [S1, S2, S3, S4])}
ROLE_COLOR = {"cloud": S1, "edge": S2, "unknown": MUTED}
''')

# ---------------------------------------------------------------- parsers
md("## 1 · Log parsers\n\nOne parser core reads every file — that is what the "
   "universal line grammar buys. No bespoke regex per file.")
code(r'''
KV = re.compile(r"(\w+)=([^\s]+)")

def parse_kv_line(line):
    """-> (ts_ns, [UPPERCASE flags], {key: value})"""
    parts = line.split()
    if not parts:
        return None, [], {}
    ts = int(parts[0]) if parts[0].isdigit() else None
    return ts, [p for p in parts[1:] if "=" not in p and p.isupper()], dict(KV.findall(line))

def num(v):
    """'55.06%' -> 55.06 ; '336' -> 336.0 ; junk/None -> nan"""
    if v is None:
        return np.nan
    try:
        return float(str(v).rstrip("%"))
    except ValueError:
        return np.nan

def read_lines(path):
    path = Path(path)
    if not path.exists():
        print(f"!! missing: {path.name}")
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]

# Raw ids -> display labels AT PARSE TIME, so no chart code ever contains a raw
# queue name. Built from the data, so a run with three clusters just works.
def build_cluster_labels():
    seen = []
    for d in RUNS.values():
        for ln in read_lines(d / "fps_cluster.log"):
            c = dict(KV.findall(ln)).get("cluster")
            if c and c not in seen:
                seen.append(c)
    return {c: (f"Cluster {c.rsplit('_', 1)[-1]}" if c.rsplit('_', 1)[-1].isdigit()
                else c) for c in sorted(seen)}

CLUSTER_LABEL = build_cluster_labels()
def scope_of(flags, kv):
    """Every rolled-up file mixes cluster-scoped and SYSTEM lines; this is the
    one place that distinction is decoded."""
    if "SYSTEM" in flags:
        return "System"
    c = kv.get("cluster")
    return CLUSTER_LABEL.get(c, c) if c else None

def parse_rate_summary(run, path):                       # fps_cluster.log
    rows = []
    for ln in read_lines(path):
        ts, flags, kv = parse_kv_line(ln)
        scope = scope_of(flags, kv)
        if scope is None:
            continue
        rows.append(dict(run=run, scope=scope, fps=num(kv.get("fps")),
                         # SYSTEM carries neither steady_fps nor share. Leave them
                         # NaN — defaulting steady_fps to fps would make the System
                         # row silently look like it had a steady-state number.
                         steady_fps=num(kv.get("steady_fps")),
                         done=num(kv.get("done")), frames=num(kv.get("frames")),
                         share=num(kv.get("share"))))
    return rows

def parse_rate_timeline(run, path):                      # fps_cluster_ns.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "window_fps" not in kv:
            continue                                     # warm-up, before the window fills
        rows.append(dict(run=run, cluster=CLUSTER_LABEL.get(kv["cluster"], kv["cluster"]),
                         ts=ts, done=int(num(kv["done"])),
                         window_fps=num(kv["window_fps"])))
    return rows

def parse_batch_done(run, path):                         # batch_done_ns.log
    """The two-arity file: one column during warm-up, two after. idx increments on
    EVERY line so the unit index stays true although only 2-column rows are kept."""
    rows, idx = [], 0
    for ln in read_lines(path):
        parts = ln.split()
        idx += 1
        if len(parts) == 2:
            rows.append(dict(run=run, batch=idx, ts=int(parts[0]),
                             window_fps=float(parts[1])))
    return rows

def parse_latency(run, path):                            # latency_cluster.log
    rows = []
    for ln in read_lines(path):
        ts, flags, kv = parse_kv_line(ln)
        scope = scope_of(flags, kv)
        if scope is None or "kind" not in kv:
            continue
        rows.append(dict(run=run, scope=scope, role=kv.get("role", "all"),
                         kind=kv["kind"], n=num(kv.get("n")),
                         mean_ms=num(kv.get("mean_ms")), p50_ms=num(kv.get("p50_ms")),
                         p95_ms=num(kv.get("p95_ms")), max_ms=num(kv.get("max_ms"))))
    return rows

def parse_util_group(run, path):                         # utilization_cluster.log
    rows = []
    for ln in read_lines(path):
        ts, flags, kv = parse_kv_line(ln)
        scope = scope_of(flags, kv)
        if scope is None:
            continue
        rows.append(dict(run=run, scope=scope,
                         role=kv.get("role", "all" if "ALL" in flags or
                                     "SYSTEM" in flags else "unknown"),
                         devices=num(kv.get("devices")),
                         utilization=num(kv.get("utilization")),
                         utilization_mean=num(kv.get("utilization_mean")),
                         busy_s=num(kv.get("busy_s")), total_s=num(kv.get("total_s")),
                         packages=num(kv.get("packages"))))
    return rows

def parse_util_device(run, path):                        # utilization.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "client" not in kv:
            continue
        rows.append(dict(run=run, client=kv["client"], role=kv.get("role", "unknown"),
                         packages=num(kv.get("packages")), busy_s=num(kv.get("busy_s")),
                         total_s=num(kv.get("total_s")),
                         utilization=num(kv.get("utilization"))))
    return rows

def parse_events(run, path):                             # events_ns.log
    rows = []
    for ln in read_lines(path):
        parts = ln.split(None, 1)                        # split ONCE: descriptions have spaces
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        rows.append(dict(run=run, ts=int(parts[0]), description=parts[1]))
    return rows

def parse_free_device(run, path):                        # free_time.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "client" not in kv or "span_s" not in kv:
            continue
        rows.append(dict(run=run, client=kv["client"], role=kv.get("role", "unknown"),
                         machine=kv.get("machine", "unknown"),
                         cluster=CLUSTER_LABEL.get(kv.get("cluster"), kv.get("cluster")),
                         span_s=num(kv.get("span_s")), busy_s=num(kv.get("busy_s")),
                         free_s=num(kv.get("free_s")), free=num(kv.get("free")),
                         gaps=num(kv.get("gaps")),
                         host_idle=num(kv.get("host_idle"))))
    return rows

def parse_free_group(run, path):                         # free_time_cluster.log
    """Six line kinds in one file; the flag says which. `agg_kind` is NOT called
    `agg` — a column with a DataFrame attribute's name is shadowed and compares
    all-False without raising (guide/08 §8)."""
    rows = []
    for ln in read_lines(path):
        ts, flags, kv = parse_kv_line(ln)
        kind = ("FREE" if "FREE" in flags else "KIND" if "KIND" in flags else
                "MACHINE" if "MACHINE" in flags else
                "SYSTEM" if "SYSTEM" in flags else
                "ALL" if "ALL" in flags else "ROLE")
        rows.append(dict(run=run, agg_kind=kind,
                         scope=scope_of(flags, kv) or "System",
                         role=kv.get("role", "all"), machine=kv.get("machine"),
                         reason=kv.get("reason"), busy_kind=kv.get("kind"),
                         free=num(kv.get("free")), free_mean=num(kv.get("free_mean")),
                         free_s=num(kv.get("free_s")), busy_s=num(kv.get("busy_s")),
                         span_s=num(kv.get("span_s")), share=num(kv.get("share")),
                         host_idle=num(kv.get("host_idle"))))
    return rows

def parse_free_series(run, path):                        # free_time_series.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "t_offset_s" not in kv:
            continue
        rows.append(dict(run=run, client=kv.get("client"), role=kv.get("role", "unknown"),
                         machine=kv.get("machine"), i=int(num(kv.get("i"))),
                         t_offset_s=num(kv.get("t_offset_s")),
                         bucket_s=num(kv.get("bucket_s")), free=num(kv.get("free"))))
    return rows

def parse_broker_ram(run, path):                         # broker_ram_ns.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if ts is None or "used_mb" not in kv:
            continue
        rows.append(dict(run=run, ts=ts, phase=kv.get("phase", "run"),
                         source=kv.get("source"), total_mb=num(kv.get("total_mb")),
                         used_mb=num(kv.get("used_mb")), used=num(kv.get("used")),
                         swap_used_mb=num(kv.get("swap_used_mb")),
                         rss_mb=num(kv.get("rabbit_rss_mb"))))
    return rows

def parse_msize_series(run, path):                       # message_size_series.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "t_offset_s" not in kv or "bytes" not in kv:
            continue
        rows.append(dict(run=run, client=kv.get("client"), i=int(num(kv.get("i"))),
                         t_offset_s=num(kv.get("t_offset_s")),
                         mb=num(kv.get("mb"))))
    return rows

def parse_msize_summary(run, path):                      # message_size.log
    rows = []
    for ln in read_lines(path):
        ts, _, kv = parse_kv_line(ln)
        if "total_mb" not in kv:
            continue
        rows.append(dict(run=run, client=kv.get("client"), role=kv.get("role"),
                         mode=kv.get("mode"), compress=kv.get("compress"),
                         n=num(kv.get("n")), total_mb=num(kv.get("total_mb")),
                         mean_mb=num(kv.get("mean_mb")), p50_mb=num(kv.get("p50_mb")),
                         p95_mb=num(kv.get("p95_mb")), max_mb=num(kv.get("max_mb")),
                         min_mb=num(kv.get("min_mb")),
                         rate_mb_s=num(kv.get("rate_mb_s")),
                         per_frame_mb=num(kv.get("per_frame_mb"))))
    return rows
''')

# ---------------------------------------------------------------- load
md("## 2 · Load every file, and print what was loaded\n\nShapes are a cheap "
   "smoke test: a frame with 0 rows means a parser silently matched nothing — "
   "usually a filename or label-map mismatch, not an empty run.")
code(r'''
# An EMPTY list becomes a DataFrame with NO columns, so a later `df.run` raises
# KeyError instead of returning an empty frame. Always declare the columns.
COLS = {
    "rate":   ["run", "scope", "fps", "steady_fps", "done", "frames", "share"],
    "tl":     ["run", "cluster", "ts", "done", "window_fps"],
    "batch":  ["run", "batch", "ts", "window_fps"],
    "lat":    ["run", "scope", "role", "kind", "n", "mean_ms", "p50_ms", "p95_ms", "max_ms"],
    "utg":    ["run", "scope", "role", "devices", "utilization", "utilization_mean",
               "busy_s", "total_s", "packages"],
    "utd":    ["run", "client", "role", "packages", "busy_s", "total_s", "utilization"],
    "ev":     ["run", "ts", "description"],
    "ftd":    ["run", "client", "role", "machine", "cluster", "span_s", "busy_s",
               "free_s", "free", "gaps", "host_idle"],
    "ftg":    ["run", "agg_kind", "scope", "role", "machine", "reason", "busy_kind",
               "free", "free_mean", "free_s", "busy_s", "span_s", "share", "host_idle"],
    "fts":    ["run", "client", "role", "machine", "i", "t_offset_s", "bucket_s", "free"],
    "ram":    ["run", "ts", "phase", "source", "total_mb", "used_mb", "used",
               "swap_used_mb", "rss_mb"],
    "msz":    ["run", "client", "i", "t_offset_s", "mb"],
    "mszs":   ["run", "client", "role", "mode", "compress", "n", "total_mb", "mean_mb",
               "p50_mb", "p95_mb", "max_mb", "min_mb", "rate_mb_s", "per_frame_mb"],
}

FILES = {
    "rate":  ("fps_cluster.log",          parse_rate_summary),
    "tl":    ("fps_cluster_ns.log",       parse_rate_timeline),
    "batch": ("batch_done_ns.log",        parse_batch_done),
    "lat":   ("latency_cluster.log",      parse_latency),
    "utg":   ("utilization_cluster.log",  parse_util_group),
    "utd":   ("utilization.log",          parse_util_device),
    "ev":    ("events_ns.log",            parse_events),
    "ftd":   ("free_time.log",            parse_free_device),
    "ftg":   ("free_time_cluster.log",    parse_free_group),
    "fts":   ("free_time_series.log",     parse_free_series),
    "ram":   ("broker_ram_ns.log",        parse_broker_ram),
    "msz":   ("message_size_series.log",  parse_msize_series),
    "mszs":  ("message_size.log",         parse_msize_summary),
}

def load_all():
    """list[dict] per file, concatenated across runs, DataFrame built ONCE.
    Never pd.concat in a loop."""
    acc = {k: [] for k in FILES}
    for run, d in RUNS.items():
        for key, (name, parser) in FILES.items():
            acc[key] += parser(run, d / name)
    return {k: pd.DataFrame(rows, columns=COLS[k]) for k, rows in acc.items()}

D = load_all()
df_rate, df_tl, df_batch = D["rate"], D["tl"], D["batch"]
df_lat, df_utg, df_utd   = D["lat"], D["utg"], D["utd"]
df_events                = D["ev"]
df_ftd, df_ftg, df_fts   = D["ftd"], D["ftg"], D["fts"]
df_ram, df_msz, df_mszs  = D["ram"], D["msz"], D["mszs"]

NAMES = {"rate": "rate summary", "tl": "rate timeline", "batch": "batch timeline",
         "lat": "latency", "utg": "utilization/cluster", "utd": "utilization/device",
         "ev": "events", "ftd": "free time/device", "ftg": "free time/rollup",
         "fts": "free time/series", "ram": "broker ram", "msz": "message size/series",
         "mszs": "message size/summary"}
for k, df in D.items():
    print(f"{NAMES[k]:<22} {df.shape}")

CLUSTERS = [c for c in CLUSTER_LABEL.values()]
SCOPES   = CLUSTERS + ["System"]
print(f"\nclusters {CLUSTERS}")

def first_run_with(df):
    """The first run, in RUN_ORDER, that actually has rows in `df`.

    Optional measurements (guide/10-12) are per-run: a directory can be complete
    for throughput and carry an empty message_size.log because the feature was
    off that day. A single-run chart that hardcodes RUN_ORDER[0] then fails on a
    perfectly valid set of runs, so ask which run has the data instead."""
    have = set(df.run) if len(df) else set()
    return next((r for r in RUN_ORDER if r in have), None)
''')

# ---------------------------------------------------------------- assertions
md("## 3 · Verify the assumptions before charting them\n\nCompute what is about "
   "to be asserted visually and **branch on the result**. Plotting two identical "
   "series draws two perfectly overlapping lines and the reader cannot tell "
   "whether the second is hidden or missing.")
code(r'''
# Same workload? Otherwise the runs are not comparable and no chart will say so —
# it will just draw two bars.
print("System totals per run:")
print(df_rate[df_rate.scope == "System"][["run", "done", "frames", "fps"]]
      .to_string(index=False))

# Utilization above 100% is a MEASUREMENT bug (overlapping intervals summed), not
# a fast device. Must be empty.
bad = df_utd[df_utd.utilization > 100]
print(f"\ndevices reporting >100% utilization: {len(bad)}"
      + ("  <-- measurement bug, fix the instrument" if len(bad) else "  (ok)"))

# Sum(service) should equal busy_s for the matching role — the two are supposed to
# be instrumented at the same points (guide/04 §2.1).
svc = df_lat[df_lat.kind == "service"]
if len(svc) and len(df_utg):
    j = (svc.merge(df_utg[df_utg.role != "all"], on=["run", "scope", "role"],
                   suffixes=("_lat", "_utg")))
    if len(j):
        j = j.assign(sum_service_s=j.n * j.mean_ms / 1000.0)
        j["rel_err_%"] = (j.sum_service_s - j.busy_s).abs() / j.busy_s * 100
        print("\nSigma(service) vs busy_s  (should agree; >1% means the two are "
              "instrumented at different points):")
        print(j[["run", "scope", "role", "sum_service_s", "busy_s", "rel_err_%"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# pipeline >> service is pure buffering, and the single most actionable reading in
# the latency file: lower the hand-off queue depth, throughput does not depend on it.
pipe = df_lat[df_lat.kind == "pipeline"]
if len(svc) and len(pipe):
    cmp_ = (svc.merge(pipe, on=["run", "scope", "role"], suffixes=("_svc", "_pipe"))
            .assign(ratio=lambda d: d.mean_ms_pipe / d.mean_ms_svc))
    worst = cmp_.ratio.max()
    print(f"\nworst pipeline/service ratio: {worst:.2f}x"
          + ("  <-- buffering dominates; lower the hand-off queue depth"
             if worst > 1.5 else "  (little buffering)"))

MULTI_RUN = len(RUN_ORDER) > 1

# Grouped-bar geometry, derived from the run count rather than assumed. A width
# fixed for two runs makes three of them wider than the category slot, and bars
# from neighbouring categories then touch — which reads as one group and is the
# kind of chart bug nobody questions, because bars are supposed to be adjacent.
BAR_GAP = 0.03                              # surface-coloured gap, never a border
BAR_W   = max(0.08, (0.82 - BAR_GAP * (len(RUN_ORDER) - 1)) / len(RUN_ORDER))
bar_off = lambda i: (i - (len(RUN_ORDER) - 1) / 2) * (BAR_W + BAR_GAP)

# Two runs are comparable only if they did the SAME amount of work (guide/05 §6);
# otherwise the comparison bar reads as a configuration effect when it is really a
# different workload, and no chart says so on its own. So pick the pair here, from
# equal done/frames, rather than letting C10 take whichever two sort first.
_tot = (df_rate[df_rate.scope == "System"].set_index("run")[["done", "frames"]]
        .to_dict("index"))
COMPARE_PAIR = next((
    (a, b) for i, a in enumerate(RUN_ORDER) for b in RUN_ORDER[i + 1:]
    if a in _tot and b in _tot and _tot[a] == _tot[b]), None)
if COMPARE_PAIR:
    print(f"\n{len(RUN_ORDER)} run(s) loaded -> C10 compares "
          f"{COMPARE_PAIR[0]} vs {COMPARE_PAIR[1]} (identical done/frames)")
elif MULTI_RUN:
    print(f"\n{len(RUN_ORDER)} run(s) loaded -> C10 skipped: no two runs share a "
          f"workload, so any comparison between them would be unlike-for-unlike "
          f"(guide/05 §6)")
else:
    print("\n1 run loaded -> comparison charts skipped (C10 needs two)")
''')


# ============================================================ CHARTS
def chart(heading, note, body):
    md(f"## {heading}\n\n{note}")
    code(body)


chart("C1 · Throughput by cluster and system",
      "The headline comparison. Cluster bars do **not** sum to the System bar and "
      "should not be expected to — each scope divides by its own span "
      "(`guide/01 §3.3`), so the sum runs above the system value whenever one "
      "cluster finishes early.",
      r'''
piv = (df_rate.pivot_table(index="scope", columns="run", values="fps")
       .reindex(SCOPES).dropna(how="all"))[RUN_ORDER]

x, width = np.arange(len(piv)), BAR_W
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for i, run in enumerate(RUN_ORDER):
    off = bar_off(i)
    bars = ax.bar(x + off, piv[run], width, label=run,
                  color=RUN_COLOR[run], **BAR_KW)
    label_bars(ax, bars, fmt="{:.2f}")

ax.set_xticks(x, list(piv.index))
ax.set_ylabel("Throughput (frames/s)")
ax.set_ylim(0, np.nanmax(piv.to_numpy()) * 1.18)
ax.set_title("Throughput by cluster and system")
ax.grid(axis="x", visible=False)
if MULTI_RUN:
    ax.legend(loc="upper left")
ax.annotate("cluster values are not additive: each divides by its own span",
            xy=(0.5, -0.16), xycoords="axes fraction", ha="center",
            fontsize=9.5, color=MUTED)
finish(fig, "01_rate_by_cluster.png")
''')

chart("C2 · Rolling window rate per cluster",
      "Faceted by run so four lines never share one axis. `sharey=True` is "
      "mandatory for small multiples — independent scales defeat the purpose. "
      "Clusters end at different x because the axis is *per-cluster* completions; "
      "that is correct.",
      r'''
fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(6.6 * len(RUN_ORDER), 4.6),
                         sharey=True, squeeze=False)
axes = axes[0]
ymax = (df_tl.window_fps.max() if len(df_tl) else 1.0) * 1.10

for ax, run in zip(axes, RUN_ORDER):
    sub_run = df_tl[df_tl.run == run]
    ends = []
    for j, cluster in enumerate(CLUSTERS):
        s = sub_run[sub_run.cluster == cluster].sort_values("done")
        if s.empty:
            continue
        colour = [S1, S2, S3, S4][j % 4]
        ax.plot(s.done, s.window_fps, color=colour, label=cluster, **LINE_KW)
        ends.append((s.iloc[-1].done, s.iloc[-1].window_fps, colour))

    # Selective label: the endpoint only. Balanced clusters converge, so their
    # end labels land on the same pixel and overprint into an unreadable smear —
    # place them after the loop and step apart the ones that are too close.
    ends.sort(key=lambda e: e[1])
    dy, prev_y = 0, None
    for x, y, colour in ends:
        dy = dy + 11 if prev_y is not None and (y - prev_y) < ymax * 0.045 else 0
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(6, dy),
                    textcoords="offset points", va="center",
                    fontsize=9, color=colour)
        prev_y = y
    ax.set_title(run)
    ax.set_xlabel("Completed batches (per cluster)")
    ax.set_ylim(0, ymax)
    ax.grid(axis="x", visible=False)

axes[0].set_ylabel("Rolling window rate (frames/s)")
if len(CLUSTERS) > 1:
    axes[0].legend(loc="upper left")           # one legend, on the first panel
fig.suptitle("Rolling window rate per cluster", fontsize=14,
             fontweight="semibold", color=INK, y=1.02)
fig.tight_layout()
finish(fig, "02_window_rate_by_cluster.png")
''')

chart("C3 · System-wide rolling window rate",
      "The authoritative series: it counts every arrival regardless of body, so a "
      "mis-tagged batch cannot move it. The line legitimately starts at x≈16 — the "
      "first `W-1` units have no window yet, and absence is used rather than `0.00` "
      "so a genuine stall stays distinguishable.",
      r'''
fig, ax = plt.subplots(figsize=(12, 4.6))
for run in RUN_ORDER:
    s = df_batch[df_batch.run == run].sort_values("batch")
    if s.empty:
        continue
    m = s.window_fps.mean()
    # The summary statistic lives IN THE LEGEND LABEL, not in a floating
    # annotation: it stays attached to its series and can never collide.
    ax.plot(s.batch, s.window_fps, color=RUN_COLOR[run],
            label=f"{run}  (mean {m:.2f})", alpha=0.9, **LINE_KW)
    ax.axhline(m, color=RUN_COLOR[run], linewidth=1.0, alpha=0.45)

ax.set_xlabel("Completed batch index (system-wide)")
ax.set_ylabel("Rolling window rate (frames/s)")
ax.set_title("System-wide rolling window rate over the run  (W = 16 batches)")
ax.set_ylim(0, df_batch.window_fps.max() * 1.10)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left")
finish(fig, "03_system_window_rate.png")
''')

chart("C4 · Window rate distribution",
      "Stability, not just the average: a high mean with a wide box is a worse "
      "result than a slightly lower mean with a tight one. Box = IQR, whiskers "
      "1.5×IQR, outliers suppressed.",
      r'''
positions, data, colours = [], [], []
for ci, cluster in enumerate(CLUSTERS):
    for ri, run in enumerate(RUN_ORDER):
        vals = df_tl[(df_tl.cluster == cluster) & (df_tl.run == run)].window_fps.values
        if not len(vals):
            continue
        positions.append(ci + bar_off(ri))
        data.append(vals)
        colours.append(RUN_COLOR[run])

fig, ax = plt.subplots(figsize=(8.6, 4.9))
bp = ax.boxplot(data, positions=positions, widths=BAR_W * 0.78, patch_artist=True,
                showfliers=False,
                medianprops=dict(color=SURFACE, linewidth=1.8),   # reads on the fill
                whiskerprops=dict(color=AXIS, linewidth=1.0),
                capprops=dict(color=AXIS, linewidth=1.0))
for patch, c in zip(bp["boxes"], colours):
    patch.set_facecolor(c); patch.set_edgecolor(SURFACE); patch.set_linewidth(1.2)

caps = []
for pos, vals in zip(positions, data):
    # Label ABOVE the whisker cap: p75 sits inside the whisker and the text
    # would land on top of it.
    q1, q3 = np.percentile(vals, [25, 75])
    cap = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
    caps.append(cap)
    ax.annotate(f"{vals.mean():.1f}", xy=(pos, cap), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=9, color=INK_2)

# Headroom for those labels. Autoscale stops at the whisker cap, so the text —
# which is drawn 7pt ABOVE it — lands on the top spine and is clipped. The
# render showed exactly that; charts are checked by looking at them.
lo, hi = ax.get_ylim()
ax.set_ylim(lo, max(hi, max(caps) + (hi - lo) * 0.08))

# ax.boxplot returns no legend handles — build Rectangle proxies. "best"
# rather than a fixed corner: which corner is free depends on where the fastest
# cluster lands, and a pinned legend covered a mean label in the render.
if MULTI_RUN:
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=RUN_COLOR[r]) for r in RUN_ORDER],
              RUN_ORDER, loc="best")
ax.set_xticks(range(len(CLUSTERS)), CLUSTERS)
ax.set_ylabel("Rolling window rate (frames/s)")
ax.set_xlabel("Cluster")
ax.set_title("Window rate distribution  (box = IQR, label = mean, outliers hidden)")
ax.grid(axis="x", visible=False)
finish(fig, "04_window_rate_distribution.png")
''')

chart("C5 · Service latency by role — two panels",
      "**This is the dual-axis replacement.** `sharey` is deliberately omitted: "
      "edge and cloud differ by roughly an order of magnitude and one shared axis "
      "would flatten the smaller. Each panel is honestly its own scale and the "
      "titles say which is which. Charted on `kind=service`, never `pipeline` — "
      "`pipeline` measures the queue depth, not the device.",
      r'''
svc = df_lat[(df_lat.kind == "service") & (df_lat.role != "all")]
roles = [r for r in ["cloud", "edge"] if r in set(svc.role)]

if not roles:
    print("no service latency rows — skipping C5")
else:
    fig, axes = plt.subplots(1, len(roles), figsize=(5.9 * len(roles), 4.8),
                             squeeze=False)          # NOT sharey — that is the point
    axes = axes[0]
    x, width = np.arange(len(CLUSTERS)), BAR_W
    for ax, role in zip(axes, roles):
        piv = (svc[svc.role == role]
               .pivot_table(index="scope", columns="run", values="mean_ms")
               .reindex(CLUSTERS)) / 1000.0          # ms -> s ONCE, at the pivot
        for i, run in enumerate(RUN_ORDER):
            if run not in piv:
                continue
            off = bar_off(i)
            b = ax.bar(x + off, piv[run], width, label=run,
                       color=RUN_COLOR[run], **BAR_KW)
            label_bars(ax, b, fmt="{:.2f}s")
        ax.set_xticks(x, CLUSTERS)
        ax.set_title(f"{role.capitalize()} devices")
        ax.set_ylim(0, np.nanmax(piv.to_numpy()) * 1.20)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean service latency (s)")
    if MULTI_RUN:
        axes[0].legend(loc="upper left")
    fig.suptitle("Mean service latency by device role  (lower is better; "
                 "panels have independent scales)",
                 fontsize=13.5, fontweight="semibold", color=INK, y=1.02)
    fig.tight_layout()
    finish(fig, "05_service_latency_by_role.png")
''')

chart("C6 · End-to-end latency profile",
      "The tail, not just the average. `e2e` spans two machines by definition, so "
      "it inherits any offset between their clocks — **indicative, not exact**. "
      "`sharey=True` here because it is the same measure across scopes.",
      r'''
e2e = df_lat[df_lat.kind == "e2e"]
stats = [("mean_ms", "Mean"), ("p50_ms", "p50"), ("p95_ms", "p95"), ("max_ms", "Max")]
scopes = [s for s in SCOPES if s in set(e2e.scope)]

if not scopes:
    print("no e2e rows — skipping C6")
else:
    fig, axes = plt.subplots(1, len(scopes), figsize=(4.9 * len(scopes), 4.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    x, width = np.arange(len(stats)), BAR_W
    ymax = np.nanmax(e2e[[c for c, _ in stats]].to_numpy()) / 1000.0
    # 24 bars want a short label, but "{:.0f}" on a 10-second e2e prints 13 and
    # 14 for 13.4 s and 13.7 s — the reader sees a gap that is not there. Pick
    # the precision from the magnitude rather than pinning it: a run whose e2e
    # is in minutes gets the compact form, a fast one keeps its tenth.
    e2e_fmt = "{:.0f}" if ymax >= 100 else "{:.1f}"
    for ax, scope in zip(axes, scopes):
        sub = e2e[e2e.scope == scope].set_index("run")
        for i, run in enumerate(RUN_ORDER):
            if run not in sub.index:
                continue
            vals = [float(np.ravel(sub.loc[run, col])[0]) / 1000.0 for col, _ in stats]
            off = bar_off(i)
            b = ax.bar(x + off, vals, width, label=run, color=RUN_COLOR[run], **BAR_KW)
            label_bars(ax, b, fmt=e2e_fmt, fontsize=8.5)
        ax.set_xticks(x, [lbl for _, lbl in stats])
        ax.set_title(scope)
        ax.set_ylim(0, ymax * 1.16)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("End-to-end latency (s)")
    if MULTI_RUN:
        axes[0].legend(loc="upper left")
    fig.suptitle("End-to-end latency profile  (lower is better · indicative — "
                 "spans two clocks)",
                 fontsize=13.5, fontweight="semibold", color=INK, y=1.02)
    fig.tight_layout()
    finish(fig, "06_e2e_latency_profile.png")
''')

chart("C7 · Utilization by cluster and role",
      "Is the work split correctly? Bars are the **pooled** ratio (Σbusy/Σtotal). "
      "The dots mark `utilization_mean`, the plain mean of the per-device ratios — "
      "**when the two diverge the group is imbalanced**, and that divergence is the "
      "signal the second number exists for.",
      r'''
# The cluster "all" rows are in here on purpose: they are the only scope besides
# SYSTEM that carries utilization_mean, so without them the pooled-vs-mean
# divergence — the whole reason the second number exists — has nowhere to show.
rows = [(s, r) for s in CLUSTERS for r in ["all", "cloud", "edge"]] + [("System", "all")]
rows = [(s, r) for s, r in rows
        if len(df_utg[(df_utg.scope == s) & (df_utg.role == r)])]
labels = [f"{s.replace('Cluster ', 'C')}\n{r}" for s, r in rows]

idx = df_utg.set_index(["run", "scope", "role"])
x, width = np.arange(len(rows)), BAR_W
fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(rows)), 4.9))

for i, run in enumerate(RUN_ORDER):
    vals, means = [], []
    for s, r in rows:
        try:
            vals.append(float(np.ravel(idx.loc[(run, s, r), "utilization"])[0]))
            means.append(float(np.ravel(idx.loc[(run, s, r), "utilization_mean"])[0]))
        except KeyError:
            vals.append(np.nan); means.append(np.nan)
    off = bar_off(i)
    b = ax.bar(x + off, vals, width, label=run, color=RUN_COLOR[run], **BAR_KW)
    label_bars(ax, b, fmt="{:.1f}%", fontsize=9 if len(RUN_ORDER) < 3 else 7.5)
    ax.plot(x + off, means, linestyle="none", marker="o", markersize=5,
            markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=1.3,
            label="mean of per-device ratios" if i == 0 else None)

ax.set_xticks(x, labels)                    # two-line labels beat rotation
ax.set_ylabel("Utilization (%)")
ax.set_ylim(0, 118)                         # percentages: fix the ceiling, never autoscale
ax.set_title("Device utilization by cluster and role")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")                # upper-left is where the tall bars are
finish(fig, "07_utilization_by_role.png")
''')

chart("C8 · Per-device utilization",
      "Straggler hunting. Sorted by `[role, value]` so classes stay blocked and "
      "rank within a class is still visible; ticks number **within** each role. A "
      "bar above 100% is a measurement bug, not a fast device — it is deliberately "
      "not clipped.",
      r'''
fig, axes = plt.subplots(1, len(RUN_ORDER), figsize=(7.0 * len(RUN_ORDER), 4.9),
                         sharey=True, squeeze=False)
axes = axes[0]
for ax, run in zip(axes, RUN_ORDER):
    sub = (df_utd[df_utd.run == run]
           .sort_values(["role", "utilization"], ascending=[True, False])
           .reset_index(drop=True))
    if sub.empty:
        continue
    pos = np.arange(len(sub))
    b = ax.bar(pos, sub.utilization,
               color=[ROLE_COLOR.get(r, MUTED) for r in sub.role],
               width=0.72, **BAR_KW)
    label_bars(ax, b, fmt="{:.0f}", fontsize=8, color=MUTED)
    # cumcount numbers WITHIN each role -> C1 C2 C3 E1 E2. A running enumerate
    # gives C1 C2 C3 E4 E5, which reads as missing devices.
    ticks = sub.groupby("role").cumcount() + 1
    ax.set_xticks(pos, [f"{r[0].upper()}{n}" for r, n in zip(sub.role, ticks)],
                  fontsize=8.5)
    ax.set_title(f"{run}  —  mean {sub.utilization.mean():.1f}%")
    ax.set_xlabel("Device  (C = cloud, E = edge)")
    ax.set_ylim(0, 118)
    ax.grid(axis="x", visible=False)

present = [r for r in ["cloud", "edge"] if r in set(df_utd.role)]
axes[0].set_ylabel("Utilization (%)")
axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[r]) for r in present],
               [r.capitalize() for r in present], loc="upper right")
fig.suptitle("Per-device utilization", fontsize=14, fontweight="semibold",
             color=INK, y=1.02)
fig.tight_layout()
finish(fig, "08_device_utilization.png")
''')

chart("C9 · Control events over the timeline",
      "Did the adaptive routing decisions move throughput? Both files are on the "
      "**server's** clock, which is the whole reason for the one-clock rule — the "
      "overlay would be meaningless otherwise. Skipped, not faked, when the run "
      "recorded no events.",
      r'''
if df_events.empty:
    print("no control events recorded — skipping C9 (the run had no route flips, "
          "or the adaptive controller was off)")
else:
    fig, axes = plt.subplots(len(RUN_ORDER), 1,
                             figsize=(13, 3.4 * len(RUN_ORDER)),
                             sharex=False, squeeze=False)   # each run has its own duration
    axes = axes[:, 0]
    ymax = df_batch.window_fps.max() * 1.15
    for ax, run in zip(axes, RUN_ORDER):
        s = df_batch[df_batch.run == run].sort_values("ts")
        if s.empty:
            continue
        t0 = int(s.ts.iloc[0])              # ns stay int; subtract a baseline BEFORE
        ax.plot((s.ts - t0) / 1e9, s.window_fps,   # dividing, or float precision loses ns
                color=RUN_COLOR[run], label=f"{run} window rate", **LINE_KW)
        ev = df_events[df_events.run == run]
        for j, (_, e) in enumerate(ev.iterrows()):
            # Event rules are chrome, not data: muted hairlines, never a series colour.
            ax.axvline((int(e.ts) - t0) / 1e9, color=MUTED, linewidth=1.0)
            ax.annotate(shorten(e.description), xy=((int(e.ts) - t0) / 1e9, ymax),
                        xytext=(3, -10 - 12 * (j % 3)),      # stagger: events cluster
                        textcoords="offset points", fontsize=8, color=MUTED, va="top")
        ax.set_title(f"{run}  —  {len(ev)} control event(s)")
        ax.set_ylabel("Window rate (frames/s)")
        ax.set_ylim(0, ymax)
        ax.grid(axis="x", visible=False)
        # NOT upper left: the event labels hang from the top of the axes and the
        # first one starts at x≈0 on a run whose first flip comes early, which
        # put the description straight through the legend text in the render.
        # The series sits high, so the bottom of the panel is the free space.
        ax.legend(loc="lower left")
    axes[-1].set_xlabel("Seconds since first completion")
    fig.suptitle("Throughput response to control events", fontsize=14,
                 fontweight="semibold", color=INK, y=1.01)
    fig.tight_layout()
    finish(fig, "09_events_over_time.png")
''')

chart("C10 · Run comparison — verdict bar",
      "**Colour keys to the verdict, never to the sign.** A +29% latency move is a "
      "regression; painting it like a +21% throughput gain is a lie the reader "
      "cannot detect. Because colour now carries status semantics, the verdict is "
      "also spelled out in text. Utilization is marked direction-neutral: lower "
      "utilization at higher throughput is a *better* system.",
      r'''
if COMPARE_PAIR is None:
    print("skipping C10 — no pair of runs with the same workload to compare")
else:
    A, B = COMPARE_PAIR
    def one(df, mask, col):
        v = df[mask][col]
        return float(v.iloc[0]) if len(v) else np.nan

    sysrate = lambda r: one(df_rate, (df_rate.run == r) & (df_rate.scope == "System"), "fps")
    syse2e  = lambda r, c: one(df_lat, (df_lat.run == r) & (df_lat.scope == "System")
                               & (df_lat.kind == "e2e"), c)
    sysutil = lambda r: one(df_utg, (df_utg.run == r) & (df_utg.scope == "System"), "utilization")

    # goal: +1 higher is better, -1 lower is better, 0 direction-neutral
    metrics = [("System throughput", sysrate(A), sysrate(B), "{:.2f} f/s", "higher is better", +1),
               ("Mean E2E latency", syse2e(A, "mean_ms") / 1000, syse2e(B, "mean_ms") / 1000,
                "{:.1f} s", "lower is better", -1),
               ("p95 E2E latency", syse2e(A, "p95_ms") / 1000, syse2e(B, "p95_ms") / 1000,
                "{:.1f} s", "lower is better", -1),
               ("System utilization", sysutil(A), sysutil(B), "{:.1f} %",
                "workload dependent", 0)]
    metrics = [m for m in metrics if np.isfinite(m[1]) and np.isfinite(m[2]) and m[2] != 0]

    pct, colours, verdicts = [], [], []
    for _, a, b, _, _, goal in metrics:
        p = (a / b - 1) * 100
        score = goal * np.sign(p)
        pct.append(p)
        colours.append(GOOD if score > 0 else BAD if score < 0 else NEUTRAL)
        verdicts.append("better" if score > 0 else "worse" if score < 0
                        else ("no change" if p == 0 else "neutral"))

    y = np.arange(len(metrics))[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(y, pct, height=0.5, color=colours, **BAR_KW)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    for yi, p, v in zip(y, pct, verdicts):
        ax.annotate(f"{p:+.1f}%  {v}", xy=(p, yi), xytext=(6 if p >= 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if p >= 0 else "right",
                    fontsize=10, fontweight="semibold", color=INK)
    # Absolute values live in the tick label; a separate column would collide.
    ax.set_yticks(y, [f"{n}\nA {f.format(a)}  ·  B {f.format(b)}\n({h})"
                      for n, a, b, f, h, _ in metrics], fontsize=9.5)
    ax.tick_params(axis="y", colors=INK_2)
    lim = max(abs(p) for p in pct) * 1.9 + 4
    ax.set_xlim(-lim, lim)                  # symmetric, or bar lengths lie about the ratio
    ax.set_xlabel(f"Change of {A} relative to {B} (%)")
    ax.set_title(f"{A} vs {B} — headline metrics  ({B} = baseline)")
    ax.annotate("Colour marks the verdict (green better / red worse), not the sign "
                "of the change", xy=(0.5, -0.22), xycoords="axes fraction",
                ha="center", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    finish(fig, "10_run_comparison.png", hide_spines=("top", "right", "left"))
''')

chart("C11 · Hero stat tile",
      "When the answer is one number, a one-bar bar chart is always wrong. The "
      "number **is** the chart.",
      r'''
run = RUN_ORDER[0]
sysrow = df_rate[(df_rate.run == run) & (df_rate.scope == "System")]
fps = float(sysrow.fps.iloc[0]) if len(sysrow) else float("nan")
done = int(sysrow.done.iloc[0]) if len(sysrow) else 0
frames = int(sysrow.frames.iloc[0]) if len(sysrow) else 0

fig, ax = plt.subplots(figsize=(4.6, 2.4))
ax.axis("off")
ax.text(0, 0.78, "System throughput", fontsize=11, color=INK_2)
ax.text(0, 0.34, f"{fps:.2f}", fontsize=40, color=INK, fontweight="semibold")
ax.text(0.56, 0.40, "frames/s", fontsize=13, color=MUTED, transform=ax.transAxes)
ax.text(0, 0.06, f"{done} batches · {frames} frames · run {run}",
        fontsize=10, color=MUTED)
finish(fig, "00_hero_throughput.png", hide_spines=())
''')

chart("C12 · Free time by device — where the run went",
      "Free time is the wall clock in which a device did **nothing at all**, "
      "computed as the run span minus the *union* of every lane's busy intervals. "
      "It is neither utilization nor `1 − utilization`: a back-pressure wait inside "
      "a unit window counts as busy for utilization and free here, and capture work "
      "on another lane counts as nothing there and busy here. Every bar is the same "
      "height because every bar is that device's whole run.",
      r'''
run = first_run_with(df_ftd)
if run is None:
    print("no free-time reports — skipping C12 (feature disabled, or no device reported)")
else:
    sub = (df_ftd[df_ftd.run == run]
           .sort_values(["role", "free"], ascending=[True, False]).reset_index(drop=True))
    pos = np.arange(len(sub))
    busy_pct = sub.busy_s / sub.span_s * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    b1 = ax.bar(pos, busy_pct, width=0.72, color=S1, label="busy (merged)", **BAR_KW)
    ax.bar(pos, sub.free, width=0.72, bottom=busy_pct, color=S4,
           label="free", **BAR_KW)
    # Yellow is a sub-3:1 fill on this surface, so its direct label is obligatory
    # relief, not decoration (guide/06 §3).
    for p, busy, free_pct in zip(pos, busy_pct, sub.free):
        ax.annotate(f"{free_pct:.0f}%", xy=(p, 100), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ticks = sub.groupby("role").cumcount() + 1
    ax.set_xticks(pos, [f"{r[0].upper()}{n}" for r, n in zip(sub.role, ticks)],
                  fontsize=8.5)
    ax.set_xlabel("Device  (C = cloud, E = edge)")
    ax.set_ylabel("Share of the device's run (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"{run} — busy vs free per device  (label = free %)")
    ax.grid(axis="x", visible=False)
    # Every bar spans the full height, so ANY in-axes legend sits on a mark.
    # Put it under the axis instead of picking the least-bad overlap.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    # Why the fleet was free. Reasons are attributed in a fixed published priority
    # so they sum to exactly the free time; whatever no reason covered arrives as
    # `unaccounted` — the overhead between instrumented stages — rather than being
    # dropped.
    ax = axes[1]
    reasons = (df_ftg[(df_ftg.run == run) & (df_ftg.agg_kind == "FREE")]
               .groupby("reason", as_index=False).free_s.sum()
               .sort_values("free_s", ascending=True))
    if reasons.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "no free-time reason breakdown", ha="center",
                color=MUTED, fontsize=10)
    else:
        total = reasons.free_s.sum()
        bars = ax.barh(np.arange(len(reasons)), reasons.free_s, height=0.6,
                       color=S3, **BAR_KW)
        for i, (v, r) in enumerate(zip(reasons.free_s, reasons.reason)):
            ax.annotate(f"{v / total * 100:.1f}%", xy=(v, i), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color=INK_2)
        ax.set_yticks(np.arange(len(reasons)), reasons.reason)
        ax.set_xlim(0, reasons.free_s.max() * 1.28)
        ax.set_xlabel("Fleet free time (s)")
        ax.set_title("Why the fleet was free")
        ax.grid(axis="y", visible=False)

    fig.suptitle("Free time — a capacity measurement, not a performance one",
                 fontsize=14, fontweight="semibold", color=INK, y=1.02)
    fig.tight_layout()
    finish(fig, "12_free_time_by_device.png")
''')

chart("C13 · When each device was idle",
      "A `(device, time-bucket, free %)` heat map. Read it against C3 on the same "
      "time axis: a band of free time on one device that lines up with a throughput "
      "dip names the stage that stalled. Sequential ramp — one hue, light→dark, "
      "never a rainbow.",
      r'''
run = first_run_with(df_fts)
if run is None:
    print("no free-time series — skipping C13")
else:
    sub = df_fts[df_fts.run == run]
    order = sub.drop_duplicates("client").sort_values(["role", "client"])
    piv = (sub.pivot_table(index="client", columns="i", values="free")
           .reindex(order.client))
    # Same C1/C2/E1… scheme as C8 and C12, so a straggler spotted in one chart
    # can be found in the others. A raw client id on the axis makes that
    # cross-reference a manual diff of two uuid prefixes.
    ticks = order.groupby("role").cumcount() + 1
    dev_label = {c: f"{r[0].upper()}{n}"
                 for c, r, n in zip(order.client, order.role, ticks)}
    bucket_s = float(sub.bucket_s.iloc[0])

    from matplotlib.colors import LinearSegmentedColormap
    # Sequential, light -> dark with INCREASING value: idle is the thing being
    # hunted, so more free must read darker. Reversing it buries the finding in
    # near-white and makes a busy fleet the salient colour.
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    cmap.set_bad(SURFACE)     # a device that ended earlier has no bucket there

    fig, ax = plt.subplots(figsize=(13, max(3.0, 0.34 * len(piv) + 1.4)))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   extent=[0, piv.shape[1] * bucket_s, len(piv) - 0.5, -0.5],
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(piv)),
                  [dev_label.get(c, str(c)) for c in piv.index], fontsize=9)
    ax.set_ylabel("Device  (C = cloud, E = edge)")
    ax.set_xlabel("Seconds since that device's own start "
                  "(device clocks — offsets are exact, but not comparable between devices)")
    ax.set_title(f"{run} — when each device was doing nothing  "
                 f"(bucket = {bucket_s:g}s)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.024)
    cb.set_label("Free (% of bucket)", color=INK_2)
    cb.outline.set_visible(False)
    finish(fig, "13_free_time_heatmap.png", hide_spines=("top", "right", "left", "bottom"))
''')

chart("C14 · Broker host memory over the run",
      "The machine we run no code on. A broker at its high-water mark does not "
      "fail — it **blocks publishers**, which on a worker looks like a stall with "
      "no local cause. Memory climbing while throughput falls is the back-pressure "
      "signature, and it is unmistakable once the two curves are stacked. `used` is "
      "`MemTotal − MemAvailable`; the phase bands are the idle / run / tail marks.",
      r'''
run = first_run_with(df_ram)
if run is None:
    print("no broker RAM samples — skipping C14 (feature disabled, or the host "
          "was unreachable; broker_ram.log records the reason)")
else:
    sub = df_ram[df_ram.run == run].sort_values("ts")
    t0 = int(sub.ts.iloc[0])
    t = (sub.ts - t0) / 1e9

    # Two measures on different scales -> TWO PANELS, never a dual axis.
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    ax = axes[0]
    PHASE_TINT = {"idle": "#f1f0ec", "run": "#e8f0fb", "tail": "#f1f0ec"}
    for phase in ["idle", "run", "tail"]:
        pt = t[sub.phase.values == phase]
        if len(pt):
            ax.axvspan(pt.min(), pt.max(), color=PHASE_TINT[phase], zorder=0)
            # ABOVE the axes, not inside it: the memory curve runs along the top
            # of this panel for the whole run, so a label at y=1.0 lands on the
            # line it is describing.
            ax.annotate(phase, xy=((pt.min() + pt.max()) / 2, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=MUTED, annotation_clip=False)
    ax.plot(t, sub.used_mb, color=S1, label="host used (MemTotal − MemAvailable)", **LINE_KW)
    ax.plot(t, sub.rss_mb, color=S2, label="broker process RSS", **LINE_KW)
    idle = sub[sub.phase == "idle"]
    if len(idle):
        base = idle.used_mb.mean()
        ax.axhline(base, color=S1, linewidth=1.0, alpha=0.45)
        # Under the rule and a third of the way in: at the right-hand end the
        # used curve comes back down towards this baseline, which is exactly
        # where the tail matters and exactly where the text collided with it.
        ax.annotate(f"idle baseline {base:.0f} MB",
                    xy=(t.iloc[len(t) // 3], base),
                    xytext=(0, -6), textcoords="offset points", ha="center",
                    va="top", fontsize=9, color=S1)
    ax.set_ylabel("Memory (MB)")
    ax.set_title(f"{run} — broker host memory  "
                 f"(source = {sub.source.iloc[0]})")
    # Both curves occupy the top and the bottom-right of this panel (host used
    # runs high, RSS runs low and the tail drops through the corner). The band
    # between them is empty for the whole run, so the legend goes there.
    ax.legend(loc="center left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    b = df_batch[df_batch.run == run].sort_values("ts")
    if len(b):
        ax.plot((b.ts - t0) / 1e9, b.window_fps, color=S3,
                label="system window rate", **LINE_KW)
        ax.legend(loc="upper right")
    ax.set_ylabel("Window rate (frames/s)")
    ax.set_xlabel("Seconds since the first RAM sample (server clock)")
    ax.grid(axis="x", visible=False)

    fig.suptitle("Broker memory against throughput — memory up while rate falls "
                 "is back-pressure", fontsize=13.5, fontweight="semibold",
                 color=INK, y=1.01)
    fig.tight_layout()
    finish(fig, "14_broker_ram.png")
''')

chart("C15 · Payload size on the wire",
      "What one worker actually puts on the wire, sampled before each publish. "
      "`mean_mb × the queue depth cap` is the RAM the broker host must hold — "
      "compare it against C14's `run_peak_over_idle`. A wide `p95/p50` spread means "
      "the payload depends on the scene, which makes any single-number bandwidth "
      "estimate optimistic.",
      r'''
run = first_run_with(df_msz)
if run is None:
    print("no message-size samples — skipping C15 (feature disabled, or the "
          "chosen worker published nothing)")
else:
    s = df_msz[df_msz.run == run].sort_values("t_offset_s")
    summary = df_mszs[df_mszs.run == run]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(s.t_offset_s, s.mb, color=S1, label=f"{s.client.iloc[0]} per message",
            **LINE_KW)
    if len(summary):
        r = summary.iloc[0]
        # mean and p95 are usually within a percent of each other on a fixed
        # payload, so their labels collide if both are offset the same way. Push
        # them apart: p95 above its line, mean below its own.
        for val, lbl, colour, dy, va in ((r.p95_mb, "p95", S2, 6, "bottom"),
                                         (r.mean_mb, "mean", S1, -7, "top")):
            ax.axhline(val, color=colour, linewidth=1.0, alpha=0.55)
            ax.annotate(f"{lbl} {val:.1f} MB", xy=(s.t_offset_s.max(), val),
                        xytext=(-4, dy), textcoords="offset points", ha="right",
                        va=va, fontsize=9, color=colour)
    ax.set_xlabel("Seconds since that worker's own first publish")
    ax.set_ylabel("Message size (MB)")
    ax.set_title("Payload size over the run")
    ax.set_ylim(0, s.mb.max() * 1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")

    ax = axes[1]
    if len(summary):
        r = summary.iloc[0]
        stat = [("min", r.min_mb), ("p50", r.p50_mb), ("p95", r.p95_mb), ("max", r.max_mb)]
        bars = ax.bar(np.arange(len(stat)), [v for _, v in stat], width=0.6,
                      color=S1, **BAR_KW)
        label_bars(ax, bars, fmt="{:.1f}")
        ax.set_xticks(np.arange(len(stat)), [k for k, _ in stat])
        ax.set_ylabel("MB per message")
        ax.set_ylim(0, r.max_mb * 1.22)
        ax.set_title(f"Distribution  ·  {int(r.n)} messages  ·  "
                     f"{r.rate_mb_s:.1f} MB/s egress")
        ax.grid(axis="x", visible=False)
        ax.annotate(f"context: mode={r['mode']} compress={r['compress']} "
                    f"per-frame {r.per_frame_mb:.3f} MB",
                    xy=(0.5, -0.20), xycoords="axes fraction", ha="center",
                    fontsize=9, color=MUTED)
    else:
        ax.axis("off")
    fig.suptitle("Message size — is a busy worker computing or shipping?",
                 fontsize=14, fontweight="semibold", color=INK, y=1.02)
    fig.tight_layout()
    finish(fig, "15_message_size.png")
''')

# ---------------------------------------------------------------- manifest
md("## 4 · Manifest")
code(r'''
print(f"{len(SAVED)} image(s) written to {IMG_DIR}\n")
for name in SAVED:
    print("  ", name)

missing = [f for f in ["01_rate_by_cluster.png", "03_system_window_rate.png",
                       "05_service_latency_by_role.png", "07_utilization_by_role.png",
                       "08_device_utilization.png"] if f not in SAVED]
print("\nrequired-file coverage:",
      "complete" if not missing else f"MISSING {missing}")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT} ({len(cells)} cells)")
