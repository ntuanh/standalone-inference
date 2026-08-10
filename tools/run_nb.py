"""run_nb.py — execute the notebook in place and report EVERY failure.

`allow_errors=True` is the important flag: it collects every failing cell in one
pass instead of one round-trip per bug. Executing writes outputs (including the
base64 PNGs) back into the .ipynb, so the delivered notebook renders without a
re-run.

    python tools/build_nb.py && python tools/run_nb.py

Fix defects in tools/build_nb.py and re-run both. Never patch the .ipynb
directly — the next build silently reverts it.
"""
import sys
from pathlib import Path

try:
    import nbformat
    from nbclient import NotebookClient
except ImportError:
    sys.exit("nbformat + nbclient + ipykernel are required: "
             "pip install nbformat nbclient ipykernel")

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "results" / "visual" / "Result Visualization.ipynb"
if not NB.exists():
    sys.exit(f"{NB} does not exist — run tools/build_nb.py first")

nb = nbformat.read(str(NB), as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3",
               resources={"metadata": {"path": str(NB.parent)}},
               allow_errors=True).execute()
nbformat.write(nb, str(NB))

fail = 0
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            fail += 1
            print(f"\n### ERROR in cell {i} ###")
            print(cell.source[:300], "\n---")
            print("\n".join(out.get("traceback", []))[-2500:])
        elif out.get("output_type") == "stream" and out.get("text", "").strip():
            print(f"[cell {i}] {out['text'].rstrip()[:1500]}")

print(f"\n=== {fail} cell error(s) ===")
sys.exit(1 if fail else 0)
