"""Follow-up patch: dedup duplicate filed-date index in
compute_historical_fundamentals. Fixes ValueError when two different
fp_end quarters were filed on the same date (catch-up filings,
amendments). Inserts one line after `qdf.set_index("__filed__")...`.

Idempotent — sentinel 'index.duplicated' in cell source skips the patch.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "Main.ipynb"
FUNC_CELL_ID = "7b19d317-a87b-4c9c-86bb-f9a5c5524c53"
SENTINEL = "index.duplicated"
INSERT_AFTER_TOKEN = 'qdf = qdf.set_index("__filed__").sort_index()'
NEW_LINE = '                qdf = qdf[~qdf.index.duplicated(keep="last")]\n'


def next_backup_path(nb_path: Path) -> Path:
    n = 1
    while True:
        cand = nb_path.with_suffix(nb_path.suffix + f".bak{n}")
        if not cand.exists():
            return cand
        n += 1


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = next((c for c in nb["cells"] if c.get("id") == FUNC_CELL_ID), None)
    if cell is None:
        print(f"FAIL: cell {FUNC_CELL_ID} not found", file=sys.stderr)
        sys.exit(1)

    source = cell["source"]
    joined = "".join(source)
    if SENTINEL in joined:
        print(f"Already patched (sentinel '{SENTINEL}' present); no-op.")
        return

    # Find the set_index("__filed__") line; insert dedup right after.
    target_idx = None
    for i, ln in enumerate(source):
        if INSERT_AFTER_TOKEN in ln:
            target_idx = i
            break
    if target_idx is None:
        print(f"FAIL: could not locate insertion point '{INSERT_AFTER_TOKEN}'", file=sys.stderr)
        sys.exit(2)

    backup = next_backup_path(NB_PATH)
    backup.write_text(NB_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    new_source = source[: target_idx + 1] + [NEW_LINE] + source[target_idx + 1 :]
    cell["source"] = new_source
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted dedup line after row {target_idx} in cell {FUNC_CELL_ID}. Backup: {backup.name}")


if __name__ == "__main__":
    main()
