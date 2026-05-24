"""Patch run_model in cell a9a9904cca1e7d46 to use the model's
trained feature order instead of real_df's column order. Fixes the
XGBoost feature_names mismatch when create_stock_features produces
columns in different orders for training vs prediction.

Idempotent — sentinel 'get_booster().feature_names' in cell source
skips the patch.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "Main.ipynb"
CELL_ID = "a9a9904cca1e7d46"
SENTINEL = "get_booster().feature_names"

OLD_BLOCK = (
    '    features = real_df.drop(["Stock", "Returns-future-1wk", "Returns-future-2wk"], axis=1).columns.values\n'
    '    model = overall_model\n'
)
NEW_BLOCK = (
    '    model = overall_model\n'
    '    features = list(model.get_booster().feature_names)  # enforce training-time column order\n'
)


def next_backup_path(nb_path: Path) -> Path:
    n = 1
    while True:
        cand = nb_path.with_suffix(nb_path.suffix + f".bak{n}")
        if not cand.exists():
            return cand
        n += 1


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = next((c for c in nb["cells"] if c.get("id") == CELL_ID), None)
    if cell is None:
        print(f"FAIL: cell {CELL_ID} not found", file=sys.stderr)
        sys.exit(1)

    source = cell["source"]
    joined = "".join(source)
    if SENTINEL in joined:
        print(f"Already patched (sentinel '{SENTINEL}' present); no-op.")
        return

    # Locate the two consecutive lines and replace.
    target_idx = None
    for i in range(len(source) - 1):
        if (source[i] == OLD_BLOCK.splitlines(keepends=True)[0]
                and source[i + 1] == OLD_BLOCK.splitlines(keepends=True)[1]):
            target_idx = i
            break
    if target_idx is None:
        print("FAIL: could not match OLD_BLOCK lines verbatim", file=sys.stderr)
        # Dump a few lines for diagnosis
        for i, ln in enumerate(source[:25]):
            print(f"  {i:3d}: {ln!r}", file=sys.stderr)
        sys.exit(2)

    backup = next_backup_path(NB_PATH)
    backup.write_text(NB_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    new_lines = NEW_BLOCK.splitlines(keepends=True)
    cell["source"] = source[:target_idx] + new_lines + source[target_idx + 2:]
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched run_model in cell {CELL_ID} (lines {target_idx}:{target_idx+2}). Backup: {backup.name}")


if __name__ == "__main__":
    main()
