#!/usr/bin/env python3
"""
Strip notebook outputs and scrub obvious stale error traces.

Usage:
  python scripts/clean_notebooks.py baselines/notebooks/*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


BAD_MD_SNIPPETS = (
    "Traceback (most recent call last)",
    "KeyError:",
    "evaluating test set!",
)


def _clean_notebook(nb: dict) -> dict:
    cells = nb.get("cells", [])
    new_cells = []
    for cell in cells:
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            new_cells.append(cell)
            continue

        if cell_type == "markdown":
            src = "".join(cell.get("source", []))
            if any(bad in src for bad in BAD_MD_SNIPPETS):
                cell["source"] = [
                    "_(Removed stale Colab error output; see upstream baseline repo logs if needed.)_"
                ]
            new_cells.append(cell)
            continue

        new_cells.append(cell)

    nb["cells"] = new_cells
    return nb


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print("Expected one or more .ipynb paths.", file=sys.stderr)
        return 2

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = _clean_notebook(data)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Cleaned: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

