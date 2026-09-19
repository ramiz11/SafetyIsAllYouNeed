#!/usr/bin/env python3
"""Run the six coordinate-bearing configurations using your own training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coordinate_pairs_v2.cli import main


if __name__ == "__main__":
    main()
