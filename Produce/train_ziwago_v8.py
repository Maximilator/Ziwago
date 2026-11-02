#!/opt/micromamba/bin/python3
"""Точка входа `Produce/train_ziwago_v8.py` — проксирует запуск основного тренера `train_ziwago_v8.py`."""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "train_ziwago_v8.py"), run_name="__main__")
