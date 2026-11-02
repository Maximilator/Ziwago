#!/opt/conda/bin/python3
"""Точка входа Глобальной Непрерывной Жизни Ziwago-Prime v6 (train_ziwago_global.py)."""
from __future__ import annotations

import os
import runpy
import sys

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Produce", "train_ziwago_global.py")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path(target, run_name="__main__")
