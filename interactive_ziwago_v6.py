#!/opt/micromamba/bin/python3
"""Точка входа Интерактивной Жизни и Стресс-Сна Ziwago-Prime v6."""
from __future__ import annotations

import os
import runpy
import sys

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Produce", "interactive_ziwago_v6.py")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path(target, run_name="__main__")
