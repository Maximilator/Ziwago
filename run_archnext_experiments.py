#!/opt/micromamba/bin/python3
"""Точка входа экспериментов масштабирования D1–D4 (Produce/archnext.txt)."""
from __future__ import annotations

import os
import runpy
import sys

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Produce", "run_archnext_experiments.py")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path(target, run_name="__main__")
