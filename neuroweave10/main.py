"""NEUROWEAVE v10.0 «Genesis» — демонстрация жизненного цикла.

    python -m neuroweave10.main [--preset tiny|poc|full] [--steps N]

Бодрствование со случайным сенсорным потоком и переключением правила
(новизна → NE-burst), затем сон с SWR-реплеем и консолидацией.
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import genesis_cfg
from .organism.system import GenesisSystem


def run_demo(preset: str = "tiny", steps: int = 120, batch: int = 1,
             verbose: bool = True) -> dict:
    t0 = time.time()
    sys = GenesisSystem(preset=preset, batch=batch).boot()
    n_in = sys.cfg.input_dim
    # Марковский сенсорный поток с инверсией на середине
    xs = [torch.rand(batch, n_in) for _ in range(steps // 2)] + \
         [torch.rand(batch, n_in).flip(-1) for _ in range(steps - steps // 2)]
    sigmas, confs, energies, awake = [], [], [], []
    n_path = 0
    for t in range(steps):
        out = sys.step(xs[t], learn=True, drive=0.6)
        sigmas.append(out.sigma); confs.append(out.confidence)
        energies.append(out.energy); awake.append(out.is_awake)
        if out.pathology:
            n_path += 1
            if verbose:
                print(f"[t={out.t}] PATHOLOGY: {out.pathology}")
            break
    rep = sys.sleep(150)
    fin = sys.report()
    dt = time.time() - t0
    if verbose:
        print(f"=== NEUROWEAVE v10.0 «Genesis» [{preset}] ===")
        print(f"ticks={sys.org.t}  wall={dt:.2f}s  "
              f"sigma={fin['sigma']:.3f}  conf={fin['confidence']:.3f}")
        print(f"sleep: {rep}")
        print(f"episodes={fin['episodes']}  m_ratio={fin['m_ratio']:.3f}  "
              f"energy={fin['energy']:.3f}")
        print(f"errors={fin['errors']}  avalanches={fin['avalanches']}")
    return {"sigma": fin["sigma"], "confidence": fin["confidence"],
            "pathology": fin["pathology"], "episodes": fin["episodes"],
            "wall_s": dt, "mean_sigma": sum(sigmas) / max(1, len(sigmas)),
            "mean_conf": sum(confs) / max(1, len(confs))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny",
                    choices=["tiny", "poc", "full"])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    run_demo(args.preset, args.steps, args.batch)
