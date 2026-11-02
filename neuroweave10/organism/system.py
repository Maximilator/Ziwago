"""GenesisSystem — фасад поверх GenesisOrganism (как CortexSystem в v9).

Отвечает за жизненный цикл: boot → run/sleep → report → save/load.
Нулевая дублирующая логика: всё делегируется организму.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ..config import GenesisConfig, genesis_cfg
from .genesis import GenesisOrganism, GenesisOutput


class GenesisSystem:
    """Фасад: загрузка, такт, сон, телеметрия, чекпоинты."""

    def __init__(self, preset: str = "poc", batch: int = 1,
                 cfg: Optional[GenesisConfig] = None):
        self.cfg = cfg or genesis_cfg(preset)
        self.org = GenesisOrganism(self.cfg, batch=batch,
                                   seed=self.cfg.seed)
        self.org.reset_state(batch)
        self.history: List[GenesisOutput] = []

    def boot(self) -> "GenesisSystem":
        return self

    def step(self, x: torch.Tensor, reward: float = 0.0,
             learn: bool = True, drive: float = 0.6) -> GenesisOutput:
        out = self.org.step(x, reward=reward, learn=learn, drive=drive)
        self.history.append(out)
        return out

    def run(self, steps: int, input_fn=None, learn: bool = True) -> List[GenesisOutput]:
        for t in range(steps):
            x = input_fn(t) if input_fn else torch.zeros(
                self.org.batch, self.cfg.input_dim)
            self.step(x, learn=learn)
            if self.org.pathology:
                break
        return self.history

    def sleep(self, ticks: int = 200) -> Dict:
        return self.org.sleep_cycle(ticks)

    def report(self) -> Dict:
        last = self.history[-1] if self.history else None
        return {
            "version": "10.0-genesis", "t": self.org.t,
            "sigma": self.org.sigma,
            "pathology": self.org.pathology,
            "is_awake": self.chem.is_awake if hasattr(self, "chem") else self.org.chem.is_awake,
            "sleep_pressure": self.org.chem.s.ado,
            "confidence": last.confidence if last else 0.0,
            "energy": last.energy if last else 1.0,
            "errors": last.prediction_errors if last else {},
            "avalanches": self.org.homeo.monitor.stats(),
            "episodes": len(self.org.hipp.episodes),
            "m_ratio": self.org._last_meta.m_ratio if self.org._last_meta else 0.0,
            "cortisol": self.org.chem.s.cort,
            "ecb": self.org.chem.s.ecb,
        }

    def save(self, path: str) -> None:
        state = {"cfg": self.cfg,
                 "encoder_phi": self.org.hier.encoder.phi,
                 "hippo": {"w_ca3": self.org.hipp.w_ca3,
                           "w_ec_dg": self.org.hipp.w_ec_dg},
                 "bg": {"w_go": self.org.bg.w_go, "w_nogo": self.org.bg.w_nogo},
                 "chem": self.org.chem.s}
        torch.save(state, path)

    @staticmethod
    def load(path: str) -> "GenesisSystem":
        st = torch.load(path, weights_only=False)
        sys = GenesisSystem(cfg=st["cfg"])
        sys.org.hier.encoder.phi = st["encoder_phi"]
        sys.org.hipp.w_ca3 = st["hippo"]["w_ca3"]
        sys.org.chem.s = st["chem"]
        return sys
