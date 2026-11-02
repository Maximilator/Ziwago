"""neuroweave9/cortex/serotonin.py — Серотонинергическая система 5-HT (Genesis modulators.py).

Биологический контур:
1. Восходящий тон дорсального ядра шва (5-HT level in [0.1, 1.5]).
2. Вычисление фактора терпения (patience factor):
   p = sigma(gain * ([5-HT] - theta_p)) in (0, 1) — 0.5 в покое.
3. Модуляция моторного порога Базальных Ганглиев (DDM threshold):
   Высокий 5-HT -> высокий порог (терпение, контроль импульсивности).
   Сеть не спамит дешёвыми пунктуационными токенами, а даёт коре время
   накопить сложный семантический контекст.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SerotoninSystem(nn.Module):
    """Ascending 5-HT / dorsal raphe tone and patience control (Genesis modulators.py)."""

    def __init__(
        self,
        baseline: float = 0.60,
        patience_gain: float = 4.0,
        patience_threshold: float = 0.50,
        tau_content_ms: float = 300.0,
        content_k: float = 0.05,
        crh_k: float = 0.50,
        content_amplitude: float = 0.50,
        stress_depletion: float = 0.30,
        tau_ms: float = 800.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.baseline = float(baseline)
        self.patience_gain = float(patience_gain)
        self.patience_threshold = float(patience_threshold)
        self.tau_content_ms = float(tau_content_ms)
        self.content_k = float(content_k)
        self.crh_k = float(crh_k)
        self.content_amplitude = float(content_amplitude)
        self.stress_depletion = float(stress_depletion)
        self.tau_ms = float(tau_ms)

        self.register_buffer("level", torch.tensor(baseline, device=device), persistent=False)
        self.register_buffer("content", torch.tensor(0.0, device=device), persistent=False)
        self._p_rest = float(
            torch.sigmoid(torch.tensor(patience_gain * (baseline - patience_threshold))).item()
        )

    def reset_state(self, device: torch.device = torch.device("cpu")) -> None:
        self.level = torch.tensor(self.baseline, device=device)
        self.content = torch.tensor(0.0, device=device)

    @torch.no_grad()
    def step(
        self,
        dt: float = 1.0,
        reward: float = 0.0,
        crh: float = 0.0,
    ) -> torch.Tensor:
        """Обновление уровня серотонина (5-HT) и накопленного удовлетворения (contentment)."""
        flow = max(float(reward), 0.0) / max(dt, 1e-6)
        alpha_c = math.exp(-dt / max(self.tau_content_ms, 1e-3))
        self.content = self.content * alpha_c + (1.0 - alpha_c) * flow

        sated = self.content / (self.content + self.content_k)
        c = max(float(crh), 0.0)
        strained = c / (c + self.crh_k)

        target = self.baseline + self.content_amplitude * sated - self.stress_depletion * strained
        alpha_t = math.exp(-dt / max(self.tau_ms, 1e-3))
        self.level = torch.clamp(self.level * alpha_t + (1.0 - alpha_t) * target, min=0.10, max=1.50)
        return self.level

    @torch.no_grad()
    def get_patience_factor(self) -> torch.Tensor:
        """Фактор терпения p in (0, 1): высокий 5-HT повышает порог DDM."""
        return torch.sigmoid(self.patience_gain * (self.level - self.patience_threshold))
