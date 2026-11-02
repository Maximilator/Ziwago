"""Астроцитарный микродомен (REGENESIS, Закон 4).

Трёхсторонний синапс: астроцит интегрирует спайковую и NE-активность
микродомена в медленную Ca²⁺-динамику (τ≈2.5 с) с латеральной щелевой
диффузией. Глиотрансмиттер мультипликативно модулирует stp_gain реальных
синапсов L4→L2/3 (пресинаптическое высвобождение) и сдвигает кальциевый
порог плато дендрита — прямое действие на физику связей, а не канал
декодера.
"""
from __future__ import annotations

import torch


class AstrocyteMicrodomain:
    """Медленная Ca²⁺-интеграция зоны → stp_gain и θ_Ca модуляция."""

    def __init__(self, n_neurons: int, tau_ca: float = 2500.0,
                 gamma_gli: float = 0.65, theta_ca: float = 0.20,
                 diffusion: float = 0.05, ne_boost: float = 1.5):
        self.n = n_neurons
        self.decay = float(torch.exp(torch.tensor(-1.0 / tau_ca)))
        self.gamma_gli = gamma_gli
        self.theta_ca = theta_ca
        self.diffusion = diffusion
        self.ne_boost = ne_boost
        self.ca = None

    def reset_state(self, batch: int, device=None, dtype=torch.float32):
        self.ca = torch.zeros(batch, self.n, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def step(self, spikes: torch.Tensor, ne: float = 0.0) -> torch.Tensor:
        """spikes [B,N] → gliotransmitter gate [N] ∈ [1, 1.5]."""
        drive = (0.35 + self.ne_boost * ne) * spikes
        if self.n > 2:
            lap = 0.5 * (torch.roll(self.ca, 1, -1)
                         + torch.roll(self.ca, -1, -1)) - self.ca
        else:
            lap = 0.0
        self.ca = (self.decay * (self.ca + self.diffusion * lap)
                   + (1.0 - self.decay) * drive * 50.0
                   ).clamp(0.0, 2.0) * (1.0 - 0.003)
        glio = torch.tanh(self.ca / max(1e-6, self.theta_ca))
        return (1.0 + self.gamma_gli * glio).clamp(1.0, 1.5)   # [B,N]

    def theta_ca_shift(self) -> float:
        """Сдвиг θ_Ca плато дендрита: активный астроцит облегчает
        Ca-плато → расширяет LTP-окно (D-серин из глии)."""
        if self.ca is None:
            return 0.0
        return -4.0 * float(self.ca.mean())   # до −4 мВ при высоком Ca
