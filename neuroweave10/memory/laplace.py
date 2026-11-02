"""Спектральный банк временных констант (REGENESIS, Закон 4).

Клетки времени Shankar & Howard (2012): геометрический спектр
экспоненциальных интеграторов τ_k = τ_min·c^k — дискретное
преобразование Лапласа входного потока. В v10 медленные шкалы
модулируют адаптацию порога нейронов L5: возбудимость клетки зависит
от длинного контекста, а не от отдельного вектора в декодере.
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class LaplaceBank:
    """O(K·D) спектральная память; медленный профиль → Θ-адаптация L5."""

    def __init__(self, feature_dim: int, num_scales: int = 8,
                 tau_min: float = 4.0, tau_max: float = 512.0):
        self.feature_dim = int(feature_dim)
        self.num_scales = max(2, int(num_scales))
        taus = torch.exp(torch.linspace(
            math.log(tau_min), math.log(tau_max), self.num_scales))
        self.taus = taus
        self.alphas = torch.exp(-1.0 / taus).view(-1, 1, 1)  # [K,1,1]
        self.state: Optional[torch.Tensor] = None            # [K,B,F]

    def reset_state(self, batch: int, device=None, dtype=torch.float32):
        self.state = torch.zeros(self.num_scales, batch,
                                 self.feature_dim, device=device,
                                 dtype=dtype)

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,F] → обновлённое состояние [K,B,F]."""
        self.state = self.alphas * self.state + x.detach().unsqueeze(0)
        return self.state

    def slow_profile(self) -> torch.Tensor:
        """Медленная половина спектра → [B,F]: длинный контекст."""
        k0 = self.num_scales // 2
        return self.state[k0:].mean(0)

    def time_cells(self) -> torch.Tensor:
        """Разностный фильтр Поста-Виддера — когда появился вход."""
        l = torch.roll(self.state, 1, 0)
        r = torch.roll(self.state, -1, 0)
        return torch.relu(self.state - 0.5 * (l + r))
