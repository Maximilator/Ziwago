"""Дендритные ветви Poirazi (Neuron 2003) + BAC-плато (Larkum 1999).

Нейрон как двухслойная сеть: K ветвей на базальном дендрите, каждая со
своей нелинейностью s(n_i) (при NMDA-спайках ~n²); сома суммирует
y = g(Σ α_i·s(n_i)). Ветвь, получившая суммарный вход выше θ_Ca,
генерирует локальный NMDA-плато (50-100 мс) — усиленный сигнал сомы.

BAC-firing: совпадение обратного спайка сомы (bAP) и дистального
апикального входа в окне ~5 мс снижает θ_Ca и вызывает Ca-спайк →
берст = сигнал ошибки/пластичности (плато усиливает трёхфакторное
правило).

Интеграция: экспоненциальный Эйлер  x ← x·e^{-dt/τ} + x_∞(1-e^{-dt/τ})
устойчив при dt≈τ (math.md §2) — используется для проводимостей.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BranchTree:
    """Базальное дендритное дерево: K ветвей × N нейронов."""
    weights: torch.Tensor    # [N,K] веса ветвей α_i
    branch_v: torch.Tensor   # [B,N,K] потенциал ветви
    plateau: torch.Tensor    # [B,N,K] оставшиеся мс плато
    bap_time: torch.Tensor   # [B,N] мс с последнего сома-спайка


class DendriticTree:
    """Базальный вход через ветви с локальными плато (полезная нагрузка:
    суперлинейная интеграция + пороговые NMDA-спайки на ветви)."""

    def __init__(self, n_neurons: int, n_in: int, n_branch: int = 4,
                 theta_ca: float = -25.0, plateau_ms: float = 60.0,
                 bac_window: float = 5.0, theta_drop: float = 10.0,
                 seed: int = 0):
        self.n, self.k, self.d = n_neurons, n_branch, n_in
        g = torch.Generator().manual_seed(seed)
        # mask[K,n_in]: какой синапс на какую ветвь (случайная привязка)
        self.mask = torch.randint(0, n_branch, (n_in,), generator=g)
        self.w_syn = torch.randn(n_branch, n_in, generator=g) * 1.2
        self.alpha = torch.rand(n_neurons, n_branch, generator=g) + 0.5
        self.theta_ca = theta_ca
        self.plateau_ms = plateau_ms
        self.bac_window = bac_window
        self.theta_drop = theta_drop

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        z = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        self.branch_v = z(self.n, self.k).fill_(-70.0)
        self.plateau = z(self.n, self.k)
        self.bap_time = z(self.n).fill_(1e9)
        self.bursts = z(self.n)

    def step(self, x_in: torch.Tensor, soma_spikes: torch.Tensor,
             dt: float = 1.0) -> torch.Tensor:
        """x_in [B,D] → проводимость к соме [B,N] + берсты."""
        # Вход на ветвь: s(n_i) = σ-нелинейность суммы веточных синапсов
        n_branch_in = x_in @ self.w_syn.t()               # [B,K]
        v = n_branch_in.unsqueeze(1).expand(-1, self.n, -1)  # [B,N,K]
        # Экспоненциальный Эйлер к -70 мВ с драйвом v
        tau = 10.0
        lam = 1.0 - torch.exp(torch.tensor(-dt / tau))
        self.branch_v = self.branch_v + (v * 60.0 - 70.0
                                         - self.branch_v) * lam
        # BAC: bAP (недавний сома-спайк) снижает θ_Ca ветви
        th = torch.where(self.bap_time.unsqueeze(-1) < self.bac_window,
                         self.theta_ca - self.theta_drop,
                         torch.full_like(self.branch_v, self.theta_ca))
        # Плато: ветвь > θ_Ca → NMDA-спайк длиной plateau_ms
        ignite = self.branch_v > th
        self.plateau = torch.where(
            ignite, torch.full_like(self.plateau, self.plateau_ms),
            (self.plateau - dt).clamp(min=0.0))
        on = (self.plateau > 0).float()
        out = (self.alpha * torch.sigmoid(v - th / 50.0) * (1.0 + 2.0 * on)
               ).sum(-1)                                  # [B,N]
        self.bursts = on.any(-1).float()                  # берст-флаг
        self._last_branch_act = torch.sigmoid(v - th / 50.0) * (1.0 + 2.0 * on)  # [B,N,K]
        self._last_on = on
        self.bap_time = torch.where(soma_spikes > 0,
                                    torch.zeros_like(self.bap_time),
                                    self.bap_time + dt)
        return out.clamp(0.0, 8.0)

    def learn(self, error: torch.Tensor, x_in: torch.Tensor,
              f_mod: float = 1.0, eta: float = 1.5e-3) -> float:
        """Локальная пластичность дендритных ветвей (Poirazi 2003; Chavlis & Poirazi 2025):
        1) Веса ветвь→сома α_{i,k} обучаются по сомато-дендритной ошибке нейрона i
           и нелинейному отклику ветви s(n_{i,k});
        2) Внутриветочные синапсы w_syn[k, j] обучаются при генерации Ca²⁺-плато на ветви k."""
        if f_mod <= 0.0 or not hasattr(self, "_last_branch_act"):
            return 0.0
        err = error[0] if error.dim() == 2 and error.shape[0] == 1 else error.mean(0)   # [N]
        x = x_in[0] if x_in.dim() == 2 and x_in.shape[0] == 1 else x_in.mean(0)          # [D]
        b_act = self._last_branch_act.mean(0)                                            # [N,K]
        lr = eta * max(0.1, min(2.5, float(f_mod)))
        # 1. Обновление α_{i,k} [N, K]: какие ветви информативны для нейрона i
        d_alpha = lr * (err.unsqueeze(1) * b_act)
        self.alpha = (self.alpha + d_alpha).clamp(0.1, 2.5)
        # 2. Внутриветочная кластеризация w_syn [K, D]: ветви с активным плато/ошибкой
        #    усиливают ко-активные входы x_j (привязанные к ветви по self.mask)
        branch_err = (err.unsqueeze(1) * (0.5 + 0.5 * self._last_on.mean(0))).mean(0)    # [K]
        branch_mask = (torch.arange(self.k, device=x.device).unsqueeze(1)
                       == self.mask.to(x.device).unsqueeze(0)).float()                   # [K,D]
        dw_syn = lr * 1.5 * torch.outer(branch_err, x) * (0.35 + 0.65 * branch_mask)
        self.w_syn = (self.w_syn + dw_syn).clamp(-3.0, 3.0)
        return float(d_alpha.abs().mean().item() + dw_syn.abs().mean().item())
