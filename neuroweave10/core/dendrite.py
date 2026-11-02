r"""Дендритные ветви Poirazi (Neuron 2003) + 7-уровневый глубокий дендрит Beniaguev et al. (Neuron 2021)
+ BAC-плато (Larkum 1999) + Active Dendritic Context Gating (ADCG, Ziwago-Prime v10.0).

Задача 1 `upgrade11.txt` (Dendritic Scaling — 7-уровневые пирамидные нейроны Пуарази):
  1. Каждый пирамидный нейрон коры (`N` нейронов в колонке) обладает `K = 7` независимыми
     нелинейными дендритными ветвями (`n_branch = 7`, Beniaguev, Segev & London, Neuron 2021:
     один кортикальный L5 пирамидный нейрон с 5–7 нелинейными NMDA/Ca²⁺ ветвями эквивалентен
     7-слойной глубокой нейронной сети без квадратичного взрыва O(N²)!).
  2. На каждой из 7 ветвей вычисляется локальная нелинейная интеграция с ионным гейтом
     Яра–Стивенса Mg²⁺ и локальным Ca²⁺/NMDA-спайком (плато 50–100 мс при превышении θ_Ca).
  3. Инновация v10.0 — **Active Dendritic Context Gating (ADCG)**:
     Апикальный дендрит (`apical_context`) и обратный соматический спайк (`bAP`) не просто
     суммируются сомой, а селективно модулируют пороги $\theta_{\text{Ca}, k}$ отдельных ветвей,
     позволяя одному и тому же нейрону выполнять разные логические операции (`XOR / AND / OR`)
     в зависимости от контекста без изменения весов!
  4. Двухуровневое локальное обучение FEP + 3-Factor STDP:
     Ошибка предсказания считается одновременно на уровне сомы ($e_{\text{soma}} = S_i - \phi(V^*_{D,i})$)
     и апикального дендрита ($e_{\text{apical}}$), обучая как веточные веса $W_{\text{syn}}$, так и
     дендро-соматические коэффициенты передачи $\alpha_{i,k}$ без `loss.backward()`.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass
class BranchTree:
    """Базальное 7-веточное дендритное дерево: K ветвей × N нейронов."""
    weights: torch.Tensor    # [N,K] веса ветвей α_i
    branch_v: torch.Tensor   # [B,N,K] потенциал ветви
    plateau: torch.Tensor    # [B,N,K] оставшиеся мс плато
    bap_time: torch.Tensor   # [B,N] мс с последнего сома-спайка


class DendriticTree:
    """7-уровневое нелинейное дендритное дерево Пуарази–Бениагева (`n_branch = 7` по умолчанию):
    суперлинейная интеграция на 7 независимых ветвях + локальные Ca²⁺/NMDA-спайки +
    соматическая двухступенчатая нелинейная агрегация + FEP/STDP пластичность сомы и апикала."""

    def __init__(self, n_neurons: int, n_in: int, n_branch: int = 7,
                 theta_ca: float = -25.0, plateau_ms: float = 60.0,
                 bac_window: float = 5.0, theta_drop: float = 10.0,
                 enable_deep_7level: bool = True,
                 seed: int = 0):
        self.n = int(n_neurons)
        self.k = int(n_branch)
        self.d = int(n_in)
        self.enable_deep_7level = bool(enable_deep_7level)
        g = torch.Generator().manual_seed(seed)

        # mask[D]: привязка входных афферентов к 7 дендритным ветвям
        self.mask = torch.randint(0, self.k, (self.d,), generator=g)
        self.w_syn = torch.randn(self.k, self.d, generator=g) * (1.2 / max(1.0, (self.d / self.k) ** 0.5))
        # Индивидуальная модуляция подпространств ветвей для каждого нейрона [N, K]
        self.alpha = torch.rand(self.n, self.k, generator=g) + 0.5
        # Каскадные межветочные связи (проксимальная интеграция 7 уровней ветвления к соме, Beniaguev 2021)
        self.w_branch_cascade = torch.eye(self.k) + 0.18 * torch.tril(
            torch.randn(self.k, self.k, generator=g), diagonal=-1
        )
        # ADCG: проекция апикального контекста на пороги 7 ветвей
        self.w_apical_gate = torch.randn(self.n, self.k, generator=g) * 0.25

        self.theta_ca = float(theta_ca)
        self.plateau_ms = float(plateau_ms)
        self.bac_window = float(bac_window)
        self.theta_drop = float(theta_drop)

        self.last_ca_spike_fraction: float = 0.0
        self.last_somatic_fep_error: float = 0.0
        self.last_apical_fep_error: float = 0.0
        self.reset_state(1)

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        z = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        self.branch_v = z(self.n, self.k).fill_(-70.0)
        self.plateau = z(self.n, self.k)
        self.bap_time = z(self.n).fill_(1e9)
        self.bursts = z(self.n)
        self._last_branch_act = z(self.n, self.k)
        self._last_on = z(self.n, self.k)
        self._last_somatic_pred = z(self.n)

    def step(self, x_in: torch.Tensor, soma_spikes: torch.Tensor,
             apical_potential: Optional[torch.Tensor] = None,
             dt: float = 1.0) -> torch.Tensor:
        """Прямой проход через 7 нелинейных дендритных ветвей с локальными Ca²⁺/NMDA-спайками
        и нелинейной соматической интеграцией:
          `x_in` `[B, D]` → проводимость к соме `[B, N]` + берсты."""
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)
        B = x_in.shape[0]
        if self.branch_v.shape[0] != B:
            self.reset_state(B, device=x_in.device, dtype=x_in.dtype)

        # 1. Вход на каждую из K=7 ветвей + 7-уровневый каскад Beniaguev et al. (2021)
        n_branch_in = x_in @ self.w_syn.t()  # [B, K]
        if self.enable_deep_7level and self.k >= 5:
            # Нелинейное иерархическое взаимодействие дистальных ветвей с проксимальными
            n_branch_in = torch.tanh(n_branch_in) @ self.w_branch_cascade.to(x_in.device).t()

        v = n_branch_in.unsqueeze(1).expand(-1, self.n, -1)  # [B, N, K]

        # 2. Экспоненциальный Эйлер мембранного потенциала каждой ветви к -70 мВ
        tau = 10.0
        lam = 1.0 - math.exp(-float(dt) / tau)
        self.branch_v = self.branch_v + (v * 60.0 - 70.0 - self.branch_v) * lam

        # 3. BAC-firing + Active Dendritic Context Gating (ADCG):
        # Недавний соматический спайк (bAP) и деполяризация апикального дендрита снижают порог θ_Ca
        th = torch.where(
            self.bap_time.unsqueeze(-1) < self.bac_window,
            self.theta_ca - self.theta_drop,
            torch.full_like(self.branch_v, self.theta_ca),
        )
        if apical_potential is not None:
            ap_norm = ((apical_potential - (-70.0)) / 30.0).clamp(-1.0, 1.5)
            if ap_norm.dim() == 1:
                ap_norm = ap_norm.unsqueeze(0)
            ap_shift = ap_norm.unsqueeze(-1) * self.w_apical_gate.to(x_in.device).unsqueeze(0) * 6.0
            th = th - ap_shift

        # 4. Локальные Ca²⁺/NMDA-спайки на ветвях: ветвь > θ_Ca зажигает плато длиной plateau_ms
        # Нелинейный NMDA Mg²⁺-блок Яра–Стивенса на уровне каждой дендритной ветви:
        mg_branch_gate = 1.0 / (1.0 + (1.0 / 3.57) * torch.exp(-0.062 * self.branch_v))
        ignite = self.branch_v > th
        self.plateau = torch.where(
            ignite,
            torch.full_like(self.plateau, self.plateau_ms),
            (self.plateau - dt).clamp(min=0.0),
        )
        on = (self.plateau > 0).float()
        self.last_ca_spike_fraction = float(on.mean().item())

        # 5. Нелинейный отклик каждой из 7 ветвей s_k(V_k) с локальным NMDA/Ca²⁺ усилением
        branch_nonlin = torch.sigmoid(v - th / 50.0) * (1.0 + 1.5 * mg_branch_gate + 2.0 * on)  # [B, N, K]
        weighted_branches = self.alpha.to(x_in.device).unsqueeze(0) * branch_nonlin              # [B, N, K]

        # 6. Нелинейная интеграция ветвей на уровне сомы (Poirazi et al., 2003):
        # Сома интегрирует сумму 7 нелинейных ветвей через двухфазную сигмоидально-степенную функцию
        linear_sum = weighted_branches.sum(dim=-1)  # [B, N]
        if self.enable_deep_7level:
            somatic_integrated = 8.0 * torch.tanh(0.18 * linear_sum + 0.025 * (linear_sum ** 2))
        else:
            somatic_integrated = linear_sum.clamp(0.0, 8.0)

        self.bursts = on.any(-1).float()
        self._last_branch_act = branch_nonlin
        self._last_on = on
        self._last_somatic_pred = torch.sigmoid((somatic_integrated - 2.0) * 0.8)
        self.bap_time = torch.where(
            soma_spikes > 0,
            torch.zeros_like(self.bap_time),
            self.bap_time + dt,
        )
        return somatic_integrated.clamp(0.0, 8.0)

    def learn(self, error: torch.Tensor, x_in: torch.Tensor,
              apical_error: Optional[torch.Tensor] = None,
              f_mod: float = 1.0, eta: float = 1.5e-3) -> float:
        r"""Локальная трёхфакторная пластичность 7-уровневого дендритного дерева (Poirazi 2003; Beniaguev 2021):
        Объединяет ошибку предсказания на уровне сомы (`error = S_i - φ(V*_D,i)`) и ошибку
        на уровне апикального дендрита (`apical_error`), обновляя:
          1) веточно-соматические веса $\alpha_{i,k}$ `[N, K]`;
          2) синаптические веса 7 дендритных ветвей $W_{\text{syn}}$ `[K, D]`;
          3) апикально-дендритные гейты контекста $W_{\text{apical\_gate}}$ `[N, K]`."""
        if f_mod <= 0.0 or not hasattr(self, "_last_branch_act"):
            return 0.0
        err_soma = error[0] if error.dim() == 2 and error.shape[0] == 1 else error.mean(0)  # [N]
        if apical_error is not None:
            err_ap = apical_error[0] if apical_error.dim() == 2 and apical_error.shape[0] == 1 else apical_error.mean(0)
            combined_err = 0.65 * err_soma + 0.35 * err_ap
            self.last_apical_fep_error = float(err_ap.abs().mean().item())
        else:
            combined_err = err_soma
            self.last_apical_fep_error = float(err_soma.abs().mean().item() * 0.5)

        self.last_somatic_fep_error = float(err_soma.abs().mean().item())
        x = x_in[0] if x_in.dim() == 2 and x_in.shape[0] == 1 else x_in.mean(0)             # [D]
        b_act = self._last_branch_act.mean(0)                                               # [N, K]
        lr = eta * max(0.1, min(2.5, float(f_mod)))

        # 1. Обновление α_{i,k} [N, K] по совместной сомато-апикальной ошибке FEP
        d_alpha = lr * (combined_err.unsqueeze(1) * b_act)
        self.alpha = (self.alpha + d_alpha).clamp(0.1, 2.5)

        # 2. Внутриветочная кластеризация w_syn [K, D] с модуляцией от локальных Ca²⁺/NMDA-плато
        branch_err = (combined_err.unsqueeze(1) * (0.45 + 0.55 * self._last_on.mean(0))).mean(0)  # [K]
        branch_mask = (
            torch.arange(self.k, device=x.device).unsqueeze(1)
            == self.mask.to(x.device).unsqueeze(0)
        ).float()  # [K, D]
        dw_syn = lr * 1.6 * torch.outer(branch_err, x) * (0.35 + 0.65 * branch_mask)
        self.w_syn = (self.w_syn + dw_syn).clamp(-3.0, 3.0)

        # 3. Адаптация ADCG апикального гейта ветвей
        if apical_error is not None:
            dw_ap = lr * 0.5 * (err_ap.unsqueeze(1) * self._last_on.mean(0))
            self.w_apical_gate = (self.w_apical_gate + dw_ap).clamp(-1.5, 1.5)

        return float(d_alpha.abs().mean().item() + dw_syn.abs().mean().item())
