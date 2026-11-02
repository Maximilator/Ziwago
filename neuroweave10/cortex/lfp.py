"""Ламинарный локальный потенциал поля (LFP) и эфаптическая связь (Ziwago-Prime v3, Уровень 6).

Биофизика:
  1. LFP — это не частота спайков, а низкочастотно фильтрованный (τ_LFP ≈ 10 мс)
     внеклеточный дипольный потенциал, создаваемый синаптическими токами пирамидальных
     дендритов и перисоматического торможения (Genesis level2/lfp.py):
       I_dipole(i) = I_FF(i) + I_FB(i) - 0.5 · I_inh(i)
  2. Эфаптическая обратная связь (Anastassiou et al., Nature Neuroscience 2011):
     Градиент внеклеточного поля между поверхностными (L2/3, L4) и глубокими (L5, L6)
     слоями колонки создаёт локальный трансмембранный сдвиг потенциала сомы (±1.5 мВ),
     синхронизируя фазу спайков соседних нейронов без синаптической задержки!
"""
from __future__ import annotations

import math
from typing import Dict

import torch


class EphapticLFP:
    """Ламинарный LFP-наблюдатель и генератор эфаптического поля колонки."""

    def __init__(self, n_neurons: int, layer_slices: Dict[str, slice],
                 tau_ms: float = 10.0, ephaptic_gain: float = 0.08):
        self.n = n_neurons
        self.slices = dict(layer_slices)
        self.tau_ms = max(1.0, float(tau_ms))
        self.ephaptic_gain = float(ephaptic_gain)
        self.layer_lfp: Dict[str, float] = {k: 0.0 for k in self.slices}
        self.history: Dict[str, list[float]] = {k: [] for k in self.slices}
        self.ephaptic_mv = torch.zeros(1, n_neurons)

    def reset_state(self, batch: int = 1) -> None:
        self.layer_lfp = {k: 0.0 for k in self.slices}
        for k in self.history:
            self.history[k].clear()
        self.ephaptic_mv = torch.zeros(batch, self.n)

    def step(self, i_ff: torch.Tensor, i_fb: torch.Tensor,
             i_inh: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Обновляет LFP каждого слоя по синаптическим токам и возвращает
        вектор эфаптического сдвига мембраны [B, N] (в мВ, ограничен ±1.5 мВ)."""
        b_sz = i_ff.shape[0]
        if self.ephaptic_mv.shape[0] != b_sz:
            self.ephaptic_mv = torch.zeros(b_sz, self.n, device=i_ff.device)
        dipole = (i_ff + i_fb - 0.5 * i_inh).mean(0)  # [N]
        alpha = math.exp(-dt / self.tau_ms)
        for name, sl in self.slices.items():
            if sl.stop > sl.start:
                raw = float(dipole[sl].mean().item())
            else:
                raw = 0.0
            val = raw + (self.layer_lfp[name] - raw) * alpha
            self.layer_lfp[name] = val
            hist = self.history[name]
            hist.append(val)
            if len(hist) > 256:
                del hist[:len(hist) - 256]

        # Внеклеточный диполь между поверхностными (L23) и глубокими (L5/L6) слоями
        l_sup = self.layer_lfp.get("L23", 0.0)
        l_deep = 0.5 * (self.layer_lfp.get("L5", 0.0) + self.layer_lfp.get("L6", 0.0))
        trans_dipole = l_sup - l_deep

        eph_1d = torch.zeros(self.n, device=i_ff.device)
        for name, sl in self.slices.items():
            if sl.stop > sl.start:
                local_field = self.layer_lfp[name] + 0.3 * trans_dipole
                eph_1d[sl] = math.tanh(self.ephaptic_gain * local_field) * 1.5
        self.ephaptic_mv = eph_1d.unsqueeze(0).expand(b_sz, -1).contiguous()
        return self.ephaptic_mv
