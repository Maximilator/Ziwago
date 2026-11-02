"""neuroweave9/cortex/endocannabinoid.py — Ретроградная эндоканнабиноидная система (Genesis volume.py).

Биологический контур (2-AG & CB1-рецепторы):
1. Синтез 2-AG (DAGL-альфа):
   d[2-AG]/dt = k_synth * sigma([Ca2+]_post - theta_ca) * (1 + 2*mglur) - k_magl * [2-AG]
2. Деградация через фермент MAGL (tau_phasic ~ 30-50 мс).
3. Ретроградная супрессия высвобождения (CB1-рецептор на пресинаптической терминали):
   P_rel = P0 * (1 - [2-AG]^n / ([2-AG]^n + IC50^n))
4. Attractor Ejection (усталость аттрактора):
   fatigue = delta * integral([eCB]) dt
   Когда колонка/нейроны воспроизводят один и тот же паттерн (зацикливание),
   высокий кальций стимулирует лавинообразный синтез 2-AG, который давит рекуррентные
   и апикальные синапсы, сбрасывая зацикленный аттрактор.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class EndocannabinoidSystem(nn.Module):
    """2-AG-mediated retrograde suppression and attractor fatigue (Genesis volume.py)."""

    def __init__(
        self,
        n_features: int = 32,
        k_synth: float = 0.18,
        theta_ca: float = 0.18,
        tau_phasic: float = 25.0,    # ms: быстрое накопление усталости за 15-25 тактов
        tau_tonic: float = 250.0,   # ms
        IC50: float = 0.32,
        hill_n: float = 2.0,
        P0: float = 1.0,
        delta_fatigue: float = 0.15,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.k_synth = float(k_synth)
        self.theta_ca = float(theta_ca)
        self.tau_phasic = float(tau_phasic)
        self.tau_tonic = float(tau_tonic)
        self.IC50 = float(IC50)
        self.hill_n = float(hill_n)
        self.P0 = float(P0)
        self.delta_fatigue = float(delta_fatigue)

        self.k_magl = 1.0 / max(self.tau_phasic, 1e-3)
        self.k_magl_tonic = 1.0 / max(self.tau_tonic, 1e-3)

        self.register_buffer("ag2", torch.zeros(1, self.n_features, device=device), persistent=False)
        self.register_buffer("tonic", torch.zeros(1, self.n_features, device=device), persistent=False)
        self.register_buffer("_ecb_integral", torch.zeros(1, self.n_features, device=device), persistent=False)

    def reset_state(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.ag2 = torch.zeros(batch_size, self.n_features, device=device, dtype=dtype)
        self.tonic = torch.zeros(batch_size, self.n_features, device=device, dtype=dtype)
        self._ecb_integral = torch.zeros(batch_size, self.n_features, device=device, dtype=dtype)

    def _align(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.n_features:
            return x
        dev = x.device
        dtype = x.dtype
        b = x.shape[0] if x.dim() > 1 else 1
        out = torch.zeros(b, self.n_features, device=dev, dtype=dtype)
        m = min(x.shape[-1], self.n_features)
        out[..., :m] = x[..., :m]
        return out

    @torch.no_grad()
    def step(
        self,
        postsynaptic_calcium: torch.Tensor,
        mglur_activation: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Обновление динамики 2-AG и вычисление коэффициента CB1-супрессии."""
        ca = self._align(postsynaptic_calcium)
        mglur = self._align(mglur_activation) if mglur_activation is not None else torch.zeros_like(ca)

        if self.ag2 is None or self.ag2.shape[0] != ca.shape[0] or self.ag2.device != ca.device:
            self.reset_state(ca.shape[0], ca.device, ca.dtype)

        # Синтез 2-AG при превышении порога кальция (активный паттерн)
        synth_drive = torch.sigmoid(10.0 * (ca - self.theta_ca))
        mglur_boost = 1.0 + 2.0 * mglur
        synthesis = self.k_synth * synth_drive * mglur_boost

        # Деградация MAGL
        degradation = self.k_magl * self.ag2
        d_ag2 = synthesis - degradation
        self.ag2 = torch.clamp(self.ag2 + d_ag2 * dt, min=0.0, max=5.0)

        # Тонический фон
        tonic_inf = self.ag2 * 0.10
        self.tonic = self.tonic + (dt / self.tau_tonic) * (tonic_inf - self.tonic)
        self.tonic.clamp_(min=0.0)

        # Интеграл для катапультирования из аттрактора
        self._ecb_integral = (self._ecb_integral + (self.ag2 + self.tonic) * dt) * (
            1.0 - dt / (self.tau_tonic * 2.0)
        )
        self._ecb_integral.clamp_(min=0.0)

        return self.get_release_modulation()

    @torch.no_grad()
    def get_release_modulation(self) -> torch.Tensor:
        """CB1-супрессия высвобождения синапсов: P_rel in [0.15, 1.0]."""
        ag_n = torch.pow(torch.clamp(self.ag2, min=1e-12), self.hill_n)
        ic_n = self.IC50 ** self.hill_n
        suppression = ag_n / (ag_n + ic_n)
        return torch.clamp(self.P0 * (1.0 - suppression), min=0.15, max=1.0)

    @torch.no_grad()
    def get_attractor_fatigue(self) -> torch.Tensor:
        return self.delta_fatigue * self._ecb_integral
