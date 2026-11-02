"""Объёмная передача (§II.3.2): NO-поле, 2-AG, H₂S→D-серин.

NO: газ, диффундирует от источника ~100 мкм за ~1 с; действует на все
синапсы в радиусе (гетеросинаптическая LTP — несвязанные синапсы
потенцируются «попутно»). Модель: источники — постсинаптические
нейроны с высоким Ca2+; поле — сумма функций Грина
  G(r,t) = exp(-r²/4Dt)/(4πDt)^{3/2} · exp(-t/τ_NO)

2-AG: ретроградный эндоканнабиноид (синтез по пост-Ca2+, супрессия
пресинаптического высвобождения P_release) + механизм «выталкивания
аттрактора»: длительная активность → усталость → выброс из аттрактора
(инсомния-свидетель при гиперактивации).

H₂S → D-серин → NMDA: астроглиальный путь; H₂S повышает доступность
ко-агониста NMDA (усиливает FB-пластичность при стрессе).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


class VolumeTransmission:
    """Поля NO/2-AG/H2S по нейронам (позиция = индекс; радиус в соседях)."""

    def __init__(self, n: int, no_tau: float = 1000.0,
                 no_d: float = 0.33, radius: int = 16,
                 h2s_tau: float = 60000.0):
        self.n = n
        self.no = torch.zeros(n)          # NO-поле по позициям
        self.h2s = torch.zeros(n)         # H2S-поле
        self.ecb_field = torch.zeros(n)   # локальный 2-AG
        self.tau_no, self.d_no = no_tau, no_d
        self.radius = radius
        self.tau_h2s = h2s_tau
        # Ядро диффузии: G(Δr) ~ exp(-Δr²/(4Dt))
        kr = torch.arange(-radius, radius + 1, dtype=torch.float32)
        self.kernel = torch.exp(-kr.pow(2) / (4.0 * no_d * 50.0))
        self.kernel /= self.kernel.sum()
        # Усталость аттрактора (2-AG ejection)
        self.attract_dur = torch.zeros(n)

    # ------------------------------------------------------------------
    def step(self, ca: torch.Tensor, spikes: torch.Tensor,
             dt_ms: float = 1.0) -> torch.Tensor:
        """ca [B,N] пост-Ca2+ → источники NO; возвращает поле LTP."""
        src = ca.mean(0)                  # [N]
        # Источник NO при Ca2+ > порога; диффузия свёрткой + распад
        emit = (src > 0.5).float() * src
        self.no = self.no * (1.0 - dt_ms / self.tau_no)
        if emit.any():
            conv = torch.nn.functional.conv1d(
                emit.view(1, 1, -1), self.kernel.view(1, 1, -1),
                padding=self.radius).view(-1)
            self.no = self.no + conv
        # H₂S: медленное накопление от активности (астроцитарное)
        self.h2s += (spikes.mean(0) * 0.001
                     - self.h2s / self.tau_h2s * dt_ms)
        # 2-AG: ретроградная супрессия — высокий Ca2+ → синтез
        self.ecb_field += torch.sigmoid((src - 0.5) * 8.0) * 0.02 \
            - self.ecb_field * (dt_ms / 10000.0)
        self.ecb_field.clamp_(0.0, 2.0)
        # Усталость аттрактора: долгая активность → выталкивание
        self.attract_dur = torch.where(spikes.mean(0) > 0.5,
                                       self.attract_dur + 1.0,
                                       torch.zeros(self.n))
        eject = (self.attract_dur > 200.0).float() * self.ecb_field
        self.last_eject = eject.clamp(0.0, 1.0)
        if eject.any():
            self.attract_dur *= (1.0 - self.last_eject)
        return self.no

    def ltp_field(self) -> torch.Tensor:
        """Поле гетеросинаптической LTP: G-нормированное NO."""
        return (self.no / self.no.max().clamp(min=1e-6)).clamp(0.0, 1.0)

    def zone_fields(self, offset: int, length: int) -> dict[str, torch.Tensor]:
        """Возвращает срезы полей NO, 2-AG, H2S и выталкивания аттрактора для зоны."""
        sl = slice(offset, min(self.n, offset + length))
        no_norm = self.ltp_field()[sl]
        ecb = self.ecb_field[sl]
        h2s = self.h2s[sl]
        ej = getattr(self, "last_eject", torch.zeros(self.n))[sl]
        return {
            "no_ltp": no_norm,
            "ecb": ecb,
            "h2s": h2s,
            "eject": ej,
        }

