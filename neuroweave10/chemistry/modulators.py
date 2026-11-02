"""Нейромодуляторное облако (§1.4): все оси и их метаболизм.

Оси: DA (phasic/tonic), NE, ACh, 5-HT, Orexin, Adenosine, Cortisol,
Lactate/ATP. Каждая — скалярный уровень с экспоненциальной τ-динамикой
и источниками (RPE→DA, novelty→NE, внимание→ACh, терпение→5HT,
циркадный/драйв→Orexin, активность→Adenosine, стресс→Cortisol,
спайки→Lactate-дефицит).

Аденозин (Borbely two-process):
  dS/dt = α·A·W - β·S·(1-W);   W=1 бодрствование, 0 — сон
Орексин vs аденозин: двааттракторная система — [OX]<θ_collapse при
S>θ_sleep → потеря wake-аттрактора → сон.

f_mod(M) = (α_DA·DA + β_NE·NE)·(1 - κ_eCB·2-AG)·Φ_CORT
           ·H(E-θ_E)·H(θ_S - S)      — гейт пластичности

Φ_CORT — кортизоловый режим (MR/GR): низкий [GC] → MR (пластичность),
высокий → GR (кристаллизация + негеномный eCB-путь).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..config import ChemistryCfg


def heaviside(x: float, th: float) -> float:
    return 1.0 if x > th else 0.0


@dataclass
class ChemState:
    da: float = 0.5
    ne: float = 0.3
    ach: float = 0.5
    ht: float = 0.3
    ox: float = 0.6
    ado: float = 0.0       # аденозин — давление сна S
    cort: float = 0.1      # кортизол [GC]
    lactate: float = 1.0   # глобальный запас ATP/лактата
    ecb: float = 0.0       # 2-AG
    no: float = 0.0        # интеграл NO
    h2s: float = 0.0       # H2S → D-serine
    zn: float = 0.5
    mg: float = 1.0


class NeuroChemistry:
    """Скалярные модуляторы с биологическими τ."""

    def __init__(self, cfg: ChemistryCfg):
        self.cfg = cfg
        self.s = ChemState()
        self.is_awake = True
        self.ticks_awake = 0
        self.ticks_asleep = 0

    # ------------------------------------------------------------------
    def phi_cort(self) -> float:
        """Φ_CORT: MR-режим (пластичность) при GC<θ_stress, GR при выше."""
        gc = self.s.cort
        th = self.cfg.theta_stress
        if gc < th:
            return 1.0                       # MR: окно пластичности
        return max(0.1, th / (gc + 1e-6))    # GR: закрытие окна

    def f_mod(self, energy_ok: float) -> float:
        """Химический гейт трёхфакторного правила."""
        c, s = self.cfg, self.s
        ecb_gate = max(0.15, min(1.0, 1.0 - c.kappa_ecb * s.ecb))
        return ((c.alpha_da * s.da + c.beta_ne * s.ne)
                * ecb_gate
                * self.phi_cort()
                * energy_ok
                * heaviside(c.sleep_threshold, s.ado))  # H(θ_S - S)

    def update(self, activity: float, rpe: float, novelty: float,
               attention: float, drive: float, stress: float,
               spikes: float, dt_ms: float = 1.0) -> ChemState:
        """Один такт облака: все оси, wake/sleep бистабильность."""
        c, s = self.cfg, self.s
        dt = dt_ms
        w = 1.0 if self.is_awake else 0.0
        # DA: фазический отклик на RPE + тонический дрейв к средней награде
        s.da += dt * (-(s.da - 0.5) / c.da_tau_tonic) + rpe * 0.05
        s.da = max(0.0, min(2.0, s.da))
        # NE: новизна/сюрприз
        s.ne += dt * (-(s.ne - 0.3) / c.ne_tau) + novelty * 0.1
        s.ne = max(0.0, min(2.0, s.ne))
        # ACh: внимание/обучение
        s.ach += dt * (-(s.ach - 0.4) / c.ach_tau) + attention * 0.05
        # 5-HT: терпение/ожидание награды
        s.ht += dt * (-(s.ht - 0.3) / c.ht_tau) + max(0.0, -rpe) * 0.02
        # Аденозин: процесс S (Borbely)
        s.ado += dt * (c.ado_wake_gain * activity * w
                       - c.ado_sleep_gain * s.ado * (1.0 - w))
        s.ado = max(0.0, min(1.0, s.ado))
        # Орексин: поддержка wake-аттрактора, падает при давлении сна
        s.ox += dt * (-(s.ox - drive) / c.ox_tau)
        s.ox = max(0.0, min(1.0, s.ox))
        # Wake/sleep: S>θ_sleep и OX<θ_collapse → потеря бодрствования
        if self.is_awake and s.ado > c.sleep_threshold \
                and s.ox < c.ox_collapse + 0.2:
            self.is_awake = False
            self.ticks_asleep = 0
        elif not self.is_awake and s.ado < c.sleep_threshold * 0.5:
            self.is_awake = True
            self.ticks_awake = 0
        self.ticks_awake += int(self.is_awake)
        self.ticks_asleep += int(not self.is_awake)
        # Кортизол: стресс (негеномный быстрый вход + медленный распад)
        s.cort += dt * (stress * 0.01 - s.cort / c.cort_tau_nongenomic)
        s.cort = max(0.0, min(2.0, s.cort))
        # Лактат/ATP: спайки тратят, покой восстанавливает (ANLS-шаттл)
        s.lactate += dt * (-spikes * c.lactate_spend
                           + c.lactate_recover * (1.0 - s.lactate))
        s.lactate = max(0.0, min(1.0, s.lactate))
        # 2-AG: синтез по Ca2+-следу (k_synth·σ(Ca-θ)) - MAGL-гидролиз
        return self.s

    def ecb_update(self, ca_mean: float, dt_ms: float = 1.0) -> float:
        """d[2-AG]/dt = k_synth·max(0, σ([Ca2+]-θ_Ca) - σ(-θ_Ca)) - k_MAGL·[2-AG]."""
        c = self.cfg
        import math
        base = 1.0 / (1.0 + math.exp(0.5 * 8.0))
        raw_sig = 1.0 / (1.0 + math.exp(-(ca_mean - 0.5) * 8.0))
        synth = c.ecb_synth * max(0.0, raw_sig - base) * (dt_ms / 20.0)
        self.s.ecb += synth - self.s.ecb * (dt_ms / min(c.ecb_tau, 250.0))
        self.s.ecb = max(0.0, min(1.2, self.s.ecb))
        return self.s.ecb

    def p_release(self, p0: float = 1.0) -> float:
        """P_release = P0·(1 - [2-AG]^n/([2-AG]^n+IC50^n)), n=2."""
        e = self.s.ecb
        ic = self.cfg.ecb_ic50
        return p0 * (1.0 - e ** 2.0 / (e ** 2.0 + ic ** 2.0))
