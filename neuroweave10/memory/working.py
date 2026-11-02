"""Рабочая память: activity-silent хранилище (Lundqvist 2018) + NMDA-бамп.

PFC держит информацию не в персистентном огне, а в кратковременной
синаптической фацилитации STP (u_syn-прочность потенциированных связей)
— энергетически дешевле: спайков нет, память жива в весах-мостах.
Пробный спайк («зонд») вызывает повторную активацию именно того
паттерна — чтение. Во время активного удержания медленный NMDA
(τ≈100 мс, Compte 2000) поддерживает бамп без высокой частоты.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WMOut:
    reactivated: torch.Tensor    # [B,N] считанный паттерн
    stored_strength: float


class SilentWorkingMemory:
    """STP-хранилище: запись через фацилитацию, чтение — пробный спайк."""

    def __init__(self, n: int, tau_facil: float = 3000.0,
                 nmda_tau: float = 100.0):
        self.n = n
        self.tau_facil = tau_facil      # фацилитация живёт секунды
        self.nmda_tau = nmda_tau        # бамп-удержание 100 мс
        self.facil = torch.zeros(n, n)  # STP-потенциация
        self.bump = torch.zeros(n)      # активный NMDA-бамп

    def write(self, pattern: torch.Tensor, strength: float = 1.0) -> None:
        """Спайк-паттерн [B,N] → фацилитация его внутренних связей."""
        p = pattern.float().mean(0)
        self.facil += strength * torch.outer(p, p)
        self.facil.clamp_(0.0, 3.0)
        self.bump = p.clone()

    def tick(self, dt: float = 1.0, active_hold: bool = False) -> None:
        """Фацилитация тает медленно; бамп держится на NMDA-τ."""
        self.facil *= 1.0 - dt / self.tau_facil
        if active_hold:
            self.bump *= 1.0 - dt / self.nmda_tau
        else:
            self.bump *= 1.0 - dt / 20.0   # без удержания бамп гаснет

    def probe(self, cue: torch.Tensor, k: int = 8) -> WMOut:
        """Пробный спайк → реактивация сохранённого паттерна (k-WTA)."""
        c = cue.float().mean(0)
        resp = (self.facil @ c) + self.bump * 0.5
        topk = resp.topk(min(k, self.n)).values[-1]
        re = (resp >= topk).float()
        return WMOut(re.unsqueeze(0), float(self.facil.abs().mean()))
