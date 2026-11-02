"""Таламус: FO-релей (§II.1.2) и HO-ядро (§II.1.4).

FO (first order, LGN/VPL): передаёт сенсорику в L4. Два режима:
  tonic — линейный релей высокой точности ([ACh]↑, [NE]↑);
  burst — нелинейный T-type Ca2+-burst «wake-up call»
          (молчание афферентов → деполяризация после гиперполяризации
          → нейрон «кричит»). Режим выбирается модуляторами и паузой
          входа (Sherman 2001).

HO (higher order, Pulvinar/MD): релей кортико-кортикального драйва —
копия L5-эфферентов идёт через HO в следующую зону (трансталамический
путь). WTA выбирает «сильнейшие сообщения»; базальные ганглии
разблокируют релей через дизингибицию (GPi/SNr → таламус).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


class ThalamusFO:
    """First-order таламус с переключением tonic/burst, T-type Ca2+ инактивацией h_T
    и TRN-гейтом (Sherman 2001, Genesis TcRelayCore)."""

    def __init__(self, n: int, name: str = "FO"):
        self.n, self.name = n, name
        self.mode = "tonic"
        self.silence_ms = torch.zeros(n)  # длительность молчания канала
        self.burst_ms = torch.zeros(n)    # оставшиеся такты burst-режима
        self.h_t = torch.zeros(n)         # доступность T-type Ca2+ каналов [0..1]
        self.gate = torch.ones(n)         # TRN-гейт [0..1]

    def step(self, x: torch.Tensor, ach: float, ne: float,
             boundary_burst: bool = False) -> torch.Tensor:
        """x [B,N] сенсорика → релейный выход в L4.

        LTS-вспышка (T-type Ca2+ burst) срабатывает только:
          - после длительного молчания (>50 мс) или во сне/пробуждении (vigilance < 0.5),
          - либо на границе предложения/клаузы или смене темы (`boundary_burst=True`),
            давая коре чёткий фазовый маркер начала новой синтаксической мысли!"""
        active = (x.abs().sum(0) > 1e-3).float()
        self.silence_ms = torch.where(active > 0, torch.zeros_like(active),
                                      self.silence_ms + 1.0)
        vigilance = ach + ne
        # В молчании, при низкой бодрости или на паузе конца предложения (boundary)
        # кальциевые каналы Т-типа де-инактивируются (h_T → 1)
        b_drive = 0.8 if boundary_burst else 0.0
        h_inf = torch.clamp((self.silence_ms / 50.0)
                            + max(0.0, 0.6 - vigilance) + b_drive, 0.0, 1.0)
        self.h_t = 0.88 * self.h_t + 0.12 * h_inf
        if vigilance < 0.5 or self.silence_ms.mean() > 50.0 or boundary_burst:
            self.mode = "burst"
        else:
            self.mode = "tonic"
        if self.mode == "burst":
            # Burst: нелинейная амплификация на границе фразы или после молчания
            gain = torch.where(self.silence_ms > 50.0,
                               torch.full_like(x[0], 3.0),
                               1.0 + 1.2 * self.h_t)
            out = torch.tanh(x * 2.0) * gain
            self.h_t = torch.where(active > 0, self.h_t * 0.25, self.h_t)
        else:
            # Внутри предложения (tonic) — чистый линейный релей без хаотичных вспышек!
            out = x
        return out * self.gate.unsqueeze(0)

    def spindle(self, t_ms: int, freq_hz: float = 10.0) -> torch.Tensor:
        """Сонное веретено: TRN-пачки ~10 Гц во время сна (NREM).
        Возвращает осциллирующий гейт — таламические реле пульсируют,
        запуская SWR-обвязку гиппокампа синхронно с корой."""
        import math
        return torch.full((self.n,), 0.5 + 0.5 * math.sin(
            2.0 * math.pi * freq_hz * t_ms / 1000.0))


class ThalamicHO:
    """Higher-order ядро (Pulvinar / MD): Core + Matrix параллельная маршрутизация
    (Jones 2001; Ziwago-Prime v4, Узел 2).

    * Core-путь (Parvalbumin+): выделяет ведущие L5-драйверы (winner_mask);
    * Matrix-путь (Calbindin+): сохраняет и синхронизирует слабые сигналы
      по фазовой когерентности LFP, не обнуляя тонкие детали контекста!"""
    def __init__(self, n_in: int, n_out: int, name: str = "HO"):
        self.n_in, self.n_out, self.name = n_in, n_out, name
        self.w = torch.randn(n_out, n_in) * 0.6
        self.w_matrix = torch.randn(n_out, n_in) * 0.25
        self.winner_mask = torch.zeros(n_in)
        self.open_gate = 0.0              # BG-гейт
        self.matrix_coherence: float = 1.0

    def step(self, l5: torch.Tensor, bg_release: float,
             lfp_coherence: float = 1.0) -> torch.Tensor:
        """l5 [B,N_in] драйв → Core + Matrix выход в следующую зону.

        bg_release: дизингибиция от BG (0=закрыт, 1=открыт)."""
        self.open_gate = bg_release
        self.matrix_coherence = float(max(0.2, min(1.5, lfp_coherence)))
        if bg_release < 0.5:
            return torch.zeros(l5.shape[0], self.n_out, device=l5.device)
        # Core-путь: отмечает топ-K сильнейших каналов для диагностики
        k = max(1, self.n_in // 4)
        thr = l5.topk(k, dim=1).values[:, -1:]
        core_mask = (l5 >= thr).float()
        self.winner_mask = core_mask.mean(0)
        # Matrix-путь: пропускает ВСЕ каналы (включая слабые фоновые обертоны),
        # модулируя их фазовой когерентностью LFP!
        core_drive = (l5 * (0.65 + 0.35 * core_mask)) @ self.w.t()
        matrix_drive = (l5 @ self.w_matrix.t()) * (0.35 * self.matrix_coherence)
        return torch.tanh(core_drive + matrix_drive) * self.open_gate
