"""Глобальное рабочее пространство (§II.4.1): Modern Hopfield Network.

Слоты K=4 (Cowan 2001) — когерентные пакеты информации d_GW.
Ignition: нелинейная фазовая транзиция (Dehaene & Changeux 2011):
  A = σ(γ_ign·(S - S_th)),  S = метрика внимания кандидата
  [NE] сдвигает порог: более резкий переход при NE↑.

Извлечение (Ramsauer et al. 2021):
  ξ_new = X·softmax(β_GW·Xᵀξ)    — один шаг Hopfield-обновления
  E(ξ)  = -½ξᵀXᵀXξ               — Lyapunov-энергия (монотонно ↓)

Бродкаст: чтение K-слотов всеми зонами (дешёвое чтение).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ..config import WorkspaceCfg


@dataclass
class GWOut:
    broadcast: torch.Tensor      # [d_gw] симфонический аккорд (все зоны вместе)
    content: List[torch.Tensor]  # активные партитуры слотов (SENS, ASSOC, PFC, CA1)
    ignition: float              # A ∈ [0,1] глобальная синхронизация
    winner_zone: int             # ведущая солирующая зона (для обратной совместимости AST)
    energy: float                # Lyapunov-энергия аккорда
    coherence_weights: Optional[List[float]] = None  # веса фазовой когерентности зон


class GlobalWorkspace:
    """Симфоническое Рабочее Пространство (Ziwago-Prime v4, Узел 2):
    Параллельная таламо-кортикальная интеграция через Modern Hopfield резонанс
    и фазовую когерентность LFP (Communication-Through-Coherence, Fries 2015)
    без подавления слабых сигналов."""

    def __init__(self, cfg: WorkspaceCfg):
        self.cfg = cfg
        self.x = torch.zeros(cfg.n_slots, cfg.d_gw)   # слоты-партитуры зон
        self.gate = torch.zeros(cfg.n_slots)          # резонансный вес каждого потока
        self.ignition = 0.0
        self.last_coherence: List[float] = []

    # ------------------------------------------------------------------
    def compete(self, candidates: List[torch.Tensor],
                ne: float,
                lfp_phases: Optional[List[float]] = None,
                beta_boost: float = 0.0) -> GWOut:
        """Интегрирует параллельные потоки от всех зон (SENS, ASSOC, PFC, CA1)
        в единый симфонический аккорд без уничтожения слабых сигналов."""
        c = self.cfg
        if not candidates:
            self.x *= c.decay
            return GWOut(torch.zeros(c.d_gw), [], 0.0, -1, 0.0, [])
        stack = torch.stack([c_.detach().float().flatten()[:c.d_gw]
                             for c_ in candidates])
        stack = torch.nn.functional.pad(
            stack, (0, max(0, c.d_gw - stack.shape[1])))[:, :c.d_gw]
        n_cand = stack.shape[0]

        # 1. Нормировка энергии пакета зоны + фазовая когерентность LFP (PLV):
        # разреженные бинарные спайки коры (SENS, ASSOC, PFC) и плотный вектор CA1
        # приводятся к сбалансированной шкале громкости оркестра (~0.8..1.2)!
        raw_norms = stack.norm(dim=1, keepdim=True).clamp(min=1e-6)
        has_signal = (raw_norms.squeeze(1) > 1e-4).float()
        stack_normed = (stack / raw_norms) * has_signal.unsqueeze(1)

        mean_field = stack_normed.mean(dim=0, keepdim=True)
        cos_coh = torch.nn.functional.cosine_similarity(
            stack_normed, mean_field.expand_as(stack_normed), dim=1).clamp(min=0.0)
        if lfp_phases is not None and len(lfp_phases) == n_cand:
            import math
            mean_phi = sum(lfp_phases) / max(1, len(lfp_phases))
            plv = torch.tensor([0.5 * (1.0 + math.cos((p - mean_phi) * 0.5)) for p in lfp_phases],
                               dtype=stack.dtype, device=stack.device)
            coh_gain = (0.55 + 0.45 * cos_coh) * (0.65 + 0.35 * plv)
        else:
            coh_gain = 0.65 + 0.35 * cos_coh

        # 2. Мягкая активация каждого параллельного потока (БЕЗ argmax-убийства остальных зон!):
        # фазовая когерентность LFP управляет весом партии каждой зоны в Симфоническом Аккорде!
        scores = has_signal * coh_gain
        beta = (c.beta_gw + float(beta_boost)) * (1.0 + 0.5 * ne)
        s_th_eff = c.s_th * 0.75
        a_raw = torch.sigmoid(c.gamma_ign * 0.5 * (scores - s_th_eff)) * has_signal
        a = torch.maximum(a_raw, 0.25 * coh_gain * has_signal)
        self.ignition = float(a_raw.max())
        lead_zone = int(a_raw.argmax()) if self.ignition > 0.5 else -1
        self.last_coherence = [round(float(w), 4) for w in a.tolist()]

        # 3. Параллельная запись: каждая зона обновляет свой собственный слот в симфонии!
        for i in range(min(n_cand, c.n_slots)):
            # Плавное вливание нового сигнала зоны в её партитуру слота i
            alpha_slot = float(max(0.25, a[i].item()))
            self.x[i] = (1.0 - alpha_slot) * (self.x[i] * c.decay) + alpha_slot * stack_normed[i]
            self.gate[i] = max(float(self.gate[i] * c.decay), float(a[i].item()))

        # 4. Взаимный резонанс Хопфилда между параллельными слотами:
        # каждый слот обогащается гармоническими обертонами других зон, но сохраняет своё лицо (75% свой + 25% резонанс)
        energy = 0.0
        for s in range(c.n_slots):
            if self.gate[s] <= 1e-3:
                continue
            xi = self.x[s]
            for _ in range(c.retrieval_steps):
                logits = beta * (self.x @ xi)
                xi_harm = self.x.t() @ torch.softmax(logits, dim=0)
                xi_new = 0.75 * xi + 0.25 * xi_harm
                xtx_xi = self.x.t() @ (self.x @ xi_new)
                energy_new = -0.5 * float((xi_new * xtx_xi).sum())
                xi = xi_new
            self.x[s] = xi
            energy = energy_new

        # 5. Симфонический бродкаст: взвешенная сумма всех одновременно звучащих партий
        broadcast = (self.x * self.gate.unsqueeze(1)).sum(0)
        return GWOut(broadcast=broadcast,
                     content=[self.x[i] for i in range(c.n_slots)
                              if self.gate[i] > 1e-3],
                     ignition=self.ignition, winner_zone=lead_zone,
                     energy=energy, coherence_weights=self.last_coherence)
