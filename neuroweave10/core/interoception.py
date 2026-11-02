"""Интероцепция: нейроны-наблюдатели, читающие состояние соседей.

Слой саморефлексии: N_int «интеро-нейронов» на каждую зону получают не
входной сигнал, а внутреннее состояние соседнего пула (средний V_S,
энергия ATP, Ca-нагрузка, локальная частота) и кодируют его в
собственную спайковую активность — материал для отчёта AST о
состоянии организма (метапознание: сеть знает, *в каком* она состоянии,
а не только *что* обрабатывает). Связь: интеро-активность идёт в AST
как признак и в GW как дополнительный кандидат — состояние самой сети
может стать содержимым сознания.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class InteroOut:
    state_vec: torch.Tensor      # [Z*N_int] активность интеро-слоя (Insula SDR)
    summary: Dict[str, float]    # агрегированное само-состояние
    predicted_vec: Optional[torch.Tensor] = None   # аллостатическое предсказание PFC/ACC
    intero_error: float = 0.0    # ошибка интероцептивного предсказания δ_intero
    allostatic_drive: float = 0.0  # упреждающий аллостатический драйв (экономия АТФ / запрос сна)


class InteroceptiveField:
    """Островковая Кора — Передняя Поясная Кора (Insula–ACC Allostatic Self-Model,
    Инновация 2: Craig 2009; Seth 2013; Barrett 2017).

    Читает висцеральное состояние пулов (v_s, energy, ca, rate) + глобальные
    гомеостатические переменные, строит упреждающее аллостатическое предсказание
    будущего метаболического состояния и выдаёт:
      - спайковый/активностный вектор Insula для Глобального Рабочего Пространства (GW);
      - ошибку интероцептивного предсказания δ_intero;
      - упреждающий аллостатический сигнал (allostatic_drive) для саморегуляции
        до наступления метаболического истощения!"""

    def __init__(self, zone_names: list, n_int: int = 16, seed: int = 0):
        self.names = zone_names
        self.n_int = n_int
        self.total_dim = len(zone_names) * n_int
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(n_int, 4, generator=g) * 0.8
        self.bias = torch.randn(n_int, generator=g) * 0.2
        # Аллостатическая предиктивная матрица ACC → Insula (предсказание s_intero(t+1) из PFC и s_intero(t))
        self.w_allostatic = torch.eye(self.total_dim) * 0.75 + torch.randn(
            self.total_dim, self.total_dim, generator=g
        ) * (0.10 / max(1.0, self.total_dim ** 0.5))
        self.w_pfc_to_insula: Optional[torch.Tensor] = None
        self.pred_vec = torch.full((self.total_dim,), 0.5)
        self.energy_trend: float = 1.0
        self.intero_error: float = 0.0
        self.allostatic_drive: float = 0.0
        self.state_vec = torch.zeros(len(zone_names), n_int)

    def read(self, zone_states: Dict[str, torch.Tensor],
             pfc_spikes: Optional[torch.Tensor] = None,
             learn: bool = True) -> InteroOut:
        """zone_states[name] = [v_s_mean, energy_mean, ca_mean, rate]."""
        feats = torch.stack([zone_states[n] for n in self.names])  # [Z,4]
        act = torch.sigmoid(feats @ self.w.t() + self.bias)        # [Z,N_int]
        self.state_vec = act
        flat_act = act.flatten()

        # Аллостатическая ошибка предсказания: сравнение реального состояния тела
        # с тем, что ожидала кора на предыдущем шаге (Seth 2013 Interoceptive Inference)
        err_vec = flat_act - self.pred_vec.to(flat_act.device)
        self.intero_error = float(err_vec.pow(2).mean().sqrt().item())

        # Локальное обучение аллостатической модели (ACC → Insula) без backprop
        if learn:
            self.w_allostatic = (
                self.w_allostatic + 0.03 * torch.outer(err_vec, flat_act.detach())
            ).clamp(-1.5, 1.5)

        # Вычисляем новое аллостатическое предсказание на следующий шаг t + Δt
        next_pred = torch.sigmoid(self.w_allostatic.to(flat_act.device) @ flat_act)
        if pfc_spikes is not None:
            pfc_flat = pfc_spikes.detach().float().mean(0) if pfc_spikes.dim() == 2 else pfc_spikes.detach().float()
            if self.w_pfc_to_insula is None or self.w_pfc_to_insula.shape[1] != pfc_flat.numel():
                g = torch.Generator().manual_seed(17)
                self.w_pfc_to_insula = torch.randn(
                    self.total_dim, pfc_flat.numel(), generator=g
                ) * (0.15 / max(1.0, pfc_flat.numel() ** 0.5))
            if learn:
                self.w_pfc_to_insula = (
                    self.w_pfc_to_insula + 0.02 * torch.outer(err_vec, pfc_flat)
                ).clamp(-1.0, 1.0)
            next_pred = torch.sigmoid(
                self.w_allostatic.to(flat_act.device) @ flat_act
                + self.w_pfc_to_insula.to(flat_act.device) @ pfc_flat
            )
        self.pred_vec = next_pred.detach()

        # Отслеживание тренда энергии АТФ (колонка 1 в feats) и кальциевой нагрузки (колонка 2)
        mean_atp = float(feats[:, 1].mean().item())
        mean_ca = float(feats[:, 2].mean().item())
        atp_drop = max(0.0, self.energy_trend - mean_atp)
        self.energy_trend = 0.85 * self.energy_trend + 0.15 * mean_atp
        # Упреждающий аллостатический драйв: активируется ДО коллапса АТФ, если падает энергия
        # или растёт висцеральная ошибка предсказания
        self.allostatic_drive = float(min(
            1.0,
            max(0.0, 0.80 - mean_atp) * 1.5 + atp_drop * 8.0 + 0.35 * self.intero_error + 0.15 * max(0.0, mean_ca - 0.5)
        ))

        summary = {f"{n}_self": float(act[i].mean())
                   for i, n in enumerate(self.names)}
        summary["intero_error"] = round(self.intero_error, 4)
        summary["allostatic_drive"] = round(self.allostatic_drive, 4)
        return InteroOut(
            state_vec=flat_act,
            summary=summary,
            predicted_vec=self.pred_vec,
            intero_error=self.intero_error,
            allostatic_drive=self.allostatic_drive,
        )
