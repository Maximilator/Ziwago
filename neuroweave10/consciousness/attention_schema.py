"""Схема внимания AST (§II.4.3): внутренняя модель внимания системы.

AST строит контроллер, отвечающий: «что сейчас в фокусе, с какой силой,
насколько это точно и почему». Схема — не само внимание, а модель для
нисходящего контроля (Graziano 2013).

Состояние: current_focus (какой модуль), gain (сила), expected_accuracy,
reason (тиггер: novelty/erreur/drive), prediction (куда сместится фокус).
Обучение: L_schema = MSE(pred, фактический фокус) + MSE(gain, факт. ошибка).
Выход: корректирующий гейн → TRN (+α_AST·AST_gain).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from ..config import AttentionSchemaCfg


@dataclass
class ASTOut:
    focus: int
    gain: torch.Tensor          # [n_modules] корректирующий гейн TRN
    expected_accuracy: float
    reason: str
    prediction: int
    report: Dict                # интроспекция


class AttentionSchema:
    """Внутренняя схема внимания с предиктивным обучением."""

    MODULES = ["SENS", "ASSOC", "PFC", "HPC", "BG"]

    def __init__(self, cfg: AttentionSchemaCfg, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w_pred = torch.randn(cfg.n_modules, cfg.d_schema,
                                  generator=g) * 0.1
        self.state = torch.zeros(cfg.d_schema)
        self.gain = torch.ones(cfg.n_modules) * 0.5
        self.exp_acc = 0.5
        self.focus = 0
        self.reason = "baseline"

    # ------------------------------------------------------------------
    def step(self, novelty: float, error: float, drive: float,
             actual_gain: float, actual_focus: int,
             learn: bool = True) -> ASTOut:
        c = self.cfg
        # Обновление состояния схемы от признаков
        feat = torch.tensor([novelty, error, drive, actual_gain,
                             float(actual_focus) / c.n_modules])
        feat = torch.nn.functional.pad(feat, (0, max(
            0, c.d_schema - feat.numel())))[:c.d_schema]
        self.state = 0.8 * self.state + 0.2 * feat
        # Фокус: argmax модуля по схеме
        logits = self.w_pred @ self.state
        self.focus = int(logits.argmax())
        self.prediction = int(logits.topk(2).indices[-1])
        # Причина фокуса: доминирующий признак
        self.reason = ["novelty", "error", "drive",
                       "baseline"][int(torch.tensor(
                           [novelty, error, drive, 0.0]).argmax())]
        self.gain = torch.softmax(logits, dim=0)
        if learn:
            # L_schema: предсказание фокуса и гейна
            tgt = torch.zeros(c.n_modules)
            tgt[actual_focus] = 1.0
            self.w_pred += c.lr * torch.outer(tgt - self.gain, self.state)
            self.exp_acc += c.lr * (actual_gain - self.exp_acc)
        report = {
            "attention_state": {
                "current_focus": self.MODULES[self.focus],
                "gain": [round(float(g), 3) for g in self.gain],
                "expected_accuracy": round(self.exp_acc, 3),
                "reason": self.reason,
                "schema_loss": float((tgt - self.gain).pow(2).mean())
                             if learn else 0.0,
                "prediction": self.MODULES[self.prediction],
            }
        }
        return ASTOut(focus=self.focus,
                      gain=self.gain * c.alpha_ast,
                      expected_accuracy=self.exp_acc,
                      reason=self.reason, prediction=self.prediction,
                      report=report)
