"""Фаза 3 Глобального Плана v8.0 (snnscience.docx §3.8, Изъян 8):
Сквозное Предиктивное Кодирование без линейного Readout-классификатора и без Backprop
(`EndToEndPredictiveReadout`: Local Dendritic Contrastive Predictive Coding + Kolen-Pollack Reciprocal Alignment).

Устраняет Изъян 8 («Зависимость от плотного линейного Readout-слоя с дельта-правилом на выходе»):
  1. Никакого внешнего линейного классификатора, обучаемого градиентным спуском или глобальным
     дельта-правилом по one-hot вектору длины V:
     - Каждое понятие/действие представлено каноническим разреженным аттрактором коры `A(y)` в пространстве L5;
     - Прямая проекция `W_ff` и обратная апикальная проекция `B_fb` выравниваются локальным
       реципрокным законом Колена–Поллака (`Kolen & Pollack 1994`):
           ΔW_ff = η * (e_post ⊗ x_pre) - λ_kp * W_ff
           ΔB_fb = η * (x_pre ⊗ e_post) - λ_kp * B_fb
       что гарантирует сходимость B_fb -> W_ff^T без транспонирования весов и без передачи градиентов!
  2. Локальный дендритный контрастивный баланс (`Predictive InfoNCE / Wake-Contrastive Hebbian`):
     - Базальный компартмент получает текущий драйв `u = W_ff @ x`, апикальный компартмент получает
       целевой прототип `A(y)` (или предсказание следующего шага) минус подавление конкурирующих
       аттракторов через PV-интернейроны (`k-WTA Negative Phase`).
  3. Сквозной выбор действия осуществляется чистым резонансом аттрактора в коре (косинусное/скалярное
     совпадение L5-активности с таблицей разреженных кодов без единой плотной матрицы классификатора).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import torch


@dataclass
class E2EPredictiveStepOut:
    predicted_class: int
    attractor_resonance: float
    apical_error_norm: float
    weight_alignment_cosine: float
    used_backprop: bool = False


class EndToEndPredictiveCodingLoop:
    """Сквозной контур предиктивного кодирования с реципрокным выравниванием Колена–Поллака
    и локальным дендритным контрастивным обучением без плотного Readout-слоя."""

    def __init__(
        self,
        d_in: int = 64,
        d_hidden: int = 96,
        d_attractor: int = 64,
        n_classes: int = 32,
        lr: float = 0.08,
        kp_decay: float = 0.015,
        seed: int = 42,
    ):
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.d_attractor = int(d_attractor)
        self.n_classes = int(n_classes)
        self.lr = float(lr)
        self.kp_decay = float(kp_decay)

        g = torch.Generator().manual_seed(seed)
        # Прямые и обратные синаптические матрицы инициализируются НЕЗАВИСИМО (без B = W^T!)
        self.w_ff1 = torch.randn(self.d_hidden, self.d_in, generator=g) * (1.0 / math.sqrt(self.d_in))
        self.b_fb1 = torch.randn(self.d_in, self.d_hidden, generator=g) * (1.0 / math.sqrt(self.d_hidden))

        self.w_ff2 = torch.randn(self.d_attractor, self.d_hidden, generator=g) * (1.0 / math.sqrt(self.d_hidden))
        self.b_fb2 = torch.randn(self.d_hidden, self.d_attractor, generator=g) * (1.0 / math.sqrt(self.d_attractor))

        # Фиксированные разреженные кортикальные аттракторы классов A(y) ∈ R^{n_classes × d_attractor}
        # (не обучаемый линейный классификатор, а ортогональные клеточные ансамбли L5!)
        raw_att = torch.randn(self.n_classes, self.d_attractor, generator=g)
        top_k = max(4, self.d_attractor // 4)
        thr = raw_att.topk(top_k, dim=-1).values[:, -1:]
        sparse_att = (raw_att >= thr).float()
        self.class_attractors = sparse_att / sparse_att.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def reciprocal_alignment_cosine(self) -> float:
        """Косинусное сходство между прямой матрицей W_ff2 и транспонированной обратной B_fb2^T
        (мерило сходимости закона Колена–Поллака без использования .t() при обучении)."""
        w_flat = self.w_ff2.flatten()
        b_flat = self.b_fb2.t().flatten()
        cos = float(
            (w_flat @ b_flat)
            / (w_flat.norm().clamp(min=1e-8) * b_flat.norm().clamp(min=1e-8))
        )
        return round(cos, 4)

    def step(
        self,
        x_in: torch.Tensor,
        target_class: int | None = None,
        da_rpe: float = 0.8,
        learn: bool = True,
    ) -> E2EPredictiveStepOut:
        """Выполняет прямой проход и (при learn=True) локальное предиктивное обновление
        по правилу Рао–Балларда + Колена–Поллака без градиентов и без линейного readout."""
        with torch.no_grad():
            x = x_in.detach().flatten()[: self.d_in]
            if x.numel() < self.d_in:
                x = torch.nn.functional.pad(x, (0, self.d_in - x.numel()))
            x = x / x.norm().clamp(min=1e-6)

            # Базальный прямой драйв слоя 1 (L2/3) и слоя 2 (L5 аттрактор)
            h1 = torch.tanh(self.w_ff1 @ x)
            u2 = self.w_ff2 @ h1
            u2_norm = u2 / u2.norm().clamp(min=1e-6)

            # Сквозное декодирование через чистый резонанс с ансамблями аттракторов A(y)
            resonances = self.class_attractors @ u2_norm
            pred_cls = int(resonances.argmax().item())
            best_res = float(resonances[pred_cls].item())

            apical_err_norm = 0.0
            if learn and target_class is not None:
                tgt = int(target_class) % self.n_classes
                target_att = self.class_attractors[tgt]
                # Негативная фаза от конкурирующего аттрактора (PV-торможение второго пика)
                res_masked = resonances.clone()
                res_masked[tgt] = -1e9
                neg_cls = int(res_masked.argmax().item())
                neg_att = self.class_attractors[neg_cls]

                # Локальная дендритная ошибка в слое аттрактора L5:
                e2 = (target_att - u2_norm) - 0.25 * torch.relu(resonances[neg_cls]) * neg_att
                apical_err_norm = float(e2.norm().item())

                # Апикальная ошибка проецируется вниз через физические обратные синапсы B_fb2
                e1 = (self.b_fb2 @ e2) * (1.0 - h1 * h1)

                eff_lr = self.lr * (0.5 + 0.5 * max(0.0, float(da_rpe)))
                # Закон Колена–Поллака для пары (W_ff2, B_fb2):
                self.w_ff2.add_(eff_lr * torch.outer(e2, h1) - self.kp_decay * self.w_ff2)
                self.b_fb2.add_(eff_lr * torch.outer(h1, e2) - self.kp_decay * self.b_fb2)

                # Закон Колена–Поллака для пары (W_ff1, B_fb1):
                self.w_ff1.add_(eff_lr * torch.outer(e1, x) - self.kp_decay * self.w_ff1)
                self.b_fb1.add_(eff_lr * torch.outer(x, e1) - self.kp_decay * self.b_fb1)

            return E2EPredictiveStepOut(
                predicted_class=pred_cls,
                attractor_resonance=round(best_res, 4),
                apical_error_norm=round(apical_err_norm, 4),
                weight_alignment_cosine=self.reciprocal_alignment_cosine(),
                used_backprop=False,
            )
