"""Мета-когнитивная голова HOT (§II.4.2).

Наблюдает GW и выдаёт:
  confidence     — «насколько система уверена в текущем решении»
  predicted_loss — L̂: предсказанная ошибка следующего шага
  surprise       — |фактический err - L̂| (мета-сюрприз)
  novelty        — расстояние GW-контента к эпизодической базе
  uncertainty    — декомпозиция epistemic (дисперсия по ансамблю
                   извлечений GW) / aleatoric (шум входа)

Обучение: L_meta = MSE + λ·Brier (калибровка уверенности).
M-ratio = meta-d'/d' — прокси метакогнитивной чувствительности.

Выход управляет поведением: confidence < θ_low гейтит действия,
увеличивает β_GW (крутизну ignition) и приоритет реплея.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional

import torch

from ..config import MetaHeadCfg


@dataclass
class MetaOut:
    confidence: float
    predicted_loss: float
    surprise: float
    novelty: float
    epistemic: float
    aleatoric: float
    m_ratio: float
    gates: Dict[str, float]  # → поведение/BG/TRN


class MetaHead:
    """Биофизическая мета-когнитивная популяция (BA10 / префронтальный полюс,
    Genesis level4/meta_head.py): фиксированная рекуррентная проекция в спайковый
    резервуар + чисто локальная синаптическая пластичность без backprop."""

    def __init__(self, cfg: MetaHeadCfg, d_gw: int, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        # Фиксированная случайная дендритная экспансия из GW в популяцию BA10 (не обучается backprop'ом!)
        self.w_in = torch.randn(cfg.d_hidden, d_gw, generator=g) / max(1.0, d_gw ** 0.5)
        self.w_rec = torch.randn(cfg.d_hidden, cfg.d_hidden, generator=g) * (0.25 / max(1.0, cfg.d_hidden ** 0.5))
        self.w_rec.fill_diagonal_(0.0)
        # Локальные синаптические веса 3 популяционных выходов: [confidence, predicted_loss, novelty]
        self.w_out = torch.zeros(3, cfg.d_hidden)
        self.b_out = torch.tensor([0.0, 0.0, -0.5])
        # Состояние популяции BA10 и след допустимости (eligibility)
        self.h_state = torch.zeros(cfg.d_hidden)
        self.trace_gw = torch.zeros(d_gw)
        self.elig = torch.zeros(3, cfg.d_hidden)
        # Бегущий центроид опыта для оценки истинной новизны
        self.gw_centroid = torch.zeros(d_gw)
        self.err_hist: List[float] = []
        self.conf_hist: List[float] = []

    # ------------------------------------------------------------------
    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h @ self.w_out.t() + self.b_out)   # [conf, L̂, nov]

    def step(self, gw_broadcast: torch.Tensor,
             gw_content: List[torch.Tensor],
             actual_error: float,
             reward: float = 0.0,
             sensory_frame: Optional[torch.Tensor] = None) -> MetaOut:
        c = self.cfg
        gw = gw_broadcast.detach().float().flatten()
        if gw.numel() != self.trace_gw.numel():
            gw = torch.nn.functional.pad(gw, (0, max(0, self.trace_gw.numel() - gw.numel())))[:self.trace_gw.numel()]
        if sensory_frame is not None:
            sf = sensory_frame.detach().float().flatten()
            sf_pad = torch.nn.functional.pad(sf, (0, max(0, gw.numel() - sf.numel())))[:gw.numel()]
            gw = 0.6 * gw + 0.4 * sf_pad
        self.trace_gw = self.trace_gw * 0.65 + gw * 0.35

        # Рекуррентный шаг популяции BA10 (без градиентов и без скрытого backprop)
        drive = self.trace_gw @ self.w_in.t() + self.h_state @ self.w_rec.t()
        self.h_state = 0.65 * self.h_state + 0.35 * torch.tanh(drive)
        out = self._readout(self.h_state)
        conf, l_hat = float(out[0]), float(out[1])

        # Истинная новизна опыта: отклонение от выученного центроида знакомых состояний
        nov_dist = float((gw - self.gw_centroid).pow(2).mean().sqrt().item())
        nov = min(1.0, math.tanh(nov_dist * 2.5))
        self.gw_centroid = 0.90 * self.gw_centroid + 0.10 * gw

        # Эпистемическая неопределённость (BA10): чувствительность популяции к мембранному
        # шуму + незнакомость паттерна относительно выученного следа (Genesis MetaHead)
        ens = [self._readout(torch.tanh(drive + torch.randn_like(drive) * 0.45))
               for _ in range(c.n_ensemble)]
        ens_var = float(torch.stack(ens).var(0, unbiased=False).mean())
        epistemic = float(min(1.0, ens_var * 15.0 + 0.35 * nov * min(1.0, actual_error)))
        # Алеаторная: шум входа — дисперсия по слотам
        if gw_content:
            aleatoric = float(torch.stack([x.float().flatten()
                                           for x in gw_content])
                              .var(0, unbiased=False).mean())
        else:
            aleatoric = 0.5
        surprise = abs(actual_error - l_hat)
        self.err_hist.append(actual_error)
        self.conf_hist.append(conf)

        # Чисто локальное однослойное дельта-правило со следом допустимости (без backprop в w_in!):
        err_clipped = min(1.0, max(0.0, float(actual_error)))
        target = torch.tensor([1.0 - err_clipped, err_clipped, nov])
        local_err = target - out  # [3]
        self.elig = 0.75 * self.elig + torch.outer(local_err, self.h_state)
        lr_eff = max(c.lr * 15.0, 0.15) * (1.0 + 0.5 * abs(float(reward)) + 0.5 * surprise)
        self.w_out = (self.w_out + lr_eff * self.elig).clamp(-3.0, 3.0)
        self.b_out = (self.b_out + lr_eff * 0.5 * local_err).clamp(-2.5, 2.5)

        # M-ratio: meta-d'/d' прокси — корр(conf, точность)
        if len(self.err_hist) > 8:
            import numpy as np
            c_arr = np.asarray(self.conf_hist[-64:])
            e_arr = 1.0 - np.asarray(self.err_hist[-64:])
            m = float(np.corrcoef(c_arr, e_arr)[0, 1]) \
                if c_arr.std() > 1e-6 and e_arr.std() > 1e-6 else 0.0
            m_ratio = m if np.isfinite(m) else 0.0
        else:
            m_ratio = 0.0
        gates = {
            "bg_threshold": 1.0 if conf < c.theta_low else 0.0,
            "beta_gw_boost": 0.3 if conf < c.theta_uncertain else 0.0,
            "replay_priority": epistemic + 0.5 * surprise,
            "trn_gain": 1.0 - conf * 0.5,
        }
        return MetaOut(confidence=conf, predicted_loss=l_hat,
                       surprise=surprise, novelty=nov,
                       epistemic=epistemic, aleatoric=aleatoric,
                       m_ratio=m_ratio, gates=gates)
