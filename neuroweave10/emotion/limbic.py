"""Лимбический Контур (Ziwago-Prime v4, Узел 3): Гипоталамус + Амигдала (BLA / CeA).

Реализует «горячую» гомеостатическую валентность и эмоциональную оценку опыта:
  1. Гипоталамус (Интероцептивный Драйв-Контроллер):
     Отслеживает 3 фундаментальных биологических дефицита (потребности):
       - D_energy  = max(0, θ_ATP - E_ATP)        — энергетический голод / усталость;
       - D_safety  = max(0, F(t) - F_comfort)     — потребность в порядке и безопасности;
       - D_novelty = max(0, θ_boredom - Surprise) — потребность в новизне (скука!).
  2. Комплекс Амигдалы (Двухпутевая схема Леду — LeDoux 1996):
     - Быстрый таламический путь («Low Road»: ThalamusFO → BLA, 1 мс, грубая оценка);
     - Точный кортикальный путь («High Road»: PFC/ASSOC → BLA, контекстная оценка + регуляция vmPFC);
     - Две конкурирующие спайковые популяции в BLA с одношаговой пластичностью (Flash-LTP):
         * A+ (Appetitive / Награда / Облегчение): активируется при снижении F(t) и утолении драйвов;
         * A- (Aversive / Угроза / Боль / Кризис): активируется при резком скачке F(t) или дефиците АТФ.
     - Центральное ядро (CeA) выдаёт:
         * Валентность V_val = r(A+) - r(A-) ∈ [-1, +1] (знак: к цели vs избегание);
         * Возбуждение S_arousal = r(A+) + r(A-) ∈ [0, 1] (сила эмоционального штампа памяти).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


@dataclass
class LimbicOut:
    valence: float          # V_val ∈ [-1, +1]: >0 награда/облегчение, <0 угроза/ошибка
    arousal: float          # S_arousal ∈ [0, 1]: интенсивность эмоции (Flash-LTP гейт)
    d_energy: float         # дефицит АТФ (усталость)
    d_safety: float         # дефицит предсказуемости (тревога/хаос)
    d_novelty: float        # дефицит новизны (скука → драйвер Active Inference!)
    appetitive: float       # активность популяции A+
    aversive: float         # активность популяции A-
    flash_ltp_gain: float   # множитель пластичности от эмоционального возбуждения


class LimbicSystem:
    """Гипоталамус + Базолатеральная/Центральная Амигдала (BLA/CeA)."""

    def __init__(self, input_dim: int, pfc_dim: int, n_bla: int = 16,
                 seed: int = 0):
        self.input_dim = input_dim
        self.pfc_dim = pfc_dim
        self.n_bla = n_bla
        g = torch.Generator().manual_seed(seed)
        # Быстрый путь (Low Road: Thalamus → BLA A+ / A-)
        self.w_low_pos = torch.rand(n_bla, input_dim, generator=g) * (0.25 / math.sqrt(max(1, input_dim)))
        self.w_low_neg = torch.rand(n_bla, input_dim, generator=g) * (0.25 / math.sqrt(max(1, input_dim)))
        # Медленный кортикальный путь (High Road: PFC → BLA + ингибиторный контроль vmPFC)
        self.w_high_pos = torch.rand(n_bla, pfc_dim, generator=g) * (0.25 / math.sqrt(max(1, pfc_dim)))
        self.w_vmpfc_inh = torch.rand(n_bla, pfc_dim, generator=g) * (0.30 / math.sqrt(max(1, pfc_dim)))

        self.r_pos = torch.zeros(n_bla)
        self.r_neg = torch.zeros(n_bla)
        self.fe_ema: float = 0.25
        self.surprise_ema: float = 0.25
        self.xt_prev: Optional[torch.Tensor] = None
        self.monotony_ema: float = 0.0
        self.last_out: Optional[LimbicOut] = None

    def reset_state(self) -> None:
        self.r_pos.zero_()
        self.r_neg.zero_()
        self.fe_ema = 0.25
        self.surprise_ema = 0.25
        self.xt_prev = None
        self.monotony_ema = 0.0
        self.last_out = None

    def step(self, x_thal: torch.Tensor, s_pfc: torch.Tensor,
             free_energy: float, intrinsic_rpe: float,
             atp_mean: float, ext_reward: float = 0.0,
             learn: bool = True) -> LimbicOut:
        """Один шаг лимбической оценки и эмоционального кондиционирования."""
        xt = x_thal.detach().float().mean(0) if x_thal.dim() == 2 else x_thal.detach().float()
        sp = s_pfc.detach().float().mean(0) if s_pfc.dim() == 2 else s_pfc.detach().float()
        if xt.numel() != self.input_dim:
            xt = torch.nn.functional.pad(xt, (0, max(0, self.input_dim - xt.numel())))[:self.input_dim]
        if sp.numel() != self.pfc_dim:
            sp = torch.nn.functional.pad(sp, (0, max(0, self.pfc_dim - sp.numel())))[:self.pfc_dim]

        # --- 1. Гипоталамус: вычисление 3 гомеостатических драйвов ---
        d_energy = max(0.0, min(1.0, (0.75 - float(atp_mean)) * 2.0))
        fe_jump = max(0.0, float(free_energy) - self.fe_ema - 0.15)
        self.fe_ema = 0.85 * self.fe_ema + 0.15 * float(free_energy)
        d_safety = max(0.0, min(1.0, fe_jump * 2.5 + max(0.0, float(free_energy) - 1.40)))

        # Сенсорное однообразие (монотонность): если входной кадр xt повторяется ("the the the"),
        # косинусное сходство с прошлым кадром ≈ 1.0 → скука D_novelty стремительно растёт!
        if self.xt_prev is not None and float(xt.norm()) > 1e-5 and float(self.xt_prev.norm()) > 1e-5:
            sim_prev = float(torch.nn.functional.cosine_similarity(
                xt.unsqueeze(0), self.xt_prev.unsqueeze(0)).item())
            is_repeat = max(0.0, (sim_prev - 0.85) / 0.15)
        else:
            is_repeat = 0.0
        self.xt_prev = xt.clone()
        self.monotony_ema = 0.65 * self.monotony_ema + 0.35 * is_repeat

        inst_surp = abs(float(intrinsic_rpe)) + fe_jump
        self.surprise_ema = 0.90 * self.surprise_ema + 0.10 * inst_surp
        low_surp_boredom = max(0.0, (0.15 - self.surprise_ema) * 5.0)
        d_novelty = max(0.0, min(1.0, (low_surp_boredom + 0.85 * self.monotony_ema) * (1.0 - d_energy)))

        # --- 2. Амигдала (BLA): быстрый таламический + кортикальный пути ---
        xt_norm = xt / xt.norm().clamp(min=1e-6)
        sp_norm = sp / sp.norm().clamp(min=1e-6)
        us_pos = max(0.0, float(ext_reward)) + max(0.0, float(intrinsic_rpe)) * 0.8
        us_neg = max(0.0, -float(ext_reward)) + d_safety * 0.7 + d_energy * 0.6

        drive_pos = 0.45 * (xt_norm @ self.w_low_pos.t()) + 0.35 * (sp_norm @ self.w_high_pos.t()) + us_pos
        # Быстрый таламический путь Леду (Low Road CS+): если стимул уже был кондиционирован
        # как угроза (w_low_neg > w_low_pos), Амигдала мгновенно зажигает A- даже до удара током!
        raw_pos = xt_norm @ self.w_low_pos.t()
        raw_neg = xt_norm @ self.w_low_neg.t()
        cs_threat = (raw_neg - raw_pos).clamp(min=0.0)
        # vmPFC + базовый тонус безопасности гасят ложную тревогу в отсутствие угрозы (us_neg и cs_threat)
        threat_present = min(1.0, us_neg * 2.0 + float(cs_threat.mean().item()) * 1.5)
        vmpfc_brake = (sp_norm @ self.w_vmpfc_inh.t()).clamp(min=0.0) + 0.30 * (1.0 - threat_present)
        drive_neg = (
            0.55 * raw_neg.clamp(min=0.0) * (0.45 + 0.55 * threat_present)
            + 0.75 * cs_threat
            + us_neg * 1.4
            - 0.45 * vmpfc_brake
        ).clamp(min=0.0)

        # Взаимное латеральное торможение между A+ и A- внутри BLA
        self.r_pos = (0.65 * self.r_pos + 0.35 * torch.tanh(drive_pos - 0.25 * self.r_neg.mean())).clamp(0.0, 1.0)
        self.r_neg = (0.65 * self.r_neg + 0.35 * torch.tanh(drive_neg - 0.25 * self.r_pos.mean())).clamp(0.0, 1.0)

        app = float(self.r_pos.mean().item())
        av = float(self.r_neg.mean().item())
        valence = max(-1.0, min(1.0, app - av))
        arousal = max(0.0, min(1.0, app + av))

        # --- 3. One-Shot эмоциональное кондиционирование (Flash-LTP в BLA с оппонентным балансом) ---
        if learn and (us_pos > 0.05 or us_neg > 0.05):
            eta_emo = 0.14 * (1.0 + arousal)
            if us_pos > 0.05:
                r_p = self.r_pos.clamp(min=0.35)
                self.w_low_pos = (self.w_low_pos + eta_emo * us_pos * torch.outer(r_p, xt_norm)).clamp(-1.5, 1.5)
                self.w_high_pos = (self.w_high_pos + eta_emo * us_pos * torch.outer(r_p, sp_norm)).clamp(-1.5, 1.5)
                self.w_low_neg = (self.w_low_neg - 0.50 * eta_emo * us_pos * torch.outer(r_p, xt_norm)).clamp(-1.5, 1.5)
            if us_neg > 0.05:
                r_n = self.r_neg.clamp(min=0.35)
                self.w_low_neg = (self.w_low_neg + eta_emo * us_neg * torch.outer(r_n, xt_norm)).clamp(-1.5, 1.5)
                self.w_low_pos = (self.w_low_pos - 0.60 * eta_emo * us_neg * torch.outer(r_n, xt_norm)).clamp(-1.5, 1.5)

        flash_ltp_gain = 1.0 + 1.5 * arousal
        out = LimbicOut(
            valence=valence,
            arousal=arousal,
            d_energy=d_energy,
            d_safety=d_safety,
            d_novelty=d_novelty,
            appetitive=app,
            aversive=av,
            flash_ltp_gain=flash_ltp_gain,
        )
        self.last_out = out
        return out
