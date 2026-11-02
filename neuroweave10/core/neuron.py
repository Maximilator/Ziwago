"""Мультикомпартментный пирамидальный нейрон Urbanczik-Senn (§1.2).

Биологическое ядро V10: сома + базальный дендрит (AMPA, feedforward)
+ апикальный дендрит (NMDA, feedback) — три связанных компартмента:

  C_m dV_S/dt  = -g_L(V_S-E_L) + g_D(V_D-V_S) + g_A(V_A-V_S)
                 - I_fast - I_metabolic
  C_m dV_D/dt  = -g_L(V_D-E_L) + g_D(V_S-V_D) + g_D AMPA(E_exc-V_D)
  C_m dV_A/dt  = -g_L(V_A-E_L) + g_A(V_S-V_A) + g_A NMDA·B(V_A)·(E_exc-V_A)

  I_fast       = g_GABA·(1+η_ALLO[ALLO])·(V_S-E_Cl)   — быстрое торможение
  I_metabolic  = k_ATP·(1-E_i)·(V_S-E_K) + k_GIRK·[Ado]_i·(V_S-E_K)
  B(V)         = 1/(1 + [Mg2+]/3.57·exp(-0.062·V))    — Jahr-Stevens 1990
  s_i(t)       = H(V_S - Θ);  Θ адаптируется как ALIF (Bellec 2018)
  φ(V)         = φ_max/(1+exp(-β(V-Θ)))               — S-функция частоты
  Ca2+         = экспоненциальный след спайков → 2-AG, NO, BDNF-пути

Два сигнала на нейрон: «обещание» дендрита φ(V*_D) для обучения и
истинный спайк S_i. Ошибка S_i-φ(V*_D,i) — внутренняя метрика зоны
(§1.3: обучение без обратного распространения).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import USNeuronCfg


def mg_block(v: torch.Tensor, mg: float = 1.0) -> torch.Tensor:
    """Jahr-Stevens: B(V) = 1/(1 + [Mg2+]/3.57·exp(-0.062V))."""
    return 1.0 / (1.0 + (mg / 3.57) * torch.exp(-0.062 * v))


@dataclass
class NeuronOut:
    s: torch.Tensor          # [B,N] спайки {0,1}
    v_s: torch.Tensor
    v_d: torch.Tensor        # базальный дендрит
    v_a: torch.Tensor        # апикальный дендрит
    rate_d: torch.Tensor     # φ(V*_D) — предсказание дендрита
    error: torch.Tensor      # S_i - φ(V*_D) — сигнал внутренней ошибки
    energy: torch.Tensor     # [B,N] ATP
    ca: torch.Tensor         # [B,N] постсинаптический Ca2+


class USNeuronPool:
    """Пул мультикомпартментных нейронов. Состояние — плоские тензоры."""

    def __init__(self, n: int, cfg: USNeuronCfg, name: str = "pyr"):
        self.n, self.cfg, self.name = n, cfg, name
        self.register_buffer = None  # не nn.Module — лёгкие пулы

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        c = self.cfg
        f = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        self.v_s = f(self.n).fill_(c.e_l)
        self.v_d = f(self.n).fill_(c.e_l)
        self.v_a = f(self.n).fill_(c.e_l)
        self.theta = f(self.n).fill_(c.theta)   # адаптивный порог Θ_i
        self.refr = f(self.n)                   # такты рефрактерности
        self.ca = f(self.n)                     # Ca2+-след
        self.energy = f(self.n).fill_(1.0)      # ATP E_i ∈ [0,1]
        self.ado_i = f(self.n)                  # локальный аденозин A1
        self.allo = f(self.n)                   # нейростероид ALLO
        self.noise = f(self.n)
        # Мускариновый M-ток (Genesis level2/neuron.py §2.30):
        # активация m_gate сравнивается с медленной базовой линией m_baseline
        self.m_gate = f(self.n).fill_(1.0)
        self.m_baseline = f(self.n).fill_(1.0)
        # Интринсик-пластичность Триша (Triesch 2005, Уровень 5):
        # индивидуальная медленная частота r_ip, собственный гейн ip_gain и базовый порог theta_base
        self.r_ip = f(self.n).fill_(0.05)
        self.ip_gain = f(self.n).fill_(1.0)
        self.theta_base = f(self.n).fill_(c.theta)
        self.met_gain = 1.0     # гомеостаз критичности → метаболический тормоз
        # Последние синаптические токи (для расчёта ламинарного LFP и эфаптики, Уровень 6)
        self.last_i_ff = f(self.n)
        self.last_i_fb = f(self.n)
        self.last_i_inh = f(self.n)
        self._spikes_hist: list = []

        # --- Lab.html F#20, F#16, F#17, F#22: eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch ---
        # При включении (enable_elif_metabolic_gate=True) нейроны с АТФ < theta_down_atp (или глубоко
        # ингибированные без афферентного входа) переходят в аппаратный энергосберегающий режим
        # 'down_phase_mask' (low-power idle), пропускающий подпороговую интеграцию и генерацию спайков,
        # и мгновенно (<20 мс) пробуждаются при восстановлении АТФ >= theta_up_atp или сильном cue!
        self.enable_elif_metabolic_gate: bool = False
        self.theta_down_atp: float = 0.58
        self.theta_up_atp: float = 0.72
        self.wake_cue_threshold: float = 0.25
        self.down_phase_mask = torch.zeros(batch, self.n, device=device, dtype=torch.bool)
        self.last_down_phase_fraction: float = 0.0
        self.cum_gated_energy_saved: float = 0.0
        self.cum_total_ticks: int = 0

    # ------------------------------------------------------------------
    def phi(self, v: torch.Tensor, theta: Optional[torch.Tensor] = None,
            ne: float = 0.0) -> torch.Tensor:
        """φ(V) = φ_max·σ(β_eff·a_i·(V-Θ)) — частота, Гц.
        Норадреналин (NE) повышает крутизну передаточной функции (Уровень 2)."""
        th = self.theta if theta is None else theta
        gain_i = getattr(self, "ip_gain", 1.0)
        beta_eff = self.cfg.beta * (1.0 + 0.6 * max(0.0, float(ne))) * gain_i
        return self.cfg.phi_max * torch.sigmoid(beta_eff * (v - th))

    def v_star_d(self) -> torch.Tensor:
        """V*_D = (g_L·E_L + g_D·V_D)/(g_L+g_D) — потенциал дендрита без спайка.

        Это «обещание» базального дендрита; используется трёхфакторным
        правилом как предсказание сомы (Urbanczik & Senn 2014)."""
        c = self.cfg
        return (c.g_l * c.e_l + c.g_d * self.v_d) / (c.g_l + c.g_d)

    def v_star_da(self) -> torch.Tensor:
        """V*_DA = V*_D + γ_A·(V_A - E_L) — интегрированный сомато-дендритный
        потенциал с учётом апикального учителя (Sacramento et al. 2018; Mikulasch et al. 2023).
        В отсутствие апикального входа (V_A = E_L) строго равен V*_D."""
        c = self.cfg
        return self.v_star_d() + 0.25 * (self.v_a - c.e_l)

    def step(self,
             g_ampa: torch.Tensor,      # [B,N] открытая проводимость AMPA
             g_nmda: torch.Tensor,      # [B,N] проводимость NMDA (апикал)
             g_gaba: torch.Tensor,      # [B,N] проводимость GABA
             i_ext: Optional[torch.Tensor] = None,
             ach: float = 0.0,          # ACh: M-ток K_v7↓ → Θ↓ (encode-режим)
             ne: float = 0.0,           # NE: α2-AR подавление шума + ↑гейн (SNR)
             histamine: float = 0.0,    # H1: sAHP-блок → берстовая активность
             nmda_gain: float = 1.0,    # Φ_DSer·Ψ_Zn·DHEA ко-агонистный гейт
             p_rel: float = 1.0,        # P_release-масштаб (NPY/Dyn/AEA/Gal)
             vth_shift: float = 0.0,    # E2-мембранный сдвиг Θ (мВ)
             gaba_tonic: float = 1.0,   # ALLO: GABA_A PAM
             g_gaba_d: Optional[torch.Tensor] = None,  # SST→апикал шунт
             ephaptic_mv: Optional[torch.Tensor] = None,  # LFP эфаптический сдвиг (мВ)
             rate_ceiling: float = 1.0, # CO→BK_Ca: калиевый тормоз частоты
             ahp_block: Optional[float] = None,  # H1-гистамин блок sAHP
             ) -> NeuronOut:
        """Один шаг интеграции (dt=1 мс), все три компартмента через exact-step."""
        c = self.cfg
        dt = c.dt
        B = self.v_s

        # --- 1. Базальный дендрит: EXACT-STEP с разделением быстрой AMPA-атаки и медленного распада Murray/Hasson (P3) ---
        g_ff_eff = g_ampa.clamp(min=0.0) * p_rel
        g_d_tot = c.g_l + c.g_d + g_ff_eff
        v_d_inf = (c.g_l * c.e_l + c.g_d * B + g_ff_eff * c.e_exc) / g_d_tot.clamp(min=1e-6)
        # P3 (Murray et al. 2014; Hasson et al. 2015): захват входного спайка (v_d_inf > v_d)
        # идёт со скоростью открытия AMPA-каналов τ_rise = min(20 мс, τ_d), а удержание
        # контекста (v_d_inf <= v_d) затухает с собственной константой зоны τ_m(g) (до 680+ мс в PFC)!
        tau_rise = min(20.0, float(c.tau_d))
        tau_base = torch.where(v_d_inf > self.v_d,
                               torch.full_like(self.v_d, tau_rise),
                               torch.full_like(self.v_d, float(c.tau_d)))
        tau_d_eff = (c.c_m * tau_base) / g_d_tot.clamp(min=1e-6)
        self.v_d = (v_d_inf + (self.v_d - v_d_inf) * torch.exp(-dt / tau_d_eff)).clamp(-100.0, 40.0)

        # --- 2. Апикальный дендрит: EXACT-STEP (20% AMPA + 80% NMDA Jahr-Stevens + SST шунт) ---
        b_v = mg_block(self.v_a, c.mg)
        eff_nmda = 0.20 + 0.80 * nmda_gain * b_v
        g_fb_eff = g_nmda.clamp(min=0.0) * eff_nmda
        g_sst = torch.zeros_like(B) if g_gaba_d is None else g_gaba_d.clamp(min=0.0) * gaba_tonic
        g_a_tot = c.g_l + c.g_a + g_fb_eff + g_sst
        v_a_inf = (c.g_l * c.e_l + c.g_a * B + g_fb_eff * c.e_exc
                   + g_sst * c.e_cl) / g_a_tot.clamp(min=1e-6)
        tau_a_eff = (c.c_m * c.tau_d_apical) / g_a_tot.clamp(min=1e-6)
        self.v_a = (v_a_inf + (self.v_a - v_a_inf) * torch.exp(-dt / tau_a_eff)).clamp(-100.0, 40.0)

        # Сохраняем токи для расчёта внеклеточного диполя LFP (Уровень 6)
        self.last_i_ff = g_ff_eff * (c.e_exc - self.v_d)
        self.last_i_fb = g_fb_eff * (c.e_exc - self.v_a)

        # --- 3. Сома: проводимости (не токи — Закон 3, шунт-физика) ---
        g_inh = g_gaba.clamp(min=0.0) * gaba_tonic * (1.0 + c.allo_gain * self.allo)
        self.last_i_inh = (g_inh + g_sst) * (self.v_s - c.e_cl).clamp(min=0.0)
        # Метаболический тормоз: KATP (ATP↓) + GIRK (Ado A1) → V_S к E_K.
        g_met = self.met_gain * (c.k_atp * (1.0 - self.energy) ** 2
                                 + c.k_girk * self.ado_i)
        # Мускариновый M-ток (Genesis §2.30): тормозит отклонение от медленной
        # базовой линии, подавляется ACh при фокусе внимания.
        m_inf = torch.sigmoid((self.v_s - (-45.0)) / 8.0)
        self.m_gate = self.m_gate + (m_inf - self.m_gate) * min(dt / 80.0, 1.0)
        self.m_baseline = self.m_baseline + (
            self.m_gate - self.m_baseline) * min(dt / 400.0, 1.0)
        m_release = 1.0 / (1.0 + ach / 0.5)
        g_m = 0.08 * m_release * (self.m_gate - self.m_baseline).clamp(min=0.0)
        h1_blk = (histamine / (histamine + 0.4)) if ahp_block is None else float(ahp_block)
        # CO (оксид углерода) открывает кальций-зависимые BK_Ca каналы (rate_ceiling),
        # защищая пирамиду от высокочастотного перегрева
        bk_gain = max(1.0, float(rate_ceiling))
        g_ahp = 0.08 * self.ca * (1.0 - h1_blk) * bk_gain
        g_gpcr = g_m + g_ahp
        ext = 0.0 if i_ext is None else i_ext
        if c.stochastic:
            # Уровень 2: Норадреналин (NE) через α2-рецепторы подавляет
            # коррелированный фоновый OU-шум, повышая отношение Сигнал/Шум (SNR)!
            ne_snr_damp = 1.0 / (1.0 + 1.5 * max(0.0, float(ne)))
            sigma_eff = c.sigma * ne_snr_damp
            self.noise = self.noise * (1.0 - dt / c.tau_ou) + \
                torch.randn_like(B) * (sigma_eff * (dt / c.tau_ou) ** 0.5)
        else:
            self.noise = torch.zeros_like(B)

        # --- сома: EXACT-STEP по проводимостям + эфаптическая связь LFP ---
        g_inh_tot = g_inh + g_met + g_gpcr  # все тормозные проводимости
        g_exc_tot = c.g_d + c.g_a           # связи с дендритами (к их V)
        g_tot = c.g_l + g_exc_tot + g_inh_tot
        eph_cur = 0.0 if ephaptic_mv is None else ephaptic_mv * c.g_l
        v_inf = (c.g_l * c.e_l + c.g_d * self.v_d + c.g_a * self.v_a
                 + g_inh * c.e_cl + (g_met + g_gpcr) * c.e_k
                 + ext + eph_cur + self.noise) / g_tot.clamp(min=1e-6)
        tau_eff = c.c_m / g_tot.clamp(min=1e-6)
        self.v_s = v_inf + (B - v_inf) * torch.exp(-dt / tau_eff)

        # --- спайк: порог Θ_i + апикальное плато (Spiker L5 AND-gate) ---
        apical_plateau = 4.0 * torch.tanh(
            (self.v_a - (-60.0)).clamp(min=0.0) / 15.0)
        theta_eff = self.theta - vth_shift - 1.5 * ach - apical_plateau \
            + c.ado_theta_gain * self.ado_i \
            - c.hist_theta_gain * histamine   # H1: бодрость ↓Θ
        s = (self.v_s > theta_eff).float() * (self.refr <= 0).float()

        # --- Lab.html F#20, F#16, F#17, F#22: eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch ---
        if getattr(self, "enable_elif_metabolic_gate", False):
            if not hasattr(self, "down_phase_mask") or self.down_phase_mask.shape != B.shape:
                self.down_phase_mask = torch.zeros_like(B, dtype=torch.bool)
            wake_cue = (g_ampa.abs() + g_nmda.abs()) >= self.wake_cue_threshold
            # Вход в даун-фазу: низкий АТФ (< theta_down_atp) ИЛИ глубокая ингибиция при отсутствии входа
            enter_down = (self.energy < self.theta_down_atp) | (
                (g_gaba > g_ampa + 0.15) & (~wake_cue) & (self.v_s < c.e_l - 1.5)
            )
            # Выход из даун-фазы (гистерезисная защёлка F#16 + реактивация <20 мс F#22):
            exit_down = (self.energy >= self.theta_up_atp) & wake_cue
            self.down_phase_mask = (self.down_phase_mask | enter_down) & (~exit_down)
            # Клетки в down-phase пропускают генерацию спайков и фиксируют мембрану в покое (low-power idle),
            # ускоренно восстанавливая АТФ в 3 раза быстрее бодрствующих нейронов!
            down_f = self.down_phase_mask.float()
            s = s * (1.0 - down_f)
            self.v_s = torch.where(self.down_phase_mask, B, self.v_s)
            self.energy = (self.energy + down_f * 0.008 * (1.0 - self.energy)).clamp(0.0, 1.0)
            self.last_down_phase_fraction = float(down_f.mean().item())
            self.cum_gated_energy_saved += self.last_down_phase_fraction
            self.cum_total_ticks += 1

        self.v_s = self.v_s - s * (self.theta - c.e_l) * 0.5
        self.refr = (self.refr - 1.0).clamp(min=0.0) + s * c.refractory

        # --- Уровень 5 + Инновация 7: Информационная Интринсик-Пластичность Триша (Triesch 2005) ---
        # Максимизирует взаимную информацию при фиксированном бюджете АТФ (экспоненциальное
        # распределение частот со средним μ = rho_tgt), пробуждая молчащие клетки, укрощая
        # гиперактивные и сохраняя редкие высокоселективные концептуальные нейроны!
        if not hasattr(self, "r_ip") or self.r_ip.shape != s.shape:
            self.r_ip = torch.full_like(s, 0.05)
            self.ip_gain = torch.ones_like(s)
            self.theta_base = torch.full_like(s, c.theta)
        alpha_ip = min(dt / 400.0, 1.0)
        self.r_ip = (1.0 - alpha_ip) * self.r_ip + alpha_ip * s
        rho_tgt = 0.05
        rate_err = self.r_ip - rho_tgt
        # Градиент энтропии Триша: ΔΘ_i ∝ (r_i - μ) + 0.25·μ·((2 + 1/μ)r_i - r_i²/μ - 1)
        triesch_grad = 0.75 * rate_err + 0.25 * rho_tgt * (
            (2.0 + 1.0 / rho_tgt) * self.r_ip - (self.r_ip ** 2) / rho_tgt - 1.0
        ).clamp(-0.2, 0.2)
        self.theta_base = (self.theta_base + 0.015 * triesch_grad).clamp(c.theta - 8.0, c.theta + 8.0)
        # Правило Триша для собственного гейна мембраны a_i:
        v_norm = ((self.v_s - c.e_l) / max(1.0, abs(c.theta - c.e_l))).clamp(0.0, 2.0)
        da_ip = 0.002 * (1.0 / self.ip_gain.clamp(min=0.4) - v_norm * (rate_err / rho_tgt))
        self.ip_gain = (self.ip_gain + da_ip).clamp(0.5, 2.0)

        # ALIF-адаптация порога с возвратом к индивидуальному порогу Триша theta_base!
        self.theta = self.theta + c.beta_adapt * s \
            - dt * (self.theta - self.theta_base) / c.tau_adapt

        # Ca2+-след спайка (постсинаптический — драйвер 2-AG/NO/BDNF)
        self.ca = self.ca * (1.0 - dt / c.tau_ca) + s

        # --- Уровень 4 + P1: Метаболизм АТФ + Сомато-дендритная и Апикальная ошибка ---
        # Спайк тратит АТФ, а в покое глиальный лактатный шаттл медленно восстанавливает E_i
        self.energy = (self.energy - s * 0.004 + (1.0 - s) * 0.0004 * (1.0 - self.energy)
                       ).clamp(min=0.0, max=1.0)
        rate_d = self.phi(self.v_star_d(), ne=ne)
        rate_da = self.phi(self.v_star_da(), ne=ne)
        # Апикальный учитель (Sacramento et al. 2018; Mikulasch et al. 2023):
        # расхождение между потенциалом с учётом апикального входа V*_DA и чисто базальным V*_D
        # передаёт непрерывный градиент целевого предсказания сверху вниз даже без спайка!
        delta_apical = (rate_da - rate_d) / c.phi_max
        self.apical_error = delta_apical
        # Метаболическая цена спайка: при дефиците АТФ (1 - E_i) избыточные спайки
        # получают отрицательную поправку в локальной ошибке, обучая сеть экономить спайки!
        spike_cost = 0.25 * (1.0 - self.energy) * s
        err_local = (s - rate_d / c.phi_max) + 0.45 * delta_apical - spike_cost
        return NeuronOut(s=s, v_s=self.v_s, v_d=self.v_d, v_a=self.v_a,
                         rate_d=rate_d, error=err_local,
                         energy=self.energy, ca=self.ca)
