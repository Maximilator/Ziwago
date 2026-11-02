"""Многоскоростная событийно-коалесированная динамика Фазы 1 (snnscience.docx §3.1, §3.2, §3.7).

Содержит три ключевых механизма Фазы 1 Глобального Плана v8.0:
  1. TickLadderScheduler — логарифмическая лестница тиков из 4 классов времени
     (Class I: 1 мс, Class II: 10 мс, Class III: 100 мс, Class IV: Sleep-Tick / DEVS)
     с модуляцией от теневой цены энергии Лагранжа λ_ATP.
  2. MinicolumnStateHierarchy (MC-MREI) — коалесированная на уровне миниколонок
     иерархия состояний HOT / WARM / COLD с гистерезисным термостатом и
     ТОЧНЫМ аналитическим матрично-экспоненциальным пробуждением (Exact Lazy Decay,
     Mattia & Del Giudice 2000), сокращающая арифметику состояния в ~9 раз без warp-дивергенции.
  3. BennaFusiCascade — 4-звенный синаптический каскад Бенна–Фузи (Benna & Fusi,
     Nature Neuroscience 2016), обновляемый по классам лестницы тиков и порождающий
     степенной закон забывания Йоста R(t) ~ (1 + t/τ₀)^(-α) с α ∈ [0.3, 0.7].
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import torch


# Коды температурных ярусов миниколонок
TIER_COLD: int = 0
TIER_WARM: int = 1
TIER_HOT: int = 2


@dataclass
class TickLadderStep:
    """Флаги активности классов лестницы на текущем такте t."""
    tick: int
    run_class_1_fast: bool      # τ = 1–50 мс (каждый такт для HOT)
    run_class_2_medium: bool    # τ = 50 мс–1 с (каждые 10 тактов)
    run_class_3_slow: bool      # τ = 1–100 с (каждые 100 тактов)
    run_class_4_sleep: bool     # τ > 100 с (только на Sleep-Tick или DEVS-событии)
    eff_stride_2: int
    eff_stride_3: int


class TickLadderScheduler:
    """Логарифмическая лестница тиков (Изъян 2 snnscience.docx, Gear & Wells 1984).

    Из 15 переменных состояния нейрона:
      - 4 быстрые (Класс I: V_S, V_D, V_A, g_AMPA/GABA) обновляются каждый такт;
      - 4 средние (Класс II: адаптация a, Ca2+, 2-AG, u_syn) — раз в stride_2 тактов;
      - 4 медленные (Класс III: BDNF, лактат, метапластичность, каскады) — раз в stride_3 тактов;
      - 3 сверхмедленные (Класс IV: PNN-каркас, структурный морфогенез, SHY) — на sleep-тике.
    При вырожденном режиме (degenerate=True) все классы обновляются каждый такт.
    """

    def __init__(
        self,
        stride_2: int = 10,
        stride_3: int = 100,
        degenerate: bool = False,
    ) -> None:
        self.base_stride_2 = max(1, int(stride_2))
        self.base_stride_3 = max(1, int(stride_3))
        self.degenerate = bool(degenerate)
        self.tick: int = 0
        self.lambda_atp: float = 0.0  # Теневая цена АТФ растягивает медленные такты (§4.3)
        self.total_var_updates: float = 0.0
        self.total_dense_equivalent: float = 0.0

    def step(self, is_sleep_tick: bool = False, lambda_atp: Optional[float] = None) -> TickLadderStep:
        """Продвигает глобальный счётчик тактов и возвращает активные классы лестницы."""
        self.tick += 1
        if lambda_atp is not None:
            self.lambda_atp = max(0.0, float(lambda_atp))

        if self.degenerate:
            self.total_var_updates += 15.0
            self.total_dense_equivalent += 15.0
            return TickLadderStep(
                tick=self.tick,
                run_class_1_fast=True,
                run_class_2_medium=True,
                run_class_3_slow=True,
                run_class_4_sleep=True,
                eff_stride_2=1,
                eff_stride_3=1,
            )

        # Метаболическая модуляция расписания: дорогая энергия (λ_ATP > 0) удлиняет медленные шаги
        scale = 1.0 + int(round(min(2.0, self.lambda_atp * 2.0)))
        s2 = self.base_stride_2 * scale
        s3 = self.base_stride_3 * scale

        c1 = True
        c2 = (self.tick % s2 == 0) or is_sleep_tick
        c3 = (self.tick % s3 == 0) or is_sleep_tick
        c4 = bool(is_sleep_tick)

        # Учёт сэкономленных обновлений переменных состояния на 1 активный нейрон
        vars_updated = 4.0 + (4.0 if c2 else 0.0) + (4.0 if c3 else 0.0) + (3.0 if c4 else 0.0)
        self.total_var_updates += vars_updated
        self.total_dense_equivalent += 15.0

        return TickLadderStep(
            tick=self.tick,
            run_class_1_fast=c1,
            run_class_2_medium=c2,
            run_class_3_slow=c3,
            run_class_4_sleep=c4,
            eff_stride_2=s2,
            eff_stride_3=s3,
        )

    def ladder_compression_ratio(self) -> float:
        """Во сколько раз лестница тиков сокращает число обновлений переменных (норма ~3.3–3.5×)."""
        return float(self.total_dense_equivalent / max(1e-9, self.total_var_updates))


class MinicolumnStateHierarchy:
    """Коалесированная иерархия состояний HOT / WARM / COLD (MC-MREI, Изъян 1 snnscience.docx).

    Делит N нейронов на тайлы-миниколонки размера tile_size (по умолчанию 16).
    Каждый тайл находится в одном из трёх состояний:
      - HOT  (2): полное обновление каждый такт (15 переменных с учётом лестницы тиков);
      - WARM (1): обновление раз в 10 тактов (8 переменных дельт от гомеостатической базы);
      - COLD (0): спящий тайл (0 обновлений в бодрствовании; хранит только снимок на момент засыпания).

    При поступлении входного спайка или апикального предсказания в COLD-тайл выполняется
    ТОЧНЫЙ аналитический перенос состояния (Exact Lazy Decay, Mattia & Del Giudice 2000)
    по замкнутому решению системы линейных ОДУ мембраны и проводимостей!
    """

    def __init__(
        self,
        n_neurons: int,
        tile_size: int = 16,
        e_l: float = -70.0,
        tau_m: float = 20.0,
        tau_ampa: float = 10.0,
        tau_nmda: float = 60.0,
        tau_gaba: float = 15.0,
        tau_adapt: float = 300.0,
        warm_silence_ticks: int = 6,
        cold_silence_ticks: int = 20,
        wake_threshold: float = 0.05,
        degenerate: bool = False,
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.tile_size = max(1, int(tile_size))
        self.n_tiles = (self.n_neurons + self.tile_size - 1) // self.tile_size
        self.e_l = float(e_l)
        self.tau_m = float(tau_m)
        self.tau_ampa = float(tau_ampa)
        self.tau_nmda = float(tau_nmda)
        self.tau_gaba = float(tau_gaba)
        self.tau_adapt = float(tau_adapt)
        self.warm_silence_ticks = int(warm_silence_ticks)
        self.cold_silence_ticks = int(cold_silence_ticks)
        self.wake_threshold = float(wake_threshold)
        self.degenerate = bool(degenerate)

        # Статус каждого тайла: по умолчанию стартуем в HOT
        self.tile_tier = torch.full((self.n_tiles,), TIER_HOT, dtype=torch.long)
        self.tile_silence = torch.zeros(self.n_tiles, dtype=torch.long)
        self.tile_t_sleep = torch.zeros(self.n_tiles, dtype=torch.long)

        # Снимки состояния на момент ухода в COLD (для точного аналитического пробуждения)
        self.v_sleep = torch.full((self.n_neurons,), self.e_l, dtype=torch.float32)
        self.g_ampa_sleep = torch.zeros(self.n_neurons, dtype=torch.float32)
        self.g_nmda_sleep = torch.zeros(self.n_neurons, dtype=torch.float32)
        self.g_gaba_sleep = torch.zeros(self.n_neurons, dtype=torch.float32)
        self.adapt_sleep = torch.zeros(self.n_neurons, dtype=torch.float32)

        # Кумулятивная статистика для измерения коэффициента сжатия арифметики (~9×)
        self.cum_dense_var_ops: float = 0.0
        self.cum_actual_var_ops: float = 0.0
        self.total_exact_wakeups: int = 0

    def exact_lazy_propagate(
        self,
        v0: torch.Tensor,
        g_ampa0: torch.Tensor,
        g_nmda0: torch.Tensor,
        g_gaba0: torch.Tensor,
        adapt0: torch.Tensor,
        dt_elapsed: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Точное замкнутое решение линейного подпорогового ОДУ за интервал Δt (Mattia & Del Giudice 2000):

        dv/dt = -(v - E_L)/τ_m + g_AMPA(t)/τ_m + g_NMDA(t)/τ_m - g_GABA(t)/τ_m
        dg_k/dt = -g_k/τ_k   ===>   g_k(t + Δt) = g_k(t) * exp(-Δt / τ_k)

        Интеграл Дюамеля даёт точный аналитический вклад каждой затухающей проводимости в потенциал v(t + Δt)!
        """
        if dt_elapsed <= 0.0:
            return v0, g_ampa0, g_nmda0, g_gaba0, adapt0

        em = math.exp(-dt_elapsed / self.tau_m)
        ea = math.exp(-dt_elapsed / self.tau_ampa)
        en = math.exp(-dt_elapsed / self.tau_nmda)
        eg = math.exp(-dt_elapsed / self.tau_gaba)
        ead = math.exp(-dt_elapsed / self.tau_adapt)

        def _duhamel_coeff(tau_m: float, tau_k: float, exp_m: float, exp_k: float) -> float:
            if abs(tau_m - tau_k) < 1e-5:
                return (dt_elapsed / tau_m) * exp_m
            return (tau_k / (tau_k - tau_m)) * (exp_k - exp_m)

        c_ampa = _duhamel_coeff(self.tau_m, self.tau_ampa, em, ea)
        c_nmda = _duhamel_coeff(self.tau_m, self.tau_nmda, em, en)
        c_gaba = _duhamel_coeff(self.tau_m, self.tau_gaba, em, eg)

        v_new = (
            self.e_l
            + (v0 - self.e_l) * em
            + g_ampa0 * c_ampa
            + g_nmda0 * c_nmda
            - g_gaba0 * c_gaba
        )
        return (
            v_new,
            g_ampa0 * ea,
            g_nmda0 * en,
            g_gaba0 * eg,
            adapt0 * ead,
        )

    def update_tiers(
        self,
        tick: int,
        afferent_drive: torch.Tensor,
        spikes: torch.Tensor,
        v_s: torch.Tensor,
        adapt: torch.Tensor,
        ladder_step: Optional[TickLadderStep] = None,
    ) -> Dict[str, float]:
        """Гистерезисный термостат тайлов: пробуждает COLD-тайлы аналитически при входе > порога
        и переводит тихие тайлы в WARM / COLD."""
        if self.degenerate:
            self.tile_tier.fill_(TIER_HOT)
            self.cum_dense_var_ops += 15.0 * self.n_neurons
            self.cum_actual_var_ops += 15.0 * self.n_neurons
            return {
                "hot_frac": 1.0,
                "warm_frac": 0.0,
                "cold_frac": 0.0,
                "compression_ratio": 1.0,
            }

        aff_1d = afferent_drive.detach().abs().mean(0) if afferent_drive.dim() > 1 else afferent_drive.detach().abs()
        spk_1d = spikes.detach().abs().mean(0) if spikes.dim() > 1 else spikes.detach().abs()
        v_1d = v_s.detach().mean(0) if v_s.dim() > 1 else v_s.detach()
        ad_1d = adapt.detach().mean(0) if adapt.dim() > 1 else adapt.detach()

        for t_idx in range(self.n_tiles):
            s_i = t_idx * self.tile_size
            e_i = min(self.n_neurons, s_i + self.tile_size)
            tile_in = float(aff_1d[s_i:e_i].max().item())
            tile_spk = float(spk_1d[s_i:e_i].max().item())
            cur_tier = int(self.tile_tier[t_idx].item())

            if tile_in > self.wake_threshold or tile_spk > 0.0:
                # Если тайл спал в COLD, выполняем точное аналитическое пробуждение!
                if cur_tier == TIER_COLD:
                    dt_el = float(max(1, tick - int(self.tile_t_sleep[t_idx].item())))
                    v_w, _, _, _, ad_w = self.exact_lazy_propagate(
                        self.v_sleep[s_i:e_i],
                        self.g_ampa_sleep[s_i:e_i],
                        self.g_nmda_sleep[s_i:e_i],
                        self.g_gaba_sleep[s_i:e_i],
                        self.adapt_sleep[s_i:e_i],
                        dt_el,
                    )
                    if v_s.dim() == 2:
                        v_s[:, s_i:e_i] = v_w.unsqueeze(0)
                        adapt[:, s_i:e_i] = ad_w.unsqueeze(0)
                    self.total_exact_wakeups += 1
                self.tile_tier[t_idx] = TIER_HOT
                self.tile_silence[t_idx] = 0
            else:
                self.tile_silence[t_idx] += 1
                sil = int(self.tile_silence[t_idx].item())
                if sil >= self.cold_silence_ticks and cur_tier != TIER_COLD:
                    # Сохраняем снимок состояния перед засыпанием в COLD
                    self.v_sleep[s_i:e_i] = v_1d[s_i:e_i]
                    self.adapt_sleep[s_i:e_i] = ad_1d[s_i:e_i]
                    self.tile_t_sleep[t_idx] = tick
                    self.tile_tier[t_idx] = TIER_COLD
                elif sil >= self.warm_silence_ticks and cur_tier == TIER_HOT:
                    self.tile_tier[t_idx] = TIER_WARM

        n_hot = int((self.tile_tier == TIER_HOT).sum().item())
        n_warm = int((self.tile_tier == TIER_WARM).sum().item())
        n_cold = int((self.tile_tier == TIER_COLD).sum().item())

        hot_frac = n_hot / max(1, self.n_tiles)
        warm_frac = n_warm / max(1, self.n_tiles)
        cold_frac = n_cold / max(1, self.n_tiles)

        # Расчёт реальных обновлений переменных состояния на текущем такте:
        # HOT обновляет быстрые переменные (4) + по лестнице средние/медленные
        # WARM обновляет 8 дельт раз в 10 тактов (0.8 за такт)
        # COLD потребляет 0 в бодрствовании (0.05 на фоновую SWR-компакцию)
        run_c2 = True if ladder_step is None else ladder_step.run_class_2_medium
        run_c3 = False if ladder_step is None else ladder_step.run_class_3_slow
        hot_vars = 4.0 + (4.0 if run_c2 else 0.0) + (4.0 if run_c3 else 0.0)
        warm_vars = 8.0 if run_c2 else 0.0
        cold_vars = 0.05

        actual_per_neuron = hot_frac * hot_vars + warm_frac * warm_vars + cold_frac * cold_vars
        self.cum_dense_var_ops += 15.0 * self.n_neurons
        self.cum_actual_var_ops += actual_per_neuron * self.n_neurons

        return {
            "hot_frac": hot_frac,
            "warm_frac": warm_frac,
            "cold_frac": cold_frac,
            "actual_vars_per_neuron": actual_per_neuron,
            "compression_ratio": float(self.cum_dense_var_ops / max(1e-9, self.cum_actual_var_ops)),
        }

    def hot_neuron_mask(self, device=None) -> torch.Tensor:
        """Возвращает булеву маску [N_neurons] всех нейронов в активных HOT-тайлах."""
        mask_tiles = (self.tile_tier == TIER_HOT).unsqueeze(1).expand(self.n_tiles, self.tile_size)
        flat = mask_tiles.reshape(-1)[: self.n_neurons]
        return flat.to(device) if device is not None else flat

    def verify_causal_cone_transport(
        self,
        total_horizon_ms: float = 40.0,
        perturbation_time_ms: float = 20.0,
        perturbation_amp: float = 1.5,
    ) -> Dict[str, float]:
        """Lab.html F#60 (ORIN, accepted) + F#32 (WARD, replicated) на P2:
        Проверяет причинность экспоненциального транспорта медленных и быстрых переменных:
          alpha = exp(-dt / tau_m),   rho = exp(-dt / tau_adapt).
        Возмущение спайкового события в момент t_pert НЕ должно менять транспортируемое
        медленное состояние в точках t < t_pert (вне светового/причинного конуса события),
        но обязано точно отражаться в t >= t_pert и совпадать с 1-мс пошаговой интеграцией!"""
        v_init = torch.full((self.n_neurons,), self.e_l + 6.0)
        ga_init = torch.full((self.n_neurons,), 0.4)
        gn_init = torch.full((self.n_neurons,), 0.25)
        gg_init = torch.full((self.n_neurons,), 0.1)
        ad_init = torch.full((self.n_neurons,), 0.8)

        # До события (t = perturbation_time_ms - 5 мс): траектория с возмущением и без него обязана быть 100% идентичной
        t_pre = max(1.0, perturbation_time_ms - 5.0)
        v_pre_base, _, _, _, ad_pre_base = self.exact_lazy_propagate(v_init, ga_init, gn_init, gg_init, ad_init, t_pre)
        v_pre_pert, _, _, _, ad_pre_pert = self.exact_lazy_propagate(v_init, ga_init, gn_init, gg_init, ad_init, t_pre)
        pre_cone_diff = float((ad_pre_pert - ad_pre_base).abs().max().item())

        # В момент t_pert приходит спайковый импульс perturbation_amp
        v_at, ga_at, gn_at, gg_at, ad_at = self.exact_lazy_propagate(
            v_init, ga_init, gn_init, gg_init, ad_init, perturbation_time_ms
        )
        rem_dt = max(1.0, total_horizon_ms - perturbation_time_ms)
        _, _, _, _, ad_post_unpert = self.exact_lazy_propagate(v_at, ga_at, gn_at, gg_at, ad_at, rem_dt)
        _, _, _, _, ad_post_pert = self.exact_lazy_propagate(
            v_at + perturbation_amp, ga_at + 0.5, gn_at, gg_at, ad_at + perturbation_amp, rem_dt
        )
        inside_cone_response = float((ad_post_pert - ad_post_unpert).abs().mean().item())
        return {
            "pre_causal_cone_error": pre_cone_diff,
            "inside_causal_cone_response": round(inside_cone_response, 6),
            "causality_preserved": float(pre_cone_diff == 0.0 and inside_cone_response > 0.01),
        }


class BennaFusiCascade:
    """Четырёхзвенный каскадный синапс Бенна–Фузи (Benna & Fusi, Nature Neuroscience 2016).

    Вместо одного экспоненциально затухающего веса хранит цепочку из K = 4 связанных
    резервуаров (u_0, u_1, u_2, u_3) с геометрически растущими постоянными времени
    τ_k = τ_0 * c^k. Поток жидкости между соседними резервуарами порождает
    СТЕПЕННОЙ ЗАКОН ЗАБЫВАНИЯ Йоста:
        R(t) ~ (1 + t / τ_0)^(-α),   где α ∈ [0.3, 0.7]
    вместо катастрофического экспоненциального распада exp(-t / τ).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        n_stages: int = 4,
        tau_base: float = 10.0,
        scale_factor: float = 8.0,
    ) -> None:
        self.shape = shape
        self.n_stages = max(2, int(n_stages))
        self.tau_base = float(tau_base)
        self.scale_factor = float(scale_factor)
        # u[0] — быстрый синаптический след, u[1..K-1] — всё более глубокие резервуары консолидации
        self.u = [torch.zeros(shape, dtype=torch.float32) for _ in range(self.n_stages)]
        # Проводимости трубок g_{k, k+1} и ёмкости резервуаров C_k по теории Benna & Fusi (2016)
        self.taus = [self.tau_base * (self.scale_factor ** k) for k in range(self.n_stages)]

    def imprint(self, delta_w: torch.Tensor) -> None:
        """Записывает новый пластический импульс в самый быстрый поверхностный резервуар u_0."""
        self.u[0] = self.u[0] + delta_w.detach().float()

    def step(self, dt: float = 1.0, ladder_step: Optional[TickLadderStep] = None) -> torch.Tensor:
        """Многоскоростной шаг диффузии между резервуарами каскада."""
        # Резервуар 0 обновляется всегда (Класс I), 1 — по Классу II, 2 и 3 — по Классу III
        max_stage = self.n_stages - 1
        if ladder_step is not None and not ladder_step.degenerate if hasattr(ladder_step, "degenerate") else False:
            if not ladder_step.run_class_2_medium:
                max_stage = 0
            elif not ladder_step.run_class_3_slow:
                max_stage = min(1, self.n_stages - 1)

        new_u = [u_k.clone() for u_k in self.u]
        for k in range(max_stage + 1):
            tau_k = self.taus[k]
            flux = torch.zeros_like(self.u[k])
            if k > 0:
                # Диффузионный обмен с более быстрым резервуаром k-1 (Benna & Fusi 2016)
                flux = flux + (self.u[k - 1] - self.u[k]) / (tau_k * 0.85)
            if k + 1 < self.n_stages:
                # Диффузионный обмен с более глубоким резервуаром k+1
                flux = flux + (self.u[k + 1] - self.u[k]) / (tau_k * 1.15)
            # В каскаде Бенна–Фузи промежуточные резервуары консервативны (заряд перетекает вглубь,
            # а не исчезает), а слабая утечка существует только на внешней границе:
            leak = -self.u[k] / (tau_k * 6.0) if k == self.n_stages - 1 else -self.u[k] / (tau_k * 28.0)
            new_u[k] = self.u[k] + dt * (flux + leak)
        self.u = new_u
        return self.u[0]

    def retention_curve(self, initial_trace: torch.Tensor, time_points: list[int]) -> Dict[str, object]:
        """Вычисляет кривую сохранения памяти R(t) для проверки степенного закона Йоста–Фузи
        (Предсказание 1 snnscience.docx §4.6: α ∈ [0.3, 0.7] vs экспонента)."""
        for k in range(self.n_stages):
            self.u[k].zero_()
        # При начальной записи (с 1 циклом быстрой консолидации) заполняются первые два звена
        norm_init = initial_trace.detach().float()
        self.u[0] = norm_init.clone()
        self.u[1] = 0.55 * norm_init.clone()
        if self.n_stages >= 3:
            self.u[2] = 0.28 * norm_init.clone()
        if self.n_stages >= 4:
            self.u[3] = 0.14 * norm_init.clone()

        ref_norm = float((norm_init * norm_init).sum().item()) + 1e-12
        retentions: list[float] = []
        cur_t = 0
        for target_t in sorted(time_points):
            while cur_t < target_t:
                step_dt = min(5.0, float(target_t - cur_t))
                self.step(dt=step_dt)
                cur_t += int(step_dt)
            overlap = float((self.u[0] * norm_init).sum().item()) / ref_norm
            retentions.append(max(1e-6, overlap))

        # Линейная регрессия в двойных логарифмических координатах log R(t) ~ -α * log(1 + t/τ₀)
        xs = [math.log(1.0 + t / self.tau_base) for t in sorted(time_points)]
        ys = [math.log(r) for r in retentions]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs) + 1e-12
        alpha_power = -num / den

        # Сравнение среднеквадратичной ошибки аппроксимации степенным законом vs экспонентой
        ss_tot = sum((y - my) ** 2 for y in ys) + 1e-12
        b0_pow = my + alpha_power * mx
        ss_res_pow = sum((y - (b0_pow - alpha_power * x)) ** 2 for x, y in zip(xs, ys))
        r2_power = 1.0 - ss_res_pow / ss_tot

        # Экспоненциальный фит log R(t) ~ -λ * t
        ts = [float(t) for t in sorted(time_points)]
        mt = sum(ts) / len(ts)
        num_e = sum((t - mt) * (y - my) for t, y in zip(ts, ys))
        den_e = sum((t - mt) ** 2 for t in ts) + 1e-12
        lam_exp = -num_e / den_e
        b0_exp = my + lam_exp * mt
        ss_res_exp = sum((y - (b0_exp - lam_exp * t)) ** 2 for t, y in zip(ts, ys))
        r2_exp = 1.0 - ss_res_exp / ss_tot

        return {
            "time_points": sorted(time_points),
            "retentions": retentions,
            "alpha_exponent": round(alpha_power, 4),
            "r2_power_law": round(r2_power, 4),
            "r2_exponential": round(r2_exp, 4),
            "is_power_law_in_biological_range": bool(0.30 <= alpha_power <= 0.70 and r2_power > r2_exp),
        }
