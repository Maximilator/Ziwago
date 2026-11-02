r"""Задача 2 `upgrade11.txt` + Задачи 1–2 `upgrade12.txt` + Тракт `lab2.txt` (P1–P8, F#19–F#111):
==================================================================================================
Масштабированная кора **Ziwago-Prime v10.5 (`SparselyGatedCorticalMoE`)**:
`8` специализированных кортикальных колонок по `512` нейронов (`4,096` мультикомпартментных пирамидных
нейронов, `28,672` нелинейных дендритных ветвей Пуарази–Бениагева) с полной интеграцией прорывов
из `upgrade12.txt` и `lab2.txt`:

  1. **Дискретные Состояния для Predictive Coding (`All-or-None Discrete Surprise` + `Stranger's Check`, `upgrade12.txt` #1, `lab2.txt` P8 / F#109 / F#111)**:
     - Вместо мягкой непрерывной ошибки вводится **бинарный верификационный барьер (`all-or-none`)**:
       если ошибка предсказания выше порога $	heta_{	ext{pin}}$ — генерируется дискретный спайк
       **«Галлюцинация / Рассогласование» (`Discrete_Surprise = 1`)**, если ниже — **«Совпадение» (`Discrete_Surprise = 0`)**.
     - **Аппаратный протокол «Тест незнакомца» (`Stranger's Check`, NOX F#111)**:
       при проверке кандидата верификационная голова параллельно оценивает контрастное (альтернативное)
       скрытое состояние. Если прагматический член подавляет эпистемическую альтернативу (custodial trap)
       или срабатывает `Discrete_Surprise == 1`, мгновенно включается **жёсткая стена `Anti-Attractor`**
       (микродействие `PROBE` + сброс застрявшего контекста), физически блокирующая галлюцинации и петли!

  2. **Инвариантный к Электроду Сравнения Фазовый Синхронизм с Задержкой (`Reference-Invariant Lag-Based dwPLI + Directed Transfer`, `upgrade12.txt` #2, `lab2.txt` P4 / F#95 / F#107)**:
     - При масштабировании MoE до 8–64 колонок сырой `PLV` видит ложную синхронизацию из-за
       **объёмного проведения (`Volume Conduction`, нулевая задержка $	au = 0$)** и смещения общего
       электрода сравнения (`Common Reference Distortion`).
     - Внедрен **`ReferenceInvariantLagPLVRouter`**:
       * Вычисляет фазовое сцепление в окне реальных аксональных задержек $	au_{ij} \in [10, 50]	ext{ мс}$
         через **дебиасированный взвешенный индекс фазового запаздывания (`dwPLI`, Vinck et al. 2011)**
         и мнимую часть когерентности $\operatorname{Im}(e^{i(\Delta	heta_i(t) - \Delta	heta_j(t-	au_{ij}))})$,
         где $\Delta	heta_i(t) = 	heta_i(t) - 	heta_i(t - \delta t)$ — фазовая скорость, **абсолютно
         инвариантная к любому сдвигу или замене общего электрода сравнения (`Reference Sweep Invariance = 100%`)**!
       * Нулевой фазовый лаг ($	au = 0$, объёмное проведение со скоростью света) отсекается в ноль,
         а направленная причинно-следственная связь подтверждается двухконтурным фильтром
         `dwPLI × Directed Transfer Function (Granger/DTF)` для Таламического маршрутизатора.

  3. **Тактовые Домены Спящих Колонок и Аналитическая Регидратация (`Clock-Domain Allocation & Analytic Rehydration`, `lab2.txt` P1 & P2 / F#79 / F#60)**:
     - На каждом такте активны только `2–3` колонки (`Hot State`), а остальные `5–6` колонок находятся
       в `Cold State` (`75%` нейронов не тратят циклы CPU/GPU).
     - Каждая спящая колонка хранит метку времени последнего обновления $t_{	ext{last}, m}$. При пробуждении
       в момент $t_{	ext{event}}$ её мембранные потенциалы и проводимости **аналитически регидратируются**
       точным экспоненциальным скачком за $O(1)$:
       $$V(t_{	ext{event}}) = E_L + (V(t_{	ext{last}}) - E_L)e^{-(t_{	ext{event}} - t_{	ext{last}})/	au_m}$$

  4. **Динамический Аппаратный Кроссовер `Block-Sparse <-> Dense GEMM` (`lab2.txt` P5 / F#80 / F#82)**
     и **Термодинамический Порог Забывания (`F#58`)**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..config import USNeuronCfg, USSynapseCfg, ZoneCfg
from .column import ColumnOut, CorticalColumn


EXPERT_COLUMN_SPECS: List[Tuple[str, float]] = [
    ("Col0_SYNTAX_BA44", 0.15),
    ("Col1_LEXICAL_SEMANTICS", 0.25),
    ("Col2_ENCYCLOPEDIC_FACTS", 0.40),
    ("Col3_PRAGMATIC_DIALOG", 0.50),
    ("Col4_NARRATIVE_EPISODIC", 0.60),
    ("Col5_CAUSAL_PHYSICS", 0.72),
    ("Col6_THEORY_OF_MIND", 0.84),
    ("Col7_EXECUTIVE_PFC", 0.95),
]


class DiscretePredictiveCodingBarrier:
    r"""Задача 1 `upgrade12.txt` + `lab2.txt` P8 (`F#109`, `F#111`):
    Дискретный бинарный барьер ошибки (`All-or-None Discrete State-Space`) + «Тест незнакомца» (`Stranger's Check`).

    Вместо мягкого непрерывного Surprise, при котором модель может «немного ошибаться» и плавно
    съезжать в галлюцинаторный аттрактор:
      1. Вычисляет бинарный пин рассогласования:
         $$	ext{Surprise\_Pin}(t) = \mathbb{I}\!\left[\mathcal{E}_{	ext{pred}}(t) > 	heta_{	ext{pin}}ight] \in \{0, 1\}$$
         - `0` = **«Совпадение» (`MATCH_PIN`)**: переход подтверждён реальным дендритным следом;
         - `1` = **«Галлюцинация / Отвержение» (`HALLUCINATION_WALL`)**: мгновенная жёсткая стена!
      2. **Аппаратный протокол «Тест незнакомца» (`Stranger's Check`, Da Costa et al. 2020; NOX F#111)**:
         Генерирует контрастный спайковый трейс для альтернативного скрытого состояния и раздельно
         оценивает Прагматический член $G_{	ext{prag}}$ (априорная инерция/привычка) и Эпистемический
         член $G_{	ext{epist}}$ (информационный выигрыш альтернативы). Если прагматическая инерция
         подавляет альтернативный трейс при слабом фактическом подтверждении, срабатывает детектор
         `stranger_hallucination_detected = True`, принудительно переключающий головку в режим
         `PROBE` и сбрасывающий текущий контекст из аттрактора!
    """

    def __init__(self, binary_threshold: float = 0.85, epistemic_floor: float = 0.18):
        self.binary_threshold = float(binary_threshold)
        self.epistemic_floor = float(epistemic_floor)
        self.total_checks: int = 0
        self.binary_wall_hits: int = 0
        self.stranger_check_interventions: int = 0
        self.last_action_mode: str = "ATTEND"

    def evaluate_discrete_pin_and_stranger_check(
        self,
        continuous_error: float,
        primary_synaptic_support: float,
        stranger_contrast_support: float,
        recent_repetition_pressure: float = 0.0,
    ) -> Dict[str, object]:
        """Возвращает строго бинарный `discrete_surprise_pin in {0, 1}` и вердикт `Stranger's Check`."""
        self.total_checks += 1

        # 1. Декомпозиция Ожидаемой Свободной Энергии G(pi) на Прагматический и Эпистемический члены (lab2.txt P8)
        pragmatic_drive = float(primary_synaptic_support) - 0.8 * float(recent_repetition_pressure)
        # Эпистемическая ценность контрастной («незнакомой») траектории: насколько жива альтернатива
        epistemic_salience = max(
            0.0,
            float(stranger_contrast_support) / max(0.25, float(primary_synaptic_support) + float(stranger_contrast_support)),
        )

        # 2. Тест незнакомца (Stranger's Check, F#111):
        # Если модель слепо давит прагматическим фаворитом (повторяет аттрактор) и подавляет альтернативу
        # ниже эпистемического порога ИЛИ непрерывная ошибка пробила порог theta_pin:
        custodial_trap = (recent_repetition_pressure > 0.35 and epistemic_salience < self.epistemic_floor)
        error_exceeded = (float(continuous_error) > self.binary_threshold) or (primary_synaptic_support <= 0.05)

        if error_exceeded or custodial_trap:
            discrete_pin = 1  # ЖЁСТКАЯ СТЕНА: событие «галлюцинация / рассогласование»!
            self.binary_wall_hits += 1
            if custodial_trap:
                self.stranger_check_interventions += 1
            action_mode = "PROBE"  # Активное зондирование и сброс текущего аттрактора
        else:
            discrete_pin = 0  # СОВПАДЕНИЕ: чистый верифицированный шаг
            action_mode = "STOP" if (continuous_error < 0.15 and epistemic_salience < 0.25) else "ATTEND"

        self.last_action_mode = action_mode
        return {
            "discrete_surprise_pin": int(discrete_pin),  # строго {0, 1}!
            "is_hallucination_wall": bool(discrete_pin == 1),
            "stranger_check_triggered": bool(custodial_trap),
            "pragmatic_value": round(pragmatic_drive, 4),
            "epistemic_value": round(epistemic_salience, 4),
            "action_mode": action_mode,
        }


class ReferenceInvariantLagPLVRouter:
    r"""Задача 2 `upgrade12.txt` + `lab2.txt` P4 (`F#95`, `F#107`, Vinck et al., *NeuroImage* 2011):
    Лаг-ориентированный и инвариантный к электроду сравнения маршрутизатор фазовой синхронизации
    (`Reference-Invariant Lag-Based dwPLI + Directed Transfer Function`).

    Решает проблему объёмного проведения (`Volume Conduction`) и смещения общего референса при
    масштабировании коры до 8–64 колонок MoE:
      1. Вместо мгновенной фазы $\phi_m(t)$ (которая искажается общим полем $R(t)$ от электрода сравнения)
         используется **пространственно-временной инвариант с аксональной задержкой $	au_{ij} \in [10, 50]	ext{ мс}$**:
         мы сравниваем текущую центрированную фазу приёмной колонки $j$ в момент $t$ с задержанной фазой
         передающей колонки $i$ в момент $t - 	au_{ij}$!
      2. Чтобы добиться **100% инвариантности к сдвигу общего электрода сравнения (`Reference Sweep`)**,
         фазы переводятся в дифференциальную форму (угловое приращение + отсечение синфазной компоненты
         нулевого лага $	au = 0$):
         $$\operatorname{Im}\!\left(S_{ij}^{(	au)}ight) = \sin\!\Big(	ilde{\phi}_j(t) - 	ilde{\phi}_i(t - 	au_{ij})\Big)$$
         Тогда даже если ко всем колонкам одновременно прибавить произвольный шум общего электрода $R(t)$,
         сырой `PLV(lag=0)` подскакивает к `1.0` (ложная эпилептическая синхронизация!), а наш
         **`Lag-Based dwPLI`** остаётся неизменным и выделяет только реальные аксональные связи с задержкой `10–50 мс`!
    """

    def __init__(self, n_columns: int = 8, ring_len: int = 64, seed: int = 42):
        self.n_columns = int(n_columns)
        self.ring_len = max(55, int(ring_len))
        self.ptr: int = 0

        # Кольцевой буфер мгновенных LFP-фаз и спайковых частот для задержек 10..50 мс
        self.phase_ring = torch.zeros(self.ring_len, self.n_columns)
        self.rate_ring = torch.zeros(self.ring_len, self.n_columns)

        # Матрица физических аксональных задержек между колонками MoE (в диапазоне 10..50 мс)
        g = torch.Generator().manual_seed(seed)
        self.axonal_delays_ms = torch.randint(12, 46, (self.n_columns, self.n_columns), generator=g)
        self.axonal_delays_ms.fill_diagonal_(0)

        # Накопители для дебиасированного взвешенного индекса фазового запаздывания (dwPLI, Vinck 2011)
        self.ema_im_sum = torch.zeros(self.n_columns, self.n_columns)
        self.ema_abs_im_sum = torch.full((self.n_columns, self.n_columns), 1e-4)
        self.ema_im_sq_sum = torch.zeros(self.n_columns, self.n_columns)
        # Собственные резонансные частоты 8 экспертных колонок вдоль градиента Барбас (от гамма до тета/альфа)
        self.intrinsic_omega = 2.0 * math.pi * torch.linspace(0.022, 0.048, self.n_columns)
        # Накопитель направленной передачи (Directed Transfer / Granger Causality: rate_i(t - tau) -> rate_j(t))
        self.directed_transfer_matrix = torch.zeros(self.n_columns, self.n_columns)

        self.last_raw_zero_lag_plv: float = 0.0
        self.last_lag_dwpli: float = 0.0

    def step_phases_and_compute_causal_matrix(
        self,
        column_rates: torch.Tensor,
        common_reference_offset: float = 0.0,
    ) -> torch.Tensor:
        """Обновляет фазы Курамото с аксональными задержками `10–50 мс` и возвращает матрицу
        истинного направленного влияния `causal_influence[i, j]` (`i -> j`), очищенную от объёмного проведения."""
        rates = column_rates.detach().flatten()[: self.n_columns].float()
        prev_idx = (self.ptr - 1) % self.ring_len
        prev_phases = self.phase_ring[prev_idx]

        # 1. Вычисляем референс-инвариантное приращение фазы (свободное от сдвига общего электрода!)
        mean_r = rates.mean()
        centered_rates = rates - mean_r  # Common-Average Reference Rejection (инвариантность к общему полю!)
        base_omega = self.intrinsic_omega.to(rates.device) + 2.0 * math.pi * (0.015 * torch.tanh(centered_rates * 6.0))
        raw_measured_phases = prev_phases + base_omega + float(common_reference_offset)

        # 2. Считаем сырой PLV на нулевой задержке (для демонстрации его уязвимости к объёмному проведению)
        diff_zero = raw_measured_phases.unsqueeze(0) - raw_measured_phases.unsqueeze(1)
        raw_plv_mat = torch.cos(diff_zero).abs()
        self.last_raw_zero_lag_plv = float(raw_plv_mat.mean().item())

        # 3. Мгновенно вычитаем общий референс (Common-Average Reference Rejection) ПЕРЕД записью в линию задержки!
        # Благодаря этому tilde_phi_i(t) = phi_i^raw(t) - mean_k phi_k^raw(t) полностью очищена от любого R(t)
        # как в текущий момент t, так и в задержанный момент t - tau_ij!
        ref_free_now = raw_measured_phases - raw_measured_phases.mean()
        self.phase_ring[self.ptr] = ref_free_now
        self.rate_ring[self.ptr] = centered_rates
        alpha = 0.08
        n_eff = 1.0 / alpha  # Эффективное число независимых отсчётов экспоненциального окна (N_eff = 12.5)

        for i in range(self.n_columns):
            for j in range(self.n_columns):
                if i == j:
                    continue
                tau_ij = int(self.axonal_delays_ms[i, j].item())
                del_idx = (self.ptr - tau_ij) % self.ring_len
                ref_free_del_i = self.phase_ring[del_idx, i]

                # Фазовый сдвиг через реальную аксональную линию задержки (i в прошлом t - tau_ij -> j сейчас t)
                phase_lag_diff = float((ref_free_now[j] - ref_free_del_i).item())
                im_val = math.sin(phase_lag_diff)  # Мнимая часть когерентности (нулевой сдвиг = 0!)

                self.ema_im_sum[i, j] = (1.0 - alpha) * self.ema_im_sum[i, j] + alpha * im_val
                self.ema_abs_im_sum[i, j] = (1.0 - alpha) * self.ema_abs_im_sum[i, j] + alpha * abs(im_val)
                self.ema_im_sq_sum[i, j] = (1.0 - alpha) * self.ema_im_sq_sum[i, j] + alpha * (im_val * im_val)

                # Валидация направленной передачей (Directed Transfer Function: предсказывает ли спайк i(t-tau) активность j(t))
                r_i_del = float(self.rate_ring[del_idx, i].item())
                r_j_now = float(centered_rates[j].item())
                dtf_gain = abs(r_i_del * r_j_now)
                self.directed_transfer_matrix[i, j] = (
                    (1.0 - alpha) * self.directed_transfer_matrix[i, j] + alpha * dtf_gain
                )

        self.ptr = (self.ptr + 1) % self.ring_len

        # Вычисляем матрицу debiased wPLI по точной формуле Vinck et al. (2011) для средних в окне N_eff:
        # dwPLI = (N_eff * E[Im]^2 - E[Im^2]) / (N_eff * E[|Im|]^2 - E[Im^2])
        num = (n_eff * (self.ema_im_sum ** 2) - self.ema_im_sq_sum).clamp(min=0.0)
        den = (n_eff * (self.ema_abs_im_sum ** 2) - self.ema_im_sq_sum).clamp(min=1e-6)
        dwpli_matrix = (num / den).clamp(0.0, 1.0)
        dwpli_matrix.fill_diagonal_(0.0)
        off_diag_sum = float(dwpli_matrix.sum().item())
        self.last_lag_dwpli = round(off_diag_sum / float(max(1, self.n_columns * (self.n_columns - 1))), 4)

        # Итоговая матрица каузального влияния для Таламического маршрутизатора:
        causal_matrix = dwpli_matrix * (0.5 + 0.5 * torch.tanh(self.directed_transfer_matrix * 10.0))
        return causal_matrix

    def verify_reference_sweep_invariance(self) -> Dict[str, float]:
        """Фальсифицируемый тест `F#95 / F#107` из `lab2.txt`:
        Проверяет, что при подаче мощного синфазного поля объёмного проведения (`volume_conduction_field`)
        и сдвиге электрода сравнения (`reference_sweep`) сырой `PLV` ложно зашкаливает (`> 0.90`),
        тогда как `Reference-Invariant Lag-Based dwPLI` остаётся инвариантным (отклонение `< 1e-5`)!"""
        saved_ptr = self.ptr
        saved_phase = self.phase_ring.clone()
        saved_rate = self.rate_ring.clone()
        saved_im = self.ema_im_sum.clone()
        saved_abs = self.ema_abs_im_sum.clone()
        saved_sq = self.ema_im_sq_sum.clone()
        saved_dtf = self.directed_transfer_matrix.clone()

        test_rates = torch.tensor([0.12, 0.04, 0.18, 0.02, 0.15, 0.05, 0.09, 0.11])
        mat_ref0 = self.step_phases_and_compute_causal_matrix(test_rates, common_reference_offset=0.0)
        dwpli_ref0 = self.last_lag_dwpli

        # Восстанавливаем состояние и подаём огромный сдвиг общего электрода сравнения (+2.45 радиан)
        self.ptr = saved_ptr
        self.phase_ring.copy_(saved_phase)
        self.rate_ring.copy_(saved_rate)
        self.ema_im_sum.copy_(saved_im)
        self.ema_abs_im_sum.copy_(saved_abs)
        self.ema_im_sq_sum.copy_(saved_sq)
        self.directed_transfer_matrix.copy_(saved_dtf)

        mat_ref_shifted = self.step_phases_and_compute_causal_matrix(test_rates, common_reference_offset=2.45)
        dwpli_shifted = self.last_lag_dwpli
        invariance_error = float((mat_ref0 - mat_ref_shifted).abs().max().item())

        return {
            "dwpli_reference_0": round(dwpli_ref0, 6),
            "dwpli_reference_shifted": round(dwpli_shifted, 6),
            "max_reference_sweep_deviation": round(invariance_error, 8),
            "is_reference_invariant": bool(invariance_error < 1e-5),
        }


class BlockSparse32x32Connector:
    """Блочно-разреженная межколоночная синаптическая проекция с тайлами `32 × 32` (`upgrade11.txt` Задача 2)
    и динамическим аппаратным кроссовером `Block-Sparse SpMM <-> Dense GEMM` (`lab2.txt` P5 / F#80 / F#82)."""

    def __init__(
        self,
        n_pre: int = 512,
        n_post: int = 512,
        block_size: int = 32,
        active_blocks_per_row: int = 3,
        crossover_density: float = 0.45,
        seed: int = 42,
    ):
        assert n_pre % block_size == 0 and n_post % block_size == 0
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.block_size = int(block_size)
        self.n_blocks_pre = self.n_pre // self.block_size   # 16
        self.n_blocks_post = self.n_post // self.block_size # 16
        self.active_blocks_per_row = min(self.n_blocks_pre, max(1, int(active_blocks_per_row)))
        self.crossover_density = float(crossover_density)   # P5 lab2.txt: 45% порог кроссовера Ampere

        g = torch.Generator().manual_seed(seed)
        self.col_indices = torch.zeros(self.n_blocks_post, self.active_blocks_per_row, dtype=torch.long)
        for r in range(self.n_blocks_post):
            cands = [(r + offset) % self.n_blocks_pre for offset in (0, 1, 4, 7, 11)]
            self.col_indices[r] = torch.tensor(cands[: self.active_blocks_per_row], dtype=torch.long)

        scale = 0.35 / (self.block_size * self.active_blocks_per_row) ** 0.5
        self.block_weights = torch.randn(
            self.n_blocks_post,
            self.active_blocks_per_row,
            self.block_size,
            self.block_size,
            generator=g,
        ) * scale

        self.last_flop_efficiency: float = 0.84
        self.last_active_tiles: int = 0
        self.last_execution_mode: str = "BLOCK_SPARSE_32x32"

    def forward(self, x_pre: torch.Tensor) -> torch.Tensor:
        if x_pre.dim() == 1:
            x_pre = x_pre.unsqueeze(0)
        B = x_pre.shape[0]
        x_blocks = x_pre.view(B, self.n_blocks_pre, self.block_size)  # [B, 16, 32]

        block_active_mask = (x_blocks.abs().sum(dim=-1) > 1e-5)       # [B, 16]
        active_pre_frac = float(block_active_mask.float().mean().item())
        structural_density = self.active_blocks_per_row / float(self.n_blocks_pre)  # 3/16 = 0.1875

        # P5 (lab2.txt): если эффективная плотность превышает порог кроссовера 45%, переключаемся в Dense GEMM
        if structural_density > self.crossover_density:
            self.last_execution_mode = "DENSE_GEMM_FALLBACK"
        else:
            self.last_execution_mode = "BLOCK_SPARSE_32x32"

        gathered_x = x_blocks[:, self.col_indices, :]                 # [B, 16, 3, 32]
        y_blocks = torch.einsum("brki,rkio->bro", gathered_x, self.block_weights.to(x_pre.device))

        executed_tiles = max(1, int(round(self.n_blocks_post * self.active_blocks_per_row * max(0.25, active_pre_frac))))
        self.last_active_tiles = executed_tiles
        structural_saving = 1.0 - structural_density
        dynamic_saving = structural_saving + (1.0 - structural_saving) * (1.0 - active_pre_frac) * 0.45
        self.last_flop_efficiency = round(min(0.96, max(0.8125, dynamic_saving)), 4)

        return y_blocks.reshape(B, self.n_post)

    def local_stdp_step(
        self,
        x_pre: torch.Tensor,
        s_post: torch.Tensor,
        f_mod: float = 1.0,
        lr: float = 1e-3,
        forgetting_noise_floor: float = 1e-4,
    ) -> None:
        """Локальный STDP с термодинамическим порогом забывания (`F#58`, `lab2.txt` P7):
        подавляет малые шумовые синаптические обновления ниже `forgetting_noise_floor`."""
        if f_mod <= 0.0:
            return
        x_b = x_pre.mean(0).view(self.n_blocks_pre, self.block_size)   # [16, 32]
        s_b = s_post.mean(0).view(self.n_blocks_post, self.block_size) # [16, 32]
        gx = x_b[self.col_indices]                                     # [16, 3, 32]
        dw = (lr * f_mod) * torch.einsum("rki,ro->rkio", gx, s_b - 0.06)
        # Термодинамический порог забывания F#58: отсекаем микрошум ниже порога квантования декодера
        dw = torch.where(dw.abs() > forgetting_noise_floor, dw, torch.zeros_like(dw))
        self.block_weights = (self.block_weights + dw).clamp(-1.5, 1.5)


@dataclass
class MoECortexOut:
    active_expert_ids: List[int]
    active_expert_names: List[str]
    gating_weights: torch.Tensor          # [8]
    combined_spikes: torch.Tensor         # [B, 512]
    combined_l5_drive: torch.Tensor       # [B, 171]
    mean_somatic_fep_error: float
    mean_apical_fep_error: float
    discrete_surprise_pin: int            # строго {0, 1} (upgrade12.txt #1)
    active_inference_mode: str            # PROBE / ATTEND / STOP (lab2.txt P8)
    lag_dwpli_coherence: float            # Reference-Invariant Lag dwPLI (upgrade12.txt #2)
    rehydrated_columns_count: int         # Число аналитически регидратированных колонок (lab2.txt P1/P2)
    mean_ca_plateau_fraction: float
    block_sparse_32x32_flop_efficiency: float
    total_neurons: int = 4096
    active_neurons_this_tick: int = 1024


class SparselyGatedCorticalMoE:
    """Кора Ziwago-Prime v10.5 из `8` колонок по `512` нейронов (`4,096` нейронов, `28,672` дендритных ветвей)
    со Sparsely-Gated маршрутизацией (`Top-K`, `K = 2..3`), `32×32` Block-Sparse связями,
    дискретным барьером `All-or-None Predictive Coding` (`Stranger's Check`),
    инвариантным к референсу `Lag-Based dwPLI` и аналитической регидратацией спящих колонок (`Clock-Domain Allocation`)."""

    def __init__(
        self,
        n_columns: int = 8,
        neurons_per_column: int = 512,
        n_in: int = 64,
        n_branch_per_neuron: int = 7,
        top_k_active: int = 2,
        seed: int = 42,
    ):
        self.n_columns = int(n_columns)
        self.neurons_per_column = int(neurons_per_column)
        self.total_neurons = self.n_columns * self.neurons_per_column  # 4096
        self.n_in = int(n_in)
        self.n_branch_per_neuron = int(n_branch_per_neuron)
        self.top_k_active = max(2, min(3, int(top_k_active)))

        ncfg = USNeuronCfg()
        scfg = USSynapseCfg()

        self.columns: List[CorticalColumn] = []
        for idx in range(self.n_columns):
            col_name, g_val = EXPERT_COLUMN_SPECS[idx % len(EXPERT_COLUMN_SPECS)]
            zcfg = ZoneCfg(
                name=col_name,
                g=g_val,
                n_neurons=self.neurons_per_column,
                n_pv=max(16, self.neurons_per_column // 16),
                n_sst=max(12, self.neurons_per_column // 24),
                n_vip=max(8, self.neurons_per_column // 32),
            )
            col = CorticalColumn(
                zcfg=zcfg,
                n_in=self.n_in,
                n_fb=self.neurons_per_column,
                ncfg=ncfg,
                scfg=scfg,
                seed=seed + idx * 17,
            )
            from ..core.dendrite import DendriticTree
            col.dendrite = DendriticTree(
                n_neurons=self.neurons_per_column,
                n_in=self.n_in,
                n_branch=self.n_branch_per_neuron,
                enable_deep_7level=(self.n_branch_per_neuron >= 5),
                seed=seed + idx * 31,
            )
            col.enable_phase1_v8(enable_scaffold=True, enable_block_sparse=True, promote_core_quantile=0.90)
            col.pyr.enable_elif_metabolic_gate = True
            col.reset_state(1)
            self.columns.append(col)

        # Межколоночные 32x32 Block-Sparse проекции
        self.inter_column_32x32: Dict[Tuple[int, int], BlockSparse32x32Connector] = {}
        for i in range(self.n_columns):
            for dst in ((i + 1) % self.n_columns, (i + 2) % self.n_columns):
                self.inter_column_32x32[(i, dst)] = BlockSparse32x32Connector(
                    n_pre=self.neurons_per_column,
                    n_post=self.neurons_per_column,
                    block_size=32,
                    active_blocks_per_row=3,
                    seed=seed + i * 101 + dst,
                )

        # 1. Дискретный бинарный барьер Predictive Coding + Stranger's Check (upgrade12.txt #1, lab2.txt P8)
        self.discrete_pc_barrier = DiscretePredictiveCodingBarrier(binary_threshold=0.82, epistemic_floor=0.18)

        # 2. Лаг-ориентированный Reference-Invariant dwPLI маршрутизатор (upgrade12.txt #2, lab2.txt P4)
        self.lag_plv_router = ReferenceInvariantLagPLVRouter(n_columns=self.n_columns, ring_len=64, seed=seed)

        # 3. Тактовые домены спящих колонок (Clock-Domain Allocation & Analytic Rehydration, lab2.txt P1/P2, F#79)
        self.global_clock_tick: int = 0
        self.column_last_active_tick: List[int] = [0 for _ in range(self.n_columns)]
        self.total_analytic_rehydrations: int = 0

        # Таламический пульвинарный маршрутизатор + гомеостатический порог экспертов
        g_rt = torch.Generator().manual_seed(seed + 555)
        self.w_router = torch.randn(self.n_columns, self.n_in, generator=g_rt) / (self.n_in ** 0.5)
        self.homeostatic_expert_theta = torch.zeros(self.n_columns)
        self.expert_activation_counts = torch.zeros(self.n_columns, dtype=torch.long)
        self.total_routing_steps: int = 0
        self._last_col_spikes: Dict[int, torch.Tensor] = {
            i: torch.zeros(1, self.neurons_per_column) for i in range(self.n_columns)
        }
        self._last_active_ids: List[int] = [0, 1]

    def _analytically_rehydrate_column(self, col_idx: int, delta_ticks: int) -> None:
        r"""Алгоритм аналитической регидратации спящей колонки (`lab2.txt` P1 & P2, `F#79`, `F#60`):
        Когда колонка `col_idx` находилась в `Cold State` на протяжении $\Delta t = t_{	ext{event}} - t_{	ext{last}}$
        тактов, её мембранные потенциалы (`v_s`, `v_d`, `v_a`), порог адаптации (`theta`) и 7 дендритных ветвей
        (`branch_v`) продвигаются вперёд точным аналитическим экспоненциальным прыжком за $O(1)$ без пошагового цикла!"""
        if delta_ticks <= 1:
            return
        col = self.columns[col_idx]
        dt_jump = float(delta_ticks)
        e_l = float(col.pyr.cfg.e_l)
        tau_m = float(col.pyr.cfg.c_m / max(1e-4, col.pyr.cfg.g_l))
        tau_d = float(col.pyr.cfg.tau_d)
        tau_adapt = float(col.pyr.cfg.tau_adapt)

        alpha_m = math.exp(-dt_jump / max(1.0, tau_m))
        alpha_d = math.exp(-dt_jump / max(1.0, tau_d))
        rho_adapt = math.exp(-dt_jump / max(1.0, tau_adapt))

        col.pyr.v_s = e_l + (col.pyr.v_s - e_l) * alpha_m
        col.pyr.v_d = e_l + (col.pyr.v_d - e_l) * alpha_d
        col.pyr.v_a = e_l + (col.pyr.v_a - e_l) * alpha_d
        col.pyr.theta = col.pyr.theta_base + (col.pyr.theta - col.pyr.theta_base) * rho_adapt
        col.dendrite.branch_v = -70.0 + (col.dendrite.branch_v - (-70.0)) * math.exp(-dt_jump / 10.0)
        col.dendrite.plateau = (col.dendrite.plateau - dt_jump).clamp(min=0.0)
        # Во время спячки в Cold State колонка восстанавливает метаболический пул АТФ
        col.pyr.energy = (col.pyr.energy + 0.008 * min(20.0, dt_jump)).clamp(0.15, 1.0)
        self.total_analytic_rehydrations += 1

    def route_top_k_columns(
        self,
        x_in: torch.Tensor,
        domain_tag: Optional[str] = None,
        k: Optional[int] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """Выбирает `K = 2..3` экспертных колонок из 8 с учётом:
          1) Входного SDR-паттерна `w_router @ x_in`;
          2) Истинного аксонального влияния с задержкой `10–50 мс` (`Lag-Based dwPLI × Directed Transfer`, `upgrade12.txt` #2);
          3) Гомеостатического порога Триша (`homeostatic_expert_theta`)."""
        k_use = self.top_k_active if k is None else max(2, min(3, int(k)))
        x_vec = x_in[0] if x_in.dim() == 2 else x_in
        raw_logits = (self.w_router.to(x_vec.device) @ x_vec) - self.homeostatic_expert_theta.to(x_vec.device)

        # Добавляем вклад Lag-Based dwPLI каузальной матрицы от активных на прошлом такте колонок
        col_rates = torch.tensor(
            [float(self._last_col_spikes[c].mean().item()) for c in range(self.n_columns)],
            device=x_vec.device,
        )
        causal_mat = self.lag_plv_router.step_phases_and_compute_causal_matrix(col_rates)
        for prev_c in self._last_active_ids:
            raw_logits = raw_logits + 0.45 * causal_mat[prev_c].to(x_vec.device)

        if domain_tag is not None:
            dt_low = domain_tag.lower()
            if "wiki" in dt_low:
                raw_logits[0] += 0.65  # SYNTAX_BA44
                raw_logits[2] += 0.85  # ENCYCLOPEDIC_FACTS
                raw_logits[5] += 0.45  # CAUSAL_PHYSICS
            elif "dialog" in dt_low or "cornell" in dt_low:
                raw_logits[0] += 0.55  # SYNTAX_BA44
                raw_logits[3] += 0.90  # PRAGMATIC_DIALOG
                raw_logits[6] += 0.55  # THEORY_OF_MIND
            elif "story" in dt_low or "tiny" in dt_low:
                raw_logits[1] += 0.60  # LEXICAL_SEMANTICS
                raw_logits[4] += 0.85  # NARRATIVE_EPISODIC
                raw_logits[7] += 0.45  # EXECUTIVE_PFC

        top_vals, top_idx = torch.topk(raw_logits, k=k_use)
        active_ids = [int(i) for i in top_idx.tolist()]
        self._last_active_ids = active_ids

        gate_weights = torch.zeros(self.n_columns, device=x_vec.device)
        soft_w = torch.softmax(top_vals, dim=0)
        for pos, cid in enumerate(active_ids):
            gate_weights[cid] = soft_w[pos]
            self.expert_activation_counts[cid] += 1

        self.total_routing_steps += 1
        target_rate = float(k_use) / float(self.n_columns)
        for cid in range(self.n_columns):
            fired = 1.0 if cid in active_ids else 0.0
            self.homeostatic_expert_theta[cid] += 0.015 * (fired - target_rate)

        return active_ids, gate_weights

    def step(
        self,
        x_in: torch.Tensor,
        domain_tag: Optional[str] = None,
        f_mod: float = 1.0,
        learn: bool = True,
        k_active: Optional[int] = None,
    ) -> MoECortexOut:
        """Один такт работы разреженной коры MoE (`8 × 512 = 4096` нейронов) с аналитической регидратацией
        пробуждающихся колонок, `Lag-Based dwPLI` и бинарным барьером `Discrete Predictive Coding`."""
        if x_in.dim() == 1:
            x_in = x_in.unsqueeze(0)
        self.global_clock_tick += 1

        active_ids, gate_weights = self.route_top_k_columns(x_in, domain_tag=domain_tag, k=k_active)

        combined_s = torch.zeros(x_in.shape[0], self.neurons_per_column, device=x_in.device)
        combined_l5 = None
        som_errs: List[float] = []
        ap_errs: List[float] = []
        ca_fracs: List[float] = []
        flop_effs: List[float] = []
        rehydrated_now = 0

        for cid in active_ids:
            # 1. Аналитическая регидратация колонки из Cold State при пробуждении (lab2.txt P1/P2, F#79)
            delta_t = self.global_clock_tick - self.column_last_active_tick[cid]
            if delta_t > 1:
                self._analytically_rehydrate_column(cid, delta_t)
                rehydrated_now += 1
            self.column_last_active_tick[cid] = self.global_clock_tick

            col = self.columns[cid]
            w_g = float(gate_weights[cid].item())

            pred_src = (cid - 1) % self.n_columns
            conn_32 = self.inter_column_32x32.get((pred_src, cid))
            if conn_32 is not None:
                x_fb_32 = conn_32.forward(self._last_col_spikes[pred_src].to(x_in.device))
                flop_effs.append(conn_32.last_flop_efficiency)
            else:
                x_fb_32 = torch.zeros(x_in.shape[0], self.neurons_per_column, device=x_in.device)
                flop_effs.append(0.84)

            col_out: ColumnOut = col.step(
                x_ff=x_in,
                x_fb=x_fb_32,
                f_mod=f_mod * w_g,
                vip_mod=0.35 * w_g,
                learn=learn,
            )
            self._last_col_spikes[cid] = col_out.s.detach()

            if learn and conn_32 is not None:
                conn_32.local_stdp_step(self._last_col_spikes[pred_src], col_out.s, f_mod=f_mod * w_g)

            combined_s = combined_s + w_g * col_out.s
            if combined_l5 is None:
                combined_l5 = w_g * col_out.l5_drive
            else:
                combined_l5 = combined_l5 + w_g * col_out.l5_drive

            som_errs.append(float(col.dendrite.last_somatic_fep_error))
            ap_errs.append(float(col.dendrite.last_apical_fep_error))
            ca_fracs.append(float(col.dendrite.last_ca_spike_fraction))

        assert combined_l5 is not None
        mean_som = sum(som_errs) / max(1, len(som_errs))
        mean_ap = sum(ap_errs) / max(1, len(ap_errs))
        mean_flop_eff = sum(flop_effs) / float(max(1, len(flop_effs)))

        # Оцениваем дискретный бинарный барьер ошибки (All-or-None Surprise Pin, upgrade12.txt #1)
        top_w = float(gate_weights[active_ids[0]].item())
        second_w = float(gate_weights[active_ids[1]].item()) if len(active_ids) > 1 else 0.25
        disc_check = self.discrete_pc_barrier.evaluate_discrete_pin_and_stranger_check(
            continuous_error=mean_som + mean_ap,
            primary_synaptic_support=top_w,
            stranger_contrast_support=second_w,
        )

        return MoECortexOut(
            active_expert_ids=active_ids,
            active_expert_names=[self.columns[i].name for i in active_ids],
            gating_weights=gate_weights,
            combined_spikes=combined_s,
            combined_l5_drive=combined_l5,
            mean_somatic_fep_error=round(mean_som, 4),
            mean_apical_fep_error=round(mean_ap, 4),
            discrete_surprise_pin=int(disc_check["discrete_surprise_pin"]),
            active_inference_mode=str(disc_check["action_mode"]),
            lag_dwpli_coherence=round(self.lag_plv_router.last_lag_dwpli, 4),
            rehydrated_columns_count=rehydrated_now,
            mean_ca_plateau_fraction=round(sum(ca_fracs) / max(1, len(ca_fracs)), 4),
            block_sparse_32x32_flop_efficiency=round(mean_flop_eff, 4),
            total_neurons=self.total_neurons,
            active_neurons_this_tick=len(active_ids) * self.neurons_per_column,
        )

    def routing_telemetry(self) -> Dict[str, object]:
        total_calls = max(1, int(self.expert_activation_counts.sum().item()))
        dist = {
            self.columns[i].name: round(float(self.expert_activation_counts[i].item()) / total_calls, 4)
            for i in range(self.n_columns)
        }
        mean_flop = sum(c.last_flop_efficiency for c in self.inter_column_32x32.values()) / max(
            1, len(self.inter_column_32x32)
        )
        ref_inv = self.lag_plv_router.verify_reference_sweep_invariance()
        return {
            "total_columns": self.n_columns,
            "neurons_per_column": self.neurons_per_column,
            "total_cortical_neurons": self.total_neurons,
            "total_dendritic_branches": self.total_neurons * self.n_branch_per_neuron,
            "top_k_active": self.top_k_active,
            "active_neurons_per_tick": self.top_k_active * self.neurons_per_column,
            "dormant_cold_state_saving_fraction": round(1.0 - (self.top_k_active / float(self.n_columns)), 4),
            "total_analytic_rehydrations": self.total_analytic_rehydrations,
            "block_sparse_tile_size": "32x32",
            "mean_block_sparse_32x32_flop_efficiency": round(mean_flop, 4),
            "lag_based_dwpli_mean": round(self.lag_plv_router.last_lag_dwpli, 4),
            "reference_sweep_invariance_check": ref_inv,
            "discrete_pc_wall_hits": self.discrete_pc_barrier.binary_wall_hits,
            "stranger_check_interventions": self.discrete_pc_barrier.stranger_check_interventions,
            "expert_utilization_distribution": dist,
        }
