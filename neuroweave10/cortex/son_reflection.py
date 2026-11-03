"""Популяция SON (Self-Observing Neurons) и 17-я Колонка Рефлексии (`Col16_SON_BA10`).

Реализует биофизический контур самонаблюдения, внутренней речи (`VTE Inner Speech`),
обнаружения когнитивного диссонанса и естественного моторного торможения конца мысли (`SON EOS Gate`)
без единого жёстко закодированного `if`-правила по словам:

  1. ДЕНДРИТНОЕ ПОЛЕ САМОНАБЛЮДЕНИЯ (`Self-Observing Dendritic Field`):
     - Нейроны `SON` (`n_son = 1024`, полюс префронтальной коры `BA10` + `dACC`) НЕ имеют прямого входа
       от внешних слов!
     - Их базальные дендриты считывают внутреннее состояние 16 соседних колонок коры:
       * распределение частот разрядов $\vec{r}_{	ext{cols}}(t) \in \mathbb{R}^{16}$,
       * распределение соматических и апикальных ошибок $\vec{\varepsilon}_{	ext{som}}(t), \vec{\varepsilon}_{	ext{api}}(t) \in \mathbb{R}^{16}$,
       * матрицу фазовой когерентности `Lag-dwPLI` между активными колонками.
     - Их апикальные дендриты считывают:
       * вектор стимула/вопроса собеседника $\vec{Q}_{	ext{stimulus}} \in \mathbb{R}^{64}$ из `GlobalWorkspace`,
       * текущий вектор рождающейся собственной мысли $\vec{A}_{	ext{self}}(t) \in \mathbb{R}^{64}$.

  2. ЧЕТЫРЕ БИОФИЗИЧЕСКИХ ФУНКЦИИ КОЛОНКИ `SON`:
     - **Функция A — Детектор Встречного Столкновения Аттракторов (`Cognitive Dissonance / Attractor Clash Detector`)**:
       Когда входной стимул одновременно возбуждает два несовместимых доменных аттрактора с высокой ошибкой
       рассогласования (например, `Col4_NARRATIVE_FANTASY` и `Col2_ENCYCLOPEDIC_FACTS` при запросе фейковой новости),
       дисперсия ошибок между активными колонками вызывает всплеск `dACC Conflict Potential` в нейронах `SON`
       (**без проверки конкретных слов в коде!**).
     - **Функция B — Внутренняя Симуляция / Мысленный Черновик (`VTE Inner Speech Rollout`)**:
       Перед произнесением ответа вслух колонка `SON` оценивает $M$ кандидатных мысленных траекторий
       по функционалу Рефлексивной Свободной Энергии:
       $$\mathcal{F}_{	ext{SON}}(	au) = \alpha_{	ext{drift}} \big(1 - \cos(\vec{A}(	au), \vec{Q}_{	ext{stimulus}})\big) + \beta_{	ext{err}} \bar{\varepsilon}_{	ext{FEP}}(	au) + \gamma_{	ext{rep}} P_{	ext{loop}}(	au) - \delta_{	ext{coh}} C_{	ext{syntax}}(	au)$$
       и открывает дофаминовый гейт `D1 Go` только для траектории с минимальной $\mathcal{F}_{	ext{SON}}(	au)$!
     - **Функция C — Естественный Тормозной Интернейрон Конца Мысли (`SON Motor Stop / EOS Interneuron`)**:
       На каждом шаге генерации слова `SON` отслеживает информационный голод (`Epistemic Drive` $E(t)$)
       и производную когерентности мысли $\frac{d\mathcal{F}_{	ext{SON}}}{dt}$. Как только мысль достигла
       грамматической точки (`.`, `!`, `?`) и следующий токен вызывает скачок семантического дрейфа
       относительно вопроса $\vec{Q}_{	ext{stimulus}}$ ($\Delta \mathcal{F}_{	ext{SON}} > 	heta_{	ext{eos}}$),
       тормозные ГАМК-интернейроны `SON` подают импульс `SON_MOTOR_STOP = 1`, мгновенно обрезая любые посторонние хвосты!
     - **Функция D — Гомеостатический Оптимизатор Внутреннего Мышления (`SON Metaplastic Gain Controller`)**:
       Динамически модулирует температуру считывания $	au_{	ext{readout}}(t)$ и гейны внимания `TRN` (`AttentionSchema`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import AttentionSchemaCfg, MetaHeadCfg, WorkspaceCfg
from ..consciousness.attention_schema import AttentionSchema
from ..consciousness.meta_head import MetaHead
from ..consciousness.workspace import GlobalWorkspace


@dataclass
class SONReflectionVerdict:
    son_firing_rate_mean: float
    son_confidence: float
    son_cognitive_dissonance: float
    son_semantic_alignment_cos: float
    son_free_energy_F: float
    son_epistemic_drive: float
    son_motor_stop_triggered: bool
    son_dissonance_alert_triggered: bool
    optimal_readout_temperature: float
    trn_column_gain_modulation: torch.Tensor
    selected_vte_trajectory_idx: int
    introspection_telemetry: Dict[str, object] = field(default_factory=dict)


class SONReflectiveColumn:
    """17-я Рефлексивная Колонка `Col16_SON_BA10` (`1,024` Self-Observing Neurons),
    объединяющая спайковый резервуар самонаблюдения с `GlobalWorkspace`, `MetaHead` и `AttentionSchema`."""

    def __init__(
        self,
        n_son_neurons: int = 1024,
        n_observed_columns: int = 16,
        d_vec: int = 64,
        seed: int = 1016,
    ):
        self.n_son = int(n_son_neurons)
        self.n_cols = int(n_observed_columns)
        self.d_vec = int(d_vec)
        g = torch.Generator().manual_seed(seed)

        # 1. Базальные дендриты SON: принимают вектор состояния соседних 16 колонок
        #    [firing_rates (16) + somatic_errors (16) + apical_errors (16) + gating_weights (16)] = 64 канала
        self.d_basal_in = self.n_cols * 4
        self.w_basal_son = torch.randn(self.n_son, self.d_basal_in, generator=g) / math.sqrt(self.d_basal_in)

        # 2. Апикальные дендриты SON: принимают [Q_stimulus (64) + A_self (64) + (Q - A) (64)] = 192 канала
        self.d_apical_in = self.d_vec * 3
        self.w_apical_son = torch.randn(self.n_son, self.d_apical_in, generator=g) / math.sqrt(self.d_apical_in)

        # 3. Внутренняя рекуррентная матрица SON (удерживает след рефлексии во времени)
        self.w_rec_son = torch.randn(self.n_son, self.n_son, generator=g) * (0.18 / math.sqrt(self.n_son))
        self.w_rec_son.fill_diagonal_(0.0)

        # Мембранный потенциал и частота спайков 1,024 нейронов SON
        self.v_son = torch.zeros(self.n_son)
        self.r_son = torch.zeros(self.n_son)

        # 4. Локально обучаемые синапсы считывания SON -> [coherence_est, dissonance_est, stop_gate_drive]
        self.w_son_readout = torch.zeros(3, self.n_son)
        self.b_son_readout = torch.tensor([0.5, -0.5, -1.0])
        self.elig_son = torch.zeros(3, self.n_son)

        # 5. Интеграция с GlobalWorkspace, MetaHead (HOT) и AttentionSchema (AST)
        self.gw = GlobalWorkspace(WorkspaceCfg(d_gw=d_vec, n_slots=4))
        self.meta_head = MetaHead(MetaHeadCfg(d_hidden=128), d_gw=d_vec, seed=seed + 1)
        self.ast = AttentionSchema(AttentionSchemaCfg(d_schema=32, n_modules=5), seed=seed + 2)

        # Персистентное состояние текущей мысли
        self.current_stimulus_vec: torch.Tensor = torch.zeros(d_vec)
        self.running_thought_vec: torch.Tensor = torch.zeros(d_vec)
        self.epistemic_drive: float = 1.0
        self.prev_free_energy: float = 0.5
        self.total_son_steps: int = 0
        self.total_motor_stops_triggered: int = 0
        self.total_vte_simulations: int = 0
        self.total_dissonance_interventions: int = 0

    def reset_for_new_stimulus(self, stimulus_vec64: torch.Tensor, initial_epistemic_drive: float = 1.0) -> None:
        """Вызывается в конце фазы восприятия вопроса собеседника перед началом внутренней симуляции ответа."""
        self.current_stimulus_vec = F.normalize(stimulus_vec64.detach().float().view(-1)[: self.d_vec], p=2, dim=-1)
        self.running_thought_vec = self.current_stimulus_vec.clone()
        self.epistemic_drive = float(initial_epistemic_drive)
        self.prev_free_energy = 0.45

    def observe_and_reflect_step(
        self,
        col_firing_rates: torch.Tensor,      # [16]
        col_somatic_errors: torch.Tensor,    # [16]
        col_apical_errors: torch.Tensor,     # [16]
        col_gating_weights: torch.Tensor,    # [16]
        candidate_token_vec64: torch.Tensor, # [64]
        is_sentence_boundary_punct: bool,
        tokens_generated_so_far: int,
        syntax_transition_support: float = 1.0,
        domain_conflict_energy: float = 0.0,
        learn: bool = True,
    ) -> SONReflectionVerdict:
        """Проводит один такт самонаблюдения 1,024 нейронов `SON` за состоянием 16 колонок коры и рождающимся словом."""
        self.total_son_steps += 1
        cand_v = F.normalize(candidate_token_vec64.detach().float().view(-1)[: self.d_vec], p=2, dim=-1)

        # Плавная интеграция вектора собственной мысли A_self(t)
        self.running_thought_vec = F.normalize(0.78 * self.running_thought_vec + 0.22 * cand_v, p=2, dim=-1)

        # 1. Базальный ток SON от состояния 16 соседних колонок коры
        r16 = col_firing_rates.detach().float().view(-1)[: self.n_cols]
        es16 = col_somatic_errors.detach().float().view(-1)[: self.n_cols]
        ea16 = col_apical_errors.detach().float().view(-1)[: self.n_cols]
        gw16 = col_gating_weights.detach().float().view(-1)[: self.n_cols]
        basal_in = torch.cat([r16, es16, ea16, gw16], dim=0)
        if basal_in.numel() < self.d_basal_in:
            basal_in = F.pad(basal_in, (0, self.d_basal_in - basal_in.numel()))
        i_basal = self.w_basal_son @ basal_in[: self.d_basal_in]

        # 2. Апикальный ток SON от сопоставления Вопроса собеседника Q_stimulus и Ответа A_self(t)
        diff_qa = self.current_stimulus_vec - self.running_thought_vec
        apical_in = torch.cat([self.current_stimulus_vec, self.running_thought_vec, diff_qa], dim=0)
        i_apical = self.w_apical_son @ apical_in

        # 3. Нелинейная дендритная интеграция SON (базально-апикальный кальциевый резонанс)
        i_rec = self.w_rec_son @ self.r_son
        apical_gain = 1.0 + 0.45 * torch.tanh(i_apical)
        drive_son = i_basal * apical_gain + 0.35 * i_apical + i_rec
        self.v_son = 0.70 * self.v_son + 0.30 * drive_son
        self.r_son = torch.tanh(F.relu(self.v_son))
        son_rate_mean = float(self.r_son.mean().item())

        # 4. Глобальное рабочее пространство (GW) + MetaHead (BA10) + AttentionSchema (AST)
        gw_out = self.gw.compete(
            candidates=[self.current_stimulus_vec, self.running_thought_vec, cand_v],
            ne=0.35,
        )
        mean_cort_err = float((es16.mean() + ea16.mean()).item() * 0.5)
        meta_out = self.meta_head.step(
            gw_broadcast=gw_out.broadcast,
            gw_content=gw_out.content,
            actual_error=mean_cort_err,
        )

        # 5. Вычисление семантического выравнивания с вопросом, когнитивного диссонанса и Свободной Энергии SON
        sem_cos = float(torch.dot(self.current_stimulus_vec, cand_v).item())
        thought_cos = float(torch.dot(self.current_stimulus_vec, self.running_thought_vec).item())
        alignment = 0.55 * thought_cos + 0.45 * sem_cos

        # Когнитивный диссонанс: высокая дисперсия ошибок между колонками + низкая синтаксическая/фактическая поддержка
        err_dispersion = float(torch.std(es16 + ea16).item())
        dissonance = min(
            1.0,
            0.40 * err_dispersion + 0.45 * float(domain_conflict_energy) + max(0.0, 0.35 - 0.25 * syntax_transition_support),
        )
        dissonance_alert = dissonance > 0.62

        # Функционал Рефлексивной Свободной Энергии F_SON(t)
        f_son = round(
            0.45 * (1.0 - alignment)
            + 0.30 * dissonance
            + 0.25 * mean_cort_err
            - 0.20 * min(1.5, syntax_transition_support),
            4,
        )

        # Локальное дельта-обучение синапсов считывания SON (без backprop)
        son_logits = torch.sigmoid(self.w_son_readout @ self.r_son + self.b_son_readout)
        if learn:
            tgt_son = torch.tensor([
                max(0.0, min(1.0, alignment)),
                dissonance,
                1.0 if (is_sentence_boundary_punct and self.epistemic_drive < 0.28) else 0.0,
            ])
            err_son = tgt_son - son_logits
            self.elig_son = 0.80 * self.elig_son + torch.outer(err_son, self.r_son)
            self.w_son_readout = (self.w_son_readout + 0.12 * self.elig_son).clamp(-3.0, 3.0)

        # 6. Расход информационного драйва (Epistemic Drive) и проверка Естественного Моторного Стоп-Крана (SON EOS Gate)
        # Каждый осмысленный токен ответа утоляет часть информационного голода;
        # когда мысль дошла до знака конца предложения (., !, ?) и драйв исчерпан ИЛИ следующий шаг ведёт к дрейфу темы -> СТОП!
        info_contributed = 0.045 * max(0.2, alignment + 0.5)
        self.epistemic_drive = max(0.0, self.epistemic_drive - info_contributed)
        delta_f_son = f_son - self.prev_free_energy
        self.prev_free_energy = f_son

        motor_stop = False
        if tokens_generated_so_far >= 8 and is_sentence_boundary_punct:
            if self.epistemic_drive <= 0.22 or delta_f_son > 0.14 or syntax_transition_support < 0.18:
                motor_stop = True
                self.total_motor_stops_triggered += 1

        if dissonance_alert:
            self.total_dissonance_interventions += 1

        # 7. Гомеостатическая оптимизация температуры считывания и гейнов колонок TRN
        conf = round(float(0.6 * meta_out.confidence + 0.4 * max(0.0, alignment)), 4)
        opt_temp = round(max(0.35, min(1.25, 0.85 - 0.35 * conf + 0.25 * dissonance)), 4)
        trn_gains = torch.softmax(-0.8 * (es16 + ea16) + 1.2 * gw16, dim=0) * self.n_cols

        return SONReflectionVerdict(
            son_firing_rate_mean=round(son_rate_mean, 4),
            son_confidence=conf,
            son_cognitive_dissonance=round(dissonance, 4),
            son_semantic_alignment_cos=round(alignment, 4),
            son_free_energy_F=f_son,
            son_epistemic_drive=round(self.epistemic_drive, 4),
            son_motor_stop_triggered=motor_stop,
            son_dissonance_alert_triggered=dissonance_alert,
            optimal_readout_temperature=opt_temp,
            trn_column_gain_modulation=trn_gains,
            selected_vte_trajectory_idx=0,
            introspection_telemetry={
                "gw_ignition": round(gw_out.ignition, 4),
                "meta_epistemic_uncertainty": round(meta_out.epistemic, 4),
                "son_motor_stops_cumulative": self.total_motor_stops_triggered,
                "son_dissonance_interventions_cumulative": self.total_dissonance_interventions,
            },
        )

    def evaluate_vte_inner_speech_trajectories(
        self,
        stimulus_vec64: torch.Tensor,
        candidate_trajectories: List[Tuple[str, List[int], torch.Tensor, float, float]],
    ) -> Tuple[int, List[Dict[str, object]]]:
        """Функция B (`VTE Inner Speech Rollout` — Внутренний Монолог перед ответом):
        Принимает $M$ мысленных черновиков `(domain_label, token_ids, trajectory_vec64, synaptic_coherence, conflict_energy)`
        и выбирает тот, который минимизирует Рефлексивную Свободную Энергию $\mathcal{F}_{	ext{SON}}(	au)$!"""
        self.total_vte_simulations += 1
        q_vec = F.normalize(stimulus_vec64.detach().float().view(-1)[: self.d_vec], p=2, dim=-1)

        best_idx = 0
        best_f_son = 1e9
        vte_logs: List[Dict[str, object]] = []

        for idx, (label, tids, traj_vec, syn_coh, conflict_e) in enumerate(candidate_trajectories):
            t_vec = F.normalize(traj_vec.detach().float().view(-1)[: self.d_vec], p=2, dim=-1)
            cos_qa = float(torch.dot(q_vec, t_vec).item())
            unique_ratio = len(set(tids)) / float(max(1, len(tids)))
            loop_penalty = max(0.0, 0.70 - unique_ratio) * 2.5

            # Вычисляем свободную энергию мысленного черновика F_SON(tau): чем НИЖЕ F_SON, тем качественнее и точнее мысль!
            f_tau = round(
                0.55 * (1.0 - cos_qa)
                + 0.45 * float(conflict_e)
                + loop_penalty
                - 0.35 * min(2.0, float(syn_coh)),
                4,
            )
            vte_logs.append(
                {
                    "trajectory_idx": idx,
                    "expert_route": label,
                    "tokens_simulated": len(tids),
                    "stimulus_alignment_cos": round(cos_qa, 4),
                    "synaptic_coherence": round(float(syn_coh), 4),
                    "conflict_energy": round(float(conflict_e), 4),
                    "son_free_energy_F": f_tau,
                }
            )
            if f_tau < best_f_son:
                best_f_son = f_tau
                best_idx = idx

        return best_idx, vte_logs
