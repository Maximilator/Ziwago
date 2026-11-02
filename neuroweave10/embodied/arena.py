"""Фаза 3 Глобального Плана v8.0 (snnscience.docx §3.10, Изъян 10):
Замкнутая Воплощённая Среда Выживания и Фуражирования (`EmbodiedForagingArena`:
Closed-Loop Sensorimotor 2D Grid-World with Food, Predators, Obstacles & Shelter).

Устраняет Изъян 10 («Отсутствие замкнутой сенсомоторной петли Действие -> Среда -> Новое Ощущение»):
  1. Действие агента `a(t) ∈ {0: NORTH, 1: SOUTH, 2: WEST, 3: EAST, 4: EAT/FORAGE, 5: SHELTER/REST}`
     непосредственно перемещает физическое тело организма на 2D-арене размера `G × G` и меняет
     его следующий сенсорный кадр `O(t+1)`, уровень энергии (АТФ), интероцептивную безопасность
     и расстояние до источников пищи и хищников!
  2. Сенсорный кадр `O(t)` кодирует эгоцентрическое поле зрения (градиенты запаха пищи,
     угрозы хищника, близости укрытия и проприоцептивные координаты MEC grid-клеток).
  3. Организм `RegenesisOrganism` принимает решения через термодинамический Активный Вывод
     `G'(π) = G(π) + λ_ATP(t) · E[ATP_cost(π)]`:
     - При падении энергии (высоком голоде и растущей теневой цене λ_ATP) он направляется к пище;
     - При приближении хищника вето миндалины (`BLA Survival Veto`) уводит его от угрозы в укрытие.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import torch


@dataclass
class ArenaStepOut:
    obs_vec: torch.Tensor          # [1, input_dim] новый сенсорный вектор после действия
    reward: float                  # подкрепление (+ при поедании пищи, - при столкновении с хищником)
    atp_delta: float               # изменение запаса энергии агента
    threat_level: float            # текущая близость хищника ∈ [0, 1]
    food_collected: int            # суммарно собрано единиц пищи
    predator_hits: int             # число столкновений с хищником
    agent_pos: Tuple[int, int]


class EmbodiedForagingArena:
    """Замкнутая 2D-среда выживания и фуражирования для проверки воплощённого Активного Вывода."""

    ACTIONS: Tuple[str, ...] = ("NORTH", "SOUTH", "WEST", "EAST", "FORAGE", "REST")

    def __init__(self, grid_size: int = 8, input_dim: int = 32, seed: int = 42):
        self.grid_size = max(4, int(grid_size))
        self.input_dim = max(16, int(input_dim))
        self.g = torch.Generator().manual_seed(seed)

        self.agent_pos: List[int] = [self.grid_size // 2, self.grid_size // 2]
        self.food_positions: List[List[int]] = [
            [1, 1],
            [1, self.grid_size - 2],
            [self.grid_size - 2, 1],
            [self.grid_size // 2, max(0, self.grid_size // 2 - 2)],
        ]
        self.predator_pos: List[int] = [self.grid_size - 1, self.grid_size - 1]
        self.shelter_pos: List[int] = [0, self.grid_size // 2]

        self.atp_reserve: float = 0.85
        self.food_collected: int = 0
        self.predator_hits: int = 0
        self.step_count: int = 0
        self.last_active_schedule_mode: str = "ATTEND"
        self.last_exact_jump_decay: float = 1.0

    def evaluate_action_exact_jump(
        self,
        action_idx: int,
        dt_action_ms: float = 15.0,
        tau_sensory_ms: float = 25.0,
    ) -> Dict[str, float]:
        """Lab.html F#63 (ORIN, verified on P8): Оценка кандидатов действий через точные экспоненциальные прыжки.
        Вместо многошагового плотного тактирования в воображении (VTE), прогнозирует затухание и сдвиг
        сенсорного следа за длительность моторного примитива Δt(a) аналитически:
            x_pred(t + Δt(a)) = x(t) * exp(-Δt(a) / τ_sensory) + Δx_action(a).
        Возвращает точный коэффициент затухания и ожидаемую свободную энергию после прыжка."""
        act = int(action_idx) % len(self.ACTIONS)
        # Разные моторные примитивы имеют разную физическую длительность Δt(a)
        action_durations_ms = {0: 14.0, 1: 14.0, 2: 14.0, 3: 14.0, 4: 22.0, 5: 35.0}
        eff_dt = float(action_durations_ms.get(act, dt_action_ms))
        exact_decay = math.exp(-eff_dt / max(1.0, float(tau_sensory_ms)))
        self.last_exact_jump_decay = round(exact_decay, 5)

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0), 5: (0, 0)}
        dr, dc = moves[act]
        nr = max(0, min(self.grid_size - 1, self.agent_pos[0] + dr))
        nc = max(0, min(self.grid_size - 1, self.agent_pos[1] + dc))
        nf, _ = self.nearest_food()
        new_f_dist = self._dist([nr, nc], nf)
        new_p_dist = self._dist([nr, nc], self.predator_pos)
        expected_surprise_after_jump = round(
            exact_decay * 0.45 * new_f_dist + (6.0 / max(0.5, new_p_dist) if new_p_dist < 3.0 else 0.0),
            4,
        )
        return {
            "action_idx": float(act),
            "dt_action_ms": eff_dt,
            "exact_exp_decay": self.last_exact_jump_decay,
            "expected_surprise_after_jump": expected_surprise_after_jump,
        }

    def schedule_probe_attend_stop(
        self,
        epistemic_uncertainty: float,
        pragmatic_salience: float,
        si_path_integral: float,
        probe_threshold: float = 0.30,
        attend_threshold: float = 0.25,
    ) -> Dict[str, object]:
        """Lab.html F#37 (HECATE, verified on P8): Диспетчер Активного Вывода `PROBE / ATTEND / STOP`
        на основе путевого интеграла Синаптического Интеллекта (`si_path_integral` Ω_ij):
          - Если эпистемическая неопределённость высока, а консолидированный интеграл SI низок:
            режим `PROBE` (запрос следа памяти / исследовательское зондирование).
          - Если прагматическая отдача и интеграл SI превышают порог:
            режим `ATTEND` (целенаправленное исполнение моторного действия).
          - Если оба члена ниже порога:
            режим `STOP` (энергосберегающая остановка без лишних спайков и дрейфа весов)."""
        epistemic_term = float(epistemic_uncertainty) / (1.0 + 2.0 * max(0.0, float(si_path_integral)))
        pragmatic_term = float(pragmatic_salience) * (0.5 + 0.5 * min(2.0, max(0.0, float(si_path_integral))))

        if epistemic_term < probe_threshold and pragmatic_term < attend_threshold:
            mode = "STOP"
        elif epistemic_term >= pragmatic_term and epistemic_term >= probe_threshold:
            mode = "PROBE"
        else:
            mode = "ATTEND"
        self.last_active_schedule_mode = mode
        return {
            "mode": mode,
            "epistemic_term": round(epistemic_term, 4),
            "pragmatic_term": round(pragmatic_term, 4),
            "si_path_integral": round(float(si_path_integral), 4),
        }

    def _dist(self, p1: List[int], p2: List[int]) -> float:
        return math.hypot(float(p1[0] - p2[0]), float(p1[1] - p2[1]))

    def nearest_food(self) -> Tuple[List[int], float]:
        best_f = min(self.food_positions, key=lambda f: self._dist(self.agent_pos, f))
        return best_f, self._dist(self.agent_pos, best_f)

    def observe(self) -> torch.Tensor:
        """Формирует сенсорный вектор O(t) [1, input_dim] из эгоцентрических градиентов пищи, хищника и укрытия."""
        obs = torch.zeros(1, self.input_dim)
        nf, f_dist = self.nearest_food()
        p_dist = self._dist(self.agent_pos, self.predator_pos)
        s_dist = self._dist(self.agent_pos, self.shelter_pos)

        # Каналы 0..3: направление на ближайшую пищу (NORTH, SOUTH, WEST, EAST)
        dy_f = float(nf[0] - self.agent_pos[0])
        dx_f = float(nf[1] - self.agent_pos[1])
        food_prox = 1.0 / (1.0 + 0.5 * f_dist)
        obs[0, 0] = max(0.0, -dy_f) * food_prox
        obs[0, 1] = max(0.0, dy_f) * food_prox
        obs[0, 2] = max(0.0, -dx_f) * food_prox
        obs[0, 3] = max(0.0, dx_f) * food_prox
        obs[0, 4] = food_prox

        # Каналы 5..9: градиент угрозы хищника
        dy_p = float(self.predator_pos[0] - self.agent_pos[0])
        dx_p = float(self.predator_pos[1] - self.agent_pos[1])
        pred_prox = 1.0 / (1.0 + 0.6 * p_dist)
        obs[0, 5] = max(0.0, -dy_p) * pred_prox
        obs[0, 6] = max(0.0, dy_p) * pred_prox
        obs[0, 7] = max(0.0, -dx_p) * pred_prox
        obs[0, 8] = max(0.0, dx_p) * pred_prox
        obs[0, 9] = pred_prox

        # Каналы 10..13: проприоцепция, укрытие и интероцептивный уровень АТФ
        obs[0, 10] = float(self.agent_pos[0]) / float(self.grid_size)
        obs[0, 11] = float(self.agent_pos[1]) / float(self.grid_size)
        obs[0, 12] = 1.0 / (1.0 + 0.5 * s_dist)
        obs[0, 13] = self.atp_reserve
        # Остальные каналы — пространственные гармоники MEC grid-клеток
        for k in range(14, self.input_dim):
            freq = 0.5 + 0.25 * (k - 14)
            obs[0, k] = 0.5 * (
                math.cos(freq * self.agent_pos[0]) + math.sin(freq * self.agent_pos[1])
            )
        return obs

    def select_embodied_action(self, lambda_atp: float, valence: float) -> int:
        """Термодинамический выбор действия по минимизации G'(π) = G(π) + λ_ATP · E[ATP_cost(π)]
        с учётом сенсорного поля арены и вето миндалины при близости хищника."""
        nf, f_dist = self.nearest_food()
        p_dist = self._dist(self.agent_pos, self.predator_pos)

        if f_dist == 0.0 and p_dist > 2.2:
            return 4  # FORAGE/EAT на клетке с пищей, если хищник не вплотную

        best_act = 0
        best_g_prime = 1e9
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0)]
        for act_idx, (dr, dc) in enumerate(moves):
            nr = max(0, min(self.grid_size - 1, self.agent_pos[0] + dr))
            nc = max(0, min(self.grid_size - 1, self.agent_pos[1] + dc))
            cand_pos = [nr, nc]
            new_f_dist = self._dist(cand_pos, nf)
            new_p_dist = self._dist(cand_pos, self.predator_pos)

            # Прагматическая ценность приближения к пище масштабируется голодом (1 + λ_ATP)
            food_drive = -(1.2 + 1.5 * max(0.0, float(lambda_atp))) * (f_dist - new_f_dist)
            # Вето выживания миндалины при приближении к хищнику (приоритет безопасности над пищей!)
            predator_veto = (25.0 / max(0.25, new_p_dist) - 2.0 * new_p_dist) if new_p_dist < 3.2 else 0.0
            # Метаболическая стоимость движения против покоя
            atp_cost = 0.02 if act_idx >= 4 else 0.06
            g_prime = food_drive + predator_veto + float(lambda_atp) * atp_cost
            if g_prime < best_g_prime:
                best_g_prime = g_prime
                best_act = act_idx
        return best_act

    def step(self, action: int) -> ArenaStepOut:
        """Выполняет физический шаг в среде, обновляет позиции агента и хищника и возвращает O(t+1)."""
        self.step_count += 1
        act = int(action) % len(self.ACTIONS)
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        atp_delta = -0.012
        reward = 0.0

        if act in moves:
            dr, dc = moves[act]
            self.agent_pos[0] = max(0, min(self.grid_size - 1, self.agent_pos[0] + dr))
            self.agent_pos[1] = max(0, min(self.grid_size - 1, self.agent_pos[1] + dc))

        # Проверяем сбор пищи
        nf, f_dist = self.nearest_food()
        if f_dist == 0.0 or (act == 4 and f_dist <= 1.0):
            self.food_collected += 1
            atp_delta += 0.22
            reward += 1.0
            # Респаун съеденной пищи в новой клетке арены
            nf[0] = int(torch.randint(0, self.grid_size, (1,), generator=self.g).item())
            nf[1] = int(torch.randint(0, self.grid_size, (1,), generator=self.g).item())
        elif act == 5:
            # Отдых вблизи укрытия восстанавливает часть энергии
            atp_delta += 0.025

        # Медленное патрулирование хищника (раз в 3 такта смещается на 1 клетку)
        if self.step_count % 3 == 0:
            p_dr = 1 if self.agent_pos[0] > self.predator_pos[0] else (-1 if self.agent_pos[0] < self.predator_pos[0] else 0)
            p_dc = 1 if self.agent_pos[1] > self.predator_pos[1] else (-1 if self.agent_pos[1] < self.predator_pos[1] else 0)
            self.predator_pos[0] = max(0, min(self.grid_size - 1, self.predator_pos[0] + p_dr))
            self.predator_pos[1] = max(0, min(self.grid_size - 1, self.predator_pos[1] + p_dc))

        p_dist = self._dist(self.agent_pos, self.predator_pos)
        if p_dist == 0.0:
            self.predator_hits += 1
            atp_delta -= 0.18
            reward -= 1.0
            # Отскок агента в безопасную зону после укуса
            self.agent_pos = list(self.shelter_pos)

        self.atp_reserve = max(0.05, min(1.0, self.atp_reserve + atp_delta))
        threat = round(1.0 / (1.0 + 0.7 * p_dist), 4)
        return ArenaStepOut(
            obs_vec=self.observe(),
            reward=round(reward, 4),
            atp_delta=round(atp_delta, 4),
            threat_level=threat,
            food_collected=self.food_collected,
            predator_hits=self.predator_hits,
            agent_pos=(self.agent_pos[0], self.agent_pos[1]),
        )
