"""Фазово-голографический синтаксический биндер (Ziwago-Prime v2.5: Grammar & Roles).

Реализует нейробиологический контур синтаксиса (поля Брока BA44/45 ↔ SMA ↔ Таламус):
1. Синтаксический слот-фазовый счётчик внутри предложения (Ding et al. 2016;
   Lisman & Jensen 2013):
   Каждая позиция в предложении (после точки/запятой/начала фразы) активирует
   свою жесткую структурную роль R_k ∈ C^D на единичной окружности:
     Role 0: DET / INIT (артикль / зачин: "the", "a", "once")
     Role 1: ADJ / MOD  (прилагательное / модификатор: "little", "big", "happy")
     Role 2: SUBJ       (субъект / герой: "girl", "dog", "cat", "lily")
     Role 3: VERB       (действие / предикат: "sat", "ran", "went", "saw")
     Role 4: PREP / AUX (предлог / связка: "to", "in", "with", "on")
     Role 5: OBJ        (объект / место: "park", "house", "tree", "ball")
     Role 6: PUNCT      (граница клаузы / точка / запятая)

2. Двойной выход:
   - `role_vec` [B, n_slots]: one-hot + фазовый след ОЖИДАЕМОЙ СЛЕДУЮЩЕЙ синтаксической
     роли (для базально-апикального гейтинга моторной коры — в слоте Действия могут
     побеждать только глаголы, а в слоте Субъекта — существительные!).
   - `context_vec` [B, D]: распакованный через PhaseVSA (Frame ⊙ conj(R_k))
     голографический вектор содержания текущего предложения (кто именно был в Role 0..2),
     удерживающий семантическую связь между Субъектом и Предикатом!
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass
class PhaseStoryOut:
    theta_phase: float          # текущая фаза θ ∈ [0, 2π)
    gamma_slot: int             # индекс ТЕКУЩЕЙ роли ∈ [0, K-1]
    next_role: int              # индекс ОЖИДАЕМОЙ СЛЕДУЮЩЕЙ роли ∈ [0, K-1]
    role_vec: torch.Tensor      # [B, K] вектор активной синтаксической роли (BA44/SMA)
    context_vec: torch.Tensor   # [B, D] вещественный фазовый контекст (PhaseVSA unbind)
    salience: float             # сила записи в текущем слоте
    boundary_reset: bool        # флаг сброса фразы (для таламического LTS-burst)


class ThetaGammaStoryBinder:
    """Синтаксический тета-гамма мультиплексор + эмерджентный гетероклинический
    аттрактор ролей Брока BA44/45 (Winnerless Competition, Rabinovich et al. 2001)
    + комплексная PhaseVSA память ролей."""

    def __init__(self, dim: int, n_slots: int = 7,
                 theta_hz: float = 8.0, gamma_hz: float = 40.0,
                 decay: float = 0.96, seed: int = 42):
        self.dim = dim
        self.n_slots = n_slots
        self.theta_hz = theta_hz
        self.gamma_hz = gamma_hz
        self.decay = decay
        g = torch.Generator().manual_seed(seed)
        # Ортогональные фазовые векторы синтаксических ролей R_k ∈ C^D на единичной окружности
        role_angles = (torch.rand(n_slots, dim, generator=g) * 2.0 - 1.0) * math.pi
        self.roles = torch.complex(torch.cos(role_angles), torch.sin(role_angles))
        # Проекция спайкового вектора в фазовые углы филлера
        self.w_angle = torch.randn(dim, dim, generator=g) / math.sqrt(max(1, dim))
        # P2: Гетероклиническая матрица синтаксических переходов W_trans[next, cur] (BA44/SMA):
        # задаёт асимметричную конкуренцию без победителя (Winnerless Competition) и допускает
        # естественные грамматические переходы с пропуском опциональных модификаторов
        # (например, DET(0) → SUBJ(2) без ADJ(1), или VERB(3) → OBJ(5)/PUNCT(6))!
        self.w_trans = torch.full((n_slots, n_slots), -0.35)
        self.w_trans.fill_diagonal_(-1.20)  # самоторможение роли (не застревать в одной роли)
        for k in range(n_slots):
            self.w_trans[(k + 1) % n_slots, k] = 1.05
        if n_slots >= 7:
            self.w_trans[2, 0] = 0.82   # DET(0) → SUBJ(2) (пропуск опционального ADJ)
            self.w_trans[3, 2] = 1.15   # SUBJ(2) → VERB(3) (ядро предикации)
            self.w_trans[5, 3] = 0.80   # VERB(3) → OBJ(5) (прямое дополнение без предлога)
            self.w_trans[6, 3] = 0.72   # VERB(3) → PUNCT(6) (непереходный глагол в конце фразы)
            self.w_trans[6, 5] = 1.10   # OBJ(5) → PUNCT(6)
            self.w_trans[0, 6] = 1.20   # PUNCT(6) → DET/INIT(0)
            self.w_trans[2, 6] = 0.78   # PUNCT(6) → SUBJ(2) (начало фразы с имени собственного)
        # Пластичная матрица распознавания синтаксической роли по сенсорному/кортикальному следу [K, D]
        self.w_lex_role = torch.zeros(n_slots, dim)
        # Текущее состояние популяции синтаксических нейронов BA44 [K]
        self.role_state = torch.zeros(n_slots)
        self.role_state[0] = 1.0
        self.prev_role: int = n_slots - 1
        # Голографический фрейм текущего предложения и глобального сюжета [D]
        self.clause_frame = torch.zeros(dim, dtype=torch.complex64)
        self.story_frame = torch.zeros(dim, dtype=torch.complex64)
        # Память субъекта истории (Role 0..2), которая переживает границы предложений
        self.subject_trace = torch.zeros(dim)
        # Буфер слотов текущего предложения [K, D]
        self.slots = torch.zeros(n_slots, dim)
        self.slot_age = torch.zeros(n_slots)
        self.t_ms: float = 0.0
        self.clause_pos: int = 0
        self._last_boundary: bool = False
        self._last_x_norm: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.clause_frame.zero_()
        self.story_frame.zero_()
        self.subject_trace.zero_()
        self.slots.zero_()
        self.slot_age.zero_()
        self.role_state.zero_()
        self.role_state[0] = 1.0
        self.prev_role = self.n_slots - 1
        self.t_ms = 0.0
        self.clause_pos = 0
        self._last_boundary = False
        self._last_x_norm = None

    def notify_boundary(self) -> None:
        """Сигнал конца предложения / клаузы (точка, восклицательный знак, запятая):
        сбрасывает синтаксический аттрактор ролей в Role 0 (Субъект/Зачин)
        и переносит тему предложения в долговременный story_frame."""
        self.story_frame = 0.85 * self.story_frame + 0.15 * self.clause_frame
        mag = self.story_frame.abs().clamp(min=1e-6)
        self.story_frame = self.story_frame / mag
        self.clause_frame.mul_(0.25)
        self.role_state.zero_()
        self.role_state[0] = 1.0
        self.prev_role = self.n_slots - 1
        self.clause_pos = 0
        self._last_boundary = True

    def _to_phasor(self, x: torch.Tensor) -> torch.Tensor:
        """Спайковый/активностный вектор [D] → комплексный фазор exp(i·ψ) ∈ C^D."""
        rms = x.pow(2).mean().sqrt().clamp(min=1e-6)
        ang = torch.tanh((x / rms) @ self.w_angle.t()) * math.pi
        return torch.complex(torch.cos(ang), torch.sin(ang))

    def learn_role(self, x_spikes: torch.Tensor, role_idx: int,
                   f_mod: float = 1.0, eta: float = 0.08) -> None:
        """P2: Локальное обучение синтаксического распознавателя BA44 (w_lex_role)
        и гетероклинической матрицы переходов (w_trans) по опыту."""
        if f_mod <= 0.0 or not (0 <= role_idx < self.n_slots):
            return
        x_mean = x_spikes.detach().mean(0) if x_spikes.dim() == 2 else x_spikes.detach()
        if x_mean.numel() != self.dim:
            if x_mean.numel() > self.dim:
                x_mean = x_mean[:self.dim]
            else:
                x_mean = torch.nn.functional.pad(x_mean, (0, self.dim - x_mean.numel()))
        x_norm = x_mean / x_mean.norm().clamp(min=1e-6)
        lr = eta * max(0.2, min(2.0, float(f_mod)))
        target = torch.full((self.n_slots,), -0.15, device=x_norm.device)
        target[role_idx] = 1.0
        pred = self.w_lex_role @ x_norm
        self.w_lex_role = (self.w_lex_role + lr * torch.outer(target - pred, x_norm)).clamp(-1.5, 2.0)
        # STDP синтаксического перехода prev_role → role_idx
        if 0 <= self.prev_role < self.n_slots and self.prev_role != role_idx:
            self.w_trans[role_idx, self.prev_role] = min(
                1.6, float(self.w_trans[role_idx, self.prev_role]) + 0.03 * lr
            )

    def step(self, x_spikes: torch.Tensor, ach: float = 0.5,
             ne: float = 0.3, dt: float = 1.0,
             is_boundary: bool = False,
             role_hint: Optional[torch.Tensor] = None,
             learn: bool = True) -> PhaseStoryOut:
        """Один шаг: распознаёт синтаксическую роль текущего слова через
        гетероклинический аттрактор Winnerless Competition (W_trans + лексический вход),
        связывает слово с его ролью в PhaseVSA и выдаёт вектор ожидаемых следующих ролей."""
        self.t_ms += dt
        b_sz = x_spikes.shape[0] if x_spikes.dim() == 2 else 1
        x_mean = x_spikes.mean(0) if x_spikes.dim() == 2 else x_spikes
        if x_mean.numel() != self.dim:
            if x_mean.numel() > self.dim:
                x_mean = x_mean[:self.dim]
            else:
                x_mean = torch.nn.functional.pad(x_mean, (0, self.dim - x_mean.numel()))
        x_norm = x_mean.detach() / x_mean.detach().norm().clamp(min=1e-6)
        self._last_x_norm = x_norm

        if is_boundary:
            self.notify_boundary()
            cur_role = self.n_slots - 1  # PUNCT / BOUNDARY роль
            next_role = 0                # После границы предложения ждём Role 0 (Зачин/Субъект)
            boundary_flag = True
            next_drive = torch.zeros(self.n_slots, device=x_spikes.device)
            next_drive[0] = 1.0
            if self.n_slots >= 3:
                next_drive[2] = 0.45     # Допускаем старт предложения сразу с Субъекта
        else:
            if self.clause_pos >= self.n_slots:
                self.notify_boundary()
            default_role = self.clause_pos % self.n_slots
            # P2: Гетероклиническое предсказание роли от предыдущего состояния BA44
            trans_prior = torch.relu(self.w_trans[:, self.prev_role])
            # Лексико-кортикальное распознавание роли текущего слова
            lex_cue = torch.relu(self.w_lex_role.to(x_norm.device) @ x_norm)
            if role_hint is not None and role_hint.numel() == self.n_slots:
                rh = role_hint.detach().flatten().to(x_norm.device)
                if float(rh.max().item()) > 0.0:
                    lex_cue = 0.45 * lex_cue + 0.85 * (rh / rh.max().clamp(min=1e-6))
            # Если мы в начале клаузы (clause_pos == 0), цикл открывается с Role 0;
            # внутри клаузы переходы возможны ТОЛЬКО вдоль открытых гетероклинических сепаратрис
            # W_trans[k_next, k_prev] > 0 (например, DET(0) → ADJ(1) или пропуск DET(0) → SUBJ(2))!
            non_punct_slots = max(1, self.n_slots - 1)
            if self.clause_pos == 0:
                cur_role = 0
            else:
                allowed_mask = (trans_prior[:non_punct_slots] > 0.10).float()
                gated_cue = lex_cue[:non_punct_slots] * allowed_mask
                if float(gated_cue.max().item()) > 0.25:
                    comb_score = (0.20 * trans_prior[:non_punct_slots] + 1.60 * gated_cue) * allowed_mask
                    cur_role = int(comb_score.argmax().item())
                else:
                    cur_role = default_role

            # Обновляем позицию фазового осциллятора и состояние популяции BA44
            self.clause_pos = cur_role + 1
            self.role_state = 0.20 * self.role_state + 0.80 * torch.nn.functional.one_hot(
                torch.tensor(cur_role), self.n_slots
            ).float()
            if learn and float(x_mean.abs().mean().item()) > 0.01:
                self.learn_role(x_mean, cur_role, f_mod=0.5 + 0.5 * ach)
            self.prev_role = cur_role

            # Вычисляем основную ожидаемую СЛЕДУЮЩУЮ роль через гетероклинический переход
            next_drive = torch.relu(self.w_trans[:, cur_role]).to(x_spikes.device)
            if float(next_drive.max().item()) > 0.0:
                next_role = int(next_drive.argmax().item())
            else:
                next_role = self.clause_pos % self.n_slots
            boundary_flag = self._last_boundary
            self._last_boundary = False

        theta_phase = (2.0 * math.pi * cur_role / max(1, self.n_slots))
        self.slot_age += dt

        act_level = float(x_mean.abs().mean().item())
        salience = act_level * (0.5 + 0.5 * ach + 0.5 * ne)
        if salience > 0.01:
            self.slots[cur_role] = 0.2 * self.slots[cur_role] + 0.8 * x_mean
            self.slot_age[cur_role] = 0.0
            # Если это группа Субъекта (Role 0..2), фиксируем субъект предложения
            if cur_role <= 2 and not is_boundary:
                self.subject_trace = 0.4 * self.subject_trace + 0.6 * x_mean
            filler = self._to_phasor(x_mean)
            bound = self.roles[cur_role] * filler
            # Жесткое голографическое связывание (PhaseVSA binding)
            self.clause_frame = self.clause_frame * self.decay + bound
            mag = self.clause_frame.abs().clamp(min=1e-6)
            self.clause_frame = self.clause_frame / mag

        # Распаковка (unbind) содержания ролей через сопряжённые векторы ролей conj(R_k):
        # Мы распаковываем Субъект (Role 0..2) и связываем его с ожидаемой следующей ролью!
        unbound_clause = (self.clause_frame.unsqueeze(0) * self.roles.conj()).real  # [K, D]
        unbound_story = (self.story_frame.unsqueeze(0) * self.roles.conj()).real    # [K, D]
        # Контекст для следующего слова: комбинация распакованного Субъекта клаузы
        # и фазовой сигнатуры ожидаемой роли R_{next_role}.real
        subj_unbound = 0.5 * unbound_clause[:3].mean(0) + 0.5 * self.subject_trace
        role_probe = self.roles[next_role].real
        ctx_1d = 0.55 * subj_unbound + 0.25 * unbound_story.mean(0) + 0.20 * role_probe
        ctx = ctx_1d.unsqueeze(0).expand(b_sz, -1).contiguous()

        # Вектор синтаксической роли [B, K]: указывает моторному пулу следующую роль
        # и плавное перекрытие текущего гамма-окна
        r_vec = torch.zeros(b_sz, self.n_slots, device=x_spikes.device)
        r_vec[:, next_role] = 1.0
        if cur_role < self.n_slots:
            r_vec[:, cur_role] += 0.35

        return PhaseStoryOut(theta_phase=theta_phase, gamma_slot=cur_role,
                             next_role=next_role, role_vec=r_vec,
                             context_vec=ctx, salience=salience,
                             boundary_reset=boundary_flag)
