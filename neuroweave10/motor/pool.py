"""Спайковый моторный пул (REGENESIS, Законы 1–2).

Заменяет MLP-декодер: `vocab` нейронов с проводимостной мембраной
(exact-step, Закон 3), ALIF-адаптацией порога и OU-шумом. Входы:

  * w_l5   — афференты от слоя L5 коры (AMPA-проводимость + STP),
  * w_hip  — гиппокамп CA1 → апикальный NMDA-вход с Mg²⁺-блоком
             (Jahr-Stevens): без базального драйва гиппокамп лишь
             «праймит» мембрану, спайк не вызывает,
  * w_lap  — медленные шкалы Лапласова банка → сдвиг возбудимости.

Выбор токена — Winner-Take-All: нейрон, выстреливший первым / чаще за
суб-такты такта, и есть выходное слово. Никакого argmax по логитам
обучаемого MLP — решение принимает спайковая гонка.

Закон 2: кратковременная депрессия Tsodyks-Markram на афферентах —
при повторном выстреле нейрона его пресинаптические везикулы истощаются
(x ← x(1−u·x), восстановление τ_D≈200 мс) и эффективность выхода падает
(out ~ 1→0.2 за ~25 мс). Петля «found found found» гасится биофизикой,
а не алгоритмическим вето.

Обучение — трёхфакторное: ошибка нейрона err_i = (i==target)·1 − act_i
(nudging в духе Urbanczik-Senn) × пре-след L5 × f_mod; пластичность
замораживается при аденилатном долге выше порога (Закон 5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import MotorCfg
from ..core.neuron import mg_block


@dataclass
class MotorOut:
    token: int                 # выбранный токен (WTA-победитель)
    spikes: torch.Tensor       # [B,V] спайки моторных нейронов
    v: torch.Tensor            # [B,V] мембранный потенциал
    margin: float              # разрыв V между 1-м и 2-м кандидатом, мВ
    stp_depleted: float        # средний ресурс x на афферентах победителя
    firing_rate: torch.Tensor  # [V] счёт спайков за такт


class MotorPool:
    """Пул спайковых моторных нейронов размером со словарь (Ziwago-Prime).

    Базальный компартмент (AMPA + STP):
      * w_l5   — афференты от слоя L5 коры (могут быть объединены со всех зон)
      * w_lat  — латеральные переходы токен(t-1)/токен(t-2) → токен(t) (STDP последовательности)
      * w_rec_m — быстрая локальная рекуррентность внутри M1
      * w_bg   — дизингибирующий гейт выбора от базальных ганглиев

    Апикальный компартмент (NMDA с Mg²⁺-блоком Jahr-Stevens + Ca²⁺-плато):
      * w_hip  — прайминг от гиппокампа CA1 (эпизодический контекст)
      * w_lap  — клетки времени и медленный спектр Лапласа (история сюжета)
      * w_gw   — бродкаст Global Workspace (сознательный фокус темы)
      * w_wm   — синаптическая рабочая память PFC (SilentWorkingMemory)
    """

    def __init__(self, vocab_size: int, n_l5: int, n_hipp: int,
                 n_lap: int, cfg: MotorCfg, seed: int = 0,
                 n_gw: int = 0, n_wm: int = 0, n_roles: int = 7):
        self.cfg = cfg
        self.vocab, self.n_l5 = vocab_size, n_l5
        self.n_hipp, self.n_lap = n_hipp, n_lap
        self.n_gw, self.n_wm = n_gw, n_wm
        self.n_roles = n_roles
        g = torch.Generator().manual_seed(seed)
        self.w_l5 = torch.randn(vocab_size, n_l5, generator=g) * 0.35
        self.w_hip = torch.randn(vocab_size, n_hipp, generator=g) * 0.20
        self.w_lap = torch.randn(vocab_size, n_lap, generator=g) * 0.15
        self.w_gw = (torch.randn(vocab_size, n_gw, generator=g) * 0.15
                     if n_gw > 0 else None)
        self.w_wm = (torch.randn(vocab_size, n_wm, generator=g) * 0.15
                     if n_wm > 0 else None)
        # Матрица синтаксических ролей (BA44/SMA → M1): какое слово допустимо
        # в какой синтаксической роли предложения (Role 0=Det/Init, Role 1=Subj/Adj,
        # Role 2=Verb/Act, Role 3=Obj, ...). Инициализируется лёгким тормозным
        # фоном (-0.25), чтобы необученные слова не пробивались в чужую роль!
        self.w_role = torch.full((vocab_size, n_roles), -0.25)
        self.w_role_slow = torch.full((vocab_size, n_roles), -0.25)
        # Латеральная матрица переходов токен→токен (STDP последовательностей в M1/SMA):
        # разделена на быстрый след w_lat и долговременный структурный w_lat_slow
        self.w_lat = torch.full((vocab_size, vocab_size), -0.15)
        self.w_lat_slow = torch.full((vocab_size, vocab_size), -0.15)
        # Триграммный контекстный след (совпадение prev2 + prev1 → next)
        self.w_lat2 = torch.full((vocab_size, vocab_size), -0.10)
        self.w_lat2_slow = torch.full((vocab_size, vocab_size), -0.10)
        # Популяционное латеральное ГАМК-торможение синтаксических конкурентов:
        # слова, делящие одну и ту же роль (например, два предлога или два существительных),
        # взаимно тормозят друг друга на следующем шаге (предотвращает "to in", "tree park")
        self.w_pop_inh = torch.zeros(vocab_size, vocab_size)
        # Долговременный базис для кортикальных и апикальных проекций
        self.w_l5_slow = self.w_l5.clone()
        self.w_lap_slow = self.w_lap.clone()
        self.w_hip_slow = self.w_hip.clone()
        # Инновация 4 & 5: Семантический прайминг от Концептуальных Клеток MTL (Concept Cells → M1)
        self.w_concept_m: Optional[torch.Tensor] = None
        self. _last_concept: Optional[torch.Tensor] = None
        # P1: Реципрокная эфферентная проекция M1 → L5 апикальных дендритов коры
        # (обучается локально по правилу Kolen-Pollack без транспонирования весов!)
        self.w_l5_fb = (torch.randn(n_l5, vocab_size, generator=g) * 0.20
                        + 0.30 * self.w_l5.t())
        self.last_l5_error = torch.zeros(n_l5)
        # Структурное здоровье актиновых шипиков в моторной коре (защита от катастрофического забывания):
        # часто подтверждаемые и консолидированные связи (H_ij → 1) защищены от стирания!
        self.lat_spine_health = torch.zeros(vocab_size, vocab_size)
        self.l5_spine_health = torch.zeros(vocab_size, n_l5)
        # Частота встречаемости целевых токенов (гомеостаз синаптического насыщения:
        # сверхчастая пунктуация "," / "." не должна подавлять содержательные слова)
        self.target_freq_ema = torch.full((vocab_size,), 1.0 / max(1, vocab_size))
        # Рекуррентные связи моторной коры (как в реальном M1)
        self.w_rec_m = torch.randn(vocab_size, vocab_size,
                                   generator=g) * 0.05
        self.w_rec_m.fill_diagonal_(0.0)
        # Базальные ганглии → моторный гейт (n_actions каналов выбора)
        self.n_bg = 0
        self.w_bg = None
        # STP Tsodyks-Markram: ресурс x и утилизация u на каждый афферент
        self.stp_x = torch.ones(vocab_size, n_l5)
        self.stp_u = torch.ones(vocab_size, n_l5) * cfg.stp_u0
        # Усталость выхода моторного нейрона (аксональная депрессия)
        self.out_fatigue = torch.ones(vocab_size)
        # Адаптивный STP: частота афферента → глубина депрессии и τ_rec
        self.rate_aff = torch.zeros(n_l5)
        self.rate_post = torch.zeros(vocab_size)
        self.rate_win = torch.zeros(vocab_size)  # медленный Ca2+/sAHP след выбранных слов
        # Следы трёхфакторного обучения
        self.z_pre = torch.zeros(n_l5)
        self.z_post = torch.zeros(vocab_size)
        self.eligibility = torch.zeros(vocab_size, n_l5)
        self.prev_winner = torch.zeros(vocab_size)
        self.prev2_winner = torch.zeros(vocab_size)
        self._last_counts = torch.zeros(vocab_size)
        self._bg_decision: Optional[torch.Tensor] = None
        self._last_hipp: Optional[torch.Tensor] = None
        self._last_lap: Optional[torch.Tensor] = None
        self._last_gw: Optional[torch.Tensor] = None
        self._last_wm: Optional[torch.Tensor] = None
        self._last_role: Optional[torch.Tensor] = None
        # Фаза 2 (snnscience.docx §3.6, Изъян 6): Двухпутевой слогово-фонемный декодер Популяционного Вектора
        from .syllabary_decoder import BGPQSyllabaryDecoder
        self.syllabary_decoder = BGPQSyllabaryDecoder(
            vocab_size=max(vocab_size, 256),
            d_in=n_l5,
            n_head=min(512, max(16, vocab_size)),
            n_syllables=max(16, int(vocab_size ** 0.5) if vocab_size > 256 else 16),
            n_articulators=32,
            seed=seed + 19,
        )

    def attach_basal_ganglia(self, n_actions: int, seed: int = 0) -> None:
        """BG→моторная кора: решение DDM открывает канал действия."""
        g = torch.Generator().manual_seed(seed)
        self.n_bg = n_actions
        self.w_bg = torch.randn(self.vocab, n_actions, generator=g) * 0.15

    def reset_state(self, batch: int, device=None, dtype=torch.float32):
        f = lambda *s: torch.zeros(batch, *s, device=device, dtype=dtype)
        c = self.cfg
        self.v = f(self.vocab).fill_(c.e_l)
        self.v_a = f(self.vocab).fill_(c.e_l)
        self.theta = f(self.vocab).fill_(c.theta0)
        self.refr = f(self.vocab)
        self.ou = f(self.vocab)
        self.counts = f(self.vocab)
        self.prev_winner.zero_()
        self.prev2_winner.zero_()
        self.rate_win.zero_()
        self.out_fatigue.fill_(1.0)
        self.stp_x.fill_(1.0)
        self.stp_u.fill_(c.stp_u0)

    def _align_l5(self, s_l5: torch.Tensor) -> torch.Tensor:
        """Авто-выравнивание размерности входа L5, если подан срез всех зон."""
        if s_l5.shape[-1] == self.n_l5:
            return s_l5
        if s_l5.shape[-1] > self.n_l5:
            return s_l5[..., :self.n_l5]
        return torch.nn.functional.pad(s_l5, (0, self.n_l5 - s_l5.shape[-1]))

    def set_prev_token(self, token_id: int) -> None:
        """Установить текущий входной токен как пресинаптический спайк в M1/SMA."""
        if 0 <= token_id < self.vocab:
            self.prev2_winner = self.prev_winner.clone() * 0.7
            self.prev_winner.zero_()
            self.prev_winner[token_id] = 1.0

    def inferred_word_roles(self, token_id: int) -> Optional[torch.Tensor]:
        """P2: Возвращает выученный синтаксический профиль слова [n_roles] из M1/BA44,
        если у слова уже сформировалась положительная ролевая привязка."""
        if not (0 <= token_id < self.vocab) or token_id <= 1:
            return None
        w_r = torch.relu(0.6 * self.w_role[token_id] + 0.4 * self.w_role_slow[token_id])
        rmax = float(w_r.max().item())
        if rmax <= 0.05:
            return None
        return torch.relu(w_r - 0.55 * rmax).detach()

    # ------------------------------------------------------------------
    def _sub_step(self, s_l5: torch.Tensor, hipp_v: Optional[torch.Tensor],
                  lap_slow: Optional[torch.Tensor], dt: float,
                  ou_on: bool,
                  gw_vec: Optional[torch.Tensor] = None,
                  wm_vec: Optional[torch.Tensor] = None,
                  role_vec: Optional[torch.Tensor] = None,
                  concept_vec: Optional[torch.Tensor] = None) -> None:
        """Один мс интеграции проводимостной мембраны + STP-обновление."""
        c = self.cfg
        s_l5 = self._align_l5(s_l5)
        if not hasattr(self, "v_a") or self.v_a.shape != self.v.shape:
            self.v_a = torch.full_like(self.v, c.e_l)

        # 2. Апикальный контекст (NMDA с 20% AMPA-поддержкой + Mg²⁺-блок):
        # рассчитывается ПЕРЕД базальным залпом, чтобы апикальное плато
        # избирательно открывало ворота только контекстно- и ролево-верному слову!
        g_apical_raw = torch.zeros_like(self.v)
        if hipp_v is not None:
            hv = hipp_v if hipp_v.dim() == 2 else hipp_v.unsqueeze(0)
            if hv.shape[-1] == self.w_hip.shape[1]:
                g_apical_raw = g_apical_raw + (hv @ self.w_hip.t()).clamp(min=0.0)
        if concept_vec is not None:
            cv = concept_vec if concept_vec.dim() == 2 else concept_vec.unsqueeze(0)
            if self.w_concept_m is None or self.w_concept_m.shape[1] != cv.shape[-1]:
                g_c = torch.Generator().manual_seed(23)
                self.w_concept_m = torch.randn(self.vocab, cv.shape[-1], generator=g_c) * 0.15
            c_gate = torch.ones_like(self.v)
            if self.prev_winner.any():
                w_lat_c = 0.5 * self.w_lat + 0.5 * self.w_lat_slow
                lat_c = self.prev_winner.unsqueeze(0) @ w_lat_c.t()
                if float(lat_c.max()) > 0.08:
                    # Апикальный концептуальный прайминг усиливает только реально выученные переходы (lat_c > 0.08)
                    c_gate = (lat_c > 0.08).float() * torch.sigmoid((lat_c - 0.05) * 10.0)
            c_proj = (cv.to(self.v.device) @ self.w_concept_m.to(self.v.device).t()).clamp(min=0.0) * c_gate
            g_apical_raw = g_apical_raw + c_proj * 0.85
        else:
            c_proj = None
        th_shift = torch.zeros_like(self.v)
        if c_proj is not None:
            th_shift = th_shift + (c_proj * 1.25).clamp(max=3.2)
        if lap_slow is not None:
            lp = lap_slow if lap_slow.dim() == 2 else lap_slow.unsqueeze(0)
            if lp.shape[-1] == self.w_lap.shape[1]:
                lap_proj = lp @ self.w_lap.t()
                g_apical_raw = g_apical_raw + lap_proj.clamp(min=0.0) * 1.2
                th_shift = th_shift + lap_proj.clamp(min=-4.0, max=4.0)
        if gw_vec is not None and self.w_gw is not None:
            gv = gw_vec if gw_vec.dim() == 2 else gw_vec.unsqueeze(0)
            if gv.shape[-1] == self.w_gw.shape[1]:
                g_apical_raw = g_apical_raw + (gv @ self.w_gw.t()).clamp(min=0.0) * 1.0
        if wm_vec is not None and self.w_wm is not None:
            wv = wm_vec if wm_vec.dim() == 2 else wm_vec.unsqueeze(0)
            if wv.shape[-1] == self.w_wm.shape[1]:
                g_apical_raw = g_apical_raw + (wv @ self.w_wm.t()).clamp(min=0.0) * 1.2
        role_gate = torch.ones_like(self.v)
        if role_vec is not None:
            rv = role_vec if role_vec.dim() == 2 else role_vec.unsqueeze(0)
            if rv.shape[-1] == self.w_role.shape[1]:
                w_role_eff = 0.5 * self.w_role + 0.5 * self.w_role_slow
                role_proj = rv @ w_role_eff.t()
                g_apical_raw = g_apical_raw + role_proj.clamp(min=0.0) * 1.1
                th_shift = th_shift + role_proj.clamp(min=-3.5, max=3.5)
                if float(role_proj.max()) > 0.0:
                    role_gate = torch.sigmoid((role_proj + 0.1) * 5.0)

        # --- Узел 5: Активный Вывод (Active Inference — Ожидаемая Свободная Энергия G_a) ---
        # Когда высока эпистемическая неопределённость (BA10) или гипоталамическая скука
        # и нет аверсивной угрозы от Амигдалы, в апикальный дендрит подаётся
        # грамматически-допустимый эпистемический ток любопытства I_info:
        explore_drive = getattr(self, "_explore_drive", 0.0)
        g_epist = torch.zeros_like(self.v)
        habit_suppress = torch.ones_like(self.v)
        if explore_drive > 0.01:
            # Редкость слова: максимальна у редких содержательных слов (freq < средней)
            # и равна 0 у частых слов и пунктуации ('.', ',', 'the', 'a')!
            freq_rel = self.target_freq_ema * self.vocab
            rarity = (1.8 - freq_rel).clamp(0.0, 1.0)
            freshness = (1.0 - self.rate_win.clamp(0.0, 0.95))
            w_lat_eff = 0.5 * self.w_lat + 0.5 * self.w_lat_slow
            if self.prev_winner.any():
                # Исследуем ТОЛЬКО те переходы, которые реально встречались в опыте (w_lat > 0.05),
                # но были редкими (например, "the -> frog", "the -> knight", "frog -> jumped")!
                lat_valid = ((self.prev_winner.unsqueeze(0) @ w_lat_eff.t()) > 0.05).float().squeeze(0)
            else:
                lat_valid = torch.ones(self.vocab, device=self.v.device)
            novelty_pot = rarity * freshness * lat_valid
            novelty_pot[:2] = 0.0  # никогда не исследовать <pad> / <unk>
            g_epist = explore_drive * 9.0 * novelty_pot.unsqueeze(0) * role_gate
            g_apical_raw = g_apical_raw + g_epist
            th_shift = th_shift + (g_epist * 1.8).clamp(max=5.5)
            # При высокой скуке (explore_drive > 0.35) норадреналин/ацетилхолин приглушают
            # приевшиеся сверхчастые шаблоны (habit_suppress), открывая дорогу редким ветвям!
            is_overlearned = (freq_rel > 1.8).float().unsqueeze(0)
            habit_suppress = 1.0 - min(0.75, explore_drive * 0.85) * is_overlearned

        # Интеграция апикального дендрита с Jahr-Stevens блоком (быстрая релаксация)
        b_v = mg_block(self.v_a)
        eff_nmda = 0.35 + 0.65 * b_v
        g_apical = g_apical_raw * eff_nmda * self.out_fatigue.unsqueeze(0)
        v_a_inf = (c.g_l * c.e_l + g_apical * c.e_exc) / (c.g_l + g_apical).clamp(min=1e-6)
        self.v_a = v_a_inf + (self.v_a - v_a_inf) * torch.exp(-dt * (c.g_l + g_apical) / 2.5)
        # Апикальное плато (Larkum BAC-firing): деполяризованный апикальный дендрит
        # снижает порог сомы и усиливает базальную передачу (коинциденция!)
        apical_gate = torch.sigmoid((self.v_a - (-55.0)) / 8.0)
        apical_plateau = 5.0 * apical_gate
        fat_gate = (0.35 + 0.65 * self.out_fatigue).unsqueeze(0)
        th_shift = ((th_shift + apical_plateau) * fat_gate).clamp(-6.0, 6.0)
        if self.prev_winner.any():
            th_shift = th_shift * (1.0 - self.prev_winner.unsqueeze(0)) - 6.0 * self.prev_winner.unsqueeze(0)

        # 1. Базальный драйв (Инновация 5: Когнитивно-Кортикальный Приоритет L5 + Broca-BG Ролевой Гейт):
        w_l5_tot = 0.5 * self.w_l5 + 0.5 * self.w_l5_slow
        w_eff = w_l5_tot * self.stp_x * self.stp_u
        g_l5 = (s_l5 @ w_eff.t()).clamp(min=0.0) * 1.15 * (0.80 + 0.20 * role_gate) * fat_gate
        g_lat_inh = torch.zeros_like(self.v)
        synfire_gate = torch.ones_like(self.v)
        if self.prev_winner.any():
            w_lat_eff = 0.5 * self.w_lat + 0.5 * self.w_lat_slow
            lat_raw = self.prev_winner.unsqueeze(0) @ w_lat_eff.t()
            # Если у текущего слова prev_winner уже есть выученные последователи (max > 0.05),
            # включается синаптический synfire-гейт: слова с невыученным/отрицательным переходом (lat_raw <= 0)
            # шунтируются и на базальном, и на апикальном дендрите, и по порогу th_shift!
            if float(lat_raw.max()) > 0.05:
                synfire_gate = (lat_raw > 0.02).float() * torch.sigmoid((lat_raw + 0.05) * 8.0)
                g_lat_inh = g_lat_inh + (1.0 - synfire_gate) * 2.5
                th_shift = th_shift * synfire_gate - 3.0 * (1.0 - synfire_gate)
            g_lat = lat_raw.clamp(min=0.0) * 3.8 * habit_suppress
            g_l5 = (g_l5 * (0.35 + 0.65 * synfire_gate) * habit_suppress
                    + g_lat * (0.75 + 0.50 * apical_gate))
        if self.prev2_winner.any():
            w_lat2_eff = 0.5 * self.w_lat2 + 0.5 * self.w_lat2_slow
            lat2_raw = self.prev2_winner.unsqueeze(0) @ w_lat2_eff.t()
            # Триграммный драйв усиливает слова, которые подтверждены контекстом prev2 -> prev1 -> target,
            # и шунтирует ложные продолжения при совпадении опорного слова (например, 'to' в SVO vs OVS, или 'to -> the -> castle' vs 'the -> knight')!
            if float(lat2_raw.max()) > 0.08:
                lat2_gate = (lat2_raw > 0.02).float() * torch.sigmoid((lat2_raw + 0.05) * 8.0)
                g_l5 = g_l5 * (0.10 + 0.90 * lat2_gate) + lat2_raw.clamp(min=0.0) * 4.2 * synfire_gate
                g_apical = g_apical * (0.15 + 0.85 * lat2_gate)
                g_lat_inh = g_lat_inh + (1.0 - lat2_gate) * 3.5
                th_shift = th_shift * (0.20 + 0.80 * lat2_gate) - 2.5 * (1.0 - lat2_gate) + (lat2_raw.clamp(min=0.0) * 2.8 * synfire_gate).clamp(max=3.8)
            else:
                g_l5 = g_l5 + lat2_raw.clamp(min=0.0) * 3.2 * synfire_gate
                th_shift = th_shift + (lat2_raw.clamp(min=0.0) * 2.2 * synfire_gate).clamp(max=3.5)
        g_l5 = g_l5 * self.out_fatigue.unsqueeze(0)
        g_apical = g_apical * (0.30 + 0.70 * synfire_gate)

        # Рекуррентный контур моторной коры
        if self._last_counts is not None:
            g_l5 = g_l5 + (self._last_counts.unsqueeze(0)
                           @ self.w_rec_m.t()).clamp(min=0.0) * 0.15
        # Базальные ганглии: решение DDM открывает канал действия
        if self.w_bg is not None and self._bg_decision is not None:
            g_l5 = g_l5 + (self._bg_decision.unsqueeze(0)
                           @ self.w_bg.t()).clamp(min=0.0) * 0.5

        # OU-шум мембран (Закон, шаг 5): полный при обучении, базовый тонус
        # фоновой синаптической бомбардировки при свободной генерации
        ou_scale = c.sigma_ou if ou_on else c.sigma_ou * 0.35
        self.ou = self.ou * (1.0 - dt / 10.0) + torch.randn_like(
            self.v) * (ou_scale * (dt / 10.0) ** 0.5)
        # Глобальное делительное торможение: WTA-пул PV + самоторможение входа + prev2
        # + популяционное торможение слов той же синтаксической роли (w_pop_inh)
        g_exc = g_l5 + 0.75 * g_apical
        g_wta = c.wta_inh * g_exc.mean(dim=1, keepdim=True)
        g_inh = g_wta.expand_as(g_l5) * 0.25 + g_lat_inh
        if self.prev_winner.any():
            # Клетки Реншоу: подавление немедленного самоповтора a->a
            # и торможение слов из той же синтаксической популяции (Noun->Noun, Prep->Prep)
            pop_inh = (self.prev_winner.unsqueeze(0) @ self.w_pop_inh.t()).clamp(min=0.0)
            g_inh = g_inh + self.prev_winner.unsqueeze(0) * 3.5 + pop_inh * 1.2
        if self.prev2_winner.any():
            # Подавление короткого пинг-понга a->b->a
            g_inh = g_inh + self.prev2_winner.unsqueeze(0) * 0.8
        # Exact-step (Закон 3): V_inf = (Σ g·E + I)/Σg
        g_tot = c.g_l + g_exc + g_inh
        v_inf = (c.g_l * c.e_l + g_exc * c.e_exc
                 + g_inh * c.e_cl + self.ou) / g_tot.clamp(min=1e-6)
        tau_eff = c.tau_m / g_tot.clamp(min=1e-6)
        self.v = v_inf + (self.v - v_inf) * torch.exp(-dt / tau_eff)
        # Спайк: порог + рефрактерность; при BAC-коинциденции (базальный спайк +
        # апикальное плато > 0.5) пирамидальный нейрон выдаёт дуплет/бёрст спайков!
        theta_eff = self.theta - th_shift
        s = (self.v > theta_eff).float() * (self.refr <= 0).float()
        bac_burst = s * (1.0 + apical_gate)
        self.v = self.v - s * (self.theta - c.e_l) * 0.5
        self.refr = (self.refr - 1.0).clamp(min=0.0) + s * c.refractory
        self.theta = self.theta + c.beta_adapt * s \
            - dt * (self.theta - c.theta0) / c.tau_adapt
        self.counts += bac_burst
        # Адаптивный STP Tsodyks-Markram
        self.rate_aff = self.rate_aff * 0.98 + s_l5.mean(0) * 0.02
        self.rate_post = self.rate_post * 0.98 + s.mean(0) * 0.02
        u_eff = (c.stp_u0 * (1.0 + 2.0 * self.rate_aff)).clamp(max=0.95)
        tau_d_eff = c.stp_tau_d / (1.0 + 3.0 * self.rate_aff)  # → ~25-30 мс
        act = s_l5.mean(0).clamp(0.0, 1.0)                   # [n_l5]
        u_rel = u_eff.unsqueeze(0) * self.stp_x              # [V,n_l5]
        self.stp_x = (self.stp_x - u_rel * act.unsqueeze(0)
                      ).clamp(min=0.0)
        self.stp_u = (self.stp_u
                      + u_eff.unsqueeze(0) * (1.0 - self.stp_u)
                      * act.unsqueeze(0)).clamp(max=1.0)
        self.stp_x += (1.0 - self.stp_x) * (dt / tau_d_eff.unsqueeze(0))
        self.stp_u += (c.stp_u0 - self.stp_u) * (dt / c.stp_tau_f)
        # Усталость выхода: спайк тратит ресурс, восстановление на интервале слова
        self.out_fatigue = (self.out_fatigue
                            - 0.25 * (1.0 + self.rate_post)
                            * s.mean(0)).clamp(min=0.10)
        self.out_fatigue += (1.0 - self.out_fatigue) * (
            dt / max(5.0, c.out_tau * 0.5))

    # ------------------------------------------------------------------
    def set_bg_decision(self, action: Optional[torch.Tensor],
                        n_actions: int = 0) -> None:
        """Решение BG-DDM как one-hot канал в моторный пул."""
        if action is None or self.w_bg is None:
            self._bg_decision = None
            return
        a = int(action.flatten()[0].item()) if torch.is_tensor(action)             else int(action)
        if a < 0:
            self._bg_decision = None
            return
        oh = torch.zeros(self.w_bg.shape[1])
        oh[min(a, self.w_bg.shape[1] - 1)] = 1.0
        self._bg_decision = oh

    def set_active_inference(self, epistemic: float = 0.0,
                             boredom: float = 0.0,
                             aversive: float = 0.0) -> float:
        """Узел 5 (Термостат Активного Вывода):
        Вычисляет коэффициент исследовательского драйва κ_explore ∈ [0, 1]
        из эпистемической неопределённости BA10 и гипоталамической скуки,
        подавляемый аверсивной тревогой Амигдалы."""
        raw = 0.8 * max(0.0, float(epistemic)) + 0.9 * max(0.0, float(boredom)) \
            - 1.2 * max(0.0, float(aversive))
        self._explore_drive = float(max(0.0, min(1.0, raw)))
        return self._explore_drive

    def top_candidates_for_vte(self, k: int = 2) -> list[int]:
        """Инновация 1 (Театр Сознания / VTE): возвращает k лучших конкурирующих
        моторных кандидатов до финального вокального сброса."""
        v_ref = getattr(self, "v_pre_reset", self.v).mean(0)
        score = (self._last_counts + 0.05 * (v_ref - self.cfg.e_l)) * self.out_fatigue - 2.2 * self.rate_win
        if self.prev_winner.any():
            w_lat_eff = 0.5 * self.w_lat + 0.5 * self.w_lat_slow
            lat_raw = self.prev_winner @ w_lat_eff.t()
            score = score + 1.2 * lat_raw - 3.0 * (lat_raw <= 0.0).float()
        if self.prev2_winner.any():
            w_lat2_eff = 0.5 * self.w_lat2 + 0.5 * self.w_lat2_slow
            lat2_raw = self.prev2_winner @ w_lat2_eff.t()
            if float(lat2_raw.max()) > 0.05:
                score = score + 2.2 * lat2_raw - 4.0 * (lat2_raw <= 0.02).float()
        if self._last_role is not None and self._last_role.numel() == self.w_role.shape[1]:
            w_role_eff = 0.5 * self.w_role + 0.5 * self.w_role_slow
            role_raw = w_role_eff @ self._last_role
            score = score + 0.8 * role_raw
        if (
            getattr(self, "_last_concept", None) is not None
            and self.w_concept_m is not None
            and self._last_concept.numel() == self.w_concept_m.shape[1]
        ):
            c_proj = torch.relu(self.w_concept_m @ self._last_concept)
            if self.prev_winner.any():
                w_lat_eff = 0.5 * self.w_lat + 0.5 * self.w_lat_slow
                c_proj = c_proj * ((self.prev_winner @ w_lat_eff.t()) > 0.05).float()
            score = score + 1.15 * c_proj
        score[:2] = -1e6  # исключаем <pad> и <unk> из мысленного соревнования
        kk = min(max(1, k), self.vocab)
        return [int(i) for i in score.topk(kk).indices.tolist()]

    def step(self, s_l5: torch.Tensor,
             hipp_v: Optional[torch.Tensor] = None,
             lap_slow: Optional[torch.Tensor] = None,
             dt: float = 1.0, ou_on: bool = True,
             gw_vec: Optional[torch.Tensor] = None,
             wm_vec: Optional[torch.Tensor] = None,
             role_vec: Optional[torch.Tensor] = None,
             concept_vec: Optional[torch.Tensor] = None) -> MotorOut:
        """Токен-шаг: sub_ticks мс интеграции + адаптивное накопление спайков
        до порога доказательства (evidence threshold) → WTA-победитель."""
        c = self.cfg
        s_l5 = self._align_l5(s_l5)
        self._last_hipp = (hipp_v.detach().mean(0) if hipp_v is not None and hipp_v.dim() == 2
                           else (hipp_v.detach() if hipp_v is not None else None))
        self._last_lap = (lap_slow.detach().mean(0) if lap_slow is not None and lap_slow.dim() == 2
                          else (lap_slow.detach() if lap_slow is not None else None))
        self._last_gw = (gw_vec.detach().mean(0) if gw_vec is not None and gw_vec.dim() == 2
                         else (gw_vec.detach() if gw_vec is not None else None))
        self._last_wm = (wm_vec.detach().mean(0) if wm_vec is not None and wm_vec.dim() == 2
                         else (wm_vec.detach() if wm_vec is not None else None))
        self._last_role = (role_vec.detach().mean(0) if role_vec is not None and role_vec.dim() == 2
                           else (role_vec.detach() if role_vec is not None else None))
        self._last_concept = (concept_vec.detach().mean(0) if concept_vec is not None and concept_vec.dim() == 2
                              else (concept_vec.detach() if concept_vec is not None else None))

        self.counts = torch.zeros_like(self.v)
        for _ in range(c.sub_ticks):
            self._sub_step(s_l5, hipp_v, lap_slow, dt, ou_on,
                           gw_vec=gw_vec, wm_vec=wm_vec, role_vec=role_vec,
                           concept_vec=concept_vec)
        # Если за базовые sub_ticks ещё не набралось достаточно спайкового
        # доказательства (evidence < 1.5 спайка), продолжаем интеграцию ещё до 3 мс
        extra = 0
        while float(self.counts.max()) < 1.5 and extra < 3:
            self._sub_step(s_l5, hipp_v, lap_slow, dt, ou_on,
                           gw_vec=gw_vec, wm_vec=wm_vec, role_vec=role_vec,
                           concept_vec=concept_vec)
            extra += 1
        counts = self.counts.mean(0)                     # [V]
        renshaw_mask = (1.0 - 1.0 * self.prev_winner - 0.70 * self.prev2_winner).clamp(0.0, 1.0)
        ahp_fatigue = (1.0 - 0.85 * self.rate_win.clamp(0.0, 0.95))
        if float(counts.max()) > 0.0:
            # При равенстве числа спайков побеждает нейрон с более высокой мембраной V,
            # с торможением Реншоу и Ca2+/sAHP-адаптацией против зацикливания аттракторов!
            score = (counts * ahp_fatigue + 1e-3 * (self.v.mean(0) - c.e_l)) * renshaw_mask - 10.0 * self.prev_winner
            token = int(score.argmax().item())           # WTA по спайкам
        else:
            v_score = (self.v.mean(0) - c.e_l) * renshaw_mask * ahp_fatigue - 25.0 * self.prev_winner - 8.0 * self.prev2_winner
            token = int(v_score.argmax().item())         # гонка мембран
        # Сохраняем распределение потенциалов ДО ресета победителя,
        # чтобы margin и диагностика видели реального лидера гонки!
        vv_pre = self.v.mean(0).clone()
        self.v_pre_reset = self.v.clone()
        top2 = vv_pre.topk(min(2, self.vocab)).values
        margin = float(top2[0] - (top2[1] if top2.numel() > 1
                                  else top2[0] - 1.0))

        # Адаптация победителя: максимальная деполяризация → максимум
        # Ca²⁺-притока → sAHP-порог растёт и без спайка (повторный
        # выбор того же нейрона или короткого 4-токенного аттрактора сам себя тормозит).
        self.rate_win.mul_(0.92)
        self.rate_win[token] += 0.25
        self.theta[:, token] += c.beta_adapt * (1.8 + 4.0 * float(self.rate_win[token]))
        # Ресурс терминала победителя истощается при каждом выборе —
        # даже в гонке мембран без спайка: выбранный канал сам себя
        # глушит на следующие шаги (рефрактерность решения).
        self.out_fatigue[token] = (self.out_fatigue[token] * 0.40)\
            .clamp(min=0.04)
        # Пост-выборный ресет: выбор канала = макро-спайк; мембрана
        # уходит к E_L — абсолютная рефрактерность решения, нейрон
        # не может выиграть следующий такт без нового драйва.
        self.v[:, token] = c.e_l
        self._last_counts = counts
        return MotorOut(token=token,
                        spikes=(self.counts > 0).float(),
                        v=self.v_pre_reset, margin=margin,
                        stp_depleted=float(self.stp_x[token].mean()),
                        firing_rate=counts)

    # ------------------------------------------------------------------
    def learn(self, target: int, s_l5: torch.Tensor, f_mod: float,
              plasticity_ok: bool = True, reward: float = 0.0,
              dt: float = 1.0) -> float:
        """Трёхфакторное обновление w_l5 + STDP последовательностей w_lat
        + синтаксических ролей w_role + апикальных синапсов контекста."""
        if not plasticity_ok or f_mod <= 0.0:
            return 0.0
        s_l5 = self._align_l5(s_l5)
        rate = self._last_counts / max(1, self.cfg.sub_ticks)
        # Если в такте ещё не было надпороговых спайков, используем
        # подпороговое предсказание мембраны φ(V_pre_reset), как в Urbanczik-Senn
        v_ref = getattr(self, "v_pre_reset", self.v).mean(0)
        soft_pred = torch.softmax((v_ref - self.theta.mean(0)) * 0.4, dim=0)
        act_pred = 0.5 * rate + 0.5 * soft_pred

        err = torch.zeros(self.vocab)
        err[target] = 1.0
        err = err - act_pred

        x = s_l5.mean(0)                                 # пре-след L5
        x_eff = 0.7 * x + 0.3 * self.z_pre
        self.eligibility = self.eligibility * (1.0 - dt / 500.0) \
            + torch.outer(err, x_eff)

        # Гомеостаз синаптического насыщения: отслеживаем частоту целевых слов;
        # сверхчастые токены (>5x от средней частоты: запятые/точки) получают
        # мягкое демпфирование, а обычные слова и существительные — полный гейн!
        self.target_freq_ema.mul_(0.998)
        self.target_freq_ema[target] += 0.002
        freq_ratio = float(self.target_freq_ema[target]) * self.vocab
        freq_balance = 1.0 / (1.0 + 0.18 * max(0.0, freq_ratio - 5.0))

        unk_scale = 0.08 if target == 1 else freq_balance
        surprise_gain = (1.0 + 0.8 * abs(reward)) * unk_scale
        eta = self.cfg.eta_motor * 8.0 * max(0.25, min(2.5, f_mod)) * surprise_gain
        dw = eta * (torch.outer(err, x_eff) + 0.25 * self.eligibility)
        # Метапластическая защита зрелых шипиков L5 от стирания (Fusi et al. 2005)
        l5_shield = torch.where(dw < 0, (1.0 - 0.85 * self.l5_spine_health).clamp(0.10, 1.0),
                                torch.ones_like(self.l5_spine_health))
        self.w_l5 = (self.w_l5 + dw * (0.4 + 0.6 * l5_shield)).clamp(-2.5, 2.5)
        self.w_l5_slow = (self.w_l5_slow + 0.08 * dw * l5_shield).clamp(-2.5, 2.5)
        if target != 1:
            self.l5_spine_health[target] = (self.l5_spine_health[target]
                                            + 0.04 * x_eff.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        # P1: Локальная реципрокная пластичность Kolen-Pollack для обратной связи M1 → L5
        # и формирование эфферентного вектора ошибки для апикальных дендритов коры!
        if not hasattr(self, "w_l5_fb") or self.w_l5_fb.shape != (self.n_l5, self.vocab):
            self.w_l5_fb = 0.5 * self.w_l5.t().clone()
        self.w_l5_fb = (0.999 * self.w_l5_fb + 0.35 * dw.t()).clamp(-2.5, 2.5)
        self.last_l5_error = torch.tanh(self.w_l5_fb @ err)

        # STDP обучения переходов между словами в моторной коре (w_lat, w_lat2):
        # если токен A предшествовал целевому токену B, синапс A→B потенцируется,
        # а синапс A→ошибочный_спайковый_победитель депрессируется (молчащие альтернативные
        # ветви ветвления не стираются — Bi & Poo 1998)!
        err_lat = torch.zeros(self.vocab)
        err_lat[target] = 1.0
        err_lat = err_lat - (0.85 * rate + 0.10 * soft_pred)
        if self.prev_winner.any() and target != 1:
            # Укрепляем актиновый каркас шипика подтверждённого перехода prev_winner → target
            self.lat_spine_health[target] = (
                self.lat_spine_health[target] + 0.14 * self.prev_winner
            ).clamp(0.0, 1.0)
            dw_lat = eta * 3.0 * torch.outer(err_lat, self.prev_winner)
            lat_shield = torch.where(dw_lat < 0,
                                     (1.0 - 0.92 * self.lat_spine_health).clamp(0.04, 1.0),
                                     torch.ones_like(self.lat_spine_health))
            self.w_lat = (self.w_lat + dw_lat * (0.30 + 0.70 * lat_shield)).clamp(-2.5, 3.0)
            # Подтверждённые опытом ветви ветвления (lat_spine_health > 0) сохраняют высокую базовую проводимость!
            branch_floor = 0.85 * self.lat_spine_health
            self.w_lat = torch.maximum(self.w_lat, branch_floor)
            self.w_lat.fill_diagonal_(-1.5)  # анти-автоповтор
            self.w_lat[1, :] = -1.5          # подавление перехода в <unk>
            self.w_lat_slow = (self.w_lat_slow + 0.15 * dw_lat * lat_shield).clamp(-2.5, 3.0)
            self.w_lat_slow = torch.maximum(self.w_lat_slow, 0.75 * self.lat_spine_health)
            self.w_lat_slow.fill_diagonal_(-1.5)
            self.w_lat_slow[1, :] = -1.5
        if self.prev2_winner.any() and target != 1:
            if not hasattr(self, "lat2_spine_health") or self.lat2_spine_health.shape != self.w_lat2.shape:
                self.lat2_spine_health = torch.zeros_like(self.w_lat2)
            self.lat2_spine_health[target] = (
                self.lat2_spine_health[target] + 0.14 * (self.prev2_winner > 0).float()
            ).clamp(0.0, 1.0)
            dw_lat2 = eta * 1.8 * torch.outer(err_lat, self.prev2_winner)
            lat2_shield = torch.where(dw_lat2 < 0,
                                      (1.0 - 0.92 * self.lat2_spine_health).clamp(0.04, 1.0),
                                      torch.ones_like(self.lat2_spine_health))
            self.w_lat2 = (self.w_lat2 + dw_lat2 * lat2_shield).clamp(-2.0, 2.5)
            self.w_lat2 = torch.maximum(self.w_lat2, 0.75 * self.lat2_spine_health)
            self.w_lat2[1, :] = -1.5
            self.w_lat2_slow = (self.w_lat2_slow + 0.15 * dw_lat2 * lat2_shield).clamp(-2.0, 2.5)
            self.w_lat2_slow = torch.maximum(self.w_lat2_slow, 0.65 * self.lat2_spine_health)
            self.w_lat2_slow[1, :] = -1.5

        # Обучение синтаксических ролей (w_role): жёсткая привязка слова к его
        # позиции/роли в предложении (Субъект vs Глагол vs Прилагательное vs Объект)
        if (self._last_role is not None
                and self._last_role.numel() == self.w_role.shape[1]
                and target != 1):
            if not hasattr(self, "role_spine_health") or self.role_spine_health.shape != self.w_role.shape:
                self.role_spine_health = torch.zeros_like(self.w_role)
            self.role_spine_health[target] = (
                self.role_spine_health[target] + 0.14 * self._last_role.clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
            dw_role = eta * 2.2 * torch.outer(err_lat, self._last_role)
            role_shield = torch.where(dw_role < 0,
                                      (1.0 - 0.92 * self.role_spine_health).clamp(0.04, 1.0),
                                      torch.ones_like(self.role_spine_health))
            self.w_role = (self.w_role + dw_role * role_shield).clamp(-2.5, 2.5)
            self.w_role = torch.maximum(self.w_role, 0.50 * self.role_spine_health)
            self.w_role[1, :] = -1.5
            self.w_role_slow = (self.w_role_slow + 0.12 * dw_role * role_shield).clamp(-2.5, 2.5)
            self.w_role_slow = torch.maximum(self.w_role_slow, 0.40 * self.role_spine_health)
            self.w_role_slow[1, :] = -1.5

        # Обучение апикальных контекстных проекций (Лаплас, Гиппокамп, GW, WM)
        if self._last_lap is not None and self._last_lap.numel() == self.w_lap.shape[1]:
            dw_lap = eta * 1.5 * torch.outer(err, self._last_lap)
            self.w_lap = (self.w_lap + dw_lap).clamp(-2.0, 2.0)
            self.w_lap_slow = (self.w_lap_slow + 0.10 * dw_lap).clamp(-2.0, 2.0)
        if self._last_hipp is not None and self._last_hipp.numel() == self.w_hip.shape[1]:
            dw_hip = eta * 1.2 * torch.outer(err, self._last_hipp)
            self.w_hip = (self.w_hip + dw_hip).clamp(-2.0, 2.0)
            self.w_hip_slow = (self.w_hip_slow + 0.10 * dw_hip).clamp(-2.0, 2.0)
        if (self.w_gw is not None and self._last_gw is not None
                and self._last_gw.numel() == self.w_gw.shape[1]):
            self.w_gw = (self.w_gw + eta * 1.2 * torch.outer(
                err, self._last_gw)).clamp(-2.0, 2.0)
        if (self.w_wm is not None and self._last_wm is not None
                and self._last_wm.numel() == self.w_wm.shape[1]):
            self.w_wm = (self.w_wm + eta * 1.2 * torch.outer(
                err, self._last_wm)).clamp(-2.0, 2.0)
        if (self.w_concept_m is not None and self._last_concept is not None
                and self._last_concept.numel() == self.w_concept_m.shape[1]
                and target != 1):
            c_err = err_lat.clone()
            c_err[:5] = 0.0  # исключаем <pad>, <unk> и артикли (the, a, an) из концептуальных якорей
            self.w_concept_m = (self.w_concept_m + eta * 1.1 * torch.outer(
                c_err, self._last_concept.to(err.device))).clamp(-2.0, 2.0)

        self.z_pre = self.z_pre * (1.0 - dt / 20.0) + x
        self.z_post = self.z_post * (1.0 - dt / 20.0) + rate
        if hasattr(self, "syllabary_decoder") and self.syllabary_decoder is not None:
            self.syllabary_decoder.learn(target, x, lr=float(eta), da_rpe=max(0.1, float(reward)))
        return float(dw.abs().mean())

    def consolidate(self, factor: float = 0.35) -> None:
        """Сонная консолидация моторной коры: перенос быстрых весов в структурные
        с актиновой защитой зрелых шипиков (lat_spine_health) от забывания."""
        # Зрелые шипики (lat_spine_health > 0.5) удерживают свой выученный положительный вес
        # и принимают только конструктивное усиление во сне!
        lat_delta = self.w_lat - self.w_lat_slow
        lat_guard = torch.where(lat_delta < 0,
                                (1.0 - 0.85 * self.lat_spine_health).clamp(0.10, 1.0),
                                torch.ones_like(self.lat_spine_health))
        self.w_lat_slow = (self.w_lat_slow + factor * lat_delta * lat_guard).clamp(-2.5, 3.0)
        # Укрепляем здоровье всех положительно консолидированных переходов
        self.lat_spine_health = (self.lat_spine_health
                                 + 0.15 * (self.w_lat_slow > 0.20).float()).clamp(0.0, 1.0)

        l5_delta = self.w_l5 - self.w_l5_slow
        l5_guard = torch.where(l5_delta < 0,
                               (1.0 - 0.80 * self.l5_spine_health).clamp(0.15, 1.0),
                               torch.ones_like(self.l5_spine_health))
        self.w_l5_slow = (self.w_l5_slow + factor * l5_delta * l5_guard).clamp(-2.5, 2.5)
        self.w_lat2_slow = ((1.0 - factor) * self.w_lat2_slow + factor * self.w_lat2).clamp(-2.0, 2.5)
        self.w_role_slow = ((1.0 - factor) * self.w_role_slow + factor * self.w_role).clamp(-2.5, 2.5)
        self.w_lap_slow = ((1.0 - factor) * self.w_lap_slow + factor * self.w_lap).clamp(-2.0, 2.0)
        self.w_hip_slow = ((1.0 - factor) * self.w_hip_slow + factor * self.w_hip).clamp(-2.0, 2.0)
        # Слова с близким положительным профилем синтаксических ролей (одна часть речи)
        # формируют взаимное ГАМК-торможение (если между ними нет прямого STDP-перехода w_lat > 0.2)
        r_pos = torch.relu(self.w_role_slow + 0.25)
        r_norm = r_pos / r_pos.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sim = (r_norm @ r_norm.t()).clamp(0.0, 1.0)
        allowed = (self.w_lat_slow > 0.15).float()
        self.w_pop_inh = (sim * (1.0 - allowed)).clamp(0.0, 1.0)
        self.out_fatigue.fill_(1.0)
        self.stp_x.fill_(1.0)
        if hasattr(self, "syllabary_decoder") and self.syllabary_decoder is not None:
            self.syllabary_decoder.consolidate_habit()
