"""Базальные ганглии: Drift-Diffusion гейт эмиссии токенов с TD Critic и Go/NoGo путями (Genesis basal_ganglia.py).

Биологический моторный контур:
1. Прямой путь Go (D1-MSN) и непрямой путь NoGo (D2-MSN):
   - w_Go стимулирует выбор токена, w_NoGo подавляет выбор (моторный тормоз).
2. TD Critic (Genesis):
   - При эмиссии токена фиксируется ожидание (pending_value).
   - Если в последующие такты возникает зацикливание (повторение того же токена, пунктуационный спам),
     критик генерирует отрицательный сигнал (dip: delta < 0).
   - Dip сбрасывает накопление DDM (evidence_acc = 0), усиливает NoGo-тормоз w_NoGo[token]
     и заставляет моторный контур немедленно переключиться на альтернативные семантические токены.
3. Серотонин (5-HT) и Ацетилхолин (ACh):
   - Высокий 5-HT повышает порог DDM (терпение, контроль импульсивности).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sampling import sample_token


class BasalGangliaGate(nn.Module):
    """Дрифт-диффузионный гейт базальных ганглиев с TD-критиком и Go/NoGo путями (Genesis)."""

    def __init__(
        self,
        dt: float = 1.0,
        base_threshold: float = 0.55,
        leak_rate: float = 0.05,
        alpha_5ht: float = 0.60,
        alpha_ach: float = 0.40,
        stn_conflict_gain: float = 0.35,
        evidence_gain: float = 8.0,
        sigma: float = 0.35,
        alpha_V: float = 0.05,
        nogo_gain: float = 2.5,
        nogo_veto_threshold: float = float("inf"),
    ):
        super().__init__()
        self.dt = float(dt)
        self.base_threshold = float(base_threshold)
        self.leak_rate = float(leak_rate)
        self.alpha_5ht = float(alpha_5ht)
        self.alpha_ach = float(alpha_ach)
        self.stn_conflict_gain = float(stn_conflict_gain)
        self.evidence_gain = float(evidence_gain)
        self.sigma = float(sigma)
        self.alpha_V = float(alpha_V)
        self.nogo_gain = float(nogo_gain)
        # Порог ВЕТО NoGo: когда вес застрявшего действия превышает порог,
        # действие ИСКЛЮЧАЕТСЯ из набора (жёсткий запрет), а не просто
        # получает вычитание константы. Это и есть «переключиться на
        # альтернативное действие»: константный штраф не может пересилить
        # margin, выросший до десятков логитов при насыщении кортикального
        # состояния. По умолчанию выключено (inf) — поведение обучения
        # не меняется ни на бит.
        self.nogo_veto_threshold = float(nogo_veto_threshold)

        self.evidence_acc: Optional[torch.Tensor] = None
        self.total_emissions: int = 0

        # Стриатум: Go / NoGo веса (Genesis basal_ganglia.py)
        self.w_Go: Optional[torch.Tensor] = None
        self.w_NoGo: Optional[torch.Tensor] = None

        # TD Critic (Genesis basal_ganglia.py)
        self.w_V: Optional[torch.Tensor] = None
        self.pending_value: float = 0.0
        self.pending_credit: float = 0.0
        self._v_elig: Optional[torch.Tensor] = None
        self.value: float = 0.0
        self._last_selected_token: int = -1
        self._last_ctx: Optional[torch.Tensor] = None
        self._ticks_since_emission: int = 0
        self._recent_emitted: List[int] = []

        # ── Стохастический моторный выбор (стриарная softmax-политика) ──────
        # temperature <= 0 -> жадный argmax: поведение по умолчанию, полная
        # воспроизводимость отчётов. Стохастика включается явно через
        # configure_sampling(): тогда контур выбирает действие из
        # распределения активностей коры (top-k / top-p / температура).
        self.temperature: float = 0.0
        self.top_k: int = 0
        self.top_p: float = 1.0
        self._seed: Optional[int] = None
        self._gen: Optional[torch.Generator] = None
        self._gen_key: Optional[str] = None

        # ── ВЕТО НА УРОВНЕ ПАТТЕРНА (n-грамм) ────────────────────────────────
        # Латеральное торможение в стриатуме действует не только на отдельные
        # действия, но и на ПАТТЕРНЫ действий. Если одна и та же n-грамма
        # повторилась подряд threshold раз, контур получает по ней вето на
        # veto_ticks тактов: кандидат, который завершил бы эту n-грамму,
        # исключается из набора. Это аналог Repetition Penalty, но реализованный
        # через базальные ганглии (D2-MSN), а не подкруткой логитов.
        self.ngram_size: int = 0            # 0 -> выключено
        self.ngram_repeat_threshold: int = 3
        self.ngram_veto_ticks: int = 20
        self._ngram_veto: Dict[Tuple[int, ...], int] = {}
        self.ngram_veto_events: int = 0

        # ── КРАТКОВРЕМЕННАЯ ДЕПРЕССИЯ (STP) НА ВЫХОДЕ СТРИАТУМА ─────────────
        # Модель Цодыкса-Маркрама: каждый синапс тратит ресурс при активации и
        # восстанавливается со временем (u -> u - a при использовании,
        # u -> u + b в покое). Здесь ресурс -- это "эфферентная копия" действия:
        # каждый произнесённый токен расходует свой ресурс, поэтому часто
        # повторяемое действие само собой устаёт и D2-MSN временно исключает
        # его из набора (как при реальном истощении синапса).
        #
        # Это биологический аналог frequency penalty, встроенный в моторный
        # контур: снаружи ничего не подкручивается, штраф живёт в состоянии
        # гейта и сам рассасывается.
        self.stp_enabled: bool = False
        self.stp_gain: float = 1.0          # сила депрессии в логитах
        self.stp_decrement: float = 0.35    # расход ресурса за эмиссию
        self.stp_recovery: float = 0.02      # восстановление за такт
        self.stp_exclude_below: float = 0.0  # порог ресурса для исключения
        self.stp_resource: Optional[torch.Tensor] = None
        self.stp_events: int = 0

        # ── ГРАДУИРОВАННЫЙ ПАТТЕРНОВЫЙ ТОРМОЗ (D2-MSN на уровне паттерна) ────
        # Жёсткое вето ловит только ТОЧНЫЙ повтор паттерна. Реальные аттракторы
        # генерации -- это ПЕРЕСТАНОВКИ нескольких слов ("happy new proud happy
        # new happy proud ..."), и точный детектор их не видит. Биология стриатума
        # мягче: D2-MSN подавляет не точную последовательность, а ПАТТЕРН
        # (action chunk), причём градуированно -- тормоз накапливается с
        # каждым повторением и затухает со временем.
        #
        # Здесь: ключ = (контекст из последних ctx_brake_size токенов,
        # произнесённый токен). Каждая эмиссия добавляет +0.8 к тормозу этой
        # пары; все тормоза затухают на 0.92 за такт. Если для текущего
        # контекста тормоз какого-то токена превысил порог -- этот токен
        # исключается из набора (действие заблокировано в этом контексте).
        self.ctx_brake_size: int = 2
        self.ctx_brake_threshold: float = 2.0
        self.ctx_brake_gain: float = 0.8
        # Затухание медленное (0.99/тик): биологически D2-MSN LTP, на котором
        # держится NoGo-тормоз, НЕ затухает за десятки тактов -- его снимает
        # диссинингибиция (SWR). При быстром затухании тормоз не успевает
        # накопиться: для пары, повторяющейся раз в ~12 токенов, равновесие
        # b* = gain/(1 - decay^k) при decay=0.92 равно всего 1.1.
        self.ctx_brake_decay: float = 0.99
        self.ctx_brake_topk: int = 64
        self._ctx_brake: Dict[Tuple[Tuple[int, ...], int], float] = {}
        self.ctx_brake_events: int = 0

    def reset_state(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.evidence_acc = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        self.pending_value = 0.0
        self.pending_credit = 0.0
        self._v_elig = None
        self.value = 0.0
        self._last_selected_token = -1
        self._last_ctx = None
        self._ticks_since_emission = 0
        self._recent_emitted = []
        if self.w_NoGo is not None:
            self.w_NoGo.zero_()

    def configure_sampling(self, temperature: float = 0.0, top_k: int = 0,
                            top_p: float = 1.0, seed: Optional[int] = None) -> None:
        """Включить/выключить стохастический моторный выбор.

        temperature <= 0  -> жадный argmax (поведение по умолчанию)
        top_k  > 0        -> оставить только k наиболее вероятных действий
        0 < top_p < 1     -> nucleus-фильтрация (top-p)
        seed              -> воспроизводимость сэмплирования
        """
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self._seed = None if seed is None else int(seed)
        self._gen = None
        self._gen_key = None

    def configure_stp(self, enabled: bool = True, gain: float = 1.0,
                      decrement: float = 0.35, recovery: float = 0.02,
                      exclude_below: float = 0.0) -> None:
        """Кратковременная депрессия (STP) на выходе стриатума.

        enabled=False -> выключено (поведение по умолчанию).
        decrement -- расход ресурса синапса за одну эмиссию действия.
        recovery  -- восстановление ресурса за такт.
        exclude_below -- если ресурс опустился ниже порога, действие
        временно исключается из набора (D2-MSN).
        """
        self.stp_enabled = bool(enabled)
        self.stp_gain = float(gain)
        self.stp_decrement = float(decrement)
        self.stp_recovery = float(recovery)
        self.stp_exclude_below = float(exclude_below)
        self.stp_resource = None

    def _stp_tick(self, device: torch.device, dtype: torch.dtype,
                  n_actions: int) -> None:
        """Восстановление ресурса (в покое синапс отдыхает)."""
        if not self.stp_enabled:
            return
        if self.stp_resource is None or self.stp_resource.shape[0] != n_actions \
                or self.stp_resource.device != device:
            self.stp_resource = torch.ones(n_actions, device=device, dtype=dtype)
        self.stp_resource = (self.stp_resource + self.stp_recovery).clamp(max=1.0)

    def _stp_emit(self, token: int) -> None:
        """Расход ресурса при эмиссии действия."""
        if not self.stp_enabled or self.stp_resource is None:
            return
        idx = int(token)
        if 0 <= idx < self.stp_resource.shape[0]:
            self.stp_resource[idx] = max(
                0.0, float(self.stp_resource[idx].item()) - self.stp_decrement)

    def _stp_apply(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Депрессия на «уставших» действиях + их исключение."""
        if not self.stp_enabled or self.stp_resource is None:
            return action_logits
        fatigue = (1.0 - self.stp_resource).to(action_logits.dtype)
        action_logits = action_logits - self.stp_gain * fatigue.unsqueeze(0)
        if self.stp_exclude_below > 0.0:
            tired = self.stp_resource < self.stp_exclude_below
            if bool(tired.any().item()) and not bool(tired.all().item()):
                action_logits = action_logits.masked_fill(
                    tired.unsqueeze(0), float("-inf"))
                self.stp_events += int(tired.sum().item())
        return action_logits

    def configure_nogo_veto(self, threshold: float) -> None:
        """Порог жёсткого вето NoGo (D2-MSN).

        threshold = inf -> вето выключено (поведение по умолчанию).
        threshold = K   -> действие, произнесённое K-раз подряд (вес NoGo >= K),
                         исключается из набора действий: контур обязан выбрать
                         альтернативу.
        """
        self.nogo_veto_threshold = float(threshold)

    def configure_ngram_veto(self, size: int = 0, repeat_threshold: int = 3,
                             veto_ticks: int = 20) -> None:
        """Паттерновое (n-граммное) торможение в стриатуме.

        size=0 -> выключено (поведение по умолчанию).
        size=N -> если одна и та же N-грамма повторилась подряд repeat_threshold
        раз, она получает вето на veto_ticks тактов: действие, которое её
        завершило бы, исключается из набора (как застрявшее).
        """
        self.ngram_size = int(size)
        self.ngram_repeat_threshold = max(2, int(repeat_threshold))
        self.ngram_veto_ticks = int(veto_ticks)
        self._ngram_veto.clear()

    def configure_ctx_brake(self, size: int = 0, threshold: float = 2.0,
                            gain: float = 0.8, decay: float = 0.99) -> None:
        """Градуированный паттерновый тормоз (размер контекста = size-1 токенов).

        size=0 -> выключено. threshold -- порог тормоза, при котором действие
        исключается из набора.
        """
        self.ctx_brake_size = int(size)
        self.ctx_brake_threshold = float(threshold)
        self.ctx_brake_gain = float(gain)
        self.ctx_brake_decay = float(decay)
        self._ctx_brake.clear()

    def _ctx_brake_apply(self, action_logits: torch.Tensor) -> torch.Tensor:
        """Исключить из набора действия с перегруженным тормозом в этом контексте."""
        if self.ctx_brake_size <= 0 or not self._ctx_brake:
            return action_logits
        n = self.ctx_brake_size
        hist = self._recent_emitted
        if len(hist) < n:
            return action_logits
        ctx = tuple(hist[-n:])
        # собираем тормоза для текущего контекста
        row: Dict[int, float] = {}
        for (c, t), b in self._ctx_brake.items():
            if c == ctx:
                row[t] = b
        if not row:
            return action_logits
        k = min(self.ctx_brake_topk, action_logits.shape[-1])
        idx = action_logits[0].topk(k).indices.tolist()
        blocked = [t for t in idx if row.get(t, 0.0) >= self.ctx_brake_threshold]
        if blocked:
            tidx = torch.tensor(blocked, dtype=torch.long, device=action_logits.device)
            action_logits = action_logits.index_fill(1, tidx, float("-inf"))
            self.ctx_brake_events += len(blocked)
        return action_logits

    def _ctx_brake_decay(self) -> None:
        if not self._ctx_brake:
            return
        for key in list(self._ctx_brake.keys()):
            b = self._ctx_brake[key] * self.ctx_brake_decay
            if b < 0.05:
                del self._ctx_brake[key]
            else:
                self._ctx_brake[key] = b

    def _ctx_brake_emit(self, token: int) -> None:
        n = self.ctx_brake_size
        if n <= 0:
            return
        hist = self._recent_emitted
        if len(hist) < n + 1:
            return
        ctx = tuple(hist[-(n + 1):-1])
        key = (ctx, int(token))
        self._ctx_brake[key] = self._ctx_brake.get(key, 0.0) + self.ctx_brake_gain

    def _detect_cycle_period(self) -> int:
        """Период застрявшего цикла действий (0 = цикла нет).

        Цикл -- это повторяющийся ПАТТЕРН действий любой длины: реальные
        аттракторы в генерации имеют период 1 ("dad dad dad"), 2 ("toy that
        toy that") или 3. Поэтому ищем наименьший p, для которого последние
        три блока длины p совпадают.
        """
        hist = self._recent_emitted
        n = self.ngram_size
        thr = max(2, self.ngram_repeat_threshold)
        for p in range(1, n + 1):
            if len(hist) < thr * p:
                continue
            unit = tuple(hist[-p:])
            blocks = [tuple(hist[-(k + 1) * p: len(hist) - k * p])
                      for k in range(thr)]
            if all(b == unit for b in blocks):
                return p
        return 0

    def _ngram_veto_tokens(self) -> List[int]:
        """Токены, которые ПРОДОЛЖАЛИ бы запрещённый паттерн.

        Если запрещён цикл-юнит g = (a, b, c) и последние len(g)-1
        произнесённых токенов равны хвосту юнита, то следующий токен 'a'
        замкнёт цикл g -> g -> g. Именно его блокируем (контекстно: только
        при совпадении хвоста).
        """
        if self.ngram_size <= 0 or not self._ngram_veto:
            return []
        hist = self._recent_emitted
        out: List[int] = []
        for unit, ticks in self._ngram_veto.items():
            if ticks <= 0 or len(unit) < 1:
                continue
            p = len(unit)
            if p == 1:
                out.append(int(unit[0]))
                continue
            if len(hist) < p - 1:
                continue
            if tuple(hist[-(p - 1):]) == tuple(unit[1:]):
                out.append(int(unit[0]))
        return out

    def _ngram_veto_tick(self) -> None:
        """Старение вето (по тикам) и детекция повторяющихся паттернов."""
        if self.ngram_size <= 0:
            return
        if self._ngram_veto:
            for unit in list(self._ngram_veto.keys()):
                left = self._ngram_veto[unit] - 1
                if left <= 0:
                    del self._ngram_veto[unit]
                else:
                    self._ngram_veto[unit] = left
        period = self._detect_cycle_period()
        if period <= 0:
            return
        unit = tuple(self._recent_emitted[-period:])
        self._ngram_veto[unit] = self.ngram_veto_ticks
        self.ngram_veto_events += 1

    def _sampling_generator(self, device: torch.device) -> Optional[torch.Generator]:
        """Ленивый генератор случайных чисел, привязанный к устройству."""
        if self._seed is None:
            return None
        key = f"{device.type}:{device.index}"
        if self._gen is None or self._gen_key != key:
            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(self._seed)
            self._gen_key = key
        return self._gen

    @torch.no_grad()
    def settle(self, delta: float) -> None:
        """Pay TD error out to the critic and close the trial (Genesis basal_ganglia.py)."""
        if self._v_elig is not None and self.w_V is not None:
            mass = float(self._v_elig.norm())
            if mass > 1e-9:
                self.w_V += self.alpha_V * float(delta) * (self._v_elig / mass)
            self.w_V.clamp_(min=0.0)
            self._v_elig.zero_()
        self.pending_value = 0.0
        self.pending_credit = 0.0

    @torch.no_grad()
    def forward(
        self,
        workspace_logits: torch.Tensor,
        serotonin_5ht: Optional[torch.Tensor] = None,
        ach_level: Optional[torch.Tensor] = None,
        orexin_level: Optional[torch.Tensor] = None,
        surprise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = workspace_logits.shape[0]
        device = workspace_logits.device
        dtype = workspace_logits.dtype
        n_actions = workspace_logits.shape[-1]

        if (
            self.evidence_acc is None
            or self.evidence_acc.shape[0] != batch_size
            or self.evidence_acc.device != device
        ):
            self.reset_state(batch_size, device, dtype)

        if self.w_V is None or self._v_elig is None or self.w_V.shape[0] != n_actions or self.w_V.device != device:
            self.w_V = torch.zeros(n_actions, device=device, dtype=dtype)
            self._v_elig = torch.zeros(n_actions, device=device, dtype=dtype)

        if self.w_NoGo is None or self.w_NoGo.shape[0] != n_actions or self.w_NoGo.device != device:
            self.w_NoGo = torch.zeros(n_actions, device=device, dtype=dtype)
            self.w_Go = torch.ones(n_actions, device=device, dtype=dtype)

        # Релаксация NoGo-тормоза со временем (decay back to baseline)
        self.w_NoGo = self.w_NoGo * 0.92

        ctx = workspace_logits.detach()[0]
        ctx_norm = float(ctx.norm())
        ctx_hat = ctx / max(ctx_norm, 1e-6)

        # Вычисление ценности текущего состояния V(s) критика
        self.value = max(float(torch.dot(self.w_V, ctx_hat)), 0.0)

        # След пригодности критика (running average of directions)
        critic_decay = math.exp(-self.dt / 500.0)
        self._v_elig = self._v_elig * critic_decay + ctx_hat * (1.0 - critic_decay)

        # ── Стриарный моторный тормоз NoGo: подавление застрявших действий ──
        # Моторные логиты с учетом NoGo-ингибирования
        nogo_penalty = self.nogo_gain * self.w_NoGo.unsqueeze(0)
        motor_logits = workspace_logits.detach() - nogo_penalty

        # Стохастический шум
        top2_raw = torch.topk(motor_logits, k=min(2, n_actions), dim=-1).values
        raw_margin = (
            (top2_raw[:, 0:1] - top2_raw[:, 1:2]).clamp(min=0.0)
            if top2_raw.shape[-1] >= 2
            else torch.zeros((batch_size, 1), device=device)
        )
        drive_gate = torch.clamp(raw_margin / 0.05, 0.0, 1.0)

        noisy_logits = motor_logits.clone()
        if self.sigma > 0.0:
            noisy_logits = noisy_logits + (self.sigma * drive_gate) * torch.randn_like(noisy_logits)

        probs = F.softmax(noisy_logits, dim=-1)
        if probs.shape[-1] >= 2:
            top2_vals, _top2_idx = torch.topk(probs, k=2, dim=-1)
            top2_logits = torch.topk(noisy_logits, k=2, dim=-1).values
            margin = (top2_logits[:, 0:1] - top2_logits[:, 1:2]).clamp(min=0.0)
            stn_conflict = (1.0 - (top2_vals[:, 0:1] - top2_vals[:, 1:2])) * top2_vals[:, 1:2]
        else:
            margin = probs[:, 0:1]
            stn_conflict = torch.zeros((batch_size, 1), device=device, dtype=dtype)

        # ── ВЕТО NoGo: застрявшее действие исключается из набора ─────────────
        # Вето применяется к НАБОРУ ДЕЙСТВИЙ, а не к margin: margin отражает
        # физическую salience и остаётся честной характеристикой состояния.
        # Нужно именно исключение, а не вычитание константы: при насыщении
        # кортикального состояния margin декодера вырастает до десятков
        # логитов, и постоянный штраф NoGo перестаёт его пересиливать.
        action_logits = noisy_logits
        if self.nogo_veto_threshold < float("inf"):
            veto = self.w_NoGo >= self.nogo_veto_threshold
            if bool(veto.any().item()) and not bool(veto.all().item()):
                action_logits = action_logits.masked_fill(veto.unsqueeze(0), float("-inf"))

        # ── Паттерновое торможение: запрещённые n-граммы ────────────────────
        self._ngram_veto_tick()
        self._ctx_brake_decay()
        blocked_tokens = self._ngram_veto_tokens()
        if blocked_tokens:
            idx = torch.tensor(blocked_tokens, dtype=torch.long, device=device)
            action_logits = action_logits.index_fill(1, idx, float("-inf"))
        # Градуированный тормоз на уровне паттерна (D2-MSN): действия,
        # повторяющиеся в одном и том же контексте, постепенно исключаются.
        action_logits = self._ctx_brake_apply(action_logits)
        # Кратковременная депрессия (STP): «уставшие» действия исключаются.
        self._stp_tick(device, dtype, action_logits.shape[-1])
        action_logits = self._stp_apply(action_logits)

        # ── Стохастический моторный выбор (стриарная softmax-политика) ──────
        # При temperature <= 0 остаётся жадный argmax (поведение по умолчанию).
        # При temperature > 0 действие выбирается из распределения активностей
        # коры: это даёт контуру возможность уйти в альтернативную гипотезу,
        # которую подготовил гиппокампальный реколл.
        if self.temperature > 0.0:
            selected_token = sample_token(
                action_logits,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                generator=self._sampling_generator(device),
            )
        else:
            selected_token = action_logits.argmax(dim=-1)

        cand_tok = int(selected_token[0].item())

        # ── TD Critic: проверка на зацикливание и генерация Dip ──────────
        dip_detected = False
        if self.pending_value > 0.0:
            self._ticks_since_emission += 1
            self.pending_value *= 0.98 ** self.dt
            self.pending_credit *= math.exp(-self.dt / 500.0)

            # Оценка информативности кандидата
            recent_counts = sum(1 for t in self._recent_emitted[-8:] if t == cand_tok)
            is_immediate_repeat = (len(self._recent_emitted) > 0 and cand_tok == self._recent_emitted[-1])
            is_punct_repeat = (cand_tok in (2, 3, 8) and recent_counts >= 2)  # . , and
            is_word_repeat = (recent_counts >= 3)

            if is_immediate_repeat or is_punct_repeat or is_word_repeat:
                # Отрицательный TD сигнал (DIP): отсутствие прогресса контекста
                reward = -0.50 if is_immediate_repeat else -0.30
                delta = reward - self.pending_value
                if delta < -0.10:
                    dip_detected = True
                    self.settle(delta)
                    # Сброс аккумулятора DDM
                    self.evidence_acc.zero_()
                    # Усиление NoGo тормоза (D2-MSN LTP) для застрявшего токена
                    self.w_NoGo[cand_tok] += 1.8 * abs(delta)
                    self.w_NoGo.clamp_(max=5.0)
            elif recent_counts == 0 and self._ticks_since_emission >= 1:
                # Новый семантический токен (награда)
                reward = 0.70
                delta = reward - self.pending_value
                self.settle(delta)

        # Дрифт DDM
        drift = (
            self.evidence_gain * (margin - self.stn_conflict_gain * stn_conflict)
            - self.leak_rate * self.evidence_acc
        )
        if self.sigma > 0.0:
            drift = drift + (self.sigma * drive_gate) * torch.randn_like(self.evidence_acc)

        self.evidence_acc = torch.clamp(self.evidence_acc + self.dt * drift, min=0.0, max=10.0)
        evidence_before_reset = self.evidence_acc.clone()

        # Модуляция порога
        mod_factor = torch.ones((batch_size, 1), device=device, dtype=dtype)
        if orexin_level is not None:
            orexin = orexin_level.detach().mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
            mod_factor = mod_factor - 0.30 * orexin
        if serotonin_5ht is not None:
            # Высокий 5-HT повышает порог DDM (терпение), предотвращая токен-спам
            s5ht = serotonin_5ht.detach().mean(dim=-1, keepdim=True).clamp(0.0, 1.5)
            mod_factor = mod_factor + self.alpha_5ht * (s5ht - 0.5)
        if ach_level is not None:
            ach_pos = torch.relu(ach_level.detach().mean(dim=-1, keepdim=True)).clamp(0.0, 1.0)
            mod_factor = mod_factor + self.alpha_ach * ach_pos
        if surprise_level is not None:
            surp = surprise_level.detach().mean(dim=-1, keepdim=True).clamp(0.0, 2.0)
            mod_factor = mod_factor + 0.25 * surp
        if dip_detected:
            mod_factor = mod_factor + 0.35

        threshold = (self.base_threshold * mod_factor).clamp(min=0.20, max=5.0)

        emit_mask = (self.evidence_acc >= threshold).to(dtype)
        emitted = float(emit_mask.sum().item())
        self.total_emissions += int(emitted)

        if emitted >= 1.0:
            emitted_tok = int(selected_token[0].item())
            self._recent_emitted.append(emitted_tok)
            if len(self._recent_emitted) > 32:
                self._recent_emitted.pop(0)

            # Установка ожидания критика (Genesis)
            self.pending_value = max(self.value, 0.50)
            self.pending_credit = 1.0
            self._v_elig = ctx_hat.clone()
            self._last_selected_token = emitted_tok
            self._last_ctx = ctx.clone()
            self._ticks_since_emission = 0

            # Лёгкое временное охлаждение только что эмитированного токена в NoGo
            self.w_NoGo[emitted_tok] += 0.80
            # Накопление градуированного паттернового тормоза: пара
            # (контекст, действие) застревает тем сильнее, чем чаще повторяется.
            self._ctx_brake_emit(emitted_tok)
            # Расход синаптического ресурса (STP): частое действие устаёт.
            self._stp_emit(emitted_tok)

        self.evidence_acc = self.evidence_acc * (1.0 - emit_mask)

        return {
            "emit_mask": emit_mask,
            "selected_token": selected_token,
            "evidence": evidence_before_reset,
            "threshold": threshold,
            "margin": margin,
            "dip": torch.tensor(1.0 if dip_detected else 0.0, device=device),
        }
