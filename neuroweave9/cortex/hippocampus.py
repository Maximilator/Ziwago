"""neuroweave9/cortex/hippocampus.py — Гиппокампальный эпизодический контур v9.0.

Интеграция трёх механизмов из Genesis:
1. ThetaOscillator (~7 Hz, период ~140 тактов): фазовое разделение кодирования (пик, theta > 0)
   и извлечения (спад/впадина, theta <= 0). Блокирует retrieve() во время фазы записи.
2. Asynchronous CA3 Hopfield: асинхронное рекуррентное восстановление дискретного аттрактора
   за 5-7 шагов в случайном порядке нейронов (step_retrieve). Инжектирует целостный эпизод,
   а не размытый вектор.
3. CA1 Comparator (Mismatch): сравнение текущего кортикального состояния с предсказанным/
   восстановленным эпизодом CA3. Высокий mismatch служит естественным триггером для recall.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroweave8.config import HippoCfg
from neuroweave8.core.sparse import k_wta


class ThetaOscillator:
    """Тета-осциллятор гиппокампа (~7 Гц, период ~140-143 мс при dt=1 мс).

    В биологии тета-ритм делит каждый цикл (~140 мс) на две половины:
    - Пик (theta > 0): фаза кодирования (высокий ацетилхолин, запись в синапсы).
    - Впадина (theta <= 0): фаза извлечения (низкий ACh, считывание CA3 -> CA1 -> кора).

    МАСШТАБ ВРЕМЕНИ (важно для генерации текста).

    Базовый режим: dt = 1 мс на шаг, 7 Гц -> 143 шага на тета-цикл. В текстовом
    потоке 1 шаг = 1 токен, поэтому гиппокамп получает ОДНУ возможность реколла
    на 143 токена. Рабочий контекст коры при этом живёт ~32 токена: к моменту,
    когда реколл наконец разрешён, контекст уже мёртв, и SWR-дисинингибиция
    не успевает разрушить застрявший кортикальный аттрактор.

    Два режима выравнивания (рантайм, веса не затрагиваются):

    * «fast»  — ВАРИАНТ А: сохраняем биологические 7 Гц, но пересчитываем
      масштаб времени модели так, чтобы тета-цикл занимал N шагов
      (эквивалентно: 1 токен = 143/N мс модельного времени). Осциллятор
      остаётся непрерывным, меняется только его скорость.
    * «token» — ВАРИАНТ Б: фаза теты считается напрямую от счётчика токенов
      (1 токен = 1/N тета-цикла), т.е. часы гиппокампа не зависят от числа
      внутренних тактов коры и всегда синхронны с потоком токенов.
    """

    def __init__(self, freq_hz: float = 7.0, dt: float = 1.0):
        self.freq_hz = float(freq_hz)
        self.base_dt = float(dt)
        self.dt = float(dt)
        self.t: float = 0.0
        self._retrieved_in_cycle: bool = False
        self._last_phase_sign: float = 1.0
        # Масштаб тета-цикла: "baseline" | "fast" | "token"
        self._mode: str = "baseline"
        self._cycle_steps: float = 0.0
        self._token_index: int = 0

    def configure(self, mode: str = "baseline", cycle_steps: float = 0.0) -> None:
        """Настроить масштаб тета-цикла (без изменения весов модели).

        mode="baseline" -- исходный режим: 143 шага на цикл (7 Гц при dt=1 мс).
        mode="fast"     -- ВАРИАНТ А: тета-цикл занимает cycle_steps шагов,
                          биологическая частота 7 Гц сохраняется.
        mode="token"    -- ВАРИАНТ Б: фаза считается от счётчика токенов,
                          1 токен = 1/cycle_steps тета-цикла.
        """
        if mode == "baseline":
            self._mode = "baseline"
            self.dt = self.base_dt
            self._cycle_steps = 0.0
        elif mode in ("fast", "token"):
            steps = float(cycle_steps)
            if steps <= 0.0:
                raise ValueError("cycle_steps должен быть > 0")
            self._mode = mode
            self._cycle_steps = steps
            # «fast»: сохраняем 7 Гц, меняем только масштаб времени модели
            self.dt = self.base_dt if mode == "token" else \
                (1000.0 / self.freq_hz) / steps
        else:
            raise ValueError(f"неизвестный режим тета-тайминга: {mode}")
        self.reset_state()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def cycle_steps(self) -> float:
        """Длина тета-цикла в шагах (0 -> исходный режим)."""
        if self._mode == "baseline":
            return 1000.0 / (self.freq_hz * self.base_dt)
        return self._cycle_steps

    def reset_state(self) -> None:
        self.t = 0.0
        self._retrieved_in_cycle = False
        self._last_phase_sign = 1.0
        self._token_index = 0

    def step(self) -> Dict[str, float]:
        if self._mode == "token":
            # ВАРИАНТ Б: фаза — строгая функция счётчика токенов
            self._token_index += 1
            frac = (self._token_index % self._cycle_steps) / self._cycle_steps
            theta = math.sin(2.0 * math.pi * frac)
            cos_theta = math.cos(2.0 * math.pi * frac)
            phase = 2.0 * math.pi * frac
        else:
            # ВАРИАНТ А / базовый: непрерывный осциллятор с пересчитанным dt
            self.t += self.dt
            two_pi_f = 2.0 * math.pi * self.freq_hz * 1e-3
            theta = math.sin(two_pi_f * self.t)  # в диапазоне [-1, 1]
            cos_theta = math.cos(two_pi_f * self.t)
            phase = (two_pi_f * self.t) % (2.0 * math.pi)

        # Переход в пик кодирования (theta > 0): сброс флага реколла для следующего цикла
        if theta > 0.0 and self._last_phase_sign <= 0.0:
            self._retrieved_in_cycle = False
        self._last_phase_sign = theta

        # Фаза извлечения: спад и впадина тета-волны (theta <= 0.0)
        is_retrieval = (theta <= 0.0)
        retrieval_ready = is_retrieval and not self._retrieved_in_cycle

        return {
            "theta": theta,
            "cos_theta": cos_theta,
            "phase": phase,
            "is_retrieval": is_retrieval,
            "is_encoding": not is_retrieval,
            "retrieval_ready": retrieval_ready,
        }

    def mark_retrieved(self) -> None:
        self._retrieved_in_cycle = True

    def current_theta(self) -> float:
        if self._mode == "token":
            frac = (self._token_index % self._cycle_steps) / self._cycle_steps
            return math.sin(2.0 * math.pi * frac)
        two_pi_f = 2.0 * math.pi * self.freq_hz * 1e-3
        return math.sin(two_pi_f * self.t)

    def is_retrieval_phase(self) -> bool:
        return self.current_theta() <= 0.0

    def is_encoding_phase(self) -> bool:
        return self.current_theta() > 0.0


class EntorhinalCortex(nn.Module):
    """Энторинальная кора (EC) — grid-cell интерфейс гиппокампа к неокортексу (из Genesis).

        РЕЖИМЫ ВЫБОРА ЭПИЗОДА:
        - "match"  -- классический nearest-match (argmax по сходству). Проблема:
          когда кора застревает, самый похожий эпизод УСИЛИВАЕТ текущую тему,
          и аттрактор не ломается.
        - "explore" -- ДИВЕРГЕНТНЫЙ реколл: эпизод выбирается из того же
          распределения сходств, но сглаженным (softmax по sims/T). Биологически
          это оправдано: pattern completion должен предлагать АЛЬТЕРНАТИВУ,
          а не копию текущего состояния. T -> 0 даёт "match", T -> inf --
          равномерный выбор по всем эпизодам.

    Преобразует реконструированный CA1 вектор через мультимасштабные
    фазовые сетки (Grid Cells / Random Fourier Features) в нелинейное
    пространственное представление и проецирует его в L5 неокортекса.
    """

    def __init__(
        self,
        n_neurons: int = 256,
        n_in: int = 512,
        n_cortex_l5: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        self.n = int(n_neurons)
        self.n_in = int(n_in)
        self.n_cortex_l5 = int(n_cortex_l5)

        gen = torch.Generator().manual_seed(seed)
        phase_offsets = torch.rand(self.n, generator=gen) * 2.0 * math.pi
        freq_scales = 0.5 + torch.rand(self.n, generator=gen) * 1.5
        w_in = torch.randn(self.n, self.n_in, generator=gen) / math.sqrt(self.n_in)
        w_out = torch.randn(self.n_cortex_l5, self.n, generator=gen) / math.sqrt(self.n)

        self.register_buffer("phase_offsets", phase_offsets, persistent=False)
        self.register_buffer("freq_scales", freq_scales, persistent=False)
        self.register_buffer("w_in", w_in, persistent=False)
        self.register_buffer("w_out", w_out, persistent=False)
        self.output: Optional[torch.Tensor] = None

    def reset_state(self, device: torch.device, dtype=torch.float32) -> None:
        self.output = torch.zeros(1, self.n, device=device, dtype=dtype)

    @torch.no_grad()
    def step(self, ca1_input: torch.Tensor) -> torch.Tensor:
        """Обновление состояния EC от CA1 (Genesis)."""
        x = ca1_input.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        x_norm = x / rms
        proj = F.linear(x_norm, self.w_in)
        grid = torch.cos(self.freq_scales.unsqueeze(0) * proj + self.phase_offsets.unsqueeze(0))
        self.output = torch.sigmoid(4.0 * grid)
        return self.output

    @torch.no_grad()
    def project_to_cortex(self, gain: float = 1.0,
                          content_gain: float = 0.0) -> torch.Tensor:
        # PROYEKCIYA EC -> L5; content_gain otdelyaet EPISOD-SPECIFICHNUYU
        # komponentu ot obshchey: vyhod EC -- sigmoid(4*cos), to est prakticheski
        # ravnomernyy po vsem kletkam, i raznye epizody otlichayutsya lish
        # malym otkloneniem. Bez otdeleniya etogo otkloneniya rekoll rabotaet
        # kak content-nezavisimyy "tolchok": kakoy epizod podstavlen, na vyhod
        # ne vliyaet.
        if self.output is None:
            return torch.zeros(1, self.n_cortex_l5, device=self.w_out.device)
        out = self.output
        ctx = F.linear(out, self.w_out)
        if content_gain > 0.0:
            dev = out - out.mean(dim=-1, keepdim=True)
            ctx = ctx + float(content_gain) * F.linear(dev, self.w_out)
        ctx_pos = F.relu(ctx)
        peak = float(ctx_pos.max().item())
        if peak > 1e-6:
            ctx_pos = (ctx_pos / peak) * gain
        return ctx_pos


class Hippocampus(nn.Module):
    """Гиппокамп v9.0: Тета-осциллятор, асинхронный CA3 Хопфилд, CA1 Mismatch и EC Grid-интерфейс."""

    def __init__(self, n_in: int, cfg: HippoCfg, seed: int = 4242, dt: float = 1.0, n_cortex_l5: int = 32):
        super().__init__()
        self.cfg = cfg
        self.n_in = int(n_in)
        self.n_dg = int(cfg.n_dg)
        self.n_cortex_l5 = int(n_cortex_l5)
        self.dt = float(dt)
        gen = torch.Generator().manual_seed(seed)

        # 1. DG: статическая проекция расширения
        w_dg = torch.randn(self.n_dg, n_in, generator=gen) / math.sqrt(max(1, n_in))
        self.register_buffer("w_dg", w_dg)
        self.k_dg = max(2, int(cfg.dg_sparsity * self.n_dg))

        # 2. CA3: автоассоциативная матрица Хопфилда (симметричная, с нулевой диагональю)
        w_ca3 = torch.randn(self.n_dg, self.n_dg, generator=gen)
        keep = torch.rand(self.n_dg, self.n_dg, generator=gen) < cfg.ca3_fan_in
        w_ca3 = w_ca3 * keep
        w_ca3.fill_diagonal_(0.0)
        # Симметризация связей (классический Хопфилд)
        w_ca3 = 0.5 * (w_ca3 + w_ca3.t())
        w_ca3 = w_ca3 / (w_ca3.abs().max() + 1e-6) * 0.90
        self.register_buffer("w_ca3", w_ca3)

        # 3. CA1: проекция CA3 -> пространство входа (реконструкция)
        self.register_buffer("w_ca1", w_dg.t() / math.sqrt(max(1, self.n_dg)))

        # 4. Энторинальная кора (EC) из Genesis: grid-cell интерфейс к неокортексу
        self.ec = EntorhinalCortex(
            n_neurons=self.n_dg, n_in=self.n_in, n_cortex_l5=self.n_cortex_l5, seed=seed + 101
        )

        # 5. Внутренний тета-осциллятор
        self.oscillator = ThetaOscillator(freq_hz=7.0, dt=dt)

        # 6. Хранилище эпизодов
        self._slots_x: Optional[torch.Tensor] = None
        self._slots_code: Optional[torch.Tensor] = None
        self._slots_err: Optional[torch.Tensor] = None
        self._slots_age: Optional[torch.Tensor] = None
        self._n_stored = 0
        self._ptr = 0

    def reset_state(self, device: torch.device, dtype=torch.float32) -> None:
        cfg = self.cfg
        self._slots_x = torch.zeros(cfg.n_slots, self.n_in, device=device, dtype=dtype)
        self._slots_code = torch.zeros(cfg.n_slots, self.n_dg, device=device, dtype=dtype)
        self._slots_err = torch.zeros(cfg.n_slots, device=device, dtype=dtype)
        self._slots_age = torch.zeros(cfg.n_slots, device=device, dtype=dtype)
        self._n_stored = 0
        self._ptr = 0
        self.oscillator.reset_state()
        self.ec.reset_state(device, dtype)

    def current_theta(self) -> float:
        return self.oscillator.current_theta()

    def is_retrieval_phase(self) -> bool:
        return self.oscillator.is_retrieval_phase()

    def is_encoding_phase(self) -> bool:
        return self.oscillator.is_encoding_phase()

    def configure_theta(self, mode: str = "baseline", cycle_steps: float = 0.0) -> None:
        """Настроить масштаб тета-цикла гиппокампа (рантайм, без весов).

        mode="baseline" -- 143 шага на цикл (исходный режим);
        mode="fast"     -- ВАРИАНТ А: цикл = cycle_steps шагов при тех же 7 Гц;
        mode="token"    -- ВАРИАНТ Б: фаза от счётчика токенов.
        """
        self.oscillator.configure(mode=mode, cycle_steps=cycle_steps)

    def _ensure(self, x: torch.Tensor) -> None:
        if self._slots_x is None or self._slots_x.device != x.device:
            self.reset_state(x.device, x.dtype)

    @torch.no_grad()
    def separate(self, x: torch.Tensor) -> torch.Tensor:
        """DG: разрежённый ортогонализованный код входа [B, n_dg]."""
        proj = F.linear(x, self.w_dg)
        return k_wta(torch.clamp(proj, min=0.0), self.k_dg)

    @torch.no_grad()
    def complete(self, code: torch.Tensor, steps: Optional[int] = None, async_hopfield: bool = True) -> torch.Tensor:
        """CA3: асинхронное аттракторное восстановление полного кода (Хопфилд из Genesis).

        За steps итераций обновляет нейроны в случайном порядке, обеспечивая
        сходимость к целостному дискретному аттрактору (эпизоду).
        """
        steps = self.cfg.ca3_steps if steps is None else int(steps)
        B = code.shape[0]
        n = self.n_dg
        dev = code.device

        if not async_hopfield:
            # Быстрый синхронный шаг для обучения/батчей
            c = code.clone()
            for _ in range(max(1, steps)):
                c = torch.tanh(self.cfg.ca3_gain * (c + F.linear(c, self.w_ca3)))
                c = k_wta(torch.clamp(c, min=0.0), self.k_dg)
            return c

        # Асинхронное обновление в случайном порядке (как в genesis/CA3Autoassociative)
        c = code.clone()
        for b in range(B):
            # Биполярный вид {-1, +1}
            state = torch.sign(c[b].clone())
            state[state == 0] = 1.0
            for _ in range(max(1, steps)):
                order = torch.randperm(n, device=dev)
                for i in order:
                    h_i = torch.dot(self.w_ca3[i], state)
                    state[i] = 1.0 if h_i >= 0.0 else -1.0
            # Возврат в разреженный вид k-WTA
            pos = torch.clamp(state, min=0.0).unsqueeze(0)
            c[b : b + 1] = k_wta(pos, self.k_dg)
        return c

    @torch.no_grad()
    def reconstruct(self, code: torch.Tensor) -> torch.Tensor:
        """CA1: DG-код -> аппроксимация исходного входа."""
        return F.linear(code, self.w_ca1)

    @torch.no_grad()
    def compute_ca1_mismatch(self, current_x: torch.Tensor, ca3_rec_x: torch.Tensor) -> float:
        """CA1 Comparator: вычисляет ошибку несоответствия между текущим входом и воспоминанием CA3.

        В Genesis: mismatch = ||ca3 - ec||^2 / N.
        Высокий mismatch означает, что кора отклонилась от эпизодического сюжета.
        """
        diff = current_x - ca3_rec_x
        mismatch = float(((diff ** 2).sum(-1) / max(1, diff.shape[-1])).mean().item())
        return mismatch

    @torch.no_grad()
    def process(self, x: torch.Tensor, theta_gate: bool = True) -> Dict[str, object]:
        """Обработка текущего входа: DG-сепарация, CA1 mismatch и условная запись в слоты.

        Если theta_gate=True, запись новых эпизодов разрешена преимущественно в фазе
        кодирования (пик тета-волны).
        """
        self._ensure(x)
        theta_st = self.oscillator.step()
        code = self.separate(x)
        completed = self.complete(code, steps=1, async_hopfield=False)
        rec = self.reconstruct(completed)

        mismatch = self.compute_ca1_mismatch(x, rec)

        # В фазе кодирования (или при высокой новизне) пишем в память
        can_store = True
        if theta_gate and not theta_st["is_encoding"] and self._n_stored >= 10:
            can_store = (mismatch >= self.cfg.novelty_threshold * 1.5)

        written = 0.0
        if can_store and (mismatch >= self.cfg.novelty_threshold or self._n_stored < self.cfg.n_slots):
            if self._n_stored < self.cfg.n_slots:
                idx = self._n_stored
                self._n_stored += 1
            else:
                idx = int(torch.argmin(self._slots_err).item())
            self._slots_x[idx] = x[0]
            self._slots_code[idx] = code[0]
            self._slots_err[idx] = mismatch
            self._slots_age[idx] = 0.0
            written = 1.0

        self._slots_age.add_(1.0)
        return {
            "code": code,
            "rec": rec,
            "mismatch": mismatch,
            "written": written,
            "theta": theta_st["theta"],
            "is_retrieval_phase": theta_st["is_retrieval"],
        }

    @torch.no_grad()
    def retrieve(
        self,
        query: Optional[torch.Tensor] = None,
        anchor_slots: Optional[List[int]] = None,
        force: bool = False,
        n_steps: int = 6,
        gain: float = 1.0,
        mode: str = "match",
        explore_temp: float = 0.25,
        content_gain: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, object]:
        """Theta-Gated Retrieval: ассоциативное извлечение эпизода через CA3->CA1->EC->Cortex.

        Правило тета-гейта (Genesis / Hasselmo):
        - Во время пика тета-волны (кодирование) retrieve ЗАПРЕЩЁН (is_retrieval = False).
        - На спаде/впадине (retrieval phase) retrieve РАЗРЕШЁН (максимум 1 реколл за цикл).
        - Асинхронный Хопфилд выполняет n_steps итераций pattern completion в CA3.
        - CA1 обновляет Энторинальную Кору (EC), генерирующую grid-паттерн в L5 неокортекса.
        """
        # Проверка фазы тета-осциллятора
        theta_val = self.oscillator.current_theta()
        is_retrieval_phase = (theta_val <= 0.0)

        if not force:
            if not is_retrieval_phase:
                dev = self.w_dg.device
                return {
                    "allowed": False,
                    "reason": "theta_encoding_phase_blocked",
                    "x_rec": torch.zeros(1, self.n_in, device=dev),
                    "code": torch.zeros(1, self.n_dg, device=dev),
                    "apical_l5": torch.zeros(1, self.n_cortex_l5, device=dev),
                    "slot_idx": -1,
                    "mismatch": 0.0,
                    "sim": 0.0,
                    "theta": theta_val,
                }
            if self.oscillator._retrieved_in_cycle:
                dev = self.w_dg.device
                return {
                    "allowed": False,
                    "reason": "theta_cycle_already_retrieved",
                    "x_rec": torch.zeros(1, self.n_in, device=dev),
                    "code": torch.zeros(1, self.n_dg, device=dev),
                    "apical_l5": torch.zeros(1, self.n_cortex_l5, device=dev),
                    "slot_idx": -1,
                    "mismatch": 0.0,
                    "sim": 0.0,
                    "theta": theta_val,
                }

        if self._n_stored == 0:
            dev = self.w_dg.device
            return {
                "allowed": True,
                "x_rec": torch.zeros(1, self.n_in, device=dev),
                "code": torch.zeros(1, self.n_dg, device=dev),
                "apical_l5": torch.zeros(1, self.n_cortex_l5, device=dev),
                "slot_idx": 0,
                "mismatch": 0.0,
                "sim": 0.0,
                "theta": theta_val,
            }

        # Фиксируем реколл для текущего тета-цикла
        self.oscillator.mark_retrieved()

        dev = self._slots_code.device
        stored_codes = self._slots_code[: self._n_stored]

        if query is not None:
            q_code = self.separate(query)
            dots = (q_code.unsqueeze(1) * stored_codes.unsqueeze(0)).sum(dim=-1)
            norms = q_code.norm(dim=-1, keepdim=True) * stored_codes.norm(dim=-1).unsqueeze(0) + 1e-8
            sims = dots / norms
            if anchor_slots is not None and len(anchor_slots) > 0:
                mask = torch.zeros(self._n_stored, device=dev)
                for s in anchor_slots:
                    if s < self._n_stored:
                        mask[s] = 1.0
                sims = sims * mask.unsqueeze(0)
            if mode == "explore" and self._n_stored > 1:
                # ДИВЕРГЕНТНЫЙ выбор: сглаженное распределение по сходствам.
                # pattern completion должен предлагать альтернативу, а не копию
                # текущего состояния (иначе реколл только усиливает аттрактор).
                probs = F.softmax(sims[0] / max(1e-6, float(explore_temp)), dim=0)
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                if float(probs.sum().item()) <= 0.0:
                    best_idx = int(torch.argmax(sims[0]).item())
                else:
                    best_idx = int(torch.multinomial(
                        probs, 1, generator=generator).item())
            else:
                best_idx = int(torch.argmax(sims[0]).item())
            best_sim = float(sims[0, best_idx].item())
        else:
            best_idx = int(torch.argmax(self._slots_err[: self._n_stored]).item())
            best_sim = 1.0

        raw_code = self._slots_code[best_idx : best_idx + 1]

        # Асинхронный Хопфилд из Genesis (5-7 шагов)
        completed_code = self.complete(raw_code, steps=n_steps, async_hopfield=True)
        # CA1 реконструкция
        x_rec = self.reconstruct(completed_code)

        # Полный путь Genesis: CA1 -> EC -> Cortex L5 (Grid-cell activation)
        ec_grid = self.ec.step(x_rec)
        apical_l5 = self.ec.project_to_cortex(gain=gain, content_gain=content_gain)

        # Вычисление CA1 Mismatch между текущим query и извлеченным эпизодом
        mismatch = 0.0
        if query is not None:
            mismatch = self.compute_ca1_mismatch(query, self._slots_x[best_idx : best_idx + 1])

        return {
            "allowed": True,
            "x_rec": x_rec,
            "code": completed_code,
            "ec_grid": ec_grid,
            "apical_l5": apical_l5,
            "slot_idx": best_idx,
            "mismatch": mismatch,
            "sim": best_sim,
            "theta": theta_val,
        }

    @torch.no_grad()
    def replay_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._n_stored == 0:
            return (torch.zeros(0, self.n_in, device=self.w_dg.device),
                    torch.zeros(0, self.n_dg, device=self.w_dg.device))
        k = min(int(n), self._n_stored)
        w = self._slots_err[: self._n_stored].clamp(min=1e-6)
        idx = torch.multinomial(w / w.sum(), k, replacement=True)
        return self._slots_x[idx].clone(), self._slots_code[idx].clone()

    def report(self) -> Dict[str, float]:
        return {
            "stored": float(self._n_stored),
            "err_mean": float(self._slots_err[: max(1, self._n_stored)].mean()),
            "fill": float(self._n_stored) / float(self.cfg.n_slots),
        }
