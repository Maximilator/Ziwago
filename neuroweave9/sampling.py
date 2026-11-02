"""sampling.py — стохастическая стратегия декодирования моторного контура NEUROWEAVE.

Зачем это нужно.

Жадный argmax по выходам коры даёт вырожденный текст («clean clean clean»):
детерминированный выбор всегда ведёт в самый вероятный токен, и моторный контур
физически не может уйти в альтернативную гипотезу. Это не дефект весов — это
дефект стратегии выбора.

В биологии моторный выбор (стриатум → паллидум → таламус) стохастичен: кору
представляют РАСПРЕДЕЛЕНИЕМ активностей, и действие выбирается из этого
распределения (softmax-политика), а не «победил один нейрон». Гиппокампальный
реколл при этом меняет само распределение (подаёт апикальный вход коры), а
значит — меняет и то, какие гипотезы вообще доступны для выбора.

Здесь мы даём контуру ровно эту возможность: выбор действия из распределения
активностей коры с фильтрацией температурой / top-k / top-p.

Порядок фильтрации (стандартный для LM-декодеров):
    1. температура    — перекалибровка остроты распределения (logits / T)
    2. top-k          — оставить k наиболее вероятных действий
    3. top-p (нуклеус)— оставить минимальный префикс с суммарной вероятностью >= p
    4. выбор          — категориальная выборка из фильтрованного распределения

Дефолт в контуре — ЖАДНЫЙ выбор (temperature <= 0), поэтому поведение
`BasalGangliaGate` по умолчанию не меняется ни на бит: все отчёты и откаты
остаются воспроизводимыми. Стохастика включается явно через
`BasalGangliaGate.configure_sampling(...)`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

class AdaptiveTemperature:
    """Управляемое temperature scaling по состоянию текста.

    Логика: если декодер УВЕРЕН (большой margin) И текст при этом ОДНООБРАЗЕН
    (низкое лексическое разнообразие в скользящем окне) -- поднимаем T
    (размываем уверенность, даём шанс редким токенам). Когда разнообразие
    восстановилось -- возвращаем T к базовому значению.

    Это не «подкрутка логитов», а обратная связь декодера со статистикой
    собственного вывода: T растёт ровно тогда, когда текст начинает
    вырождаться в низкоэнтропийный дрейф.
    """

    def __init__(self, t_min: float = 1.0, t_max: float = 1.15,
                 step: float = 0.01, margin_trigger: float = 6.0,
                 diversity_window: int = 64, diversity_trigger: float = 0.35):
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.step = float(step)
        self.margin_trigger = float(margin_trigger)
        self.diversity_window = int(diversity_window)
        self.diversity_trigger = float(diversity_trigger)
        self.t = float(t_min)

    @staticmethod
    def diversity(tokens, window: int) -> float:
        """Доля уникальных токенов в последнем окне (1.0 = все разные)."""
        if not tokens:
            return 1.0
        w = list(tokens[-window:])
        return len(set(w)) / max(1, len(w))

    def update(self, margin: float, tokens) -> float:
        div = self.diversity(tokens, self.diversity_window)
        if margin >= self.margin_trigger and div <= self.diversity_trigger:
            self.t = min(self.t_max, self.t + self.step)
        else:
            self.t = max(self.t_min, self.t - self.step)
        return self.t


__all__ = ["filter_logits", "sample_token", "SamplingConfig", "AdaptiveTemperature"]


class SamplingConfig:
    """Конфигурация стратегии декодирования (value-объект для логов и отчётов)."""

    __slots__ = ("temperature", "top_k", "top_p", "seed")

    def __init__(self, temperature: float = 0.0, top_k: int = 0,
                 top_p: float = 1.0, seed: Optional[int] = None):
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.seed = None if seed is None else int(seed)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def describe(self) -> str:
        if self.greedy:
            return "жадный (argmax)"
        return (f"T={self.temperature:g}, top_k={self.top_k}, "
                f"top_p={self.top_p:g}, seed={self.seed}")

    def __repr__(self) -> str:  # pragma: no cover - только для логов
        return f"SamplingConfig({self.describe()})"


def filter_logits(logits: torch.Tensor, temperature: float = 1.0,
                  top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Отфильтровать логиты: температура → top-k → nucleus (top-p).

    Возвращает логиты, в которых запрещённые действия помечены -inf;
    softmax по результату даёт нужное усечённое распределение.
    """
    if logits.dim() != 2:
        raise ValueError(f"ожидается (batch, vocab), получено {tuple(logits.shape)}")
    if temperature <= 0.0:
        return logits

    x = logits / float(temperature)

    # ── top-k: оставляем k самых вероятных действий ────────────────────────
    n = x.shape[-1]
    if top_k and top_k > 0 and top_k < n:
        kth = torch.topk(x, k=int(top_k), dim=-1).values[..., -1:]
        x = x.masked_fill(x < kth, float("-inf"))

    # ── top-p (nucleus): минимальный префикс с суммарной вероятностью >= p ──
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(x, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # выкидываем всё, что выходит за ядро (токен, на котором пересекли p, оставляем)
        remove = (cumulative - sorted_probs) > float(top_p)
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        x = torch.empty_like(x).scatter_(-1, sorted_idx, sorted_logits)

    return x


def sample_token(logits: torch.Tensor, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Выбрать действие (индекс токена) из распределения активностей.

    temperature <= 0 → жадный argmax (поведение по умолчанию).
    Иначе — выборка из усечённого softmax-распределения.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    filtered = filter_logits(logits, temperature=temperature,
                             top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    # защита от вырожденной строки (всё -inf) — fallback на жадный выбор
    degenerate = probs.sum(dim=-1, keepdim=True) <= 0.0
    if bool(degenerate.any().item()):
        fallback = logits.argmax(dim=-1, keepdim=True)
        probs = torch.where(degenerate, F.one_hot(fallback, probs.shape[-1]).to(probs.dtype),
                            probs)
    picked = torch.multinomial(probs, num_samples=1, generator=generator)
    return picked.squeeze(-1)
