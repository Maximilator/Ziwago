"""Единый организм: синергия v8+v9+v10 (не импорты ради названий).

Путь токен→действие:
  v9 TemporalPhaseEncoder  — токен→спайк-поезд + салиентность (вход)
  v8 sparse_drive/k_wta    — разреженный драйв в дендритном дереве (экономия)
  v10 GenesisOrganism      — таламус→кортекс→гиппокамп→БГ→GW→мета (тело)
  v9 LaplaceContextReadout — PFC→логиты словаря (выход)
Замыкание: CE-ошибка → reward→DA→TD-пластичность БГ и кортекса —
модель читает поток, ошибается, учится, спит-консолидируется.
"""
from .pipeline import UnifiedOrganism, UnifiedConfig, UnifiedReport

__all__ = ["UnifiedOrganism", "UnifiedConfig", "UnifiedReport"]
