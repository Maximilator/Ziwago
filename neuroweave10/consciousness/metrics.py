"""Дашборд метрик сознания (Knowledge/consciousness2.md §4).

Функциональные измерители по пяти теориям:
  GWT: R_bc — широта рассылки (доля зон, чей апикал изменился после
       бродкаста), γ_ign — измеренная крутизна ignition-сигмоиды
  IIT: PCI-прокси — сложность Лемпеля-Зива спайкового потока после
       точечного возмущения (Casali 2013, адаптировано);
       causal density — доля значимых лаговых корреляций
  RPT: RDI — индекс зависимости от рекуррентности (Δточности при
       W_fb=0), FFR — отношение обратных к прямым связям
  HOT: ECE — ошибка калибровки уверенности; M-ratio (в MetaHead)
  AST: L_schema, ΔU — прирост контроля внимания

Прокси, не точный Φ — честное ограничение зафиксировано в отчёте.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Optional

import torch


def lempel_ziv_complexity(bits: torch.Tensor) -> float:
    """PCI-прокси: нормированная сжимаемость бинарного потока."""
    raw = bits.flatten().to(torch.int8).cpu().numpy().tobytes()
    if len(raw) == 0:
        return 0.0
    return len(zlib.compress(raw, 9)) / len(raw)


def broadcast_reach(broadcast: Optional[torch.Tensor],
                    apical_before: Dict[str, torch.Tensor],
                    apical_after: Dict[str, torch.Tensor]) -> float:
    """R_bc: доля зон, чей апикальный потенциал изменил бродкаст."""
    if broadcast is None or not apical_before:
        return 0.0
    changed = 0
    for name, vb in apical_before.items():
        va = apical_after.get(name)
        if va is not None and float((va - vb).abs().mean()) > 1e-4:
            changed += 1
    return changed / max(1, len(apical_before))


def causal_density(spike_hist: torch.Tensor, lag: int = 3) -> float:
    """Доля |корреляций с лагом| > 0.1 между каналами — прокси Гранжера."""
    x = spike_hist.float()
    if x.shape[0] <= lag + 2 or x.shape[1] < 2:
        return 0.0
    a = x[:-lag] - x[:-lag].mean(0)
    b = x[lag:] - x[lag:].mean(0)
    num = (a.t() @ b) / max(1, a.shape[0])
    den = (a.norm(dim=0, keepdim=True).t() * b.norm(dim=0, keepdim=True)
           ).clamp(min=1e-8)
    corr = (num / den).abs()
    return float((corr > 0.1).float().mean())


def ignition_gamma(scores: List[float], access: List[float]) -> float:
    """Подгонка γ сигмоиды A=σ(γ(S−S_th)) к измеренным парам."""
    if len(scores) < 4:
        return 0.0
    import numpy as np
    s = np.asarray(scores); a = np.asarray(access).clip(1e-6, 1 - 1e-6)
    logits = np.log(a / (1 - a))
    # γ = cov(s, logits)/var(s)
    v = s.var()
    return float(np.cov(s, logits)[0, 1] / v) if v > 1e-9 else 0.0


def ece(confidences: List[float], accuracies: List[float],
        bins: int = 5) -> float:
    """Expected Calibration Error."""
    if len(confidences) < bins:
        return 0.0
    import numpy as np
    c = np.asarray(confidences); a = np.asarray(accuracies)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c >= lo) & (c < lo + 1.0 / bins)
        if m.any():
            e += m.mean() * abs(c[m].mean() - a[m].mean())
    return float(e)


class ConsciousnessDashboard:
    """Накопитель метрик по тактам + итоговый отчёт."""

    def __init__(self):
        self.spike_hist: List[torch.Tensor] = []
        self.ignition_scores: List[float] = []
        self.ignition_access: List[float] = []
        self.confs: List[float] = []
        self.accs: List[float] = []
        self.schema_losses: List[float] = []
        self.r_bc_hist: List[float] = []
        self.hist_max = 500

    def push(self, spikes: torch.Tensor, ignition_score: float,
             access: float, conf: float, acc: float,
             schema_loss: float, r_bc: float) -> None:
        self.spike_hist.append(spikes.detach().flatten().cpu())
        if len(self.spike_hist) > self.hist_max:
            self.spike_hist.pop(0)
        for lst, v in ((self.ignition_scores, ignition_score),
                       (self.ignition_access, access),
                       (self.confs, conf), (self.accs, acc),
                       (self.schema_losses, schema_loss),
                       (self.r_bc_hist, r_bc)):
            lst.append(v)
            if len(lst) > self.hist_max:
                lst.pop(0)

    def report(self, ffr: float, rdi: float = 0.0) -> Dict[str, float]:
        spikes = torch.stack(self.spike_hist[-200:]) \
            if self.spike_hist else torch.zeros(1, 1)
        pci = lempel_ziv_complexity((spikes > 0.5).float())
        return {
            "pci_lz": pci,
            "causal_density": causal_density(spikes),
            "broadcast_reach": (sum(self.r_bc_hist[-50:])
                                / max(1, len(self.r_bc_hist[-50:]))),
            "gamma_ignition": ignition_gamma(self.ignition_scores,
                                             self.ignition_access),
            "ece": ece(self.confs, self.accs),
            "l_schema": (sum(self.schema_losses[-50:])
                         / max(1, len(self.schema_losses[-50:]))),
            "ffr": ffr, "rdi": rdi,
        }
