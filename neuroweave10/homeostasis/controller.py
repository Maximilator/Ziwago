"""Гомеостатический контроллер (§1.5): σ≈1 во всех условиях.

Механизмы:
  MR-оценщик σ̂ → если σ>1: синаптическая депрессия (Levina),
                 ↓весов; σ<1: потенциация/нейрогенез;
  Intrinsic plasticity: ΔΘ = η_IP·(x - H_IP) — порог к целевой частоте;
  Vogels 2011 iSTDP: ингибиторные веса следят за пост-частотой ρ₀;
  SORN-нормировка: Σw_in ≈ const (синаптическое масштабирование).
"""
from __future__ import annotations

from typing import Dict, List

import torch

from ..config import HomeostasisCfg
from ..core.criticality import CriticalityMonitor


class HomeostaticController:
    """Регуляция критичности по всем зонам."""

    def __init__(self, cfg: HomeostasisCfg):
        self.cfg = cfg
        self.monitor = CriticalityMonitor(cfg.mr_kmax)
        self._prev_spikes: Dict[str, torch.Tensor] = {}
        self._tick = 0

    # ------------------------------------------------------------------
    def regulate(self, hierarchy, sigma: float,
                 dt_ms: float = 1.0) -> Dict[str, float]:
        """Применить регуляцию по разделённым шкалам времени (Menesse et al. 2024; §1.5):
          1. Быстрая шкала (каждый такт, ~10-50 мс): ресурс везикул u_syn (Levina 2007),
             а синапс-специфичный iSTDP Вогельса работает локально в CorticalColumn.step().
          2. Средняя шкала (покомпонентная IP, ~100-500 мс): индивидуальные пороги Θ_i
             следуют за медленным кальциевым следом нейрона ca_i с учётом τ_m(g) зоны.
          3. Медленная шкала (раз в wnorm_every тактов): мультипликативное синаптическое
             масштабирование Тёрриджано на w_slow без разрушения выученной структуры."""
        c = self.cfg
        self._tick += 1
        report = {"sigma": sigma, "depression": 0.0, "ip": 0.0,
                  "istdp": 0.0}
        do_wnorm = (c.wnorm_every > 0 and self._tick % c.wnorm_every == 0)
        zone_sigmas = getattr(self.monitor, "zone_sigmas", {})
        for col in hierarchy.cols:
            pool = col.pyr
            has_z = col.name in zone_sigmas
            z_sig = float(zone_sigmas.get(col.name, sigma))
            eff_sigma = z_sig if has_z else sigma
            # Коэффициент темпорального масштабирования зоны (Murray/Hasson):
            # медленные зоны (PFC) меняют порог IP плавнее, чем быстрые (SENS)
            tau_scale = min(1.0, 20.0 / max(20.0, float(col.zcfg.tau_m))) ** 0.35

            # 1) Быстрая шкала (Levina et al. 2007): модуляция кратковременного ресурса u_syn
            #    при отклонении σ от критичности (без ежетактового стирания w_slow!)
            if eff_sigma > c.target_sigma + c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.u_syn.mul_(0.996)
                pool.theta = (pool.theta + 0.06).clamp(-56.0, -42.0)
                col.w_pv_in = (col.w_pv_in * 1.015).clamp(0.05, 3.0)
                if has_z:
                    zone_sigmas[col.name] = float(eff_sigma - 0.12 * (eff_sigma - c.target_sigma))
                report["depression"] += 1.0
            elif eff_sigma < c.target_sigma - c.sigma_tol:
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    w.u_syn = (w.u_syn + 0.012).clamp(max=1.0)
                pool.theta = (pool.theta - 0.08).clamp(-56.0, -42.0)
                col.w_pv_in = (col.w_pv_in * 0.985).clamp(0.05, 3.0)
                if has_z:
                    zone_sigmas[col.name] = float(eff_sigma + 0.12 * (c.target_sigma - eff_sigma))

            # 2) Средняя шкала — покомпонентная Информационная IP Триша (Triesch 2005):
            #    оптимизирует экспоненциальное распределение частот при фиксированном бюджете
            if hasattr(pool, "ca") and pool.ca is not None:
                rate_vec = pool.ca.mean(0)  # [N]
                rate_mean = float(rate_vec.mean().item())
                mu = max(0.02, float(c.ip_target))
                rate_err = rate_vec - mu
                ent_grad = mu * ((2.0 + 1.0 / mu) * rate_vec - (rate_vec ** 2) / mu - 1.0).clamp(-0.3, 0.3)
                d_theta = c.ip_rate * tau_scale * (0.75 * rate_err + 0.25 * ent_grad)
                pool.theta = (pool.theta + d_theta.unsqueeze(0)).clamp(-56.0, -42.0)
                report["ip"] += float(d_theta.abs().mean().item())
            else:
                rate_mean = 0.0

            # 3) Отчёт по тормозному балансу iSTDP (сам матричный iSTDP выполняется
            #    локально в CorticalColumn.step() без дублирующего скалярного сдвига!)
            d_inh = c.inh_stdp_rate * (rate_mean - 0.06)
            report["istdp"] += abs(d_inh)

            # 4) Медленная шкала (раз в wnorm_every тактов) — мультипликативное
            #    синаптическое масштабирование Тёрриджано (сохраняет пропорции w_slow)
            if do_wnorm:
                sigma_factor = 0.995 if eff_sigma > c.target_sigma + c.sigma_tol else (
                    1.005 if eff_sigma < c.target_sigma - c.sigma_tol else 1.0
                )
                for w in (col.w_ff, col.w_fb, col.w_rec):
                    nrm = w.w_slow.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    tgt = getattr(w, "target_row_norm",
                                  torch.full_like(nrm, 0.8)).clamp(min=0.25, max=3.0)
                    scale = (tgt / nrm).clamp(0.85, 1.15) * sigma_factor
                    w.w_slow.mul_(0.9 * 1.0 + 0.1 * scale).clamp_(w.cfg.w_min, w.cfg.w_max)

            # 5) Гомеостаз критичности → метаболический тормоз KATP/GIRK:
            if eff_sigma < c.target_sigma - c.sigma_tol:
                pool.met_gain = max(0.1, getattr(pool, "met_gain", 1.0) * 0.995)
            elif eff_sigma > c.target_sigma + c.sigma_tol:
                pool.met_gain = min(2.0, getattr(pool, "met_gain", 1.0) * 1.005)
            report["met_gain"] = float(getattr(pool, "met_gain", 1.0))
        return report
