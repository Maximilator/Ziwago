"""GenesisOrganism (§III псевдокод): полный цикл 1 мс.

Фазы такта:
  0. Нейромодуляторы: RPE→DA, novelty→NE, внимание→ACh, OX/ADO
     бистабильность wake/sleep, кортизол-режим Φ_CORT, f_mod.
  1. Сенсорика: SDR → TRN-гейт → ThalamusFO (tonic/burst) → L4.
  2. Кора: FF→basal AMPA, FB→apical NMDA (SST/VIP гейт), PV k-WTA,
     HO-трансталамический драйв, GW-бродкаст, трёхфакторное обучение.
  3. Память/мотивация: HPC encode (бодр.) или SWR-реплей (сон),
     NO-поле/2-AG, BG-DDM выбор действия.
  4. Сознание: кандидаты зон → GW-Hopfield → ignition → broadcast,
     MetaHead HOT, AST интроспекция и корректирующий гейн → TRN.
  5. Гомеостаз: MR-σ̂, SOC-регуляция, IP, iSTDP, лактат.

Патологии (§4.1): эпилепсия σ>1.2>100мс, кома σ<0.7>500мс,
метаболический коллапс E<0>50% нейронов, вечный сон/бодрствование,
взрыв/коллапс весов → аварийная остановка.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ..action.basal_ganglia import BasalGanglia, BGOut
from ..chemistry.modulators import NeuroChemistry
from ..chemistry.agents import ExtendedChemistry
from ..consciousness.metrics import ConsciousnessDashboard
from ..chemistry.volume import VolumeTransmission
from ..config import GenesisConfig
from ..consciousness.attention_schema import AttentionSchema, ASTOut
from ..consciousness.meta_head import MetaHead, MetaOut
from ..consciousness.workspace import GlobalWorkspace, GWOut
from ..cortex.hierarchy import CorticalHierarchy, HierarchyOut
from ..cortex.astrocyte import AstrocyteMicrodomain
from ..emotion.limbic import LimbicOut, LimbicSystem
from ..homeostasis.controller import HomeostaticController
from ..memory.hippocampus import Hippocampus, HippoOut
from ..memory.laplace import LaplaceBank
from ..memory.phase_vsa import ThetaGammaStoryBinder
from ..thalamus.relay import ThalamicHO, ThalamusFO
from ..thalamus.trn import TRN


@dataclass
class GenesisOutput:
    t: int
    action: torch.Tensor          # выбранное действие BG (-1 = нет)
    gw_content: List[torch.Tensor]
    confidence: float
    introspection: Dict           # AST-отчёт
    sigma: float
    is_awake: bool
    sleep_pressure: float
    energy: float                 # средний ATP
    prediction_errors: Dict[str, float]
    pathology: Optional[str] = None


class GenesisOrganism:
    """Полный организм v10.0 «Genesis»."""

    def __init__(self, cfg: GenesisConfig, batch: int = 1, seed: int = 10):
        self.cfg = cfg
        self.batch = batch
        torch.manual_seed(seed)
        self.chem = NeuroChemistry(cfg.chem)
        self.hier = CorticalHierarchy(cfg, batch)
        n_total = sum(z.n_neurons for z in cfg.zones)
        self.fo = ThalamusFO(cfg.input_dim)
        self.trn = TRN(len(cfg.zones))
        l5 = lambda n: n - 2 * n // 3  # ширина L5-подмножества колонки
        self.ho = {cfg.zones[i + 1].name:
                   ThalamicHO(l5(cfg.zones[i].n_neurons),
                              cfg.zones[i].n_neurons)
                   for i in range(len(cfg.zones) - 1)}
        self.hipp = Hippocampus(cfg.hippo, seed)
        self.bg = BasalGanglia(cfg.bg, seed)
        self.gw = GlobalWorkspace(cfg.gw)
        self.meta = MetaHead(cfg.meta, cfg.gw.d_gw, seed)
        self.ast = AttentionSchema(cfg.ast, seed)
        self.volume = VolumeTransmission(n_total, cfg.chem.no_tau,
                                         cfg.chem.no_d, cfg.chem.no_radius,
                                         cfg.chem.h2s_tau)
        self.homeo = HomeostaticController(cfg.homeo)
        self.xchem = ExtendedChemistry()
        from ..memory.working import SilentWorkingMemory
        self.wm = SilentWorkingMemory(cfg.zones[-1].n_neurons)
        from ..core.interoception import InteroceptiveField
        self.intero = InteroceptiveField([z.name for z in cfg.zones])
        self.dash = ConsciousnessDashboard()
        # REGENESIS Закон 4: память и модуляторы — в физике, не в векторе
        # Астроциты микродоменов: stp_gain синапсов + θ_Ca плато дендрита
        self.astro = [AstrocyteMicrodomain(z.n_neurons) for z in cfg.zones]
        # Лапласов спектр: медленные шкалы → возбудимость L5
        self.lap = LaplaceBank(cfg.input_dim)
        # Фазово-голографический биндер сюжета (Spiker v8/v9 + Genesis)
        self.story_binder = ThetaGammaStoryBinder(
            dim=cfg.input_dim, n_slots=7, seed=seed + 5)
        self._last_vsa_ctx: Optional[torch.Tensor] = None
        # Лимбический контур (Узел 3: Гипоталамус + Амигдала BLA/CeA)
        self.limbic = LimbicSystem(cfg.input_dim, cfg.zones[-1].n_neurons,
                                   seed=seed + 9)
        self._last_limbic: Optional[LimbicOut] = None
        # Гиппокамп CA1→L5: NMDA+Mg²⁺-гейт (прайминг, не спайк)
        l5w = lambda n: n - 2 * n // 3
        self.w_hpc = {z.name: torch.randn(l5w(z.n_neurons),
                                          cfg.hippo.n_ec) * 0.2
                      for z in cfg.zones}
        self._prev_ca1: Optional[torch.Tensor] = None
        self.plasticity_blocked = False
        self.t = 0
        self.pathology: Optional[str] = None
        self._sigma_hi_ms = 0
        self._sigma_lo_ms = 0
        self._prev_spikes: Optional[torch.Tensor] = None
        self.sigma = 1.0
        self._last_meta: Optional[MetaOut] = None
        self._last_ast: Optional[ASTOut] = None

    def reset_state(self, batch: int) -> None:
        self.batch = batch
        self.hier.reset_state(batch)
        self.lap.reset_state(batch)
        self.story_binder.reset_state()
        self.limbic.reset_state()
        for a in self.astro:
            a.reset_state(batch)
        self._prev_spikes = None
        self._prev_ca1 = None
        self._last_vsa_ctx = None
        self._last_limbic = None

    # ------------------------------------------------------------------
    def _check_pathology(self, spikes_all: torch.Tensor) -> Optional[str]:
        p = self.cfg.patho
        if self.sigma > p.sigma_epilepsy:
            self._sigma_hi_ms += 1
            if self._sigma_hi_ms > p.sigma_epilepsy_ms:
                return "epilepsy"
        else:
            self._sigma_hi_ms = 0
        if self.sigma < p.sigma_coma:
            self._sigma_lo_ms += 1
            if self._sigma_lo_ms > p.sigma_coma_ms:
                return "coma"
        else:
            self._sigma_lo_ms = 0
        # Метаболический коллапс
        e = torch.cat([c.pyr.energy.flatten() for c in self.hier.cols])
        if float((e < 0.0).float().mean()) > p.energy_collapse_frac:
            return "metabolic_collapse"
        if self.chem.ticks_asleep > p.eternal_sleep_ticks:
            return "eternal_sleep"
        if self.chem.ticks_awake > p.eternal_wake_ticks:
            return "eternal_wake"
        # Веса: взрыв или коллапс
        for col in self.hier.cols:
            w_max = max(float(col.w_ff.w.abs().max()),
                        float(col.w_fb.w.abs().max()))
            if w_max > p.w_explode:
                return "weight_explosion"
            if 0.0 < w_max < p.w_collapse:
                return "weight_collapse"
        return None

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, reward: float = 0.0,
             learn: bool = True, drive: float = 0.6,
             stress: float = 0.0, patience: float = 0.0,
             is_boundary: bool = False,
             role_hint: Optional[torch.Tensor] = None) -> GenesisOutput:
        """Один такт dt=1 мс. x [B,input_dim]."""
        cfg = self.cfg
        s = self.chem.s
        self.t += 1

        # ---- Фаза 0: нейромодуляторы ----
        activity = float(self._prev_spikes.mean()) \
            if self._prev_spikes is not None else 0.05
        self.chem.update(activity=activity, rpe=reward,
                         novelty=getattr(self, "_last_novelty", 0.0),
                         attention=float(self.ast.gain.mean())
                         if hasattr(self, "ast") else 0.5,
                         drive=drive, stress=stress,
                         spikes=activity)
        e_mean = float(torch.cat(
            [c.pyr.energy.flatten() for c in self.hier.cols]).mean()) \
            if all(hasattr(c.pyr, "energy") for c in self.hier.cols) \
            else 1.0
        energy_ok = e_mean > cfg.chem.theta_energy
        # Закон 5: аденилатный долг выше порога → w_fast не обновляются
        # (пластичность — метаболически дорогой процесс).
        debt = 1.0 - e_mean
        # Фаза 2 (snnscience.docx §3.9): Теневая цена энергии Лагранжа λ_ATP(t)
        self.lambda_atp = round(max(0.0, 2.2 * debt + 0.65 * max(0.0, float(s.ado) - 0.25)), 4)
        self.plasticity_blocked = debt > cfg.chem.atp_debt_block
        f_mod = self.chem.f_mod(energy_ok)
        if self.plasticity_blocked:
            f_mod = 0.0
        awake = self.chem.is_awake

        # ---- Фаза 1: сенсорика → таламус (замыкание петли L6 + AST + MetaHead → TRN → FO) ----
        meta_gates = self._last_meta.gates if self._last_meta is not None else {}
        trn_meta_gain = float(meta_gates.get("trn_gain", 1.0))
        if awake:
            # Реальный кортикоталамический запрос от слоя L6 каждой зоны (вместо константы 0.5!)
            sector_req = torch.tensor(
                [col.l6_output() for col in self.hier.cols],
                dtype=torch.float32, device=x.device
            )
            ast_prio = (
                self.ast.gain[:len(cfg.zones)].to(x.device) * trn_meta_gain
                if self._last_ast is not None else None
            )
            trn_gate = self.trn.step(sector_req,
                                     ast_prio,
                                     vigilance=float(s.ach + s.ne))
            # Проецируем секторный гейт TRN (по числу зон) на сенсорные каналы ThalamusFO
            fo_sector_gate = (0.85 + 0.30 * (trn_gate[0] - 0.5)).clamp(0.25, 0.98)
            self.fo.gate = torch.full((self.fo.n,), float(fo_sector_gate.item()), device=x.device)
            fo_out = self.fo.step(x, s.ach, s.ne,
                                  boundary_burst=is_boundary)
        else:
            trn_gate = torch.zeros(len(cfg.zones), device=x.device)
            fo_out = torch.zeros_like(x)

        # ---- Фаза 4a (ранняя): GW-кандидаты от прошлого такта ----
        gw_b = self._gw_broadcast if hasattr(self, "_gw_broadcast") else None

        # ---- Фаза 3a: BG-дизингибиция HO + контекст PFC + обратная связь MetaHead ----
        pfc_s = self._prev_pfc if hasattr(self, "_prev_pfc") else \
            torch.zeros(self.batch, cfg.zones[-1].n_neurons)
        d1_d2 = self.xchem.d1_d2_profile(s.da)
        ne_prof = self.xchem.ne_receptor_profile(
            s.ne, surprise=self._last_meta.surprise if self._last_meta else 0.0
        )
        # Если LC/NE выдаёт сигнал Network Reset (смена правил/контекста), сбрасываем
        # застрявший бамп рабочей памяти и старый фрейм клаузы для мгновенной перестройки!
        if ne_prof["network_reset"] > 0.65 and awake:
            self.wm.bump.mul_(1.0 - 0.5 * ne_prof["network_reset"])
            self.story_binder.clause_frame.mul_(0.5)
        # PFC-контур планирования + MetaHead bg_threshold гейт + D2 гибкость:
        bg_out: BGOut = self.bg.step(
            pfc_s, s.ox, s.ado, s.ht + patience,
            da_d2_shift=d1_d2["d2_flexibility"],
            meta_threshold_boost=float(meta_gates.get("bg_threshold", 0.0)),
        )
        if reward != 0.0:
            # Опиоиды → гедонистический DA-фактор стриатума (Enk/β-End
            # усиливают обучение Go/NoGo как удовольствие от сюжета).
            opi = self.xchem.l.enkephalin + self.xchem.l.beta_endorphin
            self.bg.reward(reward, pfc_s.mean(0).detach(),
                           s.da + cfg.bg.opioid_gain * opi, s.ado)
        ho_drive: Dict[str, torch.Tensor] = {}
        l5_prev = self._prev_l5 if hasattr(self, "_prev_l5") else {}
        for z_idx, (zname, ho) in enumerate(self.ho.items(), start=1):
            src_name = [z.name for z in cfg.zones][
                [z.name for z in cfg.zones].index(zname) - 1]
            drive_in = l5_prev.get(src_name, torch.zeros(
                self.batch, ho.n_in))
            # Секторный гейт TRN также модулирует трансталамическую передачу HO между зонами!
            trn_ho_mod = float(0.6 + 0.4 * trn_gate[min(z_idx, trn_gate.numel() - 1)].item()) if awake else 0.5
            ho_drive[zname] = ho.step(
                drive_in, (0.65 + 0.35 * float(bg_out.decided)) * trn_ho_mod)

        # ---- Фаза 2: кортикальные зоны ----
        vip_mod = {z.name: float(s.ht * 0.3 + 0.1) for z in cfg.zones}
        # Расширенный химический пул: все 9 эффектных функций + рецепторные профили D1/D2 и NE
        phys_arousal = min(1.0, 0.5 * float(s.ne) + 0.5 * float(s.ach)) if awake else 0.1
        fe_damage = max(0.0, float(getattr(self, "_free_energy", 0.25)) - 0.30)
        self.xchem.update(spikes=activity, stress=stress, arousal=phys_arousal,
                          social=float(drive), threat=stress,
                          damage=fe_damage)
        da_phasic_now = float(s.da - getattr(s, "da_tonic", 0.3)) + float(getattr(self, "_intrinsic_rpe", 0.0))
        mod_ctx = {
            "ach": s.ach,
            "ne": s.ne,
            "da_phasic": da_phasic_now,
            "histamine": self.xchem.l.histamine,
            "nmda_gain": self.xchem.nmda_gain(),
            "p_rel": self.xchem.p_release_scale(),
            "vth_shift": self.xchem.vth_shift(),
            "gaba_tonic": self.xchem.gaba_tonic(),
            "disinhibition": self.xchem.disinhibition(),
            "ahp_block": self.xchem.ahp_block(),
            "snr_boost": self.xchem.snr_boost(),
            "rate_ceiling": self.xchem.rate_ceiling(),
            "d1_nmda_boost": d1_d2["d1_nmda_boost"],
            "d2_flexibility": d1_d2["d2_flexibility"],
            "network_reset": ne_prof["network_reset"],
            "bdnf": self.xchem.l.bdnf,
            "lambda_atp": self.lambda_atp,
        }
        # Закон 4: Лапласовы медленные шкалы + Фазовая VSA память ролей и сюжета
        lap_slow = self.lap.step(x).mean(0)
        vsa_out = self.story_binder.step(x, ach=s.ach, ne=s.ne, dt=cfg.dt,
                                         is_boundary=is_boundary,
                                         role_hint=role_hint,
                                         learn=learn)
        self._last_vsa_ctx = vsa_out.context_vec
        self._last_role_vec = vsa_out.role_vec
        lap_slow = lap_slow + 0.5 * vsa_out.context_vec.mean(0)
        lap_scalar = float(self.lap.slow_profile().mean())
        # Поля объёмной передачи (NO гетеро-LTP, 2-AG выталкивание, H2S) и астроциты по зонам
        gln_rate = self.xchem.glutamine_replenish_rate()
        th_shift_ext = self.xchem.theta_ltp_shift()
        z_offset = 0
        for col, ast in zip(self.hier.cols, self.astro):
            n_z = col.zcfg.n_neurons
            zf = self.volume.zone_fields(z_offset, n_z)
            z_offset += n_z
            col._mod_ctx = mod_ctx
            col._no_ltp_field = zf["no_ltp"]
            col._vol_eject = zf["eject"]
            col._hipp_in = self._prev_ca1 @ self.w_hpc[col.name].t() \
                if self._prev_ca1 is not None else None
            if self.t > 1:
                glio = ast.step(col._last_s if hasattr(col, "_last_s")
                                else torch.zeros(self.batch, n_z),
                                ne=s.ne)
                col.w_ff.stp_gain = glio.mean(0).unsqueeze(1)  # [N,1]
                col.dendrite.theta_ca = -25.0 + ast.theta_ca_shift()
            h2s_z = float(zf["h2s"].mean().item()) if zf["h2s"].numel() > 0 else 0.2
            for w in (col.w_ff, col.w_fb, col.w_rec):
                w.glutamine_rate = gln_rate
                w.set_metaplastic(gc=s.cort, e2=self.xchem.l.e2,
                                  h2s=h2s_z, zn=self.xchem.l.zn, mg=1.0,
                                  crh=self.xchem.l.crh,
                                  theta_shift_ext=th_shift_ext)
        hout: HierarchyOut = self.hier.step(
            fo_out, gw_b, ho_drive, f_mod, vip_mod, learn=learn)
        # Лапласовы медленные шкалы → adapt L5: приращение порога на
        # спайк масштабируется длинным контекстом — клетка становится
        # более/менее адаптируемой в зависимости от истории входа.
        lap_gain = 1.0 + (lap_scalar - 0.5) * 0.8
        for col in self.hier.cols:
            s_l5 = col._last_s[:, col.l5_idx] \
                if hasattr(col, "_last_s") else 0.0
            col.pyr.theta[:, col.l5_idx] += \
                s_l5 * (lap_gain - 1.0) * cfg.neuron.beta_adapt * cfg.dt
        self._prev_l5 = hout.l5
        self._prev_pfc = hout.zones[cfg.zones[-1].name].s
        self._prev_spikes = hout.spikes_all

        # ---- Уровень 3: Принцип Свободной Энергии Фристона F(t) и Эндогенная Мотивация ----
        # Свободная энергия F(t) = Σ_z Π_z · ||S_z - φ(V*_D)||² + λ_KL · KL(rate_z || ρ₀)
        precision_pi = 1.0 + 0.3 * float(s.ach + s.ne)
        # Нормированная ошибка дендритного предсказания по всем кортикальным зонам
        pred_err_sq = sum(
            float((hout.zones[col.name].rate_d / col.pyr.cfg.phi_max - hout.zones[col.name].s).pow(2).mean().item())
            for col in self.hier.cols
        ) / max(1, len(self.hier.cols))
        mean_rate = float(hout.spikes_all.mean().item())
        rho0 = 0.05
        kl_sparse = max(0.0, (mean_rate - rho0) ** 2)
        free_energy = 0.5 * precision_pi * pred_err_sq + 0.25 * kl_sparse
        if not hasattr(self, "_free_energy_ema") or self.t <= 2:
            self._free_energy_ema = free_energy
            intrinsic_rpe = 0.0
        else:
            prev_fe_ema = self._free_energy_ema
            # Эндогенный дофамин (Intrinsic Motivation): выброс DA при снижении Свободной Энергии (-dF/dt > 0)
            intrinsic_rpe = float(max(-0.5, min(0.8, (prev_fe_ema - free_energy) * 5.0)))
            self._free_energy_ema = 0.70 * prev_fe_ema + 0.30 * free_energy
        self._free_energy = free_energy
        self._intrinsic_rpe = intrinsic_rpe
        if learn and awake and intrinsic_rpe > 0.0:
            s.da = min(2.0, s.da + 0.15 * intrinsic_rpe)

        # ---- Узел 3: Лимбическая система (Гипоталамус + Амигдала BLA/CeA) ----
        limbic_out = self.limbic.step(
            fo_out if awake else x, self._prev_pfc,
            free_energy=free_energy, intrinsic_rpe=intrinsic_rpe,
            atp_mean=e_mean, ext_reward=reward, learn=learn and awake,
        )
        self._last_limbic = limbic_out
        # Эмоциональная салиентность (Flash-LTP) модулирует норадреналин и дофамин
        if awake and limbic_out.arousal > 0.1:
            s.ne = min(2.0, s.ne + 0.05 * limbic_out.arousal)
            if limbic_out.valence > 0.0:
                s.da = min(2.0, s.da + 0.06 * limbic_out.valence)

        # ---- Фаза 3b: гиппокамп, объёмная передача ----
        sens_s = hout.zones["SENS"].s if "SENS" in hout.zones else torch.zeros(self.batch, cfg.hippo.n_ec)
        sens_pad = torch.nn.functional.pad(
            sens_s, (0, max(0, cfg.hippo.n_ec - sens_s.shape[1]))
        )[:, :cfg.hippo.n_ec]
        x_ec_pad = torch.nn.functional.pad(
            x, (0, max(0, cfg.hippo.n_ec - x.shape[1]))
        )[:, :cfg.hippo.n_ec]
        ec_in = 0.55 * x_ec_pad + 0.45 * sens_pad
        err_mean = float(pred_err_sq ** 0.5)
        unc = self._last_meta.epistemic if self._last_meta else 0.5
        meta_surp = self._last_meta.surprise if self._last_meta else 0.0
        rep_prio = float(meta_gates.get("replay_priority", 0.0))
        emo_priority = abs(limbic_out.valence) + 0.5 * limbic_out.arousal
        if is_boundary:
            self.hipp.narrative_concept_trace.mul_(0.25)
        hippo_out: HippoOut = self.hipp.step(
            ec_in, s.ach, s.ado,
            learn=learn and awake,
            valence=limbic_out.valence,
            salience_boost=emo_priority + max(0.0, intrinsic_rpe),
        )
        # Эпизоды хранят grid-кодированный вектор hippo_out.ec (согласованный с w_concept)
        # и помечаются Meta-Surprise, снижением F(t), replay_priority и эмоциональной валентностью Амигдалы!
        if awake and float(x.abs().mean().item()) > 1e-4:
            self.hipp.store_episode(
                hippo_out.ec, hippo_out.dg, hippo_out.ca3,
                err_mean, unc + 0.6 * meta_surp + 0.5 * emo_priority + 0.3 * rep_prio,
                da_tag=hippo_out.vta_da_burst + max(0.0, intrinsic_rpe) + emo_priority,
            )
        self._prev_ca1 = hippo_out.ca1_out.detach()
        self._last_lap_slow = lap_slow
        if not awake:  # SWR-реплей в тройном осцилляторном замке сна (Узел 1 + Инновация 7)
            spindle_lvl = float(getattr(self, "_spindle_level", 0.7))
            def cortical_write(ec, err):
                ec_vec = ec.detach().float().flatten()
                for col in self.hier.cols:
                    # Проецируем реальный вектор реплеируемого эпизода ec через обратную проекцию w_hpc
                    # в L5/апикальное пространство зоны, обеспечивая топографическую реактивацию ансамблей!
                    w_h = self.w_hpc[col.name].to(ec_vec.device)
                    ec_pad = torch.nn.functional.pad(ec_vec, (0, max(0, w_h.shape[1] - ec_vec.numel())))[:w_h.shape[1]]
                    l5_react = torch.sigmoid(w_h @ ec_pad)  # [N_l5]
                    post_react = torch.ones(col.zcfg.n_neurons, device=ec_vec.device) * 0.5
                    post_react[col.l5_idx] = 0.5 + 0.8 * l5_react
                    for w in (col.w_ff, col.w_rec):
                        swr_boost = (1.0 + w.tag) * spindle_lvl * post_react.unsqueeze(1)
                        w.w_fast = (w.w_fast + cfg.synapse.eta * 0.08 * err
                                    * swr_boost * w.w_slow.sign() * w.mask).clamp(-0.8, 0.8)
            replayed = self.hipp.swr_replay(cortical_write)
            hippo_out.replayed = replayed
        no_field = self.volume.step(
            torch.cat([hout.zones[z.name].ca for z in cfg.zones], dim=1),
            hout.spikes_all)
        self.chem.s.no = float(no_field.mean().item())
        self.chem.s.h2s = float(self.volume.h2s.mean().item())
        ca_mean = float(torch.cat(
            [hout.zones[z.name].ca.mean(0) for z in cfg.zones]).mean())
        self.chem.ecb_update(ca_mean)
        # Нейростероидный/NO/2-AG вход нейронам
        for col in self.hier.cols:
            col.pyr.allo = (s.cort * 0.3 * torch.ones_like(col.pyr.allo))
            col.pyr.ado_i = s.ado * torch.ones_like(col.pyr.ado_i)

        # ---- Инновация 2: Замкнутый Интероцептивный Контур Insula–ACC (до GW, чтобы войти в сознание!) ----
        zone_states = {z.name: torch.tensor([
            float(hout.zones[z.name].v_s.mean()) / 100.0 + 0.5,
            float(col.pyr.energy.mean()),
            float(hout.zones[z.name].ca.mean()),
            float(hout.zones[z.name].s.mean())])
            for z, col in zip(cfg.zones, self.hier.cols)}
        intero_out = self.intero.read(zone_states, pfc_spikes=self._prev_pfc, learn=learn)
        # Упреждающая аллостатическая регуляция (Insula → Гипоталамус/Кора):
        # если аллостатический драйв высок, организм заранее повышает метаболический тормоз
        # и мягко наращивает давление сна до наступления коллапса АТФ!
        if intero_out.allostatic_drive > 0.35:
            s.ado = min(1.0, s.ado + 0.008 * intero_out.allostatic_drive)
            for col in self.hier.cols:
                col.pyr.met_gain = min(2.0, getattr(col.pyr, "met_gain", 1.0) + 0.02 * intero_out.allostatic_drive)

        # ---- Фаза 4: Симфоническая Таламо-Кортикальная Интеграция (Узел 2 + Insula + MetaHead.gates) ----
        candidates = []
        lfp_phases = []
        for z, col in zip(cfg.zones, self.hier.cols):
            # Комбинируем спайки и подпороговое дендритное состояние φ(V*_D)/φ_max,
            # чтобы фазовый портрет зоны всегда был виден в Симфоническом Аккорде!
            z_out = hout.zones[z.name]
            srate = 0.6 * z_out.s.mean(0) + 0.4 * (z_out.rate_d.mean(0) / col.pyr.cfg.phi_max)
            cand = torch.nn.functional.pad(
                srate, (0, max(0, cfg.gw.d_gw - srate.numel())))[
                :cfg.gw.d_gw]
            candidates.append(cand)
            lfp_phases.append(float(col.lfp.layer_lfp.get("L23", 0.0)))
        # Четвёртый слот GW: эпизодический контекст CA1 (+ висцеральный сигнал Insula при аллостатической тревоге!)
        ca1_pad = torch.nn.functional.pad(
            hippo_out.ca1_out.mean(0)[:cfg.gw.d_gw],
            (0, max(0, cfg.gw.d_gw - hippo_out.ca1_out.shape[1]))
        )[:cfg.gw.d_gw]
        if intero_out.allostatic_drive > 0.35:
            ins_pad = torch.nn.functional.pad(
                intero_out.state_vec[:cfg.gw.d_gw],
                (0, max(0, cfg.gw.d_gw - intero_out.state_vec.numel()))
            )[:cfg.gw.d_gw].to(ca1_pad.device)
            candidates.append(0.85 * ca1_pad + 0.15 * ins_pad)
        else:
            candidates.append(ca1_pad)
        lfp_phases.append(float(self.hipp.theta_osc()))
        gw_out: GWOut = self.gw.compete(
            candidates, s.ne, lfp_phases=lfp_phases,
            beta_boost=float(meta_gates.get("beta_gw_boost", 0.0)),
        )
        self._gw_broadcast = gw_out.broadcast
        # Передаём в BA10 MetaHead комбинированную ошибку предсказания (кора + CA1 mismatch)
        # и текущий сенсорный кадр, чтобы MetaHead мгновенно отличал знакомый опыт от нового!
        meta_err_signal = min(1.0, 0.5 * err_mean + 0.5 * min(1.0, hippo_out.mismatch))
        meta_out = self.meta.step(gw_out.broadcast, gw_out.content,
                                  meta_err_signal, reward, sensory_frame=x)
        self._last_meta, self._last_novelty = meta_out, meta_out.novelty
        ast_out = self.ast.step(meta_out.novelty, err_mean, drive,
                                actual_gain=float(trn_gate.mean())
                                if awake else 0.0,
                                actual_focus=gw_out.winner_zone
                                if gw_out.winner_zone >= 0 else 0)
        ast_out.report["self_state"] = intero_out.summary
        ast_out.report["intero_activity"] = float(intero_out.state_vec.mean())
        ast_out.report["intero_error"] = round(intero_out.intero_error, 4)
        ast_out.report["allostatic_drive"] = round(intero_out.allostatic_drive, 4)
        ast_out.report["free_energy"] = round(self._free_energy, 4)
        ast_out.report["intrinsic_rpe"] = round(self._intrinsic_rpe, 4)
        ast_out.report["lambda_atp"] = round(self.lambda_atp, 4)
        ast_out.report["gossip_plv"] = round(float(getattr(hout, "gossip_plv", 0.0)), 4)
        ast_out.report["polychronous_groups_active"] = int(getattr(hout, "polychronous_groups_active", 0))
        ast_out.report["lfp"] = {col.name: dict(col.lfp.layer_lfp)
                                 for col in self.hier.cols}
        ast_out.report["symphonic_coherence"] = gw_out.coherence_weights
        ast_out.report["limbic"] = {
            "valence": round(limbic_out.valence, 4),
            "arousal": round(limbic_out.arousal, 4),
            "d_energy": round(limbic_out.d_energy, 4),
            "d_safety": round(limbic_out.d_safety, 4),
            "d_novelty": round(limbic_out.d_novelty, 4),
        }
        ast_out.report["receptors"] = {
            "d1_nmda_boost": round(d1_d2["d1_nmda_boost"], 4),
            "d2_flexibility": round(d1_d2["d2_flexibility"], 4),
            "network_reset": round(ne_prof["network_reset"], 4),
        }
        ast_out.report["autobiographical_concepts"] = {
            "active_concepts": int((self.hipp.concept_salience > 0.2).sum().item()),
            "semantic_bridges": int((self.hipp.w_semantic_graph > 0.05).sum().item()),
        }
        self._last_intero = intero_out
        self._last_ast = ast_out
        self._last_mismatch = hippo_out.mismatch

        # ---- Фаза 5: гомеостаз и зональная критичность ----
        if self._prev_spikes is not None:
            self.sigma = self.homeo.monitor.push(
                hout.spikes_all.mean(0),
                self._prev_spikes.mean(0))
        zone_sigmas = self.homeo.monitor.push_zones(
            {
                z.name: (
                    0.65 * hout.zones[z.name].s.mean(0)
                    + 0.35 * (hout.zones[z.name].rate_d.mean(0) / col.pyr.cfg.phi_max)
                )
                for z, col in zip(cfg.zones, self.hier.cols)
            }
        )
        ast_out.report["zone_sigmas"] = {k: round(v, 4) for k, v in zone_sigmas.items()}
        hom_rep = self.homeo.regulate(self.hier, self.sigma)
        # Рабочая память PFC: спайк-паттерн → STP-фацилитация (без огня), с усилением от D1
        if awake:
            self.wm.write(self._prev_pfc > 0, strength=0.1 * d1_d2["d1_nmda_boost"])
        self.wm.tick(dt=1.0, active_hold=awake)
        # Нейромод-эффекты: novelty → NE/DA-бёрсты от CA1
        s.ne += hippo_out.lc_ne_burst * 0.1
        s.da += hippo_out.vta_da_burst * 0.05

        self.pathology = self._check_pathology(hout.spikes_all)
        errors = {z.name: float(hout.errors[z.name].abs().mean())
                  for z in cfg.zones}
        # Дашборд сознания: PCI/R_bc/γ_ign/ECE/каузальная плотность
        r_bc = float((gw_out.broadcast.abs().mean() > 1e-4)
                     * sum(float(hout.errors[z.name].abs().mean() > 1e-6)
                           for z in cfg.zones) / max(1, len(cfg.zones)))
        self.dash.push(hout.spikes_all.flatten(),
                       float(gw_out.ignition),
                       float(gw_out.broadcast.abs().mean() > 1e-6),
                       meta_out.confidence, float(1.0 - min(1.0, err_mean)),
                       float(1.0 - ast_out.expected_accuracy), r_bc)
        if not hasattr(self, "_last_dash") or self.t % 20 == 0:
            self._last_dash = self.dash.report(
                ffr=float(sum(float(c.w_fb.w.abs().mean()) for c in self.hier.cols)
                          / max(1e-9, sum(float(c.w_ff.w.abs().mean())
                                          for c in self.hier.cols))))
        return GenesisOutput(
            t=self.t, action=bg_out.action,
            gw_content=gw_out.content, confidence=meta_out.confidence,
            introspection=ast_out.report, sigma=self.sigma,
            is_awake=awake, sleep_pressure=s.ado,
            energy=float(torch.cat(
                [c.pyr.energy.flatten() for c in self.hier.cols]).mean()),
            prediction_errors=errors, pathology=self.pathology)

    # ------------------------------------------------------------------
    def run_episode(self, xs: List[torch.Tensor], learn: bool = True,
                    drive: float = 0.6) -> List[GenesisOutput]:
        outs = []
        for x in xs:
            outs.append(self.step(x, learn=learn, drive=drive))
            if self.pathology:
                break
        return outs

    def sleep_cycle(self, ticks: int = 200) -> Dict:
        """Узел 1 + Инновация 7: Двухфазный Сон (NREM Тройной Замок + REM Творческая Рекомбинация):
        1) Фаза NREM (первые 70% тактов, ACh↓, Ado↑):
           Медленные волны SO (1 Гц) × таламические веретёна (12 Гц) × топографический SWR-реплей
           переносят тегированные веса w_fast → w_slow с даунскейлингом SHY (Tononi & Cirelli 2014).
        2) Фаза REM (последние 30% тактов, ACh↑↑, NE=0):
           Химерная суперпозиция далёких значимых эпизодов (Lewis et al. 2018), формирующая
           семантические мосты в концептуальном графе и рекуррентных связях коры."""
        import math
        pre_sleep_bdnf = float(self.xchem.l.bdnf)
        pre_sleep_novelty = float(self._last_meta.novelty) if self._last_meta else 0.5
        self.chem.is_awake = False
        self.chem.s.ado = 0.8
        self.chem.s.ach = 0.1
        nrem_ticks = max(1, int(round(ticks * 0.70)))
        rem_ticks = max(1, ticks - nrem_ticks)
        n_replayed = 0
        spindle_accum = 0.0
        zone_err_accum = {z.name: 0.0 for z in self.cfg.zones}

        # --- Фаза 1: NREM (Медленноволновой сон) ---
        for i in range(nrem_ticks):
            so_up_state = 0.5 * (1.0 + math.sin(2.0 * math.pi * 1.0 * i / max(1, nrem_ticks)))
            spindle_vec = self.fo.spindle(i, freq_hz=12.0) * so_up_state
            self._spindle_level = float(spindle_vec.mean().item())
            spindle_accum += self._spindle_level
            out = self.step(torch.zeros(self.batch, self.cfg.input_dim),
                            learn=True)
            for zname, err_v in out.prediction_errors.items():
                zone_err_accum[zname] += err_v / max(1, ticks)
            n_replayed += len(getattr(self.hipp, "_dbg_replayed", []))

        eff_bdnf = max(pre_sleep_bdnf, float(self.xchem.l.bdnf))
        mean_spindle = max(0.5, spindle_accum / max(1, nrem_ticks))
        self.hier.consolidate(0.5, spindle_gain=mean_spindle,
                              zone_errors=zone_err_accum,
                              microglia_boost=self.xchem.microglial_prune_boost(),
                              bdnf=eff_bdnf)

        # --- Фаза 2: REM (Парадоксальный сон — Химерный синтез и семантическая интеграция) ---
        self.chem.s.ach = 0.95
        self.chem.s.ne = 0.02
        self._spindle_level = 0.5

        def rem_cortical_bridge(chimera_ec: torch.Tensor, bridge_err: float) -> None:
            ec_v = chimera_ec.detach().float().flatten()
            for col in self.hier.cols:
                w_h = self.w_hpc[col.name].to(ec_v.device)
                ec_pad = torch.nn.functional.pad(ec_v, (0, max(0, w_h.shape[1] - ec_v.numel())))[:w_h.shape[1]]
                l5_act = torch.sigmoid(w_h @ ec_pad)
                # Укрепляем рекуррентные ассоциативные мосты между совместно активированными L5-нейронами
                bridge_mat = torch.outer(l5_act, l5_act)
                idx = col.l5_idx
                col.w_rec.w_slow[idx[:, None], idx[None, :]] = (
                    col.w_rec.w_slow[idx[:, None], idx[None, :]]
                    + 0.015 * min(1.0, bridge_err) * bridge_mat * col.w_rec.mask[idx[:, None], idx[None, :]].float()
                ).clamp(col.w_rec.cfg.w_min, col.w_rec.cfg.w_max)
            ext_fn = getattr(self, "_rem_motor_bridge_fn", None)
            if callable(ext_fn):
                ext_fn(ec_v, bridge_err)

        rem_stats = self.hipp.rem_recombine(cortical_bridge_fn=rem_cortical_bridge)
        for _ in range(min(rem_ticks, 4)):
            out = self.step(torch.zeros(self.batch, self.cfg.input_dim), learn=True)
            for zname, err_v in out.prediction_errors.items():
                zone_err_accum[zname] += err_v / max(1, ticks)

        # Адаптивный нейрогенез DG во сне, управляемый накопленной новизной опыта и BDNF
        nov_lvl = max(pre_sleep_novelty, self._last_meta.novelty if self._last_meta else 0.5)
        n_born = self.hipp.neurogenesis(novelty_level=nov_lvl,
                                        cortisol=self.chem.s.cort,
                                        bdnf=eff_bdnf)
        sprouted_tot = sum(
            w.total_sprouted for col in self.hier.cols for w in (col.w_ff, col.w_fb, col.w_rec)
        )
        pruned_tot = sum(
            w.total_pruned for col in self.hier.cols for w in (col.w_ff, col.w_fb, col.w_rec)
        )
        self.chem.s.ado = 0.3
        self.chem.s.ox = 0.6
        self.chem.s.ach = 0.5
        self.chem.is_awake = True
        return {
            "ticks": ticks,
            "nrem_ticks": nrem_ticks,
            "rem_ticks": rem_ticks,
            "sigma": self.sigma,
            "episodes": len(self.hipp.episodes),
            "spindle_mean": round(mean_spindle, 4),
            "dg_neurogenesis": n_born,
            "bdnf": round(eff_bdnf, 4),
            "sprouted_synapses": sprouted_tot,
            "pruned_synapses": pruned_tot,
            "rem_chimera_similarity": rem_stats["rem_chimera_similarity"],
            "semantic_bridges": rem_stats["semantic_bridges"],
            "interleaved_pairs": rem_stats.get("interleaved_pairs", 0),
        }
