"""CortexOrganism — организм NEUROWEAVE v9.0 «Cortex» (The Great Merge).

Единица топологии — ЛАМИНАРНАЯ КОЛОНКА (NEUROWEAVE 8), а не плоская зона
(Spiker v6.0). Три области: A (левое, синтаксис/токены), B (левое,
локальная структура), C (правое, контекст; токенов не видит — получает
код Зоны A).

Из Spiker сохранены МАТЕМАТИЧЕСКИ ПРОВЕРЕННЫЕ компоненты:
  * LaplaceTimingBank — мультимасштабный спектр контекста декодера;
  * BasalGangliaGate — дрифт-диффузионный темп эмиссии токенов;
  * ThetaGammaOscillator — PAC-гейт окна Corpus Callosum;
  * метод обучения «одна большая партия» (train_cortex.py).

Из NEUROWEAVE 8 сохранены ИННОВАЦИОННЫЕ компоненты:
  * LaminarColumn (L4->L2/3->L5->L6 + PV/SST, дендритный AND-гейт);
  * CorpusCallosum с настоящей задержкой и GABA-фильтром;
  * phase.py: ThetaGammaBinder + PhaseVSA — структурная синхронизация
    полушарий (sync.py);
  * NeuroChemistry + MetaController (сетпоинты ACh/5-HT, NE-аларм);
  * CreditPipeline (событийный DA, Tag-and-Capture, нормировка следов);
  * Hippocampus DG->CA3->CA1 с приоритизированным реплеем;
  * SleepProtocol (реплей -> консолидация -> прунинг -> пробуждение);
  * ActiveInference (EFE-внимание -> arousal-гейт L5), StructuralPlasticity,
    EWCRegularizer, int8-квантизация.

Выход step() совместим с пайплайном Spiker (out["left"]["zone_a"]["spikes"]
и т.д.) — это позволяет переиспользовать измеренный обучающий контур без
изменений его семантики.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuroweave8.config import (
    CallosumCfg, ColumnCfg, HippoCfg, OrganismCfg, PhaseCfg,
    PlasticityCfg, SleepCfg, StructuralCfg, WorkspaceCfg, NeuronCfg, ModCfg,
)
from neuroweave8.microcircuit.column import LaminarColumn
from neuroweave8.modulation.chemistry import NeuroChemistry
from neuroweave8.modulation.controller import MetaController
from neuroweave8.topology.callosum import CorpusCallosum
from neuroweave8.topology.workspace import GlobalWorkspace
from neuroweave8.learning.credit import CreditPipeline
from neuroweave8.learning.sleep import SleepProtocol
from neuroweave8.learning.ewc import EWCRegularizer
from neuroweave8.learning.structural import StructuralPlasticity
from .hippocampus import Hippocampus
from neuroweave8.organism.slow_loop import SlowLoopController
from neuroweave8.organism.active_inference import ActiveInference

from ..memory.laplace import LaplaceTimingBank
from ..action.basal_ganglia import BasalGangliaGate
from ..modulation.pacemaker import ThetaGammaOscillator
from ..modulation.orexin import OrexinInteroceptiveLoop
from ..modulation.no_field import NitricOxideField
from ..modulation.son import SonBus
from ..topology.thalamus import Thalamus
from ..encoding import TemporalPhaseEncoder
from .astrocyte import TripartiteAstrocyte
from .sync import HemisphereSync
from .endocannabinoid import EndocannabinoidSystem
from .serotonin import SerotoninSystem


def cortex_cfg(
    vocab: int,
    input_dim: Optional[int] = None,
    scale: float = 1.0,
    seed: int = 42,
    wm_slots: int = 7,
) -> OrganismCfg:
    """Конфигурация коры v9: A,B (левое) + C (правое) из ламинарных колонок.

    scale=1.0 ~ пресет Spiker 64/48/64 по числу нейронов коры (152/152/152).
    """
    n_in = int(input_dim or vocab)
    s = float(scale)
    l23 = max(8, int(round(48 * s)))

    A = ColumnCfg(name="A", side="left", n_in=n_in, n_out=vocab,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=15.6, wm_slots=wm_slots)
    B = ColumnCfg(name="B", side="left", n_in=n_in, n_out=vocab,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=15.6, wm_slots=wm_slots)
    # C видит код A (n_l23_A), а не токены — ур. (1) канона.
    C = ColumnCfg(name="C", side="right", n_in=l23, n_out=l23,
                  n_l4=max(6, int(24 * s)), n_l23=l23, n_l5=max(8, int(32 * s)),
                  n_l6=max(6, int(24 * s)), n_pv=max(4, int(16 * s)),
                  n_sst=max(2, int(8 * s)), input_gain=5.2, wm_slots=wm_slots)
    return OrganismCfg(
        columns=[A, B, C], vocab=vocab, blank=vocab, seed=seed,
        neuron=NeuronCfg(tau_m=8.0, sigma=0.001, surrogate="fast_sigmoid"),
        callosum=CallosumCfg(delay=6, density=0.35),
        workspace=WorkspaceCfg(temperature=0.25, lateral_gain=2.5),
        plasticity=PlasticityCfg(w_fast_max=0.02),
        hippo=HippoCfg(n_slots=64),
        structural=StructuralCfg(enabled=True),
        sleep=SleepCfg(),
    )


class CortexOrganism(nn.Module):
    """Цифровой организм-кортекс: колонки + шина + память + гейт решения."""

    def __init__(
        self,
        vocab_size: int,
        input_dim: Optional[int] = None,
        scale: float = 1.0,
        dt: float = 1.0,
        num_laplace_scales: int = 8,
        vsa_dim: int = 512,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        lr_left: float = 0.020,
        lr_right: float = 0.006,
        horizon_delta_t: int = 3,
        lambda_ewc: float = 100.0,
    ):
        super().__init__()
        self.device = device
        self.dt = float(dt)
        self.vocab_size = int(vocab_size)
        self.input_dim = int(input_dim or vocab_size)
        self.cfg = cortex_cfg(vocab_size, input_dim=self.input_dim,
                              scale=scale, seed=seed)
        self.lr_left, self.lr_right = lr_left, lr_right
        self.horizon_delta_t = int(horizon_delta_t)
        self.seed = seed

        torch.manual_seed(seed)
        self.columns = nn.ModuleDict({
            c.name: LaminarColumn(c, self.cfg.neuron, self.cfg.plasticity,
                                  seed=seed + i)
            for i, c in enumerate(self.cfg.columns)})
        self.names = ["A", "B", "C"]
        self.left_names, self.right_names = ["A", "B"], ["C"]

        # Межполушарная шина C -> A (апикальный контекст L6 -> L5)
        ca, cc = self.cfg.col("A"), self.cfg.col("C")
        self.callosum = CorpusCallosum(
            cc.n_l6, ca.n_l5, delay=self.cfg.callosum.delay,
            density=self.cfg.callosum.density, seed=seed + 17)

        # Спайковые модули Spiker: осциллятор, Лапласова память, БГ
        self.oscillator = ThetaGammaOscillator(dt=dt)
        self.laplace_memory = LaplaceTimingBank(
            feature_dim=self.input_dim, num_scales=num_laplace_scales, dt=dt)
        self.basal_ganglia = BasalGangliaGate(dt=dt)
        # Порты Spiker v7.0: орексиновый интероцептивный контур и NO-поле
        self.orexin = OrexinInteroceptiveLoop(dt=dt)
        self.no_field = NitricOxideField(num_neurons=self.cfg.col("C").n_l23, dt=dt)
        # Порты Spiker v5.0/v6.0 (4-й батч аудита): таламус-TRN, астроциты,
        # SON-самонаблюдение и адаптивная миелинизация Callosum.
        self.thalamus = Thalamus(self.input_dim, dt=dt)
        self.astrocytes = nn.ModuleDict({
            n: TripartiteAstrocyte(self.cfg.col(n).n_l23, dt=dt)
            for n in self.names})
        self.son = SonBus(zone_size=self.l23_dim(), dt=dt)
        self.encoder = TemporalPhaseEncoder(self.input_dim, mode="latency")
        self.callosum.enable_adaptive_myelination(True)
        self._son_surprise: Optional[torch.Tensor] = None
        # Дивергентный реколл гиппокампа (выход из кортикального аттрактора).
        self.hippo_explore: bool = False
        self.hippo_explore_temp: float = 0.25
        # Переопределение SLTT: None -> по фазе (Фаза 1 включает, Фаза 2 нет).
        self.sltt_override: Optional[bool] = None
        # Счётчик реколлов в дивергентном режиме (диагностика: работает ли
        # расхождение вообще).
        self.explore_recalls: int = 0
        # Усиление апикального драйва от гиппокампа. По умолчанию 1.0; при
        # большем значении СОДЕРЖАНИЕ эпизода начинает преодолевать порог
        # спайков (иначе разные эпизоды дают одинаковый рисунок L5).
        self.hippo_apical_gain: float = 1.0
        # Usilenie EPISOD-SPECIFICHNOY sostavlyayushchey apikalnogo privoda.
        self.hippo_content_gain: float = 0.0
        self._astro_gains: Optional[Dict[str, torch.Tensor]] = None
        # STP Цодыкса-Маркрама на рекуррентности L2/3 — рабочая память
        for col in self.columns.values():
            if "rec" in col.syn:
                col.syn["rec"].enable_stp("depressing", dt=dt)

        # Точка слияния: структурная θ-γ синхронизация полушарий
        n_l23 = self.cfg.col("A").n_l23
        self.sync = HemisphereSync(code_dim=n_l23, vsa_dim=vsa_dim,
                                   n_slots=self.cfg.phase.n_slots, seed=seed)

        # Модули NEUROWEAVE 8
        self.workspace = GlobalWorkspace(
            3, gamma_lat=self.cfg.workspace.lateral_gain / 2.0,
            beta_focus=1.0 / max(self.cfg.workspace.temperature, 1e-3),
            storm_rate=self.cfg.workspace.storm_rate,
            storm_gain=self.cfg.workspace.storm_gain,
            competence_rho=self.cfg.workspace.competence_rho,
            competence_floor=self.cfg.workspace.competence_floor)
        self.neuro = NeuroChemistry(
            self.names, self.cfg.mod,
            ht_baseline={n: self.columns[n].ht_baseline for n in self.names},
            adjacency=[("A", "B"), ("B", "C"), ("A", "C")], device=device)
        self.controller = MetaController(
            3, cfg_lr=self.cfg.mod.ctrl_lr, cfg_sigma=self.cfg.mod.ctrl_sigma)
        self.credit = CreditPipeline(self.names, device=device,
                                     lr_fast=self.cfg.plasticity.lr_fast)
        self.hippo = Hippocampus(
            self.input_dim, self.cfg.hippo, seed=seed + 31, n_cortex_l5=self.cfg.col("A").n_l5
        )
        gen_ca1 = torch.Generator().manual_seed(seed + 777)
        w_ca1 = (torch.randn(self.cfg.col("A").n_l5, self.input_dim,
                             generator=gen_ca1) / math.sqrt(self.input_dim)).to(device)
        self.register_buffer("w_ca1_to_cortex", w_ca1, persistent=False)
        self.sleep_proto = SleepProtocol(cfg_sleep=self.cfg.sleep)
        self.structural = StructuralPlasticity(self.cfg.structural)
        self.slow = SlowLoopController(theta_sleep=self.cfg.sleep.threshold,
                                     poll_interval=200)
        self.infer = ActiveInference(3, gamma=4.0, gain=0.6)

        # Токен-голова правого полушария для конкуренции workspace
        c_feat = (cc.n_l23 + cc.n_l5
                  + (cc.n_l23 if self.columns["C"].binder is not None else 0)
                  + cc.n_in)
        self.token_head_c = nn.Linear(c_feat, vocab_size)

        self._ewc: Optional[EWCRegularizer] = None
        self.phase: int = 1
        self._steps = 0
        self._ne_rise_prev = torch.zeros(3, device=device)
        self._prev_c_in: Optional[torch.Tensor] = None
        self.last_metrics: Dict[str, float] = {}

        # Genesis: Ретроградная эндоканнабиноидная система (2-AG & CB1)
        self.ecb = nn.ModuleDict({
            n: EndocannabinoidSystem(n_features=self.cfg.col(n).n_l5, device=device)
            for n in self.names
        })
        # Genesis: Серотонинергическая система терпения (5-HT)
        self.serotonin = SerotoninSystem(device=device)

        self.to(device)

    # ------------------------------------------------------------------
    def set_phase(self, phase: int) -> None:
        """1 — изоляция (шина разорвана); 2 — сборка (апикальный путь открыт,
        w_slow заморожен); 3 — сон."""
        assert phase in (1, 2, 3)
        self.phase = phase
        if phase >= 2:
            for c in self.columns.values():
                c.connect_apical()
                c.freeze_basis()
        else:
            for c in self.columns.values():
                c.isolate()

    def reset_state(self, batch_size: int = 1,
                    device: Optional[torch.device] = None) -> None:
        dev = device or self.device
        for c in self.columns.values():
            c.reset_state(batch_size, dev)
        self.callosum.reset_state(batch_size, dev)
        self.sync.reset_state(batch_size, dev)
        self.laplace_memory.reset_state(batch_size, dev)
        self.basal_ganglia.reset_state(batch_size, dev)
        self.orexin.reset_state(batch_size, dev)
        self.no_field.reset_state(batch_size, dev)
        self.son.reset_state(batch_size, dev)
        self.thalamus.reset_state(batch_size, dev)
        for a in self.astrocytes.values():
            a.reset_state(batch_size, dev)
        self._son_surprise = None
        self._last_recall = None
        self._astro_gains = None
        self._last_activity = torch.zeros(batch_size, 1, device=dev)
        if hasattr(self, "ecb"):
            for e in self.ecb.values():
                e.reset_state(batch_size, dev)
        if hasattr(self, "serotonin"):
            self.serotonin.reset_state(dev)
        self.hippo.reset_state(dev)
        self.oscillator.reset_state(dev)
        self.credit.reset_episode()
        self._ne_rise_prev = torch.zeros(3, device=dev)
        self._apical = {n: torch.zeros(batch_size, self.cfg.col(n).n_l5,
                                       device=dev) for n in self.names}
        self._ff = {n: torch.zeros(batch_size, self.cfg.col(n).n_l4,
                                   device=dev) for n in self.names}
        self._steps = 0

    @property
    def ewc(self) -> Optional[EWCRegularizer]:
        return self._ewc

    def arm_ewc(self, loss_fn, lam: float = 50.0, n_samples: int = 32) -> float:
        params = {}
        for n, col in self.columns.items():
            for k, s in col.syn.items():
                params[f"{n}.{k}"] = s.w_slow
            if col.readout is not None:
                params[f"{n}.readout"] = col.readout.weight
        self._ewc = EWCRegularizer(params, lam=lam)
        return self._ewc.estimate_fisher(loss_fn, n_samples=n_samples)

    # ------------------------------------------------------------------
    def step_isolated(
        self,
        name: str,
        x_in: torch.Tensor,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Изолированный такт одной колонки (Фаза 1): шина разорвана,
        апикальный контекст = 0, sltt=True — суррогатный градиент живёт."""
        col = self.columns[name]
        col._ensure(x_in)
        B = x_in.shape[0]
        out = col.step(
            x_in, extra_apical=None,
            arousal=torch.zeros(B, self.cfg.col(name).n_l5, device=x_in.device),
            sltt=True, learn=learn)
        return {"spikes": out.spikes["l23"], "logits": col.logits(),
                "l5": out.spikes["l5"], "surprise": col.head.surprise}

    def _column_inputs(
        self, token_input: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """A,B <- токен; C <- выход L5 зоны A (в Фазе 1 — ноль, изоляция)."""
        B = token_input.shape[0]
        cin = self._prev_c_in
        if cin is None or cin.shape[0] != B:
            cin = torch.zeros(B, self.cfg.col("C").n_in, device=token_input.device)
        return {"A": token_input, "B": token_input, "C": cin}

    def step(
        self,
        token_input: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        sync_metrics: bool = True,
        learn: bool = True,
        extra_apical: Optional[Dict[str, torch.Tensor]] = None,
        skip_gate: bool = False,
    ) -> Dict[str, object]:
        """Один такт быстрого контура (1 мс).

        skip_gate=True -- НЕ вызывать моторный гейт (БГ) внутри такта.
        Нужен генератору: там выбор токена делает БГ по логитам ДЕКОДЕРА, и
        если гейт дополнительно сработает внутри такта на fused-логитах
        коры, то (а) аккумулятор DDM растёт дважды за тик, и (б) моторный
        тормоз NoGo остывает НЕ тот токен, который реально произносится.
        """
        B = token_input.shape[0]
        dev = token_input.device
        if not hasattr(self, "_apical") or self._apical["A"].shape[0] != B:
            self.reset_state(B, dev)

        # 1. θ-γ пейсмейкер: PAC + фазовое окно Callosum
        osc = self.oscillator.step()
        gate = osc["callosum_phase_gate"]

        # 2. Шина C->A: апикальный контекст через задержку Callosum,
        #    открытая только на пике θ (в Фазе 1 шина разорвана).
        apical = self._apical
        for t in apical.values():
            t.zero_()
        if self.phase >= 2:
            msg = self.columns["C"].prediction_spikes() * gate
            to_a, _ = self.callosum.step(
                msg, torch.zeros(B, self.callosum.dim_out, device=dev))
            # to_a -- сообщение Corpus Callosum; оно попадает в ПЕРСИСТЕНТНОЕ
            # состояние self._apical, поэтому отсоединяем его: иначе цепочка
            # шагов удерживает весь граф предыдущих шагов (утечка VRAM).
            # Градиент сюда и не нужен: в Фазе 2 кору учит кредит-пайплайн, а в
            # Фазе 1 шина разорвана.
            apical["A"] = apical["A"] + to_a.detach()
        if extra_apical is not None:
            for n, val in extra_apical.items():
                if n in apical and val is not None:
                    apical[n] = apical[n] + val

        # 2b. Гиппокамп: тета-гейтированный реколл -> апикальный контекст в L5.
        # Политика "когда вспоминать" целиком внутри гиппокампа (реколл только на спаде
        # тета-волны, не чаще раза за цикл, с SWR-дисинингибицией). Здесь мы лишь
        # ПОДКЛЮЧАЕМ существующий механизм к быстрому контуру: без этого память
        # записывается (hippo.process ниже), но никогда не читается -- кора остается
        # марковской, а апикальное плато L5 (theta_apical=0.1) никогда не срабатывает.
        if self.phase >= 2 and getattr(self, "hippo", None) is not None:
            rec = self.hippocampal_recall(query=token_input,
                                          mode=self._hippo_recall_mode(),
                                          gain=float(self.hippo_apical_gain),
                                          content_gain=float(self.hippo_content_gain))
            if rec.get("allowed"):
                for _n, _v in (rec.get("apical") or {}).items():
                    if _n in apical and _v is not None:
                        apical[_n] = apical[_n] + _v
                self._last_recall = rec

        # 3. EFE-внимание: компетентность x новизна -> arousal-гейт L5
        comp = self.workspace.track[:3].to(dev)
        nov = torch.tensor([float(self.columns[n].head.surprise)
                            for n in self.names], device=dev)
        att = self.infer.gains(comp, nov)

        # 3b. Таламус-TRN: сенсорный гейт. Surprise предыдущего такта и
        #     осциллятор модулируют пропускание входа в кору.
        z_a0 = self.neuro.index["A"]
        # Latency-салиентность (TemporalPhaseEncoder): редкий токен =
        # раньше = громче — механизм внимания на входе без обучения весов.
        ids = token_input.argmax(dim=-1)
        self.encoder.observe_tokens(ids)
        sal = self.encoder.latency_salience(ids).unsqueeze(-1)
        token_in = token_input * sal
        thal = self.thalamus(
            token_in,
            surprise=self._son_surprise,
            ach_level=self.neuro.ach[z_a0],
            ne_level=self.neuro.ne[z_a0],
            oscillator_gain=gate)
        relayed_input = thal["relayed_input"]

        # 4. Такт колонок (в Фазе 1 — sltt для суррогатного градиента);
        #    астроцитарный stp_gain — медленная фасилитация L2/3;
        #    эндоканнабиноидный ecb_scale (Genesis 2-AG) — супрессия зацикливания.
        inputs = self._column_inputs(relayed_input.to(token_input.dtype))
        outs: Dict[str, object] = {}
        sltt = self.sltt_override if self.sltt_override is not None else (self.phase == 1)
        for i, n in enumerate(self.names):
            col = self.columns[n]
            z = self.neuro.index[n]
            ecb_mod = None
            if hasattr(self, "ecb") and n in self.ecb:
                ca5 = col.trace["l5"] if ("l5" in col.trace and col.trace["l5"] is not None) else None
                ca23 = col.trace["l23"] if ("l23" in col.trace and col.trace["l23"] is not None) else None
                if ca5 is not None and ca23 is not None:
                    m = min(ca5.shape[-1], ca23.shape[-1])
                    ca = 0.5 * ca5[..., :m] + 0.5 * ca23[..., :m]
                else:
                    ca = ca5 if ca5 is not None else ca23
                if ca is not None:
                    ecb_mod = self.ecb[n].step(ca, dt=1.0)
            outs[n] = col.step(
                inputs[n], extra_apical=apical[n],
                arousal=att[i].view(1, 1).expand(B, self.cfg.col(n).n_l5),
                ach=self.neuro.ach[z], ht=self.neuro.ht[z],
                ne_rise=self._ne_rise_prev[z],
                sltt=sltt, learn=learn,
                astro_gain=(self._astro_gains.get(n)
                            if self._astro_gains else None),
                ecb_scale=ecb_mod)

        # 5. NE-алармы: транзиент surprise -> фазический выброс
        for n in self.names:
            mag = self.controller.ne_alarm(float(self.columns[n].head.surprise))
            if mag > 0.0:
                self.neuro.pulse("ne", n, mag, cap=self.cfg.mod.ne_pulse_max)

        # 6. Пластичность Фазы 2: событийные DA-импульсы -> w_fast
        if learn:
            for n in self.names:
                col = self.columns[n]
                post = {k: outs[n].spikes[k] for k in ("l4", "l23", "l5", "l6")}
                self.credit.step_column(col, inputs[n].detach(), post,
                                        lr_scale=col.lr_scale)
        if reward is not None:
            self.credit.observe_reward(float(
                reward.mean().item() if torch.is_tensor(reward) else reward))


        # 7. Вход C на следующий такт — выход L5 зоны A (ур. 1)
        a_out = self.columns["A"].output_spikes()
        n_in_c = self.cfg.col("C").n_in
        self._prev_c_in = F.pad(
            a_out.detach(), (0, max(0, n_in_c - a_out.shape[-1])))[:, :n_in_c]

        # 8. Гиппокамп + Лапласова память + синхронизация полушарий + NO-поле
        hippo_out = self.hippo.process(token_input.detach())
        laplace_out = self.laplace_memory.step(token_input)
        sync_out = self.sync.step(outs["A"].spikes["l23"].detach(),
                                  outs["C"].spikes["l23"].detach())
        no_out = self.no_field.step(outs["C"].spikes["l23"].detach())

        # 8b. SON-самонаблюдение над объединённой L2/3 корой — «это ново?»
        #     и астроцитарный Ca2+-контур: stp_gain следующего такта.
        cortical = torch.cat([outs[n].spikes["l23"] for n in self.names],
                             dim=-1)
        z_export, _apictx, son_surprise, _mloss, bus_mask = self.son(
            cortical, sltt_mode=sltt)
        self._son_surprise = (son_surprise.detach()
                              if torch.is_tensor(son_surprise) else son_surprise)
        # DA-импульс за пределы колоночных синапсов: пластичность шины SON
        # и проекций Callosum (раздача CreditPipeline → SON/шина).
        if learn and self.phase >= 2:
            da_cortex = self.neuro.da.mean().detach().view(1, -1)
            self.son.update_phase2_plasticity(
                cortical.detach(), da_cortex,
                lr_fast=self.cfg.plasticity.lr_fast)
            if self.callosum._filtered is not None:
                self.callosum.credit(
                    outs["A"].spikes["l5"].detach(),
                    self.callosum._filtered,
                    da=da_cortex, direction="fwd")
        self._astro_gains = {}
        for n in self.names:
            z = self.neuro.index[n]
            ao = self.astrocytes[n].step(
                outs[n].spikes["l23"],
                ne_level=self.neuro.ne[z],
                ach_level=self.neuro.ach[z])
            self._astro_gains[n] = ao["stp_gain"].detach()

        # 9. Workspace: конкуренция зон -> fused логиты
        logits_a = self.columns["A"].logits()
        logits_b = self.columns["B"].logits()
        logits_c = self.token_head_c(self.columns["C"].features())
        logits_list = [logits_a, logits_b, logits_c]
        bids = self.workspace.calibrated_bids(logits_list)
        alpha, winner, eff = self.workspace.compete(bids)
        fused = self.workspace.fuse(alpha, logits_list, winner=winner)

        # 10. Интероцептивный контур (Spiker §7.0): активность такта =
        #     метаболическая стоимость; энтропия выхода = давление цели.
        probs = F.softmax(fused, dim=-1)
        entropy_t = (-(probs * probs.clamp_min(1e-9).log()).sum(-1)).mean()
        firing_t = torch.cat([outs[n].spikes["l23"] for n in self.names],
                             dim=-1).mean()
        self._last_activity = firing_t.detach().reshape(-1, 1).clone()
        entropy_mean = float(entropy_t.detach())
        if entropy_mean > 0.9:
            self.orexin.register_goal("reduce_output_entropy")
        elif "reduce_output_entropy" in self.orexin.unfinished_goals:
            self.orexin.satisfy_goal("reduce_output_entropy")
        orexin_out = self.orexin.step(activity=self._last_activity)

        # 11. Базальные ганглии + Серотонин (Genesis modulators.py & basal_ganglia.py)
        z_a = self.neuro.index["A"]
        if hasattr(self, "serotonin"):
            r_val = float(sync_out["coherence"]) if "coherence" in sync_out else 0.5
            s5ht = self.serotonin.step(dt=1.0, reward=r_val, crh=float(son_surprise.mean()))
            self.neuro.ht[z_a] = s5ht.clamp(0.05, 1.0)
            serotonin_drive = s5ht.view(1, 1).expand(B, 1)
        else:
            serotonin_drive = self.neuro.ht[z_a]

        if skip_gate:
            # Моторный выбор делает вызывающий код (генератор) по логитам
            # ДЕКОДЕРА. Внутренний вызов гейта на fused-логитах коры остудил бы
            # не тот токен, который реально произносится.
            bg_out = {}
        else:
            bg_out = self.basal_ganglia(
                fused,
                serotonin_5ht=serotonin_drive,
                ach_level=self.neuro.ach[z_a],
                orexin_level=orexin_out["orexin"],
                surprise_level=son_surprise)

        # 12. Химия + медленный контур
        info = self.neuro.step(spike_sums={
            n: outs[n].spikes["l23"].detach().sum() for n in self.names})
        _ne_rise = info["ne_rise"]
        self._ne_rise_prev = (_ne_rise.detach() if torch.is_tensor(_ne_rise) else _ne_rise)
        self.slow.poll(float(self.neuro.sleep_pressure))
        self._steps += 1

        metrics = {}
        if sync_metrics:
            metrics = {
                "entropy": entropy_mean,
                "orexin": float(orexin_out["orexin"].mean()),
                "energy": float(orexin_out["energy"].mean()),
                "firing_rate": float(firing_t.detach()),
                "sleep_pressure": float(self.neuro.sleep_pressure),
                "winner_zone": float(winner.float().mean()),
                "sync_coherence": float(sync_out["coherence"]),
                "wm_occupancy": float(sync_out["occupancy"]),
                "callosum_gate": float(gate),
                "son_surprise": float(son_surprise.mean()),
                "son_bus_active": float(bus_mask.mean()),
                "trn_relay_gain": float(thal["relay_gain"].mean()),
                "astro_ca": float(torch.stack([
                    a.ca_level.mean() for a in self.astrocytes.values()
                    if a.ca_level is not None]).mean())
                if self.astrocytes["A"].ca_level is not None else 0.0,
                "astro_stp_gain": float(torch.stack([
                    g.mean() for g in self._astro_gains.values()]).mean())
                if self._astro_gains else 1.0,
            }
            self.last_metrics = metrics
        else:
            metrics = self.last_metrics

        spikes_a = outs["A"].spikes["l23"]
        spikes_b = outs["B"].spikes["l23"]
        spikes_c = outs["C"].spikes["l23"]
        spikes_a_l5 = outs["A"].spikes["l5"]
        spikes_b_l5 = outs["B"].spikes["l5"]
        spikes_c_l5 = outs["C"].spikes["l5"]
        return {
            "output_logits": fused,
            "left": {
                "zone_a": {"spikes": spikes_a, "logits": logits_a,
                           "surprise": self.columns["A"].head.surprise,
                           "modulators": {
                               "serotonin_5ht": self.neuro.ht[z_a],
                               "ach": self.neuro.ach[z_a],
                               "ne": self.neuro.ne[z_a]}},
                "zone_b": {"spikes": spikes_b, "logits": logits_b},
                "l5_spikes": {"A": spikes_a_l5, "B": spikes_b_l5}},
            "right": {
                "zone_c": {"spikes": spikes_c},
                "l5_spikes_c": spikes_c_l5,
                "token_logits": logits_c,
                "proj_state_a_future": logits_c},
            "workspace": {"wta_weights": alpha, "winner_zone_idx": winner},
            "basal_ganglia": bg_out,
            "laplace_memory": laplace_out,
            "sync": sync_out,
            "orexin": orexin_out,
            "no_field": no_out,
            "son": {"z_export": z_export, "surprise": son_surprise,
                    "bus_active": bus_mask},
            "thalamus": thal,
            "oscillator": osc,
            "hippo": hippo_out,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sleep_cycle(self, n_steps: Optional[int] = None) -> Dict[str, float]:
        self.thalamus.enter_sleep()
        rep = self.sleep_proto.run_cycle(
            self.columns.values(), self.neuro, self.hippo,
            prune_threshold=self.cfg.structural.prune_threshold,
            callosum=self.callosum)
        self.thalamus.wake()
        rep["rem_chimera_insights"] = rep.get("rem_chimera_insights", 0.0)
        if self._ewc is not None:
            self._ewc.rebaseline()
            rep["ewc_rebaselined"] = 1.0
        self.slow.register_sleep()
        # Ключи совместимости с отчётностью Spiker
        rep.setdefault("swr_replay_steps", rep.get("replay_steps", 0.0))
        rep.setdefault("rem_steps", 0.0)
        rep.setdefault("rem_chimera_insights", 0.0)
        rep["sleep_steps"] = rep.get("replay_steps", 0.0)
        return rep

    @torch.no_grad()
    def run_episode(
        self,
        token_seq: torch.Tensor,
        reward_seq: Optional[torch.Tensor] = None,
        target_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        seq_len, batch, _ = token_seq.shape
        self.reset_state(batch, token_seq.device)
        correct, total = 0, 0
        for t in range(seq_len):
            reward = reward_seq[t] if reward_seq is not None else None
            out = self.step(token_seq[t], reward=reward)
            if target_seq is not None:
                pred = out["output_logits"].argmax(dim=-1)
                correct += int((pred == target_seq[t]).sum())
                total += int(pred.numel())
        return {"steps": float(seq_len),
                "accuracy": correct / max(1, total),
                "sleep_pressure": float(self.neuro.sleep_pressure)}

    def telemetry(self) -> Dict[str, float]:
        tel = dict(self.neuro.telemetry())
        tel.update(steps=float(self._steps),
                   hippo_stored=self.hippo.report()["stored"],
                   sync_coherence=float(self.sync.coherence),
                   wm_occupancy=float(self.sync.binder.occupancy())
                   if self.sync.binder.buf is not None else 0.0,
                   bg_emissions=float(self.basal_ganglia.total_emissions),
                   son_surprise=(float(self._son_surprise.mean())
                                 if self._son_surprise is not None else 0.0),
                   astro_ca=float(torch.stack([
                       a.ca_level.mean() for a in self.astrocytes.values()
                       if a.ca_level is not None]).mean())
                   if self.astrocytes["A"].ca_level is not None else 0.0,
                   **{f"orexin_{k}": v for k, v in self.orexin.state_summary().items()})
        if hasattr(self, "ecb"):
            tel["ecb_ag2"] = float(torch.stack([e.ag2.mean() for e in self.ecb.values()]).mean())
        if hasattr(self, "serotonin"):
            tel["serotonin_5ht"] = float(self.serotonin.level.mean())
        return tel

    def swr_disinhibition(self) -> None:
        """SWR Disinhibition (Genesis):
        При успешном извлечении эпизодического воспоминания (Sharp-Wave Ripple)
        происходит временное растормаживание / сброс рабочей памяти (ThetaGammaBinder)
        и сброс аккумулятора DDM в Базальных Ганглиях, чтобы разрушить зацикленные
        аттракторы коры и позволить новому воспоминанию захватить контекст.
        """
        if hasattr(self, "sync") and self.sync is not None:
            self.sync.clear_all()
        for col in self.columns.values():
            if hasattr(col, "clear_wm"):
                col.clear_wm()
            elif col.binder is not None:
                col.binder.clear_all()
        self.basal_ganglia.reset_state(device=self.device)
        if hasattr(self, "ecb"):
            for e in self.ecb.values():
                e.reset_state(1, self.device)
        if hasattr(self, "_last_activity") and self._last_activity is not None:
            self._last_activity.zero_()

    def _hippo_recall_mode(self) -> str:
        """Режим выбора эпизода при реколле: "match" или "explore".

        ДИВЕРГЕНТНЫЙ реколл включается, когда моторный контур застревает
        (последние эмиссии почти не различаются) или когда SON сообщает о
        высокой неожиданности. Ближайший по сходству эпизод в такой момент
        только УСИЛИВАЕТ текущую тему; pattern completion должен предложить
        АЛЬТЕРНАТИВУ, чтобы кора вышла из аттрактора.

        По умолчанию выключено (hippo_explore=False) -- поведение не меняется.
        """
        if not getattr(self, "hippo_explore", False):
            return "match"
        recent = getattr(self.basal_ganglia, "_recent_emitted", [])
        if len(recent) >= 6 and len(set(recent[-6:])) <= 2:
            return "explore"
        if self._son_surprise is not None:
            try:
                if float(self._son_surprise.mean().item()) > 0.08:
                    return "explore"
            except Exception:
                pass
        return "match"

    def configure_cortex_learning(self, enabled: bool = True) -> None:
        """Разморозить/заморозить кору для обучения в Фазе 2.

        В Фазе 2 по умолчанию суррогатный градиент обрезан (sltt выключен,
        w_slow заморожен EWC) -- кора учится только в Фазе 1. Чтобы перенести
        правило В КОРУ, нужно:
          * включить sltt (градиент через ТЕКУЩИЕ синапсы; история состояний
            по-прежнему отсоединяется, поэтому утечки VRAM не будет);
          * разморозить w_slow (синапсы перестают быть «замороженными» для Фазы 2).
        """
        self.sltt_override = bool(enabled)
        for c in self.columns.values():
            for s in c.syn.values():
                if enabled:
                    s.unfreeze()
                else:
                    s.freeze_for_phase2()

    def configure_theta_clock(self, mode: str = "baseline",
                              cycle_steps: float = 0.0) -> None:
        """Масштаб тета-цикла гиппокампа (рантайм, веса НЕ затрагиваются).

        mode="baseline" -- 143 шага на цикл (7 Гц при dt=1 мс): реколл раз в 143 токена.
        mode="fast"     -- ВАРИАНТ А: тот же 7 Гц, но пересчитан масштаб времени
                          модели так, чтобы цикл занимал cycle_steps шагов.
        mode="token"    -- ВАРИАНТ Б: фаза теты считается от счётчика токенов
                          (1 токен = 1/cycle_steps тета-цикла).
        """
        if getattr(self, "hippo", None) is not None:
            self.hippo.configure_theta(mode=mode, cycle_steps=cycle_steps)

    @torch.no_grad()
    def hippocampal_recall(
        self,
        query: Optional[torch.Tensor] = None,
        anchor_slots: Optional[List[int]] = None,
        gain: float = 1.0,
        force: bool = False,
        n_steps: int = 6,
        mode: Optional[str] = None,
        explore_temp: Optional[float] = None,
        content_gain: Optional[float] = None,
    ) -> Dict[str, object]:
        """RAG через Гиппокамп: извлечение эпизода из CA3->EC и апикальный ток в кору.

        mode="explore" -- дивергентный выбор эпизода (не nearest-match);
        используется для выхода из кортикального аттрактора.
        """
        if mode is None:
            mode = self._hippo_recall_mode()
        if mode == "explore":
            self.explore_recalls += 1
        if explore_temp is None:
            explore_temp = float(getattr(self, "hippo_explore_temp", 0.25))
        ret = self.hippo.retrieve(
            query=query, anchor_slots=anchor_slots, force=force, n_steps=n_steps,
            gain=gain, mode=mode, explore_temp=explore_temp,
            content_gain=(float(self.hippo_content_gain)
                         if content_gain is None else float(content_gain)),
        )
        if not ret.get("allowed", True):
            return {
                "allowed": False,
                "reason": ret.get("reason", "blocked"),
                "x_rec": ret.get("x_rec", None),
                "slot_idx": ret.get("slot_idx", -1),
                "sim": ret.get("sim", 0.0),
                "mismatch": ret.get("mismatch", 0.0),
                "theta": ret.get("theta", 0.0),
                "apical": {"A": None, "B": None},
                "ec_grid": None,
            }

        # SWR Disinhibition: стирание зацикленного кортикального аттрактора
        self.swr_disinhibition()

        apical_a = ret["apical_l5"]
        apical_b = apical_a * 0.70

        return {
            "allowed": True,
            "x_rec": ret["x_rec"],
            "code": ret.get("code"),
            "ec_grid": ret.get("ec_grid"),
            "slot_idx": ret["slot_idx"],
            "sim": ret["sim"],
            "mismatch": ret.get("mismatch", 0.0),
            "theta": ret.get("theta", 0.0),
            "apical": {"A": apical_a, "B": apical_b},
        }

    def cortical_dim(self) -> int:
        return sum(self.cfg.col(n).n_l23 + self.cfg.col(n).n_l5
                   for n in self.names)

    def l23_dim(self) -> int:
        """Dimension of L2/3-only cortical spikes (used by SonBus, WM, etc.)."""
        return sum(self.cfg.col(n).n_l23 for n in self.names)

    def feature_dim_c(self) -> int:
        cc = self.cfg.col("C")
        return (cc.n_l23 + cc.n_l5
                + (cc.n_l23 if self.columns["C"].binder is not None else 0)
                + cc.n_in)
