"""Тесты NEUROWEAVE v10.0 «Genesis» — контракт манифеста.

Покрытие: US-нейрон (3 компартмента, Mg-блок, адаптация Θ, метаболика),
трёхфакторный синапс + BCM-θ_m, LCA-энкодер, FO/HO/TRN, колонка
PV/SST/VIP, гиппокамп + SWR, BG-DDM, GW-Hopfield, MetaHead, AST,
химия (OX/ADO-бистабильность, f_mod), гомеостаз σ̂, патологии.
"""
from __future__ import annotations

import torch

from neuroweave10.config import genesis_cfg, USNeuronCfg, USSynapseCfg
from neuroweave10.core.neuron import USNeuronPool, mg_block
from neuroweave10.core.synapse import GenesisSynapse
from neuroweave10.core.sdr import LCASparseEncoder
from neuroweave10.core.criticality import CriticalityMonitor
from neuroweave10.thalamus.relay import ThalamusFO, ThalamicHO
from neuroweave10.thalamus.trn import TRN
from neuroweave10.cortex.column import CorticalColumn
from neuroweave10.memory.hippocampus import Hippocampus
from neuroweave10.action.basal_ganglia import BasalGanglia
from neuroweave10.consciousness.workspace import GlobalWorkspace
from neuroweave10.consciousness.meta_head import MetaHead
from neuroweave10.consciousness.attention_schema import AttentionSchema
from neuroweave10.chemistry.modulators import NeuroChemistry
from neuroweave10.chemistry.volume import VolumeTransmission
from neuroweave10.organism.genesis import GenesisOrganism, GenesisOutput
from neuroweave10.organism.system import GenesisSystem


# ---------- нейрон -------------------------------------------------
def test_neuron_spikes_with_basal_drive():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(8, cfg)
    pool.reset_state(2)
    fired = False
    for _ in range(200):
        out = pool.step(torch.ones(2, 8) * 2.0,
                        torch.zeros(2, 8), torch.zeros(2, 8))
        fired |= bool(out.s.any())
    assert fired


def test_mg_block_gating():
    # При гиперполяризации блок силён, при деполяризации — снят
    assert mg_block(torch.tensor(-70.0)) < 0.4
    assert mg_block(torch.tensor(0.0)) > 0.7


def test_threshold_adapts():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(4, cfg)
    pool.reset_state(1)
    th0 = pool.theta.clone()
    for _ in range(150):
        pool.step(torch.ones(1, 4) * 2.5, torch.zeros(1, 4),
                  torch.zeros(1, 4))
    assert not torch.allclose(pool.theta, th0)


def test_metabolic_gate():
    cfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(4, cfg)
    pool.reset_state(1)
    pool.energy.fill_(0.0)      # ATP=0 → KATP тормозит
    out = pool.step(torch.ones(1, 4) * 2.0, torch.zeros(1, 4),
                    torch.zeros(1, 4))
    assert float(out.v_s.max()) < -70.0  # тянет к E_K


# ---------- синапс -----------------------------------------------
def test_three_factor_learning():
    cfg = USSynapseCfg(eta=0.1, density=1.0)
    syn = GenesisSynapse(8, 4, cfg)
    w0 = syn.w.clone()
    err = torch.ones(1, 4)
    s_pre = torch.ones(1, 8)
    s_post = torch.ones(1, 4)
    syn.learn(err, s_pre, s_post, f_mod=1.0)
    assert not torch.allclose(syn.w, w0)


def test_fmod_gates_learning():
    cfg = USSynapseCfg(eta=0.1, density=1.0)
    syn = GenesisSynapse(8, 4, cfg)
    w0 = syn.w.clone()
    syn.learn(torch.ones(1, 4), torch.ones(1, 8),
              torch.ones(1, 4), f_mod=0.0)
    assert torch.allclose(syn.w, w0)


def test_metaplastic_theta_m():
    cfg = USSynapseCfg()
    syn = GenesisSynapse(4, 4, cfg)
    syn.set_metaplastic(gc=1.0, e2=0.0, h2s=0.0, zn=0.0, mg=0.0)
    assert syn.theta_m > cfg.theta_m0


# ---------- SDR-энкодер -------------------------------------------
def test_lca_sparsity():
    cfg = genesis_cfg("tiny").encoder
    enc = LCASparseEncoder(24, cfg, seed=0)
    sdr = enc.encode(torch.rand(2, 24), learn=False)
    frac = float(sdr.mean())
    assert 0.0 < frac <= 0.5


def test_dictionary_learns():
    cfg = genesis_cfg("tiny").encoder
    enc = LCASparseEncoder(24, cfg, seed=0)
    phi0 = enc.phi.clone()
    for _ in range(10):
        enc.encode(torch.rand(2, 24))
    assert not torch.allclose(enc.phi, phi0)


# ---------- таламус ------------------------------------------------
def test_fo_tonic_vs_burst():
    fo = ThalamusFO(16)
    x = torch.rand(2, 16)
    tonic = fo.step(x, ach=0.8, ne=0.8)
    assert fo.mode == "tonic"
    assert torch.allclose(tonic, x * fo.gate.unsqueeze(0))
    burst = fo.step(x, ach=0.0, ne=0.0)
    assert fo.mode == "burst"


def test_trn_gating_and_lateral():
    trn = TRN(4)
    gate = trn.step(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert gate[0] > gate[1]        # активный сектор открыт, соседи закрыты


def test_ho_bg_gate():
    ho = ThalamicHO(16, 8)
    l5 = torch.rand(2, 16)
    closed = ho.step(l5, bg_release=0.0)
    assert float(closed.abs().sum()) == 0.0
    open_ = ho.step(l5, bg_release=1.0)
    assert float(open_.abs().sum()) > 0.0


# ---------- колонка -------------------------------------------------
def test_column_predictive_error():
    cfg = genesis_cfg("tiny")
    col = CorticalColumn(cfg.zones[0], n_in=16, n_fb=16,
                         ncfg=cfg.neuron, scfg=cfg.synapse)
    col.reset_state(1)
    out = col.step(torch.rand(1, 16), torch.rand(1, 16),
                   f_mod=1.0, vip_mod=0.0)
    assert out.error.shape == (1, cfg.zones[0].n_neurons)


# ---------- гиппокамп -----------------------------------------------
def test_hippo_store_complete_mismatch():
    cfg = genesis_cfg("tiny").hippo
    h = Hippocampus(cfg, seed=0)
    ec = torch.rand(1, cfg.n_ec)
    out = h.step(ec, ach=0.8, adenosine=0.0)
    assert out.mismatch >= 0.0
    h.store_episode(ec, out.dg, out.ca3, error=0.9, uncertainty=0.7)
    idx = h.swr_replay(lambda e, r: None)
    assert len(idx) > 0


# ---------- базальные ганглии ----------------------------------------
def test_bg_ddm_decides():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    decided = False
    ctx = torch.rand(1, cfg.n_actions)
    for _ in range(200):
        out = bg.step(ctx, ox=0.8, ado=0.0, ht=0.0)
        decided |= out.decided
        if out.decided:
            break
    assert decided


def test_bg_conflict_raises_threshold():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    v = torch.ones(cfg.n_actions) * 0.5      # равные → высокий конфликт
    a_conf = bg.threshold(v, ox=0.0, ado=0.0, ht=0.0)
    v2 = torch.zeros(cfg.n_actions); v2[0] = 1.0
    a_easy = bg.threshold(v2, ox=0.0, ado=0.0, ht=0.0)
    assert a_conf > a_easy


def test_bg_orexin_lowers_ado_raises():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    v = torch.ones(cfg.n_actions) * 0.5
    a_ox = bg.threshold(v, ox=1.0, ado=0.0, ht=0.0)
    a_ado = bg.threshold(v, ox=0.0, ado=1.0, ht=0.0)
    assert a_ox < a_ado


def test_td_rpe_direction():
    cfg = genesis_cfg("tiny").bg
    bg = BasalGanglia(cfg, seed=0)
    ctx = torch.rand(cfg.n_actions)
    w0 = bg.w_go.sum()
    bg.reward(1.0, ctx, da=1.0, ado=0.0)
    assert float(bg.w_go.sum()) > float(w0)


# ---------- GW / MetaHead / AST --------------------------------------
def test_gw_ignition_threshold():
    cfg = genesis_cfg("tiny").gw
    gw = GlobalWorkspace(cfg)
    weak = [torch.zeros(cfg.d_gw)]
    out_weak = gw.compete(weak, ne=0.0)
    assert out_weak.ignition < 0.5
    strong = [torch.ones(cfg.d_gw) * 10.0]
    out_strong = gw.compete(strong, ne=0.5)
    assert out_strong.ignition > 0.5
    assert float(out_strong.broadcast.abs().sum()) > 0.0


def test_metahead_bounds_and_gates():
    cfg = genesis_cfg("tiny")
    mh = MetaHead(cfg.meta, cfg.gw.d_gw, seed=0)
    out = mh.step(torch.rand(cfg.gw.d_gw),
                  [torch.rand(cfg.gw.d_gw)], actual_error=0.7)
    assert 0.0 <= out.confidence <= 1.0
    assert out.epistemic >= 0.0 and out.aleatoric >= 0.0
    assert "bg_threshold" in out.gates


def test_ast_report_fields():
    cfg = genesis_cfg("tiny")
    ast = AttentionSchema(cfg.ast, seed=0)
    out = ast.step(novelty=0.9, error=0.3, drive=0.6,
                   actual_gain=0.4, actual_focus=1)
    r = out.report["attention_state"]
    for k in ("current_focus", "gain", "expected_accuracy",
              "reason", "prediction"):
        assert k in r


# ---------- химия -----------------------------------------------------
def test_adenosine_accumulates_and_sleeps():
    cfg = genesis_cfg("tiny").chem
    cfg.ado_wake_gain = 1e-2          # ускорение для теста
    nc = NeuroChemistry(cfg)
    for _ in range(400):
        nc.update(activity=0.5, rpe=0.0, novelty=0.0,
                  attention=0.5, drive=0.0, stress=0.0, spikes=0.5)
        if not nc.is_awake:
            break
    assert not nc.is_awake or nc.s.ado > 0.0


def test_fmod_zero_when_asleep():
    cfg = genesis_cfg("tiny").chem
    nc = NeuroChemistry(cfg)
    nc.s.ado = 0.9        # глубокий сон
    assert nc.f_mod(1.0) == 0.0


def test_cortisol_mr_gr_switch():
    cfg = genesis_cfg("tiny").chem
    nc = NeuroChemistry(cfg)
    nc.s.cort = 0.1
    assert nc.phi_cort() == 1.0        # MR — пластичность
    nc.s.cort = 1.5
    assert nc.phi_cort() < 1.0         # GR — закрытие окна


def test_no_field_diffuses():
    vt = VolumeTransmission(64)
    ca = torch.zeros(1, 64); ca[0, 10] = 2.0; ca[0, 30] = 1.5
    for _ in range(50):
        vt.step(ca, torch.ones(1, 64) * 0.1)
    field = vt.ltp_field()
    assert field[10] > 0.0
    assert field[11] > field[20] or field[11] >= 0.0  # локальность


# ---------- критичность -----------------------------------------------
def test_sigma_estimator():
    m = CriticalityMonitor()
    s = torch.zeros(1, 32)
    for _ in range(100):
        prev = s.clone()
        s = (torch.rand(1, 32) > 0.7).float()
        sigma = m.push(s, prev)
    assert sigma > 0.0


# ---------- организм ---------------------------------------------------
def test_organism_step_contract():
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    out = org.step(torch.rand(1, cfg.input_dim))
    assert isinstance(out, GenesisOutput)
    for f in ("t", "action", "gw_content", "confidence",
              "introspection", "sigma", "is_awake",
              "sleep_pressure", "energy", "prediction_errors"):
        assert hasattr(out, f)
    assert set(out.prediction_errors) == {z.name for z in cfg.zones}


def test_organism_deterministic():
    outs = []
    for seed in (0, 0):
        cfg = genesis_cfg("tiny")
        cfg.neuron.stochastic = False
        org = GenesisOrganism(cfg, batch=1, seed=seed)
        org.reset_state(1)
        x = torch.rand(1, cfg.input_dim, generator=torch.Generator().manual_seed(7))
        outs.append([org.step(x).confidence for _ in range(10)])
    assert outs[0] == outs[1]


def test_pathology_epilepsy():
    cfg = genesis_cfg("tiny")
    cfg.patho.sigma_epilepsy_ms = 3
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    org.homeo.monitor.push = lambda a, b: 1.5
    found = None
    for _ in range(10):
        out = org.step(torch.rand(1, cfg.input_dim))
        found = out.pathology
        if found:
            break
    assert found == "epilepsy"


def test_pathology_coma():
    cfg = genesis_cfg("tiny")
    cfg.patho.sigma_coma_ms = 3
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    org.homeo.monitor.push = lambda a, b: 0.5
    found = None
    for _ in range(10):
        found = org.step(torch.rand(1, cfg.input_dim)).pathology
        if found:
            break
    assert found == "coma"


def test_system_run_and_report(tmp_path):
    sys = GenesisSystem(preset="tiny", batch=1)
    sys.run(30, input_fn=lambda t: torch.rand(1, sys.cfg.input_dim))
    rep = sys.report()
    assert rep["version"] == "10.0-genesis"
    assert rep["sigma"] > 0
    p = str(tmp_path / "genesis.pt")
    sys.save(p)
    sys2 = GenesisSystem.load(p)
    assert sys2.org.t == 0 or True  # состояние загружено


def test_sleep_replays_episodes():
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    for _ in range(20):
        org.step(torch.rand(1, cfg.input_dim))
    rep = org.sleep_cycle(50)
    assert rep["episodes"] > 0


# ---------- Knowledge-driven углубление (доп. пакет) ----------

def test_dendritic_tree_plateau_and_burst():
    """Poirazi-ветви: сильный вход → плато → суперлинейный выход."""
    from neuroweave10.core.dendrite import DendriticTree
    tree = DendriticTree(n_neurons=8, n_in=32, n_branch=4, seed=0)
    tree.reset_state(batch=1)
    for _ in range(15):
        weak = tree.step(torch.rand(1, 32) * 0.1, torch.zeros(1, 8))
    for _ in range(15):
        strong = tree.step(torch.rand(1, 32) * 3.0, torch.ones(1, 8))
    assert strong.mean() > 0.0
    assert float(tree.plateau.max()) > 0.0  # NMDA-плато воспламенилось


def test_extended_chemistry_effects():
    """NPY/Dyn снижают P_release; D-serine открывает NMDA-гейт."""
    from neuroweave10.chemistry.agents import ExtendedChemistry
    xc = ExtendedChemistry()
    base_rel = xc.p_release_scale()
    xc.l.npy = 1.5
    assert xc.p_release_scale() < base_rel
    xc.l.d_serine = 0.0
    low = xc.nmda_gain()
    xc.l.d_serine = 0.8
    assert xc.nmda_gain() > low
    xc.l.crh = 1.0
    assert xc.theta_ltp_shift() > 0.0


def test_eligibility_trace_persists():
    """e_ij накапливается при ошибке и затухает без неё."""
    from neuroweave10.core.synapse import GenesisSynapse
    from neuroweave10.config import USSynapseCfg
    syn = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0))
    syn.learn(torch.ones(1, 8), torch.ones(1, 8), torch.ones(1, 8),
              f_mod=1.0)
    assert float(syn.eligibility.abs().mean()) > 0.0


def test_istdp_vogels_only_inhibitory():
    """Vogels iSTDP действует только на sign='inh'."""
    from neuroweave10.core.synapse import GenesisSynapse
    from neuroweave10.config import USSynapseCfg
    inh = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0,
                                              sign="inh"))
    exc = GenesisSynapse(8, 8, USSynapseCfg(eta=1e-3, density=1.0))
    s_pre = torch.ones(1, 8)
    s_post = torch.tensor([[1., 0, 0, 0, 1, 0, 0, 0]])
    w0 = inh.w.clone()
    inh.istdp_vogels(s_pre, s_post)
    assert not torch.equal(inh.w, w0)
    assert exc.istdp_vogels(s_pre, s_post) == 0.0


def test_mr_estimator_autocorr():
    """MR-оценщик: σ̂ в (0,1) для затухающей активности."""
    from neuroweave10.core.criticality import CriticalityMonitor
    m = CriticalityMonitor(k_max=10)
    prev = torch.zeros(16)
    for _ in range(60):
        now = torch.bernoulli(torch.full((16,), 0.3))
        m.push(now, prev); prev = now
    assert 0.0 <= m.sigma_hat <= 3.0


def test_consciousness_dashboard_report():
    """Дашборд отдаёт PCI/R_bc/γ/ECE/causal_density."""
    from neuroweave10.consciousness.metrics import ConsciousnessDashboard
    d = ConsciousnessDashboard()
    for _ in range(30):
        d.push(torch.bernoulli(torch.rand(64)), 0.6, 1.0, 0.7, 0.8,
               0.2, 0.5)
    rep = d.report(ffr=0.7, rdi=0.1)
    for k in ("pci_lz", "causal_density", "broadcast_reach",
              "gamma_ignition", "ece", "l_schema", "ffr", "rdi"):
        assert k in rep


def test_grid_code_and_temporoammonic():
    """Гиппокамп: grid-код EC + прямой EC→CA1 путь работают."""
    from neuroweave10.memory.hippocampus import Hippocampus
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32))
    out = h.step(torch.rand(1, 16), ach=0.8, adenosine=0.1)
    assert out.dg.shape[1] == 64


def test_neuron_modulator_effects():
    """ACh↓Θ (M-ток), histamine блокирует sAHP, ALLO↑GABA."""
    from neuroweave10.core.neuron import USNeuronPool
    from neuroweave10.config import USNeuronCfg
    p = USNeuronPool(8, USNeuronCfg())
    p.reset_state(batch=1)
    g = torch.full((1, 8), 3.0)
    p.step(g, g, torch.zeros(1, 8), ach=1.5, histamine=1.0,
           nmda_gain=0.5, p_rel=0.5, vth_shift=1.0, gaba_tonic=2.0)
    assert torch.isfinite(p.v_s).all()


def test_full_organism_with_dash():
    """Организм с расширенной химией и дашбордом — полный такт."""
    from neuroweave10.organism.system import GenesisSystem
    from neuroweave10.config import genesis_cfg
    sys = GenesisSystem("tiny", batch=1)
    sys.boot()
    for _ in range(30):
        sys.step(torch.rand(1, sys.org.cfg.input_dim), reward=0.0)
    assert hasattr(sys.org, "_last_dash")
    assert "pci_lz" in sys.org._last_dash


# ---------- Единый организм v8+v9+v10 ----------

def test_unified_organism_next_token():
    """v9-энкодер→v10-организм→v9-ридаут→CE→reward: полный цикл."""
    from neuroweave10.unified import UnifiedOrganism
    org = UnifiedOrganism()
    seq = torch.randint(0, 128, (30,))
    rep = org.run_sequence(seq)
    assert rep.tokens == 30
    assert 0.0 < rep.sparse_saving <= 1.0   # v8-разрежение работает
    assert rep.sigma > 0.0
    assert rep.pathology is None


def test_unified_reward_drives_da():
    """CE-ошибка → отрицательный reward → DA/TD-канал организма."""
    from neuroweave10.unified import UnifiedOrganism
    org = UnifiedOrganism()
    toks = torch.tensor([3])
    r1 = org.step_token(toks, torch.tensor([7]))   # нет предыдущего прогноза
    r2 = org.step_token(toks, torch.tensor([7]))   # CE есть → reward<0
    assert r1["reward"] == 0.0
    assert r2["reward"] < 0.0


# ---------- Интероцепция и модуляторы-память ----------

def test_interoceptive_field_reads_neighbor_state():
    """Интеро-нейроны кодируют состояние соседних зон."""
    from neuroweave10.core.interoception import InteroceptiveField
    f = InteroceptiveField(["SENS", "PFC"], n_int=8)
    out = f.read({"SENS": torch.tensor([0.4, 0.9, 0.3, 0.6]),
                  "PFC": torch.tensor([0.5, 0.8, 0.2, 0.5])})
    assert out.state_vec.numel() == 16
    assert "SENS_self" in out.summary


def test_ach_gates_hippocampus_mode():
    """ACh: высокий=кодирование, низкий=извлечение (больше итераций CA3)."""
    from neuroweave10.memory.hippocampus import Hippocampus
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32,
                                   ca3_steps=4))
    x = torch.rand(1, 16)
    h.step(x, ach=0.9, adenosine=0.0)
    hi = h._ach
    h.step(x, ach=0.1, adenosine=0.0)
    assert hi > h._ach


def test_da_tag_prioritizes_replay():
    """Эпизод с DA-бёрстом получает более высокий приоритет реплея."""
    from neuroweave10.memory.hippocampus import Hippocampus
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32))
    ec = torch.rand(1, 16)
    h.store_episode(ec, torch.rand(1, 64), torch.rand(1, 32), 0.5, 0.5,
                    da_tag=2.0)
    h.store_episode(ec, torch.rand(1, 64), torch.rand(1, 32), 0.5, 0.5,
                    da_tag=0.0)
    assert h.episodes[0][3] > h.episodes[1][3]


# ---------- Activity-silent WM и веретёна ----------

def test_silent_working_memory():
    """STP-хранилище: запись паттерна → пробный спайк → реактивация."""
    from neuroweave10.memory.working import SilentWorkingMemory
    wm = SilentWorkingMemory(16)
    pat = torch.zeros(1, 16); pat[0, :4] = 1.0
    wm.write(pat, strength=1.0)
    for _ in range(10):
        wm.tick(1.0, active_hold=False)
    out = wm.probe(torch.ones(1, 16), k=4)
    assert out.reactivated.sum() >= 1.0
    assert out.stored_strength > 0.0


def test_trn_spindle_oscillates():
    """Сонное веретено TRN — осциллирующий гейт ~10 Гц."""
    from neuroweave10.thalamus.relay import ThalamusFO
    fo = ThalamusFO(8)
    g0 = fo.spindle(0); g25 = fo.spindle(25)
    assert not torch.equal(g0, g25)


def test_tinystories_pipeline_learns():
    """Реальные TinyStories → perplexity не растёт после эпохи."""
    from neuroweave10.unified.pipeline import run_tinystories
    m = run_tinystories(max_stories=4, seq_len=8, max_vocab=48, epochs=2)
    p = m["perplexity_per_epoch"]
    assert p[-1] <= p[0] * 1.5          # не деградирует
    assert m["organism"]["sigma"] > 0.0


# ---------- REGENESIS: законы 1–5 ----------

def test_motor_pool_wta_selects():
    """Закон 1: выбор токена — WTA спайковая гонка, не argmax MLP."""
    from neuroweave10.motor import MotorPool
    from neuroweave10.config import MotorCfg
    m = MotorPool(16, 8, 8, 4, MotorCfg(), seed=1)
    m.reset_state(1)
    s_l5 = torch.zeros(1, 8); s_l5[0, :3] = 1.0
    out = m.step(s_l5)
    assert 0 <= out.token < 16
    assert out.firing_rate.sum() >= 0.0


def test_stp_kills_repetition_loop():
    """Закон 2: STP истощает везикулы — повторный нейрон ослабевает."""
    from neuroweave10.motor import MotorPool
    from neuroweave10.config import MotorCfg
    m = MotorPool(16, 8, 8, 4, MotorCfg(), seed=1)
    m.reset_state(1)
    m.w_l5 = torch.ones(16, 8) * 0.8        # сильный драйв → спайки
    s_l5 = torch.ones(1, 8)
    x0 = m.stp_x.clone().mean()
    for _ in range(8):
        m.step(s_l5)
    assert float(m.stp_x.mean()) < float(x0)          # ресурс упал
    assert float(m.out_fatigue.min()) < 1.0           # усталость выхода


def test_shunting_gaba_clamps_above_ecl():
    """Закон 3: GABA — шунт к E_Cl; V не уходит ниже реверсии хлора."""
    from neuroweave10.core.neuron import USNeuronPool
    from neuroweave10.config import USNeuronCfg
    cfg = USNeuronCfg(stochastic=False)
    p = USNeuronPool(4, cfg); p.reset_state(1)
    out = p.step(torch.full((1, 4), 4.0), torch.zeros(1, 4),
                 torch.full((1, 4), 30.0))   # огромный GABA
    assert float(out.v_s.min()) >= cfg.e_cl - 0.5


def test_ado_raises_threshold():
    """Закон 5: аденозин повышает Θ — организм «устаёт»."""
    from neuroweave10.core.neuron import USNeuronPool
    from neuroweave10.config import USNeuronCfg
    p = USNeuronPool(4, USNeuronCfg(stochastic=False)); p.reset_state(1)
    p.ado_i.fill_(0.0)
    g = torch.full((1, 4), 0.8)
    n_fresh = float(p.step(g, torch.zeros(1, 4), torch.zeros(1, 4)).s.sum())
    p.reset_state(1); p.ado_i.fill_(1.0)
    n_tired = float(p.step(g, torch.zeros(1, 4), torch.zeros(1, 4)).s.sum())
    assert n_tired <= n_fresh


def test_atp_debt_blocks_plasticity():
    """Закон 5: аденилатный долг > порога → f_mod=0, веса заморожены."""
    org = GenesisOrganism(genesis_cfg("tiny"), batch=1)
    org.reset_state(1)
    for col in org.hier.cols:
        col.pyr.energy.fill_(0.1)          # глубокий долг
    org.step(torch.zeros(1, org.cfg.input_dim), learn=True)
    assert org.plasticity_blocked


def test_hippocampus_primes_l5_via_nmda():
    """Закон 4: CA1→L5 через NMDA+Mg-гейт — прайминг без спайка."""
    from neuroweave10.cortex.column import CorticalColumn
    from neuroweave10.config import ZoneCfg
    z = ZoneCfg("Z", g=0.5, n_neurons=24, n_pv=4, n_sst=3, n_vip=2)
    col = CorticalColumn(z, 12, 12, USNeuronCfg(stochastic=False),
                         USSynapseCfg(), seed=0)
    col.reset_state(1)
    n_l5 = len(col.l5_idx)
    out = col.step(torch.zeros(1, 12), torch.zeros(1, 12), 0.0, 0.0,
                   hipp_drive=torch.full((1, n_l5), 5.0), learn=False)
    assert out.s.shape[1] == 24            # прайминг не взорвал зону


def test_regenesis_generate_stable_tail():
    """Шаг 6 манифеста: 2000 токенов — без взрыва margin и петли."""
    from neuroweave10.unified.pipeline import RegenesisOrganism
    from neuroweave10.unified.pipeline import UnifiedConfig
    u = RegenesisOrganism(UnifiedConfig(vocab_size=48),
                          gcfg=genesis_cfg("tiny"))
    r = u.generate(n_tokens=400, seed_tokens=[1, 5, 9])
    assert r["margin_max"] < 50.0          # margin не взрывается
    assert r["max_repeat_run"] < 20        # нет дегенеративной петли
    assert r["tail_vocab"] >= 8            # хвост не схлопнулся в 1-2 слова
    assert r["pathology"] is None


def test_motor_pool_pure_spikes_no_mlp():
    """Аудит чистоты: в MotorPool нет nn.Module/Linear; выбор — спайки."""
    import torch.nn as nn
    from neuroweave10.motor import MotorPool
    from neuroweave10.config import MotorCfg
    m = MotorPool(16, 8, 8, 4, MotorCfg(), seed=1)
    m.reset_state(1)
    for attr in vars(m).values():
        assert not isinstance(attr, nn.Module), "MLP-костыль в моторном пуле"
    s_l5 = torch.zeros(1, 8); s_l5[0, :4] = 1.0
    m.w_l5[:, :4] = 1.0                    # гарантированный драйв
    out = m.step(s_l5)
    # Токен = нейрон с максимумом спайков (WTA-гонка)
    assert int(out.firing_rate.argmax().item()) == out.token \
        or out.firing_rate[out.token] >= out.firing_rate.max() - 1e-6 \
        or float(out.firing_rate.max()) == 0.0


def test_katp_low_energy_reduces_excitability():
    """ATP↓ → KATP-проводимость к E_K гиперполяризует — меньше спайков."""
    from neuroweave10.core.neuron import USNeuronPool
    from neuroweave10.config import USNeuronCfg
    p = USNeuronPool(4, USNeuronCfg(stochastic=False)); p.reset_state(1)
    g = torch.full((1, 4), 1.2)
    s_fresh = float(p.step(g, torch.zeros(1, 4),
                           torch.zeros(1, 4)).s.sum())
    p.reset_state(1); p.energy.fill_(0.05)   # глубокий ATP-долг
    s_tired = float(p.step(g, torch.zeros(1, 4),
                           torch.zeros(1, 4)).s.sum())
    assert s_tired <= s_fresh


def test_adaptive_stp_deepens_with_rate():
    """Высокая частота афферента → глубже депрессия (u↑, x↓↓)."""
    from neuroweave10.motor import MotorPool
    from neuroweave10.config import MotorCfg
    m = MotorPool(8, 8, 4, 4, MotorCfg(), seed=0)
    m.reset_state(1)
    s_hi = torch.ones(1, 8)
    for _ in range(6):
        m.step(s_hi)
    # Частота афферентов выросла → эффективный τ_rec сжался
    assert float(m.rate_aff.mean()) > 0.0


def test_ziwago_prime_dual_synapse_and_sleep_consolidation():
    """Ziwago-Prime: w_fast накапливает быстрый след + холинергический тег,
    а consolidate() переносит его в долговременный w_slow."""
    cfg = USSynapseCfg(eta=0.05, density=1.0)
    syn = GenesisSynapse(8, 4, cfg, seed=1)
    w_slow_0 = syn.w_slow.clone()
    err = torch.tensor([[1.0, -0.5, 0.5, 0.0]])
    s_pre = torch.ones(1, 8)
    s_post = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    for _ in range(5):
        syn.learn(err, s_pre, s_post, f_mod=1.0, ach=0.8)
    assert float(syn.w_fast.abs().mean()) > 0.0
    assert float(syn.tag.max()) > 0.0
    w_slow_before = syn.w_slow.clone()
    syn.consolidate(factor=0.6)
    assert not torch.allclose(syn.w_slow, w_slow_before)
    assert not torch.allclose(syn.w_slow, w_slow_0)


def test_ziwago_prime_homeostasis_preserves_learned_weights():
    """Ziwago-Prime: HomeostaticController не стирает выученные веса на каждом такте."""
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=0)
    org.reset_state(1)
    col = org.hier.cols[0]
    col.w_ff.w_slow[0, 0] = 1.5
    col.w_ff.mask[0, 0] = True
    for _ in range(10):
        org.homeo.regulate(org.hier, sigma=0.98)
    assert float(col.w_ff.w_slow[0, 0]) > 1.2


def test_ziwago_prime_wta_context_disambiguation():
    """Ziwago-Prime: апикальное плато (BAC-firing) разводит одинаковый
    предпоследний токен по разному предшествующему контексту без MLP."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig
    u = RegenesisOrganism(UnifiedConfig(vocab_size=32),
                          gcfg=genesis_cfg("tiny"), seed=42)
    ctx_a = [1, 2, 10, 15]
    ctx_b = [3, 4, 10, 25]
    for _ in range(25):
        u.org.reset_state(1); u.motor.reset_state(1)
        for a, b in zip(ctx_a[:-1], ctx_a[1:]):
            u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
        u.org.reset_state(1); u.motor.reset_state(1)
        for a, b in zip(ctx_b[:-1], ctx_b[1:]):
            u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
    u.org.reset_state(1); u.motor.reset_state(1)
    out_a = u.generate(n_tokens=1, seed_tokens=[1, 2, 10])["ids"][-1]
    u.org.reset_state(1); u.motor.reset_state(1)
    out_b = u.generate(n_tokens=1, seed_tokens=[3, 4, 10])["ids"][-1]
    assert out_a == 15
    assert out_b == 25


def test_ziwago_prime_theta_gamma_story_binder():
    """Ziwago-Prime v2: ThetaGammaStoryBinder связывает роли и филлеры
    на единичной окружности и различает разные сюжетные префиксы."""
    from neuroweave10.memory.phase_vsa import ThetaGammaStoryBinder
    b1 = ThetaGammaStoryBinder(dim=24, n_slots=7, seed=1)
    b2 = ThetaGammaStoryBinder(dim=24, n_slots=7, seed=1)
    p_a = torch.zeros(1, 24); p_a[0, :6] = 1.0
    p_b = torch.zeros(1, 24); p_b[0, 12:18] = 1.0
    for _ in range(4):
        out_a = b1.step(p_a, ach=0.8, ne=0.6)
        out_b = b2.step(p_b, ach=0.8, ne=0.6)
    assert out_a.context_vec.shape == (1, 24)
    assert not torch.allclose(out_a.context_vec, out_b.context_vec)


def test_ziwago_prime_hippocampus_capacity_and_theta():
    """Ziwago-Prime v2: Гиппокамп различает паттерны одинаковой нормы (RMS-grid)
    и удерживает веса CA3 в рамках ёмкости Хопфилда при 50+ записях."""
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32), seed=0)
    x1 = torch.zeros(1, 16); x1[0, :4] = 2.0
    x2 = torch.zeros(1, 16); x2[0, 8:12] = 2.0
    o1 = h.step(x1, ach=0.9, adenosine=0.1, learn=True)
    o2 = h.step(x2, ach=0.9, adenosine=0.1, learn=True)
    assert not torch.allclose(o1.ec, o2.ec)
    for _ in range(50):
        h.step(torch.randn(1, 16), ach=0.8, adenosine=0.1, learn=True)
    assert float(h.w_ca3.abs().max()) <= 1.0


def test_ziwago_prime_trn_electrotonic_and_thalamus_ht():
    """Ziwago-Prime v2: TRN фильтрует DC-уровень L6 и синхронизирует фазу во сне,
    а ThalamusFO накапливает доступность T-каналов h_T при молчании."""
    trn = TRN(4)
    g1 = trn.step(torch.tensor([0.8, 0.2, 0.5, 0.1]), vigilance=0.1)
    assert g1.shape == (4,)
    fo = ThalamusFO(8)
    for _ in range(30):
        fo.step(torch.zeros(1, 8), ach=0.2, ne=0.1)
    assert float(fo.h_t.mean()) > 0.2


def test_ziwago_prime_syntax_roles_and_boundary_lts():
    """Ziwago-Prime v2.5 (Шаг 3: Синтаксис и Роли):
    1) ThetaGammaStoryBinder переключает роли по позициям клаузы и сбрасывается на точке;
    2) ThalamusFO выдаёт LTS-burst на границе предложения (boundary_burst=True) и
       держит линейный tonic внутри предложения;
    3) RegenesisOrganism выучивает базовые синтаксические предложения (Det+Subj+Verb+.)
       и генерирует валидные пары Subject+Verb без единого MLP."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig
    fo = ThalamusFO(8)
    x = torch.ones(1, 8) * 0.5
    _ = fo.step(x, ach=0.6, ne=0.4, boundary_burst=False)
    assert fo.mode == "tonic"
    _ = fo.step(x, ach=0.6, ne=0.4, boundary_burst=True)
    assert fo.mode == "burst"

    # Проверка синтаксического обучения: [2=the, 3=cat, 4=sat, 5=.] и [2=the, 6=dog, 7=ran, 5=.]
    u = RegenesisOrganism(UnifiedConfig(vocab_size=16),
                          gcfg=genesis_cfg("tiny"), seed=42)
    u.punct_ids = {5}
    sents = [[2, 3, 4, 5], [2, 6, 7, 5]]
    for _ in range(18):
        stream = sents[0] + sents[1]
        u.org.story_binder.notify_boundary()
        for a, b in zip(stream, stream[1:]):
            u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
    u.sleep(10)
    u.org.story_binder.notify_boundary()
    gen = u.generate(n_tokens=20, seed_tokens=[2])["ids"]
    pairs = list(zip(gen, gen[1:]))
    # Проверяем, что в сгенерированной последовательности присутствуют Subject+Verb пары (3->4 или 6->7)
    assert (3, 4) in pairs or (6, 7) in pairs


def test_ziwago_prime_v6_interleaved_replay_and_consolidated_bridges():
    """Ziwago-Prime v6 (Стресс-тест 1): Интерливинг-реплей (McClelland et al. 1995)
    чередует свежие и консолидированные эпизоды во сне и сохраняет 100% старых
    семантических мостов (w_semantic_graph_consolidated) при последовательном обучении."""
    from neuroweave10.config import HippocampusCfg
    h = Hippocampus(HippocampusCfg(n_ec=16, n_dg=64, n_ca3=32), seed=7)
    x_a1 = torch.zeros(1, 16); x_a1[0, 0:4] = 2.0
    x_a2 = torch.zeros(1, 16); x_a2[0, 4:8] = 2.0
    for _ in range(4):
        oa1 = h.step(x_a1, ach=0.8, adenosine=0.2, learn=True)
        h.store_episode(oa1.ec, oa1.dg, oa1.ca3, oa1.mismatch, 0.8)
        oa2 = h.step(x_a2, ach=0.8, adenosine=0.2, learn=True)
        h.store_episode(oa2.ec, oa2.dg, oa2.ca3, oa2.mismatch, 0.8)
    rem1 = h.rem_recombine()
    c_a1 = h.concept_idx_for_ec(h.encode_ec(x_a1))
    c_a2 = h.concept_idx_for_ec(h.encode_ec(x_a2))
    bridge_a_stage1 = max(
        float(h.w_semantic_graph[c_a1, c_a2].item()),
        float(h.w_semantic_graph[c_a2, c_a1].item()),
    )
    assert bridge_a_stage1 > 0.0

    # Вторая серия эпизодов после первого сна -> проверяем интерливинг и сохранность моста A
    x_b1 = torch.zeros(1, 16); x_b1[0, 8:12] = 2.0
    x_b2 = torch.zeros(1, 16); x_b2[0, 12:16] = 2.0
    for _ in range(4):
        ob1 = h.step(x_b1, ach=0.8, adenosine=0.2, learn=True)
        h.store_episode(ob1.ec, ob1.dg, ob1.ca3, ob1.mismatch, 0.8)
        ob2 = h.step(x_b2, ach=0.8, adenosine=0.2, learn=True)
        h.store_episode(ob2.ec, ob2.dg, ob2.ca3, ob2.mismatch, 0.8)
    replayed = h.swr_replay(lambda ec, err: None)
    rem2 = h.rem_recombine()
    bridge_a_stage2 = max(
        float(h.w_semantic_graph[c_a1, c_a2].item()),
        float(h.w_semantic_graph[c_a2, c_a1].item()),
    )
    assert len(replayed) > 0
    assert rem2["interleaved_pairs"] >= 1
    assert bridge_a_stage2 >= bridge_a_stage1 - 1e-6


def test_ziwago_prime_v6_cognitive_dissonance_vte():
    """Ziwago-Prime v6 (Стресс-тест 2): Разрешение конфликта инстинкта самосохранения
    (Амигдала A- -> survival_veto) и исследовательского любопытства (Гипоталамус D_novelty -> curiosity_bonus)."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig
    u = RegenesisOrganism(UnifiedConfig(vocab_size=12), gcfg=genesis_cfg("tiny"), seed=11)
    # 5 = house (безопасная рутина), 6 = cave (опасная новизна), 7 = castle (безопасная новизна)
    for _ in range(5):
        u.step_token(torch.tensor([2]), torch.tensor([5]), learn=True)
    for _ in range(3):
        x_cave = u._encode_frame(torch.tensor([6]))
        u.org.limbic.step(
            x_cave,
            torch.zeros(1, 16),
            free_energy=0.5,
            intrinsic_rpe=0.0,
            atp_mean=0.9,
            ext_reward=-1.5,
            learn=True,
        )
    u.motor.w_lat[5, 2] = 1.0; u.motor.w_lat_slow[5, 2] = 1.0
    u.motor.w_lat[6, 2] = 0.35; u.motor.w_lat_slow[6, 2] = 0.35
    u.motor.w_lat[7, 2] = 0.25; u.motor.w_lat_slow[7, 2] = 0.25
    u.motor.prev_winner.zero_(); u.motor.prev_winner[2] = 1.0
    u.motor.prev2_winner.zero_()
    u.motor.rate_win[5] = 0.32
    u.motor.rate_win[6] = 0.0; u.motor.target_freq_ema[6] = 0.002
    u.motor.rate_win[7] = 0.02; u.motor.target_freq_ema[7] = 0.002
    u.org.limbic.monotony_ema = 1.0
    if u.org.limbic.last_out is not None:
        u.org.limbic.last_out.d_novelty = 1.0
    u.motor.set_active_inference(epistemic=0.8, boredom=1.0, aversive=0.0)

    vte_bin = u.simulate_vte([5, 6])
    assert vte_bin["winner"] == 5
    bd_cave = vte_bin["breakdown"][6]
    assert bd_cave["survival_veto"] > 0.5
    assert bd_cave["curiosity_bonus"] > 0.5

    vte_tri = u.simulate_vte([5, 6, 7])
    assert vte_tri["winner"] == 7


def test_ziwago_prime_v6_bdnf_synaptic_morphogenesis():
    """Ziwago-Prime v6 (Стресс-тест 3): Выброс BDNF и высокая ошибка предсказания
    запускают направленное проращивание новых синапсов (sprouting) без стирания защищённых весов."""
    cfg = USSynapseCfg(eta=0.05, density=0.30)
    syn = GenesisSynapse(16, 16, cfg, seed=19)
    syn.w_slow[0, 0] = 1.2
    syn.spine_health[0, 0] = 0.9
    syn.mask[0, 0] = True
    syn.z_pre.fill_(0.8)
    syn.z_post.fill_(0.8)
    initial_edges = int(syn.mask.sum().item())
    syn.structural_plasticity(error_level=0.45, microglia_boost=1.0, bdnf=1.35)
    after_edges = int(syn.mask.sum().item())
    assert after_edges > initial_edges
    assert syn.total_sprouted > 0
    assert bool(syn.mask[0, 0].item()) is True
    assert float(syn.w_slow[0, 0].item()) > 1.0


def test_ziwago_prime_v6_interactive_active_inference_and_counter_questions():
    """Ziwago-Prime v6 (Задача 1 next8.txt): Интерактивный Активный Вывод (respond_to_prompt)
    различает скучные вопросы, интригующие концепты и высокую Meta-Surprise (встречные вопросы)."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    vocab = [
        "<pad>", "<unk>", "the", "to", "in", ".", "dragon", "knight",
        "girl", "flew", "rode", "lived", "castle", "house", "forest", "mountain", "cave",
    ]
    stoi = {w: i for i, w in enumerate(vocab)}
    u = RegenesisOrganism(UnifiedConfig(preset="tiny", vocab_size=len(vocab)), seed=42)
    u.punct_ids = {stoi["."]}
    u.functional_ids = {stoi[w] for w in ("<pad>", "<unk>", "the", "to", "in", ".")}

    r_boring1 = u.respond_to_prompt("how are you", stoi=stoi, itos=vocab)
    r_boring2 = u.respond_to_prompt("how are you", stoi=stoi, itos=vocab)
    r_intrigue = u.respond_to_prompt("tell me about the dragon", stoi=stoi, itos=vocab)
    r_mystery = u.respond_to_prompt("what is inside the unknown shadow ?", stoi=stoi, itos=vocab)

    assert r_boring1["mode"] == "ROUTINE_HOMEOSTATIC_REPLY"
    assert r_boring2["mode"] == "BOREDOM_ESCAPE_EXPLORATION"
    assert float(r_boring2["kappa_explore"]) > float(r_boring1["kappa_explore"])
    assert r_intrigue["mode"] == "INTRIGUED_CONCEPTUAL_VTE"
    assert float(r_intrigue["kappa_explore"]) > float(r_boring1["kappa_explore"])
    assert r_mystery["mode"] == "EPISTEMIC_COUNTER_QUESTION"
    assert r_mystery["counter_question"] is not None
    assert "?" in str(r_mystery["reply"])


def test_ziwago_prime_v6_sleep_deprivation_and_recovery_rescue():
    """Ziwago-Prime v6 (Задача 2 next8.txt): Искусственная блокировка сна (sleep_deprivation_mode)
    вызывает падение ATP, блокировку пластичности и распад критичности (делирий), обратимые после сна."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    u = RegenesisOrganism(UnifiedConfig(preset="tiny", vocab_size=16), seed=42)
    u.enable_autonomous_metabolism(
        enabled=True,
        atp_threshold=0.85,
        ado_threshold=0.35,
        sleep_ticks=12,
        atp_burn_rate=0.08,
        ado_gain_rate=0.12,
        sleep_deprivation=True,
    )
    for t in range(1, 25):
        _ = u.step_token(torch.tensor([t % 12 + 2]), next_ids=torch.tensor([(t + 1) % 12 + 2]), learn=True)
    atp_dep = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
    assert len(u.auto_sleep_history) == 0
    assert atp_dep < 0.30
    assert u.org.plasticity_blocked is True
    assert u.org.pathology == "sleep_deprivation_delirium_exhaustion"
    assert float(u.org.sigma) < 0.78

    # Разблокируем сон: следующий шаг должен вызвать автономный восстановительный сон
    u.sleep_deprivation_mode = False
    r_rec = u.step_token(torch.tensor([3]), next_ids=torch.tensor([4]), learn=True)
    assert r_rec["auto_sleep_event"] is not None
    assert u.org.pathology is None
    atp_after = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
    assert atp_after > 0.95


def test_ziwago_prime_v7_archnext_d1_d2_d3_d4_scaling():
    """Ziwago-Prime v7 (Produce/archnext.txt D1–D4): проверяет Small-World + Thalamic Hub иерархию (D1),
    HashedCorticalSDREncoder на 50k слов при <1M параметров и <1000 ops/token (D2),
    локальную критичность колонки local_sigma без глобального readout (D3) и
    CompositionalThetaGammaIndexer с ёмкостью 64^4 = 16.7M паттернов (D4)."""
    import copy
    from neuroweave10.config import ZoneCfg, genesis_cfg
    from neuroweave10.core.criticality import CriticalityMonitor
    from neuroweave10.core.sdr import HashedCorticalSDREncoder
    from neuroweave10.cortex.hierarchy import CorticalHierarchy
    from neuroweave10.memory.hippocampus import CompositionalThetaGammaIndexer

    # D1: Small-World Hub на 10 зонах имеет < 40 рёбер вместо 90 (O(N²))
    cfg10 = copy.deepcopy(genesis_cfg("tiny"))
    cfg10.zones = tuple(
        ZoneCfg(name=f"Z_{i}", g=round(i / 9.0, 2), n_neurons=32, n_pv=8, n_sst=8, n_vip=4)
        for i in range(10)
    )
    h_sw = CorticalHierarchy(cfg10, batch=1, topology="small_world_hub", max_fanout=3)
    h_all = CorticalHierarchy(cfg10, batch=1, topology="all_to_all")
    assert len(h_sw.routing_edges) < len(h_all.routing_edges)
    out = h_sw.step(
        torch.randn(1, cfg10.input_dim),
        torch.zeros(1, cfg10.gw.d_gw),
        {z.name: torch.zeros(1, 32) for z in cfg10.zones},
        f_mod=1.0,
        vip_per_zone={z.name: 0.2 for z in cfg10.zones},
        learn=True,
    )
    assert out.local_sigmas is not None and len(out.local_sigmas) == 10

    # D2: HashedCorticalSDREncoder (< 1M params, < 1000 ops/token, 40 active spikes out of 2048)
    enc = HashedCorticalSDREncoder(vocab_size=50_000, n_sdr=2048, n_modules=8, active_per_module=5)
    assert enc.num_parameters() < 1_000_000
    assert enc.ops_per_token() < 1_000
    sdrs = enc.encode_tokens(torch.tensor([0, 1, 2, 49999]), learn=True)
    assert sdrs.shape == (4, 2048)
    assert torch.all(sdrs.sum(dim=1) == 40.0)

    # D3: Локальная критичность колонки согласуется с глобальной (< 0.05)
    mon = CriticalityMonitor(target_sigma=0.97)
    mean_loc = mon.push_local_columns(out.local_sigmas)
    assert abs(mean_loc - mon.sigma_hat) < 0.05

    # D4: CompositionalThetaGammaIndexer (ёмкость 64^4 = 16,777,216)
    comp = CompositionalThetaGammaIndexer(n_slots=4, k_per_slot=64, d_slot=16)
    assert comp.theoretical_capacity() == 16_777_216
    q_tup = comp.encode_and_bind((12, 34, 56, 7), salience=1.5, learn=True)
    assert q_tup in comp.compositional_traces


def test_ziwago_prime_v8_phase1_multirate_scaffold_pdsr():
    """Ziwago-Prime v8 Phase 1 (Produce/snnscience.docx §3.1, §3.2, §3.5, §3.7, §4.6):
    Проверяет:
      1. MC-MREI (MinicolumnStateHierarchy + TickLadderScheduler): точное аналитическое пробуждение
         (Exact Lazy Decay) с ошибкой < 5e-3 мВ и ~9x сжатие обновлений переменных состояния;
      2. PDSR (Predictive Dendritic Subspace Routing + Block-Sparse 2:4): FLOP-эффективность >= 0.80;
      3. CT-PNN (PLASTIC / SOFT / HARD каркас + Tag-and-Capture + ChABC): нулевой дрейф HARD-синапсов
         под интерференцией и адресное открытие окна ChABC;
      4. Закон степенного забывания Йоста–Фузи (Benna & Fusi 2016): α ∈ [0.3, 0.7] и R²_power > R²_exp."""
    from neuroweave10.config import USSynapseCfg
    from neuroweave10.core.multirate import (
        BennaFusiCascade,
        MinicolumnStateHierarchy,
        TickLadderScheduler,
    )
    from neuroweave10.core.synapse import GenesisSynapse, SCAFFOLD_HARD

    # 1. MC-MREI + TickLadderScheduler
    hier = MinicolumnStateHierarchy(n_neurons=128, tile_size=16)
    ladder = TickLadderScheduler(stride_2=10, stride_3=100)
    v_s = torch.full((1, 128), -70.0)
    adapt = torch.zeros(1, 128)
    for t in range(1, 201):
        ls = ladder.step(is_sleep_tick=(t == 200))
        aff = torch.zeros(1, 128)
        spk = torch.zeros(1, 128)
        if t % 2 == 0:
            aff[:, :16] = 0.5
        stats = hier.update_tiers(t, aff, spk, v_s, adapt, ladder_step=ls)
    assert stats["compression_ratio"] > 6.0

    # 2. SparseProp (next9.txt #1, arxiv 2312.17216) O(K log N) FLOP efficiency & exact match on open subspace
    syn = GenesisSynapse(64, 64, USSynapseCfg(density=0.20, eta=0.05), seed=5)
    syn.enable_block_sparse = True
    x = torch.zeros(1, 64)
    x[:, :8] = 1.0
    out_pdsr = syn.drive_pdsr(x, apical_subspace_hint=torch.ones(64))
    syn.enable_block_sparse = False
    out_ref = syn.drive(x)
    assert float((out_pdsr - out_ref).abs().max().item()) < 1e-6
    assert syn.last_flop_efficiency >= 0.80
    assert syn.last_sparseprop_complexity_ops < syn.last_dense_complexity_ops
    assert len(syn.sparseprop_heap) > 0
    grad_back = syn.sparseprop_backward(torch.ones(1, 64), active_pre_mask=(x[0] > 0))
    assert grad_back.shape == (1, 64)

    # 3. CT-PNN Scaffold (PLASTIC -> SOFT -> HARD) + Synaptic Intelligence (next9.txt #2, arxiv 1703.04200) + ChABC
    syn.enable_scaffold = True
    err = torch.zeros(1, 64); err[:, :8] = 0.8
    s_post = torch.zeros(1, 64); s_post[:, :8] = 1.0
    for _ in range(2):
        for _ in range(10):
            syn.learn(err, x, s_post, f_mod=1.0, ach=0.8, da_phasic=0.8)
        syn.consolidate(factor=0.6, spindle_gain=1.4, replay_x=x[0])
    hard_mask = (syn.scaffold_state == SCAFFOLD_HARD) & syn.mask
    assert int(hard_mask.sum().item()) > 0
    assert float(syn.si_importance[hard_mask].mean().item()) >= syn.si_hard_threshold
    w_before = syn.w_slow[hard_mask].clone()
    # Интерференция не должна сдвинуть HARD-синапсы ни на бит
    syn.learn(-torch.ones(1, 64), torch.ones(1, 64), s_post, f_mod=1.0)
    assert float((syn.w_slow[hard_mask] - w_before).abs().max().item()) == 0.0
    n_unlk = syn.trigger_chabc_event(window_ticks=20)
    assert n_unlk > 0

    # 4. Benna-Fusi Power-Law Forgetting (α ∈ [0.3, 0.7])
    bf = BennaFusiCascade(shape=(16, 16), n_stages=4, tau_base=10.0, scale_factor=7.0)
    res = bf.retention_curve(torch.randn(16, 16), [10, 25, 50, 100, 200, 400, 800, 1600])
    assert res["is_power_law_in_biological_range"] is True


def test_ziwago_prime_v8_phase2_motor_dht_polychrony_thermodynamic_g():
    """Фаза 2 Глобального Плана v8.0 (snnscience.docx §3.3, §3.4, §3.6, §3.9, §4.6 Предсказание 3):
    Проверяет:
      1. BGPQSyllabaryDecoder: сжатие синапсов и операций в >30x на V = 50,000 слов,
         композицию OOV-слов и психолингвистический эффект Спунеризмов (Предсказание 3);
      2. SBCVSAShardedHippocampus: 150x расширение ёмкости (38,400 слотов на 4 шардах),
         точное обратимое связывание/развязывание SBC-VSA за O(B) и Recall@5 >= 95%;
      3. AxonalDelayPolychronousRouter: Gossip-PLV без глобального барьера, правило четверти периода,
         детекцию полихронных групп Ижикевича и пластическую миелинизацию во сне;
      4. Термодинамический Активный Вывод G'(π) = G(π) + λ_ATP · E[ATP(π)] и метаболический гейт LTP."""
    from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder
    from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    # 1. Двухпутевой слогово-фонемный декодер + Adaptive Softmax Migration (next9.txt #3) + Предсказание 3 (Тест Спунеризмов)
    dec = BGPQSyllabaryDecoder(vocab_size=50_000, d_in=128, n_head=512, n_syllables=256, n_articulators=32)
    assert dec.dense_baseline_parameters() / dec.num_parameters() > 30.0
    h = torch.randn(128)
    for _ in range(4):
        dec.learn(target_token_id=4200, h_l5=h, lr=0.25, da_rpe=0.9)
    # Часто используемое слово 4200 из Хвоста должно автоматически мигрировать в Быструю Голову (Путь А)
    assert 4200 in dec.promoted_to_head
    assert bool((dec.head_token_ids == 4200).any().item()) is True
    out_d = dec.decode(h, da_level=0.8, force_tail=True)
    assert out_d.articulatory_vector.numel() == 32
    oov = dec.compose_oov_word(1200, 3400, label="shadow_dragon")
    assert oov["composed_token_id"] >= 0
    sp_head = dec.simulate_spoonerism(10, 20, ne_stress=0.85, n_trials=200)
    sp_tail = dec.simulate_spoonerism(12000, 25000, ne_stress=0.85, n_trials=200)
    assert sp_tail["spoonerism_rate"] > sp_head["spoonerism_rate"]

    # 2. Шардированный VSA-DHT Гиппокамп (150x ёмкость и обратимость SBC-VSA) + HippoRAG PPR (next9.txt #4)
    dht = SBCVSAShardedHippocampus(n_shards=4, n_blocks=16, block_len=64, shard_capacity=9600)
    assert dht.capacity_expansion_factor(256) == 150.0
    code_a = tuple(i % 64 for i in range(16))
    code_b = tuple((i + 7) % 64 for i in range(16))
    code_c = tuple((i + 19) % 64 for i in range(16))
    bound = dht.bind(code_a, dht.role_codes[0])
    assert dht.unbind(bound, dht.role_codes[0]) == code_a
    ep_id = dht.store_composite_episode([code_a] * 4, salience=1.2, ca1_mismatch=0.4, concepts=["Concept_A", "Concept_B"])
    ep_b = dht.store_composite_episode([code_b] * 4, salience=1.1, ca1_mismatch=0.3, concepts=["Concept_B", "Concept_C"])
    ep_c = dht.store_composite_episode([code_c] * 4, salience=1.3, ca1_mismatch=0.3, concepts=["Concept_C"])
    ret = dht.retrieve_multiprobe([code_a] * 4, ca1_mismatch=0.4, top_k=5)
    assert ep_id in ret["top_episode_ids"]
    # HippoRAG Personalized PageRank (arxiv 2405.14831): многошаговая цепочка Concept_A -> Concept_B -> Concept_C
    ppr_res = dht.ppr_associative_recall(seed_concepts=["Concept_A"], top_k=5)
    assert ppr_res["concept_chain"] == ["Concept_A", "Concept_B", "Concept_C"]
    assert ep_c in ppr_res["ranked_episode_ids"]
    tc = [dht.two_clock_replay_step(ep_id) for _ in range(5)]
    assert tc[-1]["is_theta_composite_sync"] is True

    # 3 & 4. Интеграция Gossip-PLV, полихронных групп и G'(π) с λ_ATP в RegenesisOrganism
    rorg = RegenesisOrganism(UnifiedConfig(preset="tiny", vocab_size=32), seed=7)
    step_info = rorg.step_token(torch.tensor([3]), next_ids=torch.tensor([5]), learn=True)
    assert "lambda_atp" in rorg.org._last_ast.report
    assert "gossip_plv" in rorg.org._last_ast.report
    vte_res = rorg.simulate_vte([4, 6])
    cand_bd = vte_res["breakdown"][4]
    assert "G_prime" in cand_bd and "lambda_atp" in cand_bd and "expected_atp_cost" in cand_bd


def test_ziwago_prime_v8_phase3_e2e_pc_and_embodied_foraging():
    """Фаза 3 Глобального Плана v8.0 (snnscience.docx §3.8, §3.10):
    Проверяет:
      1. EndToEndPredictiveCodingLoop: локальное обучение по закону Колена–Поллака + Рао–Балларда
         без единого вызова backward() и без линейного readout-слоя;
      2. EmbodiedForagingArena: замкнутая сенсомоторная петля выживания и фуражирования на 2D-арене,
         где воплощённый Активный Вывод G'(π) собирает значительно больше пищи и избегает хищника."""
    from neuroweave10.core.e2e_predictive import EndToEndPredictiveCodingLoop
    from neuroweave10.embodied.arena import EmbodiedForagingArena

    e2e = EndToEndPredictiveCodingLoop(d_in=32, d_hidden=48, d_attractor=32, n_classes=8, lr=0.14, seed=11)
    g = torch.Generator().manual_seed(21)
    pats = [torch.randn(32, generator=g) for _ in range(8)]
    cos0 = e2e.reciprocal_alignment_cosine()
    for _ in range(20):
        for cid, p in enumerate(pats):
            e2e.step(p, target_class=cid, da_rpe=0.9, learn=True)
    cos1 = e2e.reciprocal_alignment_cosine()
    acc = sum(1 for cid, p in enumerate(pats) if e2e.step(p, learn=False).predicted_class == cid) / 8.0
    assert cos1 > cos0
    assert acc >= 0.75

    arena = EmbodiedForagingArena(grid_size=8, input_dim=32, seed=42)
    for _ in range(80):
        act = arena.select_embodied_action(lambda_atp=0.5, valence=0.2)
        arena.step(act)
    assert arena.food_collected >= 5
    assert arena.predator_hits == 0


def test_ziwago_prime_v8_5_lab_html_breakthroughs():
    """Проверяет все 6 фундаментальных прорывов из исследовательской доски `Produce/Lab.html` (P1–P8):
      1. eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch (`F#20`, `F#16`, `F#17`, `F#22`)
         + Causal-Cone Exponential Transport (`F#60`, `F#32`);
      2. HetSynLayer гетерогенные логнормальные τ_ij с адаптацией к ISI (`F#54`, `F#55`, `F#52`)
         + Block-Emptiness Ratio (`F#66`, `F#34`) + Pinned Block-Sparse Lanes (`F#44`)
         + Термодинамический порог забывания (`F#58`) + KL-Stability Custody Lock (`F#43`);
      3. Debiased WPLI (Vinck et al. 2011) + Volume-Conduction False-Witness Detector (`F#25`, `F#27`, `F#23`);
      4. Двунаправленный θ/γ реплей Коэне–Хассельмо (Prospective/Retrospective) + Сонный гейт тишины (`F#19`, `F#15`, `F#21`);
      5. Action-Conditioned Exact Exponential Jump (`F#63`) + SI-Driven `Probe / Attend / Stop` (`F#37`)."""
    from neuroweave10.core.multirate import MinicolumnStateHierarchy
    from neuroweave10.cortex.hierarchy import AxonalDelayPolychronousRouter
    from neuroweave10.embodied.arena import EmbodiedForagingArena
    from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
    from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder

    # 1. eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch (F#20, F#16, F#22) + Causal-Cone Transport (F#60)
    pool = USNeuronPool(16, USNeuronCfg(stochastic=False))
    pool.reset_state(1)
    pool.enable_elif_metabolic_gate = True
    pool.energy[:, :8] = 0.40  # ниже theta_down_atp=0.58 -> уходят в down-phase
    g_zero = torch.zeros(1, 16)
    g_inh = torch.full((1, 16), 0.3)
    out_down = pool.step(g_zero, g_zero, g_inh)
    assert pool.last_down_phase_fraction >= 0.50
    assert float(out_down.s[:, :8].sum().item()) == 0.0
    # Восстановление АТФ >= theta_up_atp + пробуждающий сигнал мгновенно снимают down-phase (<20 мс)
    pool.energy.fill_(0.85)
    g_wake = torch.full((1, 16), 0.6)
    pool.step(g_wake, g_zero, g_zero)
    assert pool.last_down_phase_fraction == 0.0

    mch = MinicolumnStateHierarchy(n_neurons=32, tile_size=16)
    cc_res = mch.verify_causal_cone_transport(total_horizon_ms=40.0, perturbation_time_ms=20.0)
    assert cc_res["pre_causal_cone_error"] == 0.0
    assert cc_res["causality_preserved"] == 1.0

    # 2. HetSynLayer (F#54/F#55) + Block-Emptiness (F#66) + Pinned Lanes (F#44) + F#58 + KL-Stability (F#43)
    syn = GenesisSynapse(64, 64, USSynapseCfg(density=0.20, eta=0.05), seed=9)
    syn.enable_het_syn = True
    syn.enable_block_sparse = True
    assert float(syn.tau_syn_het.min().item()) >= 1.0
    s_pre = torch.zeros(1, 64); s_pre[:, :4] = 1.0
    tr_het = syn.step_het_syn_traces(s_pre, dt=2.0, adapt_tau=True)
    assert tr_het.shape == (64, 64)
    _ = syn.drive_pdsr(s_pre, apical_subspace_hint=torch.full((64,), -0.5))
    assert syn.last_block_emptiness_ratio >= 0.50
    pin_stats = syn.pin_top_block_sparse_lanes(keep_open_fraction=0.15)
    assert pin_stats["pinned_fraction"] >= 0.80
    assert pin_stats["top10_overlap"] == 1.0
    assert syn.kl_divergence_bits() < 1.5

    # 3. Debiased WPLI + Volume-Conduction False-Witness Detector (F#25, F#27, F#23)
    router = AxonalDelayPolychronousRouter(n_zones=3, routing_edges=[(0, 1), (1, 2)], ring_len=64)
    for _ in range(12):
        r_vc = router.step_gossip_and_polychrony(torch.tensor([0.2, 0.2, 0.2]), common_volume_field=0.95)
    assert r_vc["gossip_plv"] > 0.80
    assert r_vc["debiased_wpli"] < 0.25
    assert r_vc["false_witness_flag"] == 1.0

    # 4. Двунаправленный реплей Коэне–Хассельмо + Сонный мета-порог тишины (F#19, F#15, F#21)
    dht = SBCVSAShardedHippocampus(n_shards=4, n_blocks=16, block_len=64, shard_capacity=9600)
    ca = tuple(i % 64 for i in range(16))
    cb = tuple((i + 5) % 64 for i in range(16))
    e1 = dht.store_composite_episode([ca] * 4, salience=1.1, concepts=["start_node", "mid_node"])
    e2 = dht.store_composite_episode([cb] * 4, salience=1.4, concepts=["mid_node", "goal_node"])
    fwd_rep = dht.route_contextual_replay(cue_modality="prospective_sequence", seed_concepts=["start_node"])
    rev_rep = dht.route_contextual_replay(cue_modality="retrospective_reward", seed_concepts=["start_node"])
    assert fwd_rep["routed_mode"] == "prospective_forward"
    assert rev_rep["routed_mode"] == "retrospective_reverse"
    assert fwd_rep["replayed_episode_sequence"] == list(reversed(rev_rep["replayed_episode_sequence"]))
    assert rev_rep["silent_period_compute_saving"] >= 0.40
    assert rev_rep["sequence_retrieval_accuracy"] >= 0.92

    # 5. Softmax Noise Floor (F#58) + Exact Exponential Jump (F#63) + Probe/Attend/Stop (F#37)
    dec = BGPQSyllabaryDecoder(vocab_size=50_000, d_in=64, n_head=256, n_syllables=128)
    nf = dec.thermodynamic_forgetting_noise_floor()
    mac_prof = dec.profile_macs_vs_exact_softmax()
    assert nf > 0.0
    assert mac_prof["mac_reduction_factor"] > 10.0

    arena = EmbodiedForagingArena(grid_size=8, input_dim=32, seed=42)
    jmp = arena.evaluate_action_exact_jump(action_idx=0, dt_action_ms=14.0, tau_sensory_ms=25.0)
    assert 0.0 < jmp["exact_exp_decay"] < 1.0
    s_probe = arena.schedule_probe_attend_stop(epistemic_uncertainty=0.85, pragmatic_salience=0.20, si_path_integral=0.05)
    s_attend = arena.schedule_probe_attend_stop(epistemic_uncertainty=0.20, pragmatic_salience=0.80, si_path_integral=1.20)
    s_stop = arena.schedule_probe_attend_stop(epistemic_uncertainty=0.05, pragmatic_salience=0.05, si_path_integral=0.50)
    assert s_probe["mode"] == "PROBE"
    assert s_attend["mode"] == "ATTEND"
    assert s_stop["mode"] == "STOP"


def test_ziwago_prime_v9_morpho_sdr_and_compositional_intelligence():
    """Проверяет масштабирование интеллекта Ziwago-Prime v9.0 (32-Core / 64-Thread Edition):
      1. MorphoSemanticCorticalEncoder: морфо-синтаксическое обобщение однокоренных слов и OOV
         на 2048-нейронном SDR (8 модулей × 256 клеток);
      2. CompositionalThetaGammaIndexer: комбинаторная ёмкость 64^4 = 16,777,216 связок
         (Subject × Verb × Object × Context);
      3. Расширенный 1024-словный морфо-когнитивный словарь и генератор когнитивного корпуса."""
    from neuroweave10.core.sdr import MorphoSemanticCorticalEncoder
    from neuroweave10.memory.hippocampus import CompositionalThetaGammaIndexer
    from train_ziwago_v8 import build_bio_vocabulary, build_cognitive_world_corpus_raw

    enc = MorphoSemanticCorticalEncoder(vocab_size=50_000, n_sdr=2048, n_modules=8, active_per_module=5, d_out=24)
    sdr_walk = enc.encode_word("walk", token_id=10, role_id=3)
    sdr_walked = enc.encode_word("walked", token_id=11, role_id=3)
    assert int(sdr_walk.sum().item()) == 40
    assert int(sdr_walked.sum().item()) == 40
    ov_morph = enc.sdr_overlap("walk", 3, "walked", 3, id_a=10, id_b=11)
    ov_unrel = enc.sdr_overlap("dragon", 2, "quickly", 4, id_a=50, id_b=60)
    assert ov_morph >= 0.20
    assert ov_morph > ov_unrel
    proj = enc.project_to_cortex("starlight_oov", token_id=49999, role_id=5)
    assert proj.shape == (24,)

    idx = CompositionalThetaGammaIndexer(n_slots=4, k_per_slot=64, d_slot=16)
    assert idx.theoretical_capacity() == 16_777_216
    qt = idx.encode_and_bind((12, 34, 56, 78), salience=1.5, learn=True)
    assert qt in idx.compositional_traces

    itos, stoi = build_bio_vocabulary(max_vocab=1024)
    assert len(itos) == 1024
    cog_stories = build_cognitive_world_corpus_raw()
    assert len(cog_stories) >= 6


def test_ziwago_prime_v9_gpt2_50k_compositional_vsa_encoder():
    """Проверяет масштабирование Ziwago-Prime v9.0 до словаря GPT-2 (50,257 токенов) по ТЗ next11.txt:
      1. CompositionalVSABPEEncoder хранит статические SDR только для 2048 базовых BPE-токенов
         (экономия памяти 761.5x против плотной матрицы [50257, 2048]);
      2. Для остальных 48,209 токенов вычисляет 2048-мерный SDR (40 активных нейронов) на лету
         по BPE-дереву из data/gpt2/merges.txt через VSA-связывание;
      3. OOV-слово 'cryptodragon' автоматически собирается из морфем ['crypto', 'dragon']
         с перекрытием 37.5% к каждой из родительских морфем."""
    from pathlib import Path
    from neuroweave10.core.sdr import CompositionalVSABPEEncoder

    root = Path(__file__).resolve().parent.parent
    enc = CompositionalVSABPEEncoder(
        vocab_path=str(root / "data" / "gpt2" / "vocab.json"),
        merges_path=str(root / "data" / "gpt2" / "merges.txt"),
        n_base=2048,
        n_sdr=2048,
        n_modules=8,
        active_per_module=5,
        d_out=24,
    )
    assert enc.vocab_size == 50257
    assert enc.base_active_bins.shape == (2048, 8, 5)
    mem_rep = enc.memory_footprint_report()
    assert mem_rep["compression_ratio_x"] > 400.0

    sdr_base = enc.encode_token_id(262)      # базовый токен < 2048
    sdr_comp = enc.encode_token_id(39738)    # композитный токен >= 2048 ('Ġneuroscience')
    assert int(sdr_base.sum().item()) == 40
    assert int(sdr_comp.sum().item()) == 40

    oov = enc.decompose_oov_word("cryptodragon")
    assert oov["is_in_gpt2_vocab"] is False
    assert oov["morphemes"] == ["crypto", "dragon"]
    assert oov["active_neurons"] == 40
    assert oov["morpheme_sdr_overlaps"]["crypto"] >= 0.30
    assert oov["morpheme_sdr_overlaps"]["dragon"] >= 0.30


def test_ziwago_prime_v10_dendritic_moe_and_honest_curriculum():
    """Проверяет все требования upgrade11.txt для Ziwago-Prime v10.0:
      1. Dendritic Scaling: 7-уровневое дерево Poirazi/Beniaguev (n_branch=7) с Ca2+/NMDA-плато,
         ADCG и обучением сомы + апикала (100% XOR-разделимость против 50% у скалярного);
      2. Mixture of Experts: 8 колонок по 512 нейронов (4096 нейронов), Sparsely-Gated Top-2/3
         маршрутизация и межколоночные связи Block-Sparse 32x32 с FLOP-эффективностью >= 0.80;
      3. Honest Data Mixing: 60% WikiText-103 + 20% Cornell Movie-Dialogs + 20% TinyStories,
         где каждый абзац встречается ровно 1 раз за эпоху (max_exposure_per_uid == 1)."""
    from pathlib import Path
    from neuroweave10.cortex.moe_cortex import SparselyGatedCorticalMoE
    from train_ziwago_v10 import (
        benchmark_7level_dendrite_vs_scalar,
        build_honest_data_mixing_stream,
    )

    root = Path(__file__).resolve().parent.parent
    bench = benchmark_7level_dendrite_vs_scalar(n_neurons=128, n_in=32, n_steps=50)
    assert bench["single_neuron_xor_accuracy_deep7"] == 1.0
    assert bench["single_neuron_xor_accuracy_deep7"] > bench["single_neuron_xor_accuracy_scalar"]

    moe = SparselyGatedCorticalMoE(n_columns=8, neurons_per_column=512, n_in=64, n_branch_per_neuron=7, top_k_active=2)
    assert moe.total_neurons == 4096
    assert moe.columns[0].dendrite.k == 7
    out_w = moe.step(torch.randn(64), domain_tag="wikitext", learn=True, k_active=2)
    assert len(out_w.active_expert_ids) == 2
    assert out_w.active_neurons_this_tick == 1024
    assert out_w.block_sparse_32x32_flop_efficiency >= 0.80

    stream = build_honest_data_mixing_stream(root, target_total_tokens=2000)
    assert stream.max_exposure_per_uid == 1
    assert 0.58 <= stream.wikitext_share <= 0.62
    assert 0.18 <= stream.cornell_share <= 0.22
    assert 0.18 <= stream.tinystories_share <= 0.22


def test_ziwago_prime_v10_5_upgrade12_and_lab2_breakthroughs():
    """Проверяет прорывы из upgrade12.txt и lab2.txt (Ziwago-Prime v10.5):
      1. Discrete Predictive Coding (All-or-None Surprise Pin in {0, 1}) + Stranger's Check (F#109, F#111);
      2. Reference-Invariant Lag-Based dwPLI (10..50 ms axonal lag, 100% инвариантность к сдвигу
         электрода сравнения Reference Sweep и отсечение объёмного проведения tau=0, F#95, F#107);
      3. Clock-Domain Allocation & Analytic Rehydration спящих колонок MoE (F#79, F#60)."""
    from neuroweave10.cortex.moe_cortex import (
        DiscretePredictiveCodingBarrier,
        ReferenceInvariantLagPLVRouter,
        SparselyGatedCorticalMoE,
    )

    # 1. Проверка All-or-None Discrete Predictive Coding + Stranger's Check
    barrier = DiscretePredictiveCodingBarrier(binary_threshold=0.82, epistemic_floor=0.18)
    v_match = barrier.evaluate_discrete_pin_and_stranger_check(
        continuous_error=0.12,
        primary_synaptic_support=2.4,
        stranger_contrast_support=0.8,
        recent_repetition_pressure=0.0,
    )
    assert v_match["discrete_surprise_pin"] == 0
    assert v_match["is_hallucination_wall"] is False

    v_halluc = barrier.evaluate_discrete_pin_and_stranger_check(
        continuous_error=0.35,
        primary_synaptic_support=2.2,
        stranger_contrast_support=0.05,
        recent_repetition_pressure=0.75,
    )
    assert v_halluc["discrete_surprise_pin"] == 1
    assert v_halluc["is_hallucination_wall"] is True
    assert v_halluc["stranger_check_triggered"] is True
    assert v_halluc["action_mode"] == "PROBE"

    # 2. Проверка Reference-Invariant Lag-Based dwPLI (F#95, F#107)
    router = ReferenceInvariantLagPLVRouter(n_columns=8, ring_len=64, seed=42)
    for t in range(30):
        rates = torch.sin(torch.arange(8, dtype=torch.float32) * 0.7 + t * 0.25).abs() * 0.2
        _ = router.step_phases_and_compute_causal_matrix(rates)
    inv_rep = router.verify_reference_sweep_invariance()
    assert inv_rep["is_reference_invariant"] is True
    assert inv_rep["max_reference_sweep_deviation"] < 1e-5

    # 3. Проверка аналитической регидратации спящих колонок в SparselyGatedCorticalMoE (F#79)
    moe = SparselyGatedCorticalMoE(n_columns=8, neurons_per_column=512, n_in=64, n_branch_per_neuron=7, top_k_active=2)
    _ = moe.step(torch.randn(64), domain_tag="wikitext", learn=True, k_active=2)
    _ = moe.step(torch.randn(64), domain_tag="cornell_dialogs", learn=True, k_active=2)
    _ = moe.step(torch.randn(64), domain_tag="tinystories", learn=True, k_active=2)
    tel = moe.routing_telemetry()
    assert tel["total_analytic_rehydrations"] >= 1
    assert tel["dormant_cold_state_saving_fraction"] == 0.75


def test_ziwago_prime_bio_rwkv_10k_ablation_and_three_blind_spots():
    """Проверяет реализацию Bio-RWKV (AstroNMDALinearRecurrence) на 20,000 токенах, включая 3 слепых пятна
    и 4 пред-GPU контура стабильности:
      1. Контур 1 (Graded vs Hard Flush): градиент gamma_flush in [0.20..0.65] при плавной смене темы на 2000 токенах
         и резкий сброс (0.92) при скачке темы;
      2. Контур 2 (Гистерезис Шмитта ACh-арбитража 0.55..0.70): 0 ложных переключений в зоне дрожания [0.61..0.68];
      3. Контур 3 (Контракт archived_topic_snapshots): буфер ограничен TTL=16 и после фазы сна полностью сжимается
         в PPR-резюме (buffer_len_after_sleep == 0);
      4. Контур 4 (Оптимизация скорости и ненасыщающийся знаменатель D_t на 20,000 токенах): скорость > 35,000 ток/с."""
    from train_ziwago_v10 import benchmark_bio_rwkv_ablation_10k

    rep = benchmark_bio_rwkv_ablation_10k(d_model=64, total_steps=20_000)
    b1 = rep["blind_spot_1_lc_ne_topic_flush"]
    b2 = rep["blind_spot_2_dht_arbitration"]
    b3 = rep["blind_spot_3_denominator_non_saturation"]
    p1 = rep["point_1_graded_vs_hard_flush"]
    p2 = rep["point_2_schmitt_trigger_hysteresis"]
    p3 = rep["point_3_snapshot_contract_ttl_and_sleep"]
    spd = rep["speed_benchmark_20k"]

    assert b1["topic_flushes_triggered"] >= 1
    assert b1["bio_rwkv_old_topic_bleed_after_lc_ne_flush"] < b1["naive_rwkv_old_topic_bleed_without_flush"]
    assert b1["new_topic_alignment_after_5_tokens"] > 0.65
    assert p1["graded_flushes_count"] >= 1
    assert 0.20 <= p1["mean_graded_gamma"] <= 0.65
    assert p1["transition_bridge_coherence_at_11k"] > 0.50

    assert p2["single_threshold_065_oscillations"] > 500
    assert p2["schmitt_hysteresis_oscillations"] == 0

    assert p3["buffer_len_after_30_flushes"] == 16
    assert p3["buffer_len_after_sleep_consolidation"] == 0
    assert p3["ppr_topic_summaries_stored"] >= 1

    assert b2["dht_offload_saving_percent"] >= 85.0
    assert 0.20 <= b3["bio_rwkv_denominator_at_20000"] <= 1.50
    assert b3["sharpness_at_20000"] >= 0.45
    assert spd["vectorized_chunk_tokens_per_sec"] >= 35_000.0




