"""Конфигурация NEUROWEAVE v10.0 «Genesis».

Все величины — в мВ и мс (биологическое время; 1 такт = dt = 1 мс).
Чистые dataclass'ы без torch: сериализуемы в YAML/JSON.

Временные константы сведены по Приложению A манифеста: Mg-блок <0.1 мс,
AMPA/GABA-A 1-20 мс, таламус 5-30 мс, TRN 20-50 мс, NMDA 10-200 мс,
кортикальный градиент τ(g): SENS 20 → ASSOC 100 → PFC 1000 мс,
SWR ~125 мс, GW-цикл ~250 мс, 2-AG 1-30 с, orexin 0.5-60 с,
кортизол 10 мин - 24 ч, аденозин/лактат минуты-часы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Биофизика нейрона Urbanczik-Senn (три компартмента)
# ---------------------------------------------------------------------------
@dataclass
class USNeuronCfg:
    """Мультикомпартментный пирамидальный нейрон (манифест §1.2).

    C_m dV_S/dt = -g_L(V_S-E_L) + g_D(V_D^basal-V_S) + g_A(V_A^apical-V_S)
                  - I_fast(GABA) - I_metabolic(KATP+GIRK) + I_ext
    Спайк: V_S > Θ;  φ(V) = φ_max/(1+exp(-β(V-Θ))) — частотная передаточная.
    """
    c_m: float = 1.0
    g_l: float = 0.05           # утечка (нормировка проводимостей)
    e_l: float = -70.0          # реверсия утечки
    g_d: float = 0.25           # связь сома↔базальный дендрит
    g_a: float = 0.05           # связь сома↔апикальный дендрит
    e_exc: float = 0.0          # реверсия AMPA/NMDA
    e_cl: float = -75.0         # реверсия GABA_A Cl- (REGENESIS Закон 3)
    e_k: float = -90.0          # реверсия K+ (KATP/GIRK — метаболический ток)
    tau_d: float = 10.0         # τ дендритов (базальный)
    tau_d_apical: float = 60.0  # τ апикального дендрита (NMDA-доминант)
    phi_max: float = 100.0      # Гц — максимум φ(V)
    beta: float = 1.0           # крутизна φ (1/мВ)
    theta: float = -50.0        # базовый порог Θ
    v_reset_drop: float = 20.0  # сброс вычитанием: V -= s·(Θ-E_L)
    refractory: int = 2         # тактов абсолютной рефрактерности
    tau_adapt: float = 300.0    # τ адаптации порога (ALIF)
    beta_adapt: float = 0.02    # мВ на спайк адаптации
    tau_ca: float = 400.0       # τ постсинаптического Ca2+ (→ 2-AG, NO)
    mg: float = 1.0             # мМ Mg2+ для Jahr-Stevens блока NMDA
    k_atp: float = 0.6          # сила KATP-тока при дефиците ATP
    k_girk: float = 0.4         # сила GIRK-тока при аденозине (A1)
    hist_theta_gain: float = 3.0  # гистамин H1 → ↓Θ (антагонист аденозина)
    allo_gain: float = 0.5      # η_ALLO: нейростероидная амплификация GABA
    sigma: float = 0.02         # сома-шум (OU, мВ·√мс)
    tau_ou: float = 10.0        # τ OU-шума (коррелированный шум мембраны)
    ado_theta_gain: float = 6.0 # Ado A1 → +Θ (честная усталость, Закон 5)
    stochastic: bool = True
    dt: float = 1.0


@dataclass
class USSynapseCfg:
    """Синапс трёхфакторного правила (манифест §1.3).

    dw/dt = η · f_mod(M) · [S_i - φ(V*_D,i)] · x_j
    V*_D = (g_L E_L + g_D V_D)/(g_L+g_D) — «обещанный» потенциал дендрита.
    """
    eta: float = 1e-3           # базовая скорость обучения
    tau_x: float = 20.0         # τ пре-следа x_j
    theta_m0: float = 0.0       # базовый метапластический порог BCM
    w_min: float = -2.0
    w_max: float = 2.0
    decay: float = 0.9999       # медленная утечка веса
    density: float = 0.5        # плотность маски связей
    sign: str = "exc"           # 'exc' | 'inh'
    # Вогельс iSTDP (только inh): Δw на пре-спайк = η(x_post-α), на пост = η·x_pre
    istdp_eta: float = 1e-4
    istdp_rho0: float = 0.05    # целевая частота пост-нейрона


@dataclass
class EncoderCfg:
    """SDR-энкодер LCA (манифест §II.1.1, Rozell 2008)."""
    n_sdr: int = 256            # ширина SDR (2048 в каноне, компактно в PoC)
    sparsity_lambda: float = 0.10
    tau_lca: float = 10.0       # мс
    lca_steps: int = 50         # итераций вывода
    dict_lr: float = 0.02       # Hebbian dictionary learning
    active_frac: float = 0.08   # целевая доля активных (2-5%)


@dataclass
class ZoneCfg:
    """Кортикальная зона на градиенте Margulies (манифест §II.2.3)."""
    name: str
    g: float                    # позиция на градиенте [0..1]
    n_neurons: int = 64         # пирамидальных
    n_pv: int = 12              # перисоматическое торможение
    n_sst: int = 8              # дендритное (апикальное) торможение
    n_vip: int = 6              # дизингибиция SST
    eta: float = 1e-3

    @property
    def tau_m(self) -> float:
        """τ(g) = τ_min·(τ_max/τ_min)^g — экспоненциальный градиент."""
        return 20.0 * (1000.0 / 20.0) ** self.g


@dataclass
class HippocampusCfg:
    """Трисинаптический контур (манифест §II.3.1)."""
    n_ec: int = 64              # энторинальная кора (MEC grid + LEC)
    n_dg: int = 256             # экспансия DG, ~1-2% активных
    dg_sparsity: float = 0.02
    n_ca3: int = 128            # автоассоциативная сеть Хопфилда
    ca3_fan_in: float = 0.08
    ca3_steps: int = 6          # итераций pattern completion
    eta_ca3: float = 0.01       # one-shot: η_CA3 = 10·η_cortex
    ca1_threshold: float = 0.3  # порог mismatch → VTA/LC burst
    neurogenesis_period: int = 500   # тактов между нейрогенезом DG
    swr_period_ms: float = 125.0     # период SWR-реплея
    n_replay_batch: int = 4
    replay_beta_unc: float = 0.5     # β веса epistemic uncertainty в приоритете


@dataclass
class BasalGangliaCfg:
    """DDM + Go/NoGo/STN (манифест §II.3.3)."""
    n_actions: int = 64
    a0: float = 1.0             # базовый порог DDM
    k_stn: float = 0.5          # конфликт → порог (гиперпрямой путь)
    sigma_ddm: float = 0.10     # шум аккумулятора
    leak: float = 0.02          # утечка аккумулятора
    alpha_rl: float = 0.05      # скорость RL на Go/NoGo
    opioid_gain: float = 0.4    # опиоиды (Enk/β-End) → гедонистический DA-фактор стриатума
    gamma_td: float = 0.95      # дисконт TD
    hyperdirect_ms: float = 7.0 # STN приходит за 5-10 мс до действия
    alpha_ox: float = 0.3       # орексин ↓ порог
    beta_ox: float = 0.5        # орексин ↑ гейн δ
    alpha_ado: float = 0.5      # аденозин ↑ порог
    k_het: float = 0.5          # A2A-D2 гетеромер: ослабление D2
    ht_patience: float = 0.4    # 5-HT ↑ порог (терпение)


@dataclass
class WorkspaceCfg:
    """Global Workspace как Modern Hopfield Network (§II.4.1)."""
    d_gw: int = 128             # информационное горлышко
    n_slots: int = 4            # K = 4 (Cowan 2001)
    beta_gw: float = 8.0        # крутизна ignition (NE-модулируема)
    s_th: float = 0.5           # порог ignition (sigmoid access)
    gamma_ign: float = 12.0
    retrieval_steps: int = 3    # итераций ξ_new = X·softmax(βXᵀξ)
    decay: float = 0.92         # затухание содержимого без ignition


@dataclass
class MetaHeadCfg:
    """HOT MetaHead (§II.4.2)."""
    d_hidden: int = 64
    lambda_brier: float = 0.5   # вес Brier score в L_meta
    lr: float = 5e-3
    theta_low: float = 0.4      # confidence < θ_low → гейт действий
    theta_uncertain: float = 0.5
    theta_epistemic: float = 0.5
    n_ensemble: int = 4         # прогонов GW для декомпозиции неопределённости
    tau_trace: float = 50.0


@dataclass
class AttentionSchemaCfg:
    """AST (§II.4.3)."""
    n_modules: int = 5          # SENS, ASSOC, PFC, HPC, BG
    d_schema: int = 32
    alpha_ast: float = 0.1      # гейн-коррекция TRN
    lr: float = 2e-3


@dataclass
class ChemistryCfg:
    """Нейромодуляторное облако (§1.4)."""
    da_tau_phasic: float = 100.0     # фазический DA (RPE)
    da_tau_tonic: float = 60000.0    # тонический DA (средняя награда, Niv 2007)
    ne_tau: float = 75.0
    ach_tau: float = 500.0
    ht_tau: float = 2000.0
    ox_tau: float = 30000.0          # орексин
    ado_tau: float = 3600000.0       # аденозин (мин-часы): ускорено в PoC
    ado_wake_gain: float = 2e-5      # dS/dt = α·A·W - β·S(1-W)
    ado_sleep_gain: float = 5e-5
    sleep_threshold: float = 0.6     # S > θ_sleep → сон
    ox_collapse: float = 0.15        # [OX] < collapse → потеря wake-аттрактора
    cort_tau_nongenomic: float = 600000.0
    cort_tau_genomic: float = 43200000.0
    theta_stress: float = 0.6        # MR→GR переход
    alpha_da: float = 1.0            # вес DA в f_mod
    beta_ne: float = 0.5             # вес NE в f_mod
    kappa_ecb: float = 0.7           # 2-AG супрессия в f_mod
    theta_energy: float = 0.2        # H(E-θ_E): энергетический гейт пластичности
    lactate_tau: float = 1800000.0   # ANLS-шаттл
    lactate_spend: float = 4e-5      # трата ATP на спайк
    lactate_recover: float = 8e-5
    ecb_tau: float = 10000.0         # 2-AG фазический 1-30 с
    ecb_synth: float = 0.15
    ecb_ic50: float = 0.32
    no_tau: float = 1000.0           # NO 0.1-2 с
    no_d: float = 0.33               # эфф. диффузия (μm²/ms ≈ 3300 μm²/s)
    no_radius: int = 16              # радиус гетеросинаптической LTP (соседи)
    h2s_tau: float = 60000.0         # H2S → D-serine → NMDA (минуты)
    mg_base: float = 1.0
    zn_tau: float = 500.0            # Zn2+ миллисекундная шкала
    atp_debt_block: float = 0.5  # аденилатный долг >θ → w_fast заморожены


@dataclass
class HomeostasisCfg:
    """SOC-регуляция σ≈1 (§1.5)."""
    target_sigma: float = 0.98    # in-vivo «ревербирующий» режим (Wilting-Priesemann)
    sigma_tol: float = 0.05
    sigma_lo: float = 0.90
    sigma_hi: float = 1.05
    mr_kmax: int = 20               # горизонт MR-оценщика
    ip_rate: float = 1e-3           # intrinsic plasticity ΔΘ=η(x-H)
    ip_target: float = 0.08         # целевая частота H_IP
    inh_stdp_rate: float = 1e-4     # Vogels
    depression_tau: float = 500.0   # восстановление ресурса u_syn
    depression_delta: float = 0.15  # u→u(1-Δ) при спайке
    wnorm_every: int = 200          # нормировка весов (SORN)


@dataclass
class PathologyCfg:
    """Критические патологии — автоматическая остановка (§4.1)."""
    sigma_epilepsy: float = 1.2     # >100 мс подряд
    sigma_epilepsy_ms: int = 100
    sigma_coma: float = 0.7         # >500 мс подряд
    sigma_coma_ms: int = 500
    energy_collapse_frac: float = 0.5   # E<0 у >50% нейронов
    eternal_sleep_ticks: int = 1000000  # 1e6 в каноне; снижается для тестов
    eternal_wake_ticks: int = 1000000
    w_explode: float = 1e3
    w_collapse: float = 1e-6


@dataclass
class MotorCfg:
    """Моторная кора REGENESIS (Закон 1): спайковый пул вместо MLP.

    vocab нейронов с той же ALIF-проводимостной физикой; выбор токена —
    WTA (первый/максимум спайков за суб-такты), никакого argmax по логитам
    обучаемого декодера. STP Tsodyks-Markram на афферентах — усталость
    вместо алгоритмического Veto."""
    tau_m: float = 10.0         # мембранная τ моторных нейронов, мс
    theta0: float = -50.0       # базовый порог, мВ
    e_l: float = -70.0
    e_exc: float = 0.0
    e_cl: float = -75.0
    g_l: float = 0.1
    beta_adapt: float = 0.05    # ALIF-адаптация моторного порога
    tau_adapt: float = 200.0
    refractory: int = 3         # абсолютный рефрактерный период, тиков
    wta_inh: float = 1.2        # глобальное делительное торможение (PV-пул)
    sub_ticks: int = 5          # внутренних мс на один токен-шаг
    sigma_ou: float = 0.05      # OU-шум мембран в фазе обучения
    # STP Tsodyks-Markram на синапсах L5→моторный пул
    stp_u0: float = 0.5         # базальная утилизация везикул U
    stp_tau_d: float = 200.0    # восстановление ресурса x, мс
    stp_tau_f: float = 50.0     # распад утилизации u, мс
    out_tau: float = 25.0       # усталость выхода нейрона (вес 1→0.2 за ~30 мс)
    eta_motor: float = 2e-3     # скорость трёхфакторного обучения w_l5


@dataclass
class GenesisConfig:
    """Полная конфигурация организма Genesis."""
    input_dim: int = 64
    n_actions: int = 64
    dt: float = 1.0
    seed: int = 10

    neuron: USNeuronCfg = field(default_factory=USNeuronCfg)
    synapse: USSynapseCfg = field(default_factory=USSynapseCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    zones: List[ZoneCfg] = field(default_factory=lambda: [
        ZoneCfg("SENS", g=0.0, n_neurons=48),
        ZoneCfg("ASSOC", g=0.5, n_neurons=96),
        ZoneCfg("PFC", g=0.9, n_neurons=48),
    ])
    hippo: HippocampusCfg = field(default_factory=HippocampusCfg)
    bg: BasalGangliaCfg = field(default_factory=BasalGangliaCfg)
    gw: WorkspaceCfg = field(default_factory=WorkspaceCfg)
    meta: MetaHeadCfg = field(default_factory=MetaHeadCfg)
    ast: AttentionSchemaCfg = field(default_factory=AttentionSchemaCfg)
    chem: ChemistryCfg = field(default_factory=ChemistryCfg)
    homeo: HomeostasisCfg = field(default_factory=HomeostasisCfg)
    patho: PathologyCfg = field(default_factory=PathologyCfg)
    motor: MotorCfg = field(default_factory=MotorCfg)

    def zone(self, name: str) -> ZoneCfg:
        for z in self.zones:
            if z.name == name:
                return z
        raise KeyError(name)


def genesis_cfg(preset: str = "poc") -> GenesisConfig:
    """Пресеты: tiny (юнит-тесты) / poc (стенд) / full (канонические размеры)."""
    if preset == "tiny":
        return GenesisConfig(
            input_dim=24, n_actions=24,
            encoder=EncoderCfg(n_sdr=64, lca_steps=30),
            zones=[ZoneCfg("SENS", 0.0, 24, 4, 3, 2),
                   ZoneCfg("ASSOC", 0.5, 32, 6, 4, 3),
                   ZoneCfg("PFC", 0.9, 24, 4, 3, 2)],
            hippo=HippocampusCfg(n_ec=32, n_dg=96, n_ca3=64),
            bg=BasalGangliaCfg(n_actions=24),
            gw=WorkspaceCfg(d_gw=48),
            meta=MetaHeadCfg(d_hidden=32))
    if preset == "full":
        return GenesisConfig(
            input_dim=256, n_actions=256,
            encoder=EncoderCfg(n_sdr=2048, lca_steps=80),
            zones=[ZoneCfg("SENS", 0.0, 256), ZoneCfg("ASSOC", 0.5, 512),
                   ZoneCfg("PFC", 0.9, 256)],
            gw=WorkspaceCfg(d_gw=512))
    return GenesisConfig()
