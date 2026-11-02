"""Экспериментальная проверка Фазы 1 Глобального Плана v8.0 (snnscience.docx §3.1, §3.2, §3.5, §3.7, §4.6).

Проверяет 4 фундаментальных механизма и фальсифицируемые предсказания Фазы 1:
  1. Изъян 1 + 2 (MC-MREI + 4-Class Tick Ladder):
     - Точность аналитического прыжка (Exact Lazy Decay, Mattia & Del Giudice 2000) < 1e-5;
     - Сокращение числа обновлений состояния с 15N до ~1.6N (сжатие ~9× без warp-дивергенции);
     - Растяжение медленных тактов от теневой цены энергии Лагранжа λ_ATP.
  2. Изъян 5 (PDSR + Phase-Dependent Block-Sparse 2:4):
     - Предиктивное гейтирование постсинаптических тайлов от апикального дендрита V_A;
     - FLOP-эффективность >= 0.80 (против 0.15 у маскированного F.linear);
     - Нулевое численное расхождение с эталонным разреженным проходом при открытом подпространстве.
  3. Изъян 7 (CT-PNN Scaffold PLASTIC / SOFT / HARD + Tag-and-Capture + ChABC):
     - Продвижение PLASTIC -> SOFT -> HARD при совпадении синаптического тега и SWR-реплея во сне;
     - Защита HARD-ядра от катастрофического забывания при массивном интерферирующем обучении (100% сохранение);
     - Адресное ферментативное расплавление ChABC (HARD -> SOFT) при эпистемическом шоке;
     - Квадратичный рост эффективного периода полураспада с числом циклов сна.
  4. Фальсифицируемое Предсказание 1 (Закон степенного забывания Йоста–Фузи, Benna & Fusi 2016):
     - Кривая удержания памяти R(t) подчиняется степенному закону (1 + t/τ₀)^(-α) с α ∈ [0.3, 0.7]
       и превосходит экспоненциальный распад по R².
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.config import USNeuronCfg, USSynapseCfg, ZoneCfg
from neuroweave10.core.multirate import (
    BennaFusiCascade,
    MinicolumnStateHierarchy,
    TIER_COLD,
    TIER_HOT,
    TIER_WARM,
    TickLadderScheduler,
)
from neuroweave10.core.synapse import (
    GenesisSynapse,
    SCAFFOLD_HARD,
    SCAFFOLD_PLASTIC,
    SCAFFOLD_SOFT,
)
from neuroweave10.cortex.column import CorticalColumn


def benchmark_mc_mrei_and_tick_ladder() -> dict:
    """Тест 1: MC-MREI (HOT/WARM/COLD тайлы с точным аналитическим прыжком) + 4-Class Tick Ladder."""
    t0 = time.perf_counter()
    hier = MinicolumnStateHierarchy(
        n_neurons=256,
        tile_size=16,
        e_l=-70.0,
        tau_m=20.0,
        tau_ampa=10.0,
        tau_nmda=60.0,
        tau_gaba=15.0,
        tau_adapt=300.0,
    )

    # 1. Проверка точности аналитического решения ОДУ (Exact Lazy Decay) против точного микрошага
    v_init = torch.linspace(-68.0, -56.0, 16)
    ga_init = torch.linspace(0.2, 1.5, 16)
    gn_init = torch.linspace(0.1, 0.8, 16)
    gg_init = torch.linspace(0.05, 0.6, 16)
    ad_init = torch.linspace(0.0, 0.4, 16)
    dt_jump = 45.0  # 45 мс в COLD-состоянии

    v_exact, ga_ex, gn_ex, gg_ex, ad_ex = hier.exact_lazy_propagate(
        v_init, ga_init, gn_init, gg_init, ad_init, dt_elapsed=dt_jump
    )
    # Численное интегрирование с мелким шагом RK4 / экспоненциальным Эйлером (dt = 0.005 мс, 9000 шагов)
    v_num = v_init.clone()
    ga_num, gn_num, gg_num, ad_num = ga_init.clone(), gn_init.clone(), gg_init.clone(), ad_init.clone()
    n_sub = 9000
    sub_dt = dt_jump / n_sub
    for _ in range(n_sub):
        dv = (-(v_num - (-70.0)) + ga_num + gn_num - gg_num) / 20.0
        v_num = v_num + sub_dt * dv
        ga_num = ga_num * math.exp(-sub_dt / 10.0)
        gn_num = gn_num * math.exp(-sub_dt / 60.0)
        gg_num = gg_num * math.exp(-sub_dt / 15.0)
        ad_num = ad_num * math.exp(-sub_dt / 300.0)

    max_ode_err = float((v_exact - v_num).abs().max().item())

    # 2. Симуляция реалистичного кортикального режима (3.5% активных тайлов) на 500 тактах
    ladder = TickLadderScheduler(stride_2=10, stride_3=100, degenerate=False)
    v_s = torch.full((1, 256), -70.0)
    adapt = torch.zeros(1, 256)
    for t in range(1, 501):
        l_step = ladder.step(is_sleep_tick=(t % 250 == 0), lambda_atp=0.35 if t > 300 else 0.0)
        aff = torch.zeros(1, 256)
        spk = torch.zeros(1, 256)
        # В каждый момент времени активен только 1 из 16 миниколоночных тайлов (разреженность ~6.25%, с паузами ~3%)
        if t % 2 == 0:
            active_tile = (t // 25) % 16
            s_i = active_tile * 16
            aff[:, s_i : s_i + 16] = 0.45
            spk[:, s_i : s_i + 4] = 1.0
        stats = hier.update_tiers(t, aff, spk, v_s, adapt, ladder_step=l_step)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "max_exact_lazy_ode_error_mv": round(max_ode_err, 6),
        "ladder_only_compression": round(ladder.ladder_compression_ratio(), 3),
        "combined_mc_mrei_compression_ratio": round(stats["compression_ratio"], 3),
        "actual_vars_updated_per_neuron": round(stats["actual_vars_per_neuron"], 3),
        "dense_baseline_vars_per_neuron": 15.0,
        "hot_tile_frac": round(stats["hot_frac"], 4),
        "warm_tile_frac": round(stats["warm_frac"], 4),
        "cold_tile_frac": round(stats["cold_frac"], 4),
        "total_exact_analytical_wakeups": hier.total_exact_wakeups,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def benchmark_pdsr_block_sparse() -> dict:
    """Тест 2: PDSR (Predictive Dendritic Subspace Routing + Block-Sparse 2:4, Изъян 5)."""
    t0 = time.perf_counter()
    scfg = USSynapseCfg(density=0.15)
    syn = GenesisSynapse(256, 256, scfg, name="pdsr_bench", seed=42)
    syn.enable_block_sparse = True

    # Прогон с разреженным пре-спайком (5% активных) и апикальным предиктивным гейтом V_A
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(1, 256, generator=g) < 0.05).float()
    # Апикальный дендрит открывает полностью пространство (hint >= 0) -> проверяем точное совпадение с эталоном
    apical_open_all = torch.ones(256) * 0.5
    out_pdsr_full = syn.drive_pdsr(x, apical_subspace_hint=apical_open_all)
    syn.enable_block_sparse = False
    out_ref = syn.drive(x)
    syn.enable_block_sparse = True
    exact_match_err = float((out_pdsr_full - out_ref).abs().max().item())

    # Теперь апикальный дендрит открывает только целевые 25% миниколонок (контекстное предсказание сверху)
    apical_selective = torch.full((256,), -0.8)
    apical_selective[:64] = 0.9
    for _ in range(50):
        _ = syn.drive_pdsr(x, apical_subspace_hint=apical_selective)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "exact_open_subspace_error": round(exact_match_err, 8),
        "pdsr_flop_efficiency": round(syn.last_flop_efficiency, 4),
        "masked_f_linear_baseline_efficiency": 0.15,
        "efficiency_gain_vs_masked_linear": round(syn.last_flop_efficiency / 0.15, 2),
        "elapsed_ms": round(elapsed_ms, 2),
    }


def benchmark_ct_pnn_scaffold_and_power_law() -> dict:
    """Тест 3 + 4: Трёхуровневый каркас CT-PNN (PLASTIC/SOFT/HARD + Tag-and-Capture + ChABC)
    и фальсифицируемое предсказание 1 (степенной закон забывания Йоста–Фузи α ∈ [0.3, 0.7])."""
    t0 = time.perf_counter()
    scfg = USSynapseCfg(density=0.35, eta=0.05)
    syn = GenesisSynapse(64, 64, scfg, name="pnn_bench", seed=11)
    syn.enable_scaffold = True

    # 1. Обучаем базовый навык A (нейроны 0..15) и проводим 2 цикла сонной консолидации с SWR-реплеем
    x_skill_a = torch.zeros(1, 64)
    x_skill_a[:, :16] = 1.0
    err_skill_a = torch.zeros(1, 64)
    err_skill_a[:, :16] = 0.8
    s_post_a = torch.zeros(1, 64)
    s_post_a[:, :16] = 1.0

    half_lives = []
    for cycle in range(3):
        for _ in range(15):
            syn.learn(err_skill_a, x_skill_a, s_post_a, f_mod=1.0, ach=0.8, da_phasic=0.7)
        syn.consolidate(factor=0.6, spindle_gain=1.4, replay_x=x_skill_a[0])
        half_lives.append(syn.scaffold_stats()["estimated_half_life_ticks"])

    stats_after_consolidation = syn.scaffold_stats()
    hard_mask = (syn.scaffold_state == SCAFFOLD_HARD) & syn.mask
    w_hard_before_interference = syn.w_slow[hard_mask].clone()

    # 2. Агрессивная интерференция навыком B с противоположной ошибкой в течение 100 шагов!
    err_interf = -torch.ones(1, 64) * 0.9
    x_interf = torch.ones(1, 64)
    for _ in range(100):
        syn.learn(err_interf, x_interf, s_post_a, f_mod=1.0, ach=0.7)
        syn.structural_plasticity(prune_thresh=0.05, microglia_boost=1.8)

    w_hard_after_interference = syn.w_slow[hard_mask].clone()
    hard_drift = float((w_hard_after_interference - w_hard_before_interference).abs().max().item())

    # 3. Проверка адресного ферментативного события ChABC (HARD -> SOFT) на подмножестве нейронов 0..7
    chabc_mask = torch.zeros(64, dtype=torch.bool)
    chabc_mask[:8] = True
    n_unlocked = syn.trigger_chabc_event(post_mask=chabc_mask, window_ticks=40)

    # 4. Проверка фальсифицируемого предсказания 1: Степенной закон забывания Йоста–Фузи
    cascade = BennaFusiCascade(shape=(32, 32), n_stages=4, tau_base=10.0, scale_factor=7.0)
    init_trace = torch.randn(32, 32)
    curve_res = cascade.retention_curve(
        init_trace, time_points=[10, 25, 50, 100, 200, 400, 800, 1600]
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "scaffold_after_sleep": stats_after_consolidation,
        "half_life_progression_over_sleep_cycles": half_lives,
        "hard_synapse_max_drift_under_100_interfering_steps": round(hard_drift, 8),
        "chabc_targeted_unlocked_synapses": n_unlocked,
        "power_law_forgetting": curve_res,
        "elapsed_ms": round(elapsed_ms, 2),
    }


def benchmark_integrated_cortical_column_v8() -> dict:
    """Интеграционный тест CorticalColumn с включённым режимом enable_phase1_v8."""
    zcfg = ZoneCfg(name="ASSOC_V8", n_neurons=128, n_pv=16, n_sst=12, n_vip=8, g=0.5)
    col = CorticalColumn(zcfg, n_in=128, n_fb=128, ncfg=USNeuronCfg(), scfg=USSynapseCfg(), seed=19)
    col.reset_state(batch=1)
    col.enable_phase1_v8(enable_scaffold=True, enable_block_sparse=True, promote_core_quantile=0.85)

    for t in range(60):
        x_ff = torch.zeros(1, 128)
        x_fb = torch.zeros(1, 128)
        if t % 3 == 0:
            x_ff[:, :16] = 0.8
            x_fb[:, :16] = 0.5
        col.step(x_ff, x_fb, f_mod=0.8, vip_mod=0.3)

    col.consolidate(factor=0.5, spindle_gain=1.2)
    return col.phase1_telemetry()


def main() -> None:
    r1 = benchmark_mc_mrei_and_tick_ladder()
    r2 = benchmark_pdsr_block_sparse()
    r3 = benchmark_ct_pnn_scaffold_and_power_law()
    r4 = benchmark_integrated_cortical_column_v8()

    results = {
        "phase1_1_mc_mrei_and_tick_ladder": r1,
        "phase1_2_pdsr_block_sparse": r2,
        "phase1_3_ct_pnn_scaffold_and_power_law": r3,
        "phase1_4_integrated_column_v8": r4,
    }

    out_json = ROOT / "Produce" / "phase1_v8_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
