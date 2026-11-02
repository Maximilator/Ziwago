"""Комплексный экспериментальный бенчмарк Фазы 2 Глобального Плана Ziwago-Prime v8.0
(на основе научного исследования `Produce/snnscience.docx` §3.3, §3.4, §3.6, §3.9, §4.6).

Проверяет и сохраняет в `Produce/phase2_v8_results.json`:
  1. Изъян 6 -> `BGPQSyllabaryDecoder` (`neuroweave10/motor/syllabary_decoder.py`):
     - Сжатие параметров и операций на словаре V = 50,000 слов против плоского линейного слоя;
     - Сохранность декодирования выученных слов, дофаминовая миграция из Хвоста в Быструю Голову во сне;
     - Композиционная продуктивность (сборка незнакомых составных OOV-слов без переобучения);
     - Фальсифицируемое Предсказание 3 (`Spoonerism Psycholinguistic Test`): при повышении норадреналина
       возникают перестановки начальных слогов (Onset c_1), причём на редких словах (Tail) значительно чаще,
       чем на автоматизированных ядерных леммах (Head).
  2. Изъян 3 -> `SBCVSAShardedHippocampus` (`neuroweave10/memory/hippocampus.py`):
     - Расширение ёмкости памяти в `150.0×` (`38,400` слотов на 4 функциональных шардах против `256` плоских);
     - Точное обратимое связывание/развязывание `SBC-VSA` за `16` целочисленных сложений (без FFT);
     - `Recall@5` на масштабе `10,000` записанных эпизодов с адаптивным `Multi-Probe LSH` и ускорением против полного перебора;
     - Двухчасовой реплей (`Two-Clock θ/γ Replay`).
  3. Изъян 4 -> `AxonalDelayPolychronousRouter` (`ADR-PKR`, `neuroweave10/cortex/hierarchy.py`):
     - Физические аксональные задержки 10–50 мс и соблюдение Правила Четверти Периода f_max = 1000 / (4 * τ_ij);
     - Асинхронный `Gossip-PLV` без глобального барьера синхронизации и детекция полихронных групп Ижикевича;
     - Пластическая миелинизация задержек во сне (`Fields 2015`), повышающая фазовый резонанс.
  4. Изъян 9 -> Термодинамический Активный Вывод `G'(π) = G(π) + λ_ATP(t) · E[ATP(π)]` и метаболический гейт LTP:
     - Переключение выбора в Театре Сознания (VTE) с энергозатратной политики на экономную при дефиците АТФ (высокой λ_ATP);
     - Метаболическая фильтрация слабых LTP-обновлений при росте теневой цены энергии λ_ATP.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPIKER_ROOT = ROOT.parent / "Neuroweave Spiker"
if SPIKER_ROOT.exists() and str(SPIKER_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKER_ROOT))

from neuroweave10.config import USSynapseCfg, genesis_cfg
from neuroweave10.core.synapse import GenesisSynapse
from neuroweave10.cortex.hierarchy import CorticalHierarchy
from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig


def benchmark_1_syllabary_motor_and_spoonerisms() -> dict:
    dec = BGPQSyllabaryDecoder(
        vocab_size=50_000,
        d_in=512,
        n_head=512,
        n_syllables=256,
        n_articulators=32,
        seed=42,
    )
    dense_params = dec.dense_baseline_parameters()
    syll_params = dec.num_parameters()
    param_compression = dense_params / float(syll_params)
    ops_compression = float(50_000 * 512) / float(dec.ops_per_step())

    # Обучаем 50 слов из Хвоста (Tail) и проверяем точность их слогового декодирования
    g = torch.Generator().manual_seed(77)
    target_words = [1000 + i * 731 for i in range(50)]
    contexts = [torch.randn(512, generator=g) for _ in range(50)]
    for _ in range(12):
        for wid, ctx in zip(target_words, contexts):
            dec.learn(wid, ctx, lr=0.22, da_rpe=0.85)

    correct_tail = 0
    for wid, ctx in zip(target_words, contexts):
        out = dec.decode(ctx, da_level=0.8, force_tail=True)
        if out.token_id == wid:
            correct_tail += 1
    tail_recall = correct_tail / float(len(target_words))

    # Проверяем сонную дофаминовую миграцию частотного слова из Хвоста в Быструю Голову
    promoted_tid = dec.consolidate_habit(min_salience=0.35)
    oov_res = dec.compose_oov_word(target_words[0], target_words[1], label="neuro_weave_oov")

    # Фальсифицируемое Предсказание 3: Психолингвистический Тест Спунеризмов (Head vs Tail при росте NE)
    sp_head_high_ne = dec.simulate_spoonerism(12, 45, ne_stress=0.90, n_trials=400, seed=11)
    sp_tail_high_ne = dec.simulate_spoonerism(18400, 39200, ne_stress=0.90, n_trials=400, seed=11)
    sp_tail_low_ne = dec.simulate_spoonerism(18400, 39200, ne_stress=0.25, n_trials=400, seed=11)

    return {
        "vocab_size": 50_000,
        "d_in": 512,
        "dense_linear_parameters": dense_params,
        "syllabary_parameters": syll_params,
        "parameter_compression_ratio": round(param_compression, 2),
        "ops_compression_ratio": round(ops_compression, 2),
        "learned_tail_accuracy": round(tail_recall, 4),
        "promoted_habit_token_id": promoted_tid,
        "oov_composition": oov_res,
        "spoonerism_head_high_ne_rate": sp_head_high_ne["spoonerism_rate"],
        "spoonerism_tail_high_ne_rate": sp_tail_high_ne["spoonerism_rate"],
        "spoonerism_tail_low_ne_rate": sp_tail_low_ne["spoonerism_rate"],
        "prediction_3_verified": bool(
            sp_tail_high_ne["spoonerism_rate"] > sp_head_high_ne["spoonerism_rate"]
            and sp_tail_high_ne["spoonerism_rate"] > sp_tail_low_ne["spoonerism_rate"]
        ),
    }


def benchmark_2_vsa_dht_hippocampus() -> dict:
    dht = SBCVSAShardedHippocampus(
        n_shards=4,
        n_blocks=16,
        block_len=64,
        shard_capacity=9600,
        n_lsh_buckets=256,
        seed=42,
    )
    g = torch.Generator().manual_seed(99)
    n_episodes = 10_000
    stored_queries = []
    t0 = time.perf_counter()
    for ep in range(n_episodes):
        slots = [
            tuple(int(x) for x in torch.randint(0, 64, (16,), generator=g).tolist())
            for _ in range(4)
        ]
        ep_id = dht.store_composite_episode(slots, salience=1.0, ca1_mismatch=0.2)
        if ep < 200:
            stored_queries.append((ep_id, slots))
    store_ms = (time.perf_counter() - t0) * 1000.0

    # Проверяем точное извлечение и устойчивость к 12.5% блочного шума (2 повреждённых блока из 16)
    hits_exact = 0
    hits_noisy = 0
    speedups = []
    for ep_id, slots in stored_queries:
        res_ex = dht.retrieve_multiprobe(slots, ca1_mismatch=0.10, top_k=5)
        if ep_id in res_ex["top_episode_ids"]:
            hits_exact += 1
        speedups.append(res_ex["speedup_vs_flat_scan"])

        # Вносим шум в первые 2 блока каждого шарда и поднимаем ca1_mismatch=0.70 (включается 4-probe LSH!)
        noisy_slots = [
            ((s[0] + 5) % 64, (s[1] + 9) % 64) + s[2:]
            for s in slots
        ]
        res_ns = dht.retrieve_multiprobe(noisy_slots, ca1_mismatch=0.70, top_k=5)
        if ep_id in res_ns["top_episode_ids"]:
            hits_noisy += 1

    # Проверяем Two-Clock θ/γ реплей на 10 шагах
    tc_steps = [dht.two_clock_replay_step(0) for _ in range(10)]
    return {
        "n_shards": dht.n_shards,
        "total_distributed_capacity": dht.total_capacity(),
        "capacity_expansion_vs_flat256": round(dht.capacity_expansion_factor(256), 1),
        "episodes_stored": n_episodes,
        "store_10k_time_ms": round(store_ms, 2),
        "recall_at_5_exact_10k": round(hits_exact / float(len(stored_queries)), 4),
        "recall_at_5_noisy_multiprobe_10k": round(hits_noisy / float(len(stored_queries)), 4),
        "mean_lsh_speedup_vs_flat_scan": round(sum(speedups) / float(len(speedups)), 2),
        "theta_composite_syncs_in_10_gamma_ticks": dht.theta_bindings_completed,
        "last_two_clock_step": tc_steps[-1],
    }


def benchmark_3_polychrony_and_gossip_plv() -> dict:
    cfg = genesis_cfg("tiny")
    hier = CorticalHierarchy(cfg, batch=1, topology="small_world_hub", max_fanout=3)
    router = hier.polychronous_router
    delays_before = dict(router.edge_delays_ms)

    # Прогоняем 120 мс кортикальной динамики с периодическими сенсорными пачками
    g = torch.Generator().manual_seed(17)
    x_burst = torch.randn(1, cfg.input_dim, generator=g).abs()
    plv_trace = []
    for t in range(120):
        x_in = x_burst if (t % 12 < 4) else torch.zeros(1, cfg.input_dim)
        hout = hier.step(
            x_in,
            gw_broadcast=torch.ones(cfg.gw.d_gw) * 0.25,
            ho_drive={},
            f_mod=1.0,
            vip_per_zone={z.name: 0.2 for z in cfg.zones},
            learn=True,
            dt=1.0,
        )
        plv_trace.append(hout.gossip_plv)

    plv_pre_myelin = router.last_gossip_plv
    myelin_stats = router.consolidate_myelin(lr_myelin=0.45)
    # Ещё один шаг после пластической миелинизации во сне
    hout_post = hier.step(
        x_burst,
        gw_broadcast=torch.ones(cfg.gw.d_gw) * 0.25,
        ho_drive={},
        f_mod=1.0,
        vip_per_zone={z.name: 0.2 for z in cfg.zones},
        learn=False,
        dt=1.0,
    )
    return {
        "routing_edges_count": len(hier.routing_edges),
        "initial_edge_delays_ms": {f"{i}->{j}": v for (i, j), v in delays_before.items()},
        "quarter_period_f_max_hz": {f"{i}->{j}": v for (i, j), v in router.edge_f_max_hz.items()},
        "gossip_plv_pre_myelination": plv_pre_myelin,
        "gossip_plv_post_myelination": hout_post.gossip_plv,
        "polychronous_catalog_size": len(router.polychronous_registry),
        "myelination_sleep_stats": myelin_stats,
    }


def benchmark_4_thermodynamic_active_inference_and_ltp_gate() -> dict:
    rorg = RegenesisOrganism(UnifiedConfig(preset="tiny", vocab_size=32), seed=19)
    # 1. Сравниваем G'(π) при высоком АТФ (λ_ATP ≈ 0) и при дефиците АТФ (высокая λ_ATP)
    for col in rorg.org.hier.cols:
        col.pyr.energy.fill_(1.0)
    rorg.org.lambda_atp = 0.0
    vte_high_atp = rorg.simulate_vte([5, 9])

    for col in rorg.org.hier.cols:
        col.pyr.energy.fill_(0.45)
    rorg.org.lambda_atp = 1.45
    vte_low_atp = rorg.simulate_vte([5, 9])

    # 2. Проверяем метаболический гейт LTP на синапсе при λ_ATP = 0.0 против λ_ATP = 2.0
    syn_free = GenesisSynapse(32, 32, USSynapseCfg(density=0.5, eta=0.05), seed=3)
    syn_taxed = GenesisSynapse(32, 32, USSynapseCfg(density=0.5, eta=0.05), seed=3)
    err = torch.full((1, 32), 0.15)
    s_pre = torch.ones(1, 32)
    s_post = torch.ones(1, 32)
    w0 = syn_free.w_fast.clone()
    syn_free.learn(err, s_pre, s_post, f_mod=1.0, lambda_atp=0.0)
    syn_taxed.learn(err, s_pre, s_post, f_mod=1.0, lambda_atp=2.0)
    dw_free = float((syn_free.w_fast - w0).abs().mean().item())
    dw_taxed = float((syn_taxed.w_fast - w0).abs().mean().item())

    return {
        "vte_g_prime_cand5_at_full_atp": vte_high_atp["breakdown"][5]["G_prime"],
        "vte_g_prime_cand5_at_low_atp": vte_low_atp["breakdown"][5]["G_prime"],
        "vte_metabolic_penalty_at_low_atp": vte_low_atp["breakdown"][5]["metabolic_penalty"],
        "ltp_mean_dw_at_lambda_0": round(dw_free, 6),
        "ltp_mean_dw_at_lambda_2": round(dw_taxed, 6),
        "ltp_metabolic_suppression_ratio": round(dw_free / max(1e-9, dw_taxed), 3),
    }


def main() -> None:
    res = {
        "phase2_1_syllabary_motor_and_spoonerisms": benchmark_1_syllabary_motor_and_spoonerisms(),
        "phase2_2_vsa_dht_hippocampus": benchmark_2_vsa_dht_hippocampus(),
        "phase2_3_polychrony_and_gossip_plv": benchmark_3_polychrony_and_gossip_plv(),
        "phase2_4_thermodynamic_active_inference": benchmark_4_thermodynamic_active_inference_and_ltp_gate(),
    }
    out_path = ROOT / "Produce" / "phase2_v8_results.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
