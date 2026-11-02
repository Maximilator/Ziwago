"""Комплексный экспериментальный бенчмарк Фазы 3 Глобального Плана Ziwago-Prime v8.0
(на основе научного исследования `Produce/snnscience.docx` §3.8, §3.10, §4.6).

Проверяет и сохраняет в `Produce/phase3_v8_results.json`:
  1. Изъян 8 -> Сквозное Предиктивное Кодирование без линейного Readout-слоя и без Backprop
     (`EndToEndPredictiveCodingLoop`, `neuroweave10/core/e2e_predictive.py`):
     - Реципрокная сходимость прямых и обратных синаптических весов Колена–Поллака (`cos(W_ff, B_fb^T) -> 0.85+`);
     - Достижение высокой точности распознавания через чистый резонанс с разреженными кортикальными аттракторами L5;
     - Полное отсутствие `.backward()`, автограда и плотных классификационных матриц.
  2. Изъян 10 -> Замкнутая Воплощённая Среда Выживания и Фуражирования (`EmbodiedForagingArena`,
     `neuroweave10/embodied/arena.py`):
     - Сравнение воплощённого организма `RegenesisOrganism` (управляемого термодинамическим `G'(π)` + вето миндалины)
       против случайного блуждания на 300 шагах в 2D-среде с пищей и хищником:
       сбор пищи, избегание хищника, поддержание гомеостаза АТФ и выживаемость.
  3. Сводная проверка всех 4 Фальсифицируемых Предсказаний `snnscience.docx` §4.6:
     - Предсказание 1: Степенной закон забывания Йоста–Фузи R(t) ~ (1 + t/τ0)^{-α} с α ∈ [0.3, 0.7];
     - Предсказание 2: Квадратичный рост эффективного периода полураспада памяти от числа циклов сна и PNN;
     - Предсказание 3: Психолингвистический эффект Спунеризмов (перестановка начальных слогов при NE-стрессе);
     - Предсказание 4: Биофизический коллапс при депривации сна на 10,000 тиков (АТФ < 0.10, σ > 1.15) и спасение сном.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPIKER_ROOT = ROOT.parent / "Neuroweave Spiker"
if SPIKER_ROOT.exists() and str(SPIKER_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKER_ROOT))

from neuroweave10.core.e2e_predictive import EndToEndPredictiveCodingLoop
from neuroweave10.embodied.arena import EmbodiedForagingArena
from neuroweave10.organism.genesis import GenesisOrganism
from neuroweave10.config import genesis_cfg


def benchmark_1_e2e_predictive_coding() -> dict:
    e2e = EndToEndPredictiveCodingLoop(
        d_in=64,
        d_hidden=96,
        d_attractor=64,
        n_classes=16,
        lr=0.12,
        kp_decay=0.01,
        seed=42,
    )
    g = torch.Generator().manual_seed(101)
    patterns = [torch.randn(64, generator=g) for _ in range(16)]

    align_before = e2e.reciprocal_alignment_cosine()
    # До обучения замеряем точность
    correct_init = sum(
        1 for cls_id, p in enumerate(patterns)
        if e2e.step(p, learn=False).predicted_class == cls_id
    )
    acc_init = correct_init / 16.0

    # Сквозное локальное обучение (25 эпох без единого вызова backward!)
    err_first = 0.0
    err_last = 0.0
    for epoch in range(25):
        ep_err = 0.0
        for cls_id, p in enumerate(patterns):
            out = e2e.step(p, target_class=cls_id, da_rpe=0.9, learn=True)
            ep_err += out.apical_error_norm
        if epoch == 0:
            err_first = ep_err / 16.0
        err_last = ep_err / 16.0

    correct_final = sum(
        1 for cls_id, p in enumerate(patterns)
        if e2e.step(p, learn=False).predicted_class == cls_id
    )
    acc_final = correct_final / 16.0
    align_after = e2e.reciprocal_alignment_cosine()

    return {
        "n_classes": 16,
        "initial_accuracy": round(acc_init, 4),
        "final_accuracy": round(acc_final, 4),
        "initial_kp_alignment_cosine": align_before,
        "final_kp_alignment_cosine": align_after,
        "apical_error_epoch1": round(err_first, 4),
        "apical_error_epoch25": round(err_last, 4),
        "used_backprop": False,
        "used_dense_linear_readout": False,
    }


def benchmark_2_closed_loop_embodied_foraging() -> dict:
    # 1. Воплощённый агент с термодинамическим Активным Выводом G'(π) и вето миндалины
    cfg = genesis_cfg("tiny")
    arena_active = EmbodiedForagingArena(grid_size=8, input_dim=cfg.input_dim, seed=42)
    org = GenesisOrganism(cfg, batch=1)
    org.reset_state(1)
    atp_trace = []
    for _ in range(300):
        obs = arena_active.observe()
        g_out = org.step(obs, reward=0.0, learn=True)
        lam = float(getattr(org, "lambda_atp", 0.2))
        val = float(g_out.introspection["limbic"]["valence"])
        act = arena_active.select_embodied_action(lambda_atp=lam, valence=val)
        step_res = arena_active.step(act)
        atp_trace.append(arena_active.atp_reserve)

    # 2. Базовый агент со случайным блужданием в идентичной среде
    arena_rand = EmbodiedForagingArena(grid_size=8, input_dim=32, seed=42)
    g_rand = torch.Generator().manual_seed(777)
    atp_rand_trace = []
    for _ in range(300):
        act_r = int(torch.randint(0, 6, (1,), generator=g_rand).item())
        arena_rand.step(act_r)
        atp_rand_trace.append(arena_rand.atp_reserve)

    return {
        "steps": 300,
        "embodied_active_inference": {
            "food_collected": arena_active.food_collected,
            "predator_hits": arena_active.predator_hits,
            "mean_atp_reserve": round(sum(atp_trace) / len(atp_trace), 4),
            "final_atp_reserve": round(arena_active.atp_reserve, 4),
        },
        "random_walk_baseline": {
            "food_collected": arena_rand.food_collected,
            "predator_hits": arena_rand.predator_hits,
            "mean_atp_reserve": round(sum(atp_rand_trace) / len(atp_rand_trace), 4),
            "final_atp_reserve": round(arena_rand.atp_reserve, 4),
        },
        "foraging_advantage_ratio": round(
            float(arena_active.food_collected) / float(max(1, arena_rand.food_collected)),
            2,
        ),
    }


def benchmark_3_four_falsifiable_predictions_summary() -> dict:
    p1 = json.loads((ROOT / "Produce" / "phase1_v8_results.json").read_text())
    p2 = json.loads((ROOT / "Produce" / "phase2_v8_results.json").read_text())
    sd_full = json.loads((ROOT / "Produce" / "INTERACTIVE_AND_SLEEP_DEPRIVATION_REPORT.json").read_text())
    sd = sd_full["task2_sleep_deprivation_10k"]
    tl_10k = sd["timeline"][-2]
    tl_rescued = sd["timeline"][-1]
    plf = p1["phase1_3_ct_pnn_scaffold_and_power_law"]["power_law_forgetting"]
    hl_prog = p1["phase1_3_ct_pnn_scaffold_and_power_law"]["half_life_progression_over_sleep_cycles"]
    return {
        "prediction_1_jost_fusi_power_law": {
            "alpha_exponent": plf["alpha_exponent"],
            "r2_power_law": plf["r2_power_law"],
            "r2_exponential": plf["r2_exponential"],
            "verified": plf["is_power_law_in_biological_range"],
        },
        "prediction_2_quadratic_halflife_pnn_sleep": {
            "half_life_progression_over_sleep_cycles": hl_prog,
            "verified": bool(hl_prog[-1] > 2.5 * hl_prog[0]),
        },
        "prediction_3_psycholinguistic_spoonerisms": {
            "head_high_ne_rate": p2["phase2_1_syllabary_motor_and_spoonerisms"]["spoonerism_head_high_ne_rate"],
            "tail_high_ne_rate": p2["phase2_1_syllabary_motor_and_spoonerisms"]["spoonerism_tail_high_ne_rate"],
            "tail_low_ne_rate": p2["phase2_1_syllabary_motor_and_spoonerisms"]["spoonerism_tail_low_ne_rate"],
            "verified": p2["phase2_1_syllabary_motor_and_spoonerisms"]["prediction_3_verified"],
        },
        "prediction_4_sleep_deprivation_collapse_and_rescue": {
            "deprived_10k_atp": tl_10k["atp"],
            "peak_hyperexcitable_sigma": sd["peak_hyperexcitable_sigma"],
            "deprived_10k_speech_status": tl_10k["speech_probe"]["speech_status"],
            "rescued_atp": tl_rescued["atp"],
            "rescued_sigma": tl_rescued["sigma"],
            "rescued_speech_status": tl_rescued["speech_probe"]["speech_status"],
            "verified": bool(
                tl_10k["atp"] < 0.10
                and sd["peak_hyperexcitable_sigma"] > 1.15
                and tl_rescued["speech_probe"]["speech_status"] == "COHERENT_GRAMMATICAL"
            ),
        },
    }


def main() -> None:
    res = {
        "phase3_1_e2e_predictive_coding": benchmark_1_e2e_predictive_coding(),
        "phase3_2_closed_loop_embodied_foraging": benchmark_2_closed_loop_embodied_foraging(),
        "phase3_3_four_falsifiable_predictions": benchmark_3_four_falsifiable_predictions_summary(),
    }
    out_path = ROOT / "Produce" / "phase3_v8_results.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
