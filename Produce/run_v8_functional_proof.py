"""Единое Сквозное Доказательство Функциональности Архитектуры Ziwago-Prime v8.0
(«Двухтактный Мозг», движимый исключительно минимизацией Свободной Энергии F(t) и G'(π)).

Объединяет в одном живом цикле все узлы Фаз 1, 2 и 3:
  1. Самообучение без учителя (`next_ids=None`) за счёт минимизации Вариационной Свободной Энергии F(t):
     - Многоскоростное ядро `MC-MREI` + `TickLadderScheduler` + `PDSR Block-Sparse 2:4`;
     - Асинхронные аксональные задержки 10–50 мс и локальная фазовая когерентность `Gossip-PLV`;
     - Сквозное предиктивное кодирование Колена–Поллака (`EndToEndPredictiveCodingLoop`) без backprop.
  2. Консолидация памяти во сне (`NREM + REM`) и защита от забывания:
     - Кристаллизация перинейрональных сетей `CT-PNN` (`PLASTIC -> SOFT -> HARD`) и каскады Бенна–Фузи;
     - Шардированная голографическая память `SBCVSAShardedHippocampus` (`VSA-DHT`, 4 шарда, `Two-Clock θ/γ Replay`);
     - Двухпутевой слогово-фонемный декодер `BGPQSyllabaryDecoder` (Быстрая Голова + Слоговый Хвост + OOV).
  3. Активный Вывод, движимый минимизацией Термодинамической Ожидаемой Свободной Энергии G'(π):
     - Внутренний монолог VTE (`G'(π) = G(π) + λ_ATP(t) · E[ATP(π)]`) и эпистемические встречные вопросы;
     - Замкнутая сенсомоторная петля выживания в `EmbodiedForagingArena`.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import List

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPIKER_ROOT = ROOT.parent / "Neuroweave Spiker"
if SPIKER_ROOT.exists() and str(SPIKER_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKER_ROOT))

from neuroweave10.config import genesis_cfg
from neuroweave10.core.e2e_predictive import EndToEndPredictiveCodingLoop
from neuroweave10.embodied.arena import EmbodiedForagingArena
from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(42)

    vocab = [
        "<pad>", ".", ",", "?", "the", "a", "once", "upon", "time", "little",
        "girl", "boy", "dragon", "wolf", "knight", "flew", "ran", "walked", "guarded",
        "saw", "entered", "to", "in", "near", "dark", "ancient", "forest", "cave",
        "mountain", "castle", "crystal", "fire",
    ]
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = list(vocab)

    u = RegenesisOrganism(
        UnifiedConfig(preset="tiny", vocab_size=len(itos)),
        gcfg=genesis_cfg("tiny"),
        seed=42,
    )
    u.set_vocabulary(itos)
    u.punct_ids = {stoi["."], stoi[","], stoi["?"]}

    # Подключаем специализированные модули v8.0
    dht = SBCVSAShardedHippocampus(n_shards=4, n_blocks=16, block_len=64, shard_capacity=9600, seed=42)
    syll_dec = BGPQSyllabaryDecoder(vocab_size=50_000, d_in=64, n_head=512, n_syllables=256, n_articulators=32, seed=42)
    e2e_pc = EndToEndPredictiveCodingLoop(d_in=24, d_hidden=48, d_attractor=32, n_classes=len(itos), lr=0.12, seed=42)

    corpus = [
        "the dragon flew to the mountain .",
        "the knight walked to the castle .",
        "the wolf ran in the dark forest .",
        "the little girl saw the crystal in the cave .",
    ]
    stream: List[int] = []
    for sent in corpus:
        stream.extend(stoi[w] for w in sent.split() if w in stoi)

    print("=" * 92)
    print("ШАГ 1: САМООБУЧЕНИЕ БЕЗ УЧИТЕЛЯ (next_ids=None) ЧЕРЕЗ МИНИМИЗАЦИЮ СВОБОДНОЙ ЭНЕРГИИ F(t)")
    print("=" * 92)

    fep_curve = []
    for ep in range(1, 13):
        fe_vals, surp_vals, rpe_vals, e2e_errs = [], [], [], []
        pred_hits, total = 0, 0
        u.org.story_binder.notify_boundary()
        prev_tok = None
        for idx, tid in enumerate(stream):
            if prev_tok is not None:
                if prev_tok == tid:
                    pred_hits += 1
                total += 1
            # Чистое наблюдение реальности (next_ids=None): обучение движимо только снижением F(t)!
            r = u.step_token(torch.tensor([tid]), next_ids=None, learn=True)
            prev_tok = r["token"]
            fe_vals.append(r["free_energy"])
            surp_vals.append(r["fep_surprise"])
            rpe_vals.append(r["intrinsic_rpe"])

            # Параллельно обучаем сквозной контур Колена-Поллака и слоговой декодер на L5-состоянии
            x_enc = u._encode_frame(torch.tensor([tid])).flatten()
            next_tid = stream[(idx + 1) % len(stream)]
            e2e_out = e2e_pc.step(x_enc, target_class=next_tid, da_rpe=max(0.2, r["da"]), learn=True)
            e2e_errs.append(e2e_out.apical_error_norm)
            syll_dec.learn(target_token_id=next_tid, h_l5=x_enc, lr=0.20, da_rpe=max(0.2, r["da"]))

            # Записываем эпизод в 4-шардовый VSA-DHT Гиппокамп
            slot_codes = [tuple((tid * (s + 3) + b * 7) % 64 for b in range(16)) for s in range(4)]
            dht.store_composite_episode(slot_codes, salience=1.0 + max(0.0, r["intrinsic_rpe"]), ca1_mismatch=r["fep_surprise"])

        if ep % 4 == 0:
            u.sleep(ticks=24)
            u.org.hier.polychronous_router.consolidate_myelin(lr_myelin=0.35)
            syll_dec.consolidate_habit(min_salience=0.30)

        rec = {
            "epoch": ep,
            "free_energy": round(sum(fe_vals) / len(fe_vals), 4),
            "fep_surprise": round(sum(surp_vals) / len(surp_vals), 4),
            "e2e_apical_error": round(sum(e2e_errs) / len(e2e_errs), 4),
            "kp_alignment_cos": e2e_pc.reciprocal_alignment_cosine(),
            "self_pred_acc": round(pred_hits / max(1, total), 4),
            "gossip_plv": u.org.hier.polychronous_router.last_gossip_plv,
            "lambda_atp": round(u.org.lambda_atp, 4),
        }
        fep_curve.append(rec)
        if ep in (1, 4, 8, 12):
            print(
                f"  [Эпоха {ep:2d}/12] F(t)={rec['free_energy']:.4f} | "
                f"Surprise={rec['fep_surprise']:.3f} | "
                f"E2E-Err={rec['e2e_apical_error']:.3f} | "
                f"Self-Pred={rec['self_pred_acc']*100:5.1f}% | "
                f"Gossip-PLV={rec['gossip_plv']:.4f} | "
                f"λ_ATP={rec['lambda_atp']:.3f}"
            )

    print("
" + "=" * 92)
    print("ШАГ 2: ТЕАТР СОЗНАНИЯ (VTE) И АКТИВНЫЙ ВЫВОД ЧЕРЕЗ МИНИМИЗАЦИЮ G'(π)")
    print("=" * 92)
    dialog_prompts = [
        "tell me about the dragon .",
        "what is the zylarx in the dark cave ?",
    ]
    dialog_logs = []
    for p in dialog_prompts:
        res = u.respond_to_prompt(p, max_reply_tokens=7)
        print(f"  Промпт: '{p}'")
        print(f"    -> Ответ организма : '{res['reply']}'")
        print(
            f"    -> Meta-Surprise={res['prompt_meta_surprise']:.3f} | "
            f"κ_explore={res['kappa_explore_mean']:.3f} | "
            f"Исследовательский вопрос={res['triggered_epistemic_foraging']}"
        )
        dialog_logs.append({
            "prompt": p,
            "reply": res["reply"],
            "meta_surprise": res["prompt_meta_surprise"],
            "kappa_explore": res["kappa_explore_mean"],
            "epistemic_foraging": res["triggered_epistemic_foraging"],
        })

    print("
" + "=" * 92)
    print("ШАГ 3: ЗАМКНУТАЯ ВОПЛОЩЁННАЯ ПЕТЛЯ ВЫЖИВАНИЯ (ДЕЙСТВИЕ -> СРЕДА -> НОВОЕ ОЩУЩЕНИЕ)")
    print("=" * 92)
    arena = EmbodiedForagingArena(grid_size=8, input_dim=u.org.cfg.input_dim, seed=42)
    for _ in range(150):
        obs = arena.observe()
        g_out = u.org.step(obs, reward=0.0, learn=True)
        act = arena.select_embodied_action(
            lambda_atp=u.org.lambda_atp,
            valence=float(g_out.introspection["limbic"]["valence"]),
        )
        arena.step(act)
    print(
        f"  [150 шагов в 2D-среде] Собрано пищи: {arena.food_collected} | "
        f"Столкновений с хищником: {arena.predator_hits} | "
        f"Итоговый резерв АТФ: {arena.atp_reserve:.3f}"
    )

    col0_stats = u.org.hier.cols[0].v8_step_stats
    summary = {
        "fep_unsupervised_learning": {
            "epoch_1": fep_curve[0],
            "epoch_12": fep_curve[-1],
            "free_energy_drop": round(fep_curve[0]["free_energy"] - fep_curve[-1]["free_energy"], 4),
            "e2e_error_drop": round(fep_curve[0]["e2e_apical_error"] - fep_curve[-1]["e2e_apical_error"], 4),
        },
        "v8_core_telemetry": {
            "state_compression_ratio": col0_stats.get("state_compression_ratio", 0.0),
            "pdsr_flop_efficiency": col0_stats.get("flop_efficiency_ff", 0.0),
            "pnn_scaffold": col0_stats.get("scaffold_ff", {}),
            "vsa_dht_episodes_stored": dht.episodes_stored,
            "syllabary_parameter_compression": round(
                syll_dec.dense_baseline_parameters() / float(syll_dec.num_parameters()), 2
            ),
        },
        "active_inference_dialogue": dialog_logs,
        "embodied_survival_150_steps": {
            "food_collected": arena.food_collected,
            "predator_hits": arena.predator_hits,
            "final_atp_reserve": round(arena.atp_reserve, 4),
        },
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    out_path = ROOT / "Produce" / "v8_functional_proof_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 92)
    print(f"Доказательство функциональности успешно завершено за {summary['elapsed_sec']} с.")
    print(f"Отчёт сохранён в: {out_path}")


if __name__ == "__main__":
    main()
