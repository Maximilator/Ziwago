"""Эксперимент «Живой Мозг Ziwago-Prime v3»: Доказательство Функциональности.

Проверяет все 4 экспериментальных требования:
  1. Принцип Свободной Энергии (FEP, самообучение БЕЗ учителя `next_ids=None`):
     Организму подаётся чистый сенсорный поток текста. Отслеживаем падение
     вариационной свободной энергии F(t), падение FEP-сюрприза предсказания,
     выделение эндогенного дофамина (intrinsic_rpe > 0) и рост точности
     упреждающего угадывания следующего слова без единого внешнего лейбла!
  2. Эфаптическая связь и Ламинарный LFP:
     Логируем динамику LFP по слоям (L4, L23, L5, L6) во время чтения текста,
     трансламинарный диполь (L23 - L5/L6), эфаптический сдвиг мембраны (мВ)
     и когерентность гамма-пакетов.
  3. Био-MetaHead (BA10 без backprop):
     Показываем скачок эпистемической неопределённости, мета-сюрприза и новизны
     при переходе от хорошо знакомой фразы ("once upon a time") к совершенно
     новой фразе ("there was a girl").
  4. Свободная генерация 100 токенов (чистый WTA-моторный пул после FEP-самообучения):
     Выводим сгенерированный текст и метрики.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig
from neuroweave9.data import build_vocab_and_sequences, load_stories, tokenize_text


def main() -> int:
    t0 = time.time()
    torch.manual_seed(42)

    data_path = os.path.join(BASE_DIR, "data/10k-tinystories.parquet")
    stories = load_stories(data_path, max_stories=300)
    itos, stoi, *_ = build_vocab_and_sequences(
        stories[:250], max_vocab_size=96, seq_len=32, batch_size=1)

    # Гарантируем наличие ключевых слов эксперимента в словаре
    exp_words = [
        "once", "upon", "a", "time", ",", "there", "was", "girl", "little",
        "boy", "cat", "dog", "bird", "frog", "lily", "tim", "sat", "ran",
        "went", "played", "smiled", "jumped", "loved", "saw", "found",
        "to", "in", "on", "with", "the", "park", "house", "tree", "ball",
        "friend", "happy", "big", "small", "day", "one", "."
    ]
    for w in exp_words:
        if w not in stoi and len(itos) < 128:
            stoi[w] = len(itos)
            itos.append(w)

    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg("tiny"), seed=42)
    u.punct_ids = {stoi[p] for p in (".", "!", "?", ",") if p in stoi}

    print("=" * 88)
    print("ЭКСПЕРИМЕНТ 1: САМООБУЧЕНИЕ ПО ПРИНЦИПУ СВОБОДНОЙ ЭНЕРГИИ (БЕЗ УЧИТЕЛЯ, next_ids=None)")
    print("=" * 88, flush=True)

    # Поток живых предложений, который организм просто НАБЛЮДАЕТ (без next_ids!)
    sensory_corpus = [
        "once upon a time , there was a little girl .",
        "the little girl went to the park .",
        "she saw a big dog in the park .",
        "the big dog played with a ball .",
        "once upon a time , a little cat sat on the tree .",
        "the little boy ran to the house .",
        "tim played with his friend in the park .",
        "lily smiled and jumped .",
        "the happy bird sang in the tree .",
        "one day , lily went to the park and saw a little frog .",
    ]
    tokenized_stream: List[int] = []
    for s in sensory_corpus:
        tokenized_stream.extend(stoi[w] for w in tokenize_text(s) if w in stoi)

    fep_epochs_log = []
    for ep in range(1, 21):
        fe_list, surp_list, rpe_list = [], [], []
        self_pred_hits, total_steps = 0, 0
        u.org.story_binder.notify_boundary()
        prev_pred_tok = None
        for tid in tokenized_stream:
            if prev_pred_tok is not None:
                if prev_pred_tok == tid:
                    self_pred_hits += 1
                total_steps += 1
            # ВНИМАНИЕ: next_ids НЕ передаётся! Организм учится только из потока реальности!
            r = u.step_token(torch.tensor([tid]), next_ids=None, learn=True)
            prev_pred_tok = r["token"]
            fe_list.append(r["free_energy"])
            surp_list.append(r["fep_surprise"])
            rpe_list.append(r["intrinsic_rpe"])
        if ep % 5 == 0:
            u.sleep(ticks=15)
        rec = {
            "epoch": ep,
            "free_energy_mean": round(sum(fe_list) / len(fe_list), 4),
            "fep_surprise_mean": round(sum(surp_list) / len(surp_list), 4),
            "intrinsic_rpe_pos_ticks": sum(1 for x in rpe_list if x > 0),
            "da_level": round(float(u.org.chem.s.da), 4),
            "self_prediction_acc": round(self_pred_hits / max(1, total_steps), 4),
        }
        fep_epochs_log.append(rec)
        if ep in (1, 5, 10, 15, 20):
            print(f"  [FEP Epoch {ep:2d}/20] F(t)={rec['free_energy_mean']:.4f} | "
                  f"Surprise={rec['fep_surprise_mean']:.3f} | "
                  f"Self-Pred Acc={rec['self_prediction_acc']*100:5.1f}% | "
                  f"Endogenous DA pulses={rec['intrinsic_rpe_pos_ticks']:2d} | "
                  f"[DA]={rec['da_level']:.3f}", flush=True)

    print("=" * 88)
    print("ЭКСПЕРИМЕНТ 2: ЭФАПТИЧЕСКАЯ СВЯЗЬ И ЛАМИНАРНЫЙ LFP (L4, L23, L5, L6) + ГАММА-ПАКЕТЫ")
    print("=" * 88, flush=True)

    probe_sentence = ["once", "upon", "a", "time", ",", "there", "was", "a", "little", "girl", "."]
    lfp_trace_log = []
    u.org.story_binder.notify_boundary()
    for step_idx, word in enumerate(probe_sentence):
        tid = stoi[word]
        r = u.step_token(torch.tensor([tid]), next_ids=None, learn=False)
        sens_lfp = r["lfp"].get("SENS", {})
        pfc_lfp = r["lfp"].get("PFC", {})
        col_sens = u.org.hier.cols[0]
        eph_max = float(col_sens.lfp.ephaptic_mv.abs().max().item())
        dipole_sens = sens_lfp.get("L23", 0.0) - 0.5 * (sens_lfp.get("L5", 0.0) + sens_lfp.get("L6", 0.0))
        gamma_slot = u.org.story_binder.clause_pos % u.org.story_binder.n_slots
        entry = {
            "word": word,
            "gamma_slot": gamma_slot,
            "SENS_L4": round(sens_lfp.get("L4", 0.0), 3),
            "SENS_L23": round(sens_lfp.get("L23", 0.0), 3),
            "SENS_L5": round(sens_lfp.get("L5", 0.0), 3),
            "SENS_L6": round(sens_lfp.get("L6", 0.0), 3),
            "PFC_L23": round(pfc_lfp.get("L23", 0.0), 3),
            "PFC_L5": round(pfc_lfp.get("L5", 0.0), 3),
            "trans_dipole": round(dipole_sens, 3),
            "ephaptic_shift_mV": round(eph_max, 3),
        }
        lfp_trace_log.append(entry)
        print(f"  Token [{step_idx:2d}] '{word:<6}' (γ-slot {gamma_slot}) | "
              f"SENS LFP [L4={entry['SENS_L4']:+6.2f}, L23={entry['SENS_L23']:+6.2f}, "
              f"L5={entry['SENS_L5']:+6.2f}, L6={entry['SENS_L6']:+6.2f}] | "
              f"Dipole={entry['trans_dipole']:+6.2f} | Ephaptic={entry['ephaptic_shift_mV']:.3f} mV",
              flush=True)

    print("=" * 88)
    print("ЭКСПЕРИМЕНТ 3: БИО-METAHEAD (ПЕРЕХОД ОТ ЗНАКОМОЙ ФРАЗЫ К НОВОЙ)")
    print("  Знакомая фраза (25x повторов): 'once upon a time'")
    print("  Новая незнакомая фраза:        'there was a girl'")
    print("=" * 88, flush=True)

    # Создаём свежий экземпляр для чистого теста привыкания BA10 (habituation vs novelty)
    u_meta = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                               gcfg=genesis_cfg("tiny"), seed=77)
    familiar_words = ["once", "upon", "a", "time"]
    novel_words = ["there", "was", "a", "girl"]
    fam_ids = [stoi[w] for w in familiar_words]

    # Многократно проживаем знакомую фразу "once upon a time"
    for _ in range(25):
        u_meta.org.story_binder.notify_boundary()
        for tid in fam_ids:
            u_meta.step_token(torch.tensor([tid]), next_ids=None, learn=True)

    # Теперь идём по переходу: "once upon a time" -> "there was a girl"
    meta_transition_log = []
    u_meta.org.story_binder.notify_boundary()
    for phase_label, w_list in [("FAMILIAR", familiar_words), ("NOVEL", novel_words)]:
        for w in w_list:
            tid = stoi[w]
            r = u_meta.step_token(torch.tensor([tid]), next_ids=None, learn=True)
            m_entry = {
                "phase": phase_label,
                "word": w,
                "epistemic": round(r["epistemic"], 5),
                "novelty": round(r["novelty"], 4),
                "meta_surprise": round(r["meta_surprise"], 4),
                "fep_surprise": round(r["fep_surprise"], 4),
                "confidence": round(r["confidence"], 4),
            }
            meta_transition_log.append(m_entry)
            print(f"  [{phase_label:<8}] '{w:<6}' | "
                  f"Epistemic={m_entry['epistemic']:.5f} | "
                  f"Novelty={m_entry['novelty']:.4f} | "
                  f"Meta-Surprise={m_entry['meta_surprise']:.4f} | "
                  f"FEP-Surprise={m_entry['fep_surprise']:.3f} | "
                  f"Conf={m_entry['confidence']:.3f}", flush=True)

    print("=" * 88)
    print("ЭКСПЕРИМЕНТ 4: СВОБОДНАЯ WTA-ГЕНЕРАЦИЯ 100 ТОКЕНОВ (ПОСЛЕ ЧИСТОГО FEP-САМООБУЧЕНИЯ)")
    print("=" * 88, flush=True)

    u.org.story_binder.notify_boundary()
    seed_ids = [stoi[w] for w in ["once", "upon", "a", "time"]]
    gen = u.generate(n_tokens=100, seed_tokens=seed_ids, learn=False)
    gen_words = [itos[i] if i < len(itos) else "<unk>" for i in gen["ids"]]
    gen_text = " ".join(gen_words)
    print(f"  Метрики WTA-пула: max_repeat_run={gen['max_repeat_run']} | "
          f"tail_vocab={gen['tail_vocab']} | margin_mean={gen['margin_mean']:.2f} | "
          f"sigma={gen['sigma']:.3f} | pathology={gen['pathology']}")
    print("  Сгенерированный текст (100 токенов):")
    print(f"  {gen_text}", flush=True)

    summary = {
        "elapsed_s": round(time.time() - t0, 1),
        "fep_first_epoch": fep_epochs_log[0],
        "fep_last_epoch": fep_epochs_log[-1],
        "lfp_sample": lfp_trace_log,
        "meta_transition": meta_transition_log,
        "generation_metrics": {k: gen[k] for k in ("margin_mean", "margin_max",
                                                   "tail_vocab", "total_vocab",
                                                   "max_repeat_run", "stp_min",
                                                   "sigma", "pathology")},
        "generated_100_tokens": gen_text,
    }
    print("LIVING_BRAIN_JSON:" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
