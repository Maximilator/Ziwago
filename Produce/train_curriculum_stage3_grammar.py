"""Этап 3 куррикулума (Produce/next.txt): Обучение Синтаксису и Ролям (Ziwago-Prime v2.5).

Фазы обучения (Developmental Syntax Curriculum):
  Фаза A (Базовая грамматика 2–4 слова: Subj+Verb, Det+Noun+Verb, Det+Adj+Noun):
    Короткие жёстко структурированные предложения ("the cat sat .", "lily ran .",
    "a big dog barked .") — формирование синтаксических ролей R_k в ThetaGammaStoryBinder
    и матрицы w_role / w_lat в MotorPool.
  Фаза B (Расширенные клаузы 5–7 слов: Subj+Verb+Prep+Obj):
    "the little girl went to the park .", "a happy dog played with a ball ."
    — связывание Субъекта (Role 0..2) с Предикатом (Role 3) и Объектом (Role 5)
    через PhaseVSA и таламический LTS-burst на границах предложений.
  Фаза C (Реальные короткие предложения из корпуса TinyStories):
    Перенос выученного синтаксического скелета на живые предложения из корпуса.

Контроль: генерация 200 токенов и автоматическая проверка доли валидных
грамматических пар (Subject+Verb, Det+Adj/Noun, Prep+Det/Obj).
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import (
    RegenesisOrganism, UnifiedConfig, _checkpoint, _seq_pairs)
from neuroweave9.data import build_vocab_and_sequences, load_stories, tokenize_text


# Канонические грамматические категории для оценки синтаксической связности
DETERMINERS = {"the", "a", "an", "one", "his", "her", "once"}
ADJECTIVES = {"big", "little", "small", "happy", "old", "good", "new", "nice", "red"}
NOUNS_SUBJ = {
    "cat", "dog", "girl", "boy", "lily", "tim", "tom", "mom", "dad",
    "bird", "frog", "bear", "park", "house", "tree", "ball", "day",
    "time", "sun", "friend", "toy", "car", "book"
}
PRONOUNS = {"he", "she", "it", "they", "there"}
VERBS = {
    "sat", "ran", "went", "saw", "played", "loved", "was", "had",
    "jumped", "walked", "smiled", "looked", "found", "wanted", "said",
    "liked", "helped", "lived", "sang", "barked"
}
PREPOSITIONS = {"to", "in", "on", "with", "at", "upon", "into", "under"}
CONJUNCTIONS = {"and", "but", "so", "then"}
PUNCT = {".", ",", "!"}


def build_grammar_curriculum() -> List[List[str]]:
    """Генератор жёстко структурированных предложений для Фаз A и B."""
    phase_a = [
        "the cat sat .",
        "the dog ran .",
        "lily ran .",
        "tim jumped .",
        "lily smiled .",
        "the bird sang .",
        "a big dog ran .",
        "a little cat sat .",
        "the happy girl smiled .",
        "the little boy jumped .",
        "a small frog jumped .",
        "the big bear walked .",
        "he ran .",
        "she smiled .",
        "it jumped .",
    ]
    phase_b = [
        "the little girl went to the park .",
        "the big dog played with a ball .",
        "a happy cat sat on the tree .",
        "lily saw a little bird in the park .",
        "tim found a big ball in the house .",
        "the little boy walked to the house .",
        "she went to the park and played .",
        "he saw a big dog and smiled .",
        "once upon a time there was a little girl .",
        "one day lily went to the park .",
        "the dog loved the little girl .",
        "tim played with his friend in the park .",
        "she found a nice book in the house .",
        "the cat jumped on the big tree .",
    ]
    return [tokenize_text(s) for s in phase_a], [tokenize_text(s) for s in phase_b]


def evaluate_syntax(words: List[str]) -> Dict[str, object]:
    """Оценка грамматической структуры сгенерированного текста (200 токенов)."""
    subj_verb_pairs: List[str] = []
    det_noun_adj_pairs: List[str] = []
    prep_phrase_pairs: List[str] = []
    valid_transitions = 0
    total_transitions = max(1, len(words) - 1)

    for a, b in zip(words, words[1:]):
        is_valid = False
        # 1. Subject (Noun/Pronoun) -> Verb
        if (a in NOUNS_SUBJ or a in PRONOUNS) and b in VERBS:
            subj_verb_pairs.append(f"{a} {b}")
            is_valid = True
        # 2. Det -> Noun or Det -> Adj or Adj -> Noun
        elif (a in DETERMINERS and (b in NOUNS_SUBJ or b in ADJECTIVES)) or (
                a in ADJECTIVES and b in NOUNS_SUBJ):
            det_noun_adj_pairs.append(f"{a} {b}")
            is_valid = True
        # 3. Verb -> Prep / Det / Adj / Obj / Punct
        elif a in VERBS and (b in PREPOSITIONS or b in DETERMINERS
                             or b in ADJECTIVES or b in NOUNS_SUBJ
                             or b in PUNCT):
            is_valid = True
        # 4. Prep -> Det / Noun / Adj
        elif a in PREPOSITIONS and (b in DETERMINERS or b in NOUNS_SUBJ
                                    or b in ADJECTIVES):
            prep_phrase_pairs.append(f"{a} {b}")
            is_valid = True
        # 5. Noun -> Punct / Conj / Prep
        elif a in NOUNS_SUBJ and (b in PUNCT or b in CONJUNCTIONS
                                  or b in PREPOSITIONS):
            is_valid = True
        # 6. Punct / Conj -> Det / Pronoun / Subj / "once" / "one"
        elif (a in PUNCT or a in CONJUNCTIONS) and (
                b in DETERMINERS or b in PRONOUNS or b in NOUNS_SUBJ
                or b in ADJECTIVES):
            is_valid = True
        # 7. Idiomatic "once upon"
        elif a == "once" and b == "upon":
            is_valid = True

        if is_valid:
            valid_transitions += 1

    return {
        "grammar_Transition_Accuracy": round(valid_transitions / total_transitions, 4),
        "subj_verb_count": len(subj_verb_pairs),
        "subj_verb_examples": subj_verb_pairs[:15],
        "det_noun_count": len(det_noun_adj_pairs),
        "det_noun_examples": det_noun_adj_pairs[:15],
        "prep_phrase_count": len(prep_phrase_pairs),
    }


def main() -> int:
    t0 = time.time()
    data_path = os.path.join(BASE_DIR, "data/10k-tinystories.parquet")
    stories = load_stories(data_path, max_stories=500)
    phase_a_sents, phase_b_sents = build_grammar_curriculum()

    # Формируем словарь 128 токенов, гарантированно включающий базовую грамматику + топ TinyStories
    itos, stoi, *_ = build_vocab_and_sequences(
        stories[:400], max_vocab_size=105, seq_len=32, batch_size=1)
    core_words = sorted(
        (DETERMINERS | ADJECTIVES | NOUNS_SUBJ | PRONOUNS
         | VERBS | PREPOSITIONS | CONJUNCTIONS | PUNCT | {"sang", "barked"})
    )
    for w in core_words:
        if w not in stoi and len(itos) < 128:
            stoi[w] = len(itos)
            itos.append(w)

    unk = stoi.get("<unk>", 1)
    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg("tiny"), seed=42)
    u.punct_ids = {stoi[p] for p in (".", "!", "?", ",") if p in stoi}

    print(f"[stage3-grammar] started: vocab={len(itos)} roles={u.org.story_binder.n_slots}",
          flush=True)

    # --- ФАЗА A: Короткие синтаксические предложения в едином потоке ---
    hits_a, tot_a = 0, 0
    for ep in range(20):
        stream_a: List[int] = []
        for idx in range(len(phase_a_sents)):
            s = phase_a_sents[(idx + ep) % len(phase_a_sents)]
            stream_a.extend(stoi[w] for w in s if w in stoi)
        u.org.story_binder.notify_boundary()
        for a, b in zip(stream_a, stream_a[1:]):
            r = u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
            if ep >= 12:
                hits_a += int(r["token"] == b)
                tot_a += 1
        if (ep + 1) % 10 == 0:
            u.sleep(ticks=15)
    print(f"[Phase A: Short SV/DAN Sentences] late_acc={hits_a/max(1,tot_a):.3f} sigma={u.org.sigma:.3f}",
          flush=True)

    # --- ФАЗА B: Полные клаузы (Det+Adj+Subj+Verb+Prep+Det+Obj) в едином потоке ---
    hits_b, tot_b = 0, 0
    combined_b = phase_b_sents + phase_a_sents[:8]
    for ep in range(25):
        stream_b: List[int] = []
        for idx in range(len(combined_b)):
            s = combined_b[(idx + ep) % len(combined_b)]
            stream_b.extend(stoi[w] for w in s if w in stoi)
        u.org.story_binder.notify_boundary()
        for a, b in zip(stream_b, stream_b[1:]):
            r = u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
            if ep >= 16:
                hits_b += int(r["token"] == b)
                tot_b += 1
        if (ep + 1) % 10 == 0:
            u.sleep(ticks=20)
    print(f"[Phase B: Full SVO Clauses] late_acc={hits_b/max(1,tot_b):.3f} sigma={u.org.sigma:.3f}",
          flush=True)

    # --- ФАЗА C: Полные (без пропусков слов и с явным глаголом!) предложения из TinyStories ---
    real_short_sents: List[List[int]] = []
    for st in stories[:500]:
        toks = tokenize_text(st)
        cur_words: List[str] = []
        for w in toks:
            cur_words.append(w)
            if w in {".", "!"}:
                has_sv = any(
                    (a in NOUNS_SUBJ or a in PRONOUNS) and b in VERBS
                    for a, b in zip(cur_words, cur_words[1:])
                )
                if (4 <= len(cur_words) <= 10
                        and all(cw in stoi for cw in cur_words)
                        and has_sv):
                    real_short_sents.append([stoi[cw] for cw in cur_words])
                cur_words = []
                if len(real_short_sents) >= 120:
                    break
        if len(real_short_sents) >= 120:
            break

    hits_c, tot_c = 0, 0
    stream_c: List[int] = []
    for idx, ids in enumerate(real_short_sents):
        stream_c.extend(ids)
        ref_sent = combined_b[idx % len(combined_b)]
        stream_c.extend(stoi[w] for w in ref_sent if w in stoi)
        ref_a = phase_a_sents[idx % len(phase_a_sents)]
        stream_c.extend(stoi[w] for w in ref_a if w in stoi)
    u.org.story_binder.notify_boundary()
    for a, b in zip(stream_c, stream_c[1:]):
        r = u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
        hits_c += int(r["token"] == b)
        tot_c += 1

    u.sleep(ticks=30)
    ckpt_path = os.path.join(os.path.dirname(__file__),
                             "checkpoints/curriculum_stage3_grammar.pt")
    _checkpoint(u, ckpt_path, itos)
    print(f"[Phase C: TinyStories + Syntax Interleaved] acc={hits_c/max(1,tot_c):.3f} sigma={u.org.sigma:.3f}",
          flush=True)

    # --- ГЕНЕРАЦИЯ 200 ТОКЕНОВ И АНАЛИЗ ГРАММАТИКИ ---
    u.org.story_binder.notify_boundary()
    seed_words = ["the", "little", "girl"]
    seed_ids = [stoi[w] for w in seed_words if w in stoi]
    gen = u.generate(n_tokens=200, seed_tokens=seed_ids)
    gen_words = [itos[i] if i < len(itos) else "<unk>" for i in gen["ids"]]
    syntax_report = evaluate_syntax(gen_words)

    out = {
        "elapsed_s": round(time.time() - t0, 1),
        "vocab_size": len(itos),
        "phase_a_acc": round(hits_a / max(1, tot_a), 4),
        "phase_b_acc": round(hits_b / max(1, tot_b), 4),
        "phase_c_acc": round(hits_c / max(1, tot_c), 4),
        "gen_metrics": {k: gen[k] for k in ("margin_mean", "margin_max",
                                            "tail_vocab", "total_vocab",
                                            "max_repeat_run", "stp_min",
                                            "sigma", "pathology")},
        "syntax": syntax_report,
        "generated_200_tokens": " ".join(gen_words),
    }
    print("STAGE3_GRAMMAR_JSON:" + json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
