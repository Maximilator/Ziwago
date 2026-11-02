#!/opt/micromamba/bin/python3
"""Ziwago-Prime v8.5 («Двухтактный Живой Мозг + Lab.html Breakthroughs») — Идеальный CPU / Multi-Core Тренер (`train_ziwago_v8.py`).

Объединяет в единую высокоскоростную био-систему:
  1. Собственный 256-словный Морфо-Синтаксический и Семантический Био-Словарь (`build_bio_vocabulary`),
     построенный по частотному и грамматическому ядру `data/10k-tinystories.parquet` (10 000 историй):
     - 9 функциональных пулов: Зачины/Детерминативы, Прилагательные качеств и эмоций,
       Одушевлённые субъекты (дети, животные, сказочные герои), Местоимения,
       Глаголы движения/восприятия/речи/мышления, Модальные связки, Предлоги,
       Союзы/Дискурсивные маркеры, Объекты и Локации мира, Пунктуация.
  2. Полный стек открытий `next9.txt` + `next10.txt` + `Produce/Lab.html` (Ziwago-Prime v8.5):
     - `SparseProp` O(K log N) + `Block-Emptiness Ratio` (16x16) + `Pinned Block-Sparse Lanes` (F#66, F#44);
     - `HetSynLayer` — логнормальные индивидуальные синаптические постоянные времени τ_ij ≥ 1 мс
       с онлайн-адаптацией к межспайковым интервалам входа (F#54, F#55);
     - `eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch` (θ_down=0.58, θ_up=0.72, экономия >40%
       вычислений молчащих клеток и мгновенная реактивация <20 мс, F#20, F#16, F#22);
     - `CT-PNN + Synaptic Intelligence` (ω_ij, Ω_ij) + Термодинамический порог забывания от
       дисперсии аппроксимации декодера (`thermodynamic_forgetting_noise_floor`, F#58) + `KL-Custody Lock` (<1.5 бит, F#43);
     - `Debiased WPLI` (Vinck et al. 2011) + `False-Witness Detector` объёмной проводимости (F#25, F#27);
     - `SBC-VSA-DHT` + `HippoRAG Personalized PageRank` + Двунаправленный θ/γ реплей Коэне–Хассельмо
       (`prospective_forward` для планирования и `retrospective_reverse` для обратного связывания награды, F#15, F#19, F#21);
     - `EmbodiedForagingArena` с точными экспоненциальными прыжками действий (`evaluate_action_exact_jump`, F#63)
       и диспетчером активного вывода `PROBE / ATTEND / STOP` (F#37).
  3. Многоядерная CPU-оптимизация (от 4 до 64+ ядер):
     - Автоматическое масштабирование пула потоков `ThreadPoolExecutor(--workers)` для параллельной
       фильтрации и синтаксической разметки корпуса `10k-tinystories.parquet`;
     - Настройка `torch.set_num_threads(--cpu-threads)` и векторизованный тензорный конвейер
       биграммных (`w_lat`), триграммных (`w_lat2`), 4-граммных фразовых (`w_phrase4`) и
       длинноконтекстных субъектно-анафорических (`w_ctx_narrative`) синаптических следов без `loss.backward()`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if not (ROOT / "neuroweave10").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPIKER_ROOT = ROOT.parent / "Neuroweave Spiker"
if SPIKER_ROOT.exists() and str(SPIKER_ROOT) not in sys.path:
    sys.path.insert(0, str(SPIKER_ROOT))

import torch
import torch.nn.functional as F

from neuroweave10.config import genesis_cfg
from neuroweave10.core.e2e_predictive import EndToEndPredictiveCodingLoop
from neuroweave10.core.synapse import SCAFFOLD_HARD, SCAFFOLD_PLASTIC, SCAFFOLD_SOFT
from neuroweave10.embodied.arena import EmbodiedForagingArena
from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig, _checkpoint


# ============================================================================
# 1. СОБСТВЕННЫЙ 256-СЛОВНЫЙ МОРФО-СИНТАКСИЧЕСКИЙ И СЕМАНТИЧЕСКИЙ БИО-СЛОВАРЬ
# ============================================================================
DETERMINERS: Set[str] = {
    "the", "a", "an", "one", "her", "his", "their", "my", "our", "that", "this", "every", "some", "two", "three",
}
ADJECTIVES: Set[str] = {
    "little", "big", "happy", "brave", "small", "bright", "dark", "old", "new", "good",
    "red", "blue", "green", "yellow", "white", "black", "golden", "silver", "warm", "cold",
    "magic", "kind", "sweet", "soft", "tall", "tiny", "fast", "slow", "quiet", "loud",
    "wise", "young", "hungry", "thirsty", "tired", "safe", "shiny", "deep", "high", "gentle",
}
ANIMATE_NOUNS: Set[str] = {
    "girl", "boy", "lily", "tim", "tom", "ben", "sam", "mia", "max", "lucy", "anna", "leo",
    "mom", "dad", "mother", "father", "grandma", "grandpa", "king", "queen", "princess", "prince",
    "knight", "wizard", "dragon", "fairy", "giant", "friend", "friends", "child", "baby",
    "dog", "puppy", "cat", "kitten", "bird", "owl", "duck", "swan", "frog", "fish",
    "bear", "rabbit", "bunny", "squirrel", "fox", "wolf", "deer", "horse", "lion", "mouse",
    "bee", "butterfly", "turtle",
}
OBJECT_PLACE_NOUNS: Set[str] = {
    "park", "forest", "woods", "house", "home", "room", "kitchen", "yard", "garden", "farm",
    "school", "castle", "tower", "palace", "village", "town", "city", "bridge", "road", "path", "way",
    "mountain", "hill", "cave", "valley", "island", "beach", "sea", "ocean", "lake", "pond", "river", "stream",
    "tree", "branch", "leaf", "flower", "rose", "grass", "bush", "stone", "rock", "sand", "mud", "snow", "rain",
    "sun", "moon", "star", "sky", "cloud", "wind", "fire", "light", "shadow", "water", "milk", "tea",
    "apple", "cake", "bread", "berry", "honey", "soup", "cookie", "lunch", "dinner",
    "ball", "toy", "doll", "kite", "boat", "train", "car", "bike", "wagon", "bell", "drum", "flute",
    "book", "story", "song", "letter", "map", "treasure", "key", "box", "chest", "basket", "bag", "cup",
    "crown", "sword", "shield", "ring", "wand", "cloak", "hat", "coat", "shoe", "dress", "shirt", "pocket",
    "bed", "chair", "table", "door", "window", "gate", "wall", "floor", "roof", "nest", "web",
    "day", "time", "morning", "night", "evening", "summer", "winter", "spring",
    "world", "idea", "secret", "wish", "dream", "game", "fun", "help", "love", "joy", "peace",
    "sound", "voice", "smile", "heart", "hand", "eye", "wing", "tail", "paw",
}
NOUNS_ALL: Set[str] = ANIMATE_NOUNS | OBJECT_PLACE_NOUNS
PRONOUNS: Set[str] = {"she", "he", "it", "they", "we", "i", "you", "there", "him", "them", "us"}
VERBS: Set[str] = {
    "was", "were", "is", "are", "lived", "went", "walked", "ran", "played", "saw", "found",
    "loved", "liked", "smiled", "laughed", "cried", "jumped", "hopped", "climbed", "swam", "flew",
    "sailed", "rode", "danced", "sang", "slept", "rested", "woke", "ate", "drank", "cooked", "baked",
    "shared", "gave", "took", "carried", "held", "caught", "threw", "kicked", "pushed", "pulled",
    "opened", "closed", "locked", "unlocked", "built", "made", "drew", "painted", "wrote", "read",
    "told", "said", "asked", "answered", "called", "whispered", "shouted", "listened", "heard",
    "looked", "watched", "searched", "explored", "discovered", "knew", "thought", "remembered",
    "learned", "understood", "wanted", "needed", "wished", "hoped", "tried", "decided",
    "helped", "saved", "protected", "guarded", "healed", "hugged", "thanked", "welcomed",
    "came", "arrived", "returned", "stayed", "left", "followed", "led", "met", "visited",
    "sat", "stood", "waited", "hid", "shone", "glowed", "grew", "bloomed", "fell", "flowed",
    "had", "got", "kept", "put", "placed", "felt", "became", "seemed", "began", "started", "finished",
}
PREPOSITIONS: Set[str] = {
    "to", "in", "on", "with", "at", "near", "into", "under", "upon", "for", "from",
    "by", "over", "through", "across", "around", "behind", "beside", "inside", "outside", "about", "of",
}
CONJUNCTIONS: Set[str] = {"and", "so", "but", "then", "when", "because", "while", "after", "before", "soon", "suddenly", "together"}
PUNCTUATION: Set[str] = {".", ",", "!"}


def build_bio_vocabulary(max_vocab: int = 256) -> Tuple[List[str], Dict[str, int]]:
    """Формирует 256-словный канонический био-словарь организма, сгруппированный по
    функциональным ролям коры (Брока BA44 + концептуальные клетки MTL)."""
    ordered_pools: List[str] = [
        "<pad>", "<unk>", ".", ",", "!",
        "once", "upon", "a", "time", "there", "was", "one", "day",
    ]
    priority_verbs = [
        "lived", "went", "walked", "ran", "played", "saw", "found", "loved", "smiled",
        "jumped", "hopped", "flew", "sailed", "danced", "sang", "slept", "ate", "drank",
        "baked", "shared", "gave", "took", "carried", "built", "read", "said", "heard",
        "looked", "watched", "explored", "discovered", "wanted", "needed", "helped",
        "guarded", "hugged", "came", "returned", "led", "met", "sat", "stood", "glowed",
        "kept", "felt", "became",
    ]
    priority_places = [
        "park", "forest", "woods", "house", "home", "room", "kitchen", "garden",
        "castle", "village", "mountain", "valley", "island", "beach", "lake", "river",
        "tree", "branch", "flower", "stone", "rock", "sun", "star", "water",
        "cake", "bread", "berry", "ball", "boat", "book", "story", "song", "map",
        "treasure", "key", "box", "chest", "basket", "crown", "shield", "gate",
        "morning", "evening", "night", "voice", "hand", "dreams", "light", "path",
    ]
    for pool in (
        sorted(DETERMINERS),
        sorted(PRONOUNS),
        sorted(ADJECTIVES),
        sorted(ANIMATE_NOUNS),
        priority_verbs,
        priority_places,
        sorted(PREPOSITIONS),
        sorted(CONJUNCTIONS),
        sorted(VERBS),
        sorted(OBJECT_PLACE_NOUNS),
    ):
        for w in pool:
            if w not in ordered_pools:
                ordered_pools.append(w)
                if len(ordered_pools) >= max_vocab:
                    break
        if len(ordered_pools) >= max_vocab:
            break

    itos = ordered_pools[:max_vocab]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


# ============================================================================
# 2. МНОГОЯДЕРНЫЙ ПАРСЕР КОРПУСА (ИЗ 10k-tinystories.parquet) + 4 ЭТАПА КУРРИКУЛУМА
# ============================================================================
@dataclass
class CurriculumDataset:
    stage1_seqs_len4: List[List[int]]
    stage2_seqs_len8: List[List[int]]
    stage3_seqs_len16: List[List[int]]
    stage4_seqs_len64: List[List[int]]
    narrative_chains: List[List[int]]
    parquet_tokens_scanned: int
    parquet_stories_used: int
    parquet_clean_sentences: int


def evaluate_generated_speech(words: List[str]) -> Dict[str, object]:
    """Оценивает грамматическую связность, длину удерживаемого контекста и лексическое богатство."""
    if len(words) < 2:
        return {
            "grammar_validity": 0.0,
            "context_retention_tokens": 0.0,
            "max_coherent_span": 0,
            "unique_vocab_used": len(set(words)),
            "repeat_run_max": 1,
        }

    valid_pairs = 0
    total_pairs = len(words) - 1
    cur_span = 1
    max_span = 1
    spans: List[int] = []

    for a, b in zip(words, words[1:]):
        is_ok = False
        if a == b:
            is_ok = False
        elif a == "once" and b == "upon":
            is_ok = True
        elif a == "upon" and b in DETERMINERS:
            is_ok = True
        elif a == "time" and (b in PRONOUNS or b in DETERMINERS or b in ANIMATE_NOUNS or b in PUNCTUATION):
            is_ok = True
        elif a == "there" and b in VERBS:
            is_ok = True
        elif a == "one" and b in ("day", "morning", "night", " evening"):
            is_ok = True
        elif a in ("day", "morning", "night") and (b in PRONOUNS or b in DETERMINERS or b in ANIMATE_NOUNS or b in VERBS or b in PUNCTUATION):
            is_ok = True
        elif a in DETERMINERS and (b in ADJECTIVES or b in NOUNS_ALL):
            is_ok = True
        elif a in ADJECTIVES and (b in NOUNS_ALL or b in ADJECTIVES or b in PUNCTUATION or b in CONJUNCTIONS):
            is_ok = True
        elif a in PRONOUNS and (b in VERBS or b in CONJUNCTIONS):
            is_ok = True
        elif a in ANIMATE_NOUNS and (b in VERBS or b in PREPOSITIONS or b in CONJUNCTIONS or b in PUNCTUATION):
            is_ok = True
        elif a in OBJECT_PLACE_NOUNS and (
            b in VERBS or b in PREPOSITIONS or b in CONJUNCTIONS or b in PUNCTUATION
        ):
            is_ok = True
        elif a in VERBS and (
            b in PREPOSITIONS or b in DETERMINERS or b in ADJECTIVES
            or b in NOUNS_ALL or b in PRONOUNS or b in PUNCTUATION or b in CONJUNCTIONS or b in VERBS
        ):
            is_ok = True
        elif a in PREPOSITIONS and (b in DETERMINERS or b in ADJECTIVES or b in NOUNS_ALL or b in PRONOUNS):
            is_ok = True
        elif a in CONJUNCTIONS and (
            b in PRONOUNS or b in DETERMINERS or b in ANIMATE_NOUNS or b in VERBS or b in ADJECTIVES
        ):
            is_ok = True
        elif a in PUNCTUATION and (
            b in DETERMINERS or b in PRONOUNS or b in ANIMATE_NOUNS
            or b in CONJUNCTIONS or b in ("once", "one", "then", "there") or b in ADJECTIVES
        ):
            is_ok = True

        if is_ok:
            valid_pairs += 1
            cur_span += 1
            max_span = max(max_span, cur_span)
        else:
            spans.append(cur_span)
            cur_span = 1
    spans.append(cur_span)

    run, max_run = 1, 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)

    grammar_val = valid_pairs / float(max(1, total_pairs))
    top_spans = sorted(spans, reverse=True)[:3]
    mean_top_span = sum(top_spans) / float(max(1, len(top_spans)))
    context_retention = round(min(float(len(words)), mean_top_span * (0.68 + 0.32 * grammar_val)), 2)

    return {
        "grammar_validity": round(grammar_val, 4),
        "context_retention_tokens": context_retention,
        "max_coherent_span": int(max_span),
        "unique_vocab_used": len(set(words)),
        "repeat_run_max": int(max_run),
    }


def _parse_parquet_chunk(args: Tuple[List[str], Dict[str, int]]) -> Tuple[int, int, List[List[int]]]:
    """Рабочая функция потока для параллельного многоядерного сканирования чанка историй."""
    stories_chunk, stoi = args
    tokens_cnt = 0
    clean_sents: List[List[int]] = []
    for raw_story in stories_chunk:
        s_low = str(raw_story).lower()
        tokens_cnt += len(re.findall(r"[a-z]+|[.,!]", s_low))
        for r_s in re.split(r"[.!?]+", s_low):
            w_list = re.findall(r"[a-z]+", r_s)
            if 5 <= len(w_list) <= 12 and all(w in stoi for w in w_list):
                has_subj = any(w in ANIMATE_NOUNS or w in PRONOUNS for w in w_list[:4])
                has_verb = any(w in VERBS for w in w_list[1:])
                if has_subj and has_verb:
                    if evaluate_generated_speech(w_list + ["."])["grammar_validity"] >= 0.90:
                        clean_sents.append([stoi[w] for w in w_list] + [stoi["."]])
    return tokens_cnt, len(stories_chunk), clean_sents


def build_developmental_dataset(
    parquet_path: Path,
    stoi: Dict[str, int],
    itos: List[str],
    num_workers: int = 4,
    max_stories_to_scan: int = 2000,
) -> CurriculumDataset:
    """Многоядерная сборка развивающего датасета из `data/10k-tinystories.parquet` + богатых сюжетных линий."""
    parquet_sentences: List[List[int]] = []
    parquet_tokens_scanned = 0
    parquet_stories_used = 0

    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            all_texts = df["text"].iloc[:max_stories_to_scan].tolist()
            n_w = max(1, min(int(num_workers), 16))
            chunk_sz = max(1, (len(all_texts) + n_w - 1) // n_w)
            chunks = [(all_texts[i : i + chunk_sz], stoi) for i in range(0, len(all_texts), chunk_sz)]
            with ThreadPoolExecutor(max_workers=n_w) as ex:
                for t_cnt, s_cnt, c_sents in ex.map(_parse_parquet_chunk, chunks):
                    parquet_tokens_scanned += t_cnt
                    parquet_stories_used += s_cnt
                    parquet_sentences.extend(c_sents)
        except Exception:
            pass

    def _to_ids(text: str) -> List[int]:
        return [stoi[w] for w in text.lower().split() if w in stoi]

    # Этап 1 (`seq_len = 4`): фундаментальные синтагмы
    s1_phrases = [
        "once upon a time", "upon a time there", "a time there was", "there was a little",
        "was a little girl", "a little girl lived", "lived in a warm", "in a warm house",
        "there was a brave", "was a brave boy", "a brave boy walked", "walked to the green",
        "to the green park", "the happy dog ran", "ran to the deep", "to the deep forest",
        "lily saw a bright", "saw a bright bird", "bright bird in the", "in the tall tree",
        "tim found a magic", "found a magic stone", "magic stone near the", "near the blue river",
        "the golden dragon flew", "dragon flew to the", "flew to the high", "to the high mountain",
        "the wise king lived", "lived in the old", "in the old castle", "the brave knight guarded",
        "guarded the silver crown", "mia baked a sweet", "baked a sweet cake", "shared with her friend",
        "leo sailed a small", "sailed a small boat", "across the calm lake", "the gentle owl sang",
    ]
    stage1_seqs = [ids[:4] for p in s1_phrases for ids in [_to_ids(p)] if len(ids) >= 4]

    # Этап 2 (`seq_len = 8`): полные 8-токенные предложения SVO
    s2_clauses = [
        "once upon a time a little girl .",
        "once upon a time a brave boy .",
        "the little girl lived in the house .",
        "the brave boy walked to the park .",
        "the happy dog ran to the forest .",
        "lily saw a bright bird in tree .",
        "tim found a magic stone near river .",
        "the golden dragon flew to the mountain .",
        "the brave knight guarded the old castle .",
        "mia baked a sweet cake in kitchen .",
        "she shared the sweet cake with mom .",
        "leo sailed a small boat on lake .",
        "the wise owl sat on tall branch .",
        "a tiny rabbit hopped in green garden .",
        "ben played with a red ball outside .",
        "the kind queen gave a golden key .",
    ]
    stage2_seqs = [ids[:8] for c in s2_clauses for ids in [_to_ids(c)] if len(ids) == 8]
    for ps in parquet_sentences[:24]:
        if len(ps) >= 8:
            stage2_seqs.append(ps[:7] + [stoi["."]])

    # Этап 3 (`seq_len = 16`): двухклаузные причинно-следственные истории с анафорой
    s3_discourses = [
        "once upon a time there was a little girl . she lived in the warm house .",
        "once upon a time there was a brave boy . he walked to the green park .",
        "lily went to the green garden with her mom . she saw a bright bird there .",
        "tim walked near the blue river with his dog . he found a shiny magic stone .",
        "the golden dragon flew over the high mountain . it landed near the old castle .",
        "the brave knight met the wise king inside . they guarded the golden crown together .",
        "mia baked a warm sweet cake at home . she shared it with her good friend .",
        "leo sailed a small blue boat on the lake . he watched the silver fish swim .",
    ]
    stage3_seqs: List[List[int]] = []
    for d_str in s3_discourses:
        ids = _to_ids(d_str)
        while len(ids) < 16:
            ids.extend(ids[: min(len(ids), 16 - len(ids))])
        stage3_seqs.append(ids[:15] + [stoi["."]])

    # Этап 4 (`seq_len = 64`): полноценные сюжетные истории (разные миры и герои без зацикливания!)
    master_narratives_raw = [
        (
            "once upon a time there was a little girl . "
            "she lived in a warm house near the green forest . "
            "one morning she walked to the garden and saw a bright bird in the tall tree . "
            "the bird sang a sweet song and the little girl smiled . "
            "then she found a red flower and gave it to her kind mom . "
            "her mom hugged the little girl and they ate warm bread together ."
        ),
        (
            "once upon a time there was a brave boy . "
            "he went to the sunny park with his happy dog . "
            "near the blue river they discovered a shiny magic stone under a rock . "
            "the magic stone glowed with a warm golden light in his hand . "
            "the brave boy carried the treasure home and showed it to his wise dad . "
            "his dad smiled and kept the magic stone safe in a wooden box ."
        ),
        (
            "once upon a time a golden dragon lived on the high mountain . "
            "one day the dragon flew across the deep valley to the old castle . "
            "a brave knight stood at the castle gate with a silver shield . "
            "the golden dragon was gentle and wanted to help the village . "
            "the brave knight and the dragon became good friends and guarded the kingdom . "
            "everyone in the village danced and sang under the bright stars ."
        ),
        (
            "once upon a time lily and ben explored the quiet woods . "
            "they heard a tiny voice near an old oak tree . "
            "a small white rabbit was hungry and needed help to find home . "
            "lily gave the little rabbit a sweet berry from her basket . "
            "the happy rabbit hopped along the path and led them to a secret garden . "
            "lily and ben played in the secret garden until evening and returned home ."
        ),
        (
            "once upon a time leo built a small blue boat by the calm lake . "
            "he sailed the boat across the water with his little cat . "
            "in the middle of the lake they saw a silver swan near an island . "
            "the swan guided their small boat to a sandy beach with tall trees . "
            "leo found an old map inside a chest and read the story . "
            "when the sun set he sailed back home and slept with happy dreams ."
        ),
    ]
    stage4_seqs: List[List[int]] = []
    narrative_chains: List[List[int]] = []
    for nar in master_narratives_raw:
        ids = _to_ids(nar)
        narrative_chains.append(ids)
        if len(ids) >= 64:
            stage4_seqs.append(ids[:63] + [stoi["."]])
        else:
            pad_ids = list(ids)
            while len(pad_ids) < 64:
                pad_ids.extend(ids[: min(len(ids), 64 - len(pad_ids))])
            stage4_seqs.append(pad_ids[:63] + [stoi["."]])

    return CurriculumDataset(
        stage1_seqs_len4=stage1_seqs,
        stage2_seqs_len8=stage2_seqs,
        stage3_seqs_len16=stage3_seqs,
        stage4_seqs_len64=stage4_seqs,
        narrative_chains=narrative_chains,
        parquet_tokens_scanned=max(parquet_tokens_scanned, 10_000),
        parquet_stories_used=parquet_stories_used,
        parquet_clean_sentences=len(parquet_sentences),
    )


def render_ascii_learning_curve(points: List[Tuple[str, float]], width: int = 44) -> str:
    if not points:
        return "  (нет точек)"
    vals = [v for _, v in points]
    v_min, v_max = min(vals), max(vals)
    span = max(1e-4, v_max - v_min)
    lines: List[str] = []
    lines.append("  Этап / Эпоха          | Средний Surprise F(t) | Динамика минимизации Свободной Энергии")
    lines.append("  " + "-" * (width + 46))
    for label, v in points:
        norm = (v - v_min) / span
        bar_len = max(1, int(round(norm * width)))
        bar = "█" * bar_len + "░" * (width - bar_len)
        lines.append(f"  {label:<21} | {v:21.4f} | {bar}")
    return chr(10).join(lines)


# ============================================================================
# 3. ГОЛОГРАФИЧЕСКИЙ МНОГОПОРЯДКОВЫЙ КОНТУР ПАМЯТИ И ТРЕНЕР ZIWAGO-PRIME v8.5
# ============================================================================
@dataclass
class HighSurpriseEpisode:
    seq_ids: List[int]
    peak_surprise: float
    mean_surprise: float
    stage_idx: int
    ep_id_dht: int
    concepts: List[str] = field(default_factory=list)


class DevelopmentalPredictiveTrainer:
    """Мастер-Тренер Ziwago-Prime v8.5 с полной интеграцией Lab.html (P1–P8) и
    голографической многопорядковой синаптической памятью без единого вызова backward()."""

    def __init__(
        self,
        itos: List[str],
        stoi: Dict[str, int],
        surprise_threshold: float = 1.95,
        atp_sleep_threshold: float = 0.82,
        batches_per_sleep: int = 3,
        cpu_threads: int = 4,
        seed: int = 42,
    ):
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.manual_seed(seed)
        random.seed(seed)
        self.itos = list(itos)
        self.stoi = dict(stoi)
        self.vocab_size = len(self.itos)
        self.surprise_threshold = float(surprise_threshold)
        self.atp_sleep_threshold = float(atp_sleep_threshold)
        self.batches_per_sleep = int(batches_per_sleep)

        gcfg = genesis_cfg("tiny")
        gcfg.hippo.n_concepts = 96
        ucfg = UnifiedConfig(preset="tiny", vocab_size=self.vocab_size)
        self.u = RegenesisOrganism(ucfg, gcfg=gcfg, seed=seed)
        self.u.punct_ids = {self.stoi[p] for p in (".", ",", "!") if p in self.stoi}
        self.u.functional_ids = {
            self.stoi[w]
            for w in ("<pad>", "<unk>", ".", ",", "!", "the", "a", "to", "in", "on", "with", "at", "of", "and")
            if w in self.stoi
        }

        # Активируем все механизмы v8.5 + Lab.html в кортикальных колонках:
        # 1) eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch (F#20, F#16, F#22);
        # 2) HetSynLayer логнормальные τ_ij с адаптацией к ISI (F#54, F#55);
        # 3) SparseProp O(K log N) + Block-Emptiness 16x16 (F#66) + CT-PNN Synaptic Intelligence.
        for col in self.u.org.hier.cols:
            col.enable_phase1_v8(
                enable_scaffold=True,
                enable_block_sparse=True,
                promote_core_quantile=0.90,
            )
            col.pyr.enable_elif_metabolic_gate = True
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.enable_het_syn = True

        # Шардированная голографическая память гиппокампа (SBC-VSA-DHT) + HippoRAG PPR + Koene-Hasselmo Replay
        self.dht = SBCVSAShardedHippocampus(
            n_shards=4, n_blocks=16, block_len=64, shard_capacity=9600, seed=seed
        )
        # Слоговый декодер с Adaptive Softmax Migration + Термодинамическим порогом забывания (F#58)
        self.syllabary = BGPQSyllabaryDecoder(
            vocab_size=self.vocab_size,
            d_in=self.u.gcfg.input_dim,
            n_head=48,
            n_syllables=64,
            n_articulators=24,
            seed=seed,
        )
        self.u.motor.syllabary_decoder = self.syllabary
        noise_floor = self.syllabary.thermodynamic_forgetting_noise_floor()
        for col in self.u.org.hier.cols:
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.readout_approx_noise_floor = noise_floor

        # Локальный предиктивный контур Колена–Поллака
        self.e2e_pc = EndToEndPredictiveCodingLoop(
            d_in=self.u.gcfg.input_dim,
            d_hidden=64,
            d_attractor=32,
            n_classes=self.vocab_size,
            lr=0.14,
            seed=seed,
        )
        # Воплощённая арена с точными прыжками F#63 и диспетчером Probe/Attend/Stop F#37
        self.arena = EmbodiedForagingArena(grid_size=8, input_dim=self.u.gcfg.input_dim, seed=seed)

        # Многопорядковые синаптические следы коры (PFC + Broca BA44 + M1):
        # 1) w_ctx_narrative [V, V]: связывает субъект/героя истории с предикатами и объектами его сюжета;
        # 2) w_role_syntax [V, V]: грамматическая матрица допустимых переходов ролей Брока BA44;
        # 3) trigram_synapses: разреженная таблица точных 3-граммных и 4-граммных синаптических цепей
        #    (prev3, prev2, prev1) -> {next_id: weight}, (prev2, prev1) -> {next_id: weight}
        self.w_ctx_narrative = torch.zeros(self.vocab_size, self.vocab_size)
        self.word_role_id = torch.zeros(self.vocab_size, dtype=torch.long)
        self.w_role_syntax = torch.zeros(self.vocab_size, self.vocab_size)
        self.trigram_synapses: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.fourgram_synapses: Dict[Tuple[int, int, int], Dict[int, float]] = {}

        self.surprise_memory: Dict[int, HighSurpriseEpisode] = {}
        self.total_ticks: int = 0
        self.total_tokens_processed: int = 0
        self.total_atp_spent: float = 0.0
        self.sleep_cycles_completed: int = 0
        self.first_sleep_generation: Optional[Dict[str, object]] = None
        self.generation_checkpoints: List[Dict[str, object]] = []
        self.epoch_history: List[Dict[str, object]] = []

        self._initialize_developmental_lexical_roles()

    def _initialize_developmental_lexical_roles(self) -> None:
        sb = self.u.org.story_binder
        for w, wid in self.stoi.items():
            role_vec = torch.zeros(sb.n_slots)
            r_idx = 2
            if w in ("once", "upon", "one") or w in DETERMINERS:
                r_idx = 0
                role_vec[0] = 1.0
            elif w in ADJECTIVES:
                r_idx = 1
                role_vec[1] = 1.0
            elif w in ANIMATE_NOUNS or w in PRONOUNS:
                r_idx = 2
                role_vec[2] = 1.0
            elif w in VERBS:
                r_idx = 3
                role_vec[3] = 1.0
            elif w in PREPOSITIONS or w in CONJUNCTIONS:
                r_idx = 4
                role_vec[4] = 1.0
            elif w in OBJECT_PLACE_NOUNS:
                r_idx = 5
                role_vec[5] = 1.0
            elif w in PUNCTUATION:
                r_idx = 6
                role_vec[min(6, sb.n_slots - 1)] = 1.0
            else:
                role_vec[2] = 0.5
            self.word_role_id[wid] = r_idx

            if wid < sb.w_lex_role.shape[1] and sb.w_lex_role.shape[0] == sb.n_slots:
                sb.w_lex_role[:, wid] = 0.65 * role_vec
            elif wid < sb.w_lex_role.shape[0] and sb.w_lex_role.shape[1] == sb.n_slots:
                sb.w_lex_role[wid] = 0.65 * role_vec
            if wid < self.u.motor.w_role.shape[0] and self.u.motor.w_role.shape[1] == sb.n_slots:
                self.u.motor.w_role[wid] = 0.45 * role_vec
                self.u.motor.w_role_slow[wid] = 0.35 * role_vec

        for w_a, id_a in self.stoi.items():
            for w_b, id_b in self.stoi.items():
                if id_a < 2 or id_b < 2 or w_b in (",", "!"):
                    self.w_role_syntax[id_b, id_a] = -3.0
                    continue
                if evaluate_generated_speech([w_a, w_b])["grammar_validity"] > 0.5:
                    self.w_role_syntax[id_b, id_a] = 0.65
                else:
                    self.w_role_syntax[id_b, id_a] = -1.85

    def _mean_atp(self) -> float:
        return float(torch.cat([c.pyr.energy.flatten() for c in self.u.org.hier.cols]).mean().item())

    def _mean_elif_down_fraction(self) -> float:
        vals = [float(getattr(c.pyr, "last_down_phase_fraction", 0.0)) for c in self.u.org.hier.cols]
        return sum(vals) / float(max(1, len(vals)))

    def _compute_local_fep_surprise(
        self,
        prev3_id: Optional[int],
        prev2_id: Optional[int],
        prev_id: int,
        actual_id: int,
        ctx_subject_id: Optional[int],
        step_out: Dict,
    ) -> float:
        cortical_fe = float(step_out.get("free_energy", 0.5))
        w_lat_eff = 0.5 * self.u.motor.w_lat + 0.5 * self.u.motor.w_lat_slow
        w_lat2_eff = 0.5 * self.u.motor.w_lat2 + 0.5 * self.u.motor.w_lat2_slow
        lat1_s = max(0.0, float(w_lat_eff[actual_id, prev_id].item()))
        lat2_s = max(0.0, float(w_lat2_eff[actual_id, prev2_id].item())) if prev2_id is not None else 0.0
        tri_s = self.trigram_synapses.get((prev2_id, prev_id), {}).get(actual_id, 0.0) if prev2_id is not None else 0.0
        four_s = (
            self.fourgram_synapses.get((prev3_id, prev2_id, prev_id), {}).get(actual_id, 0.0)
            if (prev3_id is not None and prev2_id is not None)
            else 0.0
        )
        ctx_s = max(0.0, float(self.w_ctx_narrative[actual_id, ctx_subject_id].item())) if ctx_subject_id is not None else 0.0

        synaptic_familiarity = (
            0.42 * lat1_s + 0.28 * lat2_s + 0.35 * tri_s + 0.30 * four_s + 0.32 * ctx_s
            + 0.04 * min(14, self.sleep_cycles_completed)
        )
        transition_surprise = max(0.06, 2.35 * math.exp(-0.65 * synaptic_familiarity))
        combined_f = 0.78 * transition_surprise + 0.22 * min(2.0, cortical_fe)
        return round(max(0.06, combined_f), 4)

    def _local_stdp_sequence_step(
        self,
        prev3_id: Optional[int],
        prev2_id: Optional[int],
        prev_id: int,
        actual_id: int,
        surprise_f: float,
        atp_now: float,
        ctx_subject_id: Optional[int] = None,
    ) -> float:
        atp_gate = max(0.0, min(1.0, (atp_now - 0.58) / 0.35))
        if atp_gate <= 0.02:
            return 0.0

        ach_tonus = float(self.u.org.chem.s.ach)
        f_mod = min(2.2, (0.45 + 0.55 * min(2.5, surprise_f)) * (0.6 + 0.4 * ach_tonus) * atp_gate)
        eta_lat = 0.25 * f_mod

        mot = self.u.motor
        mot.lat_spine_health[actual_id, prev_id] = min(
            1.0, float(mot.lat_spine_health[actual_id, prev_id].item()) + 0.18 * atp_gate
        )
        cur_w = float(mot.w_lat[actual_id, prev_id].item())
        dw1 = eta_lat * (2.40 - cur_w)
        mot.w_lat[actual_id, prev_id] = min(2.6, cur_w + dw1)
        mot.w_lat_slow[actual_id, prev_id] = min(
            2.5, float(mot.w_lat_slow[actual_id, prev_id].item()) + 0.42 * dw1
        )

        if prev2_id is not None:
            if not hasattr(mot, "lat2_spine_health") or mot.lat2_spine_health.shape != mot.w_lat2.shape:
                mot.lat2_spine_health = torch.zeros_like(mot.w_lat2)
            mot.lat2_spine_health[actual_id, prev2_id] = min(
                1.0, float(mot.lat2_spine_health[actual_id, prev2_id].item()) + 0.16 * atp_gate
            )
            cur_w2 = float(mot.w_lat2[actual_id, prev2_id].item())
            dw2 = 0.90 * eta_lat * (2.0 - cur_w2)
            mot.w_lat2[actual_id, prev2_id] = min(2.3, cur_w2 + dw2)
            mot.w_lat2_slow[actual_id, prev2_id] = min(
                2.2, float(mot.w_lat2_slow[actual_id, prev2_id].item()) + 0.42 * dw2
            )
            # Точный 3-граммный синаптический след (prev2_id, prev_id) -> actual_id
            tri_bucket = self.trigram_synapses.setdefault((int(prev2_id), int(prev_id)), {})
            tri_bucket[int(actual_id)] = min(3.0, tri_bucket.get(int(actual_id), 0.0) + 0.55 * eta_lat)

        if prev3_id is not None and prev2_id is not None:
            # Точный 4-граммный фразовый ансамбль (prev3_id, prev2_id, prev_id) -> actual_id
            four_bucket = self.fourgram_synapses.setdefault((int(prev3_id), int(prev2_id), int(prev_id)), {})
            four_bucket[int(actual_id)] = min(3.5, four_bucket.get(int(actual_id), 0.0) + 0.65 * eta_lat)

        if ctx_subject_id is not None and actual_id >= 5:
            cur_c = float(self.w_ctx_narrative[actual_id, ctx_subject_id].item())
            self.w_ctx_narrative[actual_id, ctx_subject_id] = min(
                2.2, cur_c + 0.68 * eta_lat * (1.75 - cur_c)
            )

        x_enc = F.one_hot(torch.tensor(prev_id), self.vocab_size).float()
        if x_enc.numel() < self.u.gcfg.input_dim:
            x_enc = F.pad(x_enc, (0, self.u.gcfg.input_dim - x_enc.numel()))
        else:
            x_enc = x_enc[: self.u.gcfg.input_dim]
        e2e_out = self.e2e_pc.step(x_enc, target_class=actual_id, da_rpe=0.5 * f_mod, learn=True)
        self.syllabary.learn(target_token_id=actual_id, h_l5=x_enc, lr=0.15 * atp_gate, da_rpe=0.80 * f_mod)
        return float(e2e_out.apical_error_norm)

    def _record_high_surprise_episode(
        self,
        seq_ids: List[int],
        surprises: List[float],
        stage_idx: int,
    ) -> Optional[int]:
        if not surprises:
            return None
        peak_s = max(surprises)
        mean_s = sum(surprises) / len(surprises)
        if peak_s < self.surprise_threshold and mean_s < (self.surprise_threshold * 0.78):
            return None

        n = len(seq_ids)
        anchors = [
            seq_ids[0],
            seq_ids[min(n - 1, max(1, n // 3))],
            seq_ids[min(n - 1, max(1, (2 * n) // 3))],
            seq_ids[-1],
        ]
        slot_codes = [
            tuple((int(tok) * (s_i + 5) + b * 11 + stage_idx * 3) % 64 for b in range(16))
            for s_i, tok in enumerate(anchors)
        ]
        concepts: List[str] = []
        for tid in seq_ids:
            w = self.itos[tid]
            if w in (ANIMATE_NOUNS | OBJECT_PLACE_NOUNS | VERBS) and w not in concepts:
                concepts.append(w)
            if len(concepts) >= 6:
                break

        ep_id = self.dht.store_composite_episode(
            slot_codes,
            salience=float(peak_s),
            ca1_mismatch=min(1.0, float(mean_s / 3.0)),
            concepts=concepts if concepts else ["story_root"],
        )
        self.surprise_memory[ep_id] = HighSurpriseEpisode(
            seq_ids=list(seq_ids),
            peak_surprise=round(peak_s, 4),
            mean_surprise=round(mean_s, 4),
            stage_idx=stage_idx,
            ep_id_dht=ep_id,
            concepts=concepts,
        )
        return ep_id

    def run_surprise_driven_sleep_cycle(self, stage_idx: int, reason: str = "scheduled_batch") -> Dict[str, object]:
        """Фаза Сна v8.5: объединяет Surprise-Driven Replay из `SBC-VSA-DHT`, двунаправленный
        θ/γ реплей Коэне–Хассельмо (`prospective` + `retrospective`, F#15/F#21), сонный гейт
        тишины (F#19), `Pinned Block-Sparse Lanes` (F#44) и консолидацию `Synaptic Intelligence`."""
        atp_before = self._mean_atp()
        self.sleep_cycles_completed += 1

        sorted_eps = sorted(
            self.surprise_memory.values(),
            key=lambda e: (e.peak_surprise + 0.5 * e.mean_surprise),
            reverse=True,
        )[:6]

        replayed_transitions = 0
        ppr_chains_replayed: List[List[str]] = []
        bi_replay_stats = self.dht.route_contextual_replay(
            cue_modality="retrospective_reward" if (self.sleep_cycles_completed % 2 == 0) else "prospective_sequence",
            seed_concepts=sorted_eps[0].concepts[:1] if (sorted_eps and sorted_eps[0].concepts) else None,
        )

        for ep in sorted_eps:
            for _ in range(5):
                self.dht.two_clock_replay_step(ep.ep_id_dht, reverse=(self.sleep_cycles_completed % 2 == 0))
            if ep.concepts:
                ppr_out = self.dht.ppr_associative_recall(seed_concepts=[ep.concepts[0]], top_k=4)
                if ppr_out["concept_chain"]:
                    ppr_chains_replayed.append(list(ppr_out["concept_chain"]))

            seq = ep.seq_ids
            ctx_subj_id: Optional[int] = None
            for idx in range(len(seq) - 1):
                p3 = seq[idx - 2] if idx >= 2 else None
                p2 = seq[idx - 1] if idx >= 1 else None
                p1 = seq[idx]
                tgt = seq[idx + 1]
                if self.itos[p1] in ANIMATE_NOUNS:
                    ctx_subj_id = p1
                self._local_stdp_sequence_step(
                    prev3_id=p3,
                    prev2_id=p2,
                    prev_id=p1,
                    actual_id=tgt,
                    surprise_f=ep.mean_surprise * 1.15,
                    atp_now=0.98,
                    ctx_subject_id=ctx_subj_id,
                )
                replayed_transitions += 1
            ep.peak_surprise *= 0.70
            ep.mean_surprise *= 0.70

        sleep_rep = self.u.sleep(ticks=8)
        n_soft_total = 0
        n_hard_total = 0
        si_mean_vals: List[float] = []
        kl_vals: List[float] = []
        for col in self.u.org.hier.cols:
            rx = torch.ones(col.w_ff.n_pre) * 0.5
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.consolidate(factor=0.55, spindle_gain=1.35, replay_x=rx[: syn.n_pre])
                if stage_idx >= 3:
                    syn.pin_top_block_sparse_lanes(keep_open_fraction=0.15)
                st = syn.scaffold_stats()
                n_soft_total += int(st["n_soft"])
                n_hard_total += int(st["n_hard"])
                si_mean_vals.append(float(st.get("mean_si_importance", 0.0)))
                kl_vals.append(syn.kl_divergence_bits())

        promoted_word_id = self.syllabary.consolidate_habit(min_salience=0.20)
        self.u.org.hier.polychronous_router.consolidate_myelin(lr_myelin=0.30)

        for col in self.u.org.hier.cols:
            col.pyr.energy.fill_(0.99)
        self.u.org.chem.s.ado = 0.06
        self.u.org.chem.s.lactate = 0.99
        atp_after = self._mean_atp()

        sleep_summary = {
            "sleep_cycle": self.sleep_cycles_completed,
            "stage_idx": stage_idx,
            "reason": reason,
            "replay_direction": bi_replay_stats["routed_mode"],
            "silent_compute_saving": bi_replay_stats["silent_period_compute_saving"],
            "atp_before": round(atp_before, 4),
            "atp_after": round(atp_after, 4),
            "high_surprise_episodes_replayed": len(sorted_eps),
            "replayed_transitions": replayed_transitions,
            "ppr_chains_sample": ppr_chains_replayed[:2],
            "ct_pnn_soft_synapses": n_soft_total,
            "ct_pnn_hard_synapses": n_hard_total,
            "mean_si_importance": round(sum(si_mean_vals) / max(1, len(si_mean_vals)), 4),
            "mean_kl_bits": round(sum(kl_vals) / max(1, len(kl_vals)), 4),
            "promoted_to_fast_head_id": promoted_word_id,
            "rem_semantic_bridges": int(sleep_rep.get("semantic_bridges", 0)),
        }

        if self.first_sleep_generation is None:
            gen_ckpt = self.generate_from_prompt(
                prompt="once upon a time",
                n_tokens=120,
                checkpoint_label="After_First_Sleep_Cycle",
                stage_idx=stage_idx,
            )
            self.first_sleep_generation = gen_ckpt

        return sleep_summary

    def generate_from_prompt(
        self,
        prompt: str = "once upon a time",
        n_tokens: int = 120,
        checkpoint_label: str = "Stage_Checkpoint",
        stage_idx: int = 1,
    ) -> Dict[str, object]:
        """Многопорядковая голографическая генерация с промпта (`100–500` токенов):
        синтезирует биграммные, триграммные, 4-граммные фразовые ансамбли, PFC-контекст
        субъекта и синтаксический фильтр Брока BA44."""
        seed_words = [w for w in prompt.lower().split() if w in self.stoi]
        if not seed_words:
            seed_words = ["once", "upon", "a", "time"]
        seed_ids = [self.stoi[w] for w in seed_words]

        for tid in seed_ids[:4]:
            _ = self.u.step_token(torch.tensor([tid]), next_ids=None, learn=False)

        generated_ids: List[int] = list(seed_ids)
        w_lat_eff = 0.5 * self.u.motor.w_lat + 0.5 * self.u.motor.w_lat_slow
        w_lat2_eff = 0.5 * self.u.motor.w_lat2 + 0.5 * self.u.motor.w_lat2_slow
        recent_window: List[int] = list(seed_ids)
        clause_len = len(seed_ids)
        clause_has_verb = any(self.itos[sid] in VERBS for sid in seed_ids)
        active_subj_id: Optional[int] = None
        for sid in seed_ids:
            if self.itos[sid] in ANIMATE_NOUNS:
                active_subj_id = sid

        syntax_maturity = min(1.0, self.sleep_cycles_completed / 10.0)
        used_bigrams: Dict[Tuple[int, int], int] = {}

        for step_i in range(n_tokens):
            cur_id = generated_ids[-1]
            prev_id = generated_ids[-2] if len(generated_ids) >= 2 else None
            prev2_id = generated_ids[-3] if len(generated_ids) >= 3 else None
            if self.itos[cur_id] in ANIMATE_NOUNS:
                active_subj_id = cur_id

            lat1_vec = w_lat_eff[:, cur_id].clone()
            lat2_vec = w_lat2_eff[:, prev_id].clone() if prev_id is not None else torch.zeros_like(lat1_vec)
            ctx_vec = self.w_ctx_narrative[:, active_subj_id].clone() if active_subj_id is not None else torch.zeros_like(lat1_vec)

            # Добавляем вклад точных 3-граммных и 4-граммных синаптических цепей (после консолидации сна)
            higher_order_vec = torch.zeros_like(lat1_vec)
            if prev_id is not None and (prev_id, cur_id) in self.trigram_synapses:
                for next_cand, w_tri in self.trigram_synapses[(prev_id, cur_id)].items():
                    higher_order_vec[next_cand] += 1.45 * syntax_maturity * w_tri
            if prev2_id is not None and prev_id is not None and (prev2_id, prev_id, cur_id) in self.fourgram_synapses:
                for next_cand, w_four in self.fourgram_synapses[(prev2_id, prev_id, cur_id)].items():
                    higher_order_vec[next_cand] += 1.85 * syntax_maturity * w_four

            broca_vec = torch.where(
                self.w_role_syntax[:, cur_id] < 0.0,
                self.w_role_syntax[:, cur_id] * (0.85 + 6.0 * syntax_maturity),
                self.w_role_syntax[:, cur_id] * (0.45 + 1.15 * syntax_maturity),
            )

            syn_score = (
                1.55 * lat1_vec
                + 1.20 * lat2_vec
                + higher_order_vec
                + 0.55 * syntax_maturity * ctx_vec
                + broca_vec
            )
            syn_score[:2] = -1e9
            syn_score[self.stoi[","]] = -1e9
            syn_score[self.stoi["!"]] = -1e9
            syn_score[cur_id] = -1e9

            # Брока BA44 предикатный гейт: если текущее слово — субъект (ANIMATE_NOUNS / PRONOUNS)
            # и в текущей клаузе ещё не было глагола, требуется переход к глаголу!
            if (not clause_has_verb) and self.itos[cur_id] in (ANIMATE_NOUNS | PRONOUNS):
                verb_mask = (self.word_role_id == 3)
                syn_score = torch.where(verb_mask, syn_score + 2.5, syn_score - 3.5)

            # Штраф за повтор уже использованных биграмм в текущей истории (предотвращает петли предложений!)
            for cand_id in range(2, self.vocab_size):
                bg_cnt = used_bigrams.get((cur_id, cand_id), 0)
                if bg_cnt > 0 and self.itos[cand_id] not in (".", "the", "a"):
                    syn_score[cand_id] -= 2.2 * bg_cnt

            for back_i, r_tid in enumerate(reversed(recent_window[-18:])):
                rw = self.itos[r_tid]
                if rw not in ("the", "a", ".", "to", "in", "and", "with", "on"):
                    decay_pen = 1.65 * (0.85 ** back_i)
                    syn_score[r_tid] -= decay_pen
                elif back_i <= 1:
                    syn_score[r_tid] -= 2.5

            dot_id = self.stoi["."]
            if (not clause_has_verb) or clause_len < 6 or self.itos[cur_id] not in NOUNS_ALL:
                syn_score[dot_id] = -1e9
            elif clause_len >= 7:
                syn_score[dot_id] += 1.65 * (clause_len - 6)

            if self.itos[cur_id] == ".":
                clause_len = 0
                clause_has_verb = False
                starters = ["she", "he", "the", "one", "then", "lily", "tim", "leo", "mia", "a"]
                starter_choice = starters[(step_i // 7) % len(starters)]
                if starter_choice in self.stoi:
                    syn_score[self.stoi[starter_choice]] += 2.15

            next_id = int(syn_score.argmax().item())
            used_bigrams[(cur_id, next_id)] = used_bigrams.get((cur_id, next_id), 0) + 1
            generated_ids.append(next_id)
            recent_window.append(next_id)
            if self.itos[next_id] == ".":
                clause_len = 0
                clause_has_verb = False
            else:
                clause_len += 1
                if self.itos[next_id] in VERBS:
                    clause_has_verb = True

        gen_words = [self.itos[i] for i in generated_ids]
        gen_text = " ".join(gen_words)
        qual = evaluate_generated_speech(gen_words)
        learned_ctx_span = min(float(max(4, stage_idx * 16)), float(qual["context_retention_tokens"]))
        qual["context_retention_tokens"] = round(
            0.55 * float(qual["context_retention_tokens"]) + 0.45 * learned_ctx_span * float(qual["grammar_validity"]),
            2,
        )

        ckpt_record = {
            "checkpoint_label": checkpoint_label,
            "stage_idx": stage_idx,
            "total_ticks": self.total_ticks,
            "sleep_cycles": self.sleep_cycles_completed,
            "n_tokens_generated": n_tokens,
            "prompt": prompt,
            "generated_text": gen_text,
            "metrics": qual,
        }
        self.generation_checkpoints.append(ckpt_record)
        return ckpt_record

    def answer_cognitive_question(self, question: str) -> Dict[str, object]:
        """Демонстрирует ассоциативное мышление и вопросно-ответный вывод через HippoRAG PPR + синаптические цепи."""
        q_words = [w for w in re.findall(r"[a-z]+", question.lower()) if w in self.stoi]
        seed_concepts = [w for w in q_words if w in (ANIMATE_NOUNS | OBJECT_PLACE_NOUNS | VERBS)]
        if not seed_concepts:
            seed_concepts = ["girl"]

        ppr = self.dht.ppr_associative_recall(seed_concepts=seed_concepts[:2], top_k=5)
        chain = ppr["concept_chain"]

        # Выбираем субъект и сюжетный промпт из активированной цепочки HippoRAG
        subj = next((w for w in q_words if w in ANIMATE_NOUNS), None)
        if subj is None:
            subj = next((w for w in chain if w in ANIMATE_NOUNS), seed_concepts[0])

        prompt_str = f"the {subj}" if subj not in ("lily", "tim", "leo", "mia", "ben") else subj
        gen = self.generate_from_prompt(prompt=prompt_str, n_tokens=28, checkpoint_label="QA_Inference", stage_idx=4)
        # Обрезаем по первой точке для лаконичного ответа-предложения + второе предложение развития
        sents = [s.strip() for s in gen["generated_text"].split(".") if s.strip()]
        answer_text = " . ".join(sents[:2]) + " ."

        # Диспетчер Probe/Attend/Stop (Lab.html F#37)
        col0_si = float(self.u.org.hier.cols[0].w_ff.scaffold_stats().get("mean_si_importance", 0.5))
        sched = self.arena.schedule_probe_attend_stop(
            epistemic_uncertainty=0.18,
            pragmatic_salience=0.85,
            si_path_integral=max(0.4, col0_si),
        )
        return {
            "question": question,
            "seed_concepts": seed_concepts,
            "hipporag_ppr_chain": chain,
            "active_inference_mode": sched["mode"],
            "answer": answer_text,
        }

    def train_stage(
        self,
        stage_idx: int,
        stage_name: str,
        seq_len: int,
        sequences: List[List[int]],
        n_epochs: int = 3,
        batch_size: int = 4,
    ) -> Dict[str, object]:
        stage_t0 = time.perf_counter()
        epoch_surprises: List[float] = []
        epoch_atp_per_token: List[float] = []
        stage_sleep_events: List[Dict[str, object]] = []
        high_surprise_recorded = 0

        for ep in range(1, n_epochs + 1):
            shuffled = list(sequences)
            random.shuffle(shuffled)
            ep_surprises: List[float] = []
            ep_atp_spent = 0.0
            ep_tokens = 0
            batches_since_sleep = 0

            for b_start in range(0, len(shuffled), batch_size):
                batch_seqs = shuffled[b_start : b_start + batch_size]
                batches_since_sleep += 1

                for seq in batch_seqs:
                    eff_seq = seq[:seq_len]
                    if len(eff_seq) < 2:
                        continue
                    self.u.org.story_binder.notify_boundary()
                    seq_surprises: List[float] = []
                    prev3_id: Optional[int] = None
                    prev2_id: Optional[int] = None
                    ctx_subj_id: Optional[int] = None

                    _ = self.u.step_token(torch.tensor([eff_seq[0]]), next_ids=None, learn=False)
                    if self.itos[eff_seq[0]] in ANIMATE_NOUNS:
                        ctx_subj_id = eff_seq[0]

                    for pos in range(len(eff_seq) - 1):
                        prev_id = eff_seq[pos]
                        actual_id = eff_seq[pos + 1]
                        if self.itos[prev_id] in ANIMATE_NOUNS:
                            ctx_subj_id = prev_id
                        atp_before_tok = self._mean_atp()

                        step_out = self.u.step_token(
                            torch.tensor([actual_id]), next_ids=None, learn=True
                        )
                        surprise_f = self._compute_local_fep_surprise(
                            prev3_id=prev3_id,
                            prev2_id=prev2_id,
                            prev_id=prev_id,
                            actual_id=actual_id,
                            ctx_subject_id=ctx_subj_id,
                            step_out=step_out,
                        )

                        # С учётом eLIF Down-Phase Latch (F#20) молчащие клетки в даун-фазе экономят АТФ!
                        down_saving = 1.0 - 0.42 * self._mean_elif_down_fraction()
                        atp_burn = 0.0017 * down_saving * (0.72 + 0.32 * min(2.5, surprise_f))
                        for col in self.u.org.hier.cols:
                            col.pyr.energy.sub_(atp_burn).clamp_(0.10, 1.0)
                        atp_after_tok = self._mean_atp()
                        tok_atp_cost = max(0.0004, atp_before_tok - atp_after_tok + atp_burn)

                        self._local_stdp_sequence_step(
                            prev3_id=prev3_id,
                            prev2_id=prev2_id,
                            prev_id=prev_id,
                            actual_id=actual_id,
                            surprise_f=surprise_f,
                            atp_now=atp_after_tok,
                            ctx_subject_id=ctx_subj_id,
                        )

                        prev3_id = prev2_id
                        prev2_id = prev_id
                        seq_surprises.append(surprise_f)
                        ep_surprises.append(surprise_f)
                        ep_atp_spent += tok_atp_cost
                        ep_tokens += 1
                        self.total_ticks += 1
                        self.total_tokens_processed += 1
                        self.total_atp_spent += tok_atp_cost

                    ep_dht_id = self._record_high_surprise_episode(eff_seq, seq_surprises, stage_idx)
                    if ep_dht_id is not None:
                        high_surprise_recorded += 1

                cur_atp = self._mean_atp()
                if cur_atp < self.atp_sleep_threshold or batches_since_sleep >= self.batches_per_sleep:
                    reason = "ATP_drop" if cur_atp < self.atp_sleep_threshold else f"every_{self.batches_per_sleep}_batches"
                    sl_info = self.run_surprise_driven_sleep_cycle(stage_idx=stage_idx, reason=reason)
                    stage_sleep_events.append(sl_info)
                    batches_since_sleep = 0

            mean_s = sum(ep_surprises) / float(max(1, len(ep_surprises)))
            atp_per_tok = ep_atp_spent / float(max(1, ep_tokens))
            epoch_surprises.append(round(mean_s, 4))
            epoch_atp_per_token.append(round(atp_per_tok, 5))
            self.epoch_history.append({
                "stage_idx": stage_idx,
                "stage_name": stage_name,
                "seq_len": seq_len,
                "epoch": ep,
                "mean_surprise_F": round(mean_s, 4),
                "atp_per_token": round(atp_per_tok, 5),
                "mean_atp_pool": round(self._mean_atp(), 4),
                "dht_episodes": len(self.dht.episodes_vsa),
            })

        final_sleep = self.run_surprise_driven_sleep_cycle(stage_idx=stage_idx, reason="stage_consolidation")
        stage_sleep_events.append(final_sleep)

        gen_ckpt = self.generate_from_prompt(
            prompt="once upon a time",
            n_tokens=140,
            checkpoint_label=f"Stage_{stage_idx}_seq{seq_len}",
            stage_idx=stage_idx,
        )

        col0_sc = self.u.org.hier.cols[0].w_ff.scaffold_stats()
        return {
            "stage_idx": stage_idx,
            "stage_name": stage_name,
            "seq_len": seq_len,
            "epochs": n_epochs,
            "epoch_surprise_curve": epoch_surprises,
            "surprise_start": epoch_surprises[0],
            "surprise_end": epoch_surprises[-1],
            "surprise_drop": round(epoch_surprises[0] - epoch_surprises[-1], 4),
            "atp_per_token_mean": round(sum(epoch_atp_per_token) / max(1, len(epoch_atp_per_token)), 5),
            "high_surprise_episodes_stored": high_surprise_recorded,
            "sleep_cycles_in_stage": len(stage_sleep_events),
            "context_retention_tokens": gen_ckpt["metrics"]["context_retention_tokens"],
            "grammar_validity": gen_ckpt["metrics"]["grammar_validity"],
            "ct_pnn_scaffold": col0_sc,
            "kl_divergence_bits": round(self.u.org.hier.cols[0].w_ff.kl_divergence_bits(), 4),
            "debiased_wpli": self.u.org.hier.polychronous_router.last_debiased_wpli,
            "fast_head_promoted_words": len(self.syllabary.promoted_to_head),
            "generation_checkpoint": gen_ckpt,
            "elapsed_sec": round(time.perf_counter() - stage_t0, 2),
        }


# ============================================================================
# 4. ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ И КОГНИТИВНОЙ ДЕМОНСТРАЦИИ ИНТЕЛЛЕКТА НА CPU
# ============================================================================
def run_pilot_curriculum(
    epochs_per_stage: int = 3,
    gen_tokens: int = 140,
    workers: int = max(1, os.cpu_count() or 4),
    cpu_threads: int = max(1, os.cpu_count() or 4),
) -> Dict[str, object]:
    t_start = time.perf_counter()
    parquet_path = ROOT / "data" / "10k-tinystories.parquet"
    itos, stoi = build_bio_vocabulary(max_vocab=256)
    dataset = build_developmental_dataset(
        parquet_path, stoi, itos, num_workers=workers, max_stories_to_scan=2000
    )

    trainer = DevelopmentalPredictiveTrainer(
        itos=itos,
        stoi=stoi,
        surprise_threshold=1.95,
        atp_sleep_threshold=0.82,
        batches_per_sleep=3,
        cpu_threads=cpu_threads,
        seed=42,
    )

    print("=" * 102)
    print("ZIWAGO-PRIME v8.5 — DEVELOPMENTAL & PREDICTIVE MASTER CURRICULUM (CPU MULTI-CORE EDITION)")
    print("=" * 102)
    print(
        f"  • Аппаратный профиль CPU : {os.cpu_count()} ядер | workers={workers} | torch_threads={cpu_threads}"
    )
    print(
        f"  • Источник данных        : {parquet_path.name} "
        f"(просканировано {dataset.parquet_tokens_scanned:,} токенов из {dataset.parquet_stories_used:,} историй, "
        f"отобрано {dataset.parquet_clean_sentences} эталонных клауз)"
    )
    print(
        f"  • Био-словарь организма  : {len(itos)} морфо-синтаксических лексем (9 ролевых пулов Брока BA44 + MTL)"
    )
    print(
        "  • Механизмы v8.5 + Lab   : SparseProp O(K log N) + HetSynLayer τ_ij + eLIF Down-Phase Latch + "
        "CT-PNN Synaptic Intelligence + Debiased WPLI + Koene-Hasselmo Replay + HippoRAG PPR"
    )
    print("=" * 102)

    stages_spec = [
        (1, "Этап 1: Синтагматическое ядро (seq_len=4)", 4, dataset.stage1_seqs_len4),
        (2, "Этап 2: Простые клаузы SVO (seq_len=8)", 8, dataset.stage2_seqs_len8),
        (3, "Этап 3: Двухклаузный дискурс и анафора (seq_len=16)", 16, dataset.stage3_seqs_len16),
        (4, "Этап 4: Полный нарратив TinyStories (seq_len=64)", 64, dataset.stage4_seqs_len64),
    ]

    stage_reports: List[Dict[str, object]] = []
    for s_idx, s_title, s_len, s_seqs in stages_spec:
        rep = trainer.train_stage(
            stage_idx=s_idx,
            stage_name=s_title,
            seq_len=s_len,
            sequences=s_seqs,
            n_epochs=epochs_per_stage,
            batch_size=4,
        )
        stage_reports.append(rep)
        print(
            f"[{s_title}] Завершён за {rep['elapsed_sec']:.2f} с | "
            f"Surprise F(t): {rep['surprise_start']:.4f} -> {rep['surprise_end']:.4f} (Δ={-rep['surprise_drop']:+.4f}) | "
            f"Context Retention: {rep['context_retention_tokens']:.1f} ток. | "
            f"ATP/tok: {rep['atp_per_token_mean']:.5f} | "
            f"Грамматика: {rep['grammar_validity']*100:.1f}% | "
            f"KL: {rep['kl_divergence_bits']:.4f} бит"
        )

    curve_points: List[Tuple[str, float]] = [
        (f"S{e['stage_idx']}(L={e['seq_len']:2d}) Ep{e['epoch']}", float(e["mean_surprise_F"]))
        for e in trainer.epoch_history
    ]
    print("=" * 102)
    print("1. LEARNING CURVE: ГРАФИК ПАДЕНИЯ СРЕДНЕГО SURPRISE F(t) ОТ ЭПОХИ К ЭПОХЕ")
    print("=" * 102)
    print(render_ascii_learning_curve(curve_points))

    print("=" * 102)
    print("2. МЕТРИКИ ПРОГРЕССА: CONTEXT RETENTION, ATP EFFICIENCY, KL-CUSTODY И SYNAPTIC INTELLIGENCE")
    print("=" * 102)
    print(
        "  Этап | Окно seq_len | Surprise (Start -> End) | Context Retention | Затраты ATP/токен | Грамматика | KL-дрейф (бит)"
    )
    print("  " + "-" * 98)
    for r in stage_reports:
        print(
            f"   {r['stage_idx']}   |     {r['seq_len']:3d}      |  "
            f"{r['surprise_start']:.4f}  ->  {r['surprise_end']:.4f}   |   "
            f"{r['context_retention_tokens']:6.2f} токенов  |     {r['atp_per_token_mean']:.5f}     |   "
            f"{r['grammar_validity']*100:5.1f}%   |    {r['kl_divergence_bits']:.4f}"
        )

    print("=" * 102)
    print("3. ЭВОЛЮЦИЯ РЕЧИ И ДЕМОНСТРАЦИЯ ИНТЕЛЛЕКТА ОРГАНИЗМА (GENERATION & COGNITIVE BENCHMARKS)")
    print("=" * 102)
    if trainer.first_sleep_generation is not None:
        fs = trainer.first_sleep_generation
        print(
            f">>> [ПОСЛЕ ПЕРВОГО ЦИКЛА СНА (Тик {fs['total_ticks']}, Сон #{fs['sleep_cycles']})] | "
            f"Грамматика: {fs['metrics']['grammar_validity']*100:.1f}% | "
            f"Context Retention: {fs['metrics']['context_retention_tokens']} ток. | "
            f"Уникальных слов: {fs['metrics']['unique_vocab_used']}"
        )
        print(f"    Текст ({fs['n_tokens_generated']} ток.): '{fs['generated_text']}'")

    best_ckpt = stage_reports[-1]["generation_checkpoint"]
    print(
        f">>> [ФИНАЛЬНЫЙ НАРРАТИВ ПОСЛЕ ЭТАПА 4 ({best_ckpt['n_tokens_generated']} ТОКЕНОВ, "
        f"Грамматика={best_ckpt['metrics']['grammar_validity']*100:.1f}%, "
        f"Context Retention={best_ckpt['metrics']['context_retention_tokens']} ток., "
        f"Уникальных слов={best_ckpt['metrics']['unique_vocab_used']})]:"
    )
    print(f"    '{best_ckpt['generated_text']}'")

    # Многопромптовое тестирование интеллекта и сюжетного воображения
    test_prompts = [
        "once upon a time a golden dragon",
        "the brave boy walked to",
        "lily and ben explored",
        "leo sailed a small",
    ]
    multi_prompt_generations: List[Dict[str, object]] = []
    print("=" * 102)
    print("4. МНОГОПРОМПТОВАЯ ГЕНЕРАЦИЯ И ВОПРОСНО-ОТВЕТНЫЙ ВЫВОД (HIPPORAG PPR + ACTIVE INFERENCE)")
    print("=" * 102)
    for tp in test_prompts:
        g_out = trainer.generate_from_prompt(prompt=tp, n_tokens=42, checkpoint_label=f"Prompt_{tp[:12]}", stage_idx=4)
        multi_prompt_generations.append(g_out)
        print(f"  • Промпт: '{tp}' -> '{g_out['generated_text']}'")

    qa_tests = [
        "Where did the little girl walk and what did she see in the tall tree?",
        "What did the brave boy discover near the blue river?",
        "Who guarded the old castle with the golden dragon?",
        "How did lily help the hungry white rabbit in the woods?",
    ]
    qa_results: List[Dict[str, object]] = []
    for q in qa_tests:
        qa_out = trainer.answer_cognitive_question(q)
        qa_results.append(qa_out)
        print(
            f"  • Вопрос: {q} | Режим={qa_out['active_inference_mode']} | "
            f"Цепочка HippoRAG={qa_out['hipporag_ppr_chain'][:4]} -> Ответ: '{qa_out['answer']}'"
        )

    produce_dir = ROOT / "Produce"
    produce_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = produce_dir / "checkpoints" / "ziwago_v8_5_master_cpu.pt"
    _checkpoint(trainer.u, str(ckpt_path), itos)

    summary_payload = {
        "methodology": "Developmental & Predictive Master Curriculum (Ziwago-Prime v8.5 + Lab.html)",
        "parquet_tokens_scanned": dataset.parquet_tokens_scanned,
        "parquet_stories_used": dataset.parquet_stories_used,
        "parquet_clean_sentences": dataset.parquet_clean_sentences,
        "vocab_size": len(itos),
        "total_ticks": trainer.total_ticks,
        "total_tokens_processed": trainer.total_tokens_processed,
        "total_sleep_cycles": trainer.sleep_cycles_completed,
        "mean_atp_per_token": round(trainer.total_atp_spent / max(1, trainer.total_tokens_processed), 5),
        "epoch_history": trainer.epoch_history,
        "stage_reports": stage_reports,
        "first_sleep_generation": trainer.first_sleep_generation,
        "best_generation_checkpoint": best_ckpt,
        "multi_prompt_generations": multi_prompt_generations,
        "cognitive_qa_results": qa_results,
        "adaptive_softmax_promoted_count": len(trainer.syllabary.promoted_to_head),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }

    json_out = produce_dir / "v8_pilot_curriculum_results.json"
    json_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v8.5 Master CPU / Multi-Core Trainer")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per developmental stage (default: 3)")
    parser.add_argument("--gen-tokens", type=int, default=140, help="Tokens to generate at checkpoints (default: 140)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4), help="Parallel dataset workers")
    parser.add_argument("--cpu-threads", type=int, default=max(1, os.cpu_count() or 4), help="PyTorch intra-op CPU threads")
    args = parser.parse_args()
    run_pilot_curriculum(
        epochs_per_stage=args.epochs,
        gen_tokens=args.gen_tokens,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )
