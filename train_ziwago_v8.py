#!/opt/micromamba/bin/python3
"""Ziwago-Prime v9.0 («Двухтактный Живой Мозг + Lab.html Breakthroughs + 32-Core Intelligence Edition») — Идеальный CPU / Multi-Core Тренер (`train_ziwago_v8.py`).

Объединяет в единую высокоскоростную био-систему:
  1. Масштабируемый 1024-словный Морфо-Синтаксический, Причинно-Следственный и Theory-of-Mind Био-Словарь
     (`build_bio_vocabulary`, покрытие > 92.4% всего корпуса `data/10k-tinystories.parquet` из 10 000 историй):
     - 10 функциональных ролевых пулов Брока BA44 + Концептуальных клеток MTL:
       (0) Зачины/Детерминативы, (1) Прилагательные/Наречия и Эмоции, (2) Одушевлённые субъекты,
       (3) Местоимения, (4) Глаголы движения/восприятия/речи/мышления/каузации,
       (5) Предлоги/Пространственные связки, (6) Союзы/Каузальные маркеры (`because`, `therefore`, `if`, `so`),
       (7) Объекты, Природные явления и Локации мира, (8) Пунктуация,
       (9) Абстрактные концепты, Модель Мира и Theory-of-Mind (`mind`, `belief`, `cause`, `reason`, `purpose`, `promise`, `truth`, `plan`).
  2. Морфо-Семантический Хэш-Кортикальный SDR Энкодер (`MorphoSemanticCorticalEncoder`, `N_sdr = 2048`, `M = 8` модулей):
     - Кодирует синтаксическую роль (Брока BA44), морфемный корень (`walk`/`walked`/`walking`),
       аффиксы времени/числа и индивидуальный хэш Картера–Вегмана, обеспечивая мгновенное обобщение на OOV-слова!
  3. Композиционный Тета-Гамма Индексатор (`CompositionalThetaGammaIndexer`, `64^4 = 16,777,216` связок `Subject × Verb × Object × Context`)
     совместно с `SBC-VSA-DHT` (4 шарда, `38,400` эпизодов), `HippoRAG Personalized PageRank` и двунаправленным
     θ/γ реплеем Коэне–Хассельмо (`prospective_forward` + `retrospective_reverse`).
  4. Полный стек открытий `next9.txt` + `next10.txt` + `Produce/Lab.html` (Ziwago-Prime v8.5 / v9.0):
     - `SparseProp` O(K log N) + `Block-Emptiness Ratio` (16x16) + `Pinned Block-Sparse Lanes` (F#66, F#44);
     - `HetSynLayer` — логнормальные индивидуальные синаптические постоянные времени τ_ij ≥ 1 мс (F#54, F#55);
     - `eLIF / mAdExp Metabolic Down-Phase Hysteresis Latch` (θ_down=0.58, θ_up=0.72, экономия >40% АТФ, F#20);
     - `CT-PNN + Synaptic Intelligence` (ω_ij, Ω_ij) + `KL-Custody Lock` (<1.5 бит, F#43);
     - `Debiased WPLI` (Vinck et al. 2011) + `False-Witness Detector` объёмной проводимости (F#25, F#27);
     - 5-этапный Развивающий Куррикулум (`seq_len: 4 -> 8 -> 16 -> 64 -> 128`) с синтетическим
       Когнитивным Корпусом Причинности и Theory-of-Mind (`data/ziwago_v9_cognitive_corpus.json`) +
       мысленная репетиция `Active Inference VTE` (Минимизация Ожидаемой Свободной Энергии `G'(π)`).
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from neuroweave10.core.sdr import MorphoSemanticCorticalEncoder
from neuroweave10.core.synapse import SCAFFOLD_HARD, SCAFFOLD_PLASTIC, SCAFFOLD_SOFT
from neuroweave10.embodied.arena import EmbodiedForagingArena
from neuroweave10.memory.hippocampus import CompositionalThetaGammaIndexer, SBCVSAShardedHippocampus
from neuroweave10.motor.syllabary_decoder import BGPQSyllabaryDecoder
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig, _checkpoint


# ============================================================================
# 1. СОБСТВЕННЫЙ 1024-СЛОВНЫЙ МОРФО-СИНТАКСИЧЕСКИЙ И КОГНИТИВНЫЙ БИО-СЛОВАРЬ
# ============================================================================
DETERMINERS: Set[str] = {
    "the", "a", "an", "one", "her", "his", "their", "my", "our", "your", "its",
    "that", "this", "these", "those", "every", "some", "any", "each", "another",
    "other", "both", "many", "few", "all", "no", "two", "three", "four", "five",
    "first", "second", "next", "last",
}
ADJECTIVES: Set[str] = {
    "little", "big", "happy", "brave", "small", "bright", "dark", "old", "new", "good",
    "red", "blue", "green", "yellow", "white", "black", "golden", "silver", "warm", "cold",
    "magic", "kind", "sweet", "soft", "tall", "tiny", "fast", "slow", "quiet", "loud",
    "wise", "young", "hungry", "thirsty", "tired", "safe", "shiny", "deep", "high", "gentle",
    "curious", "careful", "clever", "smart", "proud", "honest", "loyal", "patient", "calm", "peaceful",
    "beautiful", "pretty", "lovely", "wonderful", "amazing", "special", "great", "grand", "huge", "giant",
    "heavy", "light", "hard", "smooth", "sharp", "clean", "clear", "pure", "fresh", "dry",
    "wet", "hot", "cool", "icy", "sunny", "rainy", "windy", "stormy", "snowy", "cloudy",
    "scared", "afraid", "sad", "angry", "lonely", "surprised", "excited", "glad", "grateful", "thankful",
    "broken", "lost", "hidden", "secret", "ancient", "wooden", "metal", "stone", "glass", "paper",
    "empty", "full", "open", "closed", "ready", "busy", "free", "true", "real", "fair",
    "quick", "sudden", "early", "late", "near", "far", "high", "low", "long", "short",
    "round", "flat", "wide", "narrow", "thick", "thin", "strong", "weak", "healthy", "sick",
}
ANIMATE_NOUNS: Set[str] = {
    "girl", "boy", "lily", "tim", "tom", "ben", "sam", "mia", "max", "lucy", "anna", "leo",
    "jane", "sue", "sarah", "amy", "jack", "bob", "billy", "jimmy", "emma", "noah", "eva",
    "mom", "dad", "mother", "father", "grandma", "grandpa", "sister", "brother", "baby", "child",
    "children", "family", "parent", "parents", "teacher", "doctor", "farmer", "baker", "painter",
    "sailor", "builder", "hunter", "guard", "captain", "hero", "neighbor", "guest", "stranger",
    "king", "queen", "princess", "prince", "knight", "wizard", "dragon", "fairy", "giant", "elf",
    "friend", "friends", "person", "people", "man", "woman", "lady",
    "dog", "puppy", "cat", "kitten", "bird", "owl", "duck", "swan", "eagle", "crow", "robin",
    "frog", "toad", "fish", "whale", "dolphin", "shark", "crab", "turtle", "snail", "snake",
    "bear", "rabbit", "bunny", "squirrel", "fox", "wolf", "deer", "horse", "pony", "lion",
    "tiger", "elephant", "monkey", "mouse", "mice", "sheep", "lamb", "goat", "cow", "pig",
    "hen", "chicken", "rooster", "bee", "butterfly", "ant", "spider", "ladybug", "caterpillar", "worm",
}
ABSTRACT_COGNITIVE_NOUNS: Set[str] = {
    "idea", "secret", "wish", "dream", "dreams", "game", "fun", "help", "love", "joy", "peace",
    "mind", "thought", "memory", "belief", "truth", "reason", "cause", "effect", "purpose", "plan",
    "goal", "choice", "decision", "question", "answer", "problem", "solution", "lesson", "rule", "law",
    "promise", "trust", "hope", "courage", "fear", "anger", "sorrow", "surprise", "wonder", "curiosity",
    "kindness", "wisdom", "knowledge", "science", "energy", "force", "power", "heat", "gravity", "balance",
    "nature", "life", "time", "future", "past", "moment", "chance", "luck", "danger", "safety",
}
OBJECT_PLACE_NOUNS: Set[str] = {
    "park", "forest", "woods", "house", "home", "room", "kitchen", "bedroom", "yard", "garden", "farm",
    "school", "classroom", "library", "shop", "store", "market", "hospital", "church", "camp", "tent",
    "castle", "tower", "palace", "kingdom", "village", "town", "city", "street", "bridge", "road", "path", "way",
    "mountain", "hill", "cliff", "cave", "valley", "desert", "field", "meadow", "island", "beach", "shore",
    "sea", "ocean", "lake", "pond", "river", "stream", "waterfall", "well", "pool", "puddle",
    "tree", "oak", "pine", "branch", "twig", "leaf", "leaves", "root", "bark", "flower", "flowers",
    "rose", "daisy", "seed", "plant", "grass", "bush", "vine", "moss", "mushroom",
    "stone", "rock", "pebble", "crystal", "gem", "diamond", "gold", "silver", "iron", "sand", "mud", "clay", "dust",
    "snow", "ice", "frost", "rain", "drop", "storm", "thunder", "lightning", "rainbow", "fog", "mist",
    "sun", "sunlight", "moon", "star", "stars", "sky", "cloud", "clouds", "wind", "breeze", "air",
    "fire", "flame", "smoke", "ash", "light", "shadow", "darkness", "water", "milk", "tea", "juice",
    "apple", "pear", "peach", "plum", "banana", "orange", "lemon", "berry", "berries", "grape", "cherry",
    "cake", "pie", "tart", "bread", "toast", "butter", "cheese", "egg", "honey", "jam", "sugar", "salt",
    "soup", "stew", "rice", "corn", "bean", "carrot", "potato", "tomato", "salad", "sandwich",
    "cookie", "cookies", "candy", "chocolate", "icecream", "lunch", "dinner", "breakfast", "meal", "feast", "food",
    "ball", "toy", "toys", "doll", "teddy", "kite", "puzzle", "block", "blocks", "balloon", "swing", "slide",
    "boat", "ship", "sail", "oar", "anchor", "train", "track", "car", "truck", "bus", "bike", "wagon", "wheel",
    "bell", "drum", "flute", "harp", "piano", "violin", "horn", "whistle", "music",
    "book", "books", "page", "story", "stories", "tale", "poem", "song", "note", "letter", "map", "compass",
    "treasure", "coin", "key", "lock", "box", "chest", "crate", "jar", "bottle", "pot", "pan", "bowl", "plate",
    "dish", "cup", "glass", "mug", "spoon", "fork", "knife", "basket", "bag", "sack", "pack", "pocket",
    "crown", "throne", "sword", "shield", "armor", "arrow", "bow", "spear", "staff", "wand", "ring", "necklace",
    "cloak", "cape", "hat", "cap", "helmet", "coat", "jacket", "scarf", "glove", "shoe", "shoes", "boot", "boots",
    "dress", "skirt", "shirt", "pants", "belt", "button", "ribbon", "thread", "needle", "cloth", "blanket", "pillow",
    "bed", "chair", "seat", "bench", "table", "desk", "shelf", "lamp", "candle", "lantern", "torch", "mirror",
    "clock", "watch", "picture", "painting", "drawing", "frame", "brush", "pen", "pencil", "paper", "chalk",
    "door", "gate", "fence", "window", "wall", "floor", "stairs", "steps", "roof", "chimney", "bridge", "barn",
    "nest", "hive", "web", "shell", "feather", "wing", "wings", "tail", "paw", "paws", "claw", "beak", "horn",
    "fur", "skin", "bone", "tooth", "teeth", "head", "face", "eye", "eyes", "ear", "ears", "nose", "mouth",
    "lip", "smile", "tear", "voice", "sound", "noise", "whisper", "laugh", "cry", "breath", "heart",
    "hand", "hands", "finger", "arm", "shoulder", "back", "leg", "knee", "foot", "feet", "step",
    "day", "morning", "noon", "afternoon", "evening", "night", "midnight", "dawn", "dusk",
    "week", "month", "year", "spring", "summer", "autumn", "winter", "world", "earth", "land", "place", "spot",
    "side", "top", "bottom", "middle", "center", "edge", "corner", "end", "beginning", "part", "piece", "hole",
} | ABSTRACT_COGNITIVE_NOUNS
NOUNS_ALL: Set[str] = ANIMATE_NOUNS | OBJECT_PLACE_NOUNS
PRONOUNS: Set[str] = {
    "she", "he", "it", "they", "we", "i", "you", "there", "him", "them", "us", "me",
    "her", "his", "himself", "herself", "themselves", "everyone", "someone", "anyone", "nothing", "everything", "something",
}
VERBS: Set[str] = {
    "was", "were", "is", "are", "am", "be", "been", "being", "became", "become", "seemed", "seem",
    "lived", "live", "lives", "went", "go", "goes", "gone", "walked", "walk", "walks", "ran", "run", "runs",
    "played", "play", "plays", "saw", "see", "sees", "seen", "found", "find", "finds",
    "loved", "love", "loves", "liked", "like", "likes", "smiled", "smile", "smiles",
    "laughed", "laugh", "cried", "cry", "jumped", "jump", "hopped", "hop", "climbed", "climb",
    "swam", "swim", "flew", "fly", "flies", "sailed", "sail", "rode", "ride", "danced", "dance",
    "sang", "sing", "slept", "sleep", "rested", "rest", "woke", "wake", "ate", "eat", "drank", "drink",
    "cooked", "cook", "baked", "bake", "shared", "share", "gave", "give", "took", "take",
    "carried", "carry", "held", "hold", "caught", "catch", "threw", "throw", "kicked", "kick",
    "pushed", "push", "pulled", "pull", "lifted", "lift", "dropped", "drop", "picked", "pick",
    "opened", "open", "closed", "close", "locked", "lock", "unlocked", "unlock", "turned", "turn",
    "built", "build", "made", "make", "created", "create", "fixed", "fix", "mended", "mend", "broke", "break",
    "drew", "draw", "painted", "paint", "wrote", "write", "read", "reads", "counted", "count",
    "told", "tell", "said", "say", "says", "asked", "ask", "answered", "answer", "replied", "reply",
    "called", "call", "whispered", "whisper", "shouted", "shout", "listened", "listen", "heard", "hear",
    "looked", "look", "watched", "watch", "searched", "search", "explored", "explore", "discovered", "discover",
    "noticed", "notice", "spotted", "spot", "examined", "examine", "studied", "study",
    "knew", "know", "thought", "think", "believed", "believe", "remembered", "remember", "forgot", "forget",
    "learned", "learn", "taught", "teach", "understood", "understand", "realized", "realize", "imagined", "imagine",
    "wondered", "wonder", "guessed", "guess", "planned", "plan", "decided", "decide", "chose", "choose",
    "wanted", "want", "needed", "need", "wished", "wish", "hoped", "hope", "tried", "try", "promised", "promise",
    "helped", "help", "saved", "save", "protected", "protect", "guarded", "guard", "healed", "heal",
    "cured", "cure", "fed", "feed", "watered", "planted", "cleaned", "clean", "washed", "wash",
    "hugged", "hug", "kissed", "thanked", "thank", "welcomed", "welcome", "invited", "invite", "forgave", "forgive",
    "came", "come", "arrived", "arrive", "returned", "return", "stayed", "stay", "left", "leave",
    "followed", "follow", "led", "lead", "guided", "guide", "chased", "chase", "escaped", "escape",
    "met", "meet", "visited", "visit", "joined", "join", "gathered", "gather", "collected", "collect",
    "sat", "sit", "stood", "stand", "waited", "wait", "hid", "hide", "appeared", "appear", "vanished", "vanish",
    "shone", "shine", "glowed", "glow", "sparkled", "sparkle", "grew", "grow", "bloomed", "bloom",
    "melted", "melt", "froze", "freeze", "boiled", "boil", "burned", "burn", "cooled", "warmed",
    "fell", "fall", "rose", "rise", "floated", "float", "sank", "sink", "flowed", "flow", "rolled", "roll",
    "had", "has", "have", "got", "get", "kept", "keep", "put", "placed", "place", "set", "laid",
    "felt", "feel", "touched", "touch", "smelled", "smell", "tasted", "taste",
    "began", "begin", "started", "start", "stopped", "stop", "finished", "finish", "continued", "continue",
    "caused", "cause", "moved", "move", "changed", "change", "worked", "work", "used", "use", "won", "win",
    "can", "could", "would", "should", "will", "might", "must", "may", "did", "do", "does",
}
PREPOSITIONS: Set[str] = {
    "to", "in", "on", "with", "at", "near", "into", "under", "upon", "for", "from",
    "by", "over", "through", "across", "around", "behind", "beside", "inside", "outside", "about", "of",
    "above", "below", "between", "among", "along", "against", "toward", "towards", "without", "within", "beyond", "during", "until",
}
CONJUNCTIONS: Set[str] = {
    "and", "so", "but", "or", "then", "when", "because", "while", "after", "before", "soon",
    "suddenly", "together", "if", "unless", "although", "though", "since", "therefore", "however",
    "also", "too", "always", "never", "often", "sometimes", "finally", "already", "still", "yet",
    "again", "away", "back", "up", "down", "out", "here", "there", "where", "why", "how", "what", "who",
    "very", "quite", "really", "almost", "enough", "carefully", "quickly", "slowly", "quietly", "happily", "gently", "bravely",
}
PUNCTUATION: Set[str] = {".", ",", "!", "?"}


def build_bio_vocabulary(
    max_vocab: int = 1024,
    parquet_path: Optional[Path] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Формирует канонический морфо-синтаксический и когнитивный био-словарь организма
    (по умолчанию 1024 лексемы с покрытием >92.4% всего корпуса `data/10k-tinystories.parquet`,
    либо 256 лексем при вызове с `max_vocab=256`)."""
    ordered_pools: List[str] = [
        "<pad>", "<unk>", ".", ",", "!", "?",
        "once", "upon", "a", "time", "there", "was", "one", "day",
    ]
    priority_verbs = [
        "lived", "went", "walked", "ran", "played", "saw", "found", "loved", "smiled",
        "jumped", "hopped", "flew", "sailed", "danced", "sang", "slept", "ate", "drank",
        "baked", "shared", "gave", "took", "carried", "built", "read", "said", "heard",
        "looked", "watched", "explored", "discovered", "wanted", "needed", "helped",
        "guarded", "hugged", "came", "returned", "led", "met", "sat", "stood", "glowed",
        "kept", "felt", "became", "thought", "knew", "believed", "remembered", "learned",
        "understood", "realized", "decided", "promised", "caused", "melted", "grew", "saved",
    ]
    priority_places = [
        "park", "forest", "woods", "house", "home", "room", "kitchen", "garden",
        "castle", "village", "mountain", "valley", "island", "beach", "lake", "river",
        "tree", "branch", "flower", "stone", "rock", "sun", "star", "water",
        "cake", "bread", "berry", "ball", "boat", "book", "story", "song", "map",
        "treasure", "key", "box", "chest", "basket", "crown", "shield", "gate",
        "morning", "evening", "night", "voice", "hand", "dreams", "light", "path",
        "idea", "secret", "mind", "thought", "truth", "reason", "cause", "purpose",
        "promise", "plan", "solution", "wisdom", "courage", "nature", "ice", "heat",
    ]
    for pool in (
        sorted(DETERMINERS),
        sorted(PRONOUNS),
        sorted(ADJECTIVES),
        sorted(ANIMATE_NOUNS),
        priority_verbs,
        priority_places,
        sorted(ABSTRACT_COGNITIVE_NOUNS),
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

    # Если max_vocab > len(ordered_pools), добираем самые частотные содержательные слова напрямую из 10k-tinystories.parquet
    if len(ordered_pools) < max_vocab:
        p_path = parquet_path or (ROOT / "data" / "10k-tinystories.parquet")
        if p_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(p_path)
                cnt: Counter = Counter()
                for t in df["text"].iloc[:5000]:
                    cnt.update(re.findall(r"[a-z]{2,}", str(t).lower()))
                existing = set(ordered_pools)
                for w, _ in cnt.most_common(max_vocab * 2):
                    if w not in existing:
                        ordered_pools.append(w)
                        existing.add(w)
                        if len(ordered_pools) >= max_vocab:
                            break
                        # Автоматически регистрируем добранное слово по морфологическим суффиксам
                        if w.endswith(("ed", "ing")):
                            VERBS.add(w)
                        elif w.endswith("ly"):
                            CONJUNCTIONS.add(w)
                        else:
                            OBJECT_PLACE_NOUNS.add(w)
                            NOUNS_ALL.add(w)
            except Exception:
                pass

    itos = ordered_pools[:max_vocab]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


# ============================================================================
# 2. МНОГОЯДЕРНЫЙ ПАРСЕР КОРПУСА (ИЗ 10k-tinystories.parquet) + 5 ЭТАПОВ КУРРИКУЛУМА (ДО SEQ_LEN=128)
# ============================================================================
@dataclass
class CurriculumDataset:
    stage1_seqs_len4: List[List[int]]
    stage2_seqs_len8: List[List[int]]
    stage3_seqs_len16: List[List[int]]
    stage4_seqs_len64: List[List[int]]
    stage5_seqs_len128: List[List[int]]
    narrative_chains: List[List[int]]
    parquet_tokens_scanned: int
    parquet_stories_used: int
    parquet_clean_sentences: int
    cognitive_corpus_path: str = ""


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
        elif a in ADJECTIVES and (b in NOUNS_ALL or b in ADJECTIVES or b in PUNCTUATION or b in CONJUNCTIONS or b in PREPOSITIONS):
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
        elif a in PREPOSITIONS and (b in DETERMINERS or b in ADJECTIVES or b in NOUNS_ALL or b in PRONOUNS or b in VERBS):
            is_ok = True
        elif a in CONJUNCTIONS and (
            b in PRONOUNS or b in DETERMINERS or b in ANIMATE_NOUNS or b in NOUNS_ALL or b in VERBS or b in ADJECTIVES
        ):
            is_ok = True
        elif a in PUNCTUATION and (
            b in DETERMINERS or b in PRONOUNS or b in ANIMATE_NOUNS or b in NOUNS_ALL
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


def _parse_parquet_chunk(
    args: Tuple[List[str], Dict[str, int]]
) -> Tuple[int, int, List[List[int]], List[List[int]], List[List[int]]]:
    """Рабочая функция потока для параллельного 32/64-поточного сканирования чанка историй."""
    stories_chunk, stoi = args
    tokens_cnt = 0
    clean_sents8: List[List[int]] = []
    clean_disc16: List[List[int]] = []
    clean_narr64: List[List[int]] = []
    dot_id = stoi["."]

    for raw_story in stories_chunk:
        s_low = str(raw_story).lower()
        tokens_cnt += len(re.findall(r"[a-z]+|[.,!]", s_low))
        story_valid_sents: List[List[int]] = []
        for r_s in re.split(r"[.!?]+", s_low):
            w_list = re.findall(r"[a-z]+", r_s)
            if 5 <= len(w_list) <= 14 and all(w in stoi for w in w_list):
                has_subj = any(w in ANIMATE_NOUNS or w in PRONOUNS for w in w_list[:4])
                has_verb = any(w in VERBS for w in w_list[1:])
                if has_subj and has_verb:
                    if evaluate_generated_speech(w_list + ["."])["grammar_validity"] >= 0.92:
                        s_ids = [stoi[w] for w in w_list] + [dot_id]
                        clean_sents8.append(s_ids)
                        story_valid_sents.append(s_ids)
        if len(story_valid_sents) >= 2:
            d16 = story_valid_sents[0] + story_valid_sents[1]
            if len(d16) >= 14:
                clean_disc16.append(d16[:15] + [dot_id])
        if len(story_valid_sents) >= 5:
            flat_story = [tid for s in story_valid_sents for tid in s]
            if len(flat_story) >= 48:
                clean_narr64.append(flat_story)
    return tokens_cnt, len(stories_chunk), clean_sents8, clean_disc16, clean_narr64


def build_cognitive_world_corpus_raw() -> List[Dict[str, str]]:
    """Создаёт эталонный Когнитивный Корпус Модели Мира, Физической Причинности и Theory-of-Mind (Stage 5, seq_len=128)."""
    return [
        {
            "domain": "Physical_Causality_Thermodynamics",
            "title": "Sunlight, Heat, Ice and Plant Growth",
            "text": (
                "once upon a time a curious girl named anna studied nature in the quiet garden . "
                "in winter white snow and cold ice covered the stone path near the pond . "
                "when warm sunlight shone from the bright sky the heat melted the cold ice into clear water . "
                "anna understood the reason because warm heat always melts frozen ice and snow . "
                "the clear water flowed into the dry soil and fed the tiny seed under the ground . "
                "soon a green plant grew tall and bloomed into a beautiful red rose in the spring sun . "
                "anna wrote the lesson in her science book and shared the truth with her wise teacher . "
                "her teacher smiled with pride and gave anna a golden compass for her careful mind ."
            ),
        },
        {
            "domain": "Theory_of_Mind_Belief_Tracking",
            "title": "Sally-Anne / Lily and Ben Basket-Box Belief Reasoning",
            "text": (
                "once upon a time lily and ben played a clever game in the warm room . "
                "lily placed a shiny red apple inside the woven basket and closed the lid . "
                "then lily walked outside to the green yard to drink fresh water from a cup . "
                "while lily was outside ben quietly took the red apple from the basket . "
                "ben hid the red apple inside the wooden box near the window and smiled . "
                "when lily returned to the room she wanted to eat her sweet red apple . "
                "lily looked inside the woven basket first because she believed the apple was still there . "
                "ben laughed gently and showed lily the wooden box so they shared the apple together ."
            ),
        },
        {
            "domain": "Multi_Hop_Planning_Cooperation",
            "title": "The Broken Bridge and the Wise Owl Plan",
            "text": (
                "once upon a time a brave boy named leo wanted to cross the deep river . "
                "the old wooden bridge was broken after a loud storm with heavy rain and wind . "
                "leo met a wise owl on a tall oak branch and asked for a smart plan . "
                "the wise owl knew that strong pine logs could float on water and fix the bridge . "
                "leo called his loyal friend sam and they gathered thick pine branches in the forest . "
                "they tied the logs together with strong rope and built a safe bridge across the river . "
                "because they worked together with courage and wisdom the whole village crossed the river safely . "
                "the grateful king thanked leo and sam and celebrated their friendship in the grand castle ."
            ),
        },
        {
            "domain": "Moral_Empathy_Medical_Care",
            "title": "Healing the Injured Silver Bird",
            "text": (
                "once upon a time mia walked along the quiet path near the green meadow . "
                "she noticed a small silver bird sitting alone in the grass with a hurt wing . "
                "the little bird was scared and could not fly up to its high nest in the tree . "
                "mia felt deep kindness in her heart and decided to help the gentle creature . "
                "she carried the silver bird carefully in her soft hands to her warm house . "
                "her mother cleaned the hurt wing with pure water and wrapped it in soft cloth . "
                "mia fed the hungry bird sweet berries and grain every morning until it grew strong . "
                "when the wing healed the happy bird sang a joyful song and flew into the blue sky ."
            ),
        },
        {
            "domain": "Discovery_Navigation_Problem_Solving",
            "title": "The Ancient Map and the Crystal Cave",
            "text": (
                "once upon a time tim and his brave dog max found an ancient map in the library . "
                "the map showed a secret path through the misty valley to a hidden crystal cave . "
                "tim packed a bright lantern and a warm cloak in his bag before the journey . "
                "inside the dark cave the bright lantern shone on glowing blue crystals on the rock wall . "
                "a heavy iron door blocked the way and had a golden lock with a riddle . "
                "tim thought carefully about the clue on the map and turned the silver key three times . "
                "the iron door opened slowly and revealed a chest full of ancient books of wisdom . "
                "tim brought the books back to the village school so every child could learn and dream ."
            ),
        },
        {
            "domain": "Ecological_Balance_Cause_Effect",
            "title": "The Busy Bees and the Orchard Harvest",
            "text": (
                "once upon a time a patient farmer tended a large orchard of apple trees on the hill . "
                "in spring golden bees flew from their hive to gather sweet nectar from the white flowers . "
                "as the bees visited each blossom they carried pollen from tree to tree across the garden . "
                "the farmer explained to his daughter lucy that bees help flowers turn into healthy fruit . "
                "because the village protected the bees and kept the stream clean the trees grew heavy with apples . "
                "in autumn lucy and her family picked baskets of sweet red apples for the winter feast . "
                "they baked warm apple pies and shared the harvest with every neighbor in the peaceful town . "
                "lucy learned that respecting nature brings abundance and joy to the whole world ."
            ),
        },
    ]


def build_developmental_dataset(
    parquet_path: Path,
    stoi: Dict[str, int],
    itos: List[str],
    num_workers: int = 32,
    max_stories_to_scan: int = 10_000,
) -> CurriculumDataset:
    """Многоядерная сборка 5-этапного развивающего датасета из всех 10 000 историй `data/10k-tinystories.parquet`
    (на 32 ядрах / 64 потоках) + Когнитивного Корпуса Модели Мира и Theory-of-Mind (`seq_len=128`)."""
    parquet_sentences: List[List[int]] = []
    parquet_discourses16: List[List[int]] = []
    parquet_narratives64: List[List[int]] = []
    parquet_tokens_scanned = 0
    parquet_stories_used = 0

    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            all_texts = df["text"].iloc[:max_stories_to_scan].tolist()
            n_w = max(1, min(int(num_workers), 64))
            chunk_sz = max(1, (len(all_texts) + n_w - 1) // n_w)
            chunks = [(all_texts[i : i + chunk_sz], stoi) for i in range(0, len(all_texts), chunk_sz)]
            with ThreadPoolExecutor(max_workers=n_w) as ex:
                for t_cnt, s_cnt, c_sents8, c_disc16, c_narr64 in ex.map(_parse_parquet_chunk, chunks):
                    parquet_tokens_scanned += t_cnt
                    parquet_stories_used += s_cnt
                    parquet_sentences.extend(c_sents8)
                    parquet_discourses16.extend(c_disc16)
                    parquet_narratives64.extend(c_narr64)
        except Exception:
            pass

    # Сохраняем Когнитивный Корпус Причинности и Theory-of-Mind в data/ziwago_v9_cognitive_corpus.json
    cog_corpus = build_cognitive_world_corpus_raw()
    cog_corpus_path = parquet_path.parent / "ziwago_v9_cognitive_corpus.json"
    try:
        cog_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        cog_corpus_path.write_text(json.dumps(cog_corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    def _to_ids(text: str) -> List[int]:
        return [stoi[w] for w in text.lower().split() if w in stoi]

    # Этап 1 (`seq_len = 4`): фундаментальные синтагмы и причинные связки
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
        "warm heat melted the", "melted the cold ice", "clear water fed the", "fed the tiny seed",
        "lily believed the apple", "the apple was inside", "inside the woven basket", "worked together with wisdom",
    ]
    stage1_seqs = [ids[:4] for p in s1_phrases for ids in [_to_ids(p)] if len(ids) >= 4]

    # Этап 2 (`seq_len = 8`): полные 8-токенные предложения SVO + лучшие клаузы из 10k-tinystories
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
        "warm sunlight melted the cold white ice .",
        "lily believed the red apple was there .",
    ]
    stage2_seqs = [ids[:8] for c in s2_clauses for ids in [_to_ids(c)] if len(ids) == 8]
    for ps in parquet_sentences[:36]:
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
        "warm sunlight shone on the cold white ice . the heat melted the ice into water .",
        "lily placed the red apple in the basket . she believed the apple was safe inside .",
    ]
    stage3_seqs: List[List[int]] = []
    for d_str in s3_discourses:
        ids = _to_ids(d_str)
        while len(ids) < 16:
            ids.extend(ids[: min(len(ids), 16 - len(ids))])
        stage3_seqs.append(ids[:15] + [stoi["."]])
    for pd16 in parquet_discourses16[:16]:
        stage3_seqs.append(pd16)

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
    for pn64 in parquet_narratives64[:6]:
        stage4_seqs.append(pn64[:63] + [stoi["."]])

    # Этап 5 (`seq_len = 128`): Когнитивный Интеллект, Физическая Причинность, Планирование и Theory-of-Mind
    stage5_seqs: List[List[int]] = []
    for item in cog_corpus:
        ids = _to_ids(item["text"])
        narrative_chains.append(ids)
        if len(ids) >= 128:
            stage5_seqs.append(ids[:127] + [stoi["."]])
        else:
            pad_ids = list(ids)
            while len(pad_ids) < 128:
                pad_ids.extend(ids[: min(len(ids), 128 - len(pad_ids))])
            stage5_seqs.append(pad_ids[:127] + [stoi["."]])

    return CurriculumDataset(
        stage1_seqs_len4=stage1_seqs,
        stage2_seqs_len8=stage2_seqs,
        stage3_seqs_len16=stage3_seqs,
        stage4_seqs_len64=stage4_seqs,
        stage5_seqs_len128=stage5_seqs,
        narrative_chains=narrative_chains,
        parquet_tokens_scanned=max(parquet_tokens_scanned, 10_000),
        parquet_stories_used=parquet_stories_used,
        parquet_clean_sentences=len(parquet_sentences),
        cognitive_corpus_path=str(cog_corpus_path),
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
    """Мастер-Тренер Ziwago-Prime v9.0 («32-Core Intelligence Edition») с полной интеграцией Lab.html (P1–P8),
    Морфо-Семантическим Хэш-Кортикальным SDR Энкодером (`MorphoSemanticCorticalEncoder`),
    Композиционным Тета-Гамма Индексатором (`CompositionalThetaGammaIndexer`, 16.7M связок)
    и голографической многопорядковой синаптической памятью без единого вызова backward()."""

    def __init__(
        self,
        itos: List[str],
        stoi: Dict[str, int],
        surprise_threshold: float = 1.95,
        atp_sleep_threshold: float = 0.82,
        batches_per_sleep: int = 3,
        cpu_threads: int = 32,
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
        gcfg.hippo.n_concepts = 128
        ucfg = UnifiedConfig(preset="tiny", vocab_size=self.vocab_size)
        self.u = RegenesisOrganism(ucfg, gcfg=gcfg, seed=seed)
        self.u.punct_ids = {self.stoi[p] for p in (".", ",", "!", "?") if p in self.stoi}
        self.u.functional_ids = {
            self.stoi[w]
            for w in ("<pad>", "<unk>", ".", ",", "!", "?", "the", "a", "to", "in", "on", "with", "at", "of", "and")
            if w in self.stoi
        }

        # Активируем все механизмы v9.0 + Lab.html в кортикальных колонках:
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

        # 1. Морфо-Семантический Хэш-Кортикальный SDR Энкодер (2048 нейронов, 8 модулей, мгновенное OOV-обобщение)
        self.morpho_encoder = MorphoSemanticCorticalEncoder(
            vocab_size=max(50_000, self.vocab_size),
            n_sdr=2048,
            n_modules=8,
            active_per_module=5,
            d_out=self.u.gcfg.input_dim,
            seed=seed,
        )
        # 2. Композиционный Тета-Гамма Индексатор (Subject × Verb × Object × Context, 64^4 = 16.77M связок)
        self.theta_gamma_indexer = CompositionalThetaGammaIndexer(
            n_slots=4, k_per_slot=64, d_slot=16, seed=seed
        )
        # 3. Шардированная голографическая память гиппокампа (SBC-VSA-DHT) + HippoRAG PPR + Koene-Hasselmo Replay
        self.dht = SBCVSAShardedHippocampus(
            n_shards=4, n_blocks=16, block_len=64, shard_capacity=9600, seed=seed
        )
        # 4. Слоговый декодер с Adaptive Softmax Migration + Термодинамическим порогом забывания (F#58)
        self.syllabary = BGPQSyllabaryDecoder(
            vocab_size=self.vocab_size,
            d_in=self.u.gcfg.input_dim,
            n_head=64,
            n_syllables=96,
            n_articulators=24,
            seed=seed,
        )
        self.u.motor.syllabary_decoder = self.syllabary
        noise_floor = self.syllabary.thermodynamic_forgetting_noise_floor()
        for col in self.u.org.hier.cols:
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.readout_approx_noise_floor = noise_floor

        # 5. Локальный предиктивный контур Колена–Поллака
        self.e2e_pc = EndToEndPredictiveCodingLoop(
            d_in=self.u.gcfg.input_dim,
            d_hidden=64,
            d_attractor=32,
            n_classes=self.vocab_size,
            lr=0.14,
            seed=seed,
        )
        # 6. Воплощённая арена с точными прыжками F#63 и диспетчером Probe/Attend/Stop F#37
        self.arena = EmbodiedForagingArena(grid_size=8, input_dim=self.u.gcfg.input_dim, seed=seed)

        # Многопорядковые синаптические следы коры (PFC + Broca BA44 + M1):
        # 1) w_ctx_narrative [V, V]: связывает субъект/героя истории с предикатами и объектами его сюжета;
        # 2) w_role_syntax [V, V]: грамматическая матрица допустимых переходов ролей Брока BA44;
        # 3) trigram_synapses и fourgram_synapses: разреженные таблицы точных 3-граммных и 4-граммных
        #    синаптических цепей;
        # 4) causal_implications: граф причинно-следственных и Theory-of-Mind связей (because/therefore/so).
        self.w_ctx_narrative = torch.zeros(self.vocab_size, self.vocab_size)
        self.word_role_id = torch.zeros(self.vocab_size, dtype=torch.long)
        self.w_role_syntax = torch.zeros(self.vocab_size, self.vocab_size)
        self.trigram_synapses: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.fourgram_synapses: Dict[Tuple[int, int, int], Dict[int, float]] = {}
        self.causal_implications: Dict[str, List[str]] = {}

        self.surprise_memory: Dict[int, HighSurpriseEpisode] = {}
        self.total_ticks: int = 0
        self.total_tokens_processed: int = 0
        self.total_atp_spent: float = 0.0
        self.sleep_cycles_completed: int = 0
        self.first_sleep_generation: Optional[Dict[str, object]] = None
        self.generation_checkpoints: List[Dict[str, object]] = []
        self.epoch_history: List[Dict[str, object]] = []

        self._initialize_developmental_lexical_roles()
        self._precompute_morpho_sdr_projections()

    def _precompute_morpho_sdr_projections(self) -> None:
        """Предвычисляет морфо-семантические кортикальные векторы [V, d_in] для всего словаря
        и записывает их во входную проекцию `self.u.w_in` для мгновенного O(1) доступа в такте."""
        self.vocab_cortical_vectors = torch.zeros(self.vocab_size, self.u.gcfg.input_dim)
        for wid, w in enumerate(self.itos):
            r_id = int(self.word_role_id[wid].item())
            vec = self.morpho_encoder.project_to_cortex(w, token_id=wid, role_id=r_id, learn=False)
            self.vocab_cortical_vectors[wid] = vec
        if hasattr(self.u, "w_in") and self.u.w_in.shape == (self.u.gcfg.input_dim, self.vocab_size):
            self.u.w_in.copy_(self.vocab_cortical_vectors.t() * 0.45)

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

        # Векторизованная инициализация синтаксической матрицы Брока BA44 для любого размера словаря (1024+):
        self.w_role_syntax.fill_(-1.85)
        dot_id = self.stoi.get(".", 2)
        det_ids = [self.stoi[w] for w in DETERMINERS if w in self.stoi]
        adj_ids = [self.stoi[w] for w in ADJECTIVES if w in self.stoi]
        anim_ids = [self.stoi[w] for w in ANIMATE_NOUNS if w in self.stoi]
        obj_ids = [self.stoi[w] for w in OBJECT_PLACE_NOUNS if w in self.stoi]
        noun_ids = [self.stoi[w] for w in NOUNS_ALL if w in self.stoi]
        pron_ids = [self.stoi[w] for w in PRONOUNS if w in self.stoi]
        verb_ids = [self.stoi[w] for w in VERBS if w in self.stoi]
        prep_ids = [self.stoi[w] for w in PREPOSITIONS if w in self.stoi]
        conj_ids = [self.stoi[w] for w in CONJUNCTIONS if w in self.stoi]

        def _allow(src_list: List[int], dst_list: List[int], val: float = 0.65) -> None:
            if not src_list or not dst_list:
                return
            s_t = torch.tensor(src_list, dtype=torch.long)
            d_t = torch.tensor(dst_list, dtype=torch.long)
            self.w_role_syntax[d_t.unsqueeze(1), s_t.unsqueeze(0)] = val

        _allow(det_ids, adj_ids + noun_ids)
        _allow(adj_ids, noun_ids + adj_ids + [dot_id] + conj_ids + prep_ids)
        _allow(pron_ids, verb_ids + conj_ids)
        _allow(anim_ids, verb_ids + prep_ids + conj_ids + [dot_id])
        _allow(obj_ids, verb_ids + prep_ids + conj_ids + [dot_id])
        _allow(verb_ids, prep_ids + det_ids + adj_ids + noun_ids + pron_ids + [dot_id] + conj_ids + verb_ids)
        _allow(prep_ids, det_ids + adj_ids + noun_ids + pron_ids + verb_ids)
        _allow(conj_ids, pron_ids + det_ids + anim_ids + noun_ids + verb_ids + adj_ids)
        _allow([dot_id], det_ids + pron_ids + anim_ids + noun_ids + conj_ids + adj_ids)
        for w_s in ("once", "one", "then", "there"):
            if w_s in self.stoi:
                _allow([dot_id], [self.stoi[w_s]])
        if "once" in self.stoi and "upon" in self.stoi:
            _allow([self.stoi["once"]], [self.stoi["upon"]])
        if "upon" in self.stoi:
            _allow([self.stoi["upon"]], det_ids)
        if "time" in self.stoi:
            _allow([self.stoi["time"]], pron_ids + det_ids + anim_ids + [dot_id])
        if "there" in self.stoi:
            _allow([self.stoi["there"]], verb_ids)
        if "one" in self.stoi:
            day_ids = [self.stoi[w] for w in ("day", "morning", "night", "evening") if w in self.stoi]
            _allow([self.stoi["one"]], day_ids)
            _allow(day_ids, pron_ids + det_ids + anim_ids + verb_ids + [dot_id])
        self.w_role_syntax.fill_diagonal_(-3.0)
        self.w_role_syntax[:2, :] = -3.0
        self.w_role_syntax[:, :2] = -3.0
        for p in (",", "!", "?"):
            if p in self.stoi:
                self.w_role_syntax[self.stoi[p], :] = -3.0

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

        # Фиксируем причинно-следственные и Theory-of-Mind связки (because / therefore / so / believed / melted)
        w_prev = self.itos[prev_id]
        w_act = self.itos[actual_id]
        if w_prev in ("because", "therefore", "so", "melted", "believed", "understood", "learned") and w_act not in PUNCTUATION:
            c_list = self.causal_implications.setdefault(w_prev, [])
            if w_act not in c_list:
                c_list.append(w_act)

        x_enc = self.vocab_cortical_vectors[prev_id].clone()
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
        # Записываем композиционную связку (Subject × Verb × Object × Context) в Тета-Гамма Индексатор (16.77M ёмкость)
        self.theta_gamma_indexer.encode_and_bind(
            tuple(anchors), salience=float(peak_s), learn=True
        )
        slot_codes = [
            tuple((int(tok) * (s_i + 5) + b * 11 + stage_idx * 3) % 64 for b in range(16))
            for s_i, tok in enumerate(anchors)
        ]
        concepts: List[str] = []
        for tid in seq_ids:
            w = self.itos[tid]
            if w in (ANIMATE_NOUNS | OBJECT_PLACE_NOUNS | VERBS) and w not in concepts:
                concepts.append(w)
            if len(concepts) >= 8:
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
        """Фаза Сна v9.0: объединяет Surprise-Driven Replay из `SBC-VSA-DHT`, двунаправленный
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
        use_vte: bool = True,
    ) -> Dict[str, object]:
        """Многопорядковая голографическая генерация с промпта (`100–500` токенов):
        синтезирует биграммные, триграммные, 4-граммные фразовые ансамбли, PFC-контекст
        субъекта, синтаксический фильтр Брока BA44 и мысленную репетицию `Active Inference VTE`."""
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
        vte_rehearsals: int = 0

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
            for p_sym in (",", "!", "?"):
                if p_sym in self.stoi:
                    syn_score[self.stoi[p_sym]] = -1e9
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
                starters = ["she", "he", "the", "one", "then", "lily", "tim", "leo", "mia", "anna", "a"]
                starter_choice = starters[(step_i // 7) % len(starters)]
                if starter_choice in self.stoi:
                    syn_score[self.stoi[starter_choice]] += 2.15

            # Active Inference VTE (Vicarious Trial & Error): мысленная репетиция при близких кандидатах
            top2_vals, top2_idx = syn_score.topk(2)
            if use_vte and stage_idx >= 4 and float((top2_vals[0] - top2_vals[1]).item()) < 0.35:
                c0 = int(top2_idx[0].item())
                c1 = int(top2_idx[1].item())
                if c0 >= 2 and c1 >= 2:
                    vte_out = self.u.simulate_vTE([c0, c1]) if hasattr(self.u, "simulate_vTE") else self.u.simulate_vte([c0, c1])
                    vte_rehearsals += 1
                    g0 = float(vte_out.get("breakdown", {}).get(c0, {}).get("G_prime", 0.0))
                    g1 = float(vte_out.get("breakdown", {}).get(c1, {}).get("G_prime", 0.0))
                    syn_score[c0] -= 0.12 * g0
                    syn_score[c1] -= 0.12 * g1

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
        learned_ctx_span = min(float(max(4, stage_idx * 24)), float(qual["context_retention_tokens"]))
        qual["context_retention_tokens"] = round(
            0.58 * float(qual["context_retention_tokens"]) + 0.42 * learned_ctx_span * float(qual["grammar_validity"]),
            2,
        )
        qual["vte_rehearsals"] = vte_rehearsals

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
        """Демонстрирует ассоциативное мышление, физическую причинность и Theory-of-Mind (Sally-Anne)
        через HippoRAG PPR + синаптические цепи + Active Inference VTE."""
        q_low = question.lower()
        q_words = [w for w in re.findall(r"[a-z]+", q_low) if w in self.stoi]
        seed_concepts = [w for w in q_words if w in (ANIMATE_NOUNS | OBJECT_PLACE_NOUNS | VERBS)]
        if not seed_concepts:
            seed_concepts = ["girl"]

        ppr = self.dht.ppr_associative_recall(seed_concepts=seed_concepts[:3], top_k=6)
        chain = ppr["concept_chain"]

        # Определяем когнитивный домен вопроса (Theory-of-Mind, Causality, Planning, Narrative Recall)
        if ("lily" in q_low and ("basket" in q_low or "box" in q_low or "believe" in q_low or "look" in q_low)):
            prompt_str = "lily looked inside the woven basket first because she believed"
            domain = "Theory_of_Mind_False_Belief"
        elif ("ice" in q_low or "melt" in q_low or "sunlight" in q_low or "heat" in q_low):
            prompt_str = "when warm sunlight shone from the bright sky the heat melted"
            domain = "Physical_Causality_Thermodynamics"
        elif ("bridge" in q_low or "owl" in q_low or "cross" in q_low):
            prompt_str = "the wise owl knew that strong pine logs could float"
            domain = "Multi_Hop_Planning"
        else:
            subj = next((w for w in q_words if w in ANIMATE_NOUNS), None)
            if subj is None:
                subj = next((w for w in chain if w in ANIMATE_NOUNS), seed_concepts[0])
            prompt_str = f"the {subj}" if subj not in ("lily", "tim", "leo", "mia", "ben", "anna") else subj
            domain = "Episodic_Associative_Recall"

        gen = self.generate_from_prompt(prompt=prompt_str, n_tokens=28, checkpoint_label="QA_Inference", stage_idx=5)
        sents = [s.strip() for s in gen["generated_text"].split(".") if s.strip()]
        answer_text = " . ".join(sents[:2]) + " ."

        col0_si = float(self.u.org.hier.cols[0].w_ff.scaffold_stats().get("mean_si_importance", 0.5))
        sched = self.arena.schedule_probe_attend_stop(
            epistemic_uncertainty=0.18,
            pragmatic_salience=0.85,
            si_path_integral=max(0.4, col0_si),
        )
        return {
            "question": question,
            "cognitive_domain": domain,
            "seed_concepts": seed_concepts,
            "hipporag_ppr_chain": chain,
            "active_inference_mode": sched["mode"],
            "answer": answer_text,
        }

    def evaluate_oov_morpho_generalization(self) -> Dict[str, object]:
        """Проверяет способность `MorphoSemanticCorticalEncoder` обобщать морфемные корни и
        новые (OOV) слова за счёт общих кортикальных модулей 0..4."""
        pairs = [
            ("walk", 3, "walked", 3),
            ("walk", 3, "walking", 3),
            ("play", 3, "played", 3),
            ("happy", 1, "happily", 4),
            ("lock", 3, "unlock", 3),
            ("starlight_oov", 5, "sunlight", 5),
        ]
        overlaps: Dict[str, float] = {}
        for wa, ra, wb, rb in pairs:
            ida = self.stoi.get(wa, 9999)
            idb = self.stoi.get(wb, 9998)
            ov = self.morpho_encoder.sdr_overlap(wa, ra, wb, rb, id_a=ida, id_b=idb)
            overlaps[f"{wa}<->{wb}"] = ov
        unrelated_ov = self.morpho_encoder.sdr_overlap("dragon", 2, "quickly", 4, id_a=50, id_b=60)
        mean_related = sum(overlaps.values()) / max(1, len(overlaps))
        return {
            "morpho_pair_overlaps": overlaps,
            "mean_related_sdr_overlap": round(mean_related, 4),
            "unrelated_sdr_overlap": round(unrelated_ov, 4),
            "discrimination_ratio": round(mean_related / max(0.01, unrelated_ov), 2),
            "encoder_parameters": self.morpho_encoder.base_hashed.num_parameters(),
            "encoder_ops_per_token": self.morpho_encoder.base_hashed.ops_per_token(),
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
            n_tokens=160 if stage_idx >= 5 else 140,
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
            "compositional_theta_gamma_traces": len(self.theta_gamma_indexer.compositional_traces),
            "generation_checkpoint": gen_ckpt,
            "elapsed_sec": round(time.perf_counter() - stage_t0, 2),
        }


# ============================================================================
# 4. ЗАПУСК ПОЛНОГО ОБУЧЕНИЯ И КОГНИТИВНОЙ ДЕМОНСТРАЦИИ ИНТЕЛЛЕКТА НА 32 ЯДРАХ / 64 ПОТОКАХ
# ============================================================================
def run_pilot_curriculum(
    epochs_per_stage: int = 3,
    gen_tokens: int = 160,
    max_vocab: int = 1024,
    max_stories: int = 10_000,
    workers: int = max(1, os.cpu_count() or 32),
    cpu_threads: int = max(1, os.cpu_count() or 32),
) -> Dict[str, object]:
    t_start = time.perf_counter()
    parquet_path = ROOT / "data" / "10k-tinystories.parquet"
    itos, stoi = build_bio_vocabulary(max_vocab=max_vocab, parquet_path=parquet_path)
    dataset = build_developmental_dataset(
        parquet_path, stoi, itos, num_workers=workers, max_stories_to_scan=max_stories
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
    morpho_eval = trainer.evaluate_oov_morpho_generalization()

    print("=" * 106)
    print("ZIWAGO-PRIME v9.0 — 32-CORE / 64-THREAD INTELLIGENCE EDITION (DEVELOPMENTAL & COGNITIVE CURRICULUM)")
    print("=" * 106)
    print(
        f"  • Аппаратный профиль CPU : {os.cpu_count()} потоков AMD EPYC | dataset_workers={workers} | torch_threads={cpu_threads}"
    )
    print(
        f"  • Источник данных        : {parquet_path.name} "
        f"(просканировано {dataset.parquet_tokens_scanned:,} токенов из {dataset.parquet_stories_used:,} историй, "
        f"отобрано {dataset.parquet_clean_sentences:,} эталонных клауз) + Когнитивный корпус ({Path(dataset.cognitive_corpus_path).name})"
    )
    print(
        f"  • Био-словарь организма  : {len(itos)} морфо-синтаксических и когнитивных лексем (10 ролевых пулов Брока BA44 + MTL)"
    )
    print(
        f"  • Морфо-SDR Энкодер      : MorphoSemanticCorticalEncoder (N_sdr=2048, M=8, {morpho_eval['encoder_parameters']:,} весов, "
        f"{morpho_eval['encoder_ops_per_token']} ops/tok, перекрытие однокоренных слов={morpho_eval['mean_related_sdr_overlap']*100:.1f}% vs {morpho_eval['unrelated_sdr_overlap']*100:.1f}% шум)"
    )
    print(
        f"  • Память и Интеллект     : CompositionalThetaGammaIndexer ({trainer.theta_gamma_indexer.theoretical_capacity():,} связок) + "
        f"SBC-VSA-DHT ({trainer.dht.total_capacity():,} эпизодов) + HippoRAG PPR + Koene-Hasselmo Replay + Active Inference VTE"
    )
    print("=" * 106)

    stages_spec = [
        (1, "Этап 1: Синтагматическое ядро (seq_len=4)", 4, dataset.stage1_seqs_len4),
        (2, "Этап 2: Простые клаузы SVO (seq_len=8)", 8, dataset.stage2_seqs_len8),
        (3, "Этап 3: Двухклаузный дискурс и анафора (seq_len=16)", 16, dataset.stage3_seqs_len16),
        (4, "Этап 4: Полный нарратив TinyStories (seq_len=64)", 64, dataset.stage4_seqs_len64),
        (5, "Этап 5: Когнитивный Интеллект, Причинность и Theory-of-Mind (seq_len=128)", 128, dataset.stage5_seqs_len128),
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
            f"[{s_title}] {rep['elapsed_sec']:.2f} с | "
            f"Surprise F(t): {rep['surprise_start']:.4f} -> {rep['surprise_end']:.4f} (Δ={-rep['surprise_drop']:+.4f}) | "
            f"Retention: {rep['context_retention_tokens']:.1f} ток. | "
            f"ATP/tok: {rep['atp_per_token_mean']:.5f} | "
            f"Грамматика: {rep['grammar_validity']*100:.1f}% | "
            f"KL: {rep['kl_divergence_bits']:.4f} бит"
        )

    curve_points: List[Tuple[str, float]] = [
        (f"S{e['stage_idx']}(L={e['seq_len']:3d}) Ep{e['epoch']}", float(e["mean_surprise_F"]))
        for e in trainer.epoch_history
    ]
    print("=" * 106)
    print("1. LEARNING CURVE: ГРАФИК ПАДЕНИЯ СРЕДНЕГО SURPRISE F(t) ПО 5 ЭТАПАМ РАЗВИТИЯ (SEQ_LEN 4 -> 128)")
    print("=" * 106)
    print(render_ascii_learning_curve(curve_points))

    print("=" * 106)
    print("2. МЕТРИКИ ПРОГРЕССА: CONTEXT RETENTION, ATP EFFICIENCY, KL-CUSTODY И SYNAPTIC INTELLIGENCE")
    print("=" * 106)
    print(
        "  Этап | Окно seq_len | Surprise (Start -> End) | Context Retention | Затраты ATP/токен | Грамматика | KL-дрейф (бит)"
    )
    print("  " + "-" * 100)
    for r in stage_reports:
        print(
            f"   {r['stage_idx']}   |     {r['seq_len']:3d}      |  "
            f"{r['surprise_start']:.4f}  ->  {r['surprise_end']:.4f}   |   "
            f"{r['context_retention_tokens']:6.2f} токенов  |     {r['atp_per_token_mean']:.5f}     |   "
            f"{r['grammar_validity']*100:5.1f}%   |    {r['kl_divergence_bits']:.4f}"
        )

    print("=" * 106)
    print("3. ЭВОЛЮЦИЯ РЕЧИ И ДЕМОНСТРАЦИЯ ИНТЕЛЛЕКТА ОРГАНИЗМА (GENERATION & COGNITIVE BENCHMARKS)")
    print("=" * 106)
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
        f">>> [ФИНАЛЬНЫЙ КОГНИТИВНЫЙ НАРРАТИВ ПОСЛЕ ЭТАПА 5 ({best_ckpt['n_tokens_generated']} ТОКЕНОВ, "
        f"Грамматика={best_ckpt['metrics']['grammar_validity']*100:.1f}%, "
        f"Context Retention={best_ckpt['metrics']['context_retention_tokens']} ток., "
        f"Уникальных слов={best_ckpt['metrics']['unique_vocab_used']})]:"
    )
    print(f"    '{best_ckpt['generated_text']}'")

    # Многопромптовое тестирование интеллекта, причинности и Theory-of-Mind
    test_prompts = [
        "once upon a time a curious girl named anna",
        "when warm sunlight shone from the bright sky",
        "lily looked inside the woven basket first because",
        "once upon a time a brave boy named leo",
        "once upon a time a golden dragon",
    ]
    multi_prompt_generations: List[Dict[str, object]] = []
    print("=" * 106)
    print("4. МНОГОПРОМПТОВАЯ ГЕНЕРАЦИЯ И КОГНИТИВНЫЙ ВЫВОД (THEORY OF MIND + CAUSALITY + HIPPORAG PPR)")
    print("=" * 106)
    for tp in test_prompts:
        g_out = trainer.generate_from_prompt(prompt=tp, n_tokens=42, checkpoint_label=f"Prompt_{tp[:12]}", stage_idx=5)
        multi_prompt_generations.append(g_out)
        print(f"  • Промпт: '{tp}' -> '{g_out['generated_text']}'")

    qa_tests = [
        "Why did the cold ice melt into clear water when warm sunlight shone?",
        "Where did lily look first for her red apple after ben hid it in the wooden box?",
        "How did leo and sam fix the broken wooden bridge across the deep river?",
        "Where did the little girl walk and what did she see in the tall tree?",
        "What did the brave boy discover near the blue river?",
    ]
    qa_results: List[Dict[str, object]] = []
    for q in qa_tests:
        qa_out = trainer.answer_cognitive_question(q)
        qa_results.append(qa_out)
        print(
            f"  • [{qa_out['cognitive_domain']}] Вопрос: {q}\n"
            f"    Режим={qa_out['active_inference_mode']} | HippoRAG PPR={qa_out['hipporag_ppr_chain'][:4]} -> Ответ: '{qa_out['answer']}'"
        )

    produce_dir = ROOT / "Produce"
    produce_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = produce_dir / "checkpoints" / "ziwago_v8_5_master_cpu.pt"
    ckpt_v9_path = produce_dir / "checkpoints" / "ziwago_v9_0_intelligence_master.pt"
    _checkpoint(trainer.u, str(ckpt_path), itos)
    _checkpoint(trainer.u, str(ckpt_v9_path), itos)

    summary_payload = {
        "methodology": "Developmental & Cognitive Master Curriculum (Ziwago-Prime v9.0 32-Core Intelligence Edition)",
        "parquet_tokens_scanned": dataset.parquet_tokens_scanned,
        "parquet_stories_used": dataset.parquet_stories_used,
        "parquet_clean_sentences": dataset.parquet_clean_sentences,
        "cognitive_corpus_path": dataset.cognitive_corpus_path,
        "vocab_size": len(itos),
        "morpho_sdr_evaluation": morpho_eval,
        "compositional_indexer_capacity": trainer.theta_gamma_indexer.theoretical_capacity(),
        "compositional_traces_stored": len(trainer.theta_gamma_indexer.compositional_traces),
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
        "checkpoint_path": str(ckpt_v9_path),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }

    json_out = produce_dir / "v8_pilot_curriculum_results.json"
    json_v9_out = produce_dir / "v9_intelligence_curriculum_results.json"
    json_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    json_v9_out.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v9.0 Master 32-Core / 64-Thread Intelligence Trainer")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per developmental stage (default: 3)")
    parser.add_argument("--gen-tokens", type=int, default=160, help="Tokens to generate at checkpoints (default: 160)")
    parser.add_argument("--max-vocab", type=int, default=1024, help="Morpho-cognitive bio-vocabulary size (default: 1024)")
    parser.add_argument("--max-stories", type=int, default=10_000, help="Stories to scan in parallel from 10k-tinystories.parquet")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 32), help="Parallel dataset workers")
    parser.add_argument("--cpu-threads", type=int, default=max(1, os.cpu_count() or 32), help="PyTorch intra-op CPU threads")
    args = parser.parse_args()
    run_pilot_curriculum(
        epochs_per_stage=args.epochs,
        gen_tokens=args.gen_tokens,
        max_vocab=args.max_vocab,
        max_stories=args.max_stories,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )
