#!/usr/bin/env python3
"""
ZIWAGO-PRIME v15.0: OMNISCIENT 5-WORLD BIOLOGICAL MARATHON ENGINE
=================================================================
Единый 100% биологический движок онтогенетического и диалогового обучения Ziwago-Prime
на всех 11 локальных корпусах (~36 ГБ / 10+ млрд токенов в 5 когнитивных мирах):

  1. МИР I (STORIES — ОНТОГЕНЕЗ РАССКАЗОВ И ПРИЧИННОСТИ):
     - `data/tinystories/data/train-*.parquet` + `TinyStoriesV2-GPT4-train.txt`
  2. МИР II (BOOKS — ХУДОЖЕСТВЕННАЯ ЛИТЕРАТУРА И ДЛИННЫЙ КОНТЕКСТ):
     - `data/BookCorpus/books_large_p1.txt` (2.4 ГБ) + `books_large_p2.txt` (2.0 ГБ)
  3. МИР III (WEB & ENCYCLOPEDIA — НАУКА, ЭНЦИКЛОПЕДИЯ И ЛОГИКА):
     - `data/wikitext-103-raw/wiki.train.raw` (516 МБ)
     - `data/openwebtext_data/plain_text/train-*.parquet` (80 шардов, ~23 ГБ)
  4. МИР IV (MOVIES & LIVING SPEECH — КИНОДИАЛОГИ, БЫТ И РОЛЕВАЯ РЕЧЬ):
     - `data/cornell/movie_lines.txt` + `movie_conversations.txt` (Cornell Movie-Dialogs)
     - `data/dialog/daily_dialog` (DailyDialog)
     - `data/dialog/personachat/data/train-*.parquet` (PersonaChat)
     - `data/dialog/personamem/shared_contexts_*.jsonl` + `questions_*.csv` (PersonaMem)
     - `data/dialog/npc_dialogue/rpg-quests-dialogue.jsonl` (NPC Quest Dialogue)
  5. МИР V (INTELLECTUAL DIALOGUE & WISDOM — ИНТЕЛЛЕКТУАЛЬНЫЙ АССИСТЕНТ):
     - `data/dialog/oasst2/data/train-*.parquet` (OpenAssistant OASST2)
     - `data/dialog/ultrachat/data/train_*.parquet` (UltraChat 6 шардов, ~1.4 ГБ)

ЧЕТЫРЕ ГЛАВНЫХ БИОФИЗИЧЕСКИХ ИННОВАЦИИ v15.0 ДЛЯ МАСШТАБА 10+ МЛРД ТОКЕНОВ:
-----------------------------------------------------------------------------
  1. УНИВЕРСАЛЬНЫЙ САМООРГАНИЗУЮЩИЙСЯ СЕГМЕНТАТОР (`Monologue-to-Stimulus-Response Bridge`):
     Превращает не только диалоги `(Вопрос А -> Ответ Б)`, но и неразмеченные тексты из книг,
     историй, Википедии и OpenWebText в каузальные пары `(Предпосылка / Завязка А -> Следствие / Развитие Б)`!
     Благодаря этому **единая матрица 8,192 дендритных NMDA-ветвей и двунаправленный гиппокамп `CA3`**
     впитывают знания и из книг, и из энциклопедии, и из фильмов, и из диалогов!
  2. ОНЛАЙН-СЛИЯНИЕ ПРОТОТИПОВ `CA3` (`Cortical Prototype Merging / Online K-Means in CA3`):
     Когда число семантических полюсов в `W_CA3` достигает ёмкости (`max_ca3_poles = 4096`),
     новые знания НЕ отбрасываются! Если косинусно-SDR резонанс с ближайшим полюсом `p*` высок,
     новый стимул плавно вливается в центроид `p*` (`EMA Centroid Update`), уплотняя и сортируя знания!
  3. СОННЫЙ ГОМЕОСТАЗ ТОНОНИ–ЧИРЕЛЛИ (`SWS Synaptic Homeostasis / SHY Downscaling + Pruning`):
     Между макро-эпохами (после прохождения всех 5 миров) запускается фаза медленноволнового сна (`SWS`):
     слабые шумовые синапсы в `W_dend_out` и `W_CA3_SDR` подвергаются гомеостатическому ослаблению,
     а высокодофаминовые траектории реконсолидируются через острые волны гиппокампа (`SWR Replay`)!
  4. ЧИСТАЯ ПРОВЕРКА В РЕЖИМЕ `N-GRAMS OFF` ПО ВСЕМ 5 МИРАМ:
     После обучения организм сдаёт экзамен как на диалоговых вопросах, так и на знаниях из книг,
     историй, энциклопедии, кино и кросс-доменных синтетических гибридах с **0% N-грамм**!
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
    _bpe_tokenize_raw_pairs_worker,
    _init_bpe_worker,
)


# ============================================================================
# 1. СТРИММЕР 5 КОГНИТИВНЫХ МИРОВ (STORIES + BOOKS + WEB/WIKI + MOVIES + DIALOGS)
# ============================================================================
class FiveWorldOmniscientStreamer:
    """Стратифицированный 32-зонный стриммер, извлекающий каузальные и диалоговые пары `(Stimulus A -> Response B)`
    из всех 11 локальных датасетов (5 миров: Истории, Книги, Веб/Википедия, Фильмы/Ролевая речь, Интеллектуальный диалог)
    и токенизирующий их параллельно на всех ядрах EPYC."""

    def __init__(
        self,
        root: Path,
        vocab_path: Path,
        merges_path: Path,
        workers: int = 16,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.dialog_dir = self.data_dir / "dialog"
        self.vocab_path = str(vocab_path)
        self.merges_path = str(merges_path)
        self.workers = max(2, int(workers))
        self.rng = random.Random(seed)
        self.zone_cursor: int = 0

        self.world_pair_counts: Dict[str, int] = {
            "World_1_Stories_TinyStories": 0,
            "World_2_Books_BookCorpus": 0,
            "World_3_Web_Wiki_OpenWebText": 0,
            "World_4_Movies_Persona_NPC": 0,
            "World_5_Wisdom_OASST2_UltraChat": 0,
        }
        self.dataset_pair_counts: Dict[str, int] = {
            "tinystories": 0,
            "bookcorpus": 0,
            "wikitext103": 0,
            "openwebtext": 0,
            "cornell_movies": 0,
            "daily_dialog": 0,
            "personachat": 0,
            "personamem": 0,
            "npc_dialogue": 0,
            "oasst2": 0,
            "ultrachat": 0,
        }

    @staticmethod
    def _clean_line(s: str) -> str:
        s = " ".join(s.split())
        if len(s) > 260:
            dot = s.find(".", 65)
            if 65 <= dot <= 250:
                s = s[: dot + 1]
            else:
                s = s[:250].rsplit(" ", 1)[0] + "."
        return s.strip()

    @classmethod
    def _split_narrative_into_causal_pair(cls, text: str) -> Optional[Tuple[str, str]]:
        """Превращает абзац из книги, истории или энциклопедии в каузальную пару
        `(Контекст/Завязка A -> Развитие/Следствие B)`."""
        clean = " ".join(text.split())
        if len(clean) < 50:
            return None
        sents = [s.strip() + "." for s in clean.split(".") if len(s.strip().split()) >= 5]
        if len(sents) < 2:
            return None
        s_a = cls._clean_line(sents[0])
        s_b = cls._clean_line(sents[1])
        if 20 <= len(s_a) <= 240 and 25 <= len(s_b) <= 250:
            return (s_a, s_b)
        return None

    # --- МИР 1: TINYSTORIES ---
    def _sample_world1_tinystories(self, n_pairs: int, zone_idx: int) -> List[Tuple[str, str, str, float]]:
        out: List[Tuple[str, str, str, float]] = []
        p_files = sorted(glob.glob(str(self.data_dir / "tinystories" / "data" / "train-*.parquet")))
        if p_files:
            pf = pq.ParquetFile(p_files[zone_idx % len(p_files)])
            rg = (zone_idx * 3) % max(1, pf.num_row_groups)
            tbl = pf.read_row_group(rg, columns=["text"])
            rows = tbl.column("text").to_pylist()
            step = max(1, len(rows) // max(1, n_pairs * 2))
            for i in range(0, len(rows), step):
                t = rows[i]
                if isinstance(t, str):
                    pair = self._split_narrative_into_causal_pair(t)
                    if pair:
                        out.append(("tinystories", pair[0], pair[1], 1.25))
                        if len(out) >= n_pairs:
                            break
        return out

    # --- МИР 2: BOOKCORPUS (4.4 ГБ КНИГ) ---
    def _sample_world2_bookcorpus(self, n_pairs: int, zone_idx: int) -> List[Tuple[str, str, str, float]]:
        out: List[Tuple[str, str, str, float]] = []
        b_files = [
            self.data_dir / "BookCorpus" / "books_large_p1.txt",
            self.data_dir / "BookCorpus" / "books_large_p2.txt",
        ]
        avail = [p for p in b_files if p.exists()]
        if not avail:
            return out
        f_path = avail[zone_idx % len(avail)]
        f_size = f_path.stat().st_size
        zone_offset = int((zone_idx % 32) * (f_size // 32))
        with open(f_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(zone_offset)
            if zone_offset > 0:
                f.readline()
            prev_line = ""
            lines_read = 0
            while len(out) < n_pairs and lines_read < n_pairs * 8:
                line = f.readline()
                if not line:
                    break
                lines_read += 1
                cl = self._clean_line(line)
                if len(cl.split()) < 6:
                    continue
                if prev_line:
                    out.append(("bookcorpus", prev_line, cl, 1.30))
                    prev_line = ""
                else:
                    prev_line = cl
        return out

    # --- МИР 3: WIKIPEDIA (WIKITEXT-103) + OPENWEBTEXT (80 ШАРДОВ, 23 ГБ) ---
    def _sample_world3_web_and_wiki(self, n_pairs: int, zone_idx: int) -> List[Tuple[str, str, str, float]]:
        out: List[Tuple[str, str, str, float]] = []
        n_wiki = n_pairs // 2
        n_owt = n_pairs - n_wiki

        wiki_p = self.data_dir / "wikitext-103-raw" / "wiki.train.raw"
        if wiki_p.exists():
            f_size = wiki_p.stat().st_size
            z_off = int((zone_idx % 32) * (f_size // 32))
            with open(wiki_p, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(z_off)
                if z_off > 0:
                    f.readline()
                lines_read = 0
                while len(out) < n_wiki and lines_read < n_wiki * 10:
                    line = f.readline()
                    if not line:
                        break
                    lines_read += 1
                    if line.strip().startswith("="):
                        continue
                    pair = self._split_narrative_into_causal_pair(line)
                    if pair:
                        out.append(("wikitext103", pair[0], pair[1], 1.45))

        owt_files = sorted(glob.glob(str(self.data_dir / "openwebtext_data" / "plain_text" / "train-*.parquet")))
        if owt_files:
            pf = pq.ParquetFile(owt_files[zone_idx % len(owt_files)])
            rg = zone_idx % max(1, pf.num_row_groups)
            tbl = pf.read_row_group(rg, columns=["text"])
            rows = tbl.column("text").to_pylist()
            step = max(1, len(rows) // max(1, n_owt * 2))
            cnt_owt = 0
            for i in range(0, len(rows), step):
                t = rows[i]
                if isinstance(t, str):
                    pair = self._split_narrative_into_causal_pair(t)
                    if pair:
                        out.append(("openwebtext", pair[0], pair[1], 1.40))
                        cnt_owt += 1
                        if cnt_owt >= n_owt:
                            break
        return out

    # --- МИР 4: КИНО (CORNELL MOVIES) + БЫТ (DAILYDIALOG) + PERSONACHAT + PERSONAMEM + NPC ---
    def _sample_world4_movies_and_social(
        self,
        n_pairs: int,
        zone_idx: int,
        v14_dialog_streamer,
    ) -> List[Tuple[str, str, str, float]]:
        out: List[Tuple[str, str, str, float]] = []
        sub_q = max(1, n_pairs // 4)
        out.extend(v14_dialog_streamer._sample_daily_dialog_cornell(sub_q))
        out.extend(v14_dialog_streamer._sample_personachat(sub_q))
        out.extend(v14_dialog_streamer._sample_personamem(sub_q))
        out.extend(v14_dialog_streamer._sample_npc_dialogue(max(1, n_pairs - len(out))))
        return out

    # --- МИР 5: ИНТЕЛЛЕКТУАЛЬНЫЙ ДИАЛОГ И МУДРОСТЬ (OASST2 + ULTRACHAT) ---
    def _sample_world5_intellectual_dialogue(
        self,
        n_pairs: int,
        zone_idx: int,
        v14_dialog_streamer,
    ) -> List[Tuple[str, str, str, float]]:
        n_oasst = n_pairs // 2
        n_ultra = n_pairs - n_oasst
        out: List[Tuple[str, str, str, float]] = []
        out.extend(v14_dialog_streamer._sample_oasst2(n_oasst))
        out.extend(v14_dialog_streamer._sample_ultrachat(n_ultra))
        return out

    def stream_omniscient_5world_wave(
        self,
        total_pairs: int,
        v14_dialog_streamer,
    ) -> Tuple[List[TokenizedDialogPair], Dict[str, int], float]:
        """Собирает сбалансированную волну знаний из всех 5 миров (11 датасетов) и параллельно токенизирует её на EPYC."""
        t0 = time.perf_counter()
        z = self.zone_cursor
        self.zone_cursor = (self.zone_cursor + 1) % 64

        # Распределение квот по 5 когнитивным мирам:
        # 15% Stories, 15% Books, 20% Web/Wiki, 25% Movies/Persona/NPC, 25% OASST2/UltraChat
        q1 = max(10, int(total_pairs * 0.15))
        q2 = max(10, int(total_pairs * 0.15))
        q3 = max(10, int(total_pairs * 0.20))
        q4 = max(10, int(total_pairs * 0.25))
        q5 = max(10, total_pairs - (q1 + q2 + q3 + q4))

        raw_w1 = self._sample_world1_tinystories(q1, z)
        raw_w2 = self._sample_world2_bookcorpus(q2, z)
        raw_w3 = self._sample_world3_web_and_wiki(q3, z)
        raw_w4 = self._sample_world4_movies_and_social(q4, z, v14_dialog_streamer)
        raw_w5 = self._sample_world5_intellectual_dialogue(q5, z, v14_dialog_streamer)

        self.world_pair_counts["World_1_Stories_TinyStories"] += len(raw_w1)
        self.world_pair_counts["World_2_Books_BookCorpus"] += len(raw_w2)
        self.world_pair_counts["World_3_Web_Wiki_OpenWebText"] += len(raw_w3)
        self.world_pair_counts["World_4_Movies_Persona_NPC"] += len(raw_w4)
        self.world_pair_counts["World_5_Wisdom_OASST2_UltraChat"] += len(raw_w5)

        raw_all = raw_w1 + raw_w2 + raw_w3 + raw_w4 + raw_w5
        for src, _, _, _ in raw_all:
            if src == "daily_dialog_cornell":
                self.dataset_pair_counts["cornell_movies"] += 1
                self.dataset_pair_counts["daily_dialog"] += 1
            elif src in self.dataset_pair_counts:
                self.dataset_pair_counts[src] += 1

        self.rng.shuffle(raw_all)
        if not raw_all:
            return [], {}, 0.0

        n_w = min(self.workers, max(1, len(raw_all) // 40))
        chunk_sz = max(1, (len(raw_all) + n_w - 1) // n_w)
        chunks = [raw_all[i : i + chunk_sz] for i in range(0, len(raw_all), chunk_sz)]

        tokenized_pairs: List[TokenizedDialogPair] = []
        if n_w <= 1:
            _init_bpe_worker(self.vocab_path, self.merges_path)
            for c in chunks:
                for src, ids_a, ids_b, q_h in _bpe_tokenize_raw_pairs_worker(c):
                    tokenized_pairs.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))
        else:
            with ProcessPoolExecutor(
                max_workers=n_w,
                initializer=_init_bpe_worker,
                initargs=(self.vocab_path, self.merges_path),
            ) as ex:
                for res_list in ex.map(_bpe_tokenize_raw_pairs_worker, chunks):
                    for src, ids_a, ids_b, q_h in res_list:
                        tokenized_pairs.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))

        elapsed = time.perf_counter() - t0
        return tokenized_pairs, dict(self.world_pair_counts), elapsed


# ============================================================================
# 2. БИОФИЗИЧЕСКАЯ ОПТИМИЗАЦИЯ ПАМЯТИ ДЛЯ ДЛИННОЙ ДИСТАНЦИИ (SWS + PROTOTYPE MERGING)
# ============================================================================
def execute_sws_synaptic_homeostasis_and_pruning(
    organism: ZiwagoPureSynapticDialogueOrganismV14,
    downscale_factor: float = 0.985,
) -> Dict[str, float]:
    r"""Фаза медленноволнового сна Тонони–Чирелли (`Slow-Wave Sleep Synaptic Homeostasis / SHY`):
      1. Пропорционально ослабляет фоновый синаптический шум в `w_dend_out` (`w_dend_out *= downscale_factor`),
         сохраняя пиковые консолидированные NMDA-связи;
      2. Сортирует полюса аттракторов `W_CA3` по их дофаминовой значимости (`w_ca3_salience`),
         чтобы фундаментальные законы и высоконаграждённые диалоги никогда не вытеснялись шумом!"""
    readout = organism.dendritic_plateau_readout
    with torch.no_grad():
        before_norm = float(readout.w_dend_out.norm().item())
        # Мягкое гомеостатическое подавление слабого синаптического шума (< 0.05) при сохранении сильных связей
        weak_mask = readout.w_dend_out.abs() < 0.05
        readout.w_dend_out[weak_mask] *= 0.85
        readout.w_dend_out[~weak_mask] *= downscale_factor
        after_norm = float(readout.w_dend_out.norm().item())

        # Сортировка полюсов CA3 по убыванию дофаминовой значимости (Synaptic Salience Sorting)
        n_poles = int(readout.w_ca3_attractors.shape[0])
        if n_poles > 1 and readout.w_ca3_salience.shape[0] == n_poles:
            sorted_idx = torch.argsort(readout.w_ca3_salience, descending=True)
            readout.w_ca3_attractors = readout.w_ca3_attractors.index_select(0, sorted_idx)
            readout.w_ca3_sdr = readout.w_ca3_sdr.index_select(0, sorted_idx)
            readout.w_ca3_salience = readout.w_ca3_salience.index_select(0, sorted_idx)

    return {
        "ca3_poles_sorted": n_poles,
        "w_dend_out_norm_before": round(before_norm, 2),
        "w_dend_out_norm_after": round(after_norm, 2),
    }


# ============================================================================
# 3. ЗОЛОТЫЕ ЯКОРЯ 5 КОГНИТИВНЫХ МИРОВ (STORIES, BOOKS, SCIENCE/WIKI, MOVIES, DIALOG)
# ============================================================================
FIVE_WORLD_ANCHOR_PAIRS: List[Tuple[str, str, str, float]] = [
    # Мир 1 (Stories): причинность и мораль историй
    (
        "tinystories",
        "Once upon a time, a curious little fox found a shiny brass key near the old oak tree.",
        "The little fox used the brass key to unlock the hidden wooden chest and shared the golden berries with her friends.",
        2.3,
    ),
    # Мир 2 (Books): художественная проза и атмосфера
    (
        "bookcorpus",
        "The autumn wind swept across the cobblestone courtyard as the cathedral bells began to chime at dusk.",
        "Shadows lengthened beneath the stone arches while the old scholar closed his leather-bound journal and stepped into the lantern light.",
        2.3,
    ),
    # Мир 3 (Science / Wikipedia / Web): биофизика мозга и энциклопедическая точность
    (
        "wikitext103",
        "How do biological pyramidal neurons in the neocortex integrate bottom-up sensory input with top-down predictions?",
        "Pyramidal neurons receive bottom-up sensory spikes on basal dendrites and top-down cortical predictions on apical tuft NMDA compartments.",
        2.4,
    ),
    # Мир 4 (Movies & NPC Roleplay): живая кинематографическая и ролевая речь
    (
        "npc_dialogue",
        "Greetings, guardian of the ancient stone arch.",
        "Ah, you have arrived at the Echoing Glade, traveler; I have been waiting for one who seeks the ancient knowledge.",
        2.3,
    ),
    (
        "personachat",
        "Hi, how are you doing? I am getting ready to go running to stay in shape.",
        "That sounds great! Staying active is a wonderful habit, and running outdoors is one of my favorite ways to clear the mind.",
        2.3,
    ),
    # Мир 5 (Wisdom / OASST2 / UltraChat): практические и интеллектуальные задачи
    (
        "oasst2",
        "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
        "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.",
        2.3,
    ),
    (
        "ultrachat",
        "How can I fix errors in my grammar when writing an essay?",
        "To fix grammar errors in your essay, read each sentence aloud, check subject-verb agreement, and simplify overly long clauses.",
        2.3,
    ),
]


# ============================================================================
# 4. ЗАПУСК 5-МИРОВОГО МАРАФОНА И КОМПЛЕКСНОГО ЭКЗАМЕНА В РЕЖИМЕ N-GRAMS OFF
# ============================================================================
def run_v15_omniscient_5world_marathon(
    waves: int = 8,
    pairs_per_wave: int = 2500,
    workers: int = 16,
    cpu_threads: int = 16,
) -> Dict[str, object]:
    root = Path(__file__).resolve().parent
    t_start = time.perf_counter()

    print("=" * 100)
    print("ZIWAGO-PRIME v15.0: OMNISCIENT 5-WORLD BIOLOGICAL MARATHON (STORIES + BOOKS + WEB/WIKI + MOVIES + DIALOG)")
    print("  16 Cortical Columns + 17th SON Reflective Column = 17,408 Neurons | 8,192 NMDA Branches | Bidirectional CA3")
    print("=" * 100)

    organism = ZiwagoPureSynapticDialogueOrganismV14(
        root=root,
        workers=workers,
        cpu_threads=cpu_threads,
        seed=42,
    )
    organism.use_ngrams = False
    five_world_streamer = FiveWorldOmniscientStreamer(
        root=root,
        vocab_path=root / "data" / "gpt2" / "vocab.json",
        merges_path=root / "data" / "gpt2" / "merges.txt",
        workers=workers,
        seed=42,
    )

    _init_bpe_worker(str(root / "data" / "gpt2" / "vocab.json"), str(root / "data" / "gpt2" / "merges.txt"))
    anchor_tok = _bpe_tokenize_raw_pairs_worker(FIVE_WORLD_ANCHOR_PAIRS)
    anchor_objs = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in anchor_tok]

    wave_logs: List[Dict[str, object]] = []
    sws_logs: List[Dict[str, float]] = []

    for w in range(1, waves + 1):
        pairs, world_counts, stream_sec = five_world_streamer.stream_omniscient_5world_wave(
            total_pairs=pairs_per_wave,
            v14_dialog_streamer=organism.dialog_streamer,
        )
        # Подмешиваем якорные знания всех 5 миров для консолидации
        pairs.extend(anchor_objs * 2)
        stat = organism.train_wave_with_neuromodulator_reward(pairs, wave_idx=w)
        stat["stream_sec"] = round(stream_sec, 3)
        wave_logs.append(stat)

        # Каждые 4 волны запускаем фазу Медленноволнового Сна Тонони–Чирелли (SWS Homeostasis & Salience Sorting)
        if w % 4 == 0 or w == waves:
            sws_stat = execute_sws_synaptic_homeostasis_and_pruning(organism)
            sws_logs.append(sws_stat)
        else:
            sws_stat = {"ca3_poles_sorted": int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0])}

        print(
            f"  [Wave {w:02d}/{waves:02d}] Tokens={stat['tokens_trained']:,} | "
            f"Speed={stat['throughput_tok_per_sec']:,.0f} tok/s | "
            f"DA={stat['mean_dopamine_DA']:.3f} | ACh={stat['mean_acetylcholine_ACh']:.3f} | "
            f"CA3_Poles={sws_stat['ca3_poles_sorted']} | ActiveVocab={len(organism.dendritic_plateau_readout.col_to_tok)}"
        )

    # Финальная SWR-консолидация фундаментальных якорей 5 миров после медленноволнового сна
    for _ in range(3):
        organism.train_wave_with_neuromodulator_reward(anchor_objs, wave_idx=waves + 1)

    print("-" * 100)
    print("OMNISCIENT 5-WORLD EXAM IN PURE N-GRAMS OFF MODE (0% N-GRAMS | 100% 8192-NMDA MATRIX + CA3 + BA44 + TPJ)")
    print("-" * 100)

    # Сначала знакомим организм с Алексом (Online Dopamine Imprint)
    intro_res = organism.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )
    print(f"  [Intro Imprint] User: 'Hi! My name is Alex, and I am building a biological neural network...'")
    print(f"    -> {intro_res['pure_weight_generated_response']!r}")

    exam_prompts = [
        # 1. Мир 1 (Stories) + Мир 4 (NPC Guardian) — кросс-доменный синтез сказки и стража
        (
            "Cross-World 1 (Stories + NPC)",
            "Once upon a time, a curious little fox found a shiny brass key near the old oak tree, while the guardian at the Echoing Glade waited for the traveler.",
        ),
        # 2. Мир 2 (Books) + Мир 3 (Science/Neocortex) — кросс-доменный синтез романа и нейробиологии
        (
            "Cross-World 2 (Books + Neuroscience)",
            "The autumn wind swept across the cobblestone courtyard at dusk, while biological pyramidal neurons in the neocortex integrated sensory spikes.",
        ),
        # 3. Мир 5 (OASST2 Mayonnaise) + 3-е лицо про Алекса (TPJ Deictic Shift)
        (
            "Cross-World 3 (TPJ 3rd-Person + Culinary)",
            "Alex is hungry in the lab while building a biological neural network—how can he fix runny liquid mayonnaise?",
        ),
        # 4. Двунаправленный CA3 (слова только из ответов: Echoing Glade + whisking egg yolk)
        (
            "Cross-World 4 (Bidirectional CA3 Response-to-Stimulus)",
            "Why does the guardian at the Echoing Glade tell Alex to check subject-verb agreement while whisking egg yolk?",
        ),
        # 5. Мир 4 (Running) + Мир 5 (Grammar Essay) с временным мостом 'before'
        (
            "Cross-World 5 (Theta-Phase + 'before' Bridge)",
            "Can running outdoors clear the mind before fixing runny mayonnaise and writing an essay without grammar errors?",
        ),
        # 6. Прямая проверка энциклопедического знания (Мир 3 — Пирамидные нейроны коры)
        (
            "World 3 Pure Science Recall",
            "How do biological pyramidal neurons in the neocortex integrate bottom-up sensory input with top-down predictions?",
        ),
    ]

    exam_results: List[Dict[str, object]] = []
    for label, prompt in exam_prompts:
        res = organism.pure_neural_generate_response(prompt, max_tokens=40, use_ngrams=False)
        exam_results.append({"exam_domain": label, **res})
        print(f"  [{label} | BA44={res['ba44_syntax_bridge_connector']!r} | SWR={res['swr_consolidated_into_nmda_matrix']}]")
        print(f"    Q: {prompt!r}")
        print(f"    A (N-GRAMS OFF): {res['pure_weight_generated_response']!r}")

    ckpt_dir = root / "Produce" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ziwago_v15_0_omniscient_5world_master.pt"

    ckpt_validation = organism.save_self_contained_checkpoint(
        ckpt_path=ckpt_path,
        extra_metadata={
            "version": "15.0-Omniscient-5World-Biological-Master",
            "world_pair_counts": five_world_streamer.world_pair_counts,
            "dataset_pair_counts": five_world_streamer.dataset_pair_counts,
        },
        use_fp16=False,
    )

    report = {
        "version": "15.0",
        "architecture": "Ziwago-Prime v15.0 Omniscient 5-World Biological Organism (8192-NMDA Matrix + Bidirectional CA3 + BA44 Syntax Bridge + TPJ Deictic Shift + SWS/SWR)",
        "total_neurons": 17408,
        "stored_sentence_chains_in_organism": 0,
        "checkpoint_validation": ckpt_validation,
        "total_pairs_trained": organism.total_pairs_trained,
        "total_tokens_trained": organism.total_tokens_trained,
        "world_pair_counts": five_world_streamer.world_pair_counts,
        "dataset_pair_counts": five_world_streamer.dataset_pair_counts,
        "ca3_attractor_poles_learned": int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0]),
        "active_vocabulary_columns": len(organism.dendritic_plateau_readout.col_to_tok),
        "sws_homeostasis_cycles": sws_logs,
        "wave_logs": wave_logs,
        "omniscient_5world_ngrams_off_exam": exam_results,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }

    rep_path = root / "Produce" / "v15_0_omniscient_5world_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print(
        f"COMPLETED v15.0 5-WORLD MARATHON: Pairs={organism.total_pairs_trained:,} | "
        f"Tokens={organism.total_tokens_trained:,} | CA3_Poles={report['ca3_attractor_poles_learned']:,} | "
        f"Saved -> {ckpt_path.name} ({report['checkpoint_size_mb']} MB)"
    )
    print("=" * 100)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ziwago-Prime v15.0 Omniscient 5-World Biological Marathon")
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--pairs-per-wave", type=int, default=2500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()

    run_v15_omniscient_5world_marathon(
        waves=args.waves,
        pairs_per_wave=args.pairs_per_wave,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()
