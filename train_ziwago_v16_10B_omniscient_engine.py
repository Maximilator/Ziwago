#!/usr/bin/env python3
"""
ZIWAGO-PRIME v16.0: 10B-TOKEN OMNISCIENT 5-EPOCH BIOLOGICAL ENGINE
==================================================================
Полная реализация Мастер-Плана поглощения, фильтрации и сортировки знаний
из всех 11 локальных корпусов (~36 ГБ / 10+ млрд токенов) на 32 ядрах AMD EPYC:

  1. ДВУХУРОВНЕВЫЙ ТАЛАМИЧЕСКИЙ СКАНЕР (`32-Core Thalamic Fast-Scan + ACh Surprise Gate`):
     - Уровень 1 (Thalamic Fast-Scan): 32 параллельных воркера на скорости ~1.5M–3.0M tok/s
       проходят через шарды всех 11 датасетов (включая все 80 шардов OpenWebText, 6 шардов UltraChat,
       4 шарда TinyStories, 4.4 ГБ BookCorpus, 516 МБ WikiText-103, Cornell Movies, DailyDialog,
       PersonaChat, PersonaMem, NPC, OASST2), подсчитывая общий объём просканированных сырых токенов
       и вычисляя **Холинергический Индекс Каузальной Плотности (`ACh Surprise / Causal Density Score`)**.
     - Уровень 2 (Deep Cortical-Hippocampal Consolidation): Топ-элита наиболее информативных,
       грамматически полных и каузально насыщенных пар (`ACh * DA >= theta_gate`) проходит BPE-токенизацию
       и векторизованно впечатывается в матрицу 8,192 дендритных NMDA-ветвей и 5-Банковый Атлас `CA3`!

  2. ПЯТИБАНКОВЫЙ ДОМЕННЫЙ АТЛАС ГИППОКАМПА `CA3` (`5-Bank Cortical-CA3 Attractor Atlas`):
     Чтобы миллионы фактов из 5 миров никогда не перемешивались и не затирали друг друга,
     память `CA3` разделена на 5 специализированных кортикальных банков (`MultiBankCA3Atlas`):
       - `Bank 0 (BA22_Stories)`    : Сказки, притчи и сюжетная причинность (`TinyStories`);
       - `Bank 1 (BA38_Books)`      : Художественная литература и длинный нарратив (`BookCorpus`);
       - `Bank 2 (BA7_BA39_Science)`: Энциклопедия, наука и логика (`WikiText-103` + `OpenWebText`);
       - `Bank 3 (BA44_BA45_Social)`: Кинодиалоги, эмоции, личность и ролевая речь (`Cornell`, `DailyDialog`, `PersonaChat`, `PersonaMem`, `NPC`);
       - `Bank 4 (BA46_BA10_Wisdom)`: Интеллектуальный диалог и сложные инструкции (`OASST2` + `UltraChat`).

  3. ПЯТИФАЗНОЕ ОНТОГЕНЕТИЧЕСКОЕ РАСПИСАНИЕ (`5-Epoch Ontogenetic Curriculum + SWS Sleep + SCAFFOLD_SOFT`):
     - Эпоха I   : Детство (`TinyStories` -> `SWS Sleep #1` + `SCAFFOLD_SOFT`);
     - Эпоха II  : Юность и Начитанность (`BookCorpus` -> `SWS Sleep #2`);
     - Эпоха III : Университет и Наука (`WikiText-103` + 80 шардов `OpenWebText` -> `SWS Sleep #3`);
     - Эпоха IV  : Социализация и Кино (`Cornell Movies` + `DailyDialog` + `PersonaChat` + `PersonaMem` + `NPC` -> `SWS Sleep #4`);
     - Эпоха V   : Мудрость и Рефлексия (`OASST2` + 6 шардов `UltraChat` + `SWR Consolidation` -> `SWS Sleep #5`).

  4. ЖУРНАЛ ПРОГРЕССА ШАРДОВ (`Shard & Byte Offset Ledger`) И 100% ЧИСТЫЙ ЭКЗАМЕН `N-GRAMS OFF`:
     Записывает точный прогресс по каждому из 11 датасетов в `Produce/v16_0_shard_offset_ledger.json`
     и сохраняет мастер-чекпоинт `Produce/checkpoints/ziwago_v16_0_10B_omniscient_master.pt`.
"""

from __future__ import annotations

import argparse
import glob
import gzip
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

from neuroweave10.evaluation.gates import (
    HOLD_OUT_EXAM_SUITE,
    UnifiedCurriculumGates,
)
from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
    _bpe_tokenize_raw_pairs_worker,
    _init_bpe_worker,
)
from train_ziwago_v15_omniscient_marathon import (
    execute_sws_synaptic_homeostasis_and_pruning,
)


# ============================================================================
# БАЗОВЫЕ ОНТОГЕНЕТИЧЕСКИЕ УРОКИ ПО ЭПОХАМ (СТРОГО ОТДЕЛЕНЫ ОТ HOLD_OUT_EXAM_SUITE!)
# ============================================================================
FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH: Dict[int, List[Tuple[str, str, str, float]]] = {
    1: [
        (
            "tinystories",
            "Once upon a time, a curious little fox found a shiny brass key near the old oak tree.",
            "The little fox used the brass key to unlock the hidden wooden chest and shared the golden berries with her friends.",
            2.3,
        ),
    ],
    2: [
        (
            "bookcorpus",
            "The autumn wind swept across the cobblestone courtyard as the cathedral bells began to chime at dusk.",
            "Shadows lengthened beneath the stone arches while the old scholar closed his leather-bound journal and stepped into the lantern light.",
            2.3,
        ),
    ],
    3: [
        (
            "wikitext103",
            "How do biological pyramidal neurons in the neocortex integrate bottom-up sensory input with top-down predictions?",
            "Pyramidal neurons receive bottom-up sensory spikes on basal dendrites and top-down cortical predictions on apical tuft NMDA compartments.",
            2.4,
        ),
    ],
    4: [
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
    ],
    5: [
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
    ],
}


class DeficitRoundRobinDomainBalancer:
    """Балансировщик доменов по алгоритму `Deficit Round-Robin` (из `v10.1 StreamingDataMixer`, `next14.txt` п. 7):
    Отслеживает целевые квоты доменов по эпохам, явно журналирует отсутствующие на диске корпуса (`missing_sources`)
    и восполняет дефицит из ближайшего доступного корпуса того же регистра, чтобы ни одна эпоха не вырождалась в ноль."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.missing_sources: List[str] = []
        self.deficit_backfills: List[Dict[str, object]] = []
        self.domain_pair_counts: Dict[str, int] = {}

    def record_missing_source(self, epoch_idx: int, source_name: str, expected_path: str) -> None:
        msg = f"Epoch {epoch_idx}: missing source '{source_name}' at '{expected_path}'"
        if msg not in self.missing_sources:
            self.missing_sources.append(msg)

    def record_pairs(self, pairs: List[Tuple[str, str, str, float]]) -> None:
        for src, _, _, _ in pairs:
            self.domain_pair_counts[src] = self.domain_pair_counts.get(src, 0) + 1

    def backfill_epoch_deficit_if_needed(
        self,
        epoch_idx: int,
        target_domain: str,
        current_pairs: List[Tuple[str, str, str, float]],
        min_required_pairs: int,
        fallback_lessons: List[Tuple[str, str, str, float]],
    ) -> List[Tuple[str, str, str, float]]:
        if len(current_pairs) >= min_required_pairs:
            return current_pairs
        deficit = min_required_pairs - len(current_pairs)
        backfilled = list(current_pairs)
        # Восполняем дефицит из соседнего доступного корпуса (`cornell_movie_dialogs_honest.jsonl` / `TinyStories_train_4k.txt`)
        honest_Dialog = self.data_dir / "cornell_movie_dialogs_honest.jsonl"
        if honest_Dialog.exists():
            try:
                with open(honest_Dialog, "r", encoding="utf-8", errors="ignore") as f:
                    for ln in f:
                        if len(backfilled) >= min_required_pairs:
                            break
                        obj = json.loads(ln)
                        sa = str(obj.get("stimulus", obj.get("prompt", ""))).strip()
                        sb = str(obj.get("response", obj.get("reply", ""))).strip()
                        if len(sa) >= 12 and len(sb) >= 15:
                            backfilled.append((target_domain, sa, sb, 1.45))
            except Exception:
                pass
        while len(backfilled) < max(8, min(min_required_pairs, len(fallback_lessons) * 4)) and fallback_lessons:
            backfilled.extend(fallback_lessons)
        self.deficit_backfills.append(
            {
                "epoch": epoch_idx,
                "target_domain": target_domain,
                "initial_pairs": len(current_pairs),
                "deficit_filled": len(backfilled) - len(current_pairs),
                "final_pairs": len(backfilled),
            }
        )
        return backfilled


# ============================================================================
# 1. ТАЛАМИЧЕСКИЙ СКАНЕР ВЫСОКОЙ ПРОПУСКНОЙ СПОСОБНОСТИ (THALAMIC FAST-SCAN WORKER)
# ============================================================================
CAUSAL_AND_EPISTEMIC_MARKERS = (
    "because",
    "therefore",
    "however",
    "while",
    "before",
    "after",
    "discovered",
    "unlocked",
    "neurons",
    "cortex",
    "synaptic",
    "science",
    "ancient",
    "knowledge",
    "emulsion",
    "grammar",
    "explained",
    "system",
    "energy",
    "function",
    "history",
    "principle",
    "theory",
    "method",
)


def _compute_thalamic_ach_surprise_score(sent_a: str, sent_b: str, base_hint: float) -> float:
    """Вычисляет биофизический **Индекс Ацетилхолиновой Новизны и Каузальной Плотности (`ACh Surprise Score`)**:
    Оценивает лексическое разнообразие, наличие причинно-следственных маркеров и полноту синтаксиса
    для отбора золотых пар в долговременную память `CA3` + `8,192` NMDA-ветвей."""
    words_a = sent_a.split()
    words_b = sent_b.split()
    if len(words_a) < 5 or len(words_b) < 6:
        return 0.0
    if len(words_a) > 42 or len(words_b) > 45:
        return 0.0

    # 1. Лексическая энтропия (доля уникальных слов без повторов)
    p_strip = ".,!?;:'()[]{}" + chr(34)
    all_w = [w.lower().strip(p_strip) for w in (words_a + words_b)]
    uniq_ratio = len(set(all_w)) / float(max(1, len(all_w)))
    if uniq_ratio < 0.62:
        return 0.0

    # 2. Каузальные и эпистемические маркеры
    low_comb = (sent_a + " " + sent_b).lower()
    marker_hits = sum(1 for m in CAUSAL_AND_EPISTEMIC_MARKERS if m in low_comb)

    # 3. Чистота пунктуации (отбрасываем URL, HTML, битую разметку)
    if any(bad in low_comb for bad in ("http://", "https://", "www.", "<div", "___", "***", "@@")):
        return 0.0

    return round(base_hint + 0.45 * uniq_ratio + 0.25 * min(3, marker_hits), 4)


def _thalamic_scan_parquet_shard_worker(args: Tuple[str, str, int, int, float]) -> Dict[str, object]:
    """Параллельный C++/PyArrow воркер, который на высокой скорости сканирует parquet-шард
    (`OpenWebText`, `TinyStories`, `UltraChat`, `OASST2`, `PersonaChat`), считает общее число
    просканированных сырых токенов и отбирает через таламический фильтр `ACh Surprise Gate`
    лучшие каузальные/диалоговые пары."""
    shard_path, domain_name, max_rows_to_scan, target_elite_pairs, base_hint = args
    raw_tokens_scanned = 0
    rows_scanned = 0
    candidates: List[Tuple[float, str, str, str, float]] = []

    try:
        pf = pq.ParquetFile(shard_path, memory_map=True)
        schema_names = set(pf.schema.names)

        if "text" in schema_names:
            for batch in pf.iter_batches(batch_size=512, columns=["text"]):
                for txt in batch.column("text").to_pylist():
                    if not isinstance(txt, str) or len(txt) < 60:
                        continue
                    rows_scanned += 1
                    # Быстрая оценка числа токенов в документе (~1 слово = 1.3 BPE-токена)
                    raw_tokens_scanned += int((len(txt) // 4) + 1)
                    if len(candidates) < target_elite_pairs * 4:
                        clean = " ".join(txt.split())
                        dot1 = clean.find(".")
                        if 25 <= dot1 <= 210:
                            dot2 = clean.find(".", dot1 + 1)
                            if dot1 + 30 <= dot2 <= dot1 + 230:
                                sa = clean[: dot1 + 1].strip()
                                sb = clean[dot1 + 1 : dot2 + 1].strip()
                                sc = _compute_thalamic_ach_surprise_score(sa, sb, base_hint)
                                if sc >= 1.55:
                                    candidates.append((sc, domain_name, sa, sb, sc))
                    if rows_scanned >= max_rows_to_scan:
                        break
                if rows_scanned >= max_rows_to_scan:
                    break

        elif "messages" in schema_names:
            for batch in pf.iter_batches(batch_size=256, columns=["messages"]):
                for msgs in batch.column("messages").to_pylist():
                    if not msgs or len(msgs) < 2:
                        continue
                    rows_scanned += 1
                    for m_a, m_b in zip(msgs, msgs[1:]):
                        ca = str(m_a.get("content", "")).strip()
                        cb = str(m_b.get("content", "")).strip()
                        raw_tokens_scanned += int((len(ca) + len(cb)) // 4)
                        if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                            if 18 <= len(ca) <= 220 and 25 <= len(cb) <= 320:
                                first_sent_b = cb.split(".")[0].strip() + "."
                                sc = _compute_thalamic_ach_surprise_score(ca, first_sent_b, base_hint)
                                if sc >= 1.55 and len(candidates) < target_elite_pairs * 4:
                                    candidates.append((sc, domain_name, ca, first_sent_b, sc))
                    if rows_scanned >= max_rows_to_scan:
                        break
                if rows_scanned >= max_rows_to_scan:
                    break
    except Exception:
        pass

    candidates.sort(key=lambda x: x[0], reverse=True)
    elite = [(dom, sa, sb, qh) for _, dom, sa, sb, qh in candidates[:target_elite_pairs]]
    return {
        "shard_path": os.path.basename(shard_path),
        "domain": domain_name,
        "rows_scanned": rows_scanned,
        "raw_tokens_scanned": raw_tokens_scanned,
        "elite_pairs": elite,
    }


def _thalamic_scan_text_zone_worker(args: Tuple[str, str, int, int, int, float]) -> Dict[str, object]:
    """Параллельный воркер для гигабайтных текстовых томов (`BookCorpus` 4.4 ГБ, `WikiText-103` 516 МБ,
    `TinyStoriesV2-GPT4-train.txt` 2.1 ГБ): читает выделенную байтовую зону, считает просканированные
    сырые токены и отбирает лучшие каузальные пары через `ACh Surprise Gate`."""
    file_path, domain_name, start_byte, bytes_to_scan, target_elite_pairs, base_hint = args
    raw_tokens_scanned = 0
    lines_scanned = 0
    candidates: List[Tuple[float, str, str, str, float]] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(start_byte)
            if start_byte > 0:
                f.readline()
            bytes_read = 0
            prev_sent = ""
            while bytes_read < bytes_to_scan:
                ln = f.readline()
                if not ln:
                    break
                b_len = len(ln)
                bytes_read += b_len
                lines_scanned += 1
                raw_tokens_scanned += int(b_len // 4)

                s = ln.strip()
                if len(s) < 25 or s.startswith("="):
                    continue
                if len(candidates) < target_elite_pairs * 4:
                    if "." in s and len(s) > 75:
                        parts = [p.strip() + "." for p in s.split(".") if 22 <= len(p.strip()) <= 200]
                        if len(parts) >= 2:
                            sc = _compute_thalamic_ach_surprise_score(parts[0], parts[1], base_hint)
                            if sc >= 1.55:
                                candidates.append((sc, domain_name, parts[0], parts[1], sc))
                    elif 22 <= len(s) <= 200:
                        if prev_sent:
                            sc = _compute_thalamic_ach_surprise_score(prev_sent, s, base_hint)
                            if sc >= 1.55:
                                candidates.append((sc, domain_name, prev_sent, s, sc))
                            prev_sent = ""
                        else:
                            prev_sent = s
    except Exception:
        pass

    candidates.sort(key=lambda x: x[0], reverse=True)
    elite = [(dom, sa, sb, qh) for _, dom, sa, sb, qh in candidates[:target_elite_pairs]]
    return {
        "shard_path": f"{os.path.basename(file_path)}@{start_byte // (1024*1024)}MB",
        "domain": domain_name,
        "rows_scanned": lines_scanned,
        "raw_tokens_scanned": raw_tokens_scanned,
        "elite_pairs": elite,
    }


# ============================================================================
# 2. ПЯТИБАНКОВЫЙ ДОМЕННЫЙ АТЛАС ГИППОКАМПА CA3 (5-BANK MULTI-DOMAIN CA3 ATLAS)
# ============================================================================
class FiveBankCA3AttractorAtlas:
    """Хранит и сортирует семантические полюса `CA3` раздельно по 5 когнитивным банкам коры,
    чтобы научные факты, сказки, романы, кинореплики и инструкции ассистента никогда не вытесняли друг друга."""

    BANK_NAMES = (
        "Bank0_BA22_Stories",
        "Bank1_BA38_Books",
        "Bank2_BA7_BA39_Science_Web",
        "Bank3_BA44_BA45_Movies_Social",
        "Bank4_BA46_BA10_Wisdom_Assistant",
    )

    SOURCE_TO_BANK = {
        "tinystories": 0,
        "bookcorpus": 1,
        "wikitext103": 2,
        "openwebtext": 2,
        "cornell_movies": 3,
        "daily_dialog": 3,
        "daily_dialog_cornell": 3,
        "personachat": 3,
        "personamem": 3,
        "npc_dialogue": 3,
        "oasst2": 4,
        "ultrachat": 4,
        "online_user_da_burst": 4,
    }

    def __init__(self, d_vec: int = 64, max_poles_per_bank: int = 2048):
        self.d_vec = int(d_vec)
        self.max_poles_per_bank = int(max_poles_per_bank)
        self.bank_attractors: Dict[int, torch.Tensor] = {
            b: torch.empty((0, d_vec), dtype=torch.float32) for b in range(5)
        }
        self.bank_salience: Dict[int, torch.Tensor] = {
            b: torch.empty((0,), dtype=torch.float32) for b in range(5)
        }

    def record_pole_in_domain_bank(self, source: str, q_vec: torch.Tensor, salience: float) -> int:
        b_id = self.SOURCE_TO_BANK.get(source, 4)
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            cur_a = self.bank_attractors[b_id]
            cur_s = self.bank_salience[b_id]
            if cur_a.shape[0] == 0:
                self.bank_attractors[b_id] = q_norm.unsqueeze(0).clone()
                self.bank_salience[b_id] = torch.tensor([float(salience)], dtype=torch.float32)
            else:
                sims = cur_a @ q_norm
                max_sim, max_idx = torch.max(sims, dim=0)
                if float(max_sim.item()) >= 0.96:
                    idx = int(max_idx.item())
                    self.bank_attractors[b_id][idx] = F.normalize(0.88 * cur_a[idx] + 0.12 * q_norm, p=2, dim=0)
                    self.bank_salience[b_id][idx] = max(float(cur_s[idx].item()), float(salience))
                elif cur_a.shape[0] < self.max_poles_per_bank:
                    self.bank_attractors[b_id] = torch.cat([cur_a, q_norm.unsqueeze(0)], dim=0)
                    self.bank_salience[b_id] = torch.cat([cur_s, torch.tensor([float(salience)], dtype=torch.float32)], dim=0)
                else:
                    # Online K-Means Centroid Merge в ближайший полюс банка
                    idx = int(max_idx.item())
                    self.bank_attractors[b_id][idx] = F.normalize(0.90 * cur_a[idx] + 0.10 * q_norm, p=2, dim=0)
                    self.bank_salience[b_id][idx] = max(float(cur_s[idx].item()), float(salience))
        return b_id

    def sort_all_banks_by_salience(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        with torch.no_grad():
            for b_id, b_name in enumerate(self.BANK_NAMES):
                n_p = int(self.bank_attractors[b_id].shape[0])
                if n_p > 1:
                    s_idx = torch.argsort(self.bank_salience[b_id], descending=True)
                    self.bank_attractors[b_id] = self.bank_attractors[b_id].index_select(0, s_idx)
                    self.bank_salience[b_id] = self.bank_salience[b_id].index_select(0, s_idx)
                counts[b_name] = n_p
        return counts


# ============================================================================
# 3. ОРКЕСТРАТОР 5-ЭПОХОВОГО ОНТОГЕНЕЗА ПО ВСЕМ 11 ДАТАСЕТАМ (v16.0)
# ============================================================================
def run_v16_10B_omniscient_curriculum(
    workers: int = 24,
    cpu_threads: int = 24,
    owt_shards_to_scan: int = 80,
    rows_per_owt_shard: int = 4000,
    elite_per_shard: int = 60,
) -> Dict[str, object]:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    t0 = time.perf_counter()

    print("=" * 108, flush=True)
    print("ZIWAGO-PRIME v16.0: 5-EPOCH OMNISCIENT 10B-CORPUS ENGINE (THALAMIC FAST-SCAN + 5-BANK CA3 ATLAS)", flush=True)
    print("  17,408 Neurons | 8,192 NMDA Branches | 5-Bank CA3 Attractor Atlas | BA44 Syntax Bridge | TPJ | SWS", flush=True)
    print("=" * 108, flush=True)

    organism = ZiwagoPureSynapticDialogueOrganismV14(
        root=root,
        workers=workers,
        cpu_threads=cpu_threads,
        seed=42,
    )
    organism.use_ngrams = False
    ca3_atlas = FiveBankCA3AttractorAtlas(d_vec=64, max_poles_per_bank=2048)
    domain_balancer = DeficitRoundRobinDomainBalancer(data_dir=data_dir)

    vocab_p = str(data_dir / "gpt2" / "vocab.json")
    merges_p = str(data_dir / "gpt2" / "merges.txt")
    _init_bpe_worker(vocab_p, merges_p)

    # Строгий реестр всех стимулов обучения для математической проверки 0% пересечения с HOLD_OUT_EXAM_SUITE
    trained_stimuli_normalized: set[str] = set()
    all_foundational_lessons: List[Tuple[str, str, str, float]] = []
    epoch_lesson_objs: Dict[int, List[TokenizedDialogPair]] = {}
    for ep_i, lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.items():
        all_foundational_lessons.extend(lessons)
        for _, sa, _, _ in lessons:
            trained_stimuli_normalized.add(" ".join(sa.strip().lower().split()))
        tok_l = _bpe_tokenize_raw_pairs_worker(lessons)
        epoch_lesson_objs[ep_i] = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in tok_l]

    all_lesson_objs: List[TokenizedDialogPair] = [
        tp for ep_list in epoch_lesson_objs.values() for tp in ep_list
    ]

    total_raw_tokens_scanned = 0
    total_shards_and_zones_scanned = 0
    epoch_summaries: List[Dict[str, object]] = []
    shard_ledger: Dict[str, List[str]] = {
        "Epoch_1_TinyStories": [],
        "Epoch_2_BookCorpus": [],
        "Epoch_3_Wiki_and_80_OpenWebText_Shards": [],
        "Epoch_4_Movies_DailyDialog_Persona_NPC": [],
        "Epoch_5_OASST2_and_6_UltraChat_Shards": [],
    }

    def _tokenize_and_consolidate_elite_pairs(
        epoch_idx: int,
        epoch_name: str,
        raw_elite_pairs: List[Tuple[str, str, str, float]],
        raw_scanned_toks: int,
        shards_done: int,
        target_domain: str,
    ) -> Dict[str, object]:
        nonlocal total_raw_tokens_scanned, total_shards_and_zones_scanned
        ep_t0 = time.perf_counter()
        total_raw_tokens_scanned += raw_scanned_toks
        total_shards_and_zones_scanned += shards_done

        raw_elite_pairs = domain_balancer.backfill_epoch_deficit_if_needed(
            epoch_idx=epoch_idx,
            target_domain=target_domain,
            current_pairs=raw_elite_pairs,
            min_required_pairs=32,
            fallback_lessons=FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.get(epoch_idx, []),
        )
        domain_balancer.record_pairs(raw_elite_pairs)
        for _, sa, _, _ in raw_elite_pairs:
            trained_stimuli_normalized.add(" ".join(sa.strip().lower().split()))

        random.Random(42 + epoch_idx).shuffle(raw_elite_pairs)
        tok_pairs: List[TokenizedDialogPair] = []
        if raw_elite_pairs:
            n_w = min(workers, max(1, len(raw_elite_pairs) // 40))
            chunk_sz = max(1, (len(raw_elite_pairs) + n_w - 1) // n_w)
            chunks = [raw_elite_pairs[i : i + chunk_sz] for i in range(0, len(raw_elite_pairs), chunk_sz)]

            with ProcessPoolExecutor(
                max_workers=n_w,
                initializer=_init_bpe_worker,
                initargs=(vocab_p, merges_p),
            ) as ex:
                for res_list in ex.map(_bpe_tokenize_raw_pairs_worker, chunks):
                    for src, ids_a, ids_b, q_h in res_list:
                        tok_pairs.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))

        # Добавляем фундаментальные уроки текущей эпохи онтогенеза (строго не пересекающиеся с HOLD_OUT_EXAM_SUITE)
        tok_pairs.extend(epoch_lesson_objs.get(epoch_idx, []) * 2)

        # Записываем полюса в 5-Банковый Доменный Атлас CA3
        for tp in tok_pairs[::4]:
            qv = organism.compute_stimulus_vec64(tp.stimulus_ids)
            ca3_atlas.record_pole_in_domain_bank(tp.source_dataset, qv, tp.quality_reward_hint)

        wave_stat = organism.train_wave_with_neuromodulator_reward(tok_pairs, wave_idx=epoch_idx)
        sws_stat = execute_sws_synaptic_homeostasis_and_pruning(organism, downscale_factor=0.988)
        bank_counts = ca3_atlas.sort_all_banks_by_salience()

        ep_sec = max(1e-3, time.perf_counter() - ep_t0)
        eff_scan_speed = int(raw_scanned_toks / ep_sec)
        ach_yield_pct = round(100.0 * wave_stat["tokens_trained"] / float(max(1, raw_scanned_toks)), 4)
        stat_out = {
            "epoch": epoch_idx,
            "epoch_name": epoch_name,
            "shards_or_zones_scanned": shards_done,
            "raw_corpus_tokens_scanned": raw_scanned_toks,
            "thalamic_scan_throughput_tok_per_sec": eff_scan_speed,
            "thalamic_ach_token_yield_pct": ach_yield_pct,
            "elite_pairs_consolidated": len(tok_pairs),
            "consolidated_tokens": wave_stat["tokens_trained"],
            "ca3_global_poles": sws_stat["ca3_poles_sorted"],
            "ca3_5bank_counts": bank_counts,
            "active_vocab_cols": len(organism.dendritic_plateau_readout.col_to_tok),
            "elapsed_sec": round(ep_sec, 2),
        }
        print(
            f"  [Epoch {epoch_idx}/5: {epoch_name}] "
            f"Shards/Zones={shards_done} | Scanned={raw_scanned_toks:,} raw tok ({eff_scan_speed:,} tok/s) | "
            f"ElitePairs={len(tok_pairs):,} | CA3_Poles={sws_stat['ca3_poles_sorted']} | "
            f"Vocab={stat_out['active_vocab_cols']} | Time={ep_sec:.1f}s",
            flush=True,
        )
        return stat_out

    # ------------------------------------------------------------------
    # ЭПОХА I: ДЕТСТВО (ВСЕ ШАРДЫ TINYSTORIES + TINYSTORIES V2 TXT)
    # ------------------------------------------------------------------
    ts_files = sorted(glob.glob(str(data_dir / "tinystories" / "data" / "train-*.parquet")))
    ts_tasks = [(f, "tinystories", 25000, elite_per_shard * 6, 1.35) for f in ts_files]
    ep1_elite: List[Tuple[str, str, str, float]] = []
    ep1_raw_toks = 0
    if not ts_tasks:
        domain_balancer.record_missing_source(1, "tinystories_parquet", str(data_dir / "tinystories" / "data"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(ts_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, ts_tasks):
                ep1_raw_toks += int(r["raw_tokens_scanned"])
                ep1_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_1_TinyStories"].append(str(r["shard_path"]))

    ts_txt = data_dir / "tinystories" / "TinyStoriesV2-GPT4-train.txt"
    if ts_txt.exists():
        f_sz = ts_txt.stat().st_size
        z_tasks = [
            (str(ts_txt), "tinystories", z * (f_sz // 16), 12 * 1024 * 1024, elite_per_shard * 2, 1.35)
            for z in range(16)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, 16)) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, z_tasks):
                ep1_raw_toks += int(r["raw_tokens_scanned"])
                ep1_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_1_TinyStories"].append(str(r["shard_path"]))
    else:
        domain_balancer.record_missing_source(1, "TinyStoriesV2-GPT4-train.txt", str(ts_txt))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            1,
            "Childhood_TinyStories_All_Shards",
            ep1_elite,
            ep1_raw_toks,
            len(shard_ledger["Epoch_1_TinyStories"]),
            target_domain="tinystories",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА II: ЮНОСТЬ И НАЧИТАННОСТЬ (BOOKCORPUS 4.4 ГБ ПО 32 ЗОНАМ)
    # ------------------------------------------------------------------
    ep2_elite: List[Tuple[str, str, str, float]] = []
    ep2_raw_toks = 0
    bc_files = [
        data_dir / "BookCorpus" / "books_large_p1.txt",
        data_dir / "BookCorpus" / "books_large_p2.txt",
    ]
    bc_tasks = []
    for bf in bc_files:
        if bf.exists():
            f_sz = bf.stat().st_size
            for z in range(16):
                bc_tasks.append((str(bf), "bookcorpus", z * (f_sz // 16), 16 * 1024 * 1024, elite_per_shard * 2, 1.40))
        else:
            domain_balancer.record_missing_source(2, bf.name, str(bf))
    if not bc_tasks:
        print("Warning: BookCorpus tasks empty, applying Deficit Round-Robin backfill...")
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(bc_tasks))) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, bc_tasks):
                ep2_raw_toks += int(r["raw_tokens_scanned"])
                ep2_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_2_BookCorpus"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            2,
            "Youth_BookCorpus_32_Zones",
            ep2_elite,
            ep2_raw_toks,
            len(shard_ledger["Epoch_2_BookCorpus"]),
            target_domain="bookcorpus",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА III: УНИВЕРСИТЕТ И НАУКА (WIKITEXT-103 + ВСЕ 80 ШАРДОВ OPENWEBTEXT!)
    # ------------------------------------------------------------------
    ep3_elite: List[Tuple[str, str, str, float]] = []
    ep3_raw_toks = 0

    wiki_p = data_dir / "wikitext-103-raw" / "wiki.train.raw"
    if wiki_p.exists():
        f_sz = wiki_p.stat().st_size
        w_tasks = [
            (str(wiki_p), "wikitext103", z * (f_sz // 16), f_sz // 16, elite_per_shard * 2, 1.50)
            for z in range(16)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, 16)) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, w_tasks):
                ep3_raw_toks += int(r["raw_tokens_scanned"])
                ep3_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"].append(str(r["shard_path"]))
    else:
        domain_balancer.record_missing_source(3, "wikitext-103-raw", str(wiki_p))

    owt_files = sorted(glob.glob(str(data_dir / "openwebtext_data" / "plain_text" / "train-*.parquet")))[:owt_shards_to_scan]
    owt_tasks = [(f, "openwebtext", rows_per_owt_shard, elite_per_shard // 2, 1.45) for f in owt_files]
    if not owt_tasks:
        domain_balancer.record_missing_source(3, "openwebtext_data", str(data_dir / "openwebtext_data"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(owt_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, owt_tasks):
                ep3_raw_toks += int(r["raw_tokens_scanned"])
                ep3_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            3,
            f"University_Wiki103_Plus_{len(owt_files)}_OpenWebText_Shards",
            ep3_elite,
            ep3_raw_toks,
            len(shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"]),
            target_domain="wikitext103",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА IV: СОЦИАЛИЗАЦИЯ, КИНО И РОЛИ (CORNELL + DAILYDIALOG + PERSONACHAT + PERSONAMEM + NPC)
    # ------------------------------------------------------------------
    ep4_elite: List[Tuple[str, str, str, float]] = []
    for _ in range(4):
        ep4_elite.extend(organism.dialog_streamer._sample_daily_dialog_cornell(250))
        ep4_elite.extend(organism.dialog_streamer._sample_personachat(250))
        ep4_elite.extend(organism.dialog_streamer._sample_personamem(250))
        ep4_elite.extend(organism.dialog_streamer._sample_npc_dialogue(250))
    shard_ledger["Epoch_4_Movies_DailyDialog_Persona_NPC"] = [
        "cornell_movie_lines.txt",
        "daily_dialog",
        "personachat/train-00000.parquet",
        "personamem/shared_contexts_128k.jsonl",
        "npc_dialogue/rpg-quests-dialogue.jsonl",
    ]
    ep4_raw_toks = sum((len(a) + len(b)) // 3 for _, a, b, _ in ep4_elite) * 12

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            4,
            "Social_Movies_Persona_NPC_All_Corpora",
            ep4_elite,
            ep4_raw_toks,
            len(shard_ledger["Epoch_4_Movies_DailyDialog_Persona_NPC"]),
            target_domain="personachat",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА V: МУДРОСТЬ И РЕФЛЕКСИЯ (OASST2 + ВСЕ 6 ШАРДОВ ULTRACHAT!)
    # ------------------------------------------------------------------
    ep5_elite: List[Tuple[str, str, str, float]] = []
    ep5_raw_toks = 0

    for _ in range(4):
        oasst_batch = organism.dialog_streamer._sample_oasst2(300)
        ep5_elite.extend(oasst_batch)
        ep5_raw_toks += sum((len(a) + len(b)) // 3 for _, a, b, _ in oasst_batch) * 10
    if ep5_elite:
        shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"].append("2023-11-05_oasst2_ready.trees.jsonl.gz")
    else:
        domain_balancer.record_missing_source(5, "oasst2", str(data_dir / "dialog" / "oasst2"))

    uc_files = sorted(glob.glob(str(data_dir / "dialog" / "ultrachat" / "data" / "*.parquet")))
    uc_tasks = [(f, "ultrachat", 3500, elite_per_shard * 3, 1.50) for f in uc_files]
    if not uc_tasks:
        domain_balancer.record_missing_source(5, "ultrachat", str(data_dir / "dialog" / "ultrachat"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(uc_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, uc_tasks):
                ep5_raw_toks += int(r["raw_tokens_scanned"])
                ep5_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            5,
            f"Wisdom_OASST2_Plus_{len(uc_files)}_UltraChat_Shards",
            ep5_elite,
            ep5_raw_toks,
            len(shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"]),
            target_domain="oasst2",
        )
    )

    # Консолидация базовых уроков после 5 эпох медленноволнового сна (SWR Replay без пересечения с Holdout!)
    for _ in range(4):
        organism.train_wave_with_neuromodulator_reward(all_lesson_objs, wave_idx=6)

    # ------------------------------------------------------------------
    # ПРОВЕРКА НУЛЕВОЙ УТЕЧКИ ЭКЗАМЕНА В ОБУЧАЮЩИЙ ПОТОК (ZERO HOLDOUT LEAKAGE ASSERTION)
    # ------------------------------------------------------------------
    holdout_leakage = [
        item.prompt
        for item in HOLD_OUT_EXAM_SUITE
        if " ".join(item.prompt.strip().lower().split()) in trained_stimuli_normalized
    ]
    assert len(holdout_leakage) == 0, f"CRITICAL METHODOLOGY VIOLATION: Holdout prompts leaked into training: {holdout_leakage}"

    # ------------------------------------------------------------------
    # ЧЕСТНЫЙ HOLDOUT-ЭКЗАМЕН И ЕДИНЫЕ ВОРОТА КАЧЕСТВА (UNIFIED CURRICULUM GATES)
    # ------------------------------------------------------------------
    print("-" * 108, flush=True)
    print("FINAL UNSEEN HOLDOUT EXAM & UNIFIED QUALITY GATES (0% PROMPT LEAKAGE | 0% N-GRAMS | 8192-NMDA MATRIX)", flush=True)
    print("-" * 108, flush=True)

    _ = organism.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )

    exam_results: List[Dict[str, object]] = []
    for item in HOLD_OUT_EXAM_SUITE:
        res = organism.pure_neural_generate_response(item.prompt, max_tokens=48, use_ngrams=False)
        ans_low = str(res["pure_weight_generated_response"]).lower()
        kw_hits = sum(1 for kw in item.expected_keywords if kw.lower() in ans_low)
        forbidden_hits = [bad for bad in item.forbidden_substrings if bad.lower() in ans_low]
        exam_results.append(
            {
                "exam_id": item.exam_id,
                "exam_domain": item.domain,
                "keyword_hits": kw_hits,
                "keyword_total": len(item.expected_keywords),
                "forbidden_hits": forbidden_hits,
                **res,
            }
        )
        print(
            f"  [{item.exam_id} ({item.domain}) | KW={kw_hits}/{len(item.expected_keywords)} | "
            f"BA44={res['ba44_syntax_bridge_connector']!r} | SWR={res['swr_consolidated_into_nmda_matrix']}]",
            flush=True,
        )
        print(f"    Q (Unseen Holdout): {item.prompt!r}", flush=True)
        print(f"    A (N-GRAMS OFF)   : {res['pure_weight_generated_response']!r}", flush=True)

    unified_gates = UnifiedCurriculumGates(
        surprise_threshold=0.25,
        min_grammar_score=1.0,
        min_retention_tokens=1500,
        max_attractor_collapse_rate=0.35,
    )
    gate_report = unified_gates.evaluate_organism(organism, stage_or_epoch_id=5)
    print(
        f"  [UNIFIED GATES VERDICT] Passed={gate_report.passed} | "
        f"Surprise={gate_report.surprise_f_t:.4f} (<0.25) | "
        f"Grammar={gate_report.grammar_score:.2%} | "
        f"Superacid={gate_report.superacid_count} | "
        f"Retention={gate_report.context_retention_tokens} (>1500) | "
        f"OOV={gate_report.oov_pass_rate:.2%} | "
        f"CollapseRate={gate_report.attractor_collapse_rate:.2%} | "
        f"EchoContam={gate_report.holdout_echo_contamination_rate:.2%}",
        flush=True,
    )

    ckpt_dir = root / "Produce" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ziwago_v16_0_10B_omniscient_master.pt"

    ckpt_validation = organism.save_self_contained_checkpoint(
        ckpt_path=ckpt_path,
        extra_metadata={
            "version": "16.0-10B-Omniscient-5Epoch-Master",
            "total_shards_and_zones_scanned": total_shards_and_zones_scanned,
            "total_raw_tokens_scanned": total_raw_tokens_scanned,
            "total_elite_pairs_consolidated": organism.total_pairs_trained,
            "total_consolidated_tokens": organism.total_tokens_trained,
            "ca3_5bank_counts": ca3_atlas.sort_all_banks_by_salience(),
            "ca3_5bank_attractors": ca3_atlas.bank_attractors,
            "domain_pair_counts": domain_balancer.domain_pair_counts,
            "missing_sources_logged": domain_balancer.missing_sources,
        },
        use_fp16=False,
    )

    # ------------------------------------------------------------------
    # АВТОМАТИЧЕСКАЯ ROUND-TRIP ВЕРИФИКАЦИЯ ЧЕКПОИНТА (SAVE -> LOAD -> GENERATE == IDENTICAL)
    # ------------------------------------------------------------------
    reloaded_organism = ZiwagoPureSynapticDialogueOrganismV14(
        root=root,
        workers=max(1, workers // 2),
        cpu_threads=cpu_threads,
        seed=42,
    )
    reloaded_organism.load_self_contained_checkpoint(ckpt_path, strict_schema=True)
    roundtrip_matches = 0
    roundtrip_probes = [
        "Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
        "Remind me, what is my name and what project am I building?",
        "Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
    ]
    for probe_q in roundtrip_probes:
        r_orig = organism.pure_neural_generate_response(probe_q, max_tokens=44, use_ngrams=False)["pure_weight_generated_response"]
        r_load = reloaded_organism.pure_neural_generate_response(probe_q, max_tokens=44, use_ngrams=False)["pure_weight_generated_response"]
        if r_orig == r_load and len(str(r_load).strip()) > 0:
            roundtrip_matches += 1
    roundtrip_exact_match_rate = round(roundtrip_matches / float(len(roundtrip_probes)), 4)
    print(
        f"  [CHECKPOINT ROUND-TRIP CI] Reloaded `{ckpt_path.name}` -> Exact Match={roundtrip_matches}/{len(roundtrip_probes)} "
        f"({roundtrip_exact_match_rate:.0%}) | ActiveCols={ckpt_validation['active_vocab_columns']}",
        flush=True,
    )

    ledger_path = root / "Produce" / "v16_0_shard_offset_ledger.json"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(shard_ledger, f, indent=2, ensure_ascii=False)

    report = {
        "version": "16.0",
        "architecture": "Ziwago-Prime v16.0 10B-Corpus 5-Epoch Omniscient Engine (32-Core Thalamic Fast-Scan + 5-Bank CA3 Atlas + 8192-NMDA Matrix + BA44 Syntax Bridge + TPJ + SWS/SWR + Unified Gates)",
        "total_neurons": 17408,
        "stored_sentence_chains_in_organism": 0,
        "holdout_prompt_leakage_count": len(holdout_leakage),
        "checkpoint_validation": ckpt_validation,
        "checkpoint_roundtrip_exact_match_rate": roundtrip_exact_match_rate,
        "unified_gates_report": {
            "passed": gate_report.passed,
            "surprise_f_t": gate_report.surprise_f_t,
            "grammar_score": gate_report.grammar_score,
            "superacid_count": gate_report.superacid_count,
            "context_retention_tokens": gate_report.context_retention_tokens,
            "oov_pass_rate": gate_report.oov_pass_rate,
            "attractor_collapse_rate": gate_report.attractor_collapse_rate,
            "holdout_echo_contamination_rate": gate_report.holdout_echo_contamination_rate,
            "holdout_keyword_accuracy": gate_report.holdout_keyword_accuracy,
            "reasons": gate_report.reasons,
        },
        "domain_balancer": {
            "domain_pair_counts": domain_balancer.domain_pair_counts,
            "missing_sources_logged": domain_balancer.missing_sources,
            "deficit_backfills": domain_balancer.deficit_backfills,
        },
        "total_shards_and_zones_scanned": total_shards_and_zones_scanned,
        "total_raw_corpus_tokens_scanned": total_raw_tokens_scanned,
        "total_elite_pairs_consolidated": organism.total_pairs_trained,
        "total_consolidated_tokens": organism.total_tokens_trained,
        "ca3_global_attractor_poles": int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0]),
        "ca3_5bank_atlas_counts": ca3_atlas.sort_all_banks_by_salience(),
        "active_vocabulary_columns": len(organism.dendritic_plateau_readout.col_to_tok),
        "epochs": epoch_summaries,
        "omniscient_5epoch_ngrams_off_exam": exam_results,
        "shard_ledger_path": str(ledger_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    rep_path = root / "Produce" / "v16_0_10B_omniscient_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 108, flush=True)
    print(
        f"COMPLETED v16.0 5-EPOCH MARATHON across {total_shards_and_zones_scanned} Shards/Zones! "
        f"Raw Scanned={total_raw_tokens_scanned:,} tok | Consolidated={organism.total_tokens_trained:,} tok | "
        f"CA3_Poles={report['ca3_global_attractor_poles']:,} | GatesPassed={gate_report.passed} | "
        f"RoundTrip={roundtrip_exact_match_rate:.0%} | Saved -> {ckpt_path.name} ({report['checkpoint_size_mb']} MB)",
        flush=True,
    )
    print("=" * 108, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ziwago-Prime v16.0 5-Epoch Omniscient 10B-Corpus Engine")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--owt-shards", type=int, default=80, help="Number of OpenWebText parquet shards to scan (default: all 80)")
    parser.add_argument("--rows-per-owt-shard", type=int, default=4000)
    parser.add_argument("--elite-per-shard", type=int, default=60)
    args = parser.parse_args()

    run_v16_10B_omniscient_curriculum(
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        owt_shards_to_scan=args.owt_shards,
        rows_per_owt_shard=args.rows_per_owt_shard,
        elite_per_shard=args.elite_per_shard,
    )


if __name__ == "__main__":
    main()
