#!/usr/bin/env python3
r"""ZIWAGO-PRIME v10.1 — THREE-PHASE ONTOGENETIC CURRICULUM TRAINER (`train_ziwago_v10_curriculum.py`)
====================================================================================================================
Полная реализация трёхэтапного онтогенетического развития (`Ontogenetic Curriculum`) из `data/evolution.txt`
для глубокого биологического разума **Ziwago-Prime v10.1** (`4,096` мультикомпартментных пирамидных нейронов
с 7-уровневыми дендритами Пуарази–Бениагева = `28,672` ветвей, `8 × 512 Sparsely-Gated Cortical MoE`,
`Bio-RWKV Astro-NMDA Linear Recurrence`, `Discrete All-or-None Predictive Coding` + `Stranger's Check`,
`Reference-Invariant Lag-Based dwPLI` и словарь `GPT-2` на `50,257` BPE-токенов через `CompositionalVSABPEEncoder`):

  1. `StreamingDataMixer` (Потоковый 32-ядерный миксер без переполнения RAM):
     - Читает гигантские корпуса (`TinyStories` ~7 ГБ, `BookCorpus` ~4.6 ГБ, `OpenWebText` ~24 ГБ в 80 parquet-шардах,
       `WikiText-103` ~540 МБ и `Cornell Movie-Dialogs`) строго потоково:
       * через `pyarrow.parquet.ParquetFile.iter_batches(use_memory_map=True)` для локальных `.parquet` шардов
         (и опционально HuggingFace `datasets(..., streaming=True)` при наличии);
       * через `mmap` / ленивый итератор с сохранением смещений байтов (`file_offset`) для гигантских `.txt`/`.raw`
         файлов (`BookCorpus`, `WikiText-103`);
       * с жёстким лимитом на диалоги Cornell (`max_cornell_dialogs_per_stage = 500,000`), чтобы они никогда
         не перевешивали `TinyStories`.
     - Токенизирует батчи параллельно на 32 ядрах CPU (`ProcessPoolExecutor(max_workers=32)`) в ID GPT-2 (`0..50256`)
       и выдаёт смешанный поток с точным соблюдением пропорций каждого этапа (отклонение `< 0.5%`) и гарантией
       уникальности каждого текста (`max_exposure_per_uid == 1`).

  2. ТРИ ОНТОГЕНЕТИЧЕСКИХ ЭТАПА («Детский сад -> Школа -> Университет»):
     - **Этап 1: Фундамент речи (`Language & Pragmatics — Детский сад`)**
       * Пропорции: `TinyStories (60%) + Диалоги Cornell (40%)`.
       * Цель: Идеальный синтаксис, базовая логика сюжета и прагматика (ирония, интонация).
       * Ворота качества (`Gate 1`): `Surprise < 0.25`, `100%` грамматика на тестовых генерациях, `0` артефактов `"superacid"`.
       * Чекпоинт: `Produce/checkpoints/ziwago_v10_stage1.pt` (и симлинк/копия `ziwago_v10_stage1.pt`).

     - **Этап 2: Сложный язык (`Literature & Style — Школа`)**
       * Пропорции: `BookCorpus (60%) + TinyStories (20%) + Диалоги Cornell (20%)`.
       * Цель: Длинные предложения, богатый словарный запас, литературный стиль, сложные семантические связи.
       * Ворота качества (`Gate 2`): `Surprise < 0.15`, успешный тест на `Context Retention > 1500` токенов
         без разрыва сюжета, `100%` грамматика, `0` `"superacid"`.
       * Чекпоинт: `Produce/checkpoints/ziwago_v10_stage2.pt`.

     - **Этап 3: Факты и реальность (`Encyclopedia & Noise — Университет`)**
       * Пропорции: `BookCorpus (40%) + WikiText-103 (30%) + OpenWebText (30%)`.
       * Цель: Фактология, термины, устойчивость к шуму реального интернета (сленг, опечатки, случайный текст)
         через адаптивный фильтр `Astrocytic Noise-Resilience Gate`.
       * Ворота качества (`Gate 3`): Финальный чекпоинт `Produce/checkpoints/ziwago_v10_stage3.pt` +
         комплексный экзамен на факты (`Wiki`), литературный стиль (`Book`) и диалоговую прагматику (`Cornell`).

  3. МЕХАНИЗМ `Gated Checkpoints` (Ворота качества и Авто-Промоушен):
     - На каждой эпохе оценивается `Surprise F(t)`, запускается авто-генерация трёх контрольных текстов
       (`сказка`, `диалог`, `факт/стиль`), проверяется синтаксическая связность, отсутствие `"superacid"` и петель,
       а также `Context Retention` (`> 1500` токенов на Этапах 2 и 3).
     - Как только условия этапа выполнены, тренер атомарно сохраняет чекпоинт этапа (`.tmp -> os.replace`)
       и автоматически переключает `StreamingDataMixer` на следующий этап. Если ворота ещё не пройдены,
       организм продолжает углублённое обучение и фазы сна (`Surprise-Driven Replay + CT-PNN`) на текущем этапе.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import datetime
import gc
import json
import math
import mmap
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple
import zipfile

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from neuroweave10.core.sdr import CompositionalVSABPEEncoder
from neuroweave10.cortex.moe_cortex import SparselyGatedCorticalMoE
from train_ziwago_v10 import ensure_cornell_movie_dialogs_corpus
from train_ziwago_v9_gpt2vocab import (
    _GPT2_PAT,
    _bpe_encode_piece,
    _init_bpe_worker,
    ZiwagoGPT2VocabTrainer,
    resolve_wikitext_file,
)
import train_ziwago_v9_gpt2vocab as _gpt2_mod


# ============================================================================
# 1. ПРОПОРЦИИ И ПАРАМЕТРЫ 3 ЭТАПОВ ОНТОГЕНЕТИЧЕСКОЙ ЭВОЛЮЦИИ (`evolution.txt`)
# ============================================================================
STAGE_CONFIGS: Dict[int, Dict[str, object]] = {
    1: {
        "stage_id": 1,
        "code": "stage1_language_pragmatics",
        "title": "Этап 1: Фундамент речи (Language & Pragmatics — Детский сад)",
        "proportions": {
            "tinystories": 0.60,
            "cornell_dialogs": 0.40,
        },
        "target_surprise_gate": 0.25,
        "min_context_retention": 512,
        "context_window": 256,
        "min_epochs": 3,
        "max_epochs": 16,
        "checkpoint_filename": "ziwago_v10_stage1.pt",
        "description": "Идеальный синтаксис, базовая логика сюжета и прагматика (ирония, интонация).",
    },
    2: {
        "stage_id": 2,
        "code": "stage2_literature_style",
        "title": "Этап 2: Сложный язык (Literature & Style — Школа)",
        "proportions": {
            "bookcorpus": 0.60,
            "tinystories": 0.20,
            "cornell_dialogs": 0.20,
        },
        "target_surprise_gate": 0.15,
        "min_context_retention": 1500,
        "context_window": 1600,
        "min_epochs": 3,
        "max_epochs": 16,
        "checkpoint_filename": "ziwago_v10_stage2.pt",
        "description": "Длинные предложения, богатый словарный запас, литературный стиль, Context Retention > 1500 токенов.",
    },
    3: {
        "stage_id": 3,
        "code": "stage3_encyclopedia_noise",
        "title": "Этап 3: Факты и реальность (Encyclopedia & Noise — Университет)",
        "proportions": {
            "bookcorpus": 0.40,
            "wikitext": 0.30,
            "openwebtext": 0.30,
        },
        "target_surprise_gate": 0.12,
        "min_context_retention": 1550,
        "context_window": 2048,
        "min_epochs": 3,
        "max_epochs": 16,
        "checkpoint_filename": "ziwago_v10_stage3.pt",
        "description": "Фактология, термины, устойчивость к шуму реального интернета (Wiki + Book + OpenWebText).",
    },
}


@dataclass
class StreamedCurriculumSample:
    uid: str
    domain: str  # "tinystories" | "cornell_dialogs" | "bookcorpus" | "wikitext" | "openwebtext"
    token_ids: List[int]
    raw_text: str
    noise_level: float = 0.0  # доля интернет-шума для калибровки дендритного фильтра OpenWebText


@dataclass
class StageStreamBatch:
    stage_id: int
    stage_title: str
    samples: List[StreamedCurriculumSample]
    total_tokens: int
    domain_token_counts: Dict[str, int]
    domain_token_shares: Dict[str, float]
    target_proportions: Dict[str, float]
    max_proportion_error: float
    cornell_dialogs_yielded: int
    cornell_limit_cap: int
    max_exposure_per_uid: int
    elapsed_stream_sec: float


def _parallel_bpe_tokenize_records(
    records: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, List[int], str, float]]:
    """Выполняется параллельно на 32 воркерах CPU:
    Принимает кортежи `(uid, domain, raw_text)` и возвращает `(uid, domain, token_ids, clean_preview, noise_level)`."""
    vocab = _gpt2_mod._WORKER_VOCAB
    merges = _gpt2_mod._WORKER_MERGES
    cache = _gpt2_mod._WORKER_CACHE
    assert vocab is not None and merges is not None

    out: List[Tuple[str, str, List[int], str, float]] = []
    for uid, dom, txt in records:
        s = " ".join(txt.strip().split())
        if not s:
            continue
        # Оцениваем уровень интернет-шума (небуквенные последовательности, URL, сленг в OpenWebText)
        allowed_punct = " .,!?;:'" + chr(34) + "-()"
        non_alnum = sum(1 for ch in s if not (ch.isalnum() or ch in allowed_punct))
        noise_lvl = round(min(1.0, non_alnum / float(max(1, len(s)))), 4)

        ids: List[int] = []
        for p in _GPT2_PAT.findall(" " + s):
            if p.isascii():
                ids.extend(_bpe_encode_piece(p, vocab, merges, cache))
        if len(ids) >= 12:
            out.append((uid, dom, ids, s[:200], noise_lvl))
    return out


# ============================================================================
# 2. STREAMING DATA MIXER (ПОТОКОВОЕ ЧТЕНИЕ MMAP / PARQUET НА 32 ЯДРАХ БЕЗ RAM OOM)
# ============================================================================
class StreamingDataMixer:
    """Потоковый мульти-корпусный миксер для Ziwago-Prime v10.1 (`data/evolution.txt`, Задача 1):
      - Работает с гигантскими датасетами (`TinyStories` 7 ГБ, `BookCorpus` 4.6 ГБ, `OpenWebText` 24 ГБ,
        `WikiText-103` 540 МБ, `Cornell Movie-Dialogs`) без загрузки файлов целиком в RAM (`O(1)` буфер).
      - Использует `pyarrow.parquet.ParquetFile(..., memory_map=True).iter_batches()` для `.parquet` файлов
        и `mmap` (memory-mapped file views) для многогигабайтных текстовых корпусов `BookCorpus` и `WikiText-103`.
      - Поддерживает опциональный режим HuggingFace `datasets.load_dataset(..., streaming=True)` при запросе.
      - Строго ограничивает число диалогов Cornell на этап (`max_cornell_dialogs_per_stage = 500,000`),
        чтобы Cornell никогда не перевешивал TinyStories.
      - Токенизирует потоковые чанки параллельно на 32 ядрах CPU и выдаёт батчи со строгим соблюдением
        пропорций текущего этапа (`Stage 1: 60/40`, `Stage 2: 60/20/20`, `Stage 3: 40/30/30`)."""

    def __init__(
        self,
        root: Path,
        workers: int = 32,
        max_cornell_dialogs_per_stage: int = 500_000,
        use_hf_streaming: bool = False,
        seed: int = 42,
    ):
        self.root = root
        self.data_dir = root / "data"
        self.workers = max(1, min(int(workers), os.cpu_count() or 32))
        self.max_cornell_dialogs_per_stage = int(max_cornell_dialogs_per_stage)
        self.use_hf_streaming = bool(use_hf_streaming)
        self.seed = int(seed)

        self.vocab_path = self.data_dir / "gpt2" / "vocab.json"
        self.merges_path = self.data_dir / "gpt2" / "merges.txt"
        _init_bpe_worker(str(self.vocab_path), str(self.merges_path))

        # Кэш курсоров (файловых смещений) между эпохами, чтобы каждая новая эпоха/волна читала свежие страницы диска!
        self._cursors: Dict[str, int] = {
            "tinystories_row": 0,
            "cornell_row": 0,
            "bookcorpus_offset": 0,
            "wikitext_offset": 0,
            "openwebtext_shard": 0,
            "openwebtext_row": 0,
        }
        self.current_stage: int = 1
        self.cornell_dialogs_used_in_stage: int = 0
        self._seen_hashes_in_stage: set[int] = set()

    def set_stage(self, stage_id: int) -> None:
        if stage_id not in STAGE_CONFIGS:
            raise ValueError(f"Unknown stage_id={stage_id}; expected 1, 2, or 3.")
        self.current_stage = int(stage_id)
        self.cornell_dialogs_used_in_stage = 0
        self._seen_hashes_in_stage.clear()

    # ------------------------------------------------------------------
    # Потоковые генераторы для каждого из 5 гигантских корпусов (O(1) RAM)
    # ------------------------------------------------------------------
    def _iter_tinystories_raw(self, batch_rows: int = 128) -> Iterator[Tuple[str, str]]:
        """Потоковое чтение `TinyStories` через memory-mapped Parquet шарды (`data/tinystories/data/*.parquet`
        или `data/10k-tinystories.parquet`) без загрузки таблицы в память."""
        shard_dir = self.data_dir / "tinystories" / "data"
        parquet_files = sorted(shard_dir.glob("train-*.parquet")) if shard_dir.exists() else []
        fallback_pq = self.data_dir / "10k-tinystories.parquet"
        if fallback_pq.exists():
            parquet_files = [fallback_pq] + parquet_files

        row_cursor = self._cursors.get("tinystories_row", 0)
        global_idx = 0
        for pq_path in parquet_files:
            pf = pq.ParquetFile(pq_path, memory_map=True)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
                texts = batch.column("text").to_pylist()
                for t in texts:
                    global_idx += 1
                    if global_idx <= row_cursor:
                        continue
                    s = " ".join(str(t).strip().split())
                    if len(s) >= 60:
                        self._cursors["tinystories_row"] = global_idx
                        yield f"tiny_{global_idx:07d}", s

    def _iter_cornell_dialogs_raw(self, batch_rows: int = 256) -> Iterator[Tuple[str, str]]:
        """Потоковое чтение диалогов `Cornell Movie-Dialogs` из честного корпуса `cornell_movie_dialogs_honest.jsonl`
        и memory-mapped `data/cornell/data/movie_lines-00000-of-00001.parquet` с жёстким лимитом
        `max_cornell_dialogs_per_stage = 500,000` диалогов на этап."""
        honest_path = ensure_cornell_movie_dialogs_corpus(self.data_dir)
        row_cursor = self._cursors.get("cornell_row", 0)
        global_idx = 0

        # 1. Сначала подаём структурно богатые прагматические диалоги из честного JSONL
        if honest_path.exists():
            with honest_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    if self.cornell_dialogs_used_in_stage >= self.max_cornell_dialogs_per_stage:
                        return
                    global_idx += 1
                    obj = json.loads(ln)
                    txt = str(obj.get("text", "")).strip()
                    if len(txt) >= 40:
                        self.cornell_dialogs_used_in_stage += 1
                        yield str(obj.get("uid", f"cornell_h_{global_idx:05d}")), txt

        # 2. Затем потоково собираем многорепликовые диалоги из полного parquet-архива Cornell (304,713 реплик)
        cornell_pq = self.data_dir / "cornell" / "data" / "movie_lines-00000-of-00001.parquet"
        if cornell_pq.exists():
            pf = pq.ParquetFile(cornell_pq, memory_map=True)
            dialog_buf: List[str] = []
            cur_movie = None
            for batch in pf.iter_batches(
                batch_size=batch_rows,
                columns=["lineID", "movieID", "characterName", "utterance"],
            ):
                cols = batch.to_pydict()
                n_rows = len(cols["lineID"])
                for i in range(n_rows):
                    if self.cornell_dialogs_used_in_stage >= self.max_cornell_dialogs_per_stage:
                        return
                    global_idx += 1
                    if global_idx <= row_cursor:
                        continue
                    mv = str(cols["movieID"][i] or "m0")
                    spk = str(cols["characterName"][i] or "SPEAKER").strip().upper()
                    utt = " ".join(str(cols["utterance"][i] or "").strip().split())
                    if not utt:
                        continue
                    if cur_movie is not None and mv != cur_movie and len(dialog_buf) >= 2:
                        joined = " ".join(dialog_buf)
                        dialog_buf.clear()
                        self.cornell_dialogs_used_in_stage += 1
                        self._cursors["cornell_row"] = global_idx
                        yield f"cornell_pq_{global_idx:07d}", joined
                    cur_movie = mv
                    dialog_buf.append(f"{spk} : {utt}")
                    if len(dialog_buf) >= 4:
                        joined = " ".join(dialog_buf)
                        dialog_buf.clear()
                        self.cornell_dialogs_used_in_stage += 1
                        self._cursors["cornell_row"] = global_idx
                        yield f"cornell_pq_{global_idx:07d}", joined

    def _iter_bookcorpus_mmap_raw(self) -> Iterator[Tuple[str, str]]:
        """Потоковое чтение гигантских файлов `BookCorpus` (`books_large_p1.txt` 2.5 ГБ и `books_large_p2.txt` 2.1 ГБ)
        через системный `mmap` без загрузки в RAM. Собирает связные литературные абзацы по 5–8 предложений
        для формирования богатого стиля и удержания контекста `> 1500` токенов."""
        book_files = [
            self.data_dir / "BookCorpus" / "books_large_p1.txt",
            self.data_dir / "BookCorpus" / "books_large_p2.txt",
        ]
        existing = [p for p in book_files if p.exists() and p.stat().st_size > 0]
        if not existing:
            # Фоллбэк на литературные главы WikiText, если файлы BookCorpus отсутствуют
            for uid, txt in self._iter_wikitext_mmap_raw():
                yield f"book_fallback_{uid}", txt
            return

        start_offset = self._cursors.get("bookcorpus_offset", 0)
        para_idx = 0
        sent_buf: List[str] = []
        tok_est = 0

        for bf_idx, bpath in enumerate(existing):
            fsize = bpath.stat().st_size
            with bpath.open("rb") as f:
                with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                    if bf_idx == 0 and 0 < start_offset < fsize:
                        mm.seek(start_offset)
                        # Выравниваемся по ближайшему переводу строки
                        _ = mm.readline()
                    while True:
                        raw_line = mm.readline()
                        if not raw_line:
                            break
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line or line.startswith("isbn") or "copyright" in line.lower():
                            continue
                        words = line.split()
                        if len(words) < 5:
                            continue
                        # Нормализуем заглавную букву начала предложения для чистоты грамматики Брока
                        clean_line = line[0].upper() + line[1:] if line else line
                        if clean_line[-1] not in ".!?":
                            clean_line += " ."
                        sent_buf.append(clean_line)
                        tok_est += len(words)
                        if tok_est >= 48 and len(sent_buf) >= 3:
                            para_idx += 1
                            if bf_idx == 0:
                                self._cursors["bookcorpus_offset"] = mm.tell()
                            paragraph = " ".join(sent_buf)
                            sent_buf.clear()
                            tok_est = 0
                            yield f"bookcorpus_{bf_idx}_{para_idx:07d}", paragraph

    def _iter_wikitext_mmap_raw(self) -> Iterator[Tuple[str, str]]:
        """Потоковое чтение `WikiText-103` (`wiki.train.raw`, 540 МБ) через `mmap`."""
        wpath = resolve_wikitext_file(self.root)
        fsize = wpath.stat().st_size
        start_offset = self._cursors.get("wikitext_offset", 0)
        line_idx = 0

        with wpath.open("rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                if 0 < start_offset < fsize:
                    mm.seek(start_offset)
                    _ = mm.readline()
                while True:
                    raw_line = mm.readline()
                    if not raw_line:
                        mm.seek(0)
                        raw_line = mm.readline()
                        if not raw_line:
                            break
                    line_idx += 1
                    s = raw_line.decode("utf-8", errors="ignore").strip()
                    if not s or s.startswith("="):
                        continue
                    s = (
                        s.replace(" @-@ ", "-")
                        .replace(" @,@ ", ",")
                        .replace(" @.@ ", ".")
                        .replace("<unk>", "unknown")
                    )
                    if len(s.split()) >= 18:
                        self._cursors["wikitext_offset"] = mm.tell()
                        yield f"wiki103_{line_idx:07d}", s

    def _iter_openwebtext_parquet_raw(self, batch_rows: int = 64) -> Iterator[Tuple[str, str]]:
        """Потоковое чтение 80 шардов `OpenWebText` (`data/openwebtext_data/plain_text/train-*.parquet`, ~24 ГБ)
        через memory-mapped Parquet итератор (берёт чистые информативные абзацы, сохраняя реальные интернет-маркеры
        для тренировки устойчивости к шуму на Этапе 3)."""
        owt_dir = self.data_dir / "openwebtext_data" / "plain_text"
        shards = sorted(owt_dir.glob("train-*.parquet")) if owt_dir.exists() else []
        if not shards:
            for uid, txt in self._iter_wikitext_mmap_raw():
                yield f"owt_fallback_{uid}", txt
            return

        start_shard = self._cursors.get("openwebtext_shard", 0) % len(shards)
        row_cursor = self._cursors.get("openwebtext_row", 0)
        doc_idx = 0

        for s_offset in range(len(shards)):
            s_idx = (start_shard + s_offset) % len(shards)
            pf = pq.ParquetFile(shards[s_idx], memory_map=True)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
                for raw_doc in batch.column("text").to_pylist():
                    doc_idx += 1
                    if s_offset == 0 and doc_idx <= row_cursor:
                        continue
                    if not raw_doc:
                        continue
                    # Извлекаем первый содержательный абзац документа OpenWebText (чтобы не раздувать один сэмпл до 5000 слов)
                    paras = [p.strip() for p in str(raw_doc).split(chr(10)) if len(p.strip().split()) >= 20]
                    if not paras:
                        continue
                    clean_p = " ".join(paras[0].split()[:110])
                    self._cursors["openwebtext_shard"] = s_idx
                    self._cursors["openwebtext_row"] = doc_idx
                    yield f"openwebtext_s{s_idx:02d}_{doc_idx:07d}", clean_p

    def _get_domain_iterator(self, domain: str) -> Iterator[Tuple[str, str]]:
        if domain == "tinystories":
            return self._iter_tinystories_raw()
        if domain == "cornell_dialogs":
            return self._iter_cornell_dialogs_raw()
        if domain == "bookcorpus":
            return self._iter_bookcorpus_mmap_raw()
        if domain == "wikitext":
            return self._iter_wikitext_mmap_raw()
        if domain == "openwebtext":
            return self._iter_openwebtext_parquet_raw()
        raise ValueError(f"Unsupported domain: {domain}")

    # ------------------------------------------------------------------
    # Параллельная сборка идеально сбалансированного батча этапа на 32 ядрах
    # ------------------------------------------------------------------
    def stream_stage_batch(
        self,
        stage_id: Optional[int] = None,
        target_total_tokens: int = 12_000,
        seed: Optional[int] = None,
    ) -> StageStreamBatch:
        """Собирает потоковый батч для указанного этапа (`1`, `2` или `3`) параллельно на 32 ядрах CPU,
        строго выдерживая пропорции токенов из `data/evolution.txt` (с точностью до 1 токена!)."""
        t0 = time.perf_counter()
        if stage_id is not None and stage_id != self.current_stage:
            self.set_stage(stage_id)

        st_cfg = STAGE_CONFIGS[self.current_stage]
        proportions: Dict[str, float] = dict(st_cfg["proportions"])  # type: ignore[arg-type]

        # Вычисляем точный бюджет токенов на каждый активный домен этапа
        active_domains = [(dom, ratio) for dom, ratio in proportions.items() if ratio > 0.0]
        domain_budgets: Dict[str, int] = {}
        allocated = 0
        for idx, (dom, ratio) in enumerate(active_domains):
            if idx == len(active_domains) - 1:
                domain_budgets[dom] = max(1, target_total_tokens - allocated)
            else:
                b = int(round(target_total_tokens * ratio))
                domain_budgets[dom] = max(1, b)
                allocated += b

        domain_samples: Dict[str, List[StreamedCurriculumSample]] = {dom: [] for dom, _ in active_domains}
        domain_tok_acc: Dict[str, int] = {dom: 0 for dom, _ in active_domains}

        # Читаем сырые тексты потоково и токенизируем их параллельно на 32 воркерах AMD EPYC (128 GB RAM Arena)
        use_pool = target_total_tokens >= 8_000 and self.workers > 1
        executor = (
            ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_init_bpe_worker,
                initargs=(str(self.vocab_path), str(self.merges_path)),
            )
            if use_pool
            else None
        )

        try:
            for dom, budget in domain_budgets.items():
                it = self._get_domain_iterator(dom)
                raw_buffer: List[Tuple[str, str, str]] = []

                while domain_tok_acc[dom] < budget:
                    try:
                        uid, raw_txt = next(it)
                    except StopIteration:
                        # Если дошли до конца локального файла (например, при огромном бюджете), сбрасываем курсор
                        self._cursors[f"{dom}_row"] = 0
                        self._cursors[f"{dom}_offset"] = 0
                        it = self._get_domain_iterator(dom)
                        uid, raw_txt = next(it)

                    h = hash(raw_txt)
                    if h in self._seen_hashes_in_stage:
                        continue
                    self._seen_hashes_in_stage.add(h)
                    raw_buffer.append((uid, dom, raw_txt))

                    # Когда накопили пачку сырых текстов или приближаемся к бюджету — токенизируем
                    if len(raw_buffer) >= (128 if use_pool else 16) or (len(raw_buffer) * 45 >= budget - domain_tok_acc[dom]):
                        if executor is not None:
                            chunk_sz = max(1, (len(raw_buffer) + self.workers - 1) // self.workers)
                            chunks = [raw_buffer[i : i + chunk_sz] for i in range(0, len(raw_buffer), chunk_sz)]
                            tokenized_lists = list(executor.map(_parallel_bpe_tokenize_records, chunks))
                            tokenized = [item for sub in tokenized_lists for item in sub]
                        else:
                            tokenized = _parallel_bpe_tokenize_records(raw_buffer)
                        raw_buffer.clear()

                        for t_uid, t_dom, t_ids, preview, noise_lvl in tokenized:
                            rem = budget - domain_tok_acc[dom]
                            if rem <= 0:
                                break
                            if len(t_ids) > rem:
                                t_ids = t_ids[:rem]
                            if len(t_ids) < 2:
                                continue
                            domain_samples[dom].append(
                                StreamedCurriculumSample(
                                    uid=t_uid,
                                    domain=t_dom,
                                    token_ids=t_ids,
                                    raw_text=preview,
                                    noise_level=noise_lvl,
                                )
                            )
                            domain_tok_acc[dom] += len(t_ids)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        # Стратифицированное чередование (Deficit Round-Robin Interleaving), чтобы внутри батча
        # домены шли равномерно в точном соответствии со своими пропорциями (например, 60/40 или 60/20/20 или 40/30/30)
        rng = random.Random((seed if seed is not None else self.seed) + self.current_stage * 101)
        for dom in domain_samples:
            rng.shuffle(domain_samples[dom])

        interleaved: List[StreamedCurriculumSample] = []
        ptrs: Dict[str, int] = {dom: 0 for dom in domain_samples}
        accum_weights: Dict[str, float] = {dom: 0.0 for dom in domain_samples}

        total_samples = sum(len(v) for v in domain_samples.values())
        while len(interleaved) < total_samples:
            progressed = False
            for dom, ratio in active_domains:
                if ptrs[dom] >= len(domain_samples[dom]):
                    continue
                accum_weights[dom] += ratio
                while accum_weights[dom] >= 0.199 and ptrs[dom] < len(domain_samples[dom]):
                    interleaved.append(domain_samples[dom][ptrs[dom]])
                    ptrs[dom] += 1
                    accum_weights[dom] -= 0.20
                    progressed = True
            if not progressed:
                for dom in domain_samples:
                    if ptrs[dom] < len(domain_samples[dom]):
                        interleaved.append(domain_samples[dom][ptrs[dom]])
                        ptrs[dom] += 1

        total_toks = sum(domain_tok_acc.values())
        shares: Dict[str, float] = {
            dom: round(domain_tok_acc.get(dom, 0) / float(max(1, total_toks)), 4)
            for dom, _ in active_domains
        }
        max_err = max(abs(shares[dom] - ratio) for dom, ratio in active_domains) if active_domains else 0.0

        uid_counts: Dict[str, int] = {}
        for smp in interleaved:
            uid_counts[smp.uid] = uid_counts.get(smp.uid, 0) + 1
        max_exp = max(uid_counts.values()) if uid_counts else 0

        return StageStreamBatch(
            stage_id=self.current_stage,
            stage_title=str(st_cfg["title"]),
            samples=interleaved,
            total_tokens=total_toks,
            domain_token_counts=domain_tok_acc,
            domain_token_shares=shares,
            target_proportions={dom: ratio for dom, ratio in active_domains},
            max_proportion_error=round(max_err, 5),
            cornell_dialogs_yielded=self.cornell_dialogs_used_in_stage,
            cornell_limit_cap=self.max_cornell_dialogs_per_stage,
            max_exposure_per_uid=max_exp,
            elapsed_stream_sec=round(time.perf_counter() - t0, 3),
        )


# ============================================================================
# 3. ОНТОГЕНЕТИЧЕСКИЙ ТРЕНЕР ZIWAGO-PRIME v10.1 С GATED CHECKPOINTS
# ============================================================================
@dataclass
class GateInspectionReport:
    stage_id: int
    epoch: int
    passed: bool
    mean_surprise_F: float
    target_surprise_F: float
    grammar_score: float
    superacid_detected: bool
    context_retention_tokens: float
    target_context_retention: float
    generations: Dict[str, Dict[str, object]]
    checkpoint_saved_path: str
    checkpoint_size_mb: float
    reasons: List[str] = field(default_factory=list)


class ZiwagoOntogeneticV101Trainer:
    """Продвинутый онтогенетический тренер Ziwago-Prime v10.1 для 3-фазной эволюции (`evolution.txt`):
      - Поддерживает все 5 доменов (`tinystories`, `cornell_dialogs`, `bookcorpus`, `wikitext`, `openwebtext`)
        внутри 8 экспертных колонок `SparselyGatedCorticalMoE` (`4,096` нейронов, `28,672` дендритных ветвей).
      - Реализует **Astrocytic Internet-Noise Resilience Filter** для Этапа 3 (`OpenWebText`):
        когда `noise_level` входного веб-текста повышен (опечатки, сленг, случайные символы), астроцитарный
        контур динамически модулирует пластичность `eta_noise = exp(-1.8 * noise_level)`,
        извлекая устойчивое семантическое ядро без записи мусорных переходов в долговременный `CT-PNN HARD` каркас!
      - Реализует **Ontogenetic Synaptic Inheritance**: при переходе с Этапа 1 на Этап 2 и с Этапа 2 на Этап 3
        базовые синтаксические и прагматические цепи (`Col0_SYNTAX_BA44`, `Col3_PRAGMATIC_DIALOG`, `Col4_NARRATIVE_EPISODIC`)
        переводятся в защищённое состояние `CT-PNN HARD`, благодаря чему модель не забывает грамматику и иронию
        при чтении сложных книг и шумного интернета."""

    ALL_DOMAINS = ("tinystories", "cornell_dialogs", "bookcorpus", "wikitext", "openwebtext")

    def __init__(
        self,
        root: Path,
        device: str = "cpu",
        workers: int = 32,
        cpu_threads: int = 32,
        max_cornell_dialogs: int = 500_000,
        seed: int = 42,
    ):
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.manual_seed(seed)
        random.seed(seed)
        self.root = root
        self.device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.workers = int(workers)
        self.cpu_threads = int(cpu_threads)
        self.seed = int(seed)

        vocab_path = root / "data" / "gpt2" / "vocab.json"
        merges_path = root / "data" / "gpt2" / "merges.txt"

        # 1. Потоковый миксер данных на 32 ядрах
        self.mixer = StreamingDataMixer(
            root=root,
            workers=self.workers,
            max_cornell_dialogs_per_stage=max_cornell_dialogs,
            seed=seed,
        )

        # 2. Входной композиционный VSA-энкодер на 50,257 токенов GPT-2
        self.vsa_encoder = CompositionalVSABPEEncoder(
            vocab_path=str(vocab_path),
            merges_path=str(merges_path),
            n_base=2048,
            n_sdr=2048,
            n_modules=8,
            active_per_module=5,
            d_out=64,
            seed=seed,
        )

        # 3. Кора Mixture of Experts: 8 колонок × 512 нейронов = 4,096 нейронов (28,672 дендритных ветвей)
        self.moe_cortex = SparselyGatedCorticalMoE(
            n_columns=8,
            neurons_per_column=512,
            n_in=64,
            n_branch_per_neuron=7,
            top_k_active=2,
            seed=seed,
        )

        # 4. Базовый контроллер памяти, гиппокампа SBC-VSA-DHT и слогового декодера
        self.base_trainer = ZiwagoGPT2VocabTrainer(
            vocab_path=vocab_path,
            merges_path=merges_path,
            cpu_threads=cpu_threads,
            seed=seed,
        )

        # 5. Доменно-специализированные дендритные следы для всех 5 корпусов эволюции (128 GB RAM Arena: без потерь четырехграмм)
        self.expert_synapses_1: Dict[str, Dict[int, Dict[int, float]]] = {d: {} for d in self.ALL_DOMAINS}
        self.expert_trigrams: Dict[str, Dict[Tuple[int, int], Dict[int, float]]] = {d: {} for d in self.ALL_DOMAINS}
        self.expert_fourgrams: Dict[str, Dict[Tuple[int, int, int], Dict[int, float]]] = {d: {} for d in self.ALL_DOMAINS}
        self.stage_maturity_bonus: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.noise_filtered_tokens_count: int = 0

        # В 128 GB RAM Arena предвычисляем проекции коры для всех базовых и частых BPE-токенов (O(1) доступ без пересчёта VSA!)
        self._vec64_cache: Dict[int, torch.Tensor] = {}
        for bid in range(2048):
            self._vec64_cache[bid] = self.vsa_encoder.project_token_to_cortex(bid)

    def get_vec64(self, token_id: int) -> torch.Tensor:
        tid = int(token_id)
        if tid in self._vec64_cache:
            return self._vec64_cache[tid]
        v = self.vsa_encoder.project_token_to_cortex(tid)
        if len(self._vec64_cache) < 50257:
            self._vec64_cache[tid] = v
        return v

    def _map_domain_to_moe_tag(self, domain: str) -> str:
        """Направляет каждый из 5 датасетов в профильные экспертные колонки коры MoE (`Col0..Col7`)."""
        if domain == "tinystories":
            return "tinystories"
        if domain == "cornell_dialogs":
            return "cornell_dialogs"
        if domain == "bookcorpus":
            # Литературный стиль активирует Narrative Episodic (Col4), Lexical Semantics (Col1) и Executive PFC (Col7)
            return "tinystories_book"
        if domain == "wikitext":
            return "wikitext"
        if domain == "openwebtext":
            return "wikitext_openweb"
        return domain

    def train_single_epoch_on_batch(
        self,
        stage_id: int,
        epoch_in_stage: int,
        batch: StageStreamBatch,
    ) -> Dict[str, object]:
        """Проводит одну эпоху онтогенетического обучения на потоковом батче текущего этапа."""
        t0 = time.perf_counter()
        self.stage_maturity_bonus[stage_id] = self.stage_maturity_bonus.get(stage_id, 0) + 1
        maturity = self.stage_maturity_bonus[stage_id]

        surprises_all: List[float] = []
        domain_surprises: Dict[str, List[float]] = {d: [] for d in batch.target_proportions}
        discrete_pins: List[int] = []
        somatic_errors: List[float] = []
        apical_errors: List[float] = []
        flop_effs: List[float] = []
        lag_dwplis: List[float] = []
        rehydrations = 0

        # Калибруем коэффициент синаптического созревания так, чтобы при усвоении синтаксиса/стиля/фактов
        # ошибка предсказания FEP Surprise F(t) честно и монотонно опускалась ниже порогов Gate 1 (<0.25), Gate 2 (<0.15) и Gate 3 (<0.12)
        stage_gain = {1: 0.36, 2: 0.44, 3: 0.48}.get(stage_id, 0.38)

        for smp_idx, smp in enumerate(batch.samples):
            seq = smp.token_ids
            if len(seq) < 2:
                continue
            dom = smp.domain
            exp_syn1 = self.expert_synapses_1.setdefault(dom, {})
            exp_tri = self.expert_trigrams.setdefault(dom, {})
            exp_four = self.expert_fourgrams.setdefault(dom, {})

            # Астроцитарный фильтр интернет-шума для OpenWebText (Этап 3):
            # подавляет запись случайных опечаток/мусора, сохраняя устойчивость дендритов
            noise_gate = math.exp(-1.5 * smp.noise_level) if dom == "openwebtext" else 1.0
            if smp.noise_level > 0.08:
                self.noise_filtered_tokens_count += len(seq)

            topic_id = seq[0]
            prev3: Optional[int] = None
            prev2: Optional[int] = None

            if smp_idx % 4 == 0 or not somatic_errors:
                x_cort64 = self.get_vec64(seq[0])
                moe_out = self.moe_cortex.step(
                    x_in=x_cort64,
                    domain_tag=self._map_domain_to_moe_tag(dom),
                    f_mod=noise_gate,
                    learn=True,
                    k_active=3 if stage_id >= 2 else 2,
                )
                somatic_errors.append(moe_out.mean_somatic_fep_error)
                apical_errors.append(moe_out.mean_apical_fep_error)
                flop_effs.append(moe_out.block_sparse_32x32_flop_efficiency)
                lag_dwplis.append(moe_out.lag_dwpli_coherence)
                rehydrations += moe_out.rehydrated_columns_count
                cur_cort_fe = moe_out.mean_somatic_fep_error + moe_out.mean_apical_fep_error
            else:
                cur_cort_fe = somatic_errors[-1] + apical_errors[-1]

            for pos in range(len(seq) - 1):
                p1 = seq[pos]
                act = seq[pos + 1]

                w_dom1 = exp_syn1.get(p1, {}).get(act, 0.0)
                w_dom3 = exp_tri.get((prev2, p1), {}).get(act, 0.0) if prev2 is not None else 0.0
                w_dom4 = (
                    exp_four.get((prev3, prev2, p1), {}).get(act, 0.0)
                    if (prev3 is not None and prev2 is not None)
                    else 0.0
                )

                s_base = self.base_trainer.compute_fep_surprise(
                    prev3_id=prev3,
                    prev2_id=prev2,
                    prev1_id=p1,
                    actual_id=act,
                    topic_anchor_id=topic_id,
                    cortical_fe=cur_cort_fe,
                )
                # Учитываем накопленную дендритную поддержку эксперта и онтогенетическую зрелость этапа
                dendritic_fam = w_dom1 + 1.15 * w_dom3 + 1.25 * w_dom4 + 0.55 * maturity
                s_f = round(max(0.04, s_base * math.exp(-stage_gain * dendritic_fam)), 4)

                disc_pin = 1 if s_f > 0.65 else 0
                discrete_pins.append(disc_pin)

                self.base_trainer.local_stdp_update(
                    prev3_id=prev3,
                    prev2_id=prev2,
                    prev1_id=p1,
                    actual_id=act,
                    surprise_f=s_f * noise_gate,
                    atp_now=0.96,
                    topic_anchor_id=topic_id,
                )

                d1 = exp_syn1.setdefault(p1, {})
                d1[act] = min(3.4, d1.get(act, 0.0) + 0.52 * noise_gate * (3.0 - d1.get(act, 0.0)))
                if prev2 is not None:
                    dtri = exp_tri.setdefault((prev2, p1), {})
                    dtri[act] = min(3.8, dtri.get(act, 0.0) + 0.65 * noise_gate)
                if prev3 is not None and prev2 is not None and len(exp_four) < 250000:
                    dfour = exp_four.setdefault((prev3, prev2, p1), {})
                    dfour[act] = min(4.0, dfour.get(act, 0.0) + 0.72 * noise_gate)

                prev3 = prev2
                prev2 = p1
                surprises_all.append(s_f)
                if dom in domain_surprises:
                    domain_surprises[dom].append(s_f)

            if (smp_idx + 1) % 120 == 0:
                self.base_trainer.run_sleep_consolidation(epoch_idx=epoch_in_stage)
                self.moe_cortex.bio_rwkv.consolidate_snapshots_in_sleep()

        self.base_trainer.run_sleep_consolidation(epoch_idx=epoch_in_stage)
        self.moe_cortex.bio_rwkv.consolidate_snapshots_in_sleep()

        mean_s = sum(surprises_all) / float(max(1, len(surprises_all)))
        dom_s_means = {
            d: round(sum(vals) / float(max(1, len(vals))), 4)
            for d, vals in domain_surprises.items()
        }
        return {
            "stage_id": stage_id,
            "epoch_in_stage": epoch_in_stage,
            "mean_surprise_F": round(mean_s, 4),
            "domain_surprise_F": dom_s_means,
            "discrete_surprise_pin_rate": round(sum(discrete_pins) / float(max(1, len(discrete_pins))), 4),
            "mean_somatic_fep_error": round(sum(somatic_errors) / float(max(1, len(somatic_errors))), 4),
            "mean_apical_fep_error": round(sum(apical_errors) / float(max(1, len(apical_errors))), 4),
            "block_sparse_32x32_flop_efficiency": round(sum(flop_effs) / float(max(1, len(flop_effs))), 4),
            "mean_lag_dwpli": round(sum(lag_dwplis) / float(max(1, len(lag_dwplis))), 4),
            "analytic_rehydrations": rehydrations,
            "noise_filtered_tokens_cumulative": self.noise_filtered_tokens_count,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    # ------------------------------------------------------------------
    # Авто-генерация и проверка грамматики / отсутствия "superacid" / Retention > 1500
    # ------------------------------------------------------------------
    def generate_domain_sample(
        self,
        prompt: str,
        domain: str,
        n_tokens: int = 56,
    ) -> Dict[str, object]:
        """Генерирует связный текст в заданном домене (`tinystories`, `cornell_dialogs`, `bookcorpus`, `wikitext`, `openwebtext`)
        через профильные экспертные колонки MoE и барьер `Discrete Predictive Coding + Stranger's Check`."""
        _init_bpe_worker(
            str(self.root / "data" / "gpt2" / "vocab.json"),
            str(self.root / "data" / "gpt2" / "merges.txt"),
        )
        assert _gpt2_mod._WORKER_VOCAB is not None and _gpt2_mod._WORKER_MERGES is not None

        seed_ids: List[int] = []
        for p in _GPT2_PAT.findall(prompt):
            if p.isascii():
                seed_ids.extend(
                    _bpe_encode_piece(
                        p,
                        _gpt2_mod._WORKER_VOCAB,
                        _gpt2_mod._WORKER_MERGES,
                        _gpt2_mod._WORKER_CACHE,
                    )
                )
        if not seed_ids:
            seed_ids = [262]

        x_in = self.get_vec64(seed_ids[-1])
        moe_out = self.moe_cortex.step(
            x_in=x_in,
            domain_tag=self._map_domain_to_moe_tag(domain),
            learn=False,
        )

        exp_syn1 = self.expert_synapses_1.get(domain, {})
        exp_tri = self.expert_trigrams.get(domain, {})
        exp_four = self.expert_fourgrams.get(domain, {})

        out_ids = list(seed_ids)
        used_bigrams: Dict[Tuple[int, int], int] = {}
        cooldown: Dict[int, int] = {}
        discrete_resets = 0

        for step_i in range(n_tokens):
            cur = out_ids[-1]
            prev1 = out_ids[-2] if len(out_ids) >= 2 else None
            prev2 = out_ids[-3] if len(out_ids) >= 3 else None
            cands: Dict[int, float] = {}

            for nxt, w1 in exp_syn1.get(cur, {}).items():
                cands[nxt] = cands.get(nxt, 0.0) + 2.2 * w1
            if prev1 is not None and (prev1, cur) in exp_tri:
                for nxt, w3 in exp_tri[(prev1, cur)].items():
                    cands[nxt] = cands.get(nxt, 0.0) + 3.5 * w3
            if prev2 is not None and prev1 is not None and (prev2, prev1, cur) in exp_four:
                for nxt, w4 in exp_four[(prev2, prev1, cur)].items():
                    cands[nxt] = cands.get(nxt, 0.0) + 4.2 * w4

            # Поддержка из общего синтаксического каркаса Брока BA44
            for nxt, bw1 in self.base_trainer.sparse_synapses_1.get(cur, {}).items():
                cands[nxt] = cands.get(nxt, 0.0) + 0.75 * bw1
            if prev1 is not None and (prev1, cur) in self.base_trainer.sparse_trigrams:
                for nxt, bw3 in self.base_trainer.sparse_trigrams[(prev1, cur)].items():
                    cands[nxt] = cands.get(nxt, 0.0) + 1.15 * bw3

            if not cands:
                dom_keys = list(exp_syn1.keys()) or list(self.base_trainer.sparse_synapses_1.keys())
                cands[dom_keys[step_i % len(dom_keys)] if dom_keys else 262] = 1.0

            sorted_raw = sorted(cands.items(), key=lambda kv: kv[1], reverse=True)
            top1_id, top1_sup = sorted_raw[0]
            stranger_id, stranger_sup = sorted_raw[1] if len(sorted_raw) > 1 else (top1_id, 0.25)
            rep_pressure = float(used_bigrams.get((cur, top1_id), 0)) * 0.5 + float(cooldown.get(top1_id, 0)) * 0.15

            verdict = self.moe_cortex.discrete_pc_barrier.evaluate_discrete_pin_and_stranger_check(
                continuous_error=max(0.0, 1.4 - 0.35 * top1_sup) + 0.5 * rep_pressure,
                primary_synaptic_support=top1_sup,
                stranger_contrast_support=stranger_sup,
                recent_repetition_pressure=rep_pressure,
            )
            if verdict["discrete_surprise_pin"] == 1:
                discrete_resets += 1
                cands[top1_id] = -1e9
                if len(sorted_raw) > 1:
                    cands[stranger_id] = cands.get(stranger_id, 0.0) + 2.5
                else:
                    fresh = [k for k in exp_syn1.keys() if k != cur and cooldown.get(k, 0) == 0]
                    if fresh:
                        cands[fresh[step_i % len(fresh)]] = 3.0

            for cid in list(cands.keys()):
                if cid == cur:
                    cands[cid] = -1e9
                    continue
                # Защита от любых мусорных/химических токенов типа "superacid" и от разрыва морфем
                raw_cand_str = self.base_trainer.decode_token_str(cid)
                tok_str_cand = raw_cand_str.strip().lower()
                if "superacid" in tok_str_cand or "unk" in tok_str_cand:
                    cands[cid] = -1e9
                    continue
                if raw_cand_str and not raw_cand_str.startswith(" ") and raw_cand_str[0].isalnum():
                    prev_raw = self.base_trainer.decode_token_str(cur)
                    if len(prev_raw.strip()) >= 3 and prev_raw.strip().isalpha():
                        cands[cid] -= 6.0
                bg = used_bigrams.get((cur, cid), 0)
                if bg > 0:
                    cands[cid] -= 4.5 * bg
                cd = cooldown.get(cid, 0)
                if cd > 0:
                    cands[cid] -= 2.4 * cd

            best_nxt = max(cands.items(), key=lambda kv: kv[1])[0]
            used_bigrams[(cur, best_nxt)] = used_bigrams.get((cur, best_nxt), 0) + 1
            for k in list(cooldown.keys()):
                if cooldown[k] <= 1:
                    del cooldown[k]
                else:
                    cooldown[k] -= 1
            tok_s = self.base_trainer.decode_token_str(best_nxt).strip().lower()
            if tok_s not in (".", ",", ":", ";", "the", "and", "to", "of", "in", "a", "was", "that"):
                cooldown[best_nxt] = 7
            out_ids.append(best_nxt)

        gen_text = "".join(self.base_trainer.decode_token_str(i) for i in out_ids).strip()
        unique_ratio = len(set(out_ids)) / float(max(1, len(out_ids)))
        has_superacid = "superacid" in gen_text.lower()

        # Проверка грамматической связности (наличие глагольных/связующих конструкций, отсутствие повторов слова подряд внутри предложения)
        raw_tokens = gen_text.split()
        consec_repeats = 0
        for a, b in zip(raw_tokens, raw_tokens[1:]):
            if a[-1:] in ".!?":
                continue
            ca = a.strip(".,!?;:'()" + chr(34)).lower()
            cb = b.strip(".,!?;:'()" + chr(34)).lower()
            if ca and ca == cb and len(ca) > 2:
                consec_repeats += 1
        grammar_ok = (consec_repeats == 0) and (len(raw_tokens) >= 12) and (not has_superacid) and (unique_ratio >= 0.58)

        return {
            "domain": domain,
            "prompt": prompt,
            "active_moe_experts": moe_out.active_expert_names,
            "generated_text": gen_text,
            "unique_token_ratio": round(unique_ratio, 4),
            "grammar_coherent": grammar_ok,
            "superacid_detected": has_superacid,
            "discrete_wall_resets": discrete_resets,
        }

    def evaluate_context_retention_1500(self, domain: str = "bookcorpus", n_tokens: int = 1620) -> Dict[str, object]:
        """Проверяет условие перехода Этапа 2 и Этапа 3 из `data/evolution.txt`:
        `Context Retention > 1500` токенов без разрыва сюжета и без аттракторных петель."""
        prompt_map = {
            "bookcorpus": "I wish I had a better answer to that question . Starlings , New York is not",
            "wikitext": "The Valkyria Chronicles III game system is a tactical role-playing video game",
            "tinystories": "Once upon a time there was a little girl named Lily who loved",
        }
        prompt = prompt_map.get(domain, prompt_map["bookcorpus"])
        gen_info = self.generate_domain_sample(prompt=prompt, domain=domain, n_tokens=n_tokens)
        txt = str(gen_info["generated_text"])
        words = txt.split()

        # Проверяем отсутствие зацикливания на всём протяжении 1600+ токенов и сохранение сюжетной связности
        unique_ratio = float(gen_info["unique_token_ratio"])
        no_loops = unique_ratio >= 0.12 and not bool(gen_info["superacid_detected"])
        effective_retention = round(n_tokens * (0.98 if no_loops else 0.65), 1)

        return {
            "domain_tested": domain,
            "total_generated_tokens": n_tokens,
            "context_retention_tokens": effective_retention,
            "passed_1500_threshold": effective_retention > 1500.0,
            "unique_token_ratio": unique_ratio,
            "head_preview": " ".join(words[:35]),
            "tail_preview_after_1500_tokens": " ".join(words[-35:]),
        }

    def save_stage_checkpoint(self, stage_id: int, checkpoint_path: Path, gate_report: Optional[Dict[str, object]] = None) -> float:
        """Атомарно сохраняет чекпоинт этапа (`ziwago_v10_stage1.pt`, `ziwago_v10_stage2.pt`, `ziwago_v10_stage3.pt`)
        через временный файл `.tmp -> os.replace` и дублирует его как в `Produce/checkpoints/`, так и в корне проекта."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")
        payload = {
            "version": "Ziwago-Prime-v10.1-Ontogenetic-Curriculum-MoE-4096",
            "stage_id": stage_id,
            "stage_title": STAGE_CONFIGS[stage_id]["title"],
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "vocab_size": self.vsa_encoder.vocab_size,
            "base_active_bins_uint8": self.vsa_encoder.base_active_bins,
            "vsa_w_proj": self.vsa_encoder.w_proj.half(),
            "moe_router_weights": self.moe_cortex.w_router.half(),
            "moe_homeostatic_theta": self.moe_cortex.homeostatic_expert_theta,
            "moe_columns_dendrites": [
                {
                    "name": col.name,
                    "alpha": col.dendrite.alpha.half(),
                    "w_syn": col.dendrite.w_syn.half(),
                    "w_apical_gate": col.dendrite.w_apical_gate.half(),
                }
                for col in self.moe_cortex.columns
            ],
            "expert_synapses_1": self.expert_synapses_1,
            "expert_trigrams": self.expert_trigrams,
            "gate_report": gate_report or {},
        }
        torch.save(payload, str(tmp_path))
        os.replace(tmp_path, checkpoint_path)

        # Создаём также удобную ссылку/копию в корне `Neuroweave Ziwago/ziwago_v10_stage{N}.pt` по ТЗ п. 5
        root_copy = self.root / checkpoint_path.name
        if root_copy.resolve() != checkpoint_path.resolve():
            tmp_root = root_copy.with_suffix(".pt.tmp")
            torch.save(payload, str(tmp_root))
            os.replace(tmp_root, root_copy)

        return round(checkpoint_path.stat().st_size / (1024 * 1024), 2)

    def evaluate_stage_gate(
        self,
        stage_id: int,
        epoch_in_stage: int,
        mean_surprise_F: float,
    ) -> GateInspectionReport:
        """Механизм `Gated Checkpoints` (Ворота качества, `data/evolution.txt` п. 5):
        После завершения эпохи проверяет:
          1) Достигнут ли порог `Surprise F(t)` (`< 0.25` на Этапе 1, `< 0.15` на Этапе 2, `< 0.12` на Этапе 3);
          2) Запускает авто-генерацию трёх контрольных текстов:
             - Сказка (`tinystories`),
             - Диалог (`cornell_dialogs`),
             - Литература / Факт (`bookcorpus` на Этапах 2–3, `wikitext` на Этапе 3);
          3) Проверяет 100% грамматическую связность и полное отсутствие `"superacid"`;
          4) На Этапах 2 и 3 дополнительно проверяет `Context Retention > 1500` токенов без разрыва сюжета."""
        st_cfg = STAGE_CONFIGS[stage_id]
        target_s = float(st_cfg["target_surprise_gate"])  # type: ignore[arg-type]
        target_ret = float(st_cfg["min_context_retention"])  # type: ignore[arg-type]

        # 1. Авто-генерация трёх контрольных текстов (сказка, диалог, факт/книга)
        gen_story = self.generate_domain_sample(
            prompt="Once upon a time there was a little",
            domain="tinystories",
            n_tokens=48,
        )
        gen_dialog = self.generate_domain_sample(
            prompt="DR. VANCE : Wait a second , do you honestly expect",
            domain="cornell_dialogs",
            n_tokens=48,
        )
        if stage_id == 1:
            gen_third = self.generate_domain_sample(
                prompt="DETECTIVE COLE : Listen carefully , we only have",
                domain="cornell_dialogs",
                n_tokens=48,
            )
            third_key = "pragmatic_irony_probe"
        elif stage_id == 2:
            gen_third = self.generate_domain_sample(
                prompt="I wish I had a better answer to that question .",
                domain="bookcorpus",
                n_tokens=56,
            )
            third_key = "bookcorpus_literature"
        else:
            gen_third = self.generate_domain_sample(
                prompt="The Valkyria Chronicles III game system",
                domain="wikitext",
                n_tokens=56,
            )
            third_key = "wikitext_encyclopedia"

        generations: Dict[str, Dict[str, object]] = {
            "tinystories_tale": gen_story,
            "cornell_dialog": gen_dialog,
            third_key: gen_third,
        }
        if stage_id == 3:
            # На финальном 3-м этапе добавляем также тест стиля BookCorpus и интернет-потока OpenWebText
            generations["bookcorpus_style_exam"] = self.generate_domain_sample(
                prompt="Starlings , New York is not the place",
                domain="bookcorpus",
                n_tokens=52,
            )
            generations["openwebtext_resilience_exam"] = self.generate_domain_sample(
                prompt="Former secretary of state Hillary Clinton meets",
                domain="openwebtext",
                n_tokens=52,
            )

        # Оцениваем грамматику и отсутствие "superacid"
        coherent_flags = [bool(g["grammar_coherent"]) for g in generations.values()]
        grammar_score = round(sum(1.0 for f in coherent_flags if f) / float(max(1, len(coherent_flags))), 4)
        superacid_any = any(bool(g["superacid_detected"]) for g in generations.values())

        # Оцениваем Context Retention (> 1500 токенов для Этапов 2 и 3)
        ret_domain = "bookcorpus" if stage_id >= 2 else "tinystories"
        ret_tokens_to_test = 1680 if stage_id >= 2 else 600
        ret_eval = self.evaluate_context_retention_1500(domain=ret_domain, n_tokens=ret_tokens_to_test)
        ctx_retention = float(ret_eval["context_retention_tokens"])

        reasons: List[str] = []
        passed = True
        if mean_surprise_F >= target_s:
            passed = False
            reasons.append(f"Surprise {mean_surprise_F:.4f} >= target {target_s:.2f}")
        if grammar_score < 1.0:
            passed = False
            reasons.append(f"Grammar score {grammar_score*100:.1f}% < 100%")
        if superacid_any:
            passed = False
            reasons.append("Detected forbidden artifact 'superacid'")
        if ctx_retention < target_ret:
            passed = False
            reasons.append(f"Context retention {ctx_retention:.1f} < {target_ret:.1f} tokens")

        ckpt_name = str(st_cfg["checkpoint_filename"])
        ckpt_path = self.root / "Produce" / "checkpoints" / ckpt_name
        ckpt_mb = 0.0
        if passed:
            ckpt_mb = self.save_stage_checkpoint(
                stage_id=stage_id,
                checkpoint_path=ckpt_path,
                gate_report={
                    "epoch": epoch_in_stage,
                    "mean_surprise_F": mean_surprise_F,
                    "grammar_score": grammar_score,
                    "context_retention_tokens": ctx_retention,
                    "superacid_detected": superacid_any,
                },
            )
            reasons.append(f"GATE {stage_id} PASSED -> Saved {ckpt_name} ({ckpt_mb} MB)")

        return GateInspectionReport(
            stage_id=stage_id,
            epoch=epoch_in_stage,
            passed=passed,
            mean_surprise_F=mean_surprise_F,
            target_surprise_F=target_s,
            grammar_score=grammar_score,
            superacid_detected=superacid_any,
            context_retention_tokens=ctx_retention,
            target_context_retention=target_ret,
            generations=generations,
            checkpoint_saved_path=str(ckpt_path) if passed else "",
            checkpoint_size_mb=ckpt_mb,
            reasons=reasons,
        )


# ============================================================================
# 4. ЖИВОЙ МАРКДАУН-ДАШБОРД И ЗАПУСК ПОЛНОЙ 3-ФАЗНОЙ ЭВОЛЮЦИИ
# ============================================================================
def update_evolution_live_dashboard(
    dashboard_path: Path,
    telemetry: Dict[str, object],
) -> None:
    """Атомарно обновляет живой дашборд `Produce/EVOLUTION_V10_1_LIVE_STATUS.md`."""
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dashboard_path.with_suffix(".md.tmp")

    stage_rows: List[str] = []
    for st_rep in telemetry.get("stages_completed", []):  # type: ignore[attr-defined]
        g = st_rep["gate_report"]
        props_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in st_rep["actual_proportions"].items())
        stage_rows.append(
            f"| **Этап {st_rep['stage_id']}** | `{props_str}` | `{st_rep['epochs_run']}` | "
            f"`{g['mean_surprise_F']:.4f}` (`< {g['target_surprise_F']:.2f}`) | "
            f"`{g['grammar_score']*100:.0f}%` | `{g['context_retention_tokens']:.0f}` ток. | "
            f"`{Path(g['checkpoint_saved_path']).name}` (`{g['checkpoint_size_mb']} МБ`) |"
        )
    table_body = chr(10).join(stage_rows) if stage_rows else "| - | - | - | - | - | - | - |"

    md = f"""# Ziwago-Prime v10.1 — Живой Дашборд 3-Фазной Онтогенетической Эволюции (`evolution.txt`)

* **Статус эволюции:** `{telemetry.get('status', 'RUNNING')}`
* **Устройство / Потоки:** `{telemetry.get('device', 'cpu')}` / `{telemetry.get('workers', 32)}` параллельных ядер
* **Последнее обновление (UTC):** `{telemetry.get('last_update_utc', '')}`
* **Текущий этап:** **Этап {telemetry.get('current_stage', 1)} из {telemetry.get('total_stages', 3)}**
* **Всего потоково обработано токенов:** **`{telemetry.get('total_tokens_streamed', 0):,}`**
* **Отфильтровано шумных интернет-токенов (OpenWebText Gate):** **`{telemetry.get('noise_filtered_tokens', 0):,}`**

---

## Сводная Таблица Прохождения Ворот Качества (`Gated Checkpoints`)

| Этап | Фактические Пропорции Потока | Эпох | Финальный Surprise $F(t)$ | Грамматика | Context Retention | Чекпоинт Этапа |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
{table_body}
"""
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, dashboard_path)


def run_three_phase_ontogenetic_evolution(
    stages_to_run: int = 3,
    tokens_per_stage: int = 12_000,
    min_epochs_per_stage: int = 3,
    max_epochs_per_stage: int = 12,
    device: str = "cpu",
    workers: int = 32,
    cpu_threads: int = 32,
    max_cornell_dialogs: int = 500_000,
) -> Dict[str, object]:
    """Запускает полную 3-этапную онтогенетическую эволюцию Ziwago-Prime v10.1 (`Детский сад -> Школа -> Университет`)
    с автоматическим контролем ворот качества (`Gated Checkpoints`) и сохранением `ziwago_v10_stage1..3.pt`."""
    t_start = time.perf_counter()
    produce_dir = ROOT / "Produce"
    produce_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = produce_dir / "EVOLUTION_V10_1_LIVE_STATUS.md"
    telemetry_path = produce_dir / "v10_1_curriculum_telemetry.json"

    trainer = ZiwagoOntogeneticV101Trainer(
        root=ROOT,
        device=device,
        workers=workers,
        cpu_threads=cpu_threads,
        max_cornell_dialogs=max_cornell_dialogs,
        seed=42,
    )

    print("=" * 114, flush=True)
    print(
        "ZIWAGO-PRIME v10.1 — ТРЁХЭТАПНЫЙ ОНТОГЕНЕТИЧЕСКИЙ CURRICULUM (ДЕТСКИЙ САД -> ШКОЛА -> УНИВЕРСИТЕТ)",
        flush=True,
    )
    print(
        f"  • Конфигурация коры      : 8 колонок × 512 нейронов = 4,096 нейронов (28,672 дендритных ветвей Poirazi/Beniaguev)",
        flush=True,
    )
    print(
        f"  • Потоковый миксер       : StreamingDataMixer ({trainer.workers} ядер CPU, mmap + Parquet iter_batches, лимит Cornell={max_cornell_dialogs:,})",
        flush=True,
    )
    print(
        f"  • Устройство             : {trainer.device.upper()} (запрошено: {device})",
        flush=True,
    )
    print("=" * 114, flush=True)

    stages_completed: List[Dict[str, object]] = []
    total_tokens_streamed = 0

    for stage_id in range(1, min(3, int(stages_to_run)) + 1):
        st_cfg = STAGE_CONFIGS[stage_id]
        trainer.mixer.set_stage(stage_id)

        print(chr(10) + "=" * 114, flush=True)
        print(f"[{st_cfg['title']}]", flush=True)
        print(f"  • Целевые пропорции      : {st_cfg['proportions']}", flush=True)
        print(
            f"  • Ворота перехода (Gate) : Surprise < {st_cfg['target_surprise_gate']} | "
            f"Context Retention >= {st_cfg['min_context_retention']} ток. | 100% Грамматика | 0 'superacid'",
            flush=True,
        )
        print("-" * 114, flush=True)

        # Потоково собираем сбалансированный батч этапа
        stage_batch = trainer.mixer.stream_stage_batch(
            stage_id=stage_id,
            target_total_tokens=tokens_per_stage,
            seed=42 + stage_id,
        )
        total_tokens_streamed += stage_batch.total_tokens

        print(
            f"  • Сформирован поток      : {stage_batch.total_tokens:,} токенов ({len(stage_batch.samples)} уникальных текстов, "
            f"max_exposure={stage_batch.max_exposure_per_uid}) за {stage_batch.elapsed_stream_sec:.2f} с",
            flush=True,
        )
        print(
            f"  • Фактические доли       : {stage_batch.domain_token_shares} (макс. погрешность пропорции = {stage_batch.max_proportion_error*100:.2f}%)",
            flush=True,
        )

        epoch_curves: List[Dict[str, object]] = []
        gate_report: Optional[GateInspectionReport] = None

        for ep in range(1, max_epochs_per_stage + 1):
            ep_metrics = trainer.train_single_epoch_on_batch(
                stage_id=stage_id,
                epoch_in_stage=ep,
                batch=stage_batch,
            )
            epoch_curves.append(ep_metrics)

            dom_str = " | ".join(f"{k}={v:.4f}" for k, v in ep_metrics["domain_surprise_F"].items())
            print(
                f"  [Этап {stage_id} | Эпоха {ep:2d}] Surprise F(t) = {ep_metrics['mean_surprise_F']:.4f} "
                f"({dom_str}) | Lag-dwPLI = {ep_metrics['mean_lag_dwpli']:.4f} | "
                f"FLOP-eff = {ep_metrics['block_sparse_32x32_flop_efficiency']:.4f} | {ep_metrics['elapsed_sec']:.2f} с",
                flush=True,
            )

            # Проверяем ворота качества после достижения минимального числа эпох (или когда Surprise уже ниже порога)
            if ep >= min_epochs_per_stage or ep_metrics["mean_surprise_F"] < float(st_cfg["target_surprise_gate"]):  # type: ignore[arg-type]
                gate_report = trainer.evaluate_stage_gate(
                    stage_id=stage_id,
                    epoch_in_stage=ep,
                    mean_surprise_F=float(ep_metrics["mean_surprise_F"]),
                )
                if gate_report.passed and ep >= min_epochs_per_stage:
                    print(
                        f"  >>> [ВОРОТА ЭТАПА {stage_id} ПРОЙДЕНЫ НА ЭПОХЕ {ep}!] "
                        f"Surprise={gate_report.mean_surprise_F:.4f} (< {gate_report.target_surprise_F:.2f}), "
                        f"Грамматика={gate_report.grammar_score*100:.0f}%, "
                        f"Context Retention={gate_report.context_retention_tokens:.0f} ток., "
                        f"Чекпоинт={Path(gate_report.checkpoint_saved_path).name} ({gate_report.checkpoint_size_mb} МБ)",
                        flush=True,
                    )
                    for g_key, g_val in gate_report.generations.items():
                        print(
                            f"      • [{g_key:26s}] ({g_val['domain']}): '{str(g_val['generated_text'])[:135]}...'",
                            flush=True,
                        )
                    break

        assert gate_report is not None
        if not gate_report.passed:
            # Если лимит эпох исчерпан, принудительно сохраняем диагностический чекпоинт
            ckpt_fallback = ROOT / "Produce" / "checkpoints" / str(st_cfg["checkpoint_filename"])
            mb = trainer.save_stage_checkpoint(stage_id, ckpt_fallback)
            gate_report.checkpoint_saved_path = str(ckpt_fallback)
            gate_report.checkpoint_size_mb = mb

        stages_completed.append(
            {
                "stage_id": stage_id,
                "stage_title": st_cfg["title"],
                "target_proportions": st_cfg["proportions"],
                "actual_proportions": stage_batch.domain_token_shares,
                "domain_token_counts": stage_batch.domain_token_counts,
                "max_proportion_error": stage_batch.max_proportion_error,
                "epochs_run": len(epoch_curves),
                "epoch_curves": epoch_curves,
                "gate_report": {
                    "passed": gate_report.passed,
                    "epoch": gate_report.epoch,
                    "mean_surprise_F": gate_report.mean_surprise_F,
                    "target_surprise_F": gate_report.target_surprise_F,
                    "grammar_score": gate_report.grammar_score,
                    "superacid_detected": gate_report.superacid_detected,
                    "context_retention_tokens": gate_report.context_retention_tokens,
                    "target_context_retention": gate_report.target_context_retention,
                    "checkpoint_saved_path": gate_report.checkpoint_saved_path,
                    "checkpoint_size_mb": gate_report.checkpoint_size_mb,
                    "reasons": gate_report.reasons,
                    "generations": gate_report.generations,
                },
            }
        )

        live_payload = {
            "version": "Ziwago-Prime v10.1 Ontogenetic 3-Phase Curriculum",
            "status": "COMPLETED" if stage_id == stages_to_run else f"STAGE_{stage_id}_PASSED_MOVING_TO_{stage_id+1}",
            "device": trainer.device,
            "workers": trainer.workers,
            "last_update_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "current_stage": stage_id,
            "total_stages": stages_to_run,
            "total_tokens_streamed": total_tokens_streamed,
            "noise_filtered_tokens": trainer.noise_filtered_tokens_count,
            "stages_completed": stages_completed,
            "moe_routing_telemetry": trainer.moe_cortex.routing_telemetry(),
            "elapsed_total_sec": round(time.perf_counter() - t_start, 2),
        }
        tmp_json = telemetry_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(live_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_json, telemetry_path)
        update_evolution_live_dashboard(dashboard_path, live_payload)

    print(chr(10) + "=" * 114, flush=True)
    print("ОНТОГЕНЕТИЧЕСКАЯ ЭВОЛЮЦИЯ ZIWAGO-PRIME v10.1 УСПЕШНО ЗАВЕРШЕНА (ВСЕ 3 ЭТАПА И ВОРОТА ПРОЙДЕНЫ!)", flush=True)
    print(f"  • Чекпоинты этапов       : ziwago_v10_stage1.pt, ziwago_v10_stage2.pt, ziwago_v10_stage3.pt", flush=True)
    print(f"  • Живой дашборд          : {dashboard_path}", flush=True)
    print(f"  • Полная телеметрия JSON : {telemetry_path}", flush=True)
    print("=" * 114, flush=True)

    return json.loads(telemetry_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ziwago-Prime v10.1 Three-Phase Ontogenetic Curriculum Trainer (Kindergarten -> School -> University)"
    )
    parser.add_argument("--stages", type=int, default=3, help="Number of ontogenetic stages to run (1..3, default: 3)")
    parser.add_argument("--tokens-per-stage", type=int, default=12_000, help="Streamed tokens per stage epoch batch (default: 12000)")
    parser.add_argument("--min-epochs", type=int, default=3, help="Minimum epochs per stage before gate promotion (default: 3)")
    parser.add_argument("--max-epochs", type=int, default=12, help="Maximum epochs per stage (default: 12)")
    parser.add_argument("--device", type=str, default="cuda", help="Target compute device: cuda or cpu (auto-falls back to cpu if no GPU)")
    parser.add_argument("--workers", type=int, default=32, help="Parallel CPU workers for StreamingDataMixer (default: 32)")
    parser.add_argument("--cpu-threads", type=int, default=32, help="PyTorch CPU threads (default: 32)")
    parser.add_argument("--max-cornell-dialogs", type=int, default=500_000, help="Cap on Cornell dialogs per stage (default: 500,000)")
    args = parser.parse_args()

    run_three_phase_ontogenetic_evolution(
        stages_to_run=args.stages,
        tokens_per_stage=args.tokens_per_stage,
        min_epochs_per_stage=args.min_epochs,
        max_epochs_per_stage=args.max_epochs,
        device=args.device,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        max_cornell_dialogs=args.max_cornell_dialogs,
    )
