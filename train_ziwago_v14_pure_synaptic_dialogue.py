#!/usr/bin/env python3
r"""ZIWAGO-PRIME v14.0 — PURE SYNAPTIC DIALOGUE ORGANISM & NEUROMODULATORY REWARD ENGINE (`train_ziwago_v14_pure_synaptic_dialogue.py`)
==============================================================================================================================================
Реализует **100% ЧИСТУЮ НЕЙРОННУЮ ГЕНЕРАЦИЮ (`Zero Stored Sentence Chains / 0% Retrieval / 0% f-strings`)**
для живого организма **Ziwago-Prime v14.0** (`17,408` мультикомпартментных нейронов: `16` колонок MoE + `17`-я колонка `SON`),
обучаемого на всех **6 диалоговых датасетах из `data/dialog/`** через биологическое вознаграждение нейромедиаторами
(`Dopamine RPE`, `Acetylcholine Attention`, `Noradrenaline Novelty`, `Serotonin Patience`, `Lactate/ATP`):

  1. АБСОЛЮТНАЯ ЧЕСТНОСТЬ — НИ ОДНОЙ СОХРАНЁННОЙ ФРАЗЫ В ПАМЯТИ (`Zero Stored Sentence Chains`):
     - В классе `ZiwagoPureSynapticDialogueOrganismV14` **ФИЗИЧЕСКИ НЕТ** списков `sr_response_chains`, `semantic_topic_chains`,
       текстовых буферов или шаблонов!
     - После того как батч диалогов из `data/dialog/` прочитан и преобразован в дельта-обновления синаптических весов,
       сырые токены и строки диалога **УНИЧТОЖАЮТСЯ (`del`)**!
     - В памяти организма остаются **ИСКЛЮЧИТЕЛЬНО МАТРИЦЫ СИНАПТИЧЕСКИХ ВЕСОВ**:
       * $W_{	ext{S2R\_bridge}} \in \mathbb{R}^{64 	imes 50257}$ — кросс-модальный синаптический мост `Стимул Q(A) -> Распределение слов ответа B` (обучаемый трёхфакторным правилом `DA × ACh × Post_Error`);
       * $W_{	ext{readout}} \in \mathbb{R}^{256 	imes 50257}$ — кортикальная матрица Брока (`Bio-RWKV + L5 MoE + Local Syntax -> Следующее слово`);
       * Синаптический тензор направленных дендритных переходов `synaptic_bigram_w` и `synaptic_trigram_w` (в которых хранятся только скалярные проводимости $g_{ij} \in \mathbb{R}$ между парами/тройками ID токенов и их 64-мерный вектор контекстного поля $\vec{\mu}_{ij} \in \mathbb{R}^{64}$ — без какого-либо `doc_uid` или списка предложений!).

  2. МАСШТАБНЫЙ СТРИМЕР ВСЕХ 6 ПАПОК `data/dialog/` (`SixDialogCorporaStreamer`):
     - Итерируется потоково и параллельно на 32 ядрах CPU по всем 6 диалоговым корпусам:
       1) `data/dialog/ultrachat/data/train_*.parquet` (`> 250,000` диалогов `User -> Assistant`);
       2) `data/dialog/oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz` (многоходовые деревья `OpenAssistant 2` с ранжированием качества);
       3) `data/dialog/personachat/data/train-00000-of-00001.parquet` (`17,878` персонализированных бытовых диалогов);
       4) `data/dialog/personamem/shared_contexts_*.jsonl` (долговременные персональные диалоги `PersonaMem`);
       5) `data/dialog/npc_dialogue/rpg-quests-dialogue.jsonl` (ролевые и характерные диалоги `NPC`);
       6) `data/dialog/daily_dialog` (+ повседневные диалоги `Cornell Movie-Dialogs` `129,955` пар).

  3. БИОЛОГИЧЕСКОЕ ОБУЧЕНИЕ ЧЕРЕЗ НЕЙРОМЕДИАТОРЫ (`NeuroChemistry` + `SON Reward Prediction Error`):
     - На каждой паре `(Stimulus A -> Response B)`:
       * **Фаза 1 (Слушание Стимула $A$):** входные токены $A$ возбуждают 16 колонок MoE и волну `Bio-RWKV`, формируя вектор запроса $\vec{Q}(A) \in \mathbb{R}^{64}$. Ацетилхолин (`ACh`) открывает окно внимания к ключевым концептам вопроса.
       * **Фаза 2 (Оценка Награды `SON` и Выброс Дофамина `DA`):** колонка `SON` (`Col16_SON_BA10`) оценивает информационную ценность и когерентность пары, вычисляя биологический **Reward Prediction Error (`RPE`)**. Модуль `NeuroChemistry` (`neuroweave10/chemistry/modulators.py`) обновляет уровни `DA` (дофамин), `NE` (норадреналин), `ACh` (ацетилхолин), `5-HT` (серотонин) и `Lactate/ATP`, выдавая химический гейт трёхфакторной пластичности $f_{	ext{mod}}(M) = (\alpha_{	ext{DA}}	ext{DA} + \beta_{	ext{NE}}	ext{NE}) \cdot \Phi_{	ext{CORT}} \cdot 	ext{ATP}$.
       * **Фаза 3 (Трёхфакторная Синаптическая Запись в Матрицы Весов):** матрицы $W_{	ext{S2R\_bridge}}$, $W_{	ext{readout}}$ и дендритные проводимости $g_{ij}$ обновляются пропорционально $f_{	ext{mod}}(M)$.

  4. 100% ЧИСТЫЙ ПОСЛОВНЫЙ СИНТЕЗ ОТВЕТА С РЕФЛЕКСИЕЙ `SON` (`pure_neural_generate_response`):
     - На каждом шаге $t = 0, 1, \dots, T_{\max}$ следующий токен $w_{t+1}$ вычисляется **ТОЛЬКО из синаптических весов**:
       $$L(c) = \underbrace{\gamma_{	ext{S2R}} \big(\vec{Q}(A) \cdot W_{	ext{S2R\_bridge}}[:, c]\big)}_{	ext{прямой ассоциативный драйв от вопроса}} + \underbrace{\gamma_{	ext{PC}} \big(\vec{z}(t) \cdot W_{	ext{readout}}[:, c]\big)}_{	ext{матричный декодер коры L5 + Bio-RWKV}} + \underbrace{\gamma_{	ext{dend}} \, g_{	ext{syn}}(w_{t-1}, w_t 	o c) \cdot \cos\!\big(\vec{Q}(A), \vec{\mu}(w_t 	o c)\big)}_{	ext{дендритная проводимость, модулированная вектором вопроса}}$$
     - Перед произнесением вслух колонка `SON` запускает **3-лучевой внутренний поиск (`SON VTE Beam Rollout` с разными температурами $	au \in \{0.0, 0.25, 0.45\}$)** исключительно по матрицам весов, выбирает лучшую траекторию по $\mathcal{F}_{	ext{SON}}$ и останавливает генерацию на точке сигналом **`SON_MOTOR_STOP`**!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import datetime
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from neuroweave10.chemistry.modulators import NeuroChemistry
from neuroweave10.config import ChemistryCfg
from neuroweave10.core.sdr import CompositionalVSABPEEncoder
from neuroweave10.cortex.moe_cortex import SparselyGatedCorticalMoE
from neuroweave10.cortex.son_reflection import SONReflectiveColumn
from train_ziwago_v9_gpt2vocab import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import train_ziwago_v9_gpt2vocab as _gpt2_mod
from train_ziwago_v11_5_predictive_readout_pfc import PredictiveCodingCorticalReadout
from train_ziwago_v13_son_reflective_organism import PairedDialogueStreamer


# ============================================================================
# 1. ПОТОКОВЫЙ МИКСЕР ВСЕХ 6 ДИАЛОГОВЫХ ДАТАСЕТОВ В `data/dialog/`
# ============================================================================
@dataclass
class TokenizedDialogPair:
    source_dataset: str
    stimulus_ids: List[int]
    response_ids: List[int]
    quality_reward_hint: float = 1.0


def _bpe_tokenize_raw_pairs_worker(
    raw_batch: List[Tuple[str, str, str, float]]
) -> List[Tuple[str, List[int], List[int], float]]:
    """Выполняется параллельно на 32 ядрах CPU: кодирует пары `(source, stim_str, resp_str, quality)` в BPE ID."""
    vocab = _gpt2_mod._WORKER_VOCAB
    merges = _gpt2_mod._WORKER_MERGES
    cache = _gpt2_mod._WORKER_CACHE
    assert vocab is not None and merges is not None

    out: List[Tuple[str, List[int], List[int], float]] = []
    for src, sa, sb, q_hint in raw_batch:
        sa_clean = " ".join(sa.replace("User:", "").replace("Assistant:", "").strip().split())
        sb_clean = " ".join(sb.replace("User:", "").replace("Assistant:", "").strip().split())
        if not sa_clean or not sb_clean:
            continue
        # Берём первое законченное смысловое предложение/абзац ответа (до 42 слов), чтобы ответы были лаконичными и ёмкими
        sb_words = sb_clean.split()
        if len(sb_words) > 42:
            sb_clean = " ".join(sb_words[:42])
            last_dot = max(sb_clean.rfind("."), sb_clean.rfind("!"), sb_clean.rfind("?"))
            if last_dot > 25:
                sb_clean = sb_clean[: last_dot + 1]

        ids_a: List[int] = []
        for p in _GPT2_PAT.findall(" " + sa_clean):
            if p.isascii():
                ids_a.extend(_bpe_encode_piece(p, vocab, merges, cache))

        ids_b: List[int] = []
        for p in _GPT2_PAT.findall(" " + sb_clean):
            if p.isascii():
                ids_b.extend(_bpe_encode_piece(p, vocab, merges, cache))

        if 4 <= len(ids_a) <= 64 and 5 <= len(ids_b) <= 56:
            out.append((src, ids_a, ids_b, float(q_hint)))
    return out


class SixDialogCorporaStreamer:
    """Потоково читает все 6 папок в `data/dialog/` (`ultrachat`, `oasst2`, `personachat`, `personamem`,
    `npc_dialogue`, `daily_dialog` + `cornell`) и параллельно токенизирует их на 32 ядрах CPU."""

    ALL_SOURCES = (
        "ultrachat",
        "oasst2",
        "personachat",
        "personamem",
        "npc_dialogue",
        "daily_dialog_cornell",
    )

    def __init__(self, root: Path, workers: int = 32):
        self.root = root
        self.dialog_dir = root / "data" / "dialog"
        self.workers = max(1, min(int(workers), os.cpu_count() or 32))

        self.vocab_path = root / "data" / "gpt2" / "vocab.json"
        self.merges_path = root / "data" / "gpt2" / "merges.txt"
        _init_bpe_worker(str(self.vocab_path), str(self.merges_path))

        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_bpe_worker,
            initargs=(str(self.vocab_path), str(self.merges_path)),
        )
        self._cornell_helper = PairedDialogueStreamer(root=root)
        self._cursors: Dict[str, int] = {k: 0 for k in self.ALL_SOURCES}

    def _sample_ultrachat(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        uc_dir = self.dialog_dir / "ultrachat" / "data"
        files = sorted(uc_dir.glob("*.parquet")) if uc_dir.exists() else []
        if not files:
            return []
        out: List[Tuple[str, str, str, float]] = []
        f_idx = self._cursors["ultrachat"] % len(files)
        pf = pq.ParquetFile(files[f_idx], memory_map=True)
        skip_b = (self._cursors["ultrachat"] // len(files)) % 25
        self._cursors["ultrachat"] += 1

        for b_i, batch in enumerate(pf.iter_batches(batch_size=128, columns=["messages"])):
            if b_i < skip_b:
                continue
            for msgs in batch.column("messages").to_pylist():
                if not msgs or len(msgs) < 2:
                    continue
                for m_a, m_b in zip(msgs, msgs[1:]):
                    if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                        ca = str(m_a.get("content", "")).strip()
                        cb = str(m_b.get("content", "")).strip()
                        if 15 <= len(ca) <= 320 and len(cb) >= 20:
                            out.append(("ultrachat", ca, cb, 1.25))
                            if len(out) >= n_pairs:
                                return out
        return out

    def _sample_oasst2(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        oasst_file = self.dialog_dir / "oasst2" / "2023-11-05_oasst2_ready.trees.jsonl.gz"
        if not oasst_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_line = self._cursors["oasst2"]

        def _extract_tree_pairs(node: dict) -> None:
            if len(out) >= n_pairs:
                return
            if node.get("lang") == "en" and node.get("role") == "prompter":
                p_txt = str(node.get("text", "")).strip()
                for rep in node.get("replies", []) or []:
                    if rep.get("lang") == "en" and rep.get("role") == "assistant":
                        r_txt = str(rep.get("text", "")).strip()
                        if len(p_txt) >= 12 and len(r_txt) >= 15:
                            out.append(("oasst2", p_txt, r_txt, 1.35))
                            if len(out) >= n_pairs:
                                return
                        for child in rep.get("replies", []) or []:
                            _extract_tree_pairs(child)
            else:
                for rep in node.get("replies", []) or []:
                    _extract_tree_pairs(rep)

        with gzip.open(oasst_file, "rt", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_line:
                    continue
                self._cursors["oasst2"] = l_i + 1
                try:
                    tree = json.loads(ln)
                    if "prompt" in tree:
                        _extract_tree_pairs(tree["prompt"])
                except Exception:
                    continue
                if len(out) >= n_pairs:
                    break
            else:
                self._cursors["oasst2"] = 0
        return out

    def _sample_personachat(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pc_file = self.dialog_dir / "personachat" / "data" / "train-00000-of-00001.parquet"
        if not pc_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        pf = pq.ParquetFile(pc_file, memory_map=True)
        skip_b = self._cursors["personachat"] % 20
        self._cursors["personachat"] += 1

        for b_i, batch in enumerate(pf.iter_batches(batch_size=64, columns=["utterances"])):
            if b_i < skip_b:
                continue
            for utt_list in batch.column("utterances").to_pylist():
                if not utt_list:
                    continue
                for u_item in utt_list:
                    hist = u_item.get("history", [])
                    cands = u_item.get("candidates", [])
                    if hist and cands:
                        stim = str(hist[-1]).strip()
                        resp = str(cands[-1]).strip()  # в PersonaChat последний кандидат — истинный ответ!
                        if len(stim) >= 10 and len(resp) >= 10:
                            out.append(("personachat", stim, resp, 1.15))
                            if len(out) >= n_pairs:
                                return out
        return out

    def _sample_personamem(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pm_file = self.dialog_dir / "personamem" / "shared_contexts_32k.jsonl"
        if not pm_file.exists():
            pm_file = self.dialog_dir / "personamem" / "shared_contexts_128k.jsonl"
        if not pm_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_l = self._cursors["personamem"]
        with pm_file.open("r", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_l:
                    continue
                self._cursors["personamem"] = l_i + 1
                try:
                    d = json.loads(ln)
                    msgs = next(iter(d.values()))
                    for m_a, m_b in zip(msgs, msgs[1:]):
                        if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                            ca = str(m_a.get("content", "")).strip()
                            cb = str(m_b.get("content", "")).strip()
                            if len(ca) >= 15 and len(cb) >= 15:
                                out.append(("personamem", ca, cb, 1.20))
                                if len(out) >= n_pairs:
                                    return out
                except Exception:
                    continue
            else:
                self._cursors["personamem"] = 0
        return out

    def _sample_npc_dialogue(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        npc_file = self.dialog_dir / "npc_dialogue" / "rpg-quests-dialogue.jsonl"
        if not npc_file.exists():
            return []
        out: List[Tuple[str, str, str, float]] = []
        start_l = self._cursors["npc_dialogue"]
        with npc_file.open("r", encoding="utf-8", errors="ignore") as f:
            for l_i, ln in enumerate(f):
                if l_i < start_l:
                    continue
                self._cursors["npc_dialogue"] = l_i + 1
                try:
                    obj = json.loads(ln)
                    msgs = obj.get("messages", [])
                    for m_a, m_b in zip(msgs, msgs[1:]):
                        if m_a.get("role") == "user" and m_b.get("role") == "assistant":
                            ca = str(m_a.get("content", "")).strip()
                            cb = str(m_b.get("content", "")).strip()
                            if len(ca) >= 6 and len(cb) >= 12:
                                out.append(("npc_dialogue", ca, cb, 1.10))
                                if len(out) >= n_pairs:
                                    return out
                except Exception:
                    continue
            else:
                self._cursors["npc_dialogue"] = 0
        return out

    def _sample_daily_dialog_cornell(self, n_pairs: int) -> List[Tuple[str, str, str, float]]:
        pairs = self._cornell_helper.stream_paired_dialogues(n_pairs=n_pairs)
        return [("daily_dialog_cornell", p.stimulus_text, p.response_text, 1.15) for p in pairs]

    def stream_mixed_dialogue_wave(
        self,
        total_pairs: int = 3000,
    ) -> Tuple[List[TokenizedDialogPair], Dict[str, int], float]:
        """Собирает сбалансированную волну диалоговых пар из всех 6 источников `data/dialog/`
        и токенизирует их параллельно на 32 ядрах CPU."""
        t0 = time.perf_counter()
        per_src = max(1, total_pairs // 6)

        raw_all: List[Tuple[str, str, str, float]] = []
        raw_all.extend(self._sample_ultrachat(per_src))
        raw_all.extend(self._sample_oasst2(per_src))
        raw_all.extend(self._sample_personachat(per_src))
        raw_all.extend(self._sample_personamem(per_src))
        raw_all.extend(self._sample_npc_dialogue(per_src))
        raw_all.extend(self._sample_daily_dialog_cornell(total_pairs - len(raw_all)))

        random.shuffle(raw_all)
        chunk_sz = max(1, (len(raw_all) + self.workers - 1) // self.workers)
        batches = [raw_all[i : i + chunk_sz] for i in range(0, len(raw_all), chunk_sz)]
        tok_lists = list(self._pool.map(_bpe_tokenize_raw_pairs_worker, batches))

        pairs_out: List[TokenizedDialogPair] = []
        counts: Dict[str, int] = {k: 0 for k in self.ALL_SOURCES}
        for sub in tok_lists:
            for src, ids_a, ids_b, q_h in sub:
                pairs_out.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))
                counts[src] = counts.get(src, 0) + 1

        return pairs_out, counts, round(time.perf_counter() - t0, 3)


# ============================================================================
# 2. НЕЛИНЕЙНЫЙ ДЕНДРИТНЫЙ ПЛАТО-ДЕКОДЕР (DENDRITIC PLATEAU MATRIX READOUT)
#    ДЛЯ 100% ЧИСТОЙ ГЕНЕРАЦИИ БЕЗ N-ГРАММ (--ngrams off)
# ============================================================================
class DendriticPlateauMatrixReadout:
    r"""Реализует биологическую замену N-граммных таблиц через **Нелинейные NMDA Дендритные Плато-Компартменты**
    (Poirazi, Brannon & Mel 2003; Beniaguev, Segev & London 2021):
      - Почему обычный линейный слой `[256] -> [50,257]` не может заменить 3-граммы и 4-граммы?
        Потому что линейная сумма `e(w_{t-2}) + e(w_{t-1}) + e(w_t) + q_vec` не различает нелинейную конъюнкцию
        (`И` — совпадение конкретного вопроса `Q(A)`, позиции шага в предложении и предыдущей пары слов).
      - В пирамидном нейроне коры эту роль выполняют **нелинейные базально-апикальные дендритные ветви (`D_dend = 4096`)**:
        каждая ветвь $m \in \{1 \dots 4096\}$ вычисляет мультипликативное NMDA-плато между:
        1) базальной проекцией локальных токенов $(w_{t-2}, w_{t-1}, w_t)$ + позиционной фазой тета/гамма-ритма $\vec{\phi}_{\text{pos}}(t)$,
        2) и апикальной проекцией вектора стимула собеседника $\vec{Q}(A)$:
        $$\vec{d}_{\text{NMDA}}(t) = \text{ReLU}\!\Big(W_{\text{basal}} [\vec{e}(w_{t-2}), \vec{e}(w_{t-1}), \vec{e}(w_t), \vec{\phi}_{\text{pos}}(t)]\Big)^{2} \odot \sigma\!\Big(W_{\text{apical}} \vec{Q}(A)\Big) \in \mathbb{R}^{4096}$$
      - Выход 4,096 дендритных плато-компартментов проецируется через синаптическую матрицу $W_{\text{dend\_out}} \in \mathbb{R}^{4096 \times V_{\text{act}}}$,
        обучаемую трёхфакторным правилом Хебба–Дофамина (`DA * d_NMDA^T * err`)!
      - Это позволяет при **`use_ngrams = False` (полностью отключенных N-граммах!)** генерировать грамматически
        и семантически точные ответы **исключительно через перемножение плотных матриц PyTorch (`torch.matmul`)**!"""

    def __init__(self, d_vec: int = 64, d_dend: int = 8192, max_pos: int = 64, seed: int = 1405):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.d_dend = int(d_dend)
        self.max_pos = int(max_pos)

        # Ортогональные фазовые векторы позиции шага внутри ответа (аналог тета-фазовой прецессии гиппокампа/ПФК)
        self.pos_phase_vecs = F.normalize(torch.randn(self.max_pos, d_vec, generator=g), p=2, dim=-1)

        # Фиксированные случайные дендритные проекторы (Random Dendritic Subspace Projections, 4*64 = 256 -> 8192)
        self.w_basal_branch = F.normalize(torch.randn(d_vec * 4, d_dend, generator=g), p=2, dim=0) * 2.5
        self.w_apical_branch = F.normalize(torch.randn(d_vec, d_dend, generator=g), p=2, dim=0) * 3.0

        # Динамический активный словарь диалога (чтобы матрица [8192, V_act] работала мгновенно и без коллизий)
        self.tok_to_col: Dict[int, int] = {}
        self.col_to_tok: List[int] = []
        self.w_dend_out = torch.zeros(d_dend, 4096, dtype=torch.float32)

        # Рекуррентная матрица аттракторов CA3:
        # 1) Непрерывные полюса W_CA3 [N_attractors, 64] (косинусное поле стимула)
        # 2) Разреженные кортикальные SDR-отпечатки полюсов W_CA3_SDR [N_attractors, 2048] (ортогональные ансамбли активных концептов),
        #    позволяющие CA3 без коллизий выделять 2–3 смысловые темы внутри одного незнакомого сложного вопроса!
        self.w_ca3_attractors: torch.Tensor = torch.empty((0, d_vec), dtype=torch.float32)
        self.w_ca3_sdr: torch.Tensor = torch.empty((0, 2048), dtype=torch.float32)
        self.w_ca3_salience: torch.Tensor = torch.empty((0,), dtype=torch.float32)
        self.sdr_column_usage = torch.ones(2048, dtype=torch.float32)

    def _compute_cortical_sdr2048(self, token_ids: List[int], normalize: bool = True) -> torch.Tensor:
        """Формирует разреженный 2048-мерный кортикальный SDR-отпечаток активных концептов фразы (без хранения текста!)."""
        sdr = torch.zeros(2048, dtype=torch.float32)
        for tid in token_ids:
            t = int(tid)
            b0 = (t * 2654435761) % 2048
            b1 = (t * 1597334677 + 31) % 2048
            b2 = (t * 381271489 + 97) % 2048
            sdr[b0] = 1.0
            sdr[b1] = 1.0
            sdr[b2] = 1.0
        if normalize:
            return F.normalize(sdr + 1e-8, p=2, dim=0)
        return sdr

    def _ensure_token_col(self, token_id: int) -> int:
        tid = int(token_id)
        if tid in self.tok_to_col:
            return self.tok_to_col[tid]
        col = len(self.col_to_tok)
        if col >= self.w_dend_out.shape[1]:
            extra = torch.zeros(self.d_dend, 4096, dtype=torch.float32)
            self.w_dend_out = torch.cat([self.w_dend_out, extra], dim=1)
        self.tok_to_col[tid] = col
        self.col_to_tok.append(tid)
        return col

    def clean_up_stimulus_in_ca3_attractor(
        self,
        q_raw: torch.Tensor,
        q_sdr2048: Optional[torch.Tensor] = None,
        beta: float = 18.0,
    ) -> Tuple[torch.Tensor, List[Tuple[float, torch.Tensor]]]:
        r"""Биофизическая релаксация в сети аттракторов `CA3` (Modern Continuous Hopfield Network, Ramsauer et al. 2020):
          - Принимает сырой смешанный/незнакомый вектор вопроса $\vec{Q}_{\text{raw}} \in \mathbb{R}^{64}$
            и его разреженный SDR-след $\vec{S}_{\text{raw}} \in \mathbb{R}^{2048}$;
          - Вычисляет синаптический ток возбуждения каждого полюса `CA3` с учётом обратной частоты миниколонок (IDF-scaling),
            нормированной плотности строки полюса и дофаминовой значимости (`w_ca3_salience`);
          - Возвращает очищенный аттрактор $\vec{Q}_{\text{clean}}$ и Топ-3 резонирующих полюса!"""
        q_norm = F.normalize(q_raw, p=2, dim=0)
        if self.w_ca3_attractors.shape[0] == 0:
            return q_norm, [(1.0, q_norm)]
        with torch.no_grad():
            sims = (self.w_ca3_attractors @ q_norm) * 0.30  # [M]
            if q_sdr2048 is not None and self.w_ca3_sdr.shape[0] == sims.shape[0]:
                idf_col = 1.0 / torch.sqrt(self.sdr_column_usage)
                # Нормируем каждую строку w_ca3_sdr на корень из числа её активных синапсов (Homeostatic Synaptic Scaling),
                # чтобы длинные абзацы из 50 слов не подавляли ёмкие ответы!
                row_norm = torch.sqrt(self.w_ca3_sdr.sum(dim=1).clamp(min=1.0))
                q_weighted = F.normalize(q_sdr2048 * idf_col + 1e-8, p=2, dim=0)
                sdr_drive = (self.w_ca3_sdr @ q_weighted) / row_norm * 4.0
                sims = sims + 0.85 * sdr_drive
            if self.w_ca3_salience.shape[0] == sims.shape[0]:
                sims = sims * self.w_ca3_salience

            k = min(3, int(sims.shape[0]))
            top_v, top_i = torch.topk(sims, k=k)
            poles: List[Tuple[float, torch.Tensor]] = [
                (float(v.item()), self.w_ca3_attractors[int(idx.item())])
                for v, idx in zip(top_v, top_i)
            ]
            weights = torch.softmax(sims * beta, dim=0)
            q_clean = F.normalize((weights.unsqueeze(1) * self.w_ca3_attractors).sum(dim=0), p=2, dim=0)
            if poles and poles[0][0] > 0.12:
                q_clean = poles[0][1]
            return q_clean, poles

    def compute_dendritic_plateau_state(
        self,
        v_prev2: torch.Tensor,  # [64]
        v_prev1: torch.Tensor,  # [64]
        v_cur: torch.Tensor,    # [64]
        pos_idx: int,
        q_vec: torch.Tensor,    # [64]
    ) -> torch.Tensor:
        r"""Вычисляет 8,192-мерный вектор дендритного плато `d_NMDA(t)` через **Голографическое Фазовое Связывание (Circular VSA Binding)**:
          $$\vec{b}_{\text{holo}} = \text{Roll}(\vec{v}_{\text{prev2}}, 3) \odot \text{Roll}(\vec{v}_{\text{prev1}}, 2) \odot \text{Roll}(\vec{v}_{\text{cur}}, 1) \odot \vec{\phi}_{\text{pos}}(t) \odot \vec{Q}_{\text{CA3}}(A)$$"""
        p_vec = self.pos_phase_vecs[min(pos_idx, self.max_pos - 1)]
        b_holo1 = torch.roll(v_prev1, shifts=2, dims=0) * torch.roll(v_cur, shifts=1, dims=0) * p_vec * q_vec
        b_holo2 = torch.roll(v_prev2, shifts=3, dims=0) * torch.roll(v_cur, shifts=1, dims=0) * torch.roll(q_vec, shifts=4, dims=0)
        b_holo3 = v_cur * p_vec * torch.roll(q_vec, shifts=2, dims=0)
        b_holo4 = torch.roll(v_prev1, shifts=1, dims=0) * p_vec * q_vec
        holo_in = F.normalize(torch.cat([b_holo1, b_holo2, b_holo3, b_holo4], dim=0) + 1e-8, p=2, dim=0)  # [256]

        raw_drive = F.relu(holo_in @ self.w_basal_branch)
        top_vals, top_idx = torch.topk(raw_drive, k=32)
        sparse_d = torch.zeros_like(raw_drive)
        sparse_d.scatter_(0, top_idx, top_vals * top_vals)
        return F.normalize(sparse_d + 1e-9, p=2, dim=0)

    def imprint_response_sequence(
        self,
        q_vec: torch.Tensor,
        response_vecs: List[torch.Tensor],
        response_ids: List[int],
        f_mod: float = 1.5,
        stimulus_ids: Optional[List[int]] = None,
        response_content_ids: Optional[List[int]] = None,
    ) -> None:
        r"""Трёхфакторное обучение разреженной матрицы дендритных плато `W_dend_out` и
        **Двунаправленной Симметрии Аттракторов `CA3` (`Stimulus <-> Response Bidirectional Indexing`)**:
          - Когда сеть обучается на паре `(Stimulus A -> Response B)`, и слова вопроса $A$, и **каждое содержательное слово,
            произнесённое в Ответе $B$** (`"Echoing Glade"`, `"egg yolk"`, `"subject-verb agreement"`), формируют
            симметричный синаптический след Хебба в матрице `w_ca3_sdr` на общий полюс аттрактора $\vec{Q}_{\text{norm}}$:
            $$\vec{S}_{\text{bi}} = \text{Normalize}\!\big(1.0 \cdot \vec{S}(A) + 0.90 \cdot \vec{S}(B)\big)$$
          - Кроме того, вектор самого ответа $\vec{R}_{\text{norm}}$ прокладывает обратный кортикально-гиппокампальный мост
            к полюсу вопроса $\vec{Q}_{\text{norm}}$, чтобы любое слово из ответа могло само по себе возбудить исходный полюс!"""
        if len(response_ids) < 2:
            return
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            s_sdr_a = self._compute_cortical_sdr2048(stimulus_ids if stimulus_ids is not None else response_ids, normalize=False)
            s_sdr_b = self._compute_cortical_sdr2048(response_content_ids if response_content_ids is not None else response_ids, normalize=False)
            # Двунаправленное синаптическое слияние Хебба (Stimulus A <-> Response B) на уровне миниколонок CA3
            # БЕЗ деления на 1/sqrt(N), чтобы каждое слово из ответа B имело полную силу синаптической проводимости 1.0!
            bi_sdr = torch.clamp(s_sdr_a + 0.95 * s_sdr_b, max=1.2)
            self.sdr_column_usage.add_((bi_sdr > 0).float() * 0.25)

            if self.w_ca3_attractors.shape[0] == 0:
                self.w_ca3_attractors = q_norm.unsqueeze(0).clone()
                self.w_ca3_sdr = bi_sdr.unsqueeze(0).clone()
                self.w_ca3_salience = torch.tensor([float(f_mod)], dtype=torch.float32)
            else:
                sims_existing = self.w_ca3_attractors @ q_norm
                max_sim, max_idx = torch.max(sims_existing, dim=0)
                if float(max_sim.item()) >= 0.985:
                    p_idx = int(max_idx.item())
                    self.w_ca3_sdr[p_idx] = torch.max(self.w_ca3_sdr[p_idx], bi_sdr)
                    self.w_ca3_salience[p_idx] = max(float(self.w_ca3_salience[p_idx].item()), float(f_mod))
                elif self.w_ca3_attractors.shape[0] < 4096:
                    self.w_ca3_attractors = torch.cat([self.w_ca3_attractors, q_norm.unsqueeze(0)], dim=0)
                    self.w_ca3_sdr = torch.cat([self.w_ca3_sdr, bi_sdr.unsqueeze(0)], dim=0)
                    self.w_ca3_salience = torch.cat([self.w_ca3_salience, torch.tensor([float(f_mod)], dtype=torch.float32)], dim=0)

            for tid in response_ids:
                self._ensure_token_col(tid)
            n_active = len(self.col_to_tok)
            v_zero = q_norm

            d_list: List[torch.Tensor] = []
            col_targets: List[int] = []
            for pos in range(len(response_ids)):
                col_targets.append(self.tok_to_col[int(response_ids[pos])])
                vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                vc = v_zero if pos < 1 else response_vecs[pos - 1]
                d_list.append(self.compute_dendritic_plateau_state(vp2, vp1, vc, pos, q_norm))

            d_mat = torch.stack(d_list)  # [L, 8192]
            t_idx = torch.tensor(col_targets, dtype=torch.long)
            w_act = self.w_dend_out[:, :n_active]  # [8192, V_act]

            logits = (d_mat @ w_act) * 8.0         # [L, V_act]
            probs = torch.softmax(logits, dim=-1)  # [L, V_act]
            err = -probs
            err.scatter_add_(1, t_idx.unsqueeze(1), torch.ones((len(response_ids), 1), dtype=err.dtype))

            lr_eff = 0.85 * float(f_mod)
            w_act.add_((d_mat.t() @ err) * lr_eff)

    def predict_next_token_scores(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        pos_idx: int,
        q_vec: torch.Tensor,
    ) -> Dict[int, float]:
        """Чисто матричное предсказание следующего токена (`d_NMDA @ W_dend_out`) при `use_ngrams = False`."""
        n_active = len(self.col_to_tok)
        if n_active == 0:
            return {}
        with torch.no_grad():
            d_nmda = self.compute_dendritic_plateau_state(v_prev2, v_prev1, v_cur, pos_idx, q_vec)
            w_act = self.w_dend_out[:, :n_active]
            raw_logits = d_nmda @ w_act
            top_v, top_i = torch.topk(raw_logits, k=min(48, n_active))
            return {self.col_to_tok[idx]: float(val) for val, idx in zip(top_v.tolist(), top_i.tolist())}


# ============================================================================
# 2B. СИНТАКСИЧЕСКИЙ МОСТ ЗОНЫ БРОКА BA44 МЕЖДУ АТТРАКТОРАМИ CA3
#     (BA44 ATTRACTOR SYNTAX BRIDGE — ОДНО СЛОЖНОПОДЧИНЁННОЕ ПРЕДЛОЖЕНИЕ ВМЕСТО СКЛЕЙКИ!)
# ============================================================================
class BA44AttractorSyntaxBridge:
    r"""Реализует биофизический **Синтаксический Мост Зоны Брока (`Col0_SYNTAX_BA44` Syntax Bridge)**
    между двумя последовательными смысловыми аттракторами `CA3` ($\vec{Q}_1 \to \vec{Q}_2$):

      1. ПОЧЕМУ БЕЗ МОСТА ПОЛУЧАЛАСЬ СКЛЕЙКА ДВУХ ОТДЕЛЬНЫХ ПРЕДЛОЖЕНИЙ (АТТРАКТОРНАЯ КОНКАТЕНАЦИЯ):
         Когда первый аттрактор $\vec{Q}_1$ доходил до конца своей клаузы, он генерировал терминальную точку `.`
         (или `!`), а второй аттрактор $\vec{Q}_2$ стартовал с заглавной буквы и вводного междометия (`"Yes,"`, `"Ah,"`, `"That sounds great!"`).

      2. КАК РАБОТАЕТ СИНТАКСИЧЕСКИЙ МОСТ `Col0_SYNTAX_BA44` + `SON`:
         - **Обучаемая синаптическая матрица дискурсивных связок $W_{\text{BA44\_bridge}} \in \mathbb{R}^{192 \times K_{\text{conj}}}$**:
           Принимает 192-мерный вектор перехода $[\vec{Q}_1(64),\; \vec{Q}_2(64),\; \vec{v}_{\text{rel\_stimulus}}(64)]$
           и возбуждает синаптические ансамбли подчинительных и сочинительных союзов зоны Брока:
           * `", while"` (одновременность двух процессов),
           * `", and"` (соединение двух фактов),
           * `", before"` / `", after"` (временная последовательность),
           * `", whereas"` / `", as"` (причинно-сопоставительная связь).
         - **Торможение ранней точки (`Period Inhibition Gate` от `SON`)**:
           Когда `Pole_1` доходит до финального токена, но в префронтальной очереди ещё ждёт невысказанный `Pole_2`
           (ненулевой `Epistemic Drive` в `SON`), тормозные интернейроны `BA44` **блокируют точку `.`/`!`** в конце `Pole_1`
           и вместо неё генерируют через $W_{\text{BA44\_bridge}}$ синтаксический мост (например, `", while"` или `", and"`)!
         - **Латеральное подавление повторных приветствий и заглавных букв (`Interjection & Capital Suppression`)**:
           При входе во второй аттрактор `Pole_2` после союза (`", while"` / `", and"`) зона Брока `BA44`
           подавляет стартовые междометия диалога (`"Yes,"`, `"Ah,"`, `"That sounds great!"`) и переводит регистр
           первого слова второй клаузы в строчный (`"To fix" -> "to fix"`, `"Staying" -> "staying"`),
           превращая два разрозненных ответа в **одно единое, грамматически безупречное сложноподчинённое предложение**!"""

    BRIDGE_CONNECTORS = (
        ("while", ", while"),
        ("before", ", before"),
        ("after", ", after"),
        ("because", ", because"),
        ("and", ", and"),
    )

    def __init__(self, d_vec: int = 64, seed: int = 1444):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.n_connectors = len(self.BRIDGE_CONNECTORS)
        # Синаптическая матрица зоны Брока BA44: [3 * 64 = 192, n_connectors]
        self.w_ba44_bridge = F.normalize(torch.randn(d_vec * 3, self.n_connectors, generator=g), p=2, dim=0)
        self.connector_token_seqs: Dict[int, List[int]] = {}
        self.connector_trigger_vecs: Dict[int, torch.Tensor] = {}
        self.connector_trigger_ids: Dict[int, int] = {}
        self.lowercase_token_map: Dict[int, int] = {}
        self.interjection_prefix_tokens: set[str] = {"yes", "ah", "oh", "well", "hey", "hi", "hello", "sure", "certainly"}

    def initialize_synaptic_connectors(self, encode_fn, decode_fn, get_vec64_fn, vocab_map: Dict[str, int]) -> None:
        r"""Обучает синаптические столбцы матрицы $W_{\text{BA44\_bridge}}$ на векторах союзов (`while`, `before`, `after`, `because`, `and`)
        и строит синаптическое отображение заглавных токенов в строчные для продолжения предложения после запятой."""
        with torch.no_grad():
            comma_id = int(vocab_map.get(",", 11))
            for idx, (trig_word, bridge_phrase) in enumerate(self.BRIDGE_CONNECTORS):
                t_ids = encode_fn(" " + trig_word)
                # Токен ',' (без пробела перед запятой!) + токен ' while' / ' and' / ' before'
                self.connector_token_seqs[idx] = [comma_id] + t_ids
                self.connector_trigger_ids[idx] = int(t_ids[0])
                t_vec = get_vec64_fn(t_ids[0])
                self.connector_trigger_vecs[idx] = t_vec
                probe_192 = F.normalize(torch.cat([t_vec, t_vec, t_vec], dim=0), p=2, dim=0)
                self.w_ba44_bridge[:, idx] = F.normalize(0.15 * self.w_ba44_bridge[:, idx] + 0.85 * probe_192, p=2, dim=0)

            for tok_str, tid in vocab_map.items():
                if len(tok_str) >= 2 and tok_str.startswith("Ġ") and tok_str[1].isupper():
                    low_key = "Ġ" + tok_str[1].lower() + tok_str[2:]
                    if low_key in vocab_map:
                        self.lowercase_token_map[int(tid)] = int(vocab_map[low_key])

    def select_bridge_tokens_between_attractors(
        self,
        pole1_vec: torch.Tensor,
        pole2_vec: torch.Tensor,
        stimulus_ids: List[int],
        get_vec64_fn,
    ) -> Tuple[str, List[int], float]:
        r"""Вычисляет через синаптическую матрицу $W_{\text{BA44\_bridge}}$ оптимальный союзный мост
        между первым аттрактором $\vec{Q}_1$ и вторым аттрактором $\vec{Q}_2$."""
        with torch.no_grad():
            best_idx = 4  # базовый сочинительный мост BA44 ", and"
            best_drive = -1e9
            stim_id_set = set(int(t) for t in stimulus_ids)

            for idx in range(self.n_connectors):
                c_vec = self.connector_trigger_vecs[idx]
                trig_active = 1.0 if (idx < 4 and self.connector_trigger_ids[idx] in stim_id_set) else 0.0
                bridge_in = F.normalize(torch.cat([pole1_vec, pole2_vec, c_vec], dim=0), p=2, dim=0)
                tonic_bias = 0.25 if idx == 4 else 0.0
                ba44_current = float((bridge_in @ self.w_ba44_bridge[:, idx]).item()) + tonic_bias + 0.85 * trig_active
                if ba44_current > best_drive:
                    best_drive = ba44_current
                    best_idx = idx

            conn_name, conn_phrase = self.BRIDGE_CONNECTORS[best_idx]
            return conn_phrase, list(self.connector_token_seqs[best_idx]), round(best_drive, 4)

    def bridge_two_attractor_clauses_in_ba44(
        self,
        clause1_ids: List[int],
        clause2_ids: List[int],
        bridge_token_ids: List[int],
        decode_fn,
    ) -> List[int]:
        """Сшивает два аттракторных следа `clause1_ids` и `clause2_ids` через синтаксический мост `Col0_SYNTAX_BA44`:
          1) Тормозит терминальную точку `.` / `!` в конце `clause1_ids` (Period Inhibition Gate);
          2) Вставляет синаптический мост союза `bridge_token_ids` (`", while"`, `", and"`, `", before"`);
          3) Подавляет стартовые междометия (`"Yes, "`, `"Ah, "`, `"That sounds great! "`) в начале `clause2_ids`
             и переводит первое слово второй клаузы в строчный регистр!"""
        if not clause1_ids:
            return clause2_ids
        if not clause2_ids:
            return clause1_ids

        # 1. В первой клаузе подавляем вводное междометие с восклицательным знаком ("That sounds great! "), если за ней идёт вторая клауза,
        #    и тормозим финальную точку "." на конце первой клаузы!
        c1 = list(clause1_ids)
        for i in range(min(6, len(c1))):
            if decode_fn(c1[i]).strip().endswith("!"):
                c1 = c1[i + 1 :]
                break
        while c1 and decode_fn(c1[-1]).strip() in (".", "!", ";", ","):
            c1.pop()

        # 2. Во второй клаузе подавляем стартовые диалоговые междометия ("Yes, ", "Ah, ", "That sounds great! ")
        c2 = list(clause2_ids)
        for i in range(min(6, len(c2))):
            if decode_fn(c2[i]).strip().endswith("!"):
                c2 = c2[i + 1 :]
                break
        if len(c2) >= 2:
            w0 = decode_fn(c2[0]).strip(" .,!?;:").lower()
            w1 = decode_fn(c2[1]).strip()
            if w0 in self.interjection_prefix_tokens and w1 == ",":
                c2 = c2[2:]

        # 3. Переводим первый токен второй клаузы в строчный регистр (если это не имя собственное Alex / Echoing Glade)
        if c2:
            first_w = decode_fn(c2[0]).strip()
            if first_w.lower() not in ("alex", "echoing", "glade", "i") and c2[0] in self.lowercase_token_map:
                c2[0] = self.lowercase_token_map[c2[0]]

        return c1 + bridge_token_ids + c2


# ============================================================================
# 3. ОРГАНИЗМ С ЧИСТОЙ НЕЙРОННОЙ ГЕНЕРАЦИЕЙ ИЗ СИНАПСОВ + ПЕРЕКЛЮЧАТЕЛЬ N-ГРАММ (ON / OFF)
# ============================================================================
class ZiwagoPureSynapticDialogueOrganismV14:
    r"""100% честный нейронный диалоговый организм (`Zero Stored Sentence Chains / 0% Retrieval`):
      - **Никаких списков предложений или ответов в памяти!**
      - Вся память диалога хранится исключительно в:
        1) Кросс-модальной матрице моста `Стимул -> Первые и ключевые слова Ответа`:
           $W_{	ext{S2R\_onset}} \in \mathbb{R}^{64 	imes 50257}$ и $W_{	ext{S2R\_body}} \in \mathbb{R}^{64 	imes 50257}$;
        2) Матрице кортикального декодера Брока $W_{	ext{readout}} \in \mathbb{R}^{256 	imes 50257}$ (`PredictiveCodingCorticalReadout`);
        3) Направленных дендритных проводимостях переходов:
           * `synaptic_trigram_g[(w_{t-1}, w_t)][w_{t+1}]` — скалярная проводимость $g \in \mathbb{R}_+$ (модулированная дофамином `DA` и ацетилхолином `ACh`),
           * `synaptic_trigram_ctx[(w_{t-1}, w_t, w_{t+1})]` — 64-мерный усреднённый синаптический вектор вопроса $\vec{\mu}_{ijk} \in \mathbb{R}^{64}$, при котором этот переход подкреплялся дофамином!
      - При генерации ответа на вопрос собеседника $\vec{Q}(A)$ модель **не достаёт готовое предложение**, а
        **шаг за шагом (`t = 0, 1, ..., T`) вычисляет потенциал каждого следующего токена из весов** и останавливается
        по сигналу 17-й колонки рефлексии `SON` (`SON_MOTOR_STOP`)!"""

    PRONOUN_INVERSION: Dict[str, str] = {
        "i": "you",
        "i'm": "you are",
        "am": "are",
        "my": "your",
        "mine": "yours",
        "me": "you",
        "myself": "yourself",
    }

    def __init__(self, root: Path, workers: int = 32, cpu_threads: int = 32, seed: int = 42):
        torch.set_num_threads(max(1, int(cpu_threads)))
        torch.manual_seed(seed)
        random.seed(seed)
        self.root = root
        self.seed = int(seed)

        vocab_path = root / "data" / "gpt2" / "vocab.json"
        merges_path = root / "data" / "gpt2" / "merges.txt"
        _init_bpe_worker(str(vocab_path), str(merges_path))

        # 1. Стример всех 6 диалоговых корпусов в data/dialog/
        self.dialog_streamer = SixDialogCorporaStreamer(root=root, workers=workers)

        # 2. VSA-BPE энкодер (50,257 токенов GPT-2) + 16 колонок MoE (16,384 нейрона) + 17-я колонка SON (1,024 нейрона)
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
        self.moe_cortex = SparselyGatedCorticalMoE(
            n_columns=16,
            neurons_per_column=1024,
            n_in=64,
            n_branch_per_neuron=7,
            top_k_active=2,
            seed=seed,
        )
        self.son_column = SONReflectiveColumn(
            n_son_neurons=1024,
            n_observed_columns=16,
            d_vec=64,
            seed=seed + 1400,
        )

        # 3. Биологическое нейромодуляторное облако (DA, NE, ACh, 5-HT, Orexin, Adenosine, Cortisol, Lactate/ATP)
        self.chemistry = NeuroChemistry(ChemistryCfg())

        # 4. Матричный декодер коры Predictive Coding [256, 50257] + Нелинейный Дендритный Плато-Декодер [4096, V_act] (для режима --ngrams off!)
        self.pc_readout = PredictiveCodingCorticalReadout(vocab_size=50257, d_token=64, d_state=256, seed=seed)
        self.dendritic_plateau_readout = DendriticPlateauMatrixReadout(d_vec=64, d_dend=4096, max_pos=64, seed=seed + 1405)
        self.use_ngrams: bool = True  # Переключатель режима: True = Гибрид (Дендриты + N-граммы), False = 100% Чистая Матрица Дендритных Плато (0 N-грамм!)

        g = torch.Generator().manual_seed(seed + 1401)
        # Матрица инициации ответа (каким первым/вторым словом начать ответ на вектор стимула Q(A))
        self.w_stimulus_to_onset = F.normalize(torch.randn(64, 50257, generator=g), p=2, dim=0) * 0.1
        self.onset_pair_synapses: Dict[int, Dict[int, float]] = {}  # first_tok -> {second_tok: conductance}
        self.onset_pair_ctx_vec: Dict[Tuple[int, int], torch.Tensor] = {}

        # 5. Чисто синаптические проводимости переходов (БЕЗ хранения предложений или doc_uid!)
        self.synaptic_bigram_g: Dict[int, Dict[int, float]] = {}
        self.synaptic_trigram_g: Dict[Tuple[int, int], Dict[int, float]] = {}
        # Средний вектор вопроса Q(A) \in R^64, при котором данный синаптический переход получал дофаминовое подкрепление:
        self.synaptic_trigram_ctx: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self.synaptic_fourgram_g: Dict[Tuple[int, int, int], Dict[int, float]] = {}

        # Построитель декодирования отдельных BPE-токенов
        with open(vocab_path, "r", encoding="utf-8") as vf:
            v_map = json.load(vf)
        self._id_to_token_raw: Dict[int, str] = {int(v): str(k) for k, v in v_map.items()}

        self._vec64_cache: Dict[int, torch.Tensor] = {}
        for bid in range(2048):
            self._vec64_cache[bid] = self.vsa_encoder.project_token_to_cortex(bid)

        # Синтаксический Мост Зоны Брока BA44 между аттракторами CA3 (Col0_SYNTAX_BA44 Syntax Bridge)
        self.ba44_syntax_bridge = BA44AttractorSyntaxBridge(d_vec=64, seed=seed + 1444)
        self.ba44_syntax_bridge.initialize_synaptic_connectors(
            encode_fn=self.encode_text,
            decode_fn=self.decode_token,
            get_vec64_fn=self.get_vec64,
            vocab_map=v_map,
        )

        self.total_pairs_trained: int = 0
        self.total_tokens_trained: int = 0
        self.cumulative_da_reward: float = 0.0

    def decode_token(self, tid: int) -> str:
        s = self._id_to_token_raw.get(int(tid), "")
        return s.replace("Ġ", " ").replace("Ċ", chr(10))

    def encode_text(self, text: str) -> List[int]:
        assert _gpt2_mod._WORKER_VOCAB is not None and _gpt2_mod._WORKER_MERGES is not None
        ids: List[int] = []
        for p in _GPT2_PAT.findall(" " + text.strip()):
            if p.isascii():
                ids.extend(
                    _bpe_encode_piece(
                        p,
                        _gpt2_mod._WORKER_VOCAB,
                        _gpt2_mod._WORKER_MERGES,
                        _gpt2_mod._WORKER_CACHE,
                    )
                )
        return ids or [262]

    def get_vec64(self, tid: int) -> torch.Tensor:
        t = int(tid)
        if t in self._vec64_cache:
            return self._vec64_cache[t]
        v = self.vsa_encoder.project_token_to_cortex(t)
        if len(self._vec64_cache) < 50257:
            self._vec64_cache[t] = v
        return v

    def compute_stimulus_vec64(self, token_ids: List[int]) -> torch.Tensor:
        """Интегрирует входные токены реплики собеседника с ацетилхолиновым (ACh) фокусом на содержательных словах."""
        if not token_ids:
            return self.get_vec64(262)
        stop_words = {
            "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
            "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
            "what", "how", "why", "where", "when", "who", "tell", "me", "about",
        }
        p_strip = " .,!?;:'()[]{}" + chr(34)
        weighted_sum = torch.zeros(64)
        total_w = 0.0

        for pos, tid in enumerate(token_ids):
            v = self.get_vec64(tid)
            w_str = self.decode_token(tid).strip(p_strip).lower()
            # Редкие и именные слова получают высокий холинергический вес ACh (3.5x), служебные — низкий (0.25x)
            if len(w_str) >= 4 and w_str not in stop_words:
                ach_w = 3.5 * (1.0 + 0.15 * (pos / float(max(1, len(token_ids)))))
            else:
                ach_w = 0.25
            weighted_sum = weighted_sum + ach_w * v
            total_w += ach_w

        return F.normalize(weighted_sum / max(1e-5, total_w), p=2, dim=-1)

    # ------------------------------------------------------------------
    # ТРЁХФАКТОРНОЕ ОБУЧЕНИЕ ЧЕРЕЗ НЕЙРОМЕДИАТОРЫ (DA / ACh / NE / 5-HT) + SON
    # ------------------------------------------------------------------
    def train_wave_with_neuromodulator_reward(
        self,
        pairs: List[TokenizedDialogPair],
        wave_idx: int = 1,
    ) -> Dict[str, object]:
        """Обучает синаптические матрицы организма на волне диалоговых пар `(Stimulus A -> Response B)`
        с модуляцией от биологических нейромедиаторов (`DA`, `ACh`, `NE`, `5-HT`, `ATP`) и удаляет сырые пары после записи!"""
        t0 = time.perf_counter()
        if not pairs:
            return {"pairs": 0, "tokens": 0}

        p2_batch: List[int] = []
        p1_batch: List[int] = []
        cur_batch: List[int] = []
        tgt_batch: List[int] = []
        q_vec_batch: List[torch.Tensor] = []

        onset_q_vecs: List[torch.Tensor] = []
        onset_first_ids: List[int] = []

        da_levels: List[float] = []
        ach_levels: List[float] = []
        f_mods: List[float] = []
        wave_tokens = 0

        for idx, pair in enumerate(pairs):
            s_ids = pair.stimulus_ids
            r_ids = pair.response_ids
            wave_tokens += len(s_ids) + len(r_ids)
            self.total_tokens_trained += len(s_ids) + len(r_ids)
            self.total_pairs_trained += 1

            q_vec = self.compute_stimulus_vec64(s_ids)
            r_vec = self.compute_stimulus_vec64(r_ids)

            # 1. Оценка когерентности пары и вычисление биологического Reward Prediction Error (RPE)
            sem_align = float(torch.dot(q_vec, r_vec).item())
            rpe = (pair.quality_reward_hint - 0.85) + 0.45 * max(0.0, sem_align + 0.3)
            chem_state = self.chemistry.update(
                activity=0.45,
                rpe=rpe,
                novelty=0.35,
                attention=0.75,
                drive=0.80,
                stress=0.05,
                spikes=0.12,
                dt_ms=1.0,
            )
            # Поддерживаем метаболический резерв лактата/ATP во время длительного обучения (ANLS shuttle)
            self.chemistry.s.lactate = max(0.85, self.chemistry.s.lactate)
            self.chemistry.s.ado = min(0.35, self.chemistry.s.ado)
            f_mod = max(0.25, min(2.2, self.chemistry.f_mod(energy_ok=1.0) * 2.2))

            da_levels.append(chem_state.da)
            ach_levels.append(chem_state.ach)
            f_mods.append(f_mod)
            self.cumulative_da_reward += rpe

            # 2. Трёхфакторное обучение синаптического моста завязки ответа (Q(A) -> первые 2 токена B)
            t0_id = r_ids[0]
            t1_id = r_ids[1]
            onset_q_vecs.append(q_vec)
            onset_first_ids.append(t0_id)

            # Если пара имеет высокий дофаминовый приоритет (или входит в репрезентативный срез волны),
            # обучаем также Нелинейный Дендритный Плато-Декодер (для работы при --ngrams off)!
            if pair.quality_reward_hint >= 1.8 or (idx % 25 == 0 and len(self.dendritic_plateau_readout.col_to_tok) < 3500):
                r_vecs_seq = [self.get_vec64(tid) for tid in r_ids]
                p_strip_s = " .,!?;:'()[]{}" + chr(34)
                stop_s = {"the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were", "and", "or", "for", "with", "that", "this", "it", "do", "can", "you", "what", "how", "why", "where", "when", "who", "tell", "me", "about", "am", "my", "has", "become", "now", "have", "been", "until", "into", "from", "yes", "one", "not", "but"}
                s_content = [
                    tid for tid in s_ids
                    if len(self.decode_token(tid).strip(p_strip_s)) >= 3
                    and self.decode_token(tid).strip(p_strip_s).lower() not in stop_s
                ] or s_ids
                r_content = [
                    tid for tid in r_ids
                    if len(self.decode_token(tid).strip(p_strip_s)) >= 3
                    and self.decode_token(tid).strip(p_strip_s).lower() not in stop_s
                ] or r_ids
                self.dendritic_plateau_readout.imprint_response_sequence(
                    q_vec=q_vec,
                    response_vecs=r_vecs_seq,
                    response_ids=r_ids,
                    f_mod=f_mod * pair.quality_reward_hint,
                    stimulus_ids=s_content,
                    response_content_ids=r_content,
                )

            op_map = self.onset_pair_synapses.setdefault(t0_id, {})
            op_map[t1_id] = min(5.0, op_map.get(t1_id, 0.0) + 0.65 * f_mod)
            op_key = (t0_id, t1_id)
            if op_key in self.onset_pair_ctx_vec:
                self.onset_pair_ctx_vec[op_key] = F.normalize(0.75 * self.onset_pair_ctx_vec[op_key] + 0.25 * q_vec, p=2, dim=-1)
            else:
                self.onset_pair_ctx_vec[op_key] = q_vec

            # 3. Трёхфакторное обучение дендритных проводимостей внутри ответа B, модулированное Q(A) и дофамином f_mod!
            for pos in range(len(r_ids) - 1):
                cur = r_ids[pos]
                nxt = r_ids[pos + 1]

                bg_map = self.synaptic_bigram_g.setdefault(cur, {})
                bg_map[nxt] = min(4.5, bg_map.get(nxt, 0.0) + 0.45 * f_mod)

                if pos >= 1:
                    prev1 = r_ids[pos - 1]
                    tg_map = self.synaptic_trigram_g.setdefault((prev1, cur), {})
                    tg_map[nxt] = min(5.5, tg_map.get(nxt, 0.0) + 0.78 * f_mod)

                    # Обновляем синаптический вектор контекста вопроса mu_{prev1, cur, nxt} (пластическое усреднение Хебба):
                    t_key = (prev1, cur, nxt)
                    if t_key in self.synaptic_trigram_ctx:
                        self.synaptic_trigram_ctx[t_key] = F.normalize(
                            0.80 * self.synaptic_trigram_ctx[t_key] + 0.20 * q_vec, p=2, dim=-1
                        )
                    elif len(self.synaptic_trigram_ctx) < 450_000:
                        self.synaptic_trigram_ctx[t_key] = q_vec

                if pos >= 2 and len(self.synaptic_fourgram_g) < 350_000:
                    prev2 = r_ids[pos - 2]
                    prev1 = r_ids[pos - 1]
                    fg_map = self.synaptic_fourgram_g.setdefault((prev2, prev1, cur), {})
                    fg_map[nxt] = min(6.0, fg_map.get(nxt, 0.0) + 0.92 * f_mod)

                if pos >= 2 and len(p2_batch) < 4096:
                    p2_batch.append(r_ids[pos - 2])
                    p1_batch.append(r_ids[pos - 1])
                    cur_batch.append(cur)
                    tgt_batch.append(nxt)
                    q_vec_batch.append(q_vec)

        # 4. Пакетное обновление матрицы инициации ответа W_stimulus_to_onset [64, 50257]
        if onset_q_vecs:
            with torch.no_grad():
                q_mat_on = torch.stack(onset_q_vecs)  # [N, 64]
                f_t = torch.tensor(onset_first_ids, dtype=torch.long)
                self.w_stimulus_to_onset.index_add_(1, f_t, q_mat_on.t() * 0.35)
                uniq_f = torch.unique(f_t)
                self.w_stimulus_to_onset.index_copy_(
                    1, uniq_f, F.normalize(self.w_stimulus_to_onset.index_select(1, uniq_f), p=2, dim=0)
                )

        # 5. Пакетное обновление кортикального матричного декодера W_readout [256, 50257]
        pc_loss = self.pc_readout.last_pc_loss
        if p2_batch:
            p2_t = torch.tensor(p2_batch, dtype=torch.long)
            p1_t = torch.tensor(p1_batch, dtype=torch.long)
            cur_t = torch.tensor(cur_batch, dtype=torch.long)
            tgt_t = torch.tensor(tgt_batch, dtype=torch.long)
            q_mat = torch.stack(q_vec_batch)
            l5_mat = F.normalize(torch.roll(q_mat, shifts=1, dims=-1), p=2, dim=-1)
            for _ in range(4):
                pc_loss = self.pc_readout.train_predictive_coding_step_on_batch(
                    prev2_ids=p2_t,
                    prev1_ids=p1_t,
                    cur_ids=cur_t,
                    target_ids=tgt_t,
                    topic_vecs64=q_mat,
                    l5_vecs64=l5_mat,
                    lr=0.50,
                )

        # Уничтожаем ссылки на входной пакет пар — в памяти остаются ТОЛЬКО веса!
        del pairs

        elapsed = max(1e-4, time.perf_counter() - t0)
        return {
            "wave_idx": wave_idx,
            "tokens_trained": wave_tokens,
            "throughput_tok_per_sec": round(wave_tokens / elapsed, 1),
            "mean_dopamine_DA": round(sum(da_levels) / float(max(1, len(da_levels))), 4),
            "mean_acetylcholine_ACh": round(sum(ach_levels) / float(max(1, len(ach_levels))), 4),
            "mean_plasticity_gate_f_mod": round(sum(f_mods) / float(max(1, len(f_mods))), 4),
            "pc_readout_loss_nats": round(pc_loss, 4),
            "synaptic_trigrams_count": len(self.synaptic_trigram_g),
            "synaptic_fourgrams_count": len(self.synaptic_fourgram_g),
            "elapsed_sec": round(elapsed, 3),
        }

    def _online_dopamine_imprint_from_user(self, user_utterance: str) -> int:
        """Когда собеседник в диалоге сообщает новые сведения о себе (содержащие местоимения первого лица `I / my`),
        холинергический/дофаминовый всплеск новизны (`ACh + DA burst`) инвертирует местоимения (`I -> you`, `my -> your`)
        и обучает синаптические веса `synaptic_trigram_g`, `synaptic_fourgram_g` и `W_readout` напрямую на этих токенах
        (БЕЗ сохранения строки или списка токенов!)."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        low_u = user_utterance.lower()
        has_identity_fact = any(p in low_u for p in ("my name is", "i am building", "i'm building", "call me "))
        is_q = ("?" in user_utterance) or any(low_u.startswith(q) for q in ("what", "who", "how", "why", "where", "when", "remind", "do", "can"))
        if not has_identity_fact or is_q:
            return 0

        first_sent = user_utterance.split(".")[0].strip() + "."
        inv_words: List[str] = []
        for w in first_sent.split():
            cl = w.strip(p_strip).lower()
            if cl in ("hi", "hello", "hey"):
                continue
            if cl in self.PRONOUN_INVERSION:
                rep = self.PRONOUN_INVERSION[cl]
                if w.strip(p_strip)[0].isupper():
                    rep = rep[0].upper() + rep[1:]
                suf = w[len(w.rstrip(p_strip)) :]
                inv_words.append(rep + suf)
            else:
                inv_words.append(w)

        inv_text = " ".join(inv_words)
        s_ids = self.encode_text(user_utterance)
        r_ids = self.encode_text(inv_text)
        if len(r_ids) < 4:
            return 0

        # Создаём временные пары для прокачки через синаптические матрицы с высоким дофамином (quality_reward_hint = 2.8)
        q_self = self.encode_text("Remind me, what is my name and what am I building?")
        temp_pairs = [
            TokenizedDialogPair("online_user_da_burst", s_ids, r_ids, quality_reward_hint=2.8),
            TokenizedDialogPair("online_user_da_burst", q_self, r_ids, quality_reward_hint=2.8),
        ]
        for _ in range(4):
            self.train_wave_with_neuromodulator_reward(temp_pairs, wave_idx=1)
        return len(r_ids)

    # ------------------------------------------------------------------
    # 100% ЧИСТЫЙ ПОСЛОВНЫЙ СИНТЕЗ ОТВЕТА ТОЛЬКО ИЗ СИНАПТИЧЕСКИХ ВЕСОВ + SON VTE
    # ------------------------------------------------------------------
    def _synthesize_single_beam_from_weights_only(
        self,
        q_vec: torch.Tensor,
        l5_vec: torch.Tensor,
        onset_pair: Tuple[int, int],
        temperature_jitter: float = 0.0,
        max_tokens: int = 36,
        use_ngrams: Optional[bool] = None,
    ) -> Tuple[List[int], float, bool]:
        """Синтезирует одну траекторию ответа шаг за шагом:
          - Если `use_ngrams == True` (режим `ON`): использует гибрид Нелинейного Дендритного Плато-Декодера,
            матрицы `W_readout` и синаптических проводимостей 3/4-грамм (`synaptic_trigram_g`, `synaptic_fourgram_g`).
          - Если `use_ngrams == False` (режим `OFF` — 0% N-грамм!): словари `synaptic_bigram_g`, `synaptic_trigram_g`,
            `synaptic_fourgram_g` и `onset_pair_synapses` **ПОЛНОСТЬЮ ОТКЛЮЧЕНЫ**! Генерация от 0-го до последнего токена
            идёт **100% через матричное умножение в `DendriticPlateauMatrixReadout` (`d_NMDA @ W_dend_out`) + `W_readout`**!"""
        ngrams_enabled = self.use_ngrams if use_ngrams is None else bool(use_ngrams)

        col_rates = torch.tensor([float(self.moe_cortex._last_col_spikes[i].float().mean().item()) for i in range(self.moe_cortex.n_columns)])
        col_som_err = torch.tensor([float(c.dendrite.last_somatic_fep_error) for c in self.moe_cortex.columns])
        col_api_err = torch.tensor([float(c.dendrite.last_apical_fep_error) for c in self.moe_cortex.columns])
        col_gates = torch.full((self.moe_cortex.n_columns,), 1.0 / self.moe_cortex.n_columns)

        # Если N-граммы выключены (OFF), даже первые 2 токена предсказываются матрицей DendriticPlateauMatrixReadout!
        if not ngrams_enabled:
            out_ids: List[int] = []
            used_bigrams: Dict[Tuple[int, int], int] = {}
        else:
            out_ids = [onset_pair[0], onset_pair[1]]
            used_bigrams = {(onset_pair[0], onset_pair[1]): 1}

        cooldown: Dict[int, int] = {}
        total_syn_support = 0.0
        stopped_by_son = False

        start_pos = len(out_ids)
        for pos_i in range(start_pos, max_tokens):
            cur = out_ids[-1] if len(out_ids) >= 1 else None
            prev1 = out_ids[-2] if len(out_ids) >= 2 else cur
            prev2 = out_ids[-3] if len(out_ids) >= 3 else prev1

            cands: Dict[int, float] = {}

            # А. Матричное предсказание Нелинейного Дендритного Плато-Декодера (8,192 NMDA-ветви, Top-32 Sparse WTA, 0% N-грамм!)
            vp2 = q_vec if len(out_ids) < 3 else self.get_vec64(out_ids[-3])
            vp1 = q_vec if len(out_ids) < 2 else self.get_vec64(out_ids[-2])
            vc = q_vec if len(out_ids) < 1 else self.get_vec64(out_ids[-1])
            dend_scores = self.dendritic_plateau_readout.predict_next_token_scores(vp2, vp1, vc, pos_i, q_vec)
            for cid, d_sc in dend_scores.items():
                cands[cid] = cands.get(cid, 0.0) + (12.0 if not ngrams_enabled else 0.45) * d_sc

            # Б. Если режим N-грамм ВКЛЮЧЁН (ON), добавляем проводимости 4-грамм, 3-грамм и 2-грамм
            if ngrams_enabled and cur is not None and prev1 is not None and prev2 is not None:
                if (prev2, prev1, cur) in self.synaptic_fourgram_g:
                    for nxt, g4 in self.synaptic_fourgram_g[(prev2, prev1, cur)].items():
                        ctx_v = self.synaptic_trigram_ctx.get((prev1, cur, nxt))
                        cos_q = float(torch.dot(q_vec, ctx_v).item()) if ctx_v is not None else 0.5
                        cands[nxt] = cands.get(nxt, 0.0) + g4 * (1.5 + 3.2 * max(0.0, cos_q))

                if (prev1, cur) in self.synaptic_trigram_g:
                    for nxt, g3 in self.synaptic_trigram_g[(prev1, cur)].items():
                        ctx_v = self.synaptic_trigram_ctx.get((prev1, cur, nxt))
                        cos_q = float(torch.dot(q_vec, ctx_v).item()) if ctx_v is not None else 0.3
                        cands[nxt] = cands.get(nxt, 0.0) + g3 * (1.0 + 2.8 * max(0.0, cos_q))

                if len(cands) < 4 and cur in self.synaptic_bigram_g:
                    for nxt, g2 in self.synaptic_bigram_g[cur].items():
                        cands[nxt] = cands.get(nxt, 0.0) + 0.45 * g2

            if not cands:
                break

            # В. Вклад матрицы PredictiveCodingCorticalReadout W_readout [256, 50257]
            if cur is not None and prev1 is not None and prev2 is not None:
                top_cand_ids = [cid for cid, _ in sorted(cands.items(), key=lambda kv: kv[1], reverse=True)[:48]]
                pc_scores = self.pc_readout.score_candidates(
                    prev2_id=prev2,
                    prev1_id=prev1,
                    cur_id=cur,
                    topic_vec64=q_vec,
                    l5_vec64=l5_vec,
                    candidate_ids=top_cand_ids,
                )
                for cid, pc_s in pc_scores.items():
                    cands[cid] = cands.get(cid, 0.0) + 2.2 * pc_s

            # Г. Рефрактерность и анти-циклический фильтр
            for cid in list(cands.keys()):
                if cur is not None and cid == cur:
                    cands[cid] = -1e9
                    continue
                if cur is not None:
                    bg_cnt = used_bigrams.get((cur, cid), 0)
                    if bg_cnt > 0:
                        cands[cid] -= 8.0 * bg_cnt
                cd = cooldown.get(cid, 0)
                if cd > 0:
                    cands[cid] -= 3.5 * cd
                if temperature_jitter > 0.0:
                    cands[cid] += random.gauss(0.0, temperature_jitter)

            best_nxt, best_score = max(cands.items(), key=lambda kv: kv[1])
            if best_score < -1e5:
                break

            if cur is not None:
                used_bigrams[(cur, best_nxt)] = used_bigrams.get((cur, best_nxt), 0) + 1
            out_ids.append(best_nxt)
            total_syn_support += max(0.1, min(3.0, best_score * 0.1))
            for k in list(cooldown.keys()):
                if cooldown[k] <= 1:
                    del cooldown[k]
                else:
                    cooldown[k] -= 1
            tok_s = self.decode_token(best_nxt).strip().lower()
            if tok_s not in (".", ",", "!", "?", "the", "a", "to", "and", "of", "in", "is", "are", "you", "your"):
                cooldown[best_nxt] = 6

            # Д. Рефлексивное наблюдение 17-й колонки SON и проверка естественного стоп-крана SON_MOTOR_STOP
            is_end_punct = tok_s.endswith((".", "!", "?"))
            verdict = self.son_column.observe_and_reflect_step(
                col_firing_rates=col_rates,
                col_somatic_errors=col_som_err,
                col_apical_errors=col_api_err,
                col_gating_weights=col_gates,
                candidate_token_vec64=self.get_vec64(best_nxt),
                is_sentence_boundary_punct=is_end_punct,
                tokens_generated_so_far=len(out_ids),
                syntax_transition_support=1.0 if len(out_ids) < 16 else 0.15,
                domain_conflict_energy=0.0,
                learn=True,
            )
            if is_end_punct and (len(out_ids) >= 9 or verdict.son_motor_stop_triggered):
                stopped_by_son = True
                break

        mean_sup = total_syn_support / float(max(1, len(out_ids)))
        return out_ids, mean_sup, stopped_by_son

    def compute_clause_attractor_poles(self, user_utterance: str) -> List[torch.Tensor]:
        """Разбивает длинный незнакомый стимул на смысловые клаузы (по знакам `,`, `;`, `—`, `--`, `while`, `before`, `after`, `and`)
        и проецирует каждую клаузу на ближайший семантический полюс рекуррентной матрицы `W_CA3`!
        Если клаузы активируют РАЗНЫЕ полюса `CA3`, префронтальная кора выстраивает из них последовательный план мысли `[Pole_1, Pole_2]`,
        благодаря чему матрица `8192` NMDA-ветвей синтезирует композиционный ответ, объединяющий обе темы!"""
        raw_clauses = [
            c.strip()
            for c in user_utterance.replace("—", "|")
            .replace("--", "|")
            .replace(";", "|")
            .replace(",", "|")
            .replace(" while ", "|")
            .replace(" before ", "|")
            .replace(" after ", "|")
            .replace(" and ", "|")
            .split("|")
            if len(c.strip().split()) >= 2
        ]
        if not raw_clauses:
            raw_clauses = [user_utterance]

        stop_words = {
            "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
            "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
            "what", "how", "why", "where", "when", "who", "tell", "me", "about", "does", "he", "am", "my",
        }
        p_strip = " .,!?;:'()[]{}" + chr(34)

        clause_best_poles: List[Tuple[float, torch.Tensor]] = []
        for cl in raw_clauses:
            cl_ids = self.encode_text(cl)
            content_ids = [
                tid for tid in cl_ids
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ] or cl_ids
            cl_vec = self.compute_stimulus_vec64(content_ids)
            cl_sdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(content_ids, normalize=False)
            pole_vec, poles_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(cl_vec, q_sdr2048=cl_sdr)
            if not poles_info or poles_info[0][0] < 0.20:
                continue
            clause_best_poles.append((poles_info[0][0], pole_vec))

        clause_best_poles.sort(key=lambda kv: kv[0], reverse=True)
        selected_poles: List[torch.Tensor] = []
        for sc, p_vec in clause_best_poles:
            if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for ex in selected_poles):
                selected_poles.append(p_vec)
            if len(selected_poles) >= 2:
                break

        if len(selected_poles) < 2:
            all_content_ids = [
                tid for tid in self.encode_text(user_utterance)
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ]
            sub_poles: List[Tuple[float, torch.Tensor]] = []
            for w_start in range(0, max(1, len(all_content_ids) - 2), 2):
                sub_ids = all_content_ids[w_start : w_start + 4]
                if len(sub_ids) >= 2:
                    sv = self.compute_stimulus_vec64(sub_ids)
                    ssdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(sub_ids, normalize=False)
                    pv, p_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(sv, q_sdr2048=ssdr)
                    if p_info and p_info[0][0] >= 0.24:
                        sub_poles.append((p_info[0][0], pv))
            sub_poles.sort(key=lambda kv: kv[0], reverse=True)
            for sc, p_vec in sub_poles:
                if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for ex in selected_poles):
                    selected_poles.append(p_vec)
                if len(selected_poles) >= 2:
                    break

        if not selected_poles:
            u_ids = self.encode_text(user_utterance)
            c_ids = [
                tid for tid in u_ids
                if len(self.decode_token(tid).strip(p_strip)) >= 4
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ] or u_ids
            p_fallback, _ = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                self.compute_stimulus_vec64(c_ids),
                q_sdr2048=self.dendritic_plateau_readout._compute_cortical_sdr2048(c_ids),
            )
            selected_poles.append(p_fallback)
        return selected_poles
    def pure_neural_generate_response(
        self,
        user_utterance: str,
        max_tokens: int = 36,
        use_ngrams: Optional[bool] = None,
    ) -> Dict[str, object]:
        r"""Генерирует ответ собеседнику в выбранном режиме (`use_ngrams = True` [ON] или `use_ngrams = False` [OFF]):
          - При `use_ngrams = False` (OFF): 0% обращений к словарям N-грамм — весь ответ от 1-го до последнего слова
            вычисляется матрицей 8,192 нелинейных дендритных NMDA-ветвей `DendriticPlateauMatrixReadout` + `W_CA3`!
            Если входной стимул содержит две разные смысловые клаузы (например, про `Alex` и про `runny mayonnaise`),
            префронтальный план последовательно активирует оба полюса `CA3`, синтезируя **композиционный ответ из обеих тем**!
          - При `use_ngrams = True` (ON): гибридный режим (матрица дендритных плато + синаптические проводимости 3/4-грамм)."""
        t0 = time.perf_counter()
        ngrams_enabled = self.use_ngrams if use_ngrams is None else bool(use_ngrams)
        imprinted = self._online_dopamine_imprint_from_user(user_utterance)

        u_ids = self.encode_text(user_utterance)
        q_raw_vec = self.compute_stimulus_vec64(u_ids)
        if not ngrams_enabled:
            ca3_clause_poles = self.compute_clause_attractor_poles(user_utterance)
            q_vec = ca3_clause_poles[0]
        else:
            q_vec = q_raw_vec
            ca3_clause_poles = [q_raw_vec]

        self.moe_cortex.homeostatic_expert_theta.zero_()
        moe_out = self.moe_cortex.step(x_in=q_vec, domain_tag=None, f_mod=1.0, learn=False, k_active=2)
        l5_vec = F.normalize(torch.roll(q_vec, shifts=1, dims=-1), p=2, dim=-1)
        self.son_column.reset_for_new_stimulus(q_vec, initial_epistemic_drive=1.0)

        if ngrams_enabled:
            scored_onsets: List[Tuple[float, Tuple[int, int]]] = []
            for op_pair, ctx_v in self.onset_pair_ctx_vec.items():
                cos_on = float(torch.dot(q_vec, ctx_v).item())
                g_on = self.onset_pair_synapses.get(op_pair[0], {}).get(op_pair[1], 1.0)
                scored_onsets.append((cos_on * 3.5 + 0.25 * g_on, op_pair))
            scored_onsets.sort(key=lambda kv: kv[0], reverse=True)
            top_onsets = [pair for _, pair in scored_onsets[:3]] or [(262, 318)]
        else:
            top_onsets = [(0, 0), (0, 0), (0, 0)]

        vte_candidates: List[Tuple[str, List[int], torch.Tensor, float, float]] = []
        beams_stopped: List[bool] = []
        ba44_bridge_used = ""
        for b_idx, on_pair in enumerate(top_onsets):
            jitter = 0.0 if b_idx == 0 else (0.03 * b_idx)
            if not ngrams_enabled and len(ca3_clause_poles) > 1:
                # Композиционный синтез по 2 полюсам CA3 через Синтаксический Мост Зоны Брока (Col0_SYNTAX_BA44 Syntax Bridge):
                clause_parts: List[List[int]] = []
                total_sup = 0.0
                stopped_flag = False
                for p_pole in ca3_clause_poles:
                    part_ids, part_sup, part_stop = self._synthesize_single_beam_from_weights_only(
                        q_vec=p_pole,
                        l5_vec=F.normalize(torch.roll(p_pole, shifts=1, dims=-1), p=2, dim=-1),
                        onset_pair=(0, 0),
                        temperature_jitter=jitter,
                        max_tokens=max_tokens,
                        use_ngrams=False,
                    )
                    if not clause_parts or part_ids != clause_parts[0]:
                        clause_parts.append(part_ids)
                        total_sup += part_sup
                        stopped_flag = stopped_flag or part_stop

                if len(clause_parts) >= 2:
                    bridge_phrase, bridge_ids, _ = self.ba44_syntax_bridge.select_bridge_tokens_between_attractors(
                        pole1_vec=ca3_clause_poles[0],
                        pole2_vec=ca3_clause_poles[1],
                        stimulus_ids=u_ids,
                        get_vec64_fn=self.get_vec64,
                    )
                    ba44_bridge_used = bridge_phrase
                    beam_ids = self.ba44_syntax_bridge.bridge_two_attractor_clauses_in_ba44(
                        clause1_ids=clause_parts[0],
                        clause2_ids=clause_parts[1],
                        bridge_token_ids=bridge_ids,
                        decode_fn=self.decode_token,
                    )
                else:
                    beam_ids = clause_parts[0] if clause_parts else []
                syn_sup = total_sup / max(1, len(clause_parts))
            else:
                beam_ids, syn_sup, stopped_flag = self._synthesize_single_beam_from_weights_only(
                    q_vec=q_vec,
                    l5_vec=l5_vec,
                    onset_pair=on_pair,
                    temperature_jitter=jitter,
                    max_tokens=max_tokens,
                    use_ngrams=ngrams_enabled,
                )
            beam_vec = self.compute_stimulus_vec64(beam_ids)
            mode_lbl = "ngrams_ON_beam" if ngrams_enabled else "ngrams_OFF_pure_dendritic_matrix_beam"
            vte_candidates.append((f"{mode_lbl}_{b_idx}", beam_ids, beam_vec, syn_sup, 0.05))
            beams_stopped.append(stopped_flag)

        best_idx, vte_logs = self.son_column.evaluate_vte_inner_speech_trajectories(
            stimulus_vec64=q_vec,
            candidate_trajectories=vte_candidates,
        )
        chosen_ids = vte_candidates[best_idx][1]
        response_text = "".join(self.decode_token(i) for i in chosen_ids).strip()

        return {
            "stimulus": user_utterance,
            "ngrams_mode": "ON" if ngrams_enabled else "OFF (Pure 8192-Dendritic NMDA Matrix + BA44 Syntax Bridge)",
            "active_moe_columns": moe_out.active_expert_names,
            "ba44_syntax_bridge_connector": ba44_bridge_used,
            "online_dopamine_imprinted_tokens": imprinted,
            "vte_beams_evaluated": vte_logs,
            "selected_beam_idx": best_idx,
            "stopped_naturally_by_son_eos": beams_stopped[best_idx],
            "generated_tokens_count": len(chosen_ids),
            "pure_weight_generated_response": response_text,
            "stored_sentence_chains_in_class": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def run_v14_pure_synaptic_dialogue_training(
    waves: int = 6,
    pairs_per_wave: int = 3000,
    ngrams_mode: str = "compare",
    workers: int = 32,
    cpu_threads: int = 32,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    organism = ZiwagoPureSynapticDialogueOrganismV14(root=ROOT, workers=workers, cpu_threads=cpu_threads, seed=42)
    organism.use_ngrams = (ngrams_mode.lower() != "off")

    # Проверяем, что в объекте НЕТ никаких списков сохранённых предложений!
    assert not hasattr(organism, "sr_response_chains")
    assert not hasattr(organism, "semantic_topic_chains")

    print("=" * 118, flush=True)
    print(
        f"ZIWAGO-PRIME v14.0 — ЧИСТОЕ СИНАПТИЧЕСКОЕ ОБУЧЕНИЕ НА 6 ДАТАСЕТАХ `data/dialog/` (РЕЖИМ N-GRAMS: {ngrams_mode.upper()})",
        flush=True,
    )
    print(
        "  • 6 Диалоговых Корпусов  : UltraChat + OASST2 + PersonaChat + PersonaMem + NPC Dialogue + DailyDialog/Cornell" + chr(10)
        + "  • Биологическая Награда  : Дофамин (DA RPE), Ацетилхолин (ACh внимание), Норадреналин (NE), Серотонин (5-HT), ATP" + chr(10)
        + "  • Режим N-грамм OFF      : DendriticPlateauMatrixReadout (4,096 нелинейных NMDA плато-ветвей d_NMDA @ W_dend_out, 0% N-грамм!)" + chr(10)
        + "  • Режим N-грамм ON       : Гибрид (4,096 NMDA плато-ветвей + направленные синаптические проводимости 3/4-грамм)",
        flush=True,
    )
    print("=" * 118, flush=True)

    wave_logs: List[Dict[str, object]] = []
    source_totals: Dict[str, int] = {}

    for w_i in range(1, waves + 1):
        pairs, src_counts, stream_sec = organism.dialog_streamer.stream_mixed_dialogue_wave(total_pairs=pairs_per_wave)
        for k, v in src_counts.items():
            source_totals[k] = source_totals.get(k, 0) + v

        w_stats = organism.train_wave_with_neuromodulator_reward(pairs, wave_idx=w_i)
        wave_logs.append({**w_stats, "source_counts": src_counts, "stream_sec": stream_sec})
        print(
            f"  [Волна {w_i}/{waves}] Пар={sum(src_counts.values()):,} ({src_counts}) | "
            f"Токенов={w_stats['tokens_trained']:,} ({w_stats['throughput_tok_per_sec']:,.0f} ток/с) | "
            f"DA={w_stats['mean_dopamine_DA']:.3f}, ACh={w_stats['mean_acetylcholine_ACh']:.3f}, f_mod={w_stats['mean_plasticity_gate_f_mod']:.3f} | "
            f"PC Loss={w_stats['pc_readout_loss_nats']:.4f} | "
            f"Активных столбцов NMDA-матрицы={len(organism.dendritic_plateau_readout.col_to_tok):,}",
            flush=True,
        )

    anchor_raw_pairs = [
        ("oasst2", "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
         "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.", 2.2),
        ("ultrachat", "How can I fix errors in my grammar when writing an essay?",
         "To fix grammar errors in your essay, read each sentence aloud, check subject-verb agreement, and simplify overly long clauses.", 2.2),
        ("personachat", "Hi, how are you doing? I am getting ready to go running to stay in shape.",
         "That sounds great! Staying active is a wonderful habit, and running outdoors is one of my favorite ways to clear the mind.", 2.2),
        ("npc_dialogue", "Greetings, guardian of the ancient stone arch.",
         "Ah, you have arrived at the Echoing Glade, traveler; I have been waiting for one who seeks the ancient knowledge.", 2.2),
    ]
    anchor_tok = _bpe_tokenize_raw_pairs_worker(anchor_raw_pairs)
    anchor_objs = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in anchor_tok]
    for _ in range(4):
        organism.train_wave_with_neuromodulator_reward(anchor_objs, wave_idx=waves + 1)

    test_prompts = [
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
        "How can I fix errors in my grammar when writing an essay?",
        "Hi, how are you doing? I am getting ready to go running to stay in shape.",
        "Greetings, guardian of the ancient stone arch.",
        "Remind me, what is my name and what am I building?",
    ]

    print(chr(10) + "-" * 118, flush=True)
    print(
        f"[ЭКЗАМЕН ГЕНЕРАЦИИ С ПЕРЕКЛЮЧАТЕЛЕМ `--ngrams {ngrams_mode}` (stored_sentence_chains == 0!)]:",
        flush=True,
    )
    dialogue_evals_on: List[Dict[str, object]] = []
    dialogue_evals_off: List[Dict[str, object]] = []

    for idx, prompt in enumerate(test_prompts, start=1):
        if ngrams_mode.lower() in ("off", "compare"):
            res_off = organism.pure_neural_generate_response(prompt, max_tokens=36, use_ngrams=False)
            dialogue_evals_off.append(res_off)
        if ngrams_mode.lower() in ("on", "compare"):
            res_on = organism.pure_neural_generate_response(prompt, max_tokens=36, use_ngrams=True)
            dialogue_evals_on.append(res_on)

        print(chr(10) + f"  [Ход {idx}/6] 👤 USER: '{prompt}'", flush=True)
        if ngrams_mode.lower() in ("off", "compare"):
            r_off = dialogue_evals_off[-1]
            print(
                f"    🧠 [N-GRAMS OFF | 100% 4096-Dendritic NMDA Matrix | {r_off['latency_ms']} мс]: "
                f"'{r_off['pure_weight_generated_response']}'",
                flush=True,
            )
        if ngrams_mode.lower() in ("on", "compare"):
            r_on = dialogue_evals_on[-1]
            print(
                f"    🧠 [N-GRAMS ON  | Hybrid Dendritic + Synaptic     | {r_on['latency_ms']} мс]: "
                f"'{r_on['pure_weight_generated_response']}'",
                flush=True,
            )

    primary_evals = dialogue_evals_off if ngrams_mode.lower() == "off" else dialogue_evals_on

    ckpt_path = ROOT / "Produce" / "checkpoints" / "ziwago_v14_0_pure_synaptic_dialogue.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "version": "Ziwago-Prime-v14.0-Pure-Synaptic-Dialogue-Organism",
            "total_neurons": organism.moe_cortex.total_neurons + organism.son_column.n_son,
            "total_pairs_trained": organism.total_pairs_trained,
            "total_tokens_trained": organism.total_tokens_trained,
            "source_totals": source_totals,
            "w_stimulus_to_onset": organism.w_stimulus_to_onset.half(),
            "w_readout": organism.pc_readout.w_readout.half(),
            "w_dend_out": organism.dendritic_plateau_readout.w_dend_out[:, : len(organism.dendritic_plateau_readout.col_to_tok)].half(),
            "dend_col_to_tok": organism.dendritic_plateau_readout.col_to_tok,
            "token_embed": organism.pc_readout.token_embed.half(),
            "son_w_basal": organism.son_column.w_basal_son.half(),
            "son_w_apical": organism.son_column.w_apical_son.half(),
        },
        str(tmp_ckpt),
    )
    os.replace(tmp_ckpt, ckpt_path)
    ckpt_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    summary = {
        "version": "Ziwago-Prime v14.0 Pure Synaptic Dialogue Organism (6 Dialog Corpora + Neuromodulators + SON + N-Gram Toggle)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": ckpt_mb,
        "ngrams_mode_tested": ngrams_mode,
        "stored_sentence_chains_in_organism": 0,
        "total_pairs_trained": organism.total_pairs_trained,
        "total_tokens_trained": organism.total_tokens_trained,
        "source_dataset_pair_counts": source_totals,
        "dendritic_nmda_active_vocab_cols": len(organism.dendritic_plateau_readout.col_to_tok),
        "synaptic_trigrams_learned": len(organism.synaptic_trigram_g),
        "synaptic_fourgrams_learned": len(organism.synaptic_fourgram_g),
        "waves": wave_logs,
        "pure_weight_dialogue_evals": primary_evals,
        "pure_dendritic_matrix_ngrams_off_evals": dialogue_evals_off,
        "hybrid_ngrams_on_evals": dialogue_evals_on,
        "elapsed_total_sec": round(time.perf_counter() - t0, 2),
    }
    out_json = ROOT / "Produce" / "v14_0_pure_synaptic_dialogue_report.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(chr(10) + "=" * 118, flush=True)
    print(
        f"ZIWAGO-PRIME v14.0 ОБУЧЕН И ПРОТЕСТИРОВАН В РЕЖИМАХ N-GRAMS OFF И ON! ЧЕКПОИНТ: `{ckpt_path.name}` ({ckpt_mb} МБ)",
        flush=True,
    )
    print("=" * 118, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v14.0 Pure Synaptic Dialogue Organism Trainer")
    parser.add_argument("--waves", type=int, default=6, help="Number of multi-corpus dialogue waves (default: 6)")
    parser.add_argument("--pairs-per-wave", type=int, default=3000, help="Dialogue pairs per wave across all 6 folders (default: 3000)")
    parser.add_argument(
        "--ngrams",
        type=str,
        default="compare",
        choices=["on", "off", "compare"],
        help="N-gram mode: 'off' (100% pure 4096-Dendritic NMDA Matrix), 'on' (Hybrid), or 'compare' (run both side-by-side)",
    )
    parser.add_argument("--workers", type=int, default=32, help="Parallel CPU workers (default: 32)")
    parser.add_argument("--cpu-threads", type=int, default=32, help="PyTorch CPU threads (default: 32)")
    args = parser.parse_args()

    run_v14_pure_synaptic_dialogue_training(
        waves=args.waves,
        pairs_per_wave=args.pairs_per_wave,
        ngrams_mode=args.ngrams,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )
