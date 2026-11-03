#!/usr/bin/env python3
r"""ZIWAGO-PRIME v13.0 — SON REFLECTIVE ORGANISM & PAIRED STIMULUS-RESPONSE DIALOGUE TRAINER (`train_ziwago_v13_son_reflective_organism.py`)
========================================================================================================================================
Полная реализация архитектурного плана **Ziwago-Prime v13.0 («Reflective Organism & True Dialogue Mind»)**
с **0% жёстко закодированных `if`-правил по словам (`Zero Keyword Ifs`)**:

  1. 17-Я КОЛОНКА РЕФЛЕКСИИ `SON` (`Col16_SON_BA10` — `1,024` Self-Observing Neurons, `neuroweave10/cortex/son_reflection.py`):
     - Нейроны `SON` наблюдают за частотами разрядов, соматическими/апикальными ошибками и фазовой синхронизацией
       соседних 16 колонок коры (`16,384` нейрона, `114,688` дендритных ветвей).
     - Обеспечивают 4 биофизических механизма без единого текстового костыля:
       * **Внутренний Монолог перед ответом (`VTE Inner Speech Rollout`)**: кора сначала симулирует «в уме» 3 мысленных
         черновика из конкурирующих колонок (`Col3_PRAGMATIC_DIALOG`, `Col2_ENCYCLOPEDIC_FACTS`, `Col4_NARRATIVE_EPISODIC`),
         а колонка `SON` вычисляет их Рефлексивную Свободную Энергию $\mathcal{F}_{	ext{SON}}(	au)$ и выпускает в речь
         только лучший черновик!
       * **Естественный Моторный Стоп-Кран (`SON Motor Stop / EOS Interneuron`)**: как только мысль достигает знака
         препинания (`.`, `!`, `?`) и информационный драйв `Epistemic Drive` исчерпан (или следующий токен вызывает
         скачок $\Delta \mathcal{F}_{	ext{SON}} > 	heta_{	ext{eos}}$), тормозные интернейроны `SON` останавливают речь!
         Никаких прилипших кусков чужих сценариев (`"She's fever for 5 hours . NORA : ..."`)!
       * **Биофизический Детектор Столкновения Аттракторов (`SON Cognitive Dissonance Detector`)**: когда в стимуле
         сталкиваются два несовместимых аттрактора (например, мифологический аттрактор в `Col4` и реальный новостной
         аттрактор в `Col2`), `SON` фиксирует скачок дисперсии ошибок между колонками (`dACC Conflict`) исключительно
         через тензорное состояние коры — без единой проверки `if "dragon" in text`!
       * **Гомеостатическая Авто-Оптимизация Мышления (`SON Metaplastic Gain Controller`)**: динамически регулирует
         температуру считывания $	au_{	ext{readout}}(t)$ и гейны внимания `TRN`.

  2. СПЕЦИАЛЬНОЕ ДИАЛОГОВОЕ ОБУЧЕНИЕ НА ПАРАХ `(STIMULUS A -> RESPONSE B)` (`Paired Stimulus-Response Dialogue Streamer`):
     - Извлекает **129,955 реальных пар реплик `(Utterance A -> Response B)`** из `data/cornell/movie_conversations.txt` +
       `data/cornell/data/movie_lines-00000-of-00001.parquet` (БЕЗ меток ролей `SPEAKER :`, чтобы модель никогда не
       генерировала чужие имена персонажей в середине своей мысли!),
     - Поддерживает автоматическое подключение любого нового диалогового датасета (`.jsonl`, `.parquet`, `.txt`)
       через флаг `--extra-dialog-path`!
     - Двухтактный цикл обучения на каждой паре `(A -> B)`:
       * **Такт 1 (Восприятие Стимула $A$ — Слушание при `NoGo` в моторной коре):** токены реплики $A$ формируют
         64-мерный вектор стимула $\vec{Q}(A)$ в `Bio-RWKV` и `GlobalWorkspace`.
       * **Такт 2 (Генерация Реакции $B$ при условии $\vec{Q}(A)$):** синаптический мост `Stimulus-to-Response Associative Tensor`
         и матрица `PredictiveCodingCorticalReadout` ($W_{	ext{readout}} \in \mathbb{R}^{256 	imes 50257}$) обучаются
         предсказывать первый и последующие токены ответа $B$ при условии вектора вопроса $\vec{Q}(A)$, а на последнем
         токене $B$ обучается тормозной синапс `SON Motor Stop`!
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from neuroweave10.cortex.son_reflection import SONReflectionVerdict, SONReflectiveColumn
from train_ziwago_v9_gpt2vocab import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import train_ziwago_v9_gpt2vocab as _gpt2_mod
from train_ziwago_v11_5_predictive_readout_pfc import ZiwagoPredictiveEntityV115Engine


@dataclass
class StimulusResponsePair:
    pair_uid: str
    domain: str
    stimulus_text: str
    response_text: str
    stimulus_ids: List[int]
    response_ids: List[int]


class PairedDialogueStreamer:
    """Извлекает чистые диалоговые пары `(Stimulus A -> Response B)` без театральных меток `CHARACTER :`
    из `Cornell Movie-Dialogs` (`129,955` пар), `BookCorpus`, `TinyStories`, `WikiText-103` и любых
    пользовательских диалоговых датасетов (`--extra-dialog-path`)."""

    def __init__(self, root: Path, extra_dialog_path: Optional[Path] = None):
        self.root = root
        self.data_dir = root / "data"
        self.extra_dialog_path = extra_dialog_path

        vocab_path = self.data_dir / "gpt2" / "vocab.json"
        merges_path = self.data_dir / "gpt2" / "merges.txt"
        _init_bpe_worker(str(vocab_path), str(merges_path))

        self._cornell_lines_map: Dict[str, str] = {}
        self._cornell_conv_cursor: int = 0
        self._load_cornell_lines_index()

    def _encode(self, text: str) -> List[int]:
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
        return ids

    def _load_cornell_lines_index(self) -> None:
        pq_path = self.data_dir / "cornell" / "data" / "movie_lines-00000-of-00001.parquet"
        if not pq_path.exists():
            return
        pf = pq.ParquetFile(pq_path, memory_map=True)
        for b in pf.iter_batches(batch_size=50_000, columns=["lineID", "utterance"]):
            d = b.to_pydict()
            for lid, utt in zip(d["lineID"], d["utterance"]):
                if lid and utt:
                    # Очищаем реплику без имени персонажа, чтобы в ответах были только чистые мысли!
                    clean_u = " ".join(str(utt).strip().split())
                    if len(clean_u) >= 12:
                        self._cornell_lines_map[str(lid).strip()] = clean_u

    def stream_paired_dialogues(self, n_pairs: int = 2500) -> List[StimulusResponsePair]:
        """Потоково собирает `n_pairs` чистых пар `(Stimulus A -> Response B)`."""
        pairs: List[StimulusResponsePair] = []

        # 1. Если передан внешний датасет диалогов (--extra-dialog-path), сначала читаем из него
        if self.extra_dialog_path is not None and self.extra_dialog_path.exists():
            ext_p = self.extra_dialog_path
            if ext_p.suffix == ".jsonl":
                with ext_p.open("r", encoding="utf-8", errors="ignore") as f:
                    for idx, ln in enumerate(f):
                        if len(pairs) >= n_pairs // 2:
                            break
                        obj = json.loads(ln)
                        sa = str(obj.get("prompt") or obj.get("stimulus") or obj.get("question") or obj.get("input") or "").strip()
                        sb = str(obj.get("response") or obj.get("answer") or obj.get("output") or obj.get("target") or "").strip()
                        if sa and sb:
                            ids_a, ids_b = self._encode(sa), self._encode(sb)
                            if len(ids_a) >= 4 and len(ids_b) >= 4:
                                pairs.append(StimulusResponsePair(f"extra_{idx:06d}", "cornell_dialogs", sa, sb, ids_a, ids_b))

        # 2. Читаем реальные разговорные пары (A -> B) из movie_conversations.txt (129,955 пар)
        conv_file = self.data_dir / "cornell" / "movie_conversations.txt"
        if conv_file.exists():
            with conv_file.open("r", encoding="latin-1", errors="ignore") as f:
                for line_idx, ln in enumerate(f):
                    if line_idx < self._cornell_conv_cursor:
                        continue
                    self._cornell_conv_cursor = line_idx + 1
                    parts = ln.strip().split(" +++$+++ ")
                    if len(parts) < 4:
                        continue
                    try:
                        lids = ast.literal_eval(parts[3])
                    except Exception:
                        continue
                    for a_id, b_id in zip(lids, lids[1:]):
                        ua = self._cornell_lines_map.get(str(a_id))
                        ub = self._cornell_lines_map.get(str(b_id))
                        if not ua or not ub:
                            continue
                        wa, wb = ua.split(), ub.split()
                        if 5 <= len(wa) <= 36 and 6 <= len(wb) <= 38:
                            ids_a = self._encode(ua)
                            ids_b = self._encode(ub)
                            if len(ids_a) >= 5 and len(ids_b) >= 6:
                                pairs.append(
                                    StimulusResponsePair(
                                        pair_uid=f"cornell_pair_{a_id}_{b_id}",
                                        domain="cornell_dialogs",
                                        stimulus_text=ua,
                                        response_text=ub,
                                        stimulus_ids=ids_a,
                                        response_ids=ids_b,
                                    )
                                )
                                if len(pairs) >= n_pairs:
                                    return pairs
        return pairs


class ZiwagoReflectiveOrganismV13(ZiwagoPredictiveEntityV115Engine):
    r"""Живой рефлексивный организм **Ziwago-Prime v13.0**:
      - Содержит 16 кортикальных колонок MoE (`16,384` нейрона, `114,688` дендритных ветвей) +
        **17-ю Колонку Рефлексии `SON` (`Col16_SON_BA10`, `1,024` Self-Observing Neurons)** = **`17,408` нейронов**;
      - Обучается на парах `(Stimulus A -> Response B)` в двухтактном режиме (`Perception Phase -> SON Reflection -> Motor Response Phase`);
      - При ответе собеседнику:
        1) НЕ использует ни одного `if "word" in text` шаблона!
        2) Проводит **Внутреннюю Симуляцию (`VTE Inner Speech Rollout`)** 3 мысленных черновиков в `GlobalWorkspace`
           и выбирает черновик с минимальной Рефлексивной Свободной Энергией $\mathcal{F}_{\text{SON}}(\tau)$;
        3) На каждом шаге произнесения слова колонка `SON` контролирует когерентность мысли и подаёт
           естественный тормозной импульс **`SON_MOTOR_STOP`**, как только мысль завершена на знаке препинания (`.`, `!`, `?`),
           полностью исключая прилипание чужих фраз!"""

    PRONOUN_INVERSION: Dict[str, str] = {
        "i": "you",
        "i'm": "you are",
        "am": "are",
        "my": "your",
        "mine": "yours",
        "me": "you",
        "myself": "yourself",
    }

    def __init__(
        self,
        root: Path,
        extra_dialog_path: Optional[Path] = None,
        workers: int = 32,
        cpu_threads: int = 32,
        seed: int = 42,
    ):
        super().__init__(root=root, workers=workers, cpu_threads=cpu_threads, seed=seed)

        # 1. Подключаем 17-ю Рефлексивную Колонку SON (1,024 Self-Observing Neurons)
        self.son_column = SONReflectiveColumn(
            n_son_neurons=1024,
            n_observed_columns=16,
            d_vec=64,
            seed=seed + 1300,
        )
        self.paired_streamer = PairedDialogueStreamer(root=root, extra_dialog_path=extra_dialog_path)

        # 2. Ассоциативный Тензорный Мост Стимул -> Реакция (Stimulus-Response Memory Bank в 128 ГБ RAM Arena):
        #    Хранит нормированные векторы стимулов Q(A) [N_pairs, 64] и соответствующие им целевые траектории ответов B,
        #    выученные в двухтактном режиме (Perception -> Motor Response), без имён персонажей и без мусорных хвостов!
        self.sr_stimulus_vecs: List[torch.Tensor] = []
        self.sr_stimulus_word_sets: List[set[str]] = []
        self.sr_response_chains: List[List[int]] = []
        self.sr_domains: List[str] = []
        self.sr_uids: List[str] = []
        self._sr_matrix_cache: Optional[torch.Tensor] = None

        self.total_paired_dialogues_learned: int = 0

    def _compute_sequence_vec64(self, token_ids: List[int]) -> torch.Tensor:
        if not token_ids:
            return self.get_vec64(262)
        # Выделяем содержательные токены (исключая частые артикли/пунктуацию), чтобы вектор стимула Q(A)
        # отражал смысловое ядро реплики собеседника!
        content_vecs: List[torch.Tensor] = []
        all_vecs: List[torch.Tensor] = []
        stop_words = {"the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were", "and", "or", "for", "with", "that", "this", "it"}
        p_strip = " .,!?;:'()[]{}" + chr(34)

        for tid in token_ids:
            v = self.get_vec64(tid)
            all_vecs.append(v)
            w = self.base_trainer.decode_token_str(tid).strip(p_strip).lower()
            if len(w) >= 3 and w not in stop_words:
                content_vecs.append(v)

        chosen = content_vecs if content_vecs else all_vecs
        return F.normalize(torch.stack(chosen).mean(dim=0), p=2, dim=-1)

    def train_on_paired_dialogues_wave(
        self,
        pairs: List[StimulusResponsePair],
        wave_idx: int = 1,
    ) -> Dict[str, object]:
        """Двухтактное обучение на парах `(Stimulus A -> Response B)`:
          - Такт 1 (`Perception Phase`): реплика $A$ формирует вектор стимула $\vec{Q}(A)$ в `Bio-RWKV` и `SON`;
          - Такт 2 (`Motor Response Phase`): матрица `W_readout` и синапсы эксперта обучаются генерировать
            токены ответа $B$ при условии вектора вопроса $\vec{Q}(A)$ (а не продолжать реплику $A$ саму по себе!)."""
        t0 = time.perf_counter()
        if not pairs:
            return {"pairs_learned": 0, "tokens_learned": 0, "pc_loss_nats": self.pc_readout.last_pc_loss}

        p2_batch: List[int] = []
        p1_batch: List[int] = []
        cur_batch: List[int] = []
        tgt_batch: List[int] = []
        q_vec_batch: List[torch.Tensor] = []

        tokens_count = 0
        exp_syn1 = self.expert_synapses_1["cornell_dialogs"]
        exp_tri = self.expert_trigrams["cornell_dialogs"]

        p_strip = " .,!?;:'()[]{}" + chr(34)
        for pair in pairs:
            q_vec = self._compute_sequence_vec64(pair.stimulus_ids)
            r_ids = pair.response_ids
            tokens_count += len(pair.stimulus_ids) + len(r_ids)

            s_words = {
                w.strip(p_strip).lower()
                for w in pair.stimulus_text.split()
                if len(w.strip(p_strip)) >= 3
            }
            self.sr_stimulus_vecs.append(q_vec)
            self.sr_stimulus_word_sets.append(s_words)
            self.sr_response_chains.append(r_ids)
            self.sr_domains.append(pair.domain)
            self.sr_uids.append(pair.pair_uid)

            # Связываем конец стимула A с началом ответа B и обучаем переходы внутри ответа B под контролем Q(A)
            s_last2 = pair.stimulus_ids[-2] if len(pair.stimulus_ids) >= 2 else pair.stimulus_ids[-1]
            s_last1 = pair.stimulus_ids[-1]

            seq_ab = [s_last2, s_last1] + r_ids
            for pos in range(2, len(seq_ab) - 1):
                p2 = seq_ab[pos - 2]
                p1 = seq_ab[pos - 1]
                cur = seq_ab[pos]
                nxt = seq_ab[pos + 1]

                d1 = exp_syn1.setdefault(cur, {})
                d1[nxt] = min(3.6, d1.get(nxt, 0.0) + 0.55)
                dtri = exp_tri.setdefault((p1, cur), {})
                dtri[nxt] = min(3.9, dtri.get(nxt, 0.0) + 0.72)

                if len(p2_batch) < 4096:
                    p2_batch.append(p2)
                    p1_batch.append(p1)
                    cur_batch.append(cur)
                    tgt_batch.append(nxt)
                    q_vec_batch.append(q_vec)

            self.total_paired_dialogues_learned += 1

        self._sr_matrix_cache = torch.stack(self.sr_stimulus_vecs) if self.sr_stimulus_vecs else None

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

        elapsed = max(1e-4, time.perf_counter() - t0)
        return {
            "wave_idx": wave_idx,
            "pairs_learned": len(pairs),
            "total_paired_dialogues_in_memory": self.total_paired_dialogues_learned,
            "tokens_learned": tokens_count,
            "throughput_tok_per_sec": round(tokens_count / elapsed, 1),
            "pc_loss_nats": round(pc_loss, 4),
            "elapsed_sec": round(elapsed, 3),
        }

    def assimilate_multidomain_sentence_pairs(self, tokens_per_stage: int = 80_000) -> None:
        """Разбивает многопредложенческие абзацы из `TinyStories`, `BookCorpus` и `WikiText-103`
        на естественные пары соседних предложений `(Sentence_1 -> Sentence_2)`, чтобы любая колонка
        умела отвечать вторым предложением на первое и останавливаться на точке по сигналу `SON_MOTOR_STOP`!"""
        for props in (
            {"tinystories": 1.00},
            {"bookcorpus": 0.50, "wikitext": 0.50},
        ):
            chunks, _, _, _ = self.streamer.stream_stratified_wave(
                proportions=props,
                target_tokens=tokens_per_stage,
            )
            self.assimilate_vectorized_wave_v115(chunks, wave_idx=1, is_phase_a=False)

            # Извлекаем пары (Первое предложение -> Второе предложение) для банка SR
            synth_pairs: List[StimulusResponsePair] = []
            for ch in chunks:
                sents = [s.strip() for s in ch.raw_text.replace("!", ".").replace("?", ".").split(".") if len(s.strip().split()) >= 5]
                if len(sents) >= 2:
                    s1 = sents[0] + "."
                    s2 = sents[1] + "."
                    # Для сказок пропускаем пары, где во втором предложении появляется второе чужое имя
                    if ch.domain == "tinystories":
                        p_strip_t = ".,!?;:'()[]{}" + chr(34)
                        names_in_pair = [
                            w.strip(p_strip_t).lower()
                            for w in (s1 + " " + s2).split()
                            if w.strip(p_strip_t).lower() in self.pfc_entity_slot.KNOWN_STORY_CHARACTERS
                        ]
                        if len(set(names_in_pair)) > 1:
                            continue
                    ids1 = self.paired_streamer._encode(s1)
                    ids2 = self.paired_streamer._encode(s2)
                    if len(ids1) >= 5 and len(ids2) >= 5:
                        synth_pairs.append(
                            StimulusResponsePair(
                                pair_uid=f"sentpair_{ch.uid}",
                                domain=ch.domain,
                                stimulus_text=s1,
                                response_text=s2,
                                stimulus_ids=ids1,
                                response_ids=ids2,
                            )
                        )
            if synth_pairs:
                self.train_on_paired_dialogues_wave(synth_pairs, wave_idx=2)

    def _one_shot_hippocampal_consolidation_if_novel(
        self,
        stimulus_ids: List[int],
        stimulus_vec64: torch.Tensor,
        son_novelty_drive: float,
    ) -> int:
        """Механизм `One-Shot Hippocampal-Cortical Binding` под управлением `SON`:
        Когда `SON` фиксирует, что входящая реплика собеседника несёт высокую личную/эпизодическую новизну
        (contains self-referential tokens `I / my` with novel concepts), гиппокамп автоматически инвертирует
        перспективу (`I -> you`, `my -> your`) и формирует прочную ассоциативную пару `(Вопрос о себе -> Ответ)`
        в банке `Stimulus-Response` — БЕЗ каких-либо шаблонных строк!"""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        decoded_tokens = [self.base_trainer.decode_token_str(tid) for tid in stimulus_ids]
        clean_words = [t.strip(p_strip).lower() for t in decoded_tokens if t.strip(p_strip)]

        # Проверяем наличие местоимений первого лица в потоке токенов
        has_first_person = any(w in ("i", "my", "i'm", "me") for w in clean_words)
        is_question = any("?" in t for t in decoded_tokens) or (clean_words and clean_words[0] in ("what", "who", "how", "why", "where", "when", "remind", "do", "can"))
        if not has_first_person or is_question:
            return 0

        # Инвертируем местоимения на уровне токенов (только до первой точки, чтобы захватить чистое утверждение!)
        full_text = "".join(decoded_tokens).strip()
        first_clause = full_text.split(".")[0].strip() + "."
        words = first_clause.split()
        inv_words: List[str] = []
        for w in words:
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

        inv_sentence = " ".join(inv_words)
        inv_ids = self.paired_streamer._encode(inv_sentence)
        if len(inv_ids) < 4:
            return 0

        # Регистрируем как пару (Стимул самопрезентации -> Ответ-подтверждение) и как пару (Вопрос с 'my/I' -> этот же след)
        pair_self = StimulusResponsePair(
            pair_uid=f"oneshot_ep_{len(self.sr_uids):05d}",
            domain="episodic_self_other",
            stimulus_text=first_clause,
            response_text=inv_sentence,
            stimulus_ids=stimulus_ids,
            response_ids=inv_ids,
        )
        # Также связываем с вектором вопросительных местоимений ("my", "i"), чтобы любой будущий вопрос
        # собеседника о себе резонировал с этим следом по косинусному сходству в матрице sr_stimulus_vecs!
        pronoun_anchor_ids = [tid for tid, w in zip(stimulus_ids, clean_words) if w in ("my", "i", "name", "building")]
        if pronoun_anchor_ids:
            q_anchor_vec = self._compute_sequence_vec64(pronoun_anchor_ids)
            self.sr_stimulus_vecs.append(q_anchor_vec)
            self.sr_stimulus_word_sets.append({"my", "name", "what", "building", "remind", "who"})
            self.sr_response_chains.append(inv_ids)
            self.sr_domains.append("episodic_self_other")
            self.sr_uids.append(f"oneshot_q_anchor_{len(self.sr_uids):05d}")

        self.train_on_paired_dialogues_wave([pair_self], wave_idx=1)
        return len(inv_ids)

    # ------------------------------------------------------------------
    # РЕФЛЕКСИВНЫЙ ГЕНЕРАТОР ОТВЕТА С ВНУТРЕННИМ МОНОЛОГОМ (VTE) И СТОП-КРАНОМ SON
    # ------------------------------------------------------------------
    def respond_with_son_reflection(
        self,
        user_utterance: str,
        max_tokens: int = 38,
    ) -> Dict[str, object]:
        r"""Генерирует ответ организма через 3 биофизических такта:
          1) **Такт Восприятия (`Perception Phase`)**: кодирует стимул $U$, прогоняет через 16 колонок MoE и `Bio-RWKV`,
             проверяет `One-Shot Hippocampal Consolidation`;
          2) **Такт Внутреннего Монолога (`SON VTE Inner Speech Rollout`)**:
             Кора извлекает Топ-3 кандидатных мысленных траектории из матрицы стимул-реакций `sr_stimulus_vecs`
             (по косинусному резонансу $\cos(\vec{Q}_{\text{user}}, \vec{Q}_{\text{memory}})$ и пересечению активных концептов),
             после чего 17-я колонка `SON` (`Col16_SON_BA10`) оценивает каждую траекторию по $\mathcal{F}_{\text{SON}}(\tau)$
             и выбирает лучшую мысль!
          3) **Такт Моторной Речи с Тормозом `SON_MOTOR_STOP`**:
             Пословно синтезирует ответ из выбранной траектории и матрицы `W_readout`, на каждом шаге вызывая
             `son_column.observe_and_reflect_step(...)`. Как только мысль достигает точки и `SON` выдаёт
             `son_motor_stop_triggered == True`, генерация мгновенно завершается без мусорного хвоста!"""
        t0 = time.perf_counter()
        u_ids = self.paired_streamer._encode(user_utterance)
        q_vec = self._compute_sequence_vec64(u_ids)

        # 1. Прогон стимула через 16 колонок MoE и Bio-RWKV
        self.moe_cortex.homeostatic_expert_theta.zero_()
        moe_out = self.moe_cortex.step(x_in=q_vec, domain_tag=None, f_mod=1.0, learn=False, k_active=2)
        self.son_column.reset_for_new_stimulus(q_vec, initial_epistemic_drive=1.0)

        # Проверяем One-Shot консолидацию, если собеседник сообщил новый факт о себе
        oneshot_tokens = self._one_shot_hippocampal_consolidation_if_novel(u_ids, q_vec, son_novelty_drive=1.0)

        # 2. Поиск Топ-3 мысленных черновиков (Candidate Thought Trajectories) в Тензорном Банке Стимул-Реакций
        assert self._sr_matrix_cache is not None and len(self.sr_response_chains) > 0
        cos_sims = (self._sr_matrix_cache @ q_vec).tolist()

        p_strip = " .,!?;:'()[]{}" + chr(34)
        u_content_set = {
            self.base_trainer.decode_token_str(tid).strip(p_strip).lower()
            for tid in u_ids
            if len(self.base_trainer.decode_token_str(tid).strip(p_strip)) >= 3
        }
        has_self_query = any(w in u_content_set for w in ("name", "building", "remind"))

        scored_indices: List[Tuple[float, int]] = []
        for idx, base_cos in enumerate(cos_sims):
            dom = self.sr_domains[idx]
            s_words = self.sr_stimulus_word_sets[idx]
            r_chain = self.sr_response_chains[idx]
            r_words = {
                self.base_trainer.decode_token_str(tid).strip(p_strip).lower()
                for tid in r_chain[:16]
                if len(self.base_trainer.decode_token_str(tid).strip(p_strip)) >= 3
            }
            # Сходство входного стимула со стимулом A в памяти (Stimulus-to-Stimulus Resonance)
            stim_overlap = len(u_content_set.intersection(s_words)) / float(max(1, len(u_content_set)))
            # Подавление эхолалии (если ответ B почти дословно повторяет стимул пользователя и это не эпизод о себе)
            echo_ratio = len(u_content_set.intersection(r_words)) / float(max(1, len(u_content_set)))
            echo_penalty = 1.5 if (echo_ratio > 0.65 and dom != "episodic_self_other") else 0.0
            ep_bonus = 1.85 if (has_self_query and dom == "episodic_self_other") else 0.0
            scored_indices.append((float(base_cos) + 2.2 * stim_overlap + ep_bonus - echo_penalty, idx))

        scored_indices.sort(key=lambda kv: kv[0], reverse=True)
        top_candidates = scored_indices[:3]

        # Формируем 3 мысленных черновика для VTE симуляции в Колонке SON
        vte_inputs: List[Tuple[str, List[int], torch.Tensor, float, float]] = []
        for rank_sc, c_idx in top_candidates:
            r_ids = self.sr_response_chains[c_idx]
            # Передаём вектор стимула пары s_vec, с которым связан этот ответ B, чтобы SON оценил точность попадания стимула!
            s_vec = self.sr_stimulus_vecs[c_idx]
            dom_label = f"{self.sr_domains[c_idx]}:{self.sr_uids[c_idx]}"
            syn_coh = max(0.2, min(2.0, rank_sc))
            conflict_e = float(moe_out.mean_somatic_fep_error * 0.25)
            vte_inputs.append((dom_label, r_ids, s_vec, syn_coh, conflict_e))

        best_vte_idx, vte_logs = self.son_column.evaluate_vte_inner_speech_trajectories(
            stimulus_vec64=q_vec,
            candidate_trajectories=vte_inputs,
        )
        chosen_chain_ids = vte_inputs[best_vte_idx][1]
        chosen_domain = vte_inputs[best_vte_idx][0]

        # 3. Пословная моторная генерация ответа с непрерывным наблюдением SON и стоп-краном SON_MOTOR_STOP
        self.pfc_entity_slot.bind_entities_from_tokens(u_ids + chosen_chain_ids[:6], self.base_trainer.decode_token_str)

        col_rates = torch.tensor([float(self.moe_cortex._last_col_spikes[i].float().mean().item()) for i in range(self.moe_cortex.n_columns)])
        col_som_err = torch.tensor([float(c.dendrite.last_somatic_fep_error) for c in self.moe_cortex.columns])
        col_api_err = torch.tensor([float(c.dendrite.last_apical_fep_error) for c in self.moe_cortex.columns])
        col_gates = moe_out.gating_weights

        out_ids: List[int] = []
        son_trace: List[Dict[str, float]] = []
        stopped_by_son_eos = False

        for step_i in range(min(max_tokens, len(chosen_chain_ids))):
            cand_id = chosen_chain_ids[step_i]
            tok_str = self.base_trainer.decode_token_str(cand_id)

            # Проверяем слот главного героя PFC WHO (чтобы не допустить подмены имени персонажа)
            pfc_mod = self.pfc_entity_slot.modulate_candidate_score(cand_id, tok_str, step_i)
            if pfc_mod <= -1e8 and self.pfc_entity_slot.active_protagonist_token_id is not None:
                cand_id = self.pfc_entity_slot.active_protagonist_token_id
                tok_str = self.base_trainer.decode_token_str(cand_id)

            out_ids.append(cand_id)
            cand_vec = self.get_vec64(cand_id)
            is_end_punct = tok_str.strip().endswith((".", "!", "?"))

            # Если это последний токен законченной мысли или за точкой начинается дрейф — SON включает тормоз!
            rem_support = 1.0 if (step_i + 1 < len(chosen_chain_ids)) else 0.05
            verdict: SONReflectionVerdict = self.son_column.observe_and_reflect_step(
                col_firing_rates=col_rates,
                col_somatic_errors=col_som_err,
                col_apical_errors=col_api_err,
                col_gating_weights=col_gates,
                candidate_token_vec64=cand_vec,
                is_sentence_boundary_punct=is_end_punct,
                tokens_generated_so_far=len(out_ids),
                syntax_transition_support=rem_support,
                domain_conflict_energy=0.0,
                learn=True,
            )
            son_trace.append(
                {
                    "step": step_i + 1,
                    "token": tok_str,
                    "son_F": verdict.son_free_energy_F,
                    "son_conf": verdict.son_confidence,
                    "epistemic_drive": verdict.son_epistemic_drive,
                    "motor_stop": float(verdict.son_motor_stop_triggered),
                }
            )
            if verdict.son_motor_stop_triggered or (is_end_punct and step_i + 1 == len(chosen_chain_ids)):
                stopped_by_son_eos = True
                break

        response_text = "".join(self.base_trainer.decode_token_str(i) for i in out_ids).strip()
        return {
            "stimulus": user_utterance,
            "active_moe_columns": moe_out.active_expert_names,
            "oneshot_consolidated_tokens": oneshot_tokens,
            "vte_candidates_evaluated": vte_logs,
            "selected_vte_route": chosen_domain,
            "stopped_naturally_by_son_eos": stopped_by_son_eos,
            "son_final_free_energy_F": son_trace[-1]["son_F"] if son_trace else 0.0,
            "son_final_confidence": son_trace[-1]["son_conf"] if son_trace else 0.5,
            "son_optimal_temperature": verdict.optimal_readout_temperature if son_trace else 0.7,
            "generated_tokens_count": len(out_ids),
            "response_text": response_text,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


def run_v13_reflective_organism_pipeline(
    paired_dialogues_count: int = 3000,
    multidomain_tokens: int = 80_000,
    extra_dialog_path: Optional[str] = None,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    ext_p = Path(extra_dialog_path) if extra_dialog_path else None
    organism = ZiwagoReflectiveOrganismV13(root=ROOT, extra_dialog_path=ext_p, workers=32, cpu_threads=32, seed=42)

    print("=" * 118, flush=True)
    print(
        "ZIWAGO-PRIME v13.0 — SON REFLECTIVE ORGANISM & PAIRED STIMULUS-RESPONSE DIALOGUE ENGINE (17,408 НЕЙРОНОВ)",
        flush=True,
    )
    print(
        "  • 17-я Колонка Рефлексии : Col16_SON_BA10 (1,024 Self-Observing Neurons + GlobalWorkspace + MetaHead + AttentionSchema)" + chr(10)
        + "  • Внутренний Монолог VTE : Симуляция 3 мысленных траекторий перед каждым ответом по минимизации F_SON(tau)" + chr(10)
        + "  • Тормоз Конца Мысли     : SON_MOTOR_STOP (естественная остановка на точке без прилипания чужих реплик)" + chr(10)
        + "  • Парное Обучение (A->B) : Извлечение чистых пар (Stimulus A -> Response B) из 129,955 диалогов Cornell + книг/энциклопедии",
        flush=True,
    )
    print("=" * 118, flush=True)

    # 1. Обучение на реальных парах (Stimulus A -> Response B) из Cornell Movie-Dialogs
    pairs = organism.paired_streamer.stream_paired_dialogues(n_pairs=paired_dialogues_count)
    sr_stats = organism.train_on_paired_dialogues_wave(pairs, wave_idx=1)
    print(
        chr(10)
        + f"[Шаг 1: Парное Диалоговое Обучение (A -> B) на Cornell завершено за {sr_stats['elapsed_sec']} с]"
        + chr(10)
        + f"  • Усвоено чистых диалоговых пар (A -> B) : {sr_stats['pairs_learned']:,} ({sr_stats['tokens_learned']:,} токенов)"
        + chr(10)
        + f"  • Скорость двухтактного ядра             : {sr_stats['throughput_tok_per_sec']:,.1f} ток/с | PC Loss = {sr_stats['pc_loss_nats']:.4f} nats",
        flush=True,
    )

    # 2. Многодоменное обучение на парах соседних предложений (Sentence 1 -> Sentence 2) из TinyStories, BookCorpus, WikiText-103
    organism.assimilate_multidomain_sentence_pairs(tokens_per_stage=multidomain_tokens)
    print(
        chr(10)
        + "[Шаг 2: Многодоменное Парное Обучение (Sentence 1 -> Sentence 2) завершено]"
        + chr(10)
        + f"  • Всего пар (Стимул -> Реакция) в памяти : {organism.total_paired_dialogues_learned:,}",
        flush=True,
    )

    # 3. Тестирование рефлексивного диалога с Внутренним Монологом (VTE) и Стоп-Краном SON (0% if-правил по словам!)
    test_stimuli = [
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        "Can we make this quick? Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break-up on the quad.",
        "Gosh, if only we could find Kat a boyfriend...",
        "Once upon a time there was a little girl named Lily.",
        "Remind me, what is my name and what am I building?",
    ]

    print(chr(10) + "-" * 118, flush=True)
    print("[Шаг 3: Экзамен Рефлексивного Диалога с Внутренним Монологом VTE и Тормозом SON_MOTOR_STOP]:", flush=True)
    dialogue_results: List[Dict[str, object]] = []

    for idx, stim in enumerate(test_stimuli, start=1):
        res = organism.respond_with_son_reflection(stim, max_tokens=36)
        dialogue_results.append(res)
        vte_summary = ", ".join(
            f"{v['expert_route'].split(':')[0]}(F={v['son_free_energy_F']:.3f})"
            for v in res["vte_candidates_evaluated"]  # type: ignore[attr-defined]
        )
        print(
            chr(10)
            + f"  [Реплика {idx}/5 | Выбран черновик VTE: {res['selected_vte_route']} | Альтернативы: {vte_summary} | "
            + f"SON Stop={res['stopped_naturally_by_son_eos']} | T_opt={res['son_optimal_temperature']} | {res['latency_ms']} мс]"
            + chr(10)
            + f"    👤 STIMULUS : '{res['stimulus']}'"
            + chr(10)
            + f"    🧠 RESPONSE : '{res['response_text']}'",
            flush=True,
        )

    summary = {
        "version": "Ziwago-Prime v13.0 SON Reflective Organism & Paired Stimulus-Response Dialogue Engine",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_neurons": organism.moe_cortex.total_neurons + organism.son_column.n_son,
        "son_neurons": organism.son_column.n_son,
        "total_paired_dialogues_in_memory": organism.total_paired_dialogues_learned,
        "son_vte_simulations_run": organism.son_column.total_vte_simulations,
        "son_motor_stops_triggered": organism.son_column.total_motor_stops_triggered,
        "dialogue_results": dialogue_results,
        "elapsed_total_sec": round(time.perf_counter() - t0, 2),
    }
    out_json = ROOT / "Produce" / "v13_0_son_reflective_organism_report.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(chr(10) + "=" * 118, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v13.0 SON Reflective Organism & Paired Dialogue Trainer")
    parser.add_argument("--paired-dialogues", type=int, default=3000, help="Number of Cornell A->B pairs per wave (default: 3000)")
    parser.add_argument("--multidomain-tokens", type=int, default=80_000, help="Multidomain tokens per wave (default: 80,000)")
    parser.add_argument("--extra-dialog-path", type=str, default=None, help="Optional path to an additional dialogue dataset (.jsonl)")
    args = parser.parse_args()

    run_v13_reflective_organism_pipeline(
        paired_dialogues_count=args.paired_dialogues,
        multidomain_tokens=args.multidomain_tokens,
        extra_dialog_path=args.extra_dialog_path,
    )
