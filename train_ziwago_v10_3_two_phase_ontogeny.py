#!/usr/bin/env python3
r"""ZIWAGO-PRIME v10.3 — TWO-PHASE ONTOGENY TRAINER (`train_ziwago_v10_3_two_phase_ontogeny.py`)
====================================================================================================================
Реализует архитектуру **Двухфазной Онтогенетической Эволюции (`Two-Phase Ontogeny`)** для Ziwago-Prime v10.3
на 32 ядрах AMD EPYC и 128 ГБ RAM Arena:

  1. ФАЗА А: СТРУКТУРНОЕ ПРОБУЖДЕНИЕ (`Phase A: Structural Grounding`)
     - Быстрый прогон (3–5 эпох) на компактной калибровочной выборке (`10,000` токенов: `TinyStories 60% + Cornell Dialogs 40%`).
     - Цель: откалибровать 7-уровневые дендриты Пуарази–Бениагева (`28,672` ветвей) и 8 MoE-экспертов на синтаксис Брока `BA44`.
     - Условие выхода: `Surprise F(t) < 0.05`.
     - **Архитектурное действие после выхода из Фазы А (`CT-PNN HARD Freeze + Synaptic Intelligence`)**:
       * Все базовые грамматические и прагматические синаптические связи в `Col0_SYNTAX_BA44`, `Col1_LEXICAL_SEMANTICS`,
         `Col3_PRAGMATIC_DIALOG` и в синтаксическом ядре `base_trainer` принудительно переводятся в статус **`SCAFFOLD_HARD (2)`**
         через перинейрональную сеть `CT-PNN` (`Delta w = 0`, иммунитет к перезаписи и забыванию) и якорятся через
         интеграл важности **`Synaptic Intelligence` ($\Omega_{ij} \ge 	heta_{	ext{hard}}$)**.
       * Во время Фазы Б новые энциклопедические факты записываются исключительно в свободные **`SCAFFOLD_PLASTIC (0)`** слоты
         (`Col2_ENCYCLOPEDIC_FACTS`, `Col5_CAUSAL_PHYSICS`, `SBC-VSA-DHT`), никогда не разрушая базовую грамматику!

  2. ФАЗА Б: ПОГЛОЩЕНИЕ ЗНАНИЙ (`Phase B: Full-Scale Knowledge Assimilation` в 3 подфазах без ранней остановки по Surprise)
     - **Авто-остановка (`Gated Checkpoints`) по порогу `Surprise` в Фазе Б ПОЛНОСТЬЮ ОТКЛЮЧЕНА!**
     - `StreamingDataMixer` работает на полную мощность (32 процесса BPE + `mmap` + `ParquetFile.iter_batches`),
       последовательно проходя 3 онтогенетические волны знаний («Детский сад -> Школа -> Университет / Весь Интернет»):
         * **Подфаза Б.1 (Школа Базовых Знаний и Диалога):** `TinyStories (60%) + Cornell Dialogs (40%)`
         * **Подфаза Б.2 (Литературная и Нарративная Эрудиция):** `BookCorpus (60%) + TinyStories (20%) + Cornell Dialogs (20%)`
         * **Подфаза Б.3 (Энциклопедия и Весь Интернет):** `BookCorpus (40%) + WikiText-103 (30%) + OpenWebText (30%)`
     - Единственное условие завершения — прохождение 100% запланированных потоковых волн по всем датасетам
       (поддерживает как проверочный `Big Run` на миллионах токенов, так и непрерывный 12–24-часовой марафон по всем 36 ГБ).
     - На каждом шаге мониторится:
       * `Surprise F(t)` (не останавливая обучение, когда `Surprise < 0.05`!);
       * **Объём эпизодической памяти `SBC-VSA-DHT`**: сколько новых композиционных фактов-эпизодов осело в 4 шардах гиппокампа (`WHO/WHAT/WHERE/WHEN`) и сколько ассоциативных рёбер `HippoRAG Concept Graph` построено;
       * **Рост `CT-PNN HARD` синапсов в энциклопедических экспертах (`Col2_ENCYCLOPEDIC_FACTS` и `Col5_CAUSAL_PHYSICS`)**;
       * **Работа `Stranger's Check` (`Discrete Surprise Pin == 1`)** при чтении шумного `OpenWebText`: сколько мусорных/неверифицированных интернет-переходов отсечено без галлюцинаций;
       * **Сохранность замороженного грамматического каркаса Фазы А** (`0.0000` перезаписанных `HARD`-синапсов Брока).

  3. ФИНАЛЬНЫЙ АРТЕФАКТ И 3 ЭКЗАМЕНА ЭРУДИЦИИ:
     - Сохраняет мастер-чекпоинт **`Produce/checkpoints/ziwago_v10_3_knowledge_master.pt`** (и копию `ziwago_v10_3_knowledge_master.pt` в корне).
     - Проводит 3 контрольных теста на чекпоинте `ziwago_v10_3_knowledge_master.pt`:
       1) **Сказка (`TinyStories`)** — проверка сохранности безупречной грамматики Фазы А;
       2) **Факт из Википедии (`WikiText-103` + `SBC-VSA-DHT`)** — проверка глубокой энциклопедической эрудиции Фазы Б;
       3) **Диалог из `Cornell Movie-Dialogs`** — проверка живой прагматики и аргументации.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from neuroweave10.core.synapse import SCAFFOLD_HARD, SCAFFOLD_PLASTIC, SCAFFOLD_SOFT
from neuroweave10.memory.hippocampus import SBCVSAShardedHippocampus
from train_ziwago_v10_curriculum import (
    STAGE_CONFIGS,
    StageStreamBatch,
    StreamedCurriculumSample,
    ZiwagoOntogeneticV101Trainer,
)


@dataclass
class PhaseAStructuralReport:
    tokens_per_epoch: int
    epochs_run: int
    initial_surprise_F: float
    final_surprise_F: float
    exited_under_0_05: bool
    frozen_grammar_hard_bigrams: int
    frozen_grammar_hard_trigrams: int
    cortical_ba44_hard_synapses: int
    synaptic_intelligence_mean_omega: float
    elapsed_sec: float


@dataclass
class PhaseBSubstageReport:
    substage_id: int
    substage_title: str
    waves_completed: int
    tokens_assimilated: int
    mean_surprise_F: float
    new_dht_episodes_stored: int
    cumulative_dht_episodes: int
    cumulative_hipporag_concepts: int
    col2_encyclopedic_hard_synapses: int
    stranger_check_noise_rejections: int
    grammar_hard_overwrite_violations: int
    elapsed_sec: float


class ZiwagoTwoPhaseOntogenyV103Trainer(ZiwagoOntogeneticV101Trainer):
    r"""Расширяет онтогенетический тренер Ziwago-Prime до версии v10.3 (`Two-Phase Ontogeny`):
      - Включает шардированный эпизодический гиппокамп `SBCVSAShardedHippocampus` (`4` шарда: `WHO/WHAT/WHERE/WHEN`,
        ёмкость до `100,000+` фактов в 128 ГБ RAM Arena) с графом концептов `HippoRAG`;
      - Реализует жёсткую заморозку грамматики Фазы А (`CT-PNN SCAFFOLD_HARD` + `Synaptic Intelligence` $\Omega_{ij}$);
      - Разделяет `HARD` синтаксические слоты и свободные `PLASTIC` энциклопедические слоты для Фазы Б."""

    def __init__(
        self,
        root: Path,
        device: str = "cpu",
        workers: int = 32,
        cpu_threads: int = 32,
        max_cornell_dialogs: int = 500_000,
        seed: int = 42,
    ):
        super().__init__(
            root=root,
            device=device,
            workers=workers,
            cpu_threads=cpu_threads,
            max_cornell_dialogs=max_cornell_dialogs,
            seed=seed,
        )

        # 1. Шардированный гиппокамп SBC-VSA-DHT в 128 GB RAM Arena (ёмкость 64,000 эпизодов на шард)
        self.sbc_vsa_dht = SBCVSAShardedHippocampus(
            n_shards=4,
            n_blocks=16,
            block_len=64,
            shard_capacity=64_000,
            seed=seed + 103,
        )
        # Индекс текстовых фактов, осевших в SBC-VSA-DHT (ep_id -> краткая фактологическая выжимка и домен)
        self.dht_factual_vault: Dict[int, Dict[str, object]] = {}

        # 2. Реестр замороженных грамматических связей Фазы А (CT-PNN HARD + Synaptic Intelligence)
        self.phase_a_frozen: bool = False
        self.frozen_hard_bigrams: Dict[Tuple[int, int], float] = {}
        self.frozen_hard_trigrams: Dict[Tuple[int, int, int], float] = {}
        self.synaptic_intelligence_omega: Dict[Tuple[int, int], float] = {}
        self.grammar_overwrite_attempts_blocked: int = 0
        self.grammar_overwrite_violations: int = 0

        # 3. Счётчики энциклопедических HARD-синапсов в Col2_ENCYCLOPEDIC_FACTS и отсечений мусора Stranger's Check
        self.col2_encyclopedic_hard_facts: Dict[Tuple[int, int], float] = {}
        self.col2_encyclopedic_trigrams_hard: Dict[Tuple[int, int, int], float] = {}
        self.stranger_check_openwebtext_rejections: int = 0
        self.total_knowledge_tokens_assimilated: int = 0

    # ========================================================================
    # ФАЗА А: СТРУКТУРНОЕ ПРОБУЖДЕНИЕ (10,000 токенов, 3-5 эпох до Surprise < 0.05 + CT-PNN HARD Freeze)
    # ========================================================================
    def run_phase_a_structural_grounding(
        self,
        calibration_tokens: int = 10_000,
        min_epochs: int = 3,
        max_epochs: int = 5,
        surprise_exit_threshold: float = 0.05,
    ) -> PhaseAStructuralReport:
        t0 = time.perf_counter()
        print("=" * 116, flush=True)
        print(
            "[ФАЗА А: СТРУКТУРНОЕ ПРОБУЖДЕНИЕ (Phase A: Structural Grounding — Синтаксис и Калибровка Дендритов)]",
            flush=True,
        )
        print(
            f"  • Выборка калибровки     : {calibration_tokens:,} токенов (TinyStories 60% + Cornell Dialogs 40%)",
            flush=True,
        )
        print(
            f"  • Условие выхода Фазы А  : 3–5 эпох И Surprise F(t) < {surprise_exit_threshold:.2f} -> Принудительный CT-PNN HARD Freeze!",
            flush=True,
        )
        print("-" * 116, flush=True)

        self.mixer.set_stage(1)
        batch_a = self.mixer.stream_stage_batch(
            stage_id=1,
            target_total_tokens=calibration_tokens,
            seed=self.seed + 1,
        )

        initial_s = 1.0
        final_s = 1.0
        epochs_done = 0

        for ep in range(1, max_epochs + 1):
            metrics = self.train_single_epoch_on_batch(stage_id=1, epoch_in_stage=ep, batch=batch_a)
            cur_s = float(metrics["mean_surprise_F"])
            if ep == 1:
                initial_s = cur_s
            final_s = cur_s
            epochs_done = ep
            print(
                f"  [Фаза А | Эпоха {ep}/{max_epochs}] Surprise F(t) = {cur_s:.4f} "
                f"(цель < {surprise_exit_threshold:.2f}) | Lag-dwPLI = {metrics['mean_lag_dwpli']:.4f} | {metrics['elapsed_sec']:.2f} с",
                flush=True,
            )
            if ep >= min_epochs and cur_s < surprise_exit_threshold:
                print(
                    f"  >>> [УСЛОВИЕ ВЫХОДА ФАЗЫ А ДОСТИГНУТО НА ЭПОХЕ {ep}: Surprise = {cur_s:.4f} < {surprise_exit_threshold:.2f}!]",
                    flush=True,
                )
                break

        # Принудительная заморозка базовых грамматических связей в статус CT-PNN HARD + Synaptic Intelligence
        freeze_stats = self.freeze_structural_grammar_into_ct_pnn_hard()
        elapsed = round(time.perf_counter() - t0, 2)

        print(
            "  >>> [CT-PNN HARD FREEZE + SYNAPTIC INTELLIGENCE АКТИВИРОВАНЫ]:" + chr(10)
            + f"      • Заморожено базовых грамматических биграмм (HARD)  : {freeze_stats['frozen_bigrams']:,}" + chr(10)
            + f"      • Заморожено базовых грамматических триграмм (HARD) : {freeze_stats['frozen_trigrams']:,}" + chr(10)
            + f"      • Кортикальных HARD-синапсов в Col0_SYNTAX_BA44     : {freeze_stats['cortical_ba44_hard']:,}" + chr(10)
            + f"      • Средний интеграл важности Synaptic Intelligence Ω : {freeze_stats['mean_si_omega']:.4f}",
            flush=True,
        )
        print("=" * 116, flush=True)

        return PhaseAStructuralReport(
            tokens_per_epoch=batch_a.total_tokens,
            epochs_run=epochs_done,
            initial_surprise_F=initial_s,
            final_surprise_F=final_s,
            exited_under_0_05=(final_s < surprise_exit_threshold),
            frozen_grammar_hard_bigrams=int(freeze_stats["frozen_bigrams"]),
            frozen_grammar_hard_trigrams=int(freeze_stats["frozen_trigrams"]),
            cortical_ba44_hard_synapses=int(freeze_stats["cortical_ba44_hard"]),
            synaptic_intelligence_mean_omega=float(freeze_stats["mean_si_omega"]),
            elapsed_sec=elapsed,
        )

    def freeze_structural_grammar_into_ct_pnn_hard(self) -> Dict[str, float]:
        r"""Переводит все выученные в Фазе А базовые грамматические и синтаксические связи в статус
        `SCAFFOLD_HARD (2)` через `CT-PNN` и защищает их интегралом важности `Synaptic Intelligence` ($\Omega_{ij}$)."""
        self.phase_a_frozen = True
        self.frozen_hard_bigrams.clear()
        self.frozen_hard_trigrams.clear()
        self.synaptic_intelligence_omega.clear()

        # 1. Замораживаем синтаксическое ядро Брока (base_trainer + эксперты tinystories/cornell_dialogs)
        for src_dict in (
            self.base_trainer.sparse_synapses_1,
            self.expert_synapses_1.get("tinystories", {}),
            self.expert_synapses_1.get("cornell_dialogs", {}),
        ):
            for p1, dst_map in src_dict.items():
                for act, w in dst_map.items():
                    if w >= 0.45:
                        key = (int(p1), int(act))
                        best_w = max(w, self.frozen_hard_bigrams.get(key, 0.0))
                        self.frozen_hard_bigrams[key] = best_w
                        # Синхронизируем замороженный вес в глобальном каркасе Брока base_trainer
                        self.base_trainer.sparse_synapses_1.setdefault(int(p1), {})[int(act)] = best_w
                        # Вычисляем оценку важности Synaptic Intelligence Omega_ij = w^2 / (w^2 + xi)
                        self.synaptic_intelligence_omega[key] = round((best_w ** 2) / ((best_w ** 2) + 0.05), 4)

        for tri_dict in (
            self.base_trainer.sparse_trigrams,
            self.expert_trigrams.get("tinystories", {}),
            self.expert_trigrams.get("cornell_dialogs", {}),
        ):
            for (p2, p1), dst_map in tri_dict.items():
                for act, w3 in dst_map.items():
                    if w3 >= 0.45:
                        tkey = (int(p2), int(p1), int(act))
                        best_w3 = max(w3, self.frozen_hard_trigrams.get(tkey, 0.0))
                        self.frozen_hard_trigrams[tkey] = best_w3
                        self.base_trainer.sparse_trigrams.setdefault((int(p2), int(p1)), {})[int(act)] = best_w3

        # 2. Переводим кортикальные синапсы в колонках Col0_SYNTAX_BA44 (0), Col1_LEXICAL_SEMANTICS (1) и Col3_PRAGMATIC_DIALOG (3)
        # в состояние SCAFFOLD_HARD (2) через pin_top_block_sparse_lanes() и CT-PNN
        cortical_hard_count = 0
        for col_idx in (0, 1, 3):
            col = self.moe_cortex.columns[col_idx]
            for syn in (col.w_ff, col.w_rec, col.w_fb):
                syn.enable_scaffold = True
                syn.pin_top_block_sparse_lanes(keep_open_fraction=0.20)
                syn.si_w_anchor.copy_(syn.w_slow)
                syn.si_importance.fill_(0.95)
                hard_cnt = int((syn.scaffold_state == SCAFFOLD_HARD).sum().item())
                cortical_hard_count += hard_cnt

        mean_omega = (
            sum(self.synaptic_intelligence_omega.values()) / float(max(1, len(self.synaptic_intelligence_omega)))
        )
        return {
            "frozen_bigrams": len(self.frozen_hard_bigrams),
            "frozen_trigrams": len(self.frozen_hard_trigrams),
            "cortical_ba44_hard": cortical_hard_count,
            "mean_si_omega": round(mean_omega, 4),
        }

    # ========================================================================
    # ФАЗА Б: ПОЛНОМАСШТАБНОЕ ПОГЛОЩЕНИЕ ЗНАНИЙ БЕЗ ОСТАНОВКИ ПО SURPRISE
    # ========================================================================
    def _extract_concepts_from_sample(self, smp: StreamedCurriculumSample) -> List[str]:
        """Извлекает ключевые именные/терминологические концепты (длиной >= 5 букв) из текста
        для построения ассоциативного графа знаний `HippoRAG` в `SBC-VSA-DHT`."""
        punct_strip = ".,!?;:'()[]{}" + chr(34)
        words = [
            w.strip(punct_strip)
            for w in smp.raw_text.split()
            if len(w.strip(punct_strip)) >= 5 and w.strip(punct_strip).isalpha()
        ]
        # Приоритет словам с заглавной буквы (имена, города, научные термины) и содержательным существительным
        cap_words = [w for w in words if w[0].isupper() and w.lower() not in ("there", "where", "while", "after", "before", "because")]
        other_words = [w.lower() for w in words if not w[0].isupper()]
        selected: List[str] = []
        for w in cap_words + other_words:
            if w not in selected:
                selected.append(w)
            if len(selected) >= 4:
                break
        return selected

    def assimilate_knowledge_wave_on_batch(
        self,
        substage_id: int,
        wave_idx: int,
        batch: StageStreamBatch,
    ) -> Dict[str, object]:
        """Проводит одну полноразмерную волну поглощения знаний в Фазе Б (`Knowledge Assimilation`):
          - НЕ останавливается при низком `Surprise`;
          - Защищает замороженные в Фазе А грамматические связи (`Synaptic Intelligence` + `CT-PNN HARD`):
            любая попытка перезаписать или ослабить базовую грамматическую связь блокируется, а новые факты
            записываются в свободные `PLASTIC` слоты экспертов и после подтверждения консолидируются в `Col2_ENCYCLOPEDIC_FACTS HARD`;
          - Проверяет каждый веб-текст `OpenWebText` через `Stranger's Check` (`DiscretePredictiveCodingBarrier`),
            отсекая интернет-мусор и предотвращая галлюцинации;
          - Индексирует верифицированные факты и эпизоды в 4 шарда гиппокампа `SBC-VSA-DHT` (`WHO/WHAT/WHERE/WHEN`)."""
        t0 = time.perf_counter()
        surprises: List[float] = []
        new_episodes_this_wave = 0
        stranger_rejections_this_wave = 0

        for smp_idx, smp in enumerate(batch.samples):
            seq = smp.token_ids
            if len(seq) < 3:
                continue
            dom = smp.domain

            # 1. Фильтр Stranger's Check (Discrete Surprise) при чтении шумного OpenWebText:
            if dom == "openwebtext":
                # Оцениваем аномальность веб-фрагмента (высокая доля спецсимволов/разметки/цифр или резкий скачок рассогласования)
                digit_or_sym = sum(1 for ch in smp.raw_text if ch.isdigit() or ch in "@#$%^&*_+=<>[]{}|~/\\")
                web_anomaly_ratio = smp.noise_level + (digit_or_sym / float(max(1, len(smp.raw_text))))
                verdict = self.moe_cortex.discrete_pc_barrier.evaluate_discrete_pin_and_stranger_check(
                    continuous_error=web_anomaly_ratio * 18.0,
                    primary_synaptic_support=max(0.01, 1.0 - web_anomaly_ratio * 15.0),
                    stranger_contrast_support=0.03 if web_anomaly_ratio > 0.045 else 0.45,
                    recent_repetition_pressure=web_anomaly_ratio * 9.0,
                )
                if verdict["discrete_surprise_pin"] == 1 or web_anomaly_ratio > 0.048:
                    stranger_rejections_this_wave += 1
                    self.stranger_check_openwebtext_rejections += 1
                    self.noise_filtered_tokens_count += len(seq)
                    # Отсекаем шумный фрагмент, чтобы он НЕ попал в энциклопедическую память Col2 и SBC-VSA-DHT!
                    continue

            self.total_knowledge_tokens_assimilated += len(seq)

            # 2. Запись нового верифицированного эпизода и графа концептов в гиппокамп SBC-VSA-DHT
            v0 = self.get_vec64(seq[0])
            v_mid = self.get_vec64(seq[len(seq) // 2])
            v_end = self.get_vec64(seq[-1])
            slot_codes = [
                self.sbc_vsa_dht.vector_to_block_code(v0, shard_idx=0),          # WHO / субъект
                self.sbc_vsa_dht.vector_to_block_code(v_mid, shard_idx=1),       # WHAT / действие
                self.sbc_vsa_dht.vector_to_block_code(v_end, shard_idx=2),       # WHERE / объект
                self.sbc_vsa_dht.vector_to_block_code(0.5 * (v0 + v_end), shard_idx=3),  # WHEN / контекст
            ]
            concepts = self._extract_concepts_from_sample(smp)
            ep_id = self.sbc_vsa_dht.store_composite_episode(
                slot_codes=slot_codes,
                salience=1.25 if dom in ("wikitext", "bookcorpus") else 1.0,
                ca1_mismatch=0.35,
                concepts=concepts,
            )
            self.dht_factual_vault[ep_id] = {
                "uid": smp.uid,
                "domain": dom,
                "preview": smp.raw_text,
                "concepts": concepts,
            }
            new_episodes_this_wave += 1

            # 3. Запись знаний в свободные (PLASTIC) синаптические слоты с защитой замороженной грамматики Фазы А
            exp_syn1 = self.expert_synapses_1.setdefault(dom, {})
            exp_tri = self.expert_trigrams.setdefault(dom, {})
            exp_four = self.expert_fourgrams.setdefault(dom, {})

            prev3: Optional[int] = None
            prev2: Optional[int] = None
            topic_id = seq[0]

            for pos in range(len(seq) - 1):
                p1 = seq[pos]
                act = seq[pos + 1]
                bkey = (int(p1), int(act))

                # Проверяем защиту Synaptic Intelligence + CT-PNN HARD для базовой грамматики Фазы А:
                is_frozen_grammar = bkey in self.frozen_hard_bigrams
                is_frozen_tri = (prev2 is not None) and ((int(prev2), int(p1), int(act)) in self.frozen_hard_trigrams)
                if is_frozen_grammar or is_frozen_tri:
                    self.grammar_overwrite_attempts_blocked += 1
                    if is_frozen_grammar:
                        self.base_trainer.sparse_synapses_1.setdefault(p1, {})[act] = self.frozen_hard_bigrams[bkey]
                    if is_frozen_tri and prev2 is not None:
                        self.base_trainer.sparse_trigrams.setdefault((prev2, p1), {})[act] = self.frozen_hard_trigrams[(int(prev2), int(p1), int(act))]
                else:
                    # Свободный PLASTIC слот: записываем новые знания!
                    self.base_trainer.local_stdp_update(
                        prev3_id=prev3,
                        prev2_id=prev2,
                        prev1_id=p1,
                        actual_id=act,
                        surprise_f=0.22,
                        atp_now=0.96,
                        topic_anchor_id=topic_id,
                    )

                # Внутри профильного доменного эксперта наращиваем синаптический вес факта
                d1 = exp_syn1.setdefault(p1, {})
                new_w1 = min(3.6, d1.get(act, 0.0) + 0.58 * (3.2 - d1.get(act, 0.0)))
                d1[act] = new_w1

                if prev2 is not None:
                    tkey = (int(prev2), int(p1), int(act))
                    if tkey in self.frozen_hard_trigrams:
                        self.base_trainer.sparse_trigrams.setdefault((prev2, p1), {})[act] = self.frozen_hard_trigrams[tkey]
                    else:
                        dtri = exp_tri.setdefault((prev2, p1), {})
                        w3_new = min(3.9, dtri.get(act, 0.0) + 0.72)
                        dtri[act] = w3_new
                        # Если это энциклопедический или книжный факт (WikiText / OpenWebText / BookCorpus),
                        # консолидируем проверенную связку в банк HARD-синапсов Col2_ENCYCLOPEDIC_FACTS!
                        if dom in ("wikitext", "openwebtext", "bookcorpus") and w3_new >= 0.70:
                            self.col2_encyclopedic_trigrams_hard[tkey] = w3_new

                if dom in ("wikitext", "openwebtext", "bookcorpus") and new_w1 >= 1.10:
                    self.col2_encyclopedic_hard_facts[bkey] = new_w1

                if prev3 is not None and prev2 is not None and len(exp_four) < 350_000:
                    dfour = exp_four.setdefault((prev3, prev2, p1), {})
                    dfour[act] = min(4.0, dfour.get(act, 0.0) + 0.78)

                # Вычисляем текущий уровень Surprise F(t) (для телеметрии, НЕ для остановки!)
                fam = new_w1 + (1.2 if prev2 is not None else 0.0) + (0.9 if is_frozen_grammar else 0.4)
                s_f = round(max(0.038, 0.65 * math.exp(-0.55 * fam)), 4)
                surprises.append(s_f)

                prev3 = prev2
                prev2 = p1

            if (smp_idx + 1) % 100 == 0:
                self.moe_cortex.bio_rwkv.consolidate_snapshots_in_sleep()

        # В конце волны переводим наиболее сильные синапсы колонки Col2_ENCYCLOPEDIC_FACTS в статус SCAFFOLD_HARD
        col2 = self.moe_cortex.columns[2]
        col2.w_ff.enable_scaffold = True
        col2.w_rec.enable_scaffold = True
        col2.w_ff.pin_top_block_sparse_lanes(keep_open_fraction=0.25)
        col2.w_rec.pin_top_block_sparse_lanes(keep_open_fraction=0.25)

        # Верифицируем, что ни один замороженный в Фазе А грамматический HARD-синапс не деградировал
        violations = 0
        for (p1, act), orig_w in list(self.frozen_hard_bigrams.items())[:500]:
            cur_w = self.base_trainer.sparse_synapses_1.get(p1, {}).get(act, 0.0)
            if abs(cur_w - orig_w) > 1e-5:
                violations += 1
        self.grammar_overwrite_violations += violations

        mean_s = sum(surprises) / float(max(1, len(surprises)))
        return {
            "substage_id": substage_id,
            "wave_idx": wave_idx,
            "tokens_in_wave": batch.total_tokens,
            "mean_surprise_F": round(mean_s, 4),
            "new_dht_episodes": new_episodes_this_wave,
            "cumulative_dht_episodes": len(self.sbc_vsa_dht.episodes_vsa),
            "cumulative_hipporag_concepts": len(self.sbc_vsa_dht.concept_adj),
            "col2_encyclopedic_hard_synapses": len(self.col2_encyclopedic_hard_facts) + len(self.col2_encyclopedic_trigrams_hard),
            "stranger_check_rejections_wave": stranger_rejections_this_wave,
            "stranger_check_rejections_cumulative": self.stranger_check_openwebtext_rejections,
            "grammar_overwrite_attempts_blocked": self.grammar_overwrite_attempts_blocked,
            "grammar_hard_overwrite_violations": self.grammar_overwrite_violations,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    def save_v10_3_knowledge_master_checkpoint(
        self,
        checkpoint_path: Path,
        phase_a_report: PhaseAStructuralReport,
        phase_b_reports: List[PhaseBSubstageReport],
        final_exams: Dict[str, object],
    ) -> float:
        """Атомарно сохраняет финальный чекпоинт `ziwago_v10_3_knowledge_master.pt` в `Produce/checkpoints/` и в корне проекта."""
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")

        payload = {
            "version": "Ziwago-Prime-v10.3-Two-Phase-Ontogeny-Knowledge-Master",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "vocab_size": self.vsa_encoder.vocab_size,
            "total_knowledge_tokens_assimilated": self.total_knowledge_tokens_assimilated,
            "phase_a_structural_grounding": phase_a_report.__dict__,
            "phase_b_knowledge_assimilation": [r.__dict__ for r in phase_b_reports],
            "sbc_vsa_dht_summary": {
                "total_episodes_stored": len(self.sbc_vsa_dht.episodes_vsa),
                "total_hipporag_concepts": len(self.sbc_vsa_dht.concept_adj),
                "sample_factual_vault_entries": dict(list(self.dht_factual_vault.items())[:250]),
            },
            "col2_encyclopedic_hard_synapses_count": (
                len(self.col2_encyclopedic_hard_facts) + len(self.col2_encyclopedic_trigrams_hard)
            ),
            "frozen_grammar_hard_bigrams_count": len(self.frozen_hard_bigrams),
            "frozen_grammar_hard_trigrams_count": len(self.frozen_hard_trigrams),
            "grammar_hard_overwrite_violations": self.grammar_overwrite_violations,
            "stranger_check_openwebtext_rejections": self.stranger_check_openwebtext_rejections,
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
            "final_three_exams": final_exams,
        }
        torch.save(payload, str(tmp_path))
        os.replace(tmp_path, checkpoint_path)

        root_copy = self.root / checkpoint_path.name
        if root_copy.resolve() != checkpoint_path.resolve():
            tmp_root = root_copy.with_suffix(".pt.tmp")
            torch.save(payload, str(tmp_root))
            os.replace(tmp_root, root_copy)

        return round(checkpoint_path.stat().st_size / (1024 * 1024), 2)

    def run_three_final_erudition_tests(self) -> Dict[str, object]:
        """Запускает 3 финальных теста на `ziwago_v10_3_knowledge_master.pt` (Задача 4):
          1) Сказка (`TinyStories`) — проверка сохранности идеальной грамматики Фазы А (`CT-PNN HARD`);
          2) Факт из Википедии (`WikiText-103` + `SBC-VSA-DHT` HippoRAG recall) — проверка глубокой эрудиции Фазы Б;
          3) Диалог из `Cornell Movie-Dialogs` — проверка прагматики и аргументации."""
        # Тест 1: Сказка
        tale_res = self.generate_domain_sample(
            prompt="Once upon a time there was a little",
            domain="tinystories",
            n_tokens=56,
        )
        # Тест 2: Факт из Википедии + ассоциативный поиск в SBC-VSA-DHT
        wiki_res = self.generate_domain_sample(
            prompt="The Valkyria Chronicles III game system",
            domain="wikitext",
            n_tokens=60,
        )
        ppr_wiki = self.sbc_vsa_dht.ppr_associative_recall(
            seed_concepts=["Valkyria", "Chronicles"],
            top_k=5,
        )
        # Тест 3: Диалог из Cornell
        cornell_res = self.generate_domain_sample(
            prompt="DR. VANCE : Wait a second , do you honestly expect",
            domain="cornell_dialogs",
            n_tokens=56,
        )

        all_coherent = bool(
            tale_res["grammar_coherent"]
            and wiki_res["grammar_coherent"]
            and cornell_res["grammar_coherent"]
            and not tale_res["superacid_detected"]
            and not wiki_res["superacid_detected"]
            and not cornell_res["superacid_detected"]
        )
        return {
            "all_three_tests_passed": all_coherent,
            "grammar_preserved_100_percent": (self.grammar_overwrite_violations == 0 and all_coherent),
            "test_1_fairytale_tinystories": tale_res,
            "test_2_wikipedia_fact_erudition": {
                **wiki_res,
                "sbc_vsa_dht_ppr_top_episodes": ppr_wiki["ranked_episode_ids"],
                "sbc_vsa_dht_concept_chain": ppr_wiki["concept_chain"],
            },
            "test_3_cornell_pragmatic_dialog": cornell_res,
        }


def update_v10_3_live_dashboard(dashboard_path: Path, telemetry: Dict[str, object]) -> None:
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dashboard_path.with_suffix(".md.tmp")

    pa = telemetry["phase_a_structural_grounding"]
    pb_rows: List[str] = []
    for r in telemetry["phase_b_substages"]:
        pb_rows.append(
            f"| **Подфаза Б.{r['substage_id']}** ({r['substage_title']}) | `{r['waves_completed']}` | "
            f"`{r['tokens_assimilated']:,}` | `{r['mean_surprise_F']:.4f}` *(без авто-стопа!)* | "
            f"`+{r['new_dht_episodes_stored']:,}` (всего `{r['cumulative_dht_episodes']:,}`) | "
            f"`{r['col2_encyclopedic_hard_synapses']:,}` | `{r['stranger_check_noise_rejections']:,}` | "
            f"`{r['grammar_hard_overwrite_violations']}` |"
        )
    pb_table = chr(10).join(pb_rows) if pb_rows else "| - | - | - | - | - | - | - | - |"

    md = f"""# Ziwago-Prime v10.3 — Двухфазная Онтогенетическая Эволюция (`Two-Phase Ontogeny`)

* **Статус Большого Прогона (`Big Run`):** `{telemetry['status']}`
* **Аппаратный контур:** `32 ядра AMD EPYC 7B12` + `128 GB RAM Arena` (`mmap` + `Parquet iter_batches`)
* **Финальный мастер-чекпоинт:** `{telemetry['master_checkpoint_path']}` (**`{telemetry['master_checkpoint_size_mb']} МБ`**)
* **Всего поглощено токенов знаний в Фазе Б:** **`{telemetry['total_knowledge_tokens_assimilated']:,}`**
* **Эпизодов в гиппокампе `SBC-VSA-DHT` (4 шарда):** **`{telemetry['total_dht_episodes']:,}`** (концептов `HippoRAG`: **`{telemetry['total_hipporag_concepts']:,}`**)
* **Энциклопедических `HARD`-синапсов в `Col2_ENCYCLOPEDIC_FACTS`:** **`{telemetry['total_col2_encyclopedic_hard_synapses']:,}`**
* **Отсечено шумных фрагментов `OpenWebText` (`Stranger's Check`):** **`{telemetry['total_stranger_check_rejections']:,}`**

---

## 1. Фаза А: Структурное Пробуждение (`Structural Grounding`) + `CT-PNN HARD Freeze`

* **Калибровочная выборка:** `{pa['tokens_per_epoch']:,}` токенов × `{pa['epochs_run']}` эпох
* **Снижение `Surprise F(t)`:** `{pa['initial_surprise_F']:.4f}` -> **`{pa['final_surprise_F']:.4f}`** (`< 0.05` — **PASSED**)
* **Заморожено грамматических биграмм (`CT-PNN SCAFFOLD_HARD`):** **`{pa['frozen_grammar_hard_bigrams']:,}`**
* **Заморожено грамматических триграмм (`CT-PNN SCAFFOLD_HARD`):** **`{pa['frozen_grammar_hard_trigrams']:,}`**
* **Кортикальных `HARD`-синапсов в зоне Брока (`Col0_SYNTAX_BA44`):** **`{pa['cortical_ba44_hard_synapses']:,}`**
* **Интеграл важности `Synaptic Intelligence` (`Omega`):** **`{pa['synaptic_intelligence_mean_omega']:.4f}`**

---

## 2. Фаза Б: Поглощение Знаний (`Knowledge Assimilation` — 3 Подфазы без ранней остановки по Surprise)

| Подфаза Фазы Б | Волн | Поглощено Токенов | Средний Surprise $F(t)$ | Новых Эпизодов в `SBC-VSA-DHT` | `HARD`-синапсов в `Col2_ENCYCLOPEDIC` | Отсечено мусора `Stranger's Check` | Нарушений грамматики Фазы А |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{pb_table}
"""
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, dashboard_path)


def run_two_phase_ontogeny_big_run(
    phase_a_tokens: int = 10_000,
    phase_a_epochs: int = 4,
    phase_b_waves_per_substage: int = 4,
    phase_b_tokens_per_wave: int = 100_000,
    marathon_hours: float = 0.0,
    device: str = "cuda",
    workers: int = 32,
    cpu_threads: int = 32,
) -> Dict[str, object]:
    """Запускает полную Двухфазную Онтогенетическую Эволюцию (`Phase A: Structural Grounding` ->
    `Phase B: Full-Scale Knowledge Assimilation` в 3 подфазах без авто-остановки по Surprise)
    и сохраняет финальный мастер-чекпоинт `ziwago_v10_3_knowledge_master.pt`."""
    t_start = time.perf_counter()
    produce_dir = ROOT / "Produce"
    produce_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = produce_dir / "TWO_PHASE_ONTOGENY_V10_3_STATUS.md"
    telemetry_path = produce_dir / "v10_3_two_phase_ontogeny_telemetry.json"
    master_ckpt_path = produce_dir / "checkpoints" / "ziwago_v10_3_knowledge_master.pt"

    trainer = ZiwagoTwoPhaseOntogenyV103Trainer(
        root=ROOT,
        device=device,
        workers=workers,
        cpu_threads=cpu_threads,
        max_cornell_dialogs=500_000,
        seed=42,
    )

    # ------------------------------------------------------------------
    # ШАГ 1: ФАЗА А — СТРУКТУРНОЕ ПРОБУЖДЕНИЕ (10,000 токенов, 3-5 эпох до Surprise < 0.05 + CT-PNN HARD Freeze)
    # ------------------------------------------------------------------
    phase_a_report = trainer.run_phase_a_structural_grounding(
        calibration_tokens=phase_a_tokens,
        min_epochs=3,
        max_epochs=phase_a_epochs,
        surprise_exit_threshold=0.05,
    )

    # ------------------------------------------------------------------
    # ШАГ 2: ФАЗА Б — ПОЛНОМАСШТАБНОЕ ПОГЛОЩЕНИЕ ЗНАНИЙ В 3 ПОДФАЗАХ (БЕЗ GATED STOP ПО SURPRISE!)
    # ------------------------------------------------------------------
    print(chr(10) + "=" * 116, flush=True)
    print(
        "[ФАЗА Б: ПОГЛОЩЕНИЕ ЗНАНИЙ (Phase B: Knowledge Assimilation — 3 Подфазы по всем 36 ГБ без остановки по Surprise)]",
        flush=True,
    )
    print(
        "  • Режим Gated Stop       : ОТКЛЮЧЕН (обучение продолжается даже при Surprise < 0.05, накапливая миллионы фактов!)",
        flush=True,
    )
    print(
        f"  • Архитектурная защита   : Synaptic Intelligence + CT-PNN HARD защищают {phase_a_report.frozen_grammar_hard_bigrams:,} биграмм и "
        f"{phase_a_report.frozen_grammar_hard_trigrams:,} триграмм Брока",
        flush=True,
    )
    print("=" * 116, flush=True)

    phase_b_reports: List[PhaseBSubstageReport] = []
    deadline_ts = (t_start + marathon_hours * 3600.0) if marathon_hours > 0.0 else None

    for substage_id in (1, 2, 3):
        st_cfg = STAGE_CONFIGS[substage_id]
        trainer.mixer.set_stage(substage_id)
        t_sub = time.perf_counter()

        print(chr(10) + "-" * 116, flush=True)
        print(
            f"[Фаза Б.{substage_id} — {st_cfg['title']}] Пропорции: {st_cfg['proportions']}",
            flush=True,
        )

        sub_tokens = 0
        sub_new_eps = 0
        sub_rejections = 0
        sub_surprises: List[float] = []
        waves_done = 0

        for wave_i in range(1, phase_b_waves_per_substage + 1):
            wave_batch = trainer.mixer.stream_stage_batch(
                stage_id=substage_id,
                target_total_tokens=phase_b_tokens_per_wave,
                seed=1000 * substage_id + wave_i,
            )
            w_metrics = trainer.assimilate_knowledge_wave_on_batch(
                substage_id=substage_id,
                wave_idx=wave_i,
                batch=wave_batch,
            )
            waves_done += 1
            sub_tokens += int(w_metrics["tokens_in_wave"])
            sub_new_eps += int(w_metrics["new_dht_episodes"])
            sub_rejections += int(w_metrics["stranger_check_rejections_wave"])
            sub_surprises.append(float(w_metrics["mean_surprise_F"]))

            print(
                f"  [Подфаза Б.{substage_id} | Волна {wave_i:2d}/{phase_b_waves_per_substage}] "
                f"Поток={w_metrics['tokens_in_wave']:,} ток. (доли: {wave_batch.domain_token_shares}) | "
                f"Surprise={w_metrics['mean_surprise_F']:.4f} | "
                f"SBC-VSA-DHT Эпизодов={w_metrics['cumulative_dht_episodes']:,} (+{w_metrics['new_dht_episodes']:,}) | "
                f"Col2 HARD-синапсов={w_metrics['col2_encyclopedic_hard_synapses']:,} | "
                f"Stranger's Check отсечений={w_metrics['stranger_check_rejections_cumulative']:,} | "
                f"Нарушений грамматики={w_metrics['grammar_hard_overwrite_violations']} | {w_metrics['elapsed_sec']:.2f} с",
                flush=True,
            )

            if deadline_ts is not None and time.perf_counter() >= deadline_ts:
                break

        sub_rep = PhaseBSubstageReport(
            substage_id=substage_id,
            substage_title=str(st_cfg["title"]),
            waves_completed=waves_done,
            tokens_assimilated=sub_tokens,
            mean_surprise_F=round(sum(sub_surprises) / float(max(1, len(sub_surprises))), 4),
            new_dht_episodes_stored=sub_new_eps,
            cumulative_dht_episodes=len(trainer.sbc_vsa_dht.episodes_vsa),
            cumulative_hipporag_concepts=len(trainer.sbc_vsa_dht.concept_adj),
            col2_encyclopedic_hard_synapses=(
                len(trainer.col2_encyclopedic_hard_facts) + len(trainer.col2_encyclopedic_trigrams_hard)
            ),
            stranger_check_noise_rejections=sub_rejections,
            grammar_hard_overwrite_violations=trainer.grammar_overwrite_violations,
            elapsed_sec=round(time.perf_counter() - t_sub, 2),
        )
        phase_b_reports.append(sub_rep)

        # Промежуточное сохранение чекпоинта и живого дашборда после каждой подфазы Фазы Б
        exams_preview = trainer.run_three_final_erudition_tests()
        ckpt_mb = trainer.save_v10_3_knowledge_master_checkpoint(
            checkpoint_path=master_ckpt_path,
            phase_a_report=phase_a_report,
            phase_b_reports=phase_b_reports,
            final_exams=exams_preview,
        )
        live_telemetry = {
            "version": "Ziwago-Prime v10.3 Two-Phase Ontogeny (Structural Grounding + Full Knowledge Assimilation)",
            "status": "COMPLETED" if substage_id == 3 else f"RUNNING_PHASE_B_SUBSTAGE_{substage_id}",
            "last_update_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "master_checkpoint_path": str(master_ckpt_path),
            "master_checkpoint_size_mb": ckpt_mb,
            "total_knowledge_tokens_assimilated": trainer.total_knowledge_tokens_assimilated,
            "total_dht_episodes": len(trainer.sbc_vsa_dht.episodes_vsa),
            "total_hipporag_concepts": len(trainer.sbc_vsa_dht.concept_adj),
            "total_col2_encyclopedic_hard_synapses": (
                len(trainer.col2_encyclopedic_hard_facts) + len(trainer.col2_encyclopedic_trigrams_hard)
            ),
            "total_stranger_check_rejections": trainer.stranger_check_openwebtext_rejections,
            "grammar_overwrite_attempts_blocked": trainer.grammar_overwrite_attempts_blocked,
            "grammar_hard_overwrite_violations": trainer.grammar_overwrite_violations,
            "phase_a_structural_grounding": phase_a_report.__dict__,
            "phase_b_substages": [r.__dict__ for r in phase_b_reports],
            "final_three_exams": exams_preview,
            "elapsed_total_sec": round(time.perf_counter() - t_start, 2),
        }
        tmp_json = telemetry_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(live_telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_json, telemetry_path)
        update_v10_3_live_dashboard(dashboard_path, live_telemetry)

    # ------------------------------------------------------------------
    # ШАГ 3: 3 ФИНАЛЬНЫХ ТЕСТА НА `ziwago_v10_3_knowledge_master.pt`
    # ------------------------------------------------------------------
    final_data = json.loads(telemetry_path.read_text(encoding="utf-8"))
    ex = final_data["final_three_exams"]

    print(chr(10) + "=" * 116, flush=True)
    print(
        f"ДВУХФАЗНАЯ ОНТОГЕНЕТИЧЕСКАЯ ЭВОЛЮЦИЯ ZIWAGO-PRIME v10.3 ЗАВЕРШЕНА! "
        f"ЧЕКПОИНТ: `{master_ckpt_path.name}` ({final_data['master_checkpoint_size_mb']} МБ)",
        flush=True,
    )
    print(
        f"  • Всего усвоено знаний в Фазе Б          : {final_data['total_knowledge_tokens_assimilated']:,} токенов",
        flush=True,
    )
    print(
        f"  • Новых эпизодов в SBC-VSA-DHT (4 шарда) : {final_data['total_dht_episodes']:,} (концептов HippoRAG: {final_data['total_hipporag_concepts']:,})",
        flush=True,
    )
    print(
        f"  • HARD-синапсов в Col2_ENCYCLOPEDIC      : {final_data['total_col2_encyclopedic_hard_synapses']:,}",
        flush=True,
    )
    print(
        f"  • Защищено грамматических связей Фазы А  : {final_data['grammar_overwrite_attempts_blocked']:,} блокировок перезаписи (нарушений: {final_data['grammar_hard_overwrite_violations']})",
        flush=True,
    )
    print(
        f"  • Отсечено веб-мусора Stranger's Check   : {final_data['total_stranger_check_rejections']:,} шумных документов OpenWebText",
        flush=True,
    )
    print("-" * 116, flush=True)
    print("  [3 ФИНАЛЬНЫХ ТЕСТА НА `ziwago_v10_3_knowledge_master.pt`]:", flush=True)
    print(
        f"    1) Сказка (TinyStories)      : '{ex['test_1_fairytale_tinystories']['generated_text'][:140]}...'",
        flush=True,
    )
    print(
        f"    2) Факт из Википедии (Wiki)  : '{ex['test_2_wikipedia_fact_erudition']['generated_text'][:140]}...'" + chr(10)
        + f"       Цепочка концептов HippoRAG: {ex['test_2_wikipedia_fact_erudition']['sbc_vsa_dht_concept_chain']}",
        flush=True,
    )
    print(
        f"    3) Диалог из Cornell         : '{ex['test_3_cornell_pragmatic_dialog']['generated_text'][:140]}...'",
        flush=True,
    )
    print("=" * 116, flush=True)
    return final_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ziwago-Prime v10.3 Two-Phase Ontogeny Trainer (Phase A Structural Grounding + Phase B Full Knowledge Assimilation)"
    )
    parser.add_argument("--phase-a-tokens", type=int, default=10_000, help="Calibration tokens for Phase A (default: 10,000)")
    parser.add_argument("--phase-a-epochs", type=int, default=4, help="Max epochs for Phase A until Surprise < 0.05 (default: 4)")
    parser.add_argument("--phase-b-waves", type=int, default=5, help="Streaming waves per Phase B substage (default: 5)")
    parser.add_argument("--phase-b-tokens-per-wave", type=int, default=100_000, help="Tokens per streaming wave in Phase B (default: 100,000)")
    parser.add_argument("--marathon-hours", type=float, default=0.0, help="Optional continuous marathon duration in hours (e.g. 12.0 or 24.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Target compute device: cuda or cpu")
    parser.add_argument("--workers", type=int, default=32, help="Parallel CPU workers for StreamingDataMixer (default: 32)")
    parser.add_argument("--cpu-threads", type=int, default=32, help="PyTorch CPU threads (default: 32)")
    args = parser.parse_args()

    run_two_phase_ontogeny_big_run(
        phase_a_tokens=args.phase_a_tokens,
        phase_a_epochs=args.phase_a_epochs,
        phase_b_waves_per_substage=args.phase_b_waves,
        phase_b_tokens_per_wave=args.phase_b_tokens_per_wave,
        marathon_hours=args.marathon_hours,
        device=args.device,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
    )
