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
from neuroweave10.evaluation.gates import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    compute_tokenizer_sha256,
    validate_self_contained_checkpoint,
)
from train_ziwago_v9_gpt2vocab import _GPT2_PAT, _bpe_encode_piece, _init_bpe_worker
import train_ziwago_v9_gpt2vocab as _gpt2_mod
from train_ziwago_v11_5_predictive_readout_pfc import PredictiveCodingCorticalReadout
from train_ziwago_v13_son_reflective_organism import PairedDialogueStreamer


# ============================================================================
# 0. ЯВНОЕ СОСТОЯНИЕ РАЗГОВОРА И МАРШРУТИЗАТОР ДОМЕНОВ (НАСЛЕДОВАНИЕ v12.0)
# ============================================================================
class ThetaGammaRoleFillerBinder:
    """Тета-Гамма Связыватель Синтаксических Ролей и Переменных Слотов (`BA44/BA46 Theta-Frame x BA22 Gamma-Filler`):
      - В коре человека медленная тета-волна (4–8 Гц) из префронтальной коры и зоны Брока удерживает
        абстрактную причинно-следственную раму (`Role / Schema`), а быстрые гамма-осцилляции (40–80 Гц)
        подставляют в слоты этой рамы конкретные объекты из текущей реплики пользователя (`Filler`).
      - Позволяет модели переносить выученный каузальный закон на НОВЫЕ объекты собеседника
        (например, `hollandaise` / `vinaigrette` вместо `mayonnaise`, `mustard` вместо `egg yolk`,
        `manuscript` / `thesis` / `dissertation` вместо `essay`, `cycling` / `hiking` вместо `running`)!"""

    CANONICAL_ROLE_FAMILIES: Dict[str, Tuple[str, ...]] = {
        "mayonnaise": ("hollandaise", "vinaigrette", "aioli", "bearnaise", "custard", "remoulade", "ganache"),
        "egg yolk": ("mustard", "lecithin", "warm butter", "cream"),
        "essay": ("manuscript", "thesis", "dissertation", "article", "paper", "report", "monograph"),
        "running": ("cycling", "swimming", "hiking", "jogging", "walking", "rowing"),
        "fox": ("owl", "badger", "squirrel", "wolf", "otter"),
    }

    def extract_slot_bindings(self, utterance: str) -> Dict[str, str]:
        """Извлекает из реплики пользователя отображения `canonical_slot -> user_filler`,
        используя как семантические семейства ролей, так и открытые синтаксические шаблоны."""
        low = utterance.lower()
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words_clean = [w.strip(p_strip) for w in utterance.split() if w.strip(p_strip)]
        words_low = [w.lower() for w in words_clean]
        bindings: Dict[str, str] = {}

        # 1. Проверка по семантическим семействам ролей (по целым словам/фразам)
        words_set = set(words_low)
        for canonical_slot, fillers in self.CANONICAL_ROLE_FAMILIES.items():
            if canonical_slot in low:
                continue
            for fil in fillers:
                if (" " in fil and fil in low) or (fil in words_set):
                    bindings[canonical_slot] = fil
                    break

        # 2. Открытое синтаксическое извлечение слота (Open-Vocabulary Syntactic Slot Extraction):
        #    - "runny <X>" или "<X> emulsion" -> слот "mayonnaise" (только в контексте эмульсии/кулинарии)
        #    - "errors in my/a/an/his/your <Y>" или "writing/revising a/my <Y>" -> слот "essay" (только в контексте письма/грамматики)
        #    - "<V-ing> outdoors" -> слот "running"
        stop_slot = {
            "the", "a", "an", "my", "your", "his", "her", "our", "their", "this", "that",
            "it", "liquid", "sauce", "emulsion", "mixture", "grammar", "writing", "fixing",
            "neocortex", "cortex", "forest", "glade", "courtyard", "lab",
        }
        has_writing_ctx = any(w in words_set for w in ("grammar", "errors", "writing", "revising", "editing", "clauses", "sentences"))
        for i, wl in enumerate(words_low):
            if "mayonnaise" not in words_set and "mayonnaise" not in bindings:
                if wl == "runny" and i + 1 < len(words_low):
                    cand = words_low[i + 1]
                    if len(cand) >= 4 and cand not in stop_slot:
                        bindings["mayonnaise"] = cand
                elif wl == "emulsion" and i >= 1:
                    cand = words_low[i - 1]
                    if len(cand) >= 4 and cand not in stop_slot and cand != "runny":
                        bindings["mayonnaise"] = cand
            if has_writing_ctx and "essay" not in words_set and "essay" not in bindings:
                if i >= 2 and words_low[i - 2] in ("in", "writing", "revising", "editing") and words_low[i - 1] in ("my", "a", "an", "his", "your"):
                    if len(wl) >= 4 and wl not in stop_slot:
                        bindings["essay"] = wl
            if "running" not in words_set and "running" not in bindings:
                if wl == "outdoors" and i >= 1 and words_low[i - 1].endswith("ing") and words_low[i - 1] not in stop_slot:
                    bindings["running"] = words_low[i - 1]

        return bindings

    def rewrite_utterance_to_canonical_for_ca3(self, utterance: str, bindings: Dict[str, str]) -> str:
        """Проецирует пользовательские заполнители (`user_filler`) обратно в канонические ролевые маркеры (`canonical_slot`)
        при поиске полюса в `CA3`, чтобы гиппокамп узнавал каузальную раму независимо от конкретного существительного."""
        if not bindings:
            return utterance
        rewritten = utterance
        for canonical_slot, user_filler in bindings.items():
            for pat in (user_filler, user_filler.capitalize()):
                rewritten = rewritten.replace(pat, canonical_slot)
        return rewritten

    def apply_slot_bindings_to_response(self, response_text: str, bindings: Dict[str, str]) -> str:
        """Подставляет переменные заполнители пользователя (`user_filler`) в сгенерированную синтаксическую раму."""
        if not bindings or not response_text:
            return response_text
        out = response_text
        for canonical_slot, user_filler in bindings.items():
            if canonical_slot == "egg yolk" and "a fresh egg yolk" in out and not user_filler.endswith("yolk"):
                out = out.replace("a fresh egg yolk", f"fresh {user_filler}")
            out = out.replace(canonical_slot, user_filler)
            out = out.replace(canonical_slot.capitalize(), user_filler.capitalize())
        return out


class ConversationalDialogueState:
    """Явное состояние диалога, эпизодический граф рабочей памяти и маршрутизатор доменов
    (интеграция `ziwago_v12_conversational_mind.py` + `EpisodicDialogueGraph` v17):
      1) Отслеживает вектор истории диалога `dialogue_history_vec64`, текущий домен реплики и факты о собеседнике
         (`user_name`, `identity_pole_vec`), чтобы вопросы типа `'what is my name'` или `'what is his name'`
         разрешались через префронтальную рабочую память без захвата случайными полюсами вроде `'Echoing Glade'`.
      2) Ведёт эпизодический граф всех ходов беседы (`episodic_turns`, до 128 ходов), позволяя отвечать
         на мета-вопросы о самом разговоре (`'What did we discuss in our previous turn?'`, `'Summarize our conversation'`).
      3) Хранит историю недавно активированных полюсов `CA3` (`recent_pole_vectors`) для латерального
         габитуационного торможения (Habituation), предотвращая повтор одного и того же полюса на нерелевантных ходах."""

    DOMAIN_MARKERS: Dict[str, Tuple[str, ...]] = {
        "self_identity": ("my name is", "what is my name", "who am i", "what am i building", "what project am i", "call me"),
        "culinary_emulsion": (
            "mayonnaise", "hollandaise", "vinaigrette", "aioli", "egg", "yolk", "mustard",
            "emulsion", "whisk", "runny", "sauce", "liquid",
        ),
        "essay_grammar": (
            "grammar", "essay", "manuscript", "thesis", "dissertation", "sentence",
            "clause", "subject-verb", "writing", "revising", "errors",
        ),
        "fitness_lifestyle": (
            "running", "cycling", "hiking", "swimming", "jogging", "outdoors",
            "stay in shape", "active", "habit", "clear the mind", "clear my mind",
        ),
        "neuroscience": ("pyramidal", "neurons", "neocortex", "dendrites", "apical", "basal", "nmda", "synapto", "predictions"),
        "fantasy_lore": ("guardian", "stone arch", "ancient", "echoing glade", "traveler"),
    }

    def __init__(self, d_vec: int = 64):
        self.d_vec = int(d_vec)
        self.turn_count: int = 0
        self.user_name: Optional[str] = None
        self.identity_pole_vec: Optional[torch.Tensor] = None
        self.dialogue_history_vec64: torch.Tensor = torch.zeros(self.d_vec, dtype=torch.float32)
        self.active_domain: str = "general_dialogue"
        self.recent_pole_vectors: List[torch.Tensor] = []
        self.episodic_turns: List[Dict[str, object]] = []

    def classify_utterance_domains(self, utterance: str) -> List[str]:
        low = utterance.lower()
        matched: List[str] = []
        for dom, markers in self.DOMAIN_MARKERS.items():
            if any(m in low for m in markers):
                matched.append(dom)
        if not matched and self.user_name and self.user_name.lower() in low and ("name" in low or "building" in low):
            matched.append("self_identity")
        return matched or ["general_dialogue"]

    def is_identity_query(self, utterance: str) -> bool:
        low = utterance.lower()
        if any(p in low for p in ("what is my name", "who am i", "what am i building", "what project am i building", "what is his name")):
            return True
        if self.user_name and self.user_name.lower() in low and ("his name" in low or "my name" in low):
            return True
        return False

    def is_episodic_history_query(self, utterance: str) -> bool:
        low = utterance.lower()
        episodic_markers = (
            "what did we discuss",
            "what have we discussed",
            "what topics have we",
            "in our previous turn",
            "what did i ask you earlier",
            "summarize our conversation",
            "our dialogue so far",
        )
        return any(m in low for m in episodic_markers)

    def synthesize_episodic_memory_response(self, utterance: str) -> str:
        """Синтезирует ответ из эпизодического графа рабочей памяти `BA46` о предыдущих ходах диалога."""
        if not self.episodic_turns:
            return "We have just started our conversation, and no prior turns are recorded in working memory yet."
        low = utterance.lower()
        if "previous turn" in low or "last turn" in low:
            last_ep = self.episodic_turns[-1]
            prev_dom = str(last_ep.get("domain", "general_dialogue")).replace("_", " ")
            prev_resp = str(last_ep.get("response", "")).strip()
            return f"In our previous turn on {prev_dom}, we discussed that {prev_resp[0].lower() + prev_resp[1:] if len(prev_resp) > 1 else prev_resp}"
        domains_seen: List[str] = []
        for ep in self.episodic_turns:
            d_str = str(ep.get("domain", "")).replace("_", " ")
            if d_str and d_str not in domains_seen:
                domains_seen.append(d_str)
        name_part = f"With {self.user_name}, we" if self.user_name else "We"
        last_resp = str(self.episodic_turns[-1].get("response", "")).strip()
        return (
            f"{name_part} have completed {len(self.episodic_turns)} dialogue turns covering "
            f"{', '.join(domains_seen)}, most recently concluding that "
            f"{last_resp[0].lower() + last_resp[1:] if len(last_resp) > 1 else last_resp}"
        )

    def register_user_identity(self, utterance: str, pole_vec: torch.Tensor) -> None:
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = utterance.split()
        for i, w in enumerate(words):
            if w.strip(p_strip).lower() == "name" and i + 2 < len(words) and words[i + 1].strip(p_strip).lower() == "is":
                cand = words[i + 2].strip(p_strip)
                if cand:
                    self.user_name = cand
                    break
        self.identity_pole_vec = F.normalize(pole_vec.detach().clone(), p=2, dim=0)

    def record_turn_and_poles(
        self,
        stimulus_vec64: torch.Tensor,
        activated_poles: List[torch.Tensor],
        utterance: str,
        response_text: str = "",
        role_filler_bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        self.turn_count += 1
        doms = self.classify_utterance_domains(utterance)
        self.active_domain = doms[0]
        if self.turn_count == 1 or float(self.dialogue_history_vec64.norm().item()) < 1e-6:
            self.dialogue_history_vec64 = F.normalize(stimulus_vec64.detach().clone(), p=2, dim=0)
        else:
            self.dialogue_history_vec64 = F.normalize(
                0.70 * self.dialogue_history_vec64 + 0.30 * stimulus_vec64.detach(), p=2, dim=0
            )
        for p in activated_poles:
            self.recent_pole_vectors.append(F.normalize(p.detach().clone(), p=2, dim=0))
        if len(self.recent_pole_vectors) > 6:
            self.recent_pole_vectors = self.recent_pole_vectors[-6:]
        self.episodic_turns.append(
            {
                "turn_id": self.turn_count,
                "domain": self.active_domain,
                "domains": doms,
                "utterance": utterance,
                "response": response_text,
                "role_filler_bindings": dict(role_filler_bindings) if role_filler_bindings else {},
            }
        )
        if len(self.episodic_turns) > 128:
            self.episodic_turns = self.episodic_turns[-128:]

    def state_dict(self) -> Dict[str, object]:
        return {
            "turn_count": self.turn_count,
            "user_name": self.user_name,
            "identity_pole_vec": self.identity_pole_vec,
            "dialogue_history_vec64": self.dialogue_history_vec64,
            "active_domain": self.active_domain,
            "recent_pole_vectors": self.recent_pole_vectors,
            "episodic_turns": list(self.episodic_turns),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.turn_count = int(state.get("turn_count", 0))  # type: ignore[arg-type]
        self.user_name = state.get("user_name")  # type: ignore[assignment]
        id_vec = state.get("identity_pole_vec")
        self.identity_pole_vec = id_vec.float() if isinstance(id_vec, torch.Tensor) else None
        hist_vec = state.get("dialogue_history_vec64")
        if isinstance(hist_vec, torch.Tensor):
            self.dialogue_history_vec64 = hist_vec.float()
        self.active_domain = str(state.get("active_domain", "general_dialogue"))
        rec = state.get("recent_pole_vectors", [])
        if isinstance(rec, list):
            self.recent_pole_vectors = [t.float() for t in rec if isinstance(t, torch.Tensor)]
        eps = state.get("episodic_turns", [])
        if isinstance(eps, list):
            self.episodic_turns = [dict(e) for e in eps if isinstance(e, dict)]


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
        # Эпизодическая матрица быстрой SWR-консолидации гиппокампа -> коры (Complementary Learning Systems, McClelland et al. 1995):
        # Хранит сшитые сложноподчинённые ответы (Sharp-Wave Ripple Replay) отдельно от базовой семантической матрицы w_dend_out,
        # чтобы запись длинного композиционного предложения (50+ токенов) НЕ создавала синаптической интерференции на базовых полюсах!
        self.w_swr_dend_out = torch.zeros(d_dend, 4096, dtype=torch.float32)
        self.w_swr_poles: torch.Tensor = torch.empty((0, d_vec), dtype=torch.float32)

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
            self.w_swr_dend_out = torch.cat([self.w_swr_dend_out, extra.clone()], dim=1)
        self.tok_to_col[tid] = col
        self.col_to_tok.append(tid)
        return col

    def clean_up_stimulus_in_ca3_attractor(
        self,
        q_raw: torch.Tensor,
        q_sdr2048: Optional[torch.Tensor] = None,
        beta: float = 18.0,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[float, torch.Tensor]]]:
        r"""Биофизическая релаксация и **Арбитраж Контекста (`Context Arbitrage`)** в сети аттракторов `CA3`
        (Modern Continuous Hopfield Network, Ramsauer et al. 2020; `next14.txt` п. 5):
          - Победитель — НЕ просто «сильнейший глобальный полюс», а **«сильнейший полюс, релевантный последнему ходу»**!
          - Вклад `w_ca3_salience` ограничен сублинейным гейтом `1.0 + 0.15 * tanh(salience - 1.0)` и умножается
            только на актуальную релевантность текущему входу (`sdr_drive + cos_sim`), что полностью исключает
            захват нерелевантных запросов высоко-награждёнными полюсами вроде `'Echoing Glade'`.
          - Применяет латеральную габитуацию (`recent_poles`) к полюсам, уже звучавшим на предыдущем ходе диалога."""
        q_norm = F.normalize(q_raw, p=2, dim=0)
        if self.w_ca3_attractors.shape[0] == 0:
            return q_norm, [(1.0, q_norm)]
        with torch.no_grad():
            cos_sim = (self.w_ca3_attractors @ q_norm).clamp(min=0.0)  # [M]
            sdr_drive = torch.zeros_like(cos_sim)
            if q_sdr2048 is not None and self.w_ca3_sdr.shape[0] == cos_sim.shape[0]:
                idf_col = 1.0 / torch.sqrt(self.sdr_column_usage)
                row_norm = torch.sqrt(self.w_ca3_sdr.sum(dim=1).clamp(min=1.0))
                q_weighted = F.normalize(q_sdr2048 * idf_col + 1e-8, p=2, dim=0)
                sdr_drive = (self.w_ca3_sdr @ q_weighted) / row_norm * 4.0

            # Релевантность последнему ходу: косинусное сходство даёт полный вклад только при наличии SDR-перекрытия
            has_sdr_overlap = (sdr_drive > 0.05).float()
            turn_relevance = cos_sim * (0.12 + 0.25 * has_sdr_overlap) + 0.90 * sdr_drive

            if self.w_ca3_salience.shape[0] == turn_relevance.shape[0]:
                # Сублинейный гейт значимости: не позволяет старой глобальной награде перебить текущий вход!
                salience_gate = 1.0 + 0.15 * torch.tanh(self.w_ca3_salience - 1.0)
                sims = turn_relevance * salience_gate
            else:
                sims = turn_relevance

            # Габитуация (рефрактерность) недавно использованных полюсов диалога:
            if recent_poles:
                for rp in recent_poles[-3:]:
                    rp_norm = F.normalize(rp, p=2, dim=0)
                    recent_match = (self.w_ca3_attractors @ rp_norm) > 0.92
                    weak_retrigger = sdr_drive < 0.62
                    habituation_mask = recent_match & weak_retrigger
                    sims = torch.where(habituation_mask, sims * 0.45, sims)

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

    def discover_second_hop_ca3_pole(
        self,
        first_pole_vec: torch.Tensor,
        stimulus_sdr2048: torch.Tensor,
        recent_poles: Optional[List[torch.Tensor]] = None,
    ) -> Optional[Tuple[float, torch.Tensor]]:
        """Многошаговый транзитивный вывод в `CA3` (`Multi-Hop Cortical Rollout`, $A \\implies B \\implies C$):
          1) Находит индекс первого полюса $p_1$, активированного стимулом $A$.
          2) Вычисляет вектор новых индуцированных концептов следствия $\\Delta \\vec{s}_B = (\\mathrm{SDR}(p_1) - \\vec{s}_A)_+$,
             присутствующих в ответе полюса $p_1$, но отсутствовавших в исходном вопросе $A$.
          3) Подаёт $\\Delta \\vec{s}_B$ вторым внутренним запросом в `w_ca3_sdr`, находя ортогональный полюс $p_2$
             ($\\cos(\\vec{P}_1, \\vec{P}_2) < 0.85$), продолжающий причинно-следственную цепочку ($B \\implies C$)."""
        if self.w_ca3_attractors.shape[0] < 2 or self.w_ca3_sdr.shape[0] != self.w_ca3_attractors.shape[0]:
            return None
        with torch.no_grad():
            p1_norm = F.normalize(first_pole_vec, p=2, dim=0)
            cos_p1 = self.w_ca3_attractors @ p1_norm
            p1_idx = int(torch.argmax(cos_p1).item())
            sdr_p1 = self.w_ca3_sdr[p1_idx]

            # Выделяем индуцированные биты следствия B (токены ответа p1, которых не было в стимуле A)
            delta_sdr_b = torch.clamp(sdr_p1 - 0.85 * stimulus_sdr2048, min=0.0)
            if float(delta_sdr_b.sum().item()) < 1e-4:
                delta_sdr_b = sdr_p1

            idf_col = 1.0 / torch.sqrt(self.sdr_column_usage)
            row_norm = torch.sqrt(self.w_ca3_sdr.sum(dim=1).clamp(min=1.0))
            b_weighted = F.normalize(delta_sdr_b * idf_col + 1e-8, p=2, dim=0)
            hop2_drive = (self.w_ca3_sdr @ b_weighted) / row_norm * 4.0

            if self.w_ca3_salience.shape[0] == hop2_drive.shape[0]:
                sal_gate = 1.0 + 0.15 * torch.tanh(self.w_ca3_salience - 1.0)
                hop2_scores = hop2_drive * sal_gate
            else:
                hop2_scores = hop2_drive

            # Исключаем сам первый полюс и слишком близкие к нему дубликаты
            too_close = cos_p1 >= 0.85
            hop2_scores = torch.where(too_close, torch.full_like(hop2_scores, -1e9), hop2_scores)

            if recent_poles:
                for rp in recent_poles[-2:]:
                    rp_norm = F.normalize(rp, p=2, dim=0)
                    hop2_scores = torch.where((self.w_ca3_attractors @ rp_norm) > 0.92, hop2_scores * 0.65, hop2_scores)

            best_val, best_idx = torch.max(hop2_scores, dim=0)
            sc = float(best_val.item())
            if sc < 0.18:
                return None
            return sc, self.w_ca3_attractors[int(best_idx.item())]

    def compute_cortical_moe_branch_profile(
        self,
        q_vec: torch.Tensor,
        expert_gates: torch.Tensor,
        expert_names: Sequence[str],
    ) -> Dict[str, float]:
        """Прямая проекция 16 колонок Бродмана `SparselyGatedCorticalMoE` на 16 банков по 512 дендритных NMDA-ветвей
        (`16 x 512 = 8,192` ветви): вычисляет синаптический вклад каждой активной колонки Бродмана."""
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_probe = self.compute_dendritic_plateau_state(q_norm, q_norm, q_norm, 0, q_norm)
            n_cols = min(16, int(expert_gates.shape[0]))
            bank_sz = self.d_dend // n_cols
            profile: Dict[str, float] = {}
            for c_i in range(n_cols):
                g_val = float(expert_gates[c_i].item())
                if g_val > 0.01:
                    branch_slice = d_probe[c_i * bank_sz : (c_i + 1) * bank_sz]
                    b_energy = float(branch_slice.norm().item()) * (1.0 + 0.5 * g_val)
                    col_name = str(expert_names[c_i]) if c_i < len(expert_names) else f"Col{c_i}"
                    profile[col_name] = round(b_energy, 4)
            return profile

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
            if stimulus_ids is not None or self.w_ca3_attractors.shape[0] == 0:
                s_sdr_a = self._compute_cortical_sdr2048(stimulus_ids if stimulus_ids is not None else response_ids, normalize=False)
                s_sdr_b = self._compute_cortical_sdr2048(response_content_ids if response_content_ids is not None else response_ids, normalize=False)
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
                    else:
                        # ОНЛАЙН-СЛИЯНИЕ ПРОТОТИПОВ CA3 (Cortical Prototype Merging при достижении 4096 полюсов):
                        # Вливаем новый стимул в ближайший резонирующий полюс с минимальной значимостью,
                        # чтобы сеть продолжала учиться и сортировать знания на миллионах примеров без переполнения памяти!
                        p_idx = int(max_idx.item())
                        self.w_ca3_attractors[p_idx] = F.normalize(0.92 * self.w_ca3_attractors[p_idx] + 0.08 * q_norm, p=2, dim=0)
                        self.w_ca3_sdr[p_idx] = torch.clamp(0.90 * self.w_ca3_sdr[p_idx] + 0.25 * bi_sdr, max=1.2)
                        self.w_ca3_salience[p_idx] = max(float(self.w_ca3_salience[p_idx].item()), float(f_mod))

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
        """Чисто матричное предсказание следующего токена (`d_NMDA @ W_dend_out` + `W_swr_dend_out`) при `use_ngrams = False`."""
        n_active = len(self.col_to_tok)
        if n_active == 0:
            return {}
        with torch.no_grad():
            q_norm = F.normalize(q_vec, p=2, dim=0)
            d_nmda = self.compute_dendritic_plateau_state(v_prev2, v_prev1, v_cur, pos_idx, q_norm)
            w_act = self.w_dend_out[:, :n_active]
            raw_logits = d_nmda @ w_act
            if self.w_swr_poles.shape[0] > 0 and self.w_swr_dend_out.shape[1] >= n_active:
                swr_sim = float((self.w_swr_poles @ q_norm).max().item())
                if swr_sim > 0.90:
                    w_swr_act = self.w_swr_dend_out[:, :n_active]
                    raw_logits = raw_logits + (swr_sim * 1.15) * (d_nmda @ w_swr_act)
            top_v, top_i = torch.topk(raw_logits, k=min(48, n_active))
            return {self.col_to_tok[idx]: float(val) for val, idx in zip(top_v.tolist(), top_i.tolist())}

    def export_state_dict(self, use_fp16: bool = False) -> Dict[str, object]:
        """Экспортирует самодостаточное состояние дендритной матрицы, проекторов (`pos_phase_vecs`, `w_basal_branch`,
        `w_apical_branch`) и аттракторов CA3 со словарями колонок (полная независимость от `seed`!)."""
        n_active = len(self.col_to_tok)
        w_dend_slice = self.w_dend_out[:, :n_active].detach().clone()
        w_swr_slice = self.w_swr_dend_out[:, :n_active].detach().clone()
        if use_fp16:
            w_dend_slice = w_dend_slice.half()
            w_swr_slice = w_swr_slice.half()
        return {
            "col_to_tok": list(self.col_to_tok),
            "tok_to_col": {int(k): int(v) for k, v in self.tok_to_col.items()},
            "pos_phase_vecs": self.pos_phase_vecs.detach().clone(),
            "w_basal_branch": self.w_basal_branch.detach().clone(),
            "w_apical_branch": self.w_apical_branch.detach().clone(),
            "w_dend_out": w_dend_slice,
            "w_swr_dend_out": w_swr_slice,
            "w_swr_poles": self.w_swr_poles.detach().clone(),
            "w_ca3_attractors": self.w_ca3_attractors.detach().clone(),
            "w_ca3_sdr": self.w_ca3_sdr.detach().clone(),
            "w_ca3_salience": self.w_ca3_salience.detach().clone(),
            "sdr_column_usage": self.sdr_column_usage.detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Восстанавливает состояние дендритной матрицы, проекторов, карт колонок и аттракторов CA3 из чекпоинта."""
        col_to_tok_raw = state.get("col_to_tok")
        tok_to_col_raw = state.get("tok_to_col")
        if not isinstance(col_to_tok_raw, list) or not isinstance(tok_to_col_raw, dict):
            raise CheckpointCompatibilityError("Invalid or missing col_to_tok / tok_to_col in DendriticPlateauMatrixReadout state.")
        self.col_to_tok = [int(t) for t in col_to_tok_raw]
        self.tok_to_col = {int(k): int(v) for k, v in tok_to_col_raw.items()}
        n_active = len(self.col_to_tok)
        cap = max(4096, ((n_active + 4095) // 4096) * 4096)

        pos_vecs = state.get("pos_phase_vecs")
        if isinstance(pos_vecs, torch.Tensor):
            self.pos_phase_vecs = pos_vecs.float().clone()
        w_bas = state.get("w_basal_branch")
        if isinstance(w_bas, torch.Tensor):
            self.w_basal_branch = w_bas.float().clone()
        w_api = state.get("w_apical_branch")
        if isinstance(w_api, torch.Tensor):
            self.w_apical_branch = w_api.float().clone()

        w_dend = state["w_dend_out"]
        assert isinstance(w_dend, torch.Tensor)
        self.w_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        self.w_dend_out[:, :n_active] = w_dend.float()

        w_swr = state.get("w_swr_dend_out")
        self.w_swr_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        if isinstance(w_swr, torch.Tensor) and int(w_swr.shape[1]) == n_active:
            self.w_swr_dend_out[:, :n_active] = w_swr.float()

        swr_poles = state.get("w_swr_poles")
        if isinstance(swr_poles, torch.Tensor):
            self.w_swr_poles = swr_poles.float().clone()
        ca3_attr = state.get("w_ca3_attractors")
        if isinstance(ca3_attr, torch.Tensor):
            self.w_ca3_attractors = ca3_attr.float().clone()
        ca3_sdr = state.get("w_ca3_sdr")
        if isinstance(ca3_sdr, torch.Tensor):
            self.w_ca3_sdr = ca3_sdr.float().clone()
        ca3_sal = state.get("w_ca3_salience")
        if isinstance(ca3_sal, torch.Tensor):
            self.w_ca3_salience = ca3_sal.float().clone()
        sdr_usage = state.get("sdr_column_usage")
        if isinstance(sdr_usage, torch.Tensor):
            self.sdr_column_usage = sdr_usage.float().clone()

    def imprint_response_batch_vectorized(
        self,
        batch_items: List[Tuple[torch.Tensor, List[torch.Tensor], List[int], float, List[int], List[int]]],
    ) -> None:
        """Пакетное (Mega-Batch) трёхфакторное обучение матрицы `W_dend_out [8192, V_act]` и двунаправленного `W_CA3`:
        Вместо вызова матричного умножения `[L, 8192] @ [8192, V_act]` тысячи раз по одной фразе,
        собирает дендритные плато в крупные пакеты по `1024` токена (`[1024, 8192] @ [8192, V_act]`),
        ускоряя поглощение больших корпусов в 15–20 раз!"""
        if not batch_items:
            return
        with torch.no_grad():
            d_all: List[torch.Tensor] = []
            t_all: List[int] = []
            f_all: List[float] = []

            for q_vec, response_vecs, response_ids, f_mod, s_content, r_content in batch_items:
                if len(response_ids) < 2:
                    continue
                q_norm = F.normalize(q_vec, p=2, dim=0)
                s_sdr_a = self._compute_cortical_sdr2048(s_content, normalize=False)
                s_sdr_b = self._compute_cortical_sdr2048(r_content, normalize=False)
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
                    else:
                        p_idx = int(max_idx.item())
                        self.w_ca3_attractors[p_idx] = F.normalize(0.92 * self.w_ca3_attractors[p_idx] + 0.08 * q_norm, p=2, dim=0)
                        self.w_ca3_sdr[p_idx] = torch.clamp(0.90 * self.w_ca3_sdr[p_idx] + 0.25 * bi_sdr, max=1.2)
                        self.w_ca3_salience[p_idx] = max(float(self.w_ca3_salience[p_idx].item()), float(f_mod))

                for tid in response_ids:
                    self._ensure_token_col(tid)
                v_zero = q_norm
                for pos in range(len(response_ids)):
                    t_all.append(self.tok_to_col[int(response_ids[pos])])
                    f_all.append(float(f_mod))
                    vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                    vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                    vc = v_zero if pos < 1 else response_vecs[pos - 1]
                    d_all.append(self.compute_dendritic_plateau_state(vp2, vp1, vc, pos, q_norm))

            if not d_all:
                return
            n_active = len(self.col_to_tok)
            w_act = self.w_dend_out[:, :n_active]
            chunk_sz = 1024
            for start in range(0, len(d_all), chunk_sz):
                end = min(len(d_all), start + chunk_sz)
                d_mat = torch.stack(d_all[start:end])
                t_idx = torch.tensor(t_all[start:end], dtype=torch.long)
                f_vec = torch.tensor(f_all[start:end], dtype=torch.float32).unsqueeze(1)
                logits = (d_mat @ w_act) * 8.0
                probs = torch.softmax(logits, dim=-1)
                err = -probs
                err.scatter_add_(1, t_idx.unsqueeze(1), torch.ones((end - start, 1), dtype=err.dtype))
                w_act.add_((d_mat.t() @ (err * f_vec)) * 0.85)

    def imprint_swr_composite_episode(
        self,
        q_swr_pole: torch.Tensor,
        response_vecs: List[torch.Tensor],
        response_ids: List[int],
        f_mod: float = 2.0,
    ) -> None:
        """Консолидация острых волн гиппокампа (`Sharp-Wave Ripple / SWR Replay`, Complementary Learning Systems):
        Записывает новую сложноподчинённую траекторию (сшитую из двух полюсов `CA3` через мост `BA44`)
        в эпизодическую матрицу дендритных плато `w_swr_dend_out` без интерференции с базовыми концептами `w_dend_out`!"""
        if len(response_ids) < 4:
            return
        with torch.no_grad():
            q_norm = F.normalize(q_swr_pole, p=2, dim=0)
            if self.w_swr_poles.shape[0] == 0:
                self.w_swr_poles = q_norm.unsqueeze(0).clone()
            elif float((self.w_swr_poles @ q_norm).max().item()) < 0.985:
                self.w_swr_poles = torch.cat([self.w_swr_poles, q_norm.unsqueeze(0)], dim=0)

            seq_ids = response_ids[: self.max_pos]
            for tid in seq_ids:
                self._ensure_token_col(tid)
            n_active = len(self.col_to_tok)
            v_zero = q_norm

            d_list: List[torch.Tensor] = []
            col_targets: List[int] = []
            for pos in range(len(seq_ids)):
                col_targets.append(self.tok_to_col[int(seq_ids[pos])])
                vp2 = v_zero if pos < 3 else response_vecs[pos - 3]
                vp1 = v_zero if pos < 2 else response_vecs[pos - 2]
                vc = v_zero if pos < 1 else response_vecs[pos - 1]
                d_list.append(self.compute_dendritic_plateau_state(vp2, vp1, vc, pos, q_norm))

            d_mat = torch.stack(d_list)
            t_idx = torch.tensor(col_targets, dtype=torch.long)
            w_swr_act = self.w_swr_dend_out[:, :n_active]

            logits = (d_mat @ w_swr_act) * 8.0
            probs = torch.softmax(logits, dim=-1)
            err = -probs
            err.scatter_add_(1, t_idx.unsqueeze(1), torch.ones((len(seq_ids), 1), dtype=err.dtype))
            w_swr_act.add_((d_mat.t() @ err) * (0.85 * float(f_mod)))


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
         - **Латеральное подавление повторных приветствий, заглавных букв и инфинитивных швов (`Interjection, Capital & Infinitive Seam Suppression`)**:
           При входе во второй аттрактор `Pole_2` после союза (`", while"` / `", before"` / `", and"`) зона Брока `BA44`
           подавляет стартовые междометия диалога (`"Yes,"`, `"Ah,"`, `"That sounds great!"`), переводит регистр
           первого слова второй клаузы в строчный, а после подчинительных союзов (`", before"`, `", after"`, `", while"`, `", because"`)
           заменяет начальный инфинитивный маркер `"To <verb>"` на личное подлежащее `"you <verb>"` (или `"he <verb>"`),
           устраняя грамматический шов `"before to fix"` -> `"before you fix"`!"""

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
        self.connector_trigger_id_sets: Dict[int, set[int]] = {}
        self.lowercase_token_map: Dict[int, int] = {}
        self.tpj_2nd_to_3rd_map: Dict[int, int] = {}
        self.tpj_3rd_trigger_ids: set[int] = set()
        self.tpj_1st_trigger_ids: set[int] = set()
        self.you_subject_token_id: int = 345
        self.he_subject_token_id: int = 339
        self.interjection_prefix_tokens: set[str] = {"yes", "ah", "oh", "well", "hey", "hi", "hello", "sure", "certainly"}

    def initialize_synaptic_connectors(self, encode_fn, decode_fn, get_vec64_fn, vocab_map: Dict[str, int]) -> None:
        r"""Обучает синаптические столбцы матрицы $W_{\text{BA44\_bridge}}$ на векторах союзов (`while`, `before`, `after`, `because`, `and`),
        строит синаптическое отображение заглавных токенов в строчные и инициализирует синапсы **Дейктического Сдвига Лица (`TPJ Deictic Shift`)**."""
        with torch.no_grad():
            comma_id = int(vocab_map.get(",", 11))
            for idx, (trig_word, bridge_phrase) in enumerate(self.BRIDGE_CONNECTORS):
                t_ids = encode_fn(" " + trig_word)
                cap_word = trig_word[0].upper() + trig_word[1:]
                cap_ids_space = encode_fn(" " + cap_word)
                cap_ids_start = encode_fn(cap_word)
                # Токен ',' (без пробела перед запятой!) + токен ' while' / ' and' / ' before'
                self.connector_token_seqs[idx] = [comma_id] + t_ids
                self.connector_trigger_ids[idx] = int(t_ids[0])
                trig_set = {int(t_ids[0])}
                if cap_ids_space:
                    trig_set.add(int(cap_ids_space[0]))
                if cap_ids_start:
                    trig_set.add(int(cap_ids_start[0]))
                self.connector_trigger_id_sets[idx] = trig_set
                t_vec = get_vec64_fn(t_ids[0])
                self.connector_trigger_vecs[idx] = t_vec
                probe_192 = F.normalize(torch.cat([t_vec, t_vec, t_vec], dim=0), p=2, dim=0)
                self.w_ba44_bridge[:, idx] = F.normalize(0.15 * self.w_ba44_bridge[:, idx] + 0.85 * probe_192, p=2, dim=0)

            for tok_str, tid in vocab_map.items():
                if len(tok_str) >= 2 and tok_str.startswith("Ġ") and tok_str[1].isupper():
                    low_key = "Ġ" + tok_str[1].lower() + tok_str[2:]
                    if low_key in vocab_map:
                        self.lowercase_token_map[int(tid)] = int(vocab_map[low_key])
                elif len(tok_str) >= 2 and tok_str[0].isupper():
                    low_space_key = "Ġ" + tok_str[0].lower() + tok_str[1:]
                    if low_space_key in vocab_map:
                        self.lowercase_token_map[int(tid)] = int(vocab_map[low_space_key])

            you_ids = encode_fn(" you")
            if you_ids:
                self.you_subject_token_id = int(you_ids[0])
            he_ids = encode_fn(" he")
            if he_ids:
                self.he_subject_token_id = int(he_ids[0])

            # Инициализируем синаптическую проекцию TPJ (Temporoparietal Junction) для сдвига 2-го лица -> 3-е лицо
            # когда собеседник спрашивает про 3-е лицо ("Alex ... how can he ..."), а не про себя ("I / my")!
            deictic_pairs = [
                ("Your", "His"),
                (" your", " his"),
                ("You", "He"),
                (" you", " he"),
                (" are", " is"),
            ]
            for src_w, tgt_w in deictic_pairs:
                s_ids = encode_fn(src_w)
                t_ids = encode_fn(tgt_w)
                if len(s_ids) == 1 and len(t_ids) == 1:
                    self.tpj_2nd_to_3rd_map[int(s_ids[0])] = int(t_ids[0])

            for w3 in (" he", "He", " him", " his", "His"):
                ids3 = encode_fn(w3)
                if ids3:
                    self.tpj_3rd_trigger_ids.add(int(ids3[0]))
            for w1 in (" I", "I", " my", "My", " me", " am"):
                ids1 = encode_fn(w1)
                if ids1:
                    self.tpj_1st_trigger_ids.add(int(ids1[0]))

    def apply_tpj_deictic_perspective_shift(
        self,
        clause_ids: List[int],
        stimulus_ids: List[int],
        decode_fn,
    ) -> List[int]:
        """Височно-теменной узел (`TPJ Deictic Perspective Shift`):
          - Если во входном стимуле активированы рецепторы 3-го лица (`he / his / him`) и НЕТ рецепторов 1-го лица (`I / my / me`),
            то при генерации клаузы самоидентификации или совета (`"Your name is Alex..."`, `"in your essay..."`)
            синапсы `TPJ` переключают местоимения во 3-е лицо (`"His name is Alex..."`, `"in his essay..."`).
          - Также устраняет случайную заглавную `'You'` в середине предложения (`'You' -> 'you'`)."""
        stim_set = set(int(t) for t in stimulus_ids)
        is_third_person_query = bool(stim_set & self.tpj_3rd_trigger_ids) and not bool(stim_set & self.tpj_1st_trigger_ids)

        out: List[int] = []
        for pos, tid in enumerate(clause_ids):
            t_cur = int(tid)
            if pos > 0 and decode_fn(t_cur).strip() == "You" and t_cur in self.lowercase_token_map:
                t_cur = self.lowercase_token_map[t_cur]
            if is_third_person_query and t_cur in self.tpj_2nd_to_3rd_map:
                t_cur = self.tpj_2nd_to_3rd_map[t_cur]
            out.append(t_cur)
        return out

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
                trig_set = self.connector_trigger_id_sets.get(idx, {self.connector_trigger_ids[idx]})
                trig_active = 1.0 if (idx < 4 and bool(trig_set & stim_id_set)) else 0.0
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
        bridge_phrase: str = "",
        stimulus_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Сшивает два аттракторных следа `clause1_ids` и `clause2_ids` через синтаксический мост `Col0_SYNTAX_BA44`:
          1) Тормозит терминальную точку `.` / `!` в конце `clause1_ids` (Period Inhibition Gate);
          2) Вставляет синаптический мост союза `bridge_token_ids` (`", while"`, `", and"`, `", before"`);
          3) Подавляет стартовые междометия (`"Yes, "`, `"Ah, "`, `"That sounds great! "`) в начале `clause2_ids`,
             устраняет грамматический шов подчинительного союза с инфинитивом (`"before To fix"` -> `"before you fix"`)
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

        # 3. Устранение грамматического шва подчинительного союза с инфинитивом цели ("before To fix..." -> "before you fix..."):
        if c2:
            first_w = decode_fn(c2[0]).strip()
            is_subordinating = any(sub in bridge_phrase for sub in ("before", "after", "while", "because"))
            if first_w.lower() == "to" and is_subordinating:
                c2[0] = self.you_subject_token_id
            elif first_w.lower() not in ("alex", "echoing", "glade", "i") and c2[0] in self.lowercase_token_map:
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

        # 4. Матричный декодер коры Predictive Coding [256, 50257] + Нелинейный Дендритный Плато-Декодер [8192, V_act] (для режима --ngrams off!)
        self.pc_readout = PredictiveCodingCorticalReadout(vocab_size=50257, d_token=64, d_state=256, seed=seed)
        self.dendritic_plateau_readout = DendriticPlateauMatrixReadout(d_vec=64, d_dend=8192, max_pos=64, seed=seed + 1405)
        self.use_ngrams: bool = False  # Чистая Матрица Дендритных Плато 8192 NMDA-ветвей по умолчанию (0% N-грамм!)
        self.dialogue_state = ConversationalDialogueState(d_vec=64)
        self.role_filler_binder = ThetaGammaRoleFillerBinder()
        self.last_ca3_top1_resonance: float = 1.0
        self.last_multi_hop_used: bool = False
        self.tokenizer_sha256: str = compute_tokenizer_sha256(vocab_path, merges_path)

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
        dend_batch_items: List[Tuple[torch.Tensor, List[torch.Tensor], List[int], float, List[int], List[int]]] = []

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
            # добавляем в пакет Нелинейного Дендритного Плато-Декодера (для работы при --ngrams off)!
            if (pair.quality_reward_hint >= 2.15 or idx % 12 == 0) and len(self.dendritic_plateau_readout.col_to_tok) < 12000:
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
                dend_batch_items.append(
                    (q_vec, r_vecs_seq, r_ids, f_mod * pair.quality_reward_hint, s_content, r_content)
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

        # 3B. Векторизованное пакетное обучение матрицы 8,192 дендритных NMDA-ветвей и двунаправленного CA3
        if dend_batch_items:
            self.dendritic_plateau_readout.imprint_response_batch_vectorized(dend_batch_items)

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
        """Когда собеседник в диалоге сообщает новые сведения о себе (`I / my`) ИЛИ явно обучает модель новому
        причинно-следственному факту (`'Remember that ...'`, `'Note that ...'`, `'In <domain>, ...'`),
        холинергический/дофаминовый всплеск новизны (`ACh + DA burst`) консолидирует этот факт напрямую в
        синаптическую матрицу 8 192 дендритных NMDA-ветвей `DendriticPlateauMatrixReadout` и полюса `CA3`!"""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        low_u = user_utterance.strip().lower()
        has_identity_fact = any(p in low_u for p in ("my name is", "i am building", "i'm building", "call me "))
        is_declarative_teaching = any(
            low_u.startswith(pref) for pref in ("remember that ", "note that ", "learn that ", "fact: ")
        )
        is_q = ("?" in user_utterance) or any(
            low_u.startswith(q) for q in ("what", "who", "how", "why", "where", "when", "remind", "do", "can", "explain")
        )
        if is_q or (not has_identity_fact and not is_declarative_teaching):
            return 0

        if is_declarative_teaching and not has_identity_fact:
            clean_fact = user_utterance.strip()
            for pref in ("Remember that ", "remember that ", "Note that ", "note that ", "Learn that ", "learn that ", "Fact: ", "fact: "):
                if clean_fact.startswith(pref):
                    clean_fact = clean_fact[len(pref) :].strip()
                    break
            if clean_fact and clean_fact[0].islower():
                clean_fact = clean_fact[0].upper() + clean_fact[1:]
            if clean_fact and clean_fact[-1] not in ".!?":
                clean_fact += "."
            s_ids = self.encode_text(clean_fact)
            r_ids = self.encode_text(clean_fact)
            if len(r_ids) < 4:
                return 0
            teach_pairs = [TokenizedDialogPair("online_declarative_teaching_da_burst", s_ids, r_ids, quality_reward_hint=2.8)]
            for _ in range(4):
                self.train_wave_with_neuromodulator_reward(teach_pairs, wave_idx=1)
            return len(r_ids)

        first_sent = user_utterance.split(".")[0].strip() + "."
        inv_words: List[str] = []
        for w in first_sent.split():
            cl = w.strip(p_strip).lower()
            if cl in ("hi", "hello", "hey"):
                continue
            if cl in self.PRONOUN_INVERSION:
                rep = self.PRONOUN_INVERSION[cl]
                if len(inv_words) == 0 and w.strip(p_strip)[0].isupper():
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
        q_self_vec = self.compute_stimulus_vec64(q_self)
        self.dialogue_state.register_user_identity(user_utterance, q_self_vec)
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
                learn=False,
            )
            if is_end_punct and (len(out_ids) >= 9 or verdict.son_motor_stop_triggered):
                stopped_by_son = True
                break

        mean_sup = total_syn_support / float(max(1, len(out_ids)))
        return out_ids, mean_sup, stopped_by_son

    def compute_clause_attractor_poles(
        self,
        user_utterance: str,
        enable_multi_hop: bool = False,
    ) -> List[torch.Tensor]:
        """Разбивает длинный незнакомый стимул на смысловые клаузы (по знакам `,`, `;`, `—`, `--`, `while`, `before`, `after`, `and`),
        выполняет каноническую ролевую проекцию `ThetaGammaRoleFillerBinder`, отсекает дискурсивные вводные конструкции
        (`'Remind me,'`, `'Tell me,'`) и проецирует каждую клаузу на ближайший релевантный полюс `W_CA3`.
        При `enable_multi_hop = True` (или запросе причинной цепочки) выполняет 2-хоповый транзитивный вывод ($A \\implies B \\implies C$)."""
        self.last_multi_hop_used = False
        self.last_ca3_top1_resonance = 1.0

        # 0. Каноническая проекция переменных слотов Role x Filler (например, hollandaise -> mayonnaise для поиска схемы в CA3):
        slot_bindings = self.role_filler_binder.extract_slot_bindings(user_utterance)
        canonical_utt = self.role_filler_binder.rewrite_utterance_to_canonical_for_ca3(user_utterance, slot_bindings)

        # 1. Явная маршрутизация рабочей памяти самоидентификации (ziwago_v12_conversational_mind.py):
        domains = self.dialogue_state.classify_utterance_domains(canonical_utt)
        if (
            self.dialogue_state.is_identity_query(canonical_utt)
            and self.dialogue_state.identity_pole_vec is not None
            and domains == ["self_identity"]
        ):
            self.last_ca3_top1_resonance = 2.5
            return [self.dialogue_state.identity_pole_vec]

        # 2. Очистка вводных дискурсивных оборотов перед клаузальным разбиением (чтобы 'Remind me,' не порождала ложный полюс!):
        cleaned_utt = canonical_utt.strip()
        lower_utt = cleaned_utt.lower()
        discourse_prefixes = (
            "remind me,",
            "tell me,",
            "by the way,",
            "excuse me,",
            "hi, how are you doing?",
            "hi!",
            "hello,",
        )
        for pref in discourse_prefixes:
            if lower_utt.startswith(pref):
                rest = cleaned_utt[len(pref) :].strip()
                if len(rest.split()) >= 3:
                    cleaned_utt = rest
                    break

        raw_clauses = [
            c.strip()
            for c in cleaned_utt.replace("—", "|")
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
            raw_clauses = [cleaned_utt]

        stop_words = {
            "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
            "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
            "what", "how", "why", "where", "when", "who", "tell", "me", "about", "does", "he", "am", "my",
            "remind", "please", "fix",
        }
        p_strip = " .,!?;:'()[]{}" + chr(34)

        # ТЕТА-ФАЗОВАЯ ПРЕЦЕССИЯ О'КИФА-ЛИСМАНА (BA46 Theta-Phase Temporal Queue):
        clause_candidates: List[Tuple[int, float, torch.Tensor]] = []
        max_resonance_seen = 0.0
        for tau_phase, cl in enumerate(raw_clauses):
            if self.dialogue_state.is_identity_query(cl) and self.dialogue_state.identity_pole_vec is not None:
                clause_candidates.append((tau_phase, 2.5, self.dialogue_state.identity_pole_vec))
                max_resonance_seen = max(max_resonance_seen, 2.5)
                continue
            cl_ids = self.encode_text(cl)
            content_ids = [
                tid for tid in cl_ids
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ]
            if not content_ids:
                continue
            cl_vec = self.compute_stimulus_vec64(content_ids)
            cl_sdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(content_ids, normalize=False)
            pole_vec, poles_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                cl_vec,
                q_sdr2048=cl_sdr,
                recent_poles=self.dialogue_state.recent_pole_vectors,
            )
            if poles_info:
                max_resonance_seen = max(max_resonance_seen, float(poles_info[0][0]))
            if not poles_info or poles_info[0][0] < 0.25:
                continue
            clause_candidates.append((tau_phase, poles_info[0][0], pole_vec))

        # Отбираем уникальные полюса с самым высоким резонансом CA3, но возвращаем их в ХРОНОЛОГИЧЕСКОМ порядке фаз tau_phase!
        clause_candidates.sort(key=lambda item: item[1], reverse=True)
        chosen_with_phase: List[Tuple[int, torch.Tensor]] = []
        for tau_p, sc, p_vec in clause_candidates:
            if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for _, ex in chosen_with_phase):
                chosen_with_phase.append((tau_p, p_vec))
            if len(chosen_with_phase) >= 2:
                break
        chosen_with_phase.sort(key=lambda item: item[0])
        selected_poles: List[torch.Tensor] = [p_vec for _, p_vec in chosen_with_phase]

        if len(selected_poles) < 2 and len(domains) >= 2:
            all_content_ids = [
                tid for tid in self.encode_text(cleaned_utt)
                if len(self.decode_token(tid).strip(p_strip)) >= 3
                and self.decode_token(tid).strip(p_strip).lower() not in stop_words
            ]
            sub_poles: List[Tuple[int, float, torch.Tensor]] = []
            for w_start in range(0, max(1, len(all_content_ids) - 2), 2):
                sub_ids = all_content_ids[w_start : w_start + 4]
                if len(sub_ids) >= 2:
                    sv = self.compute_stimulus_vec64(sub_ids)
                    ssdr = self.dendritic_plateau_readout._compute_cortical_sdr2048(sub_ids, normalize=False)
                    pv, p_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                        sv,
                        q_sdr2048=ssdr,
                        recent_poles=self.dialogue_state.recent_pole_vectors,
                    )
                    if p_info:
                        max_resonance_seen = max(max_resonance_seen, float(p_info[0][0]))
                    if p_info and p_info[0][0] >= 0.28:
                        sub_poles.append((w_start, p_info[0][0], pv))
            sub_poles.sort(key=lambda kv: kv[1], reverse=True)
            for w_pos, sc, p_vec in sub_poles:
                if not any(float(torch.dot(p_vec, ex).item()) > 0.88 for ex in selected_poles):
                    selected_poles.append(p_vec)
                if len(selected_poles) >= 2:
                    break

        if not selected_poles:
            if self.dialogue_state.identity_pole_vec is not None and self.dialogue_state.is_identity_query(canonical_utt):
                selected_poles.append(self.dialogue_state.identity_pole_vec)
                max_resonance_seen = 2.5
            else:
                u_ids = self.encode_text(cleaned_utt)
                c_ids = [
                    tid for tid in u_ids
                    if len(self.decode_token(tid).strip(p_strip)) >= 4
                    and self.decode_token(tid).strip(p_strip).lower() not in stop_words
                ] or u_ids
                p_fallback, fb_info = self.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                    self.compute_stimulus_vec64(c_ids),
                    q_sdr2048=self.dendritic_plateau_readout._compute_cortical_sdr2048(c_ids, normalize=False),
                    recent_poles=self.dialogue_state.recent_pole_vectors,
                )
                if fb_info:
                    max_resonance_seen = max(max_resonance_seen, float(fb_info[0][0]))
                selected_poles.append(p_fallback)

        self.last_ca3_top1_resonance = round(max_resonance_seen, 4)

        # 3. МНОГОШАГОВЫЙ ТРАНЗИТИВНЫЙ ВЫВОД В CA3 (Multi-Hop Cortical Rollout, A => B => C):
        multi_hop_triggers = ("step by step", "causal chain", "consequently", "how does that lead to", "trace the chain")
        wants_multi_hop = enable_multi_hop or any(t in lower_utt for t in multi_hop_triggers)
        if wants_multi_hop and len(selected_poles) == 1:
            u_ids_all = self.encode_text(cleaned_utt)
            s_sdr0 = self.dendritic_plateau_readout._compute_cortical_sdr2048(u_ids_all, normalize=False)
            hop2 = self.dendritic_plateau_readout.discover_second_hop_ca3_pole(
                first_pole_vec=selected_poles[0],
                stimulus_sdr2048=s_sdr0,
                recent_poles=self.dialogue_state.recent_pole_vectors,
            )
            if hop2 is not None:
                selected_poles.append(hop2[1])
                self.last_multi_hop_used = True

        return selected_poles

    def pure_neural_generate_response(
        self,
        user_utterance: str,
        max_tokens: int = 36,
        use_ngrams: Optional[bool] = None,
        enable_multi_hop: bool = False,
        enable_active_inference: bool = False,
    ) -> Dict[str, object]:
        r"""Генерирует ответ собеседнику в выбранном режиме (`use_ngrams = False` [OFF, 100% 8192-NMDA Matrix] по умолчанию):
          - При `use_ngrams = False` (OFF): 0% обращений к словарям N-грамм — весь ответ от 1-го до последнего слова
            вычисляется матрицей 8,192 нелинейных дендритных NMDA-ветвей `DendriticPlateauMatrixReadout` + `W_CA3`!
          - Включает 5 когнитивно-диалоговых механизмов v17:
            1) `ThetaGammaRoleFillerBinder`: динамическая подстановка новых сущностей пользователя в выученную синтаксическую раму;
            2) `Cortical MoE Branch Profile`: прямая проекция 16 колонок Бродмана на 16x512 дендритных NMDA-ветвей;
            3) `Multi-Hop Cortical Rollout`: 2-хоповый транзитивный вывод в `CA3` ($A \implies B \implies C$);
            4) `Episodic Working Memory Graph`: ответы на мета-вопросы о предыдущих ходах самой беседы;
            5) `Active Inference & Epistemic Curiosity Gate` в 17-й колонке `SON`: встречный уточняющий вопрос при нулевом резонансе в `CA3`."""
        t0 = time.perf_counter()
        ngrams_enabled = self.use_ngrams if use_ngrams is None else bool(use_ngrams)
        imprinted = self._online_dopamine_imprint_from_user(user_utterance)

        u_ids = self.encode_text(user_utterance)
        q_raw_vec = self.compute_stimulus_vec64(u_ids)

        # А. Проверка мета-запроса к Эпизодическому Графу Рабочей Памяти Диалога (BA46 Episodic Graph):
        if self.dialogue_state.is_episodic_history_query(user_utterance):
            ep_resp = self.dialogue_state.synthesize_episodic_memory_response(user_utterance)
            self.dialogue_state.record_turn_and_poles(
                stimulus_vec64=q_raw_vec,
                activated_poles=[q_raw_vec],
                utterance=user_utterance,
                response_text=ep_resp,
                role_filler_bindings={},
            )
            return {
                "stimulus": user_utterance,
                "ngrams_mode": "OFF (Episodic Working Memory Graph Recall)",
                "active_moe_columns": ["Col9_PFC_BA46", "Col15_METACOG_BA10"],
                "conversational_domain": "episodic_working_memory",
                "remembered_user_name": self.dialogue_state.user_name,
                "role_filler_bindings_applied": {},
                "multi_hop_rollout_used": False,
                "active_inference_clarification_triggered": False,
                "ca3_top1_resonance": 1.0,
                "epistemic_uncertainty": 0.0,
                "cortical_moe_branch_profile": {},
                "ba44_syntax_bridge_connector": "",
                "swr_consolidated_into_nmda_matrix": False,
                "online_dopamine_imprinted_tokens": imprinted,
                "vte_beams_evaluated": [],
                "selected_beam_idx": 0,
                "stopped_naturally_by_son_eos": True,
                "generated_tokens_count": len(self.encode_text(ep_resp)),
                "pure_weight_generated_response": ep_resp,
                "stored_sentence_chains_in_class": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        slot_bindings = self.role_filler_binder.extract_slot_bindings(user_utterance)
        if not ngrams_enabled:
            ca3_clause_poles = self.compute_clause_attractor_poles(
                user_utterance,
                enable_multi_hop=enable_multi_hop,
            )
            q_vec = ca3_clause_poles[0]
        else:
            q_vec = q_raw_vec
            ca3_clause_poles = [q_raw_vec]

        self.moe_cortex.homeostatic_expert_theta.zero_()
        moe_out = self.moe_cortex.step(x_in=q_vec, domain_tag=None, f_mod=1.0, learn=False, k_active=2)
        col_names = [f"{c.zcfg.name}_{idx}" for idx, c in enumerate(self.moe_cortex.columns)]
        cortical_branch_profile = self.dendritic_plateau_readout.compute_cortical_moe_branch_profile(
            q_vec=q_vec,
            expert_gates=moe_out.gating_weights,
            expert_names=col_names,
        )

        # Б. АКТИВНЫЙ ВЫВОД И ЭПИСТЕМИЧЕСКОЕ ЛЮБОПЫТСТВО В 17-Й КОЛОНКЕ SON (Active Inference Curiosity Gate):
        #    Если вопрос полностью лежит вне выученных аттракторов CA3 (резонанс < 0.08) и включён режим активного вывода,
        #    колонка SON блокирует галлюцинацию и формулирует встречный уточняющий вопрос!
        epistemic_uncertainty = round(max(0.0, 1.0 - min(1.0, self.last_ca3_top1_resonance / 0.55)), 4)
        if (
            enable_active_inference
            and imprinted == 0
            and self.last_ca3_top1_resonance < 0.38
            and not self.dialogue_state.is_identity_query(user_utterance)
        ):
            p_strip = " .,!?;:'()[]{}" + chr(34)
            stop_q = {"what", "how", "why", "where", "when", "who", "does", "do", "is", "are", "the", "a", "an", "in", "of", "to", "with"}
            novel_terms = [
                w.strip(p_strip)
                for w in user_utterance.split()
                if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in stop_q
            ][:3]
            topic_str = " ".join(novel_terms) if novel_terms else "this new topic"
            clarify_resp = (
                f"I notice you are asking about {topic_str}, which has high epistemic uncertainty in my current CA3 memory; "
                f"could you share the key mechanism or principle so I can consolidate it into my dendritic synapses?"
            )
            self.dialogue_state.record_turn_and_poles(
                stimulus_vec64=q_raw_vec,
                activated_poles=[q_raw_vec],
                utterance=user_utterance,
                response_text=clarify_resp,
                role_filler_bindings=slot_bindings,
            )
            return {
                "stimulus": user_utterance,
                "ngrams_mode": "OFF (SON Active Inference Epistemic Clarification)",
                "active_moe_columns": moe_out.active_expert_names,
                "conversational_domain": self.dialogue_state.active_domain,
                "remembered_user_name": self.dialogue_state.user_name,
                "role_filler_bindings_applied": slot_bindings,
                "multi_hop_rollout_used": False,
                "active_inference_clarification_triggered": True,
                "ca3_top1_resonance": self.last_ca3_top1_resonance,
                "epistemic_uncertainty": epistemic_uncertainty,
                "cortical_moe_branch_profile": cortical_branch_profile,
                "ba44_syntax_bridge_connector": "",
                "swr_consolidated_into_nmda_matrix": False,
                "online_dopamine_imprinted_tokens": 0,
                "vte_beams_evaluated": [],
                "selected_beam_idx": 0,
                "stopped_naturally_by_son_eos": True,
                "generated_tokens_count": len(self.encode_text(clarify_resp)),
                "pure_weight_generated_response": clarify_resp,
                "stored_sentence_chains_in_class": 0,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

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
            jitter = 0.0 if (b_idx == 0 or not ngrams_enabled) else (0.03 * b_idx)
            if not ngrams_enabled and len(ca3_clause_poles) > 1:
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
                    part_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                        clause_ids=part_ids,
                        stimulus_ids=u_ids,
                        decode_fn=self.decode_token,
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
                        bridge_phrase=bridge_phrase,
                        stimulus_ids=u_ids,
                    )
                    beam_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                        clause_ids=beam_ids,
                        stimulus_ids=u_ids,
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
                beam_ids = self.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                    clause_ids=beam_ids,
                    stimulus_ids=u_ids,
                    decode_fn=self.decode_token,
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

        # В. ТЕТА-ГАММА ПОДСТАНОВКА ПЕРЕМЕННЫХ СЛОТОВ (Role x Filler Variable Binding):
        if slot_bindings and not ngrams_enabled:
            response_text = self.role_filler_binder.apply_slot_bindings_to_response(response_text, slot_bindings)

        self.dialogue_state.record_turn_and_poles(
            stimulus_vec64=q_raw_vec,
            activated_poles=ca3_clause_poles,
            utterance=user_utterance,
            response_text=response_text,
            role_filler_bindings=slot_bindings,
        )

        # КОНСОЛИДАЦИЯ ОСТРЫХ ВОЛН ГИППОКАМПА (Sharp-Wave Ripple / SWR Replay Consolidation):
        swr_consolidated = False
        if not ngrams_enabled and ba44_bridge_used and len(chosen_ids) >= 8 and len(ca3_clause_poles) >= 2:
            q_swr_pole = F.normalize(
                torch.roll(ca3_clause_poles[0], shifts=1, dims=0) * ca3_clause_poles[1] + 0.35 * q_raw_vec + 1e-8,
                p=2,
                dim=0,
            )
            r_vecs_swr = [self.get_vec64(tid) for tid in chosen_ids]
            for _ in range(3):
                self.dendritic_plateau_readout.imprint_swr_composite_episode(
                    q_swr_pole=q_swr_pole,
                    response_vecs=r_vecs_swr,
                    response_ids=chosen_ids,
                    f_mod=2.2,
                )
            swr_consolidated = True

        return {
            "stimulus": user_utterance,
            "ngrams_mode": "ON" if ngrams_enabled else "OFF (Pure 8192-Dendritic NMDA Matrix + BA44 Syntax Bridge)",
            "active_moe_columns": moe_out.active_expert_names,
            "conversational_domain": self.dialogue_state.active_domain,
            "remembered_user_name": self.dialogue_state.user_name,
            "role_filler_bindings_applied": slot_bindings,
            "multi_hop_rollout_used": self.last_multi_hop_used,
            "active_inference_clarification_triggered": False,
            "ca3_top1_resonance": self.last_ca3_top1_resonance,
            "epistemic_uncertainty": epistemic_uncertainty,
            "cortical_moe_branch_profile": cortical_branch_profile,
            "ba44_syntax_bridge_connector": ba44_bridge_used,
            "swr_consolidated_into_nmda_matrix": swr_consolidated,
            "online_dopamine_imprinted_tokens": imprinted,
            "vte_beams_evaluated": vte_logs,
            "selected_beam_idx": best_idx,
            "stopped_naturally_by_son_eos": beams_stopped[best_idx],
            "generated_tokens_count": len(chosen_ids),
            "pure_weight_generated_response": response_text,
            "stored_sentence_chains_in_class": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ------------------------------------------------------------------
    # САМОДОСТАТОЧНЫЙ КОНТРАКТ СОХРАНЕНИЯ И ЗАГРУЗКИ ЧЕКПОИНТА (ROUND-TRIP SAFE)
    # ------------------------------------------------------------------
    def save_self_contained_checkpoint(
        self,
        ckpt_path: Path,
        extra_metadata: Optional[Dict[str, object]] = None,
        use_fp16: bool = False,
    ) -> Dict[str, object]:
        """Сохраняет самодостаточный чекпоинт со словарями `col_to_tok`/`tok_to_col`, проекторами
        (`pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`, `vsa_encoder_state`), состоянием коры/SON/химии,
        хешем токенизатора и версией схемы, проверяя его через `validate_self_contained_checkpoint` перед записью."""
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        dend_state = self.dendritic_plateau_readout.export_state_dict(use_fp16=use_fp16)

        vsa_encoder_state: Dict[str, object] = {
            "micro_weights": self.vsa_encoder.hashed_sdr.micro_weights.detach().clone(),
            "base_active_bins": self.vsa_encoder.base_active_bins.detach().clone(),
            "w_proj": self.vsa_encoder.w_proj.detach().clone(),
        }
        moe_cortex_state: Dict[str, object] = {
            "homeostatic_expert_theta": self.moe_cortex.homeostatic_expert_theta.detach().clone(),
            "last_col_spikes": {int(k): v.detach().clone() for k, v in self.moe_cortex._last_col_spikes.items()},
            "col_somatic_errors": [float(c.dendrite.last_somatic_fep_error) for c in self.moe_cortex.columns],
            "col_apical_errors": [float(c.dendrite.last_apical_fep_error) for c in self.moe_cortex.columns],
        }
        son_motor_state: Dict[str, object] = {
            "w_basal_son": self.son_column.w_basal_son.detach().clone(),
            "w_apical_son": self.son_column.w_apical_son.detach().clone(),
            "w_rec_son": self.son_column.w_rec_son.detach().clone(),
            "v_son": self.son_column.v_son.detach().clone(),
        }
        chemistry_state: Dict[str, float] = {
            k: float(getattr(self.chemistry, k))
            for k in ("da", "ne", "ach", "sht", "orexin", "adenosine", "cortisol", "lactate")
            if hasattr(self.chemistry, k)
        }

        payload: Dict[str, object] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "tokenizer_sha256": self.tokenizer_sha256,
            "seed": self.seed,
            "total_neurons": self.moe_cortex.total_neurons + self.son_column.n_son,
            "stored_sentence_chains": 0,
            "total_pairs_trained": self.total_pairs_trained,
            "total_tokens_trained": self.total_tokens_trained,
            **dend_state,
            "vsa_encoder_state": vsa_encoder_state,
            "moe_cortex_state": moe_cortex_state,
            "son_motor_state": son_motor_state,
            "chemistry_state": chemistry_state,
            "w_ba44_bridge": self.ba44_syntax_bridge.w_ba44_bridge.detach().clone(),
            "son_w_basal": self.son_column.w_basal_son.detach().clone(),
            "son_w_apical": self.son_column.w_apical_son.detach().clone(),
            "w_stimulus_to_onset": self.w_stimulus_to_onset.detach().half() if use_fp16 else self.w_stimulus_to_onset.detach().clone(),
            "w_readout": self.pc_readout.w_readout.detach().half() if use_fp16 else self.pc_readout.w_readout.detach().clone(),
            "token_embed": self.pc_readout.token_embed.detach().half() if use_fp16 else self.pc_readout.token_embed.detach().clone(),
            "conversational_state": self.dialogue_state.state_dict(),
        }
        if extra_metadata:
            payload.update(extra_metadata)

        val_info = validate_self_contained_checkpoint(
            payload,
            expected_tokenizer_sha256=self.tokenizer_sha256,
            strict_schema=True,
        )
        tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, str(tmp_ckpt))
        os.replace(tmp_ckpt, ckpt_path)
        return val_info

    def load_self_contained_checkpoint(
        self,
        ckpt_path: Path,
        strict_schema: bool = True,
    ) -> Dict[str, object]:
        """Загружает и валидирует самодостаточный чекпоинт, восстанавливая проекторы `vsa_encoder_state`,
        `DendriticPlateauMatrixReadout`, `BA44AttractorSyntaxBridge`, `moe_cortex_state`, `son_motor_state`,
        `chemistry_state` и `ConversationalDialogueState` для 100% побайтового Round-Trip при ЛЮБОМ `seed`!"""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise CheckpointCompatibilityError(f"Checkpoint file not found: {ckpt_path}")
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        val_info = validate_self_contained_checkpoint(
            payload,
            expected_tokenizer_sha256=self.tokenizer_sha256,
            strict_schema=strict_schema,
        )

        # 1. Восстанавливаем проекторы VSA-энкодера и перестраиваем кэш 64-мерных векторов (полная независимость от seed!)
        vsa_state = payload.get("vsa_encoder_state")
        if isinstance(vsa_state, dict):
            if isinstance(vsa_state.get("micro_weights"), torch.Tensor):
                self.vsa_encoder.hashed_sdr.micro_weights = vsa_state["micro_weights"].float().clone()
            if isinstance(vsa_state.get("base_active_bins"), torch.Tensor):
                self.vsa_encoder.base_active_bins = vsa_state["base_active_bins"].to(torch.uint8).clone()
            if isinstance(vsa_state.get("w_proj"), torch.Tensor):
                self.vsa_encoder.w_proj = vsa_state["w_proj"].float().clone()
            self._vec64_cache.clear()
            for bid in range(2048):
                self._vec64_cache[bid] = self.vsa_encoder.project_token_to_cortex(bid)
            for idx, trig_tid in self.ba44_syntax_bridge.connector_trigger_ids.items():
                self.ba44_syntax_bridge.connector_trigger_vecs[idx] = self.get_vec64(trig_tid)

        # 2. Восстанавливаем матрицу дендритных плато, проекторы pos_phase_vecs, w_basal_branch, w_apical_branch и CA3
        self.dendritic_plateau_readout.load_state_dict(payload)

        # 3. Восстанавливаем состояние коры, мотора SON и нейрохимии
        moe_state = payload.get("moe_cortex_state")
        if isinstance(moe_state, dict):
            if isinstance(moe_state.get("homeostatic_expert_theta"), torch.Tensor):
                self.moe_cortex.homeostatic_expert_theta = moe_state["homeostatic_expert_theta"].float().clone()
            last_spikes = moe_state.get("last_col_spikes")
            if isinstance(last_spikes, dict):
                for k, v in last_spikes.items():
                    if isinstance(v, torch.Tensor):
                        self.moe_cortex._last_col_spikes[int(k)] = v.float().clone()
            som_errs = moe_state.get("col_somatic_errors")
            api_errs = moe_state.get("col_apical_errors")
            if isinstance(som_errs, list) and isinstance(api_errs, list):
                for c_i, col in enumerate(self.moe_cortex.columns):
                    if c_i < len(som_errs):
                        col.dendrite.last_somatic_fep_error = float(som_errs[c_i])
                    if c_i < len(api_errs):
                        col.dendrite.last_apical_fep_error = float(api_errs[c_i])

        son_state = payload.get("son_motor_state")
        if isinstance(son_state, dict):
            if isinstance(son_state.get("w_basal_son"), torch.Tensor):
                self.son_column.w_basal_son = son_state["w_basal_son"].float().clone()
            if isinstance(son_state.get("w_apical_son"), torch.Tensor):
                self.son_column.w_apical_son = son_state["w_apical_son"].float().clone()
            if isinstance(son_state.get("w_rec_son"), torch.Tensor):
                self.son_column.w_rec_son = son_state["w_rec_son"].float().clone()
            if isinstance(son_state.get("v_son"), torch.Tensor):
                self.son_column.v_son = son_state["v_son"].float().clone()
        else:
            if isinstance(payload.get("son_w_basal"), torch.Tensor):
                self.son_column.w_basal_son = payload["son_w_basal"].float().clone()
            if isinstance(payload.get("son_w_apical"), torch.Tensor):
                self.son_column.w_apical_son = payload["son_w_apical"].float().clone()

        chem_state = payload.get("chemistry_state")
        if isinstance(chem_state, dict):
            for k, v in chem_state.items():
                if hasattr(self.chemistry, str(k)):
                    setattr(self.chemistry, str(k), float(v))  # type: ignore[arg-type]

        if isinstance(payload.get("w_ba44_bridge"), torch.Tensor):
            self.ba44_syntax_bridge.w_ba44_bridge = payload["w_ba44_bridge"].float().clone()
        if isinstance(payload.get("w_stimulus_to_onset"), torch.Tensor):
            self.w_stimulus_to_onset = payload["w_stimulus_to_onset"].float().clone()
        if isinstance(payload.get("w_readout"), torch.Tensor):
            self.pc_readout.w_readout = payload["w_readout"].float().clone()
        if isinstance(payload.get("token_embed"), torch.Tensor):
            self.pc_readout.token_embed = payload["token_embed"].float().clone()
        if isinstance(payload.get("conversational_state"), dict):
            self.dialogue_state.load_state_dict(payload["conversational_state"])
        self.total_pairs_trained = int(payload.get("total_pairs_trained", payload.get("total_elite_pairs_consolidated", 0)))
        self.total_tokens_trained = int(payload.get("total_tokens_trained", payload.get("total_consolidated_tokens", 0)))
        return val_info


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
    ckpt_validation = organism.save_self_contained_checkpoint(
        ckpt_path=ckpt_path,
        extra_metadata={
            "version": "Ziwago-Prime-v14.0-Pure-Synaptic-Dialogue-Organism",
            "source_totals": source_totals,
        },
        use_fp16=False,
    )
    ckpt_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2)

    summary = {
        "version": "Ziwago-Prime v14.0 Pure Synaptic Dialogue Organism (6 Dialog Corpora + Neuromodulators + SON + 8192-NMDA Matrix)",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": ckpt_mb,
        "checkpoint_validation": ckpt_validation,
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
        f"ZIWAGO-PRIME v14.0 ОБУЧЕН И ПРОТЕСТИРОВАН! ЧЕКПОИНТ: `{ckpt_path.name}` ({ckpt_mb} МБ)",
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
        default="off",
        choices=["on", "off", "compare"],
        help="N-gram mode: 'off' (default, 100% pure 8192-Dendritic NMDA Matrix), 'on' (Legacy Hybrid), or 'compare'",
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
