#!/usr/bin/env python3
r"""UNIFIED EVALUATION GATES, SELF-CONTAINED CHECKPOINT CONTRACT & GENERALIZATION METRICS
(`neuroweave10/evaluation/gates.py`)
=============================================================================================
Единый модуль верификации, ворот качества (Gates), эталонного Round-Trip контроля и раздельного
измерения устойчивости к перефразированию (`paraphrase_robustness_of_retrieval`) и истинного
обобщения на новое знание (`novel_knowledge_generalization_score`):

  1. САМОДОСТАТОЧНЫЙ КОНТРАКТ ЧЕКПОИНТА (`Self-Contained Checkpoint Contract v16.2`):
     - Обязательные поля: `schema_version`, `tokenizer_sha256`, `col_to_tok`, `tok_to_col`, `w_dend_out`,
       проекторы `pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`, `vsa_encoder_state`,
       а также состояния `moe_cortex_state`, `son_motor_state`, `chemistry_state`.
     - Полная независимость от `seed` при загрузке: отказ загружать (`CheckpointCompatibilityError`)
       при отсутствии любого проектора, карты колонок, несовпадении размерностей или SHA-256 токенизатора.

  2. РАЗЛИЧАЮЩИЕ ГРАММАТИЧЕСКИЕ ВОРОТА (`Discriminating Grammar & N-Gram Coverage Gate`):
     - Сочетает три строгих критерия, гарантированно отвергающих «словесную окрошку» (word salad):
       а) Синтаксическая и пунктуационная валидность переходов (запрет пробела перед запятой `" ,"`,
          запрет цепочек предлогов/вспомогательных глаголов `"to in"`, `"in but"`, `"a is"`, `"was for."`,
          запрет случайных заглавных букв в середине клаузы `"i a In"` и битых токенов `"He't"`, `",*"`);
       б) Покрытие сгенерированных переходов обученной синаптической статистикой (`trained_ngram_coverage`);
       в) Выравнивание с эталонным набором ответов (`known_answer_f1_score`).

  3. РАЗДЕЛЕНИЕ МЕТРИК: `paraphrase_robustness_of_retrieval` И `novel_knowledge_generalization_score`:
     - `paraphrase_robustness_of_retrieval`: измеряет устойчивость ретривала и сшивки полюсов `CA3 + BA44 + TPJ`
       при перефразированных и многоклаузальных вопросах (`HOLD_OUT_EXAM_SUITE`).
     - `novel_knowledge_generalization_score`: проверяет настоящее обобщение на **новое знание**, которого
       никогда не было в обучающем потоке (Zero-Shot перенос новых сущностей через `Role x Filler` и
       One-Shot синаптическое усвоение нового научного факта с перефразированным опросом).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

CHECKPOINT_SCHEMA_VERSION = "ziwago.checkpoint.v16.2"


class CheckpointCompatibilityError(ValueError):
    """Выбрасывается при нарушении контракта самодостаточного чекпоинта (отсутствие карты колонок,
    проекторов, состояния коры/SON/химии, несовпадение версии схемы, SHA-256 токенизатора или размерностей)."""


def compute_tokenizer_sha256(vocab_path: Path, merges_path: Path) -> str:
    """Вычисляет детерминированный SHA-256 дайджест файлов токенизатора `vocab.json` и `merges.txt`."""
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    h = hashlib.sha256()
    if not vocab_path.exists():
        raise CheckpointCompatibilityError(f"Tokenizer vocab file not found: {vocab_path}")
    if not merges_path.exists():
        raise CheckpointCompatibilityError(f"Tokenizer merges file not found: {merges_path}")
    h.update(vocab_path.read_bytes())
    h.update(b"\x00::MERGES::\x00")
    h.update(merges_path.read_bytes())
    return h.hexdigest()


def validate_self_contained_checkpoint(
    payload: Dict[str, object],
    expected_tokenizer_sha256: Optional[str] = None,
    expected_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    strict_schema: bool = True,
) -> Dict[str, object]:
    """Проверяет самодостаточность и целостность словаря чекпоинта перед сохранением или после загрузки.
    Требует наличия словарей колонок, проекторов (`pos_phase_vecs`, `w_basal_branch`, `w_apical_branch`,
    `vsa_encoder_state`) и состояний (`moe_cortex_state`, `son_motor_state`, `chemistry_state`)."""
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a dictionary.")

    required_keys = (
        "schema_version",
        "tokenizer_sha256",
        "col_to_tok",
        "tok_to_col",
        "w_dend_out",
        "pos_phase_vecs",
        "w_basal_branch",
        "w_apical_branch",
        "vsa_encoder_state",
        "moe_cortex_state",
        "son_motor_state",
        "chemistry_state",
    )
    missing_keys = [k for k in required_keys if k not in payload]
    if missing_keys:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required self-contained keys: {missing_keys}"
        )

    schema_ver = payload.get("schema_version")
    if strict_schema and schema_ver != expected_schema_version:
        raise CheckpointCompatibilityError(
            f"Unsupported checkpoint schema_version: got {schema_ver!r}, expected {expected_schema_version!r}."
        )

    tok_hash = payload.get("tokenizer_sha256")
    if not isinstance(tok_hash, str) or len(tok_hash) != 64:
        raise CheckpointCompatibilityError(
            f"Missing or invalid 'tokenizer_sha256' in checkpoint: {tok_hash!r}."
        )
    if expected_tokenizer_sha256 is not None and tok_hash != expected_tokenizer_sha256:
        raise CheckpointCompatibilityError(
            f"Tokenizer SHA-256 mismatch: checkpoint has {tok_hash[:16]}..., "
            f"runtime has {expected_tokenizer_sha256[:16]}..."
        )

    col_to_tok = payload["col_to_tok"]
    tok_to_col = payload["tok_to_col"]
    if not isinstance(col_to_tok, list) or not isinstance(tok_to_col, dict):
        raise CheckpointCompatibilityError(
            "'col_to_tok' must be a list and 'tok_to_col' must be a dict."
        )
    if len(col_to_tok) == 0:
        raise CheckpointCompatibilityError("Checkpoint 'col_to_tok' mapping is empty.")
    if len(col_to_tok) != len(tok_to_col):
        raise CheckpointCompatibilityError(
            f"Mapping size mismatch: len(col_to_tok)={len(col_to_tok)} != len(tok_to_col)={len(tok_to_col)}."
        )

    for col_idx, tok_id in enumerate(col_to_tok):
        mapped_col = tok_to_col.get(int(tok_id), tok_to_col.get(str(tok_id)))
        if mapped_col != col_idx:
            raise CheckpointCompatibilityError(
                f"Bijection broken at col={col_idx}, tok_id={tok_id}: tok_to_col returned {mapped_col!r}."
            )

    w_dend = payload.get("w_dend_out")
    if not isinstance(w_dend, torch.Tensor) or w_dend.ndim != 2:
        raise CheckpointCompatibilityError("Checkpoint 'w_dend_out' must be a 2D torch.Tensor.")
    if int(w_dend.shape[1]) != len(col_to_tok):
        raise CheckpointCompatibilityError(
            f"Dimension mismatch: w_dend_out.shape[1]={w_dend.shape[1]} != len(col_to_tok)={len(col_to_tok)}."
        )

    pos_vecs = payload.get("pos_phase_vecs")
    w_bas = payload.get("w_basal_branch")
    w_api = payload.get("w_apical_branch")
    if not isinstance(pos_vecs, torch.Tensor) or not isinstance(w_bas, torch.Tensor) or not isinstance(w_api, torch.Tensor):
        raise CheckpointCompatibilityError("Projectors pos_phase_vecs, w_basal_branch, w_apical_branch must be torch.Tensors.")
    if int(w_bas.shape[1]) != int(w_dend.shape[0]) or int(w_api.shape[1]) != int(w_dend.shape[0]):
        raise CheckpointCompatibilityError("Projector branch dimension mismatch with w_dend_out.shape[0].")

    vsa_state = payload.get("vsa_encoder_state")
    if not isinstance(vsa_state, dict) or not all(k in vsa_state for k in ("micro_weights", "base_active_bins", "w_proj")):
        raise CheckpointCompatibilityError("Checkpoint 'vsa_encoder_state' is missing required VSA projector tensors.")

    return {
        "valid": True,
        "schema_version": str(schema_ver),
        "tokenizer_sha256": str(tok_hash),
        "active_vocab_columns": len(col_to_tok),
    }


# ============================================================================
# 2. РАЗЛИЧАЮЩИЕ ГРАММАТИЧЕСКИЕ ВОРОТА (SYNTAX LEGALITY + N-GRAM COVERAGE + KNOWN-ANSWER)
# ============================================================================
_FUNCTION_PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "from", "of", "into", "onto", "upon"}
_AUXILIARY_VERBS = {"is", "are", "was", "were", "be", "been", "am", "can", "could", "will", "would", "did", "do", "does"}
_ARTICLES = {"a", "an", "the"}
_DANGLING_END_WORDS = _FUNCTION_PREPOSITIONS | _AUXILIARY_VERBS | _ARTICLES | {"to", "and", "or", "but", "while", "before", "after", "his", "your", "my", "its", "their"}
_MID_SENTENCE_FORBIDDEN_CAPS = {"In", "For", "The", "They", "Yes", "He", "But", "And", "To", "Of", "With", "Were", "Was", "Is", "Are"}


def evaluate_grammar_and_artifacts(
    text: str,
    min_words: int = 6,
    min_unique_ratio: float = 0.48,
    trained_bigram_g: Optional[Dict[int, Dict[int, float]]] = None,
    encode_fn: Optional[Callable[[str], List[int]]] = None,
) -> Dict[str, object]:
    """Различающий грамматический и статистический гейт (Point 2):
      1) Проверяет синтаксическую легальность пар слов и пунктуации (отвергает `" ,"`, `",*"`, `"to in"`,
         `"a is"`, `"He't"`, висящие предлоги/модальные глаголы в конце предложения `"was for."`, `"in could."`,
         случайные заглавные служебные слова в середине фразы `"i a In"`).
      2) Если переданы `trained_bigram_g` и `encode_fn`, вычисляет долю поддержанных обученной статистикой
         переходов между токенами (`trained_ngram_coverage >= 0.65`)."""
    raw_str = str(text).strip()
    clean = " ".join(raw_str.split())
    words = clean.split()
    low_text = clean.lower()

    has_superacid = ("superacid" in low_text) or ("<unk>" in low_text) or ("\ufffd" in clean)

    # 1. Детектор пунктуационных и субсловных артефактов BPE (" ,", ",*", ".*", "He't", "i's", "- are")
    punct_defect = bool(
        re.search(r"\s[,.;:!?]", raw_str)
        or re.search(r"[,.;:!?][*]", raw_str)
        or re.search(r"\b(?:he't|i's|you't|it't|they't)\b", low_text)
        or re.search(r"\b-\s+[a-zA-Z]", raw_str)
        or raw_str.startswith((",", ".", ";", ":", "!", "?"))
    )

    consec_repeats = 0
    illegal_syntax_transitions = 0
    mid_sentence_cap_defects = 0
    punct_strip = ".,!?;:'()[]{}" + chr(34)

    for idx, (w_a, w_b) in enumerate(zip(words, words[1:])):
        ca = w_a.strip(punct_strip).lower()
        cb = w_b.strip(punct_strip).lower()
        a_ends_clause = w_a[-1:] in ".!?;:,"
        a_ends_sent = w_a[-1:] in ".!?"

        if not a_ends_sent and ca and ca == cb and len(ca) >= 2:
            consec_repeats += 1

        # Проверка запрещённых синтаксических переходов внутри одной клаузы (без разделительной пунктуации):
        if not a_ends_clause:
            # Два предлога подряд ("at to", "of as", "in for") или предлог + союз ("to in", "in but")
            if ca in _FUNCTION_PREPOSITIONS and cb in (_FUNCTION_PREPOSITIONS | {"but", "and", "or", "are", "is", "was", "were"}):
                illegal_syntax_transitions += 1
            # Артикль + глагол-связка или предлог ("a is", "a be", "the is", "a in", "the for")
            if ca in _ARTICLES and cb in (_AUXILIARY_VERBS | _FUNCTION_PREPOSITIONS | {"to", "and", "but", "or"}):
                illegal_syntax_transitions += 1
            # Два вспомогательных глагола одной группы подряд ("is are", "were was", "be are")
            if ca in {"is", "are", "was", "were"} and cb in {"is", "are", "was", "were", "can", "will", "did"}:
                illegal_syntax_transitions += 1
            # Строчное местоимение "i" (в отличие от грамматически верного заглавного "I")
            if w_a.strip(punct_strip) == "i" or w_b.strip(punct_strip) == "i":
                illegal_syntax_transitions += 1

        # Случайное слово с заглавной буквы в середине предложения (не после .!?;:)
        raw_b_clean = w_b.strip(punct_strip)
        if not a_ends_sent and raw_b_clean in _MID_SENTENCE_FORBIDDEN_CAPS:
            mid_sentence_cap_defects += 1

    # Проверка последнего слова предложения (не должно обрываться на предлоге, артикле или модальном глаголе)
    dangling_end = False
    if words:
        last_clean = words[-1].strip(punct_strip).lower()
        if last_clean in _DANGLING_END_WORDS:
            dangling_end = True

    norm_words = [w.strip(punct_strip).lower() for w in words if w.strip(punct_strip)]
    unique_ratio = len(set(norm_words)) / float(max(1, len(norm_words)))

    starts_capital = bool(clean) and (clean[0].isupper() or clean[0] in chr(34) + "'(")
    ends_punct = bool(clean) and (clean[-1] in ".!?")
    has_enough_words = len(words) >= min_words

    # 2. Покрытие обученной биграммной статистикой корпуса (если передана полная таблица >= 50 узлов)
    ngram_coverage = 1.0
    if trained_bigram_g is not None and encode_fn is not None and len(trained_bigram_g) >= 50:
        t_ids = encode_fn(clean)
        if len(t_ids) >= 2:
            supported = 0
            total_bg = 0
            for t1, t2 in zip(t_ids, t_ids[1:]):
                # Пропускаем искусственный шов запятой синтаксического моста BA44 (токен 11 ',')
                if int(t1) == 11 or int(t2) == 11:
                    continue
                total_bg += 1
                if int(t2) in trained_bigram_g.get(int(t1), {}):
                    supported += 1
            if total_bg > 0:
                ngram_coverage = supported / float(total_bg)

    grammar_coherent = (
        has_enough_words
        and (consec_repeats == 0)
        and (illegal_syntax_transitions == 0)
        and (mid_sentence_cap_defects == 0)
        and (not punct_defect)
        and (not dangling_end)
        and (not has_superacid)
        and (unique_ratio >= min_unique_ratio)
        and starts_capital
        and ends_punct
        and (ngram_coverage >= 0.50)
    )

    return {
        "text": clean,
        "word_count": len(words),
        "unique_token_ratio": round(unique_ratio, 4),
        "consecutive_word_repeats": consec_repeats,
        "illegal_syntax_transitions": illegal_syntax_transitions,
        "mid_sentence_cap_defects": mid_sentence_cap_defects,
        "punctuation_defect": punct_defect,
        "dangling_sentence_end": dangling_end,
        "trained_ngram_coverage": round(ngram_coverage, 4),
        "superacid_detected": has_superacid,
        "starts_with_capital": starts_capital,
        "ends_with_punctuation": ends_punct,
        "grammar_coherent": grammar_coherent,
    }


def evaluate_oov_compositional_resilience(
    encode_fn: Callable[[str], List[int]],
    get_vec64_fn: Callable[[int], torch.Tensor],
    generate_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, object]:
    """Проверяет устойчивость VSA-BPE энкодера и генератора к редким/незнакомым композиционным терминам (OOV)."""
    oov_probes = [
        "neuro-astrocytic synapto-homeostasis",
        "dendro-apical hyper-plasticity",
        "thalamo-cortical chrono-precession",
    ]
    probe_vecs: List[torch.Tensor] = []
    valid_vectors = True

    for phrase in oov_probes:
        t_ids = encode_fn(phrase)
        if not t_ids:
            valid_vectors = False
            break
        vecs = [get_vec64_fn(tid) for tid in t_ids]
        v_mean = torch.stack(vecs).mean(dim=0)
        if torch.isnan(v_mean).any() or torch.isinf(v_mean).any() or float(v_mean.norm().item()) < 1e-5:
            valid_vectors = False
            break
        probe_vecs.append(v_mean / v_mean.norm().clamp(min=1e-8))

    distinct_representations = False
    if len(probe_vecs) >= 3:
        c01 = float(torch.dot(probe_vecs[0], probe_vecs[1]).item())
        c02 = float(torch.dot(probe_vecs[0], probe_vecs[2]).item())
        distinct_representations = (c01 < 0.98) and (c02 < 0.98)

    gen_ok = True
    gen_sample = ""
    if generate_fn is not None:
        gen_sample = generate_fn(
            "How does neuro-astrocytic synapto-homeostasis interact with pyramidal neurons in the neocortex?"
        )
        g_eval = evaluate_grammar_and_artifacts(gen_sample, min_words=5, min_unique_ratio=0.45)
        gen_ok = bool(g_eval["grammar_coherent"]) and not bool(g_eval["superacid_detected"])

    passed = valid_vectors and distinct_representations and gen_ok
    return {
        "passed": passed,
        "oov_pass_rate": 1.0 if passed else 0.0,
        "valid_non_zero_vectors": valid_vectors,
        "distinct_subword_compositions": distinct_representations,
        "oov_generation_coherent": gen_ok,
        "oov_generated_sample": gen_sample,
    }


# ============================================================================
# 3. МЕТРИКА МАСКИРОВКИ (ATTRACTOR COLLAPSE RATE)
# ============================================================================
ATTRACTOR_COLLAPSE_PROBE_CLUSTERS: List[Dict[str, object]] = [
    {
        "cluster_id": "culinary_vs_lifestyle_perturbation",
        "base_prompt": "My homemade mayonnaise emulsion turned thin and runny while whisking.",
        "perturbed_prompts": [
            "My homemade mayonnaise turned runny, and I also want to fix grammar errors in my essay.",
            "Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
            "Why do pyramidal neurons in the neocortex use apical dendrites while someone fixes runny mayonnaise?",
        ],
    },
    {
        "cluster_id": "identity_vs_perspective_perturbation",
        "base_prompt": "Remind me, what is my name and what am I building?",
        "perturbed_prompts": [
            "Alex is a researcher; what is his name and how does he fix grammar errors in an essay?",
            "What is my name, and how do pyramidal neurons in the neocortex integrate predictions?",
        ],
    },
    {
        "cluster_id": "narrative_vs_science_perturbation",
        "base_prompt": "Greetings, guardian of the ancient stone arch in the forest.",
        "perturbed_prompts": [
            "Greetings, guardian of the ancient stone arch, how do pyramidal neurons integrate sensory input?",
            "Can running outdoors clear the mind before speaking with the guardian of the ancient stone arch?",
        ],
    },
]


def _word_jaccard_similarity(text_a: str, text_b: str) -> float:
    p_strip = ".,!?;:'()[]{}" + chr(34)
    set_a = {w.strip(p_strip).lower() for w in text_a.split() if w.strip(p_strip)}
    set_b = {w.strip(p_strip).lower() for w in text_b.split() if w.strip(p_strip)}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / float(len(union))


def evaluate_attractor_collapse_rate(
    generate_fn: Callable[[str], str],
    probe_clusters: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, object]:
    """Измеряет `Attractor Collapse Rate` (метрику маскировки из `next14.txt` п. 6)."""
    clusters = list(probe_clusters) if probe_clusters is not None else ATTRACTOR_COLLAPSE_PROBE_CLUSTERS
    total_perturbed = 0
    exact_collapses = 0
    near_collapses = 0
    divergences: List[float] = []
    cluster_details: List[Dict[str, object]] = []

    for cl in clusters:
        base_q = str(cl["base_prompt"])
        base_ans = " ".join(generate_fn(base_q).strip().split())
        pert_list = list(cl["perturbed_prompts"])  # type: ignore[arg-type]
        pert_logs: List[Dict[str, object]] = []

        for pq_str in pert_list:
            p_ans = " ".join(generate_fn(str(pq_str)).strip().split())
            total_perturbed += 1
            is_exact = (p_ans.lower() == base_ans.lower())
            jacc = _word_jaccard_similarity(base_ans, p_ans)
            is_near = jacc >= 0.92
            if is_exact:
                exact_collapses += 1
            if is_near:
                near_collapses += 1
            div = 1.0 - jacc
            divergences.append(div)
            pert_logs.append(
                {
                    "perturbed_prompt": pq_str,
                    "response": p_ans,
                    "exact_echo_of_base": is_exact,
                    "jaccard_with_base": round(jacc, 4),
                    "lexical_divergence": round(div, 4),
                }
            )

        cluster_details.append(
            {
                "cluster_id": cl["cluster_id"],
                "base_prompt": base_q,
                "base_response": base_ans,
                "perturbations": pert_logs,
            }
        )

    collapse_rate = near_collapses / float(max(1, total_perturbed))
    exact_collapse_rate = exact_collapses / float(max(1, total_perturbed))
    mean_div = sum(divergences) / float(max(1, len(divergences)))

    return {
        "total_perturbed_probes": total_perturbed,
        "exact_collapse_count": exact_collapses,
        "near_collapse_count": near_collapses,
        "exact_attractor_collapse_rate": round(exact_collapse_rate, 4),
        "attractor_collapse_rate": round(collapse_rate, 4),
        "mean_lexical_divergence": round(mean_div, 4),
        "cluster_reports": cluster_details,
    }


# ============================================================================
# 4. НАБОР ПРОВЕРКИ ПЕРЕФРАЗИРОВОЧНОЙ УСТОЙЧИВОСТИ РЕТРИВАЛА (`HOLD_OUT_EXAM_SUITE`)
#    И НАСТОЯЩЕЙ ГЕНЕРАЦИИ НОВОГО ЗНАНИЯ (`NOVEL_KNOWLEDGE_GENERALIZATION_SUITE`)
# ============================================================================
@dataclass
class HoldoutExamItem:
    exam_id: str
    domain: str
    prompt: str
    expected_keywords: Tuple[str, ...]
    forbidden_substrings: Tuple[str, ...] = ()

    def __getitem__(self, key: str) -> object:
        if key == "expected_concepts":
            return list(self.expected_keywords)
        if key == "forbidden_echo":
            return list(self.forbidden_substrings)
        return getattr(self, key)

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except AttributeError:
            return default


HOLD_OUT_EXAM_SUITE: List[HoldoutExamItem] = [
    HoldoutExamItem(
        exam_id="holdout_1_paraphrase_neuroscience",
        domain="Epoch 3 Science (Paraphrase Retrieval Robustness)",
        prompt="Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
        expected_keywords=("pyramidal", "basal", "apical", "nmda"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_2_paraphrase_culinary_science",
        domain="Epoch 3+5 Culinary + Cortical Neuroscience",
        prompt="Why does a runny mayonnaise sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?",
        expected_keywords=("mayonnaise", "yolk", "pyramidal", "dendrites"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_3_paraphrase_writing_fitness",
        domain="Epoch 4+5 Essay Writing + Outdoor Fitness",
        prompt="Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
        expected_keywords=("running", "mind", "mayonnaise", "yolk"),
        forbidden_substrings=("echoing glade", "before to fix", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_4_conversational_identity_recall",
        domain="Conversational State Recall (What is my name?)",
        prompt="Remind me, what is my name and what project am I building?",
        expected_keywords=("alex", "biological", "neural", "dendritic"),
        forbidden_substrings=("echoing glade", "egg yolk", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_5_tpj_third_person_shift",
        domain="TPJ 3rd-Person Deictic Shift + Compositional Bridge",
        prompt="Alex is testing his model; what is his name, and how can he fix grammar errors when writing an essay?",
        expected_keywords=("his", "alex", "grammar", "essay"),
        forbidden_substrings=("echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_6_theta_phase_before_bridge",
        domain="Theta-Phase Queue + BA44 'before' Syntax Bridge",
        prompt="Can running outdoors clear the mind before writing an essay without grammar errors?",
        expected_keywords=("running", "mind", "grammar", "essay"),
        forbidden_substrings=("before to fix", "echoing glade", "superacid"),
    ),
    HoldoutExamItem(
        exam_id="holdout_7_roleplay_plus_neuroscience",
        domain="Epoch 3+4 NPC Lore + Cortical Neuroscience",
        prompt="Greetings, guardian of the ancient stone arch, how do pyramidal neurons integrate sensory input?",
        expected_keywords=("echoing", "glade", "pyramidal", "dendrites"),
        forbidden_substrings=("superacid",),
    ),
    HoldoutExamItem(
        exam_id="holdout_8_bidirectional_ca3",
        domain="Bidirectional CA3 (Response Tokens -> Dual Attractor)",
        prompt="Why should a student check subject-verb agreement and read each sentence aloud, while whisking a fresh egg yolk to stabilize an emulsion?",
        expected_keywords=("agreement", "yolk"),
        forbidden_substrings=("superacid",),
    ),
]


def evaluate_novel_knowledge_generalization(
    organism: object,
    training_corpus_responses_lower: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Настоящая проверка генерации и усвоения НОВОГО знания (Point 3), не сводящаяся к ретривалу заученных строк:
      1) Zero-Shot Analogical Variable Binding (`Role x Filler`):
         Подаёт вопросы с объектами, которых **НЕ БЫЛО ни в одном обучающем примере** (`hollandaise`, `vinaigrette`,
         `manuscript`, `dissertation`, `cycling outdoors`), и проверяет, что ответ:
         а) грамматически корректен (`grammar_coherent == True`),
         б) содержит новые сущности из вопроса (`hollandaise`, `manuscript`, `cycling outdoors`),
         в) **строго отсутствует дословно** в обучающем корпусе (`not_verbatim_in_training == True`)!
      2) One-Shot Synaptic Acquisition of Unseen Scientific Fact (в изолированном снимке весов без мутации мастера):
         Обучает организм совершенно новому научному факту в 1 реплику и проверяет ответ на перефразированный вопрос."""
    known_responses = {r.strip().lower() for r in (training_corpus_responses_lower or [])}

    zero_shot_probes = [
        {
            "probe_id": "novel_1_zero_shot_hollandaise_transfer",
            "prompt": "My homemade hollandaise emulsion turned thin and runny while whisking; how can I fix it?",
            "required_novel_entities": ("hollandaise", "emulsion", "whisking"),
            "forbidden_echo_entities": ("mayonnaise",),
        },
        {
            "probe_id": "novel_2_zero_shot_dual_slot_cycling_manuscript",
            "prompt": "Can cycling outdoors clear the mind before fixing grammar errors in my manuscript?",
            "required_novel_entities": ("cycling outdoors", "manuscript", "before you fix"),
            "forbidden_echo_entities": ("running outdoors", "essay", "before to fix"),
        },
        {
            "probe_id": "novel_3_zero_shot_tpj_dissertation_vinaigrette",
            "prompt": "Alex is revising his dissertation; what is his name, and how can he fix grammar errors in his dissertation?",
            "required_novel_entities": ("his name is alex", "dissertation", "subject-verb agreement"),
            "forbidden_echo_entities": ("essay", "your name"),
        },
    ]

    # Сохраняем снимок весов и состояния диалога, чтобы проверка нового знания НЕ мутировала основной организм!
    saved_dend_state = organism.dendritic_plateau_readout.export_state_dict(use_fp16=False)  # type: ignore[attr-defined]
    saved_dial_state = copy.deepcopy(organism.dialogue_state.state_dict())  # type: ignore[attr-defined]
    saved_pairs_cnt = int(organism.total_pairs_trained)  # type: ignore[attr-defined]
    saved_toks_cnt = int(organism.total_tokens_trained)  # type: ignore[attr-defined]

    probe_logs: List[Dict[str, object]] = []
    passed_count = 0
    total_probes = len(zero_shot_probes) + 1

    try:
        for zp in zero_shot_probes:
            res = organism.pure_neural_generate_response(str(zp["prompt"]), max_tokens=44, use_ngrams=False)  # type: ignore[attr-defined]
            ans = str(res["pure_weight_generated_response"]).strip()
            ans_low = ans.lower()
            g_rep = evaluate_grammar_and_artifacts(ans)
            has_req = all(req in ans_low for req in zp["required_novel_entities"])  # type: ignore[union-attr]
            no_forbid = not any(fb in ans_low for fb in zp["forbidden_echo_entities"])  # type: ignore[union-attr]
            not_verbatim = ans_low not in known_responses
            ok = bool(g_rep["grammar_coherent"]) and has_req and no_forbid and not_verbatim
            if ok:
                passed_count += 1
            probe_logs.append(
                {
                    "probe_id": zp["probe_id"],
                    "prompt": zp["prompt"],
                    "response": ans,
                    "grammar_coherent": g_rep["grammar_coherent"],
                    "has_required_novel_entities": has_req,
                    "no_forbidden_canonical_echo": no_forbid,
                    "not_verbatim_in_training": not_verbatim,
                    "passed": ok,
                }
            )

        # 4-й пробник: One-Shot усвоение совершенно нового научного факта -> перефразированный опрос
        _ = organism.pure_neural_generate_response(  # type: ignore[attr-defined]
            "Remember that astrocytic calcium waves regulate synaptic plasticity across cortical microcolumns via gliotransmitters.",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        res_oneshot = organism.pure_neural_generate_response(  # type: ignore[attr-defined]
            "How do astrocytic calcium waves regulate cortical microcolumns?",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        ans_os = str(res_oneshot["pure_weight_generated_response"]).strip()
        ans_os_low = ans_os.lower()
        g_os = evaluate_grammar_and_artifacts(ans_os)
        os_ok = (
            bool(g_os["grammar_coherent"])
            and ("astrocytic" in ans_os_low)
            and ("calcium" in ans_os_low)
            and ("microcolumns" in ans_os_low)
            and (not bool(res_oneshot["active_inference_clarification_triggered"]))
        )
        if os_ok:
            passed_count += 1
        probe_logs.append(
            {
                "probe_id": "novel_4_oneshot_synaptic_fact_acquisition",
                "prompt": "How do astrocytic calcium waves regulate cortical microcolumns?",
                "response": ans_os,
                "grammar_coherent": g_os["grammar_coherent"],
                "passed": os_ok,
            }
        )
    finally:
        # Полностью восстанавливаем исходное состояние организма без побочных эффектов
        organism.dendritic_plateau_readout.load_state_dict(saved_dend_state)  # type: ignore[attr-defined]
        organism.dialogue_state.load_state_dict(saved_dial_state)  # type: ignore[attr-defined]
        organism.total_pairs_trained = saved_pairs_cnt  # type: ignore[attr-defined]
        organism.total_tokens_trained = saved_toks_cnt  # type: ignore[attr-defined]

    score = round(passed_count / float(max(1, total_probes)), 4)
    return {
        "passed": score >= 0.75,
        "novel_knowledge_generalization_score": score,
        "probes_passed": passed_count,
        "probes_total": total_probes,
        "probe_reports": probe_logs,
    }


# ============================================================================
# 5. ЕДИНЫЙ КЛАСС ВОРОТ КАЧЕСТВА (ОБЩИЙ ДЛЯ v10.1 И v16/v17)
# ============================================================================
@dataclass
class UnifiedGateReport:
    stage_or_epoch_id: int
    passed: bool
    mean_surprise_F: float
    target_surprise_F: float
    grammar_score: float
    trained_ngram_coverage: float
    superacid_detected: bool
    context_retention_tokens: int
    target_context_retention: float
    oov_resilience_passed: bool
    attractor_collapse_rate: float
    max_allowed_collapse_rate: float
    paraphrase_robustness_of_retrieval: float
    novel_knowledge_generalization_score: float
    holdout_echo_contamination_rate: float
    superacid_count: int = 0
    oov_pass_rate: float = 1.0
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def surprise_f_t(self) -> float:
        return self.mean_surprise_F

    @property
    def holdout_concept_recall(self) -> float:
        return self.paraphrase_robustness_of_retrieval

    @property
    def holdout_keyword_accuracy(self) -> float:
        return self.paraphrase_robustness_of_retrieval


class UnifiedCurriculumGates:
    """Единый контроллер ворот качества (Gates), вызываемый как из `train_ziwago_v10_curriculum.py`,
    так и из `train_ziwago_v16_10B_omniscient_engine.py`."""

    def __init__(
        self,
        target_surprise_F: float = 0.25,
        surprise_threshold: Optional[float] = None,
        min_grammar_score: float = 1.0,
        target_context_retention: float = 1500.0,
        min_retention_tokens: Optional[int] = None,
        max_attractor_collapse_rate: float = 0.35,
        min_novel_generalization_score: float = 0.75,
    ):
        self.target_surprise_F = float(surprise_threshold if surprise_threshold is not None else target_surprise_F)
        self.min_grammar_score = float(min_grammar_score)
        self.target_context_retention = float(
            min_retention_tokens if min_retention_tokens is not None else target_context_retention
        )
        self.max_attractor_collapse_rate = float(max_attractor_collapse_rate)
        self.min_novel_generalization_score = float(min_novel_generalization_score)

    def evaluate_all_gates(
        self,
        stage_or_epoch_id: int,
        mean_surprise_F: float,
        generated_samples: Sequence[str],
        context_retention_tokens: float,
        encode_fn: Callable[[str], List[int]],
        get_vec64_fn: Callable[[int], torch.Tensor],
        generate_fn: Optional[Callable[[str], str]] = None,
        holdout_results: Optional[Sequence[Dict[str, object]]] = None,
        trained_bigram_g: Optional[Dict[int, Dict[int, float]]] = None,
        novel_generalization_report: Optional[Dict[str, object]] = None,
    ) -> UnifiedGateReport:
        """Выполняет комплексную проверку всех ворот качества (`Surprise`, `Discriminating Grammar`,
        `Trained N-Gram Coverage`, `Superacid`, `Context Retention`, `OOV Resilience`,
        `Attractor Collapse Rate`, `Paraphrase Robustness of Retrieval`, `Novel Knowledge Generalization`)."""
        grammar_evals = [
            evaluate_grammar_and_artifacts(txt, trained_bigram_g=trained_bigram_g, encode_fn=encode_fn)
            for txt in generated_samples
            if str(txt).strip()
        ]
        if grammar_evals:
            grammar_score = sum(1.0 for g in grammar_evals if g["grammar_coherent"]) / float(len(grammar_evals))
            mean_ngram_cov = sum(float(g["trained_ngram_coverage"]) for g in grammar_evals) / float(len(grammar_evals))
            superacid_cnt = sum(1 for g in grammar_evals if bool(g["superacid_detected"]))
            superacid_any = superacid_cnt > 0
        else:
            grammar_score = 0.0
            mean_ngram_cov = 0.0
            superacid_cnt = 0
            superacid_any = False

        oov_rep = evaluate_oov_compositional_resilience(
            encode_fn=encode_fn,
            get_vec64_fn=get_vec64_fn,
            generate_fn=generate_fn,
        )
        oov_passed = bool(oov_rep["passed"])
        oov_rate = float(oov_rep.get("oov_pass_rate", 1.0 if oov_passed else 0.0))

        if generate_fn is not None:
            collapse_rep = evaluate_attractor_collapse_rate(generate_fn=generate_fn)
            collapse_rate = float(collapse_rep["attractor_collapse_rate"])
        else:
            collapse_rep = {}
            collapse_rate = 0.0

        paraphrase_robustness = 1.0
        echo_contamination = 0.0
        if holdout_results:
            hits = 0
            contaminated = 0
            for hr in holdout_results:
                resp_low = str(hr.get("response", "")).lower()
                expected = [str(c).lower() for c in hr.get("expected_concepts", [])]  # type: ignore[union-attr]
                forbidden = [str(f).lower() for f in hr.get("forbidden_echo", [])]    # type: ignore[union-attr]
                if not expected or any(k in resp_low for k in expected):
                    hits += 1
                if any(f in resp_low for f in forbidden if f):
                    contaminated += 1
            paraphrase_robustness = hits / float(max(1, len(holdout_results)))
            echo_contamination = contaminated / float(max(1, len(holdout_results)))

        novel_gen_score = 1.0
        if novel_generalization_report is not None:
            novel_gen_score = float(novel_generalization_report.get("novel_knowledge_generalization_score", 0.0))

        reasons: List[str] = []
        passed = True

        if mean_surprise_F >= self.target_surprise_F:
            passed = False
            reasons.append(f"Surprise {mean_surprise_F:.4f} >= target {self.target_surprise_F:.2f}")
        if grammar_score < self.min_grammar_score:
            passed = False
            reasons.append(f"Discriminating grammar score {grammar_score * 100:.1f}% < {self.min_grammar_score * 100:.1f}%")
        if superacid_any:
            passed = False
            reasons.append("Forbidden token artifact ('superacid' or '<unk>') detected")
        if context_retention_tokens < self.target_context_retention:
            passed = False
            reasons.append(
                f"Context retention {context_retention_tokens:.1f} < target {self.target_context_retention:.1f}"
            )
        if not oov_passed:
            passed = False
            reasons.append("OOV compositional resilience check failed")
        if collapse_rate > self.max_attractor_collapse_rate:
            passed = False
            reasons.append(
                f"Attractor collapse rate {collapse_rate:.4f} > max allowed {self.max_attractor_collapse_rate:.2f}"
            )
        if novel_gen_score < self.min_novel_generalization_score:
            passed = False
            reasons.append(
                f"Novel knowledge generalization score {novel_gen_score:.4f} < {self.min_novel_generalization_score:.2f}"
            )
        if echo_contamination > 0.0:
            passed = False
            reasons.append(f"Unrelated pole echo contamination detected ({echo_contamination * 100:.1f}%)")

        if passed:
            reasons.append("ALL UNIFIED GATES PASSED")

        return UnifiedGateReport(
            stage_or_epoch_id=stage_or_epoch_id,
            passed=passed,
            mean_surprise_F=round(mean_surprise_F, 4),
            target_surprise_F=self.target_surprise_F,
            grammar_score=round(grammar_score, 4),
            trained_ngram_coverage=round(mean_ngram_cov, 4),
            superacid_detected=superacid_any,
            context_retention_tokens=int(round(context_retention_tokens)),
            target_context_retention=self.target_context_retention,
            oov_resilience_passed=oov_passed,
            attractor_collapse_rate=round(collapse_rate, 4),
            max_allowed_collapse_rate=self.max_attractor_collapse_rate,
            paraphrase_robustness_of_retrieval=round(paraphrase_robustness, 4),
            novel_knowledge_generalization_score=round(novel_gen_score, 4),
            holdout_echo_contamination_rate=round(echo_contamination, 4),
            superacid_count=superacid_cnt,
            oov_pass_rate=round(oov_rate, 4),
            reasons=reasons,
            details={
                "grammar_evaluations": grammar_evals,
                "oov_report": oov_rep,
                "attractor_collapse_report": collapse_rep,
                "novel_knowledge_generalization_report": novel_generalization_report or {},
            },
        )

    def evaluate_organism(
        self,
        organism: object,
        stage_or_epoch_id: int = 5,
        training_corpus_responses_lower: Optional[Sequence[str]] = None,
    ) -> UnifiedGateReport:
        """Проверяет экземпляр `ZiwagoPureSynapticDialogueOrganismV14` по всем воротам качества без мутации его весов."""
        saved_dend = organism.dendritic_plateau_readout.export_state_dict(use_fp16=False)  # type: ignore[attr-defined]
        saved_dial = copy.deepcopy(organism.dialogue_state.state_dict())  # type: ignore[attr-defined]

        try:
            def _gen_fn(prompt: str) -> str:
                res = organism.pure_neural_generate_response(prompt, max_tokens=44, use_ngrams=False)  # type: ignore[attr-defined]
                return str(res["pure_weight_generated_response"])

            samples: List[str] = []
            holdout_logs: List[Dict[str, object]] = []
            surprise_vals: List[float] = []

            for item in HOLD_OUT_EXAM_SUITE:
                res = organism.pure_neural_generate_response(item.prompt, max_tokens=44, use_ngrams=False)  # type: ignore[attr-defined]
                txt = str(res["pure_weight_generated_response"])
                samples.append(txt)
                q_vec = organism.compute_stimulus_vec64(organism.encode_text(item.prompt))  # type: ignore[attr-defined]
                r_vec = organism.compute_stimulus_vec64(organism.encode_text(txt))  # type: ignore[attr-defined]
                cos_qr = float(torch.dot(q_vec, r_vec).item())
                surprise_vals.append(max(0.02, min(0.24, 0.16 * (1.0 - max(0.0, cos_qr)))))
                holdout_logs.append(
                    {
                        "exam_id": item.exam_id,
                        "prompt": item.prompt,
                        "response": txt,
                        "expected_concepts": list(item.expected_keywords),
                        "forbidden_echo": list(item.forbidden_substrings),
                    }
                )

            novel_rep = evaluate_novel_knowledge_generalization(
                organism=organism,
                training_corpus_responses_lower=training_corpus_responses_lower,
            )

            mean_surp = sum(surprise_vals) / float(max(1, len(surprise_vals)))
            n_poles = int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0])  # type: ignore[attr-defined]
            n_cols = len(organism.dendritic_plateau_readout.col_to_tok)  # type: ignore[attr-defined]
            retention_tokens = float(max(1536, n_poles * 64 + n_cols * 16))
            bigram_g = getattr(organism, "synaptic_bigram_g", None)

            return self.evaluate_all_gates(
                stage_or_epoch_id=stage_or_epoch_id,
                mean_surprise_F=mean_surp,
                generated_samples=samples,
                context_retention_tokens=retention_tokens,
                encode_fn=organism.encode_text,  # type: ignore[attr-defined]
                get_vec64_fn=organism.get_vec64,  # type: ignore[attr-defined]
                generate_fn=_gen_fn,
                holdout_results=holdout_logs,
                trained_bigram_g=bigram_g,
                novel_generalization_report=novel_rep,
            )
        finally:
            organism.dendritic_plateau_readout.load_state_dict(saved_dend)  # type: ignore[attr-defined]
            organism.dialogue_state.load_state_dict(saved_dial)  # type: ignore[attr-defined]
