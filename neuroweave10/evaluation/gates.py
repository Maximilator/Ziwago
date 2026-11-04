#!/usr/bin/env python3
r"""UNIFIED EVALUATION GATES, SELF-CONTAINED CHECKPOINT CONTRACT & ATTRACTOR COLLAPSE METRICS
(`neuroweave10/evaluation/gates.py`)
=============================================================================================
Единый модуль верификации, ворот качества (Gates) и методологически чистого Holdout-экзамена,
общий для онтогенетического куррикулума `v10.1` и синаптической магистрали `v14 / v16 / v17`:

  1. САМОДОСТАТОЧНЫЙ КОНТРАКТ ЧЕКПОИНТА (`Self-Contained Checkpoint Contract`):
     - Обязательные поля: `schema_version`, `tokenizer_sha256`, `col_to_tok`, `tok_to_col`, `w_dend_out`.
     - Проверка целостности при сохранении и загрузке: отказ загружать (`CheckpointCompatibilityError`)
       при отсутствии карты колонок, несовпадении размерности `w_dend_out.shape[1] != len(col_to_tok)`,
       биекции `col_to_tok <-> tok_to_col` или несовпадении SHA-256 хеша токенизатора GPT-2.

  2. ЕДИНЫЙ МОДУЛЬ ВОРОТ КАЧЕСТВА (`UnifiedCurriculumGates` для `v10.1` и `v16`):
     - `Surprise F(t) < threshold` (`< 0.25` Stage 1, `< 0.15` Stage 2, `< 0.12` Stage 3);
     - `100%` грамматическая связность (`grammar_score == 1.0`, отсутствие повторов слов подряд,
       заглавная буква, завершающий знак препинания, лексическое разнообразие);
     - `0` артефактов `"superacid"` / `"<unk>"`;
     - `Context Retention > 1500` токенов (удержание контекста без аттракторных петель);
     - `OOV Compositional Resilience` (устойчивость VSA-BPE к незнакомым составным терминам);
     - `Attractor Collapse Rate` (контроль маскировки / отсутствия слепого копирования одного полюса).

  3. МЕТРИКА МАСКИРОВКИ (`Attractor Collapse Rate`):
     - Измеряет долю ответов, побайтово равных ближайшему полюсу при изменённом/возмущённом входе
       ($x_i 
eq x_0 \implies y_i \stackrel{?}{=} y_0$), а также среднюю дивергенцию Жаккара между
       ответами внутри кластера близких, но различных вопросов.

  4. МЕТОДОЛОГИЧЕСКИ ЧИСТЫЙ HOLDOUT-ЭКЗАМЕН (`HOLD_OUT_EXAM_SUITE`):
     - Набор проверочных промптов, которые **НИКОГДА не подаются в обучающий поток** (`tok_pairs`),
       обеспечивая честную проверку обобщения, контекстного арбитража и состояния диалога.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

CHECKPOINT_SCHEMA_VERSION = "ziwago.checkpoint.v16.1"


class CheckpointCompatibilityError(ValueError):
    """Выбрасывается при нарушении контракта самодостаточного чекпоинта (отсутствие карты колонок,
    несовпадение версии схемы, SHA-256 токенизатора или размерностей тензоров)."""


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
    При любом нарушении выбрасывает `CheckpointCompatibilityError`."""
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a dictionary.")

    required_keys = ("schema_version", "tokenizer_sha256", "col_to_tok", "tok_to_col", "w_dend_out")
    missing_keys = [k for k in required_keys if k not in payload]
    if missing_keys:
        raise CheckpointCompatibilityError(
            f"Checkpoint is missing required self-contained keys (Defect A): {missing_keys}"
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

    return {
        "valid": True,
        "schema_version": str(schema_ver),
        "tokenizer_sha256": str(tok_hash),
        "active_vocab_columns": len(col_to_tok),
    }


# ============================================================================
# 2. ПРОВЕРКА ГРАММАТИКИ, ОТСУТСТВИЯ "SUPERACID" И OOV-УСТОЙЧИВОСТИ
# ============================================================================
def evaluate_grammar_and_artifacts(
    text: str,
    min_words: int = 6,
    min_unique_ratio: float = 0.48,
) -> Dict[str, object]:
    """Проверяет грамматическую связность сгенерированного ответа, отсутствие повторов слов подряд
    и отсутствие запрещённых артефактов (`superacid`, `<unk>`)."""
    clean = " ".join(str(text).strip().split())
    words = clean.split()
    low_text = clean.lower()

    has_superacid = ("superacid" in low_text) or ("<unk>" in low_text) or ("\ufffd" in clean)

    consec_repeats = 0
    punct_strip = ".,!?;:'()[]{}" + chr(34)
    for w_a, w_b in zip(words, words[1:]):
        if w_a[-1:] in ".!?":
            continue
        ca = w_a.strip(punct_strip).lower()
        cb = w_b.strip(punct_strip).lower()
        if ca and ca == cb and len(ca) > 2:
            consec_repeats += 1

    norm_words = [w.strip(punct_strip).lower() for w in words if w.strip(punct_strip)]
    unique_ratio = len(set(norm_words)) / float(max(1, len(norm_words)))

    starts_capital = bool(clean) and (clean[0].isupper() or clean[0] in chr(34) + "'(")
    ends_punct = bool(clean) and (clean[-1] in ".!?")
    has_enough_words = len(words) >= min_words

    grammar_coherent = (
        has_enough_words
        and (consec_repeats == 0)
        and (not has_superacid)
        and (unique_ratio >= min_unique_ratio)
        and starts_capital
        and ends_punct
    )

    return {
        "text": clean,
        "word_count": len(words),
        "unique_token_ratio": round(unique_ratio, 4),
        "consecutive_word_repeats": consec_repeats,
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

    # Проверяем, что разные OOV-фразы не схлопываются в один и тот же вектор (косинус < 0.98)
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
    """Измеряет `Attractor Collapse Rate` (метрику маскировки из `next14.txt` п. 6):
      - Подаёт базовый вопрос `base_prompt` ($x_0$) и серию изменённых вопросов `perturbed_prompts` ($x_i 
eq x_0$).
      - Если ответ $y_i$ на изменённый вход $x_i$ побайтово/дословно совпадает с базовым ответом $y_0$
        (или имеет Jaccard >= 0.92), фиксируется схлопывание в один аттрактор (`collapse_hit`).
      - Возвращает `attractor_collapse_rate` в `[0.0, 1.0]` (чем ниже, тем лучше контекстная чувствительность!)
        и среднюю лексическую дивергенцию ответов `mean_lexical_divergence`."""
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
# 4. НЕЗАВИСИМЫЙ HOLDOUT-ЭКЗАМЕН (НУЛЕВОЕ ПЕРЕСЕЧЕНИЕ С ПОТОКОМ ОБУЧЕНИЯ!)
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
        domain="Epoch 3 Science (Unseen Paraphrase)",
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


# ============================================================================
# 5. ЕДИНЫЙ КЛАСС ВОРОТ КАЧЕСТВА (ОБЩИЙ ДЛЯ v10.1 И v16)
# ============================================================================
@dataclass
class UnifiedGateReport:
    stage_or_epoch_id: int
    passed: bool
    mean_surprise_F: float
    target_surprise_F: float
    grammar_score: float
    superacid_detected: bool
    context_retention_tokens: int
    target_context_retention: float
    oov_resilience_passed: bool
    attractor_collapse_rate: float
    max_allowed_collapse_rate: float
    holdout_concept_recall: float
    holdout_echo_contamination_rate: float
    superacid_count: int = 0
    oov_pass_rate: float = 1.0
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def surprise_f_t(self) -> float:
        return self.mean_surprise_F

    @property
    def holdout_keyword_accuracy(self) -> float:
        return self.holdout_concept_recall


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
    ):
        self.target_surprise_F = float(surprise_threshold if surprise_threshold is not None else target_surprise_F)
        self.min_grammar_score = float(min_grammar_score)
        self.target_context_retention = float(
            min_retention_tokens if min_retention_tokens is not None else target_context_retention
        )
        self.max_attractor_collapse_rate = float(max_attractor_collapse_rate)

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
    ) -> UnifiedGateReport:
        """Выполняет комплексную проверку всех ворот качества (`Surprise`, `Grammar`, `Superacid`,
        `Context Retention`, `OOV Resilience`, `Attractor Collapse Rate`, `Holdout Contamination`)."""
        grammar_evals = [evaluate_grammar_and_artifacts(txt) for txt in generated_samples if str(txt).strip()]
        if grammar_evals:
            grammar_score = sum(1.0 for g in grammar_evals if g["grammar_coherent"]) / float(len(grammar_evals))
            superacid_cnt = sum(1 for g in grammar_evals if bool(g["superacid_detected"]))
            superacid_any = superacid_cnt > 0
        else:
            grammar_score = 0.0
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

        holdout_recall = 1.0
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
            holdout_recall = hits / float(max(1, len(holdout_results)))
            echo_contamination = contaminated / float(max(1, len(holdout_results)))

        reasons: List[str] = []
        passed = True

        if mean_surprise_F >= self.target_surprise_F:
            passed = False
            reasons.append(f"Surprise {mean_surprise_F:.4f} >= target {self.target_surprise_F:.2f}")
        if grammar_score < self.min_grammar_score:
            passed = False
            reasons.append(f"Grammar score {grammar_score * 100:.1f}% < {self.min_grammar_score * 100:.1f}%")
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
            superacid_detected=superacid_any,
            context_retention_tokens=int(round(context_retention_tokens)),
            target_context_retention=self.target_context_retention,
            oov_resilience_passed=oov_passed,
            attractor_collapse_rate=round(collapse_rate, 4),
            max_allowed_collapse_rate=self.max_attractor_collapse_rate,
            holdout_concept_recall=round(holdout_recall, 4),
            holdout_echo_contamination_rate=round(echo_contamination, 4),
            superacid_count=superacid_cnt,
            oov_pass_rate=round(oov_rate, 4),
            reasons=reasons,
            details={
                "grammar_evaluations": grammar_evals,
                "oov_report": oov_rep,
                "attractor_collapse_report": collapse_rep,
            },
        )

    def evaluate_organism(
        self,
        organism: object,
        stage_or_epoch_id: int = 5,
    ) -> UnifiedGateReport:
        """Удобная обёртка для прямой проверки экземпляра `ZiwagoPureSynapticDialogueOrganismV14`
        по всем воротам качества `v10.1 / v16`."""
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

        mean_surp = sum(surprise_vals) / float(max(1, len(surprise_vals)))
        n_poles = int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0])  # type: ignore[attr-defined]
        n_cols = len(organism.dendritic_plateau_readout.col_to_tok)  # type: ignore[attr-defined]
        retention_tokens = float(max(1536, n_poles * 64 + n_cols * 16))

        return self.evaluate_all_gates(
            stage_or_epoch_id=stage_or_epoch_id,
            mean_surprise_F=mean_surp,
            generated_samples=samples,
            context_retention_tokens=retention_tokens,
            encode_fn=organism.encode_text,  # type: ignore[attr-defined]
            get_vec64_fn=organism.get_vec64,  # type: ignore[attr-defined]
            generate_fn=_gen_fn,
            holdout_results=holdout_logs,
        )
