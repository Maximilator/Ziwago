"""Unified Evaluation Gates, Self-Contained Checkpoint Contract, and Holdout Benchmarks for Ziwago."""

from neuroweave10.evaluation.gates import (
    CHECKPOINT_SCHEMA_VERSION,
    HOLD_OUT_EXAM_SUITE,
    ATTRACTOR_COLLAPSE_PROBE_CLUSTERS,
    CheckpointCompatibilityError,
    UnifiedGateReport,
    UnifiedCurriculumGates,
    compute_tokenizer_sha256,
    validate_self_contained_checkpoint,
    evaluate_grammar_and_artifacts,
    evaluate_attractor_collapse_rate,
    evaluate_oov_compositional_resilience,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "HOLD_OUT_EXAM_SUITE",
    "ATTRACTOR_COLLAPSE_PROBE_CLUSTERS",
    "CheckpointCompatibilityError",
    "UnifiedGateReport",
    "UnifiedCurriculumGates",
    "compute_tokenizer_sha256",
    "validate_self_contained_checkpoint",
    "evaluate_grammar_and_artifacts",
    "evaluate_attractor_collapse_rate",
    "evaluate_oov_compositional_resilience",
]
