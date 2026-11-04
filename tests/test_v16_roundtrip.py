#!/usr/bin/env python3
"""
CI Test Suite for Ziwago v14/v16/v17 Self-Contained Checkpoints, Context Arbitrage,
BA44 Grammatical Bridge Seam Suppression, Unified Curriculum Gates, and the 5 Cognitive
Dialogue Mechanisms (Theta-Gamma Role-Filler Binding, 16-Column MoE NMDA Branch Profile,
Multi-Hop Transitive Rollout, Episodic Working Memory Graph, and SON Active Inference).
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.evaluation.gates import (
    HOLD_OUT_EXAM_SUITE,
    CheckpointCompatibilityError,
    UnifiedCurriculumGates,
    validate_self_contained_checkpoint,
)
from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
    _bpe_tokenize_raw_pairs_worker,
    _init_bpe_worker,
)
from train_ziwago_v16_10B_omniscient_engine import (
    FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH,
)


def _build_trained_test_organism() -> ZiwagoPureSynapticDialogueOrganismV14:
    vocab_p = str(ROOT / "data" / "gpt2" / "vocab.json")
    merges_p = str(ROOT / "data" / "gpt2" / "merges.txt")
    _init_bpe_worker(vocab_p, merges_p)

    organism = ZiwagoPureSynapticDialogueOrganismV14(
        root=ROOT,
        workers=4,
        cpu_threads=4,
        seed=42,
    )
    organism.use_ngrams = False

    all_lessons = [
        lesson
        for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values()
        for lesson in ep_lessons
    ]
    tok_lessons = _bpe_tokenize_raw_pairs_worker(all_lessons)
    lesson_objs = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in tok_lessons]

    for w_i in range(1, 5):
        organism.train_wave_with_neuromodulator_reward(lesson_objs, wave_idx=w_i)

    # Register online user identity ("Alex")
    _ = organism.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )
    return organism


class TestZiwagoV16RoundTripAndGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.organism = _build_trained_test_organism()

    def test_v16_save_load_byte_for_byte_roundtrip(self) -> None:
        """Round-Trip CI: save -> load into fresh organism -> byte-for-byte identical generation."""
        probes = [
            "Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
            "Remind me, what is my name and what project am I building?",
            "Alex is testing his model; what is his name, and how can he fix grammar errors when writing an essay?",
            "Before fixing a runny mayonnaise emulsion, how does running outdoors help clear the mind?",
            "Can running outdoors clear the mind before writing an essay without grammar errors?",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ziwago_roundtrip_test.pt"
            val_info = self.organism.save_self_contained_checkpoint(ckpt_path, use_fp16=False)
            self.assertTrue(val_info["valid"])
            self.assertGreater(int(val_info["active_vocab_columns"]), 20)

            expected_outputs = [
                self.organism.pure_neural_generate_response(q, max_tokens=44, use_ngrams=False)[
                    "pure_weight_generated_response"
                ]
                for q in probes
            ]

            reloaded = ZiwagoPureSynapticDialogueOrganismV14(
                root=ROOT,
                workers=2,
                cpu_threads=4,
                seed=42,
            )
            load_info = reloaded.load_self_contained_checkpoint(ckpt_path, strict_schema=True)
            self.assertTrue(load_info["valid"])

            for q, exp_out in zip(probes, expected_outputs):
                actual_out = reloaded.pure_neural_generate_response(q, max_tokens=44, use_ngrams=False)[
                    "pure_weight_generated_response"
                ]
                self.assertEqual(
                    actual_out,
                    exp_out,
                    f"Round-trip mismatch for {q!r}: {actual_out!r} != {exp_out!r}",
                )

    def test_v16_checkpoint_rejects_broken_maps_and_mismatch(self) -> None:
        """Verifies that missing col_to_tok, broken bijection, or wrong tokenizer hash raises CheckpointCompatibilityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "valid_ckpt.pt"
            self.organism.save_self_contained_checkpoint(ckpt_path, use_fp16=False)
            base_payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

            # 1. Missing col_to_tok (Defect A from next14.txt)
            bad_no_map = copy.deepcopy(base_payload)
            del bad_no_map["col_to_tok"]
            with self.assertRaisesRegex(CheckpointCompatibilityError, "missing required self-contained keys"):
                validate_self_contained_checkpoint(bad_no_map)

            # 2. Broken bijection between col_to_tok and tok_to_col
            bad_bijection = copy.deepcopy(base_payload)
            first_tok = bad_bijection["col_to_tok"][0]
            bad_bijection["tok_to_col"][first_tok] = 999999
            with self.assertRaisesRegex(CheckpointCompatibilityError, "Bijection broken"):
                validate_self_contained_checkpoint(bad_bijection)

            # 3. Wrong tokenizer SHA-256
            with self.assertRaisesRegex(CheckpointCompatibilityError, "Tokenizer SHA-256 mismatch"):
                validate_self_contained_checkpoint(base_payload, expected_tokenizer_sha256="0" * 64)

            # 4. Wrong schema version
            bad_schema = copy.deepcopy(base_payload)
            bad_schema["schema_version"] = "ziwago-legacy-v0.0"
            with self.assertRaisesRegex(CheckpointCompatibilityError, "Unsupported checkpoint schema_version"):
                validate_self_contained_checkpoint(bad_schema, strict_schema=True)

    def test_v16_context_arbitrage_and_ba44_grammar_seam(self) -> None:
        """Verifies no 'Echoing Glade' leakage on identity queries and no 'before to fix' grammatical seam."""
        # 1. Identity query must not pull in NPC 'Echoing Glade'
        res_id = self.organism.pure_neural_generate_response(
            "Remind me, what is my name and what project am I building?",
            max_tokens=40,
            use_ngrams=False,
        )
        txt_id = str(res_id["pure_weight_generated_response"])
        self.assertIn("Alex", txt_id)
        self.assertIn("biological neural network", txt_id)
        self.assertNotIn("Echoing Glade", txt_id)

        # 2. Temporal 'before' bridge with infinitive clause must produce 'before you fix' and never 'before to fix'
        res_before = self.organism.pure_neural_generate_response(
            "Can running outdoors clear the mind before writing an essay without grammar errors?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_before = str(res_before["pure_weight_generated_response"]).lower()
        self.assertNotIn("before to fix", txt_before)
        self.assertIn("running outdoors", txt_before)
        self.assertIn("before you fix grammar errors", txt_before)

    def test_v16_unified_curriculum_gates_and_zero_leakage(self) -> None:
        """Verifies zero prompt overlap with HOLD_OUT_EXAM_SUITE and passing UnifiedCurriculumGates."""
        lesson_stimuli = {
            " ".join(sa.strip().lower().split())
            for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values()
            for _, sa, _, _ in ep_lessons
        }
        for item in HOLD_OUT_EXAM_SUITE:
            norm_p = " ".join(item.prompt.strip().lower().split())
            self.assertNotIn(norm_p, lesson_stimuli, f"Holdout prompt leaked into lessons: {item.prompt}")

        gates = UnifiedCurriculumGates(
            surprise_threshold=0.25,
            min_grammar_score=1.0,
            min_retention_tokens=1500,
            max_attractor_collapse_rate=0.35,
        )
        rep = gates.evaluate_organism(self.organism, stage_or_epoch_id=5)
        self.assertTrue(rep.passed, f"Unified gates failed: {rep.reasons}")
        self.assertLess(rep.surprise_f_t, 0.25)
        self.assertEqual(rep.grammar_score, 1.0)
        self.assertEqual(rep.superacid_count, 0)
        self.assertGreater(rep.context_retention_tokens, 1500)
        self.assertEqual(rep.oov_pass_rate, 1.0)
        self.assertLessEqual(rep.attractor_collapse_rate, 0.35)
        self.assertEqual(rep.holdout_echo_contamination_rate, 0.0)

    def test_v17_theta_gamma_role_filler_binding(self) -> None:
        """Verifies dynamic Theta-Gamma Role x Filler variable binding on novel entities (hollandaise, cycling, manuscript)."""
        res_holl = self.organism.pure_neural_generate_response(
            "My homemade hollandaise emulsion turned thin and runny while whisking; how can I fix it?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_holl = str(res_holl["pure_weight_generated_response"]).lower()
        self.assertIn("hollandaise", txt_holl)
        self.assertEqual(res_holl["role_filler_bindings_applied"].get("mayonnaise"), "hollandaise")  # type: ignore[union-attr]

        res_comp = self.organism.pure_neural_generate_response(
            "Can cycling outdoors clear the mind before fixing grammar errors in my manuscript?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_comp = str(res_comp["pure_weight_generated_response"]).lower()
        self.assertIn("cycling outdoors", txt_comp)
        self.assertIn("manuscript", txt_comp)
        self.assertNotIn("before to fix", txt_comp)

    def test_v17_cortical_moe_branch_profile_and_multi_hop_rollout(self) -> None:
        """Verifies 16-column Brodmann MoE branch energy profile and 2-hop transitive CA3 rollout (A => B => C)."""
        res_mh = self.organism.pure_neural_generate_response(
            "Step by step, how do biological pyramidal neurons in the neocortex integrate sensory spikes?",
            max_tokens=44,
            use_ngrams=False,
            enable_multi_hop=True,
        )
        self.assertTrue(bool(res_mh["multi_hop_rollout_used"]))
        self.assertTrue(len(res_mh["cortical_moe_branch_profile"]) >= 1)  # type: ignore[arg-type]
        self.assertIn("pyramidal neurons", str(res_mh["pure_weight_generated_response"]).lower())

    def test_v17_episodic_working_memory_and_active_inference_teaching_loop(self) -> None:
        """Verifies Episodic Working Memory Graph recall and SON Active Inference Curiosity -> Online Teaching -> Recall."""
        # 1. Episodic Working Memory Graph query
        res_ep = self.organism.pure_neural_generate_response(
            "What topics have we discussed in our conversation so far?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_ep = str(res_ep["pure_weight_generated_response"])
        self.assertEqual(res_ep["conversational_domain"], "episodic_working_memory")
        self.assertIn("Alex", txt_ep)

        # 2. Active Inference Curiosity Gate on completely unseen topic (before teaching)
        res_unknown = self.organism.pure_neural_generate_response(
            "How do excitons travel in quantum photosynthesis?",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        self.assertTrue(bool(res_unknown["active_inference_clarification_triggered"]))
        self.assertIn("epistemic uncertainty", str(res_unknown["pure_weight_generated_response"]).lower())

        # 3. Online Declarative Dopamine Teaching Turn
        res_teach = self.organism.pure_neural_generate_response(
            "Remember that in quantum photosynthesis, excitons travel through chromophores via coherent wavepackets.",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        self.assertGreater(int(res_teach["online_dopamine_imprinted_tokens"]), 0)  # type: ignore[arg-type]

        # 4. Subsequent query on the newly taught topic now recalls from NMDA weights without clarification!
        res_learned = self.organism.pure_neural_generate_response(
            "How do excitons travel in quantum photosynthesis?",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        self.assertFalse(bool(res_learned["active_inference_clarification_triggered"]))
        txt_learned = str(res_learned["pure_weight_generated_response"]).lower()
        self.assertIn("excitons", txt_learned)
        self.assertIn("chromophores", txt_learned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
