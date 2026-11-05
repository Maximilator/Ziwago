#!/usr/bin/env python3
"""
CI Test Suite for Ziwago v14/v16/v17:
  1) Golden-File Round-Trip CI with a different seed (seed=99999 != 42);
  2) Projector & Cortex/SON/Chemistry State Export Contract;
  3) Discriminating Grammar & Trained N-Gram Coverage Gate (rejecting word salad);
  4) Paraphrase Robustness of Retrieval vs. Novel Knowledge Generalization;
  5) 5 v17 Cognitive Dialogue Mechanisms (Theta-Gamma Role-Filler Binding,
     16-Column MoE NMDA Branch Profile, Multi-Hop CA3 Rollout, Episodic Graph,
     and SON Active Inference Curiosity -> Online Dopamine Teaching).
"""

from __future__ import annotations

import copy
import json
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
    MASTER_CKPT,
    CheckpointCompatibilityError,
    UnifiedCurriculumGates,
    evaluate_grammar_and_artifacts,
    evaluate_novel_knowledge_generalization,
    resolve_master_checkpoint_path,
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
        cls.golden_path = ROOT / "tests" / "golden_v16_expected_outputs.json"
        cls.golden_data = json.loads(cls.golden_path.read_text(encoding="utf-8"))

    def test_v16_golden_file_roundtrip_with_different_seed(self) -> None:
        """Point 1, 4 & 5: Save checkpoint -> Load into fresh organism with DIFFERENT seed (seed=99999)
        -> Verify 100% byte-for-byte match against Golden Answers File (golden_v16_expected_outputs.json)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ziwago_roundtrip_test.pt"
            val_info = self.organism.save_self_contained_checkpoint(ckpt_path, use_fp16=False)
            self.assertTrue(val_info["valid"])
            self.assertGreater(int(val_info["active_vocab_columns"]), 20)

            # Deliberately instantiate with seed=99999 (!= 42) to prove zero dependency on seed!
            reloaded = ZiwagoPureSynapticDialogueOrganismV14(
                root=ROOT,
                workers=2,
                cpu_threads=4,
                seed=99999,
            )
            load_info = reloaded.load_self_contained_checkpoint(ckpt_path, strict_schema=True)
            self.assertTrue(load_info["valid"])

            for g_item in self.golden_data["holdout_paraphrase_retrieval_golden"]:
                q = str(g_item["prompt"])
                gold_expected = str(g_item["expected_response"])
                actual_out = reloaded.pure_neural_generate_response(q, max_tokens=48, use_ngrams=False)[
                    "pure_weight_generated_response"
                ]
                self.assertEqual(
                    actual_out,
                    gold_expected,
                    f"Golden Round-Trip mismatch (seed=99999) for {g_item['exam_id']}: GOT={actual_out!r} != GOLD={gold_expected!r}",
                )

            for n_item in self.golden_data["novel_knowledge_generalization_golden"]:
                q = str(n_item["prompt"])
                gold_expected = str(n_item["expected_response"])
                actual_out = reloaded.pure_neural_generate_response(q, max_tokens=48, use_ngrams=False)[
                    "pure_weight_generated_response"
                ]
                self.assertEqual(
                    actual_out,
                    gold_expected,
                    f"Golden Novel Generalization mismatch (seed=99999) for {n_item['probe_id']}: GOT={actual_out!r} != GOLD={gold_expected!r}",
                )

    def test_v16_checkpoint_rejects_broken_maps_projectors_and_states(self) -> None:
        """Points 1, 4 & 5: Verifies that missing col_to_tok, missing projectors (pos_phase_vecs, vsa_encoder_state),
        missing cortex/SON/chemistry states, broken bijection, or wrong tokenizer hash raises CheckpointCompatibilityError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "valid_ckpt.pt"
            self.organism.save_self_contained_checkpoint(ckpt_path, use_fp16=False)
            base_payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

            # 1. Missing col_to_tok, pos_phase_vecs, vsa_encoder_state, or moe_cortex_state
            for req_key in ("col_to_tok", "pos_phase_vecs", "w_basal_branch", "vsa_encoder_state", "moe_cortex_state", "son_motor_state", "chemistry_state"):
                bad_missing = copy.deepcopy(base_payload)
                del bad_missing[req_key]
                with self.assertRaisesRegex(CheckpointCompatibilityError, "missing required self-contained keys"):
                    validate_self_contained_checkpoint(bad_missing)

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

    def test_v16_discriminating_grammar_gate_rejects_word_salad(self) -> None:
        """Point 2: Verifies that the discriminating grammar & n-gram coverage gate strictly rejects
        BPE word salad that starts with a capital letter and ends with a period."""
        word_salad_samples = [
            "To in but to are, i a In you more , at- are for the, you with he, Yes were In for The there the this will at to his and neural be i of as was for.",
            "While and to its a be in are, and before He't was the and is by a one to , you an , look.",
            "A is it is for I, his the to but for in's me ' it the him i's they, in could.",
        ]
        for bad_txt in word_salad_samples:
            rep = evaluate_grammar_and_artifacts(
                bad_txt,
                trained_bigram_g=self.organism.synaptic_bigram_g,
                encode_fn=self.organism.encode_text,
            )
            self.assertFalse(
                bool(rep["grammar_coherent"]),
                f"Discriminating grammar gate failed to reject word salad: {bad_txt!r} -> {rep}",
            )

        valid_txt = "Pyramidal neurons receive bottom-up sensory spikes on basal dendrites and top-down cortical predictions on apical tuft NMDA compartments."
        rep_good = evaluate_grammar_and_artifacts(
            valid_txt,
            trained_bigram_g=self.organism.synaptic_bigram_g,
            encode_fn=self.organism.encode_text,
        )
        self.assertTrue(bool(rep_good["grammar_coherent"]))
        self.assertGreaterEqual(float(rep_good["trained_ngram_coverage"]), 0.80)

    def test_v16_context_arbitrage_and_ba44_grammar_seam(self) -> None:
        """Verifies no 'Echoing Glade' leakage on identity queries and no 'before to fix' grammatical seam."""
        res_id = self.organism.pure_neural_generate_response(
            "Remind me, what is my name and what project am I building?",
            max_tokens=40,
            use_ngrams=False,
        )
        txt_id = str(res_id["pure_weight_generated_response"])
        self.assertIn("Alex", txt_id)
        self.assertIn("biological neural network", txt_id)
        self.assertNotIn("Echoing Glade", txt_id)

        res_before = self.organism.pure_neural_generate_response(
            "Can running outdoors clear the mind before writing an essay without grammar errors?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_before = str(res_before["pure_weight_generated_response"]).lower()
        self.assertNotIn("before to fix", txt_before)
        self.assertIn("running outdoors", txt_before)
        self.assertIn("before you fix grammar errors", txt_before)

    def test_v16_unified_gates_paraphrase_robustness_and_novel_generalization(self) -> None:
        """Point 3: Verifies both paraphrase_robustness_of_retrieval and novel_knowledge_generalization_score."""
        lesson_stimuli = {
            " ".join(sa.strip().lower().split())
            for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values()
            for _, sa, _, _ in ep_lessons
        }
        lesson_responses_lower = [
            sb.strip().lower()
            for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values()
            for _, _, sb, _ in ep_lessons
        ]
        for item in HOLD_OUT_EXAM_SUITE:
            norm_p = " ".join(item.prompt.strip().lower().split())
            self.assertNotIn(norm_p, lesson_stimuli, f"Holdout prompt leaked into lessons: {item.prompt}")

        gates = UnifiedCurriculumGates(
            surprise_threshold=0.25,
            min_grammar_score=1.0,
            min_retention_tokens=1500,
            max_attractor_collapse_rate=0.35,
            min_novel_generalization_score=0.75,
        )
        rep = gates.evaluate_organism(
            self.organism,
            stage_or_epoch_id=5,
            training_corpus_responses_lower=lesson_responses_lower,
        )
        self.assertTrue(rep.passed, f"Unified gates failed: {rep.reasons}")
        self.assertLess(rep.surprise_f_t, 0.25)
        self.assertEqual(rep.grammar_score, 1.0)
        self.assertGreaterEqual(rep.trained_ngram_coverage, 0.65)
        self.assertEqual(rep.superacid_count, 0)
        self.assertGreater(rep.context_retention_tokens, 1500)
        self.assertEqual(rep.oov_pass_rate, 1.0)
        self.assertEqual(rep.paraphrase_robustness_of_retrieval, 1.0)
        self.assertEqual(rep.novel_knowledge_generalization_score, 1.0)
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
        res_ep = self.organism.pure_neural_generate_response(
            "What topics have we discussed in our conversation so far?",
            max_tokens=44,
            use_ngrams=False,
        )
        txt_ep = str(res_ep["pure_weight_generated_response"])
        self.assertEqual(res_ep["conversational_domain"], "episodic_working_memory")
        self.assertIn("Alex", txt_ep)

        res_unknown = self.organism.pure_neural_generate_response(
            "How do excitons travel in quantum photosynthesis?",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        self.assertTrue(bool(res_unknown["active_inference_clarification_triggered"]))
        self.assertIn("epistemic uncertainty", str(res_unknown["pure_weight_generated_response"]).lower())

        res_teach = self.organism.pure_neural_generate_response(
            "Remember that in quantum photosynthesis, excitons travel through chromophores via coherent wavepackets.",
            max_tokens=44,
            use_ngrams=False,
            enable_active_inference=True,
        )
        self.assertGreater(int(res_teach["online_dopamine_imprinted_tokens"]), 0)  # type: ignore[arg-type]

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

    def test_v16_master_checkpoint_disk_reproduces_golden_with_seed_99999(self) -> None:
        """Directly loads MASTER_CKPT with seed=99999 (failing hard if missing instead of skipping)
        and verifies byte-for-byte reproduction of tests/golden_v16_expected_outputs.json."""
        self.assertTrue(MASTER_CKPT.exists(), f"Missing canonical MASTER_CKPT: {MASTER_CKPT}")
        master_ckpt = resolve_master_checkpoint_path(ROOT)

        master_org = ZiwagoPureSynapticDialogueOrganismV14(
            root=ROOT,
            workers=2,
            cpu_threads=4,
            seed=99999,
        )
        val_info = master_org.load_self_contained_checkpoint(master_ckpt, strict_schema=True)
        self.assertTrue(val_info["valid"])
        self.assertGreaterEqual(int(val_info["active_vocab_columns"]), 9000)

        for g_item in self.golden_data["holdout_paraphrase_retrieval_golden"]:
            q = str(g_item["prompt"])
            gold_expected = str(g_item["expected_response"])
            actual_out = master_org.pure_neural_generate_response(q, max_tokens=48, use_ngrams=False)[
                "pure_weight_generated_response"
            ]
            self.assertEqual(
                actual_out,
                gold_expected,
                f"Master checkpoint mismatch (seed=99999) for {g_item['exam_id']}: GOT={actual_out!r} != GOLD={gold_expected!r}",
            )

    def test_v17_snn_assistant_all_five_breakthroughs(self) -> None:
        """Tests the 5 v17.0 Biological SNN Assistant breakthroughs end-to-end:
        1) Arcuate Fasciculus BPE token-ID pointer injection (0% string .replace);
        2) Protected Core Bank w_core_dend_out;
        3) Closed-loop MoE + DiscretePredictiveCodingBarrier Stranger's Check;
        4) Sequential Attractor Episode Chains (w_ca3_episode_links, 2-sentence paragraphs);
        5) Basal Ganglia D1/D2 Action Selection + Autonomic SWS Consolidation."""
        from ziwago_assistant import BasalGangliaActionSelector, ZiwagoBiologicalSNNAssistant

        assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=2, cpu_threads=4, seed=99999)
        self.assertGreater(int(assistant.organism.dendritic_plateau_readout.w_core_dend_out.shape[1]), 0)

        # 1. Arcuate Fasciculus BPE token-ID pointer injection
        r_arc = assistant.chat("Can cycling outdoors clear the mind before fixing grammar errors in my manuscript?")
        self.assertGreaterEqual(int(r_arc["arcuate_token_pointer_injections"]), 2)  # type: ignore[arg-type]
        self.assertIn("cycling outdoors", str(r_arc["pure_weight_generated_response"]).lower())
        self.assertIn("manuscript", str(r_arc["pure_weight_generated_response"]).lower())

        # 2. Active Inference Clarification before teaching
        r_clar = assistant.chat("How do excitons travel in quantum photosynthesis?")
        self.assertEqual(
            r_clar["basal_ganglia_selected_action"],
            BasalGangliaActionSelector.ACTION_ACTIVE_INFERENCE_CLARIFY,
        )
        self.assertTrue(bool(r_clar["active_inference_clarification_triggered"]))

        # 3. Multi-sentence teaching with CA3 horizontal episode link (p1 -> p2)
        r_teach = assistant.chat(
            "Remember that in quantum photosynthesis, excitons travel through chromophores via coherent wavepackets. "
            "Coherent quantum wavepackets protect photosynthetic energy transfer from thermal dissipation."
        )
        self.assertEqual(
            r_teach["basal_ganglia_selected_action"],
            BasalGangliaActionSelector.ACTION_ONLINE_DA_CONSOLIDATION,
        )
        self.assertEqual(int(r_teach["episode_teaching_telemetry"]["episode_links_created"]), 1)  # type: ignore[index]

        # 4. Multi-sentence paragraph generation via CA3 Sequential Attractor Chain (Sentences == 2)
        r_para = assistant.chat("Explain in detail how excitons travel in quantum photosynthesis?")
        self.assertEqual(
            r_para["basal_ganglia_selected_action"],
            BasalGangliaActionSelector.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS,
        )
        self.assertEqual(int(r_para["episode_chain_sentences_generated"]), 2)  # type: ignore[arg-type]
        txt_para = str(r_para["pure_weight_generated_response"]).lower()
        self.assertIn("chromophores", txt_para)
        self.assertIn("thermal dissipation", txt_para)

        # 5. Autonomic Slow-Wave Sleep consolidation
        sws = assistant.sleep_and_consolidate()
        self.assertTrue(bool(sws["core_bank_replay_applied"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
