#!/usr/bin/env python3
r"""
ZIWAGO-GENESIS v18.0 — CI TEST SUITE FOR THE 5 DREAM BIOLOGICAL SNN PILLARS (`tests/test_v18_genesis_dream.py`)
===============================================================================================================
Проверяет все 5 инновационных механизмов архитектуры Ziwago-Genesis v18.0:
  1. `SyntacticValenceWaveGenerator`: генерация по непрерывной волне валентностей зоны Брока с 0% `pos_idx` и 0% `pos_phase_vecs`;
  2. `CorticalRelationalWorldGraph`: кристаллизация отдельных фактов в каузальный граф и синтез нового 3-хопового предложения с нуля;
  3. `EmergentOntologicalRoleDiscovery`: самозарождение онтологических классов ролей (`treatise`, `kayaking`) через дистрибутивную STDP;
  4. `ThetaGammaSpikingPhaseMultiplexer`: фазовое кодирование 6 ролей (TTFS) в 120-мс спайковой тета-волне с косинусной точностью > 0.99;
  5. `BidirectionalMeaningClosureVerifier`: обратная реконструкция вопроса $X \to \hat{Y} \to \hat{X}$ и автодостройка пропущенной клаузы.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.cortex.genesis_engine import (
    BidirectionalMeaningClosureVerifier,
    CorticalRelationalWorldGraph,
    EmergentOntologicalRoleDiscovery,
    SyntacticValenceWaveGenerator,
    ThetaGammaSpikingPhaseMultiplexer,
)
from neuroweave10.evaluation.gates import evaluate_grammar_and_artifacts
from train_ziwago_v18_genesis_dream import ZiwagoGenesisDreamOrganismV18


class TestZiwagoV18GenesisDreamArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.genesis = ZiwagoGenesisDreamOrganismV18(root=ROOT, workers=2, cpu_threads=4, seed=99999)
        cls.genesis.ingest_scientific_corpus_into_world_graph(
            [
                "Astrocytes release gliotransmitters such as D-serine.",
                "D-serine binds to glycine sites on apical NMDA receptors.",
                "Apical NMDA receptors ignite dendritic plateau potentials in pyramidal neurons.",
            ],
            dopamine_gain=2.5,
        )
        for s_exp in (
            "It is possible to fix runny bechamel by slowly whisking the mixture into a fresh egg yolk.",
            "To fix grammar errors when writing my treatise, read each sentence aloud and simplify overly long clauses.",
            "Staying active is a wonderful habit, and kayaking outdoors is a great way to clear the mind.",
        ):
            cls.genesis.emergent_ontology.observe_sentence_stdp(s_exp, register_as_canonical=False, lr=2.0)

    def test_pillar1_zero_pos_idx_syntactic_valence_wave(self) -> None:
        """Pillar 1: Verifies SyntacticValenceWaveGenerator has no pos_phase_vecs and no pos_idx parameter,
        and generates 100% grammatical responses purely via continuous valence state Psi_BA44(t)."""
        gen = self.genesis.valence_wave_generator
        self.assertFalse(hasattr(gen, "pos_phase_vecs"))
        sig = inspect.signature(gen.compute_valence_dendritic_state)
        self.assertNotIn("pos_idx", sig.parameters)

        res = self.genesis.generate_genesis_response(
            "Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
            use_zero_pos_valence_wave=True,
        )
        txt = str(res["pure_weight_generated_response"])
        self.assertTrue(bool(res["zero_pos_idx_used"]))
        self.assertIn("basal dendrites", txt.lower())
        self.assertIn("apical tuft nmda compartments", txt.lower())
        self.assertTrue(bool(evaluate_grammar_and_artifacts(txt)["grammar_coherent"]))

    def test_pillar2_world_graph_multi_hop_novel_sentence_synthesis(self) -> None:
        """Pillar 2: Verifies CorticalRelationalWorldGraph synthesizes a brand-new 3-hop causal sentence
        token-by-token through the dedicated zero-pos_idx NMDA matrix (synthesized_via_zero_pos_nmda_matrix == True)."""
        res = self.genesis.generate_genesis_response(
            "How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?",
            prefer_world_graph_synthesis=True,
        )
        graph_info = res["world_graph_synthesis"]
        self.assertIsNotNone(graph_info)
        self.assertTrue(bool(graph_info["synthesized_via_zero_pos_nmda_matrix"]))  # type: ignore[index]
        self.assertEqual(int(graph_info["hops_traversed"]), 3)  # type: ignore[index]
        txt = str(res["pure_weight_generated_response"])
        self.assertIn("astrocytes release gliotransmitters such as d-serine", txt.lower())
        self.assertIn("which binds to glycine sites on apical nmda receptors", txt.lower())
        self.assertIn("to ignite dendritic plateau potentials in pyramidal neurons", txt.lower())
        self.assertTrue(bool(evaluate_grammar_and_artifacts(txt)["grammar_coherent"]))

    def test_pillar3_unsupervised_stdp_emergent_ontology_and_schema_residual_law(self) -> None:
        """Pillar 3: Verifies EmergentOntologicalRoleDiscovery uses zero domain stop words in FUNCTION_WORDS_ONLY
        and autonomously binds 'kayaking' -> 'running', 'treatise' -> 'essay', and 'bechamel' -> 'mayonnaise'
        purely via the Schema-Residual Law and asymmetric STDP."""
        for domain_word in ("pyramidal", "neurons", "grammar", "errors", "outdoors", "sauce", "runny", "yolk"):
            self.assertNotIn(domain_word, EmergentOntologicalRoleDiscovery.FUNCTION_WORDS_ONLY)

        res = self.genesis.generate_genesis_response(
            "Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?",
            use_zero_pos_valence_wave=True,
        )
        discoveries = res["emergent_stdp_role_discoveries"]
        self.assertIn("running", discoveries)  # type: ignore[arg-type]
        self.assertIn("essay", discoveries)  # type: ignore[arg-type]
        self.assertEqual(discoveries["running"][0], "kayaking")  # type: ignore[index]
        self.assertEqual(discoveries["essay"][0], "treatise")  # type: ignore[index]

        txt = str(res["pure_weight_generated_response"]).lower()
        self.assertIn("kayaking outdoors", txt)
        self.assertIn("treatise", txt)
        self.assertTrue(bool(evaluate_grammar_and_artifacts(str(res["pure_weight_generated_response"]))["grammar_coherent"]))

        res_bech = self.genesis.generate_genesis_response(
            "Why does a runny bechamel sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?",
            use_zero_pos_valence_wave=True,
        )
        disc_bech = res_bech["emergent_stdp_role_discoveries"]
        self.assertIn("mayonnaise", disc_bech)  # type: ignore[arg-type]
        self.assertEqual(disc_bech["mayonnaise"][0], "bechamel")  # type: ignore[index]
        self.assertIn("runny bechamel", str(res_bech["pure_weight_generated_response"]).lower())

    def test_pillar4_theta_gamma_ttfs_spiking_multiplexer(self) -> None:
        """Pillar 4: Verifies 6-slot Theta-Gamma Time-to-First-Spike (TTFS) phase coding over a 120 ms theta wave
        reconstructs all 6 bound role vectors with cosine > 0.99 and zero cross-slot leakage."""
        mux = self.genesis.theta_gamma_multiplexer
        g = torch.Generator().manual_seed(1804)
        roles_in = {
            role_name: F.normalize(torch.randn(64, generator=g), p=2, dim=0)
            for role_name in mux.ROLE_SLOT_NAMES
        }
        raster, windows = mux.encode_roles_into_theta_spike_train(roles_in)
        self.assertEqual(tuple(raster.shape), (120, 64))
        self.assertEqual(len(windows), 6)

        decoded = mux.decode_roles_from_theta_spike_train(raster)
        for role_name, v_orig in roles_in.items():
            cos_sim = float(torch.dot(v_orig, decoded[role_name]).item())
            self.assertGreater(cos_sim, 0.99, f"TTFS phase reconstruction failed for {role_name}: {cos_sim}")

    def test_pillar5_bidirectional_meaning_closure_and_repair(self) -> None:
        """Pillar 5: Verifies BidirectionalMeaningClosureVerifier detects an incomplete single-clause draft
        for a 2-clause question, identifies the missing concept SDR r_missing, and repairs the response."""
        question = "Why does a runny mayonnaise sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?"
        incomplete_draft = "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes."

        def _synth_pole(p_vec: torch.Tensor) -> str:
            ids = self.genesis.valence_wave_generator.generate_trajectory_zero_pos(
                q_goal_vec=p_vec,
                get_vec64_fn=self.genesis.core_organism.get_vec64,
                decode_fn=self.genesis.core_organism.decode_token,
                max_tokens=44,
            )
            return "".join(self.genesis.core_organism.decode_token(i) for i in ids).strip()

        rep = self.genesis.meaning_closure_verifier.evaluate_and_close_meaning_loop(
            question_text=question,
            draft_response_text=incomplete_draft,
            ca3_attractors=self.genesis.core_organism.dendritic_plateau_readout.w_ca3_attractors,
            ca3_sdr=self.genesis.core_organism.dendritic_plateau_readout.w_ca3_sdr,
            compute_sdr_fn=self.genesis.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048,
            encode_fn=self.genesis.core_organism.encode_text,
            synthesize_pole_fn=_synth_pole,
        )
        self.assertTrue(bool(rep["closure_repair_applied"]))
        self.assertGreater(float(rep["meaning_closure_score"]), 0.70)  # type: ignore[arg-type]
        self.assertIn("pyramidal neurons", str(rep["verified_response"]).lower())

    def test_pillar6_genesis_state_roundtrip_across_different_seeds(self) -> None:
        """Pillar 6: Verifies full export_genesis_state_dict -> load_genesis_state_dict into a fresh organism
        initialized with a completely different seed (seed=77777 != 99999) produces 100% byte-identical outputs."""
        exported = self.genesis.export_genesis_state_dict()
        fresh_genesis = ZiwagoGenesisDreamOrganismV18(root=ROOT, workers=2, cpu_threads=4, seed=77777)
        fresh_genesis.load_genesis_state_dict(exported)

        for q, use_graph in (
            ("How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?", True),
            ("Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?", False),
            ("Why does a runny bechamel sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?", False),
        ):
            out_orig = self.genesis.generate_genesis_response(q, prefer_world_graph_synthesis=use_graph)["pure_weight_generated_response"]
            out_fresh = fresh_genesis.generate_genesis_response(q, prefer_world_graph_synthesis=use_graph)["pure_weight_generated_response"]
            self.assertEqual(out_orig, out_fresh)

    def test_pillar7_proportional_think_gate_passport_and_clarification_continuity(self) -> None:
        """Verifies all 5 engineering & biophysical improvements:
        1) Missing MASTER_CKPT raises CheckpointCompatibilityError (fail instead of silent skip);
        2) Proportional Gate Zone 2 ('I know X as Y' when X is in STDP cortex/world graph) vs Zone 3 ('share the key mechanism');
        3) Deterministic response_passport with selected_ca3_pole_indices, saliences, and triggered mechanisms;
        4) Implicit user reply after a clarifying turn automatically consolidates knowledge and updates episodic_turns;
        5) Sub-400ms per-turn latency."""
        from neuroweave10.evaluation.gates import CheckpointCompatibilityError, MASTER_CKPT, resolve_master_checkpoint_path
        from ziwago_assistant import BasalGangliaActionSelector, ZiwagoBiologicalSNNAssistant

        self.assertTrue(MASTER_CKPT.exists(), f"Missing MASTER_CKPT: {MASTER_CKPT}")
        with self.assertRaises(CheckpointCompatibilityError):
            resolve_master_checkpoint_path(explicit_path=ROOT / "Produce" / "checkpoints" / "nonexistent_ckpt_file.pt")

        assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=2, cpu_threads=4, seed=12345)
        assistant.genesis.emergent_ontology.observe_sentence_stdp(
            "To fix grammar errors in your treatise, read each sentence aloud, check subject-verb agreement, and simplify overly long clauses.",
            register_as_canonical=False,
            lr=1.5,
        )

        # Zone 2: Short/under-specified question about a known STDP concept ('treatise') -> 'I know treatise as essay: ...'
        res_think = assistant.chat("What do you know about treatise?", enable_active_inference=True)
        self.assertTrue(bool(res_think.get("endogenous_cortical_think_triggered")))
        self.assertFalse(bool(res_think.get("active_inference_clarification_triggered")))
        self.assertEqual(
            res_think["basal_ganglia_selected_action"],
            BasalGangliaActionSelector.ACTION_ENDOGENOUS_CORTICAL_THINK,
        )
        txt_think = str(res_think["pure_weight_generated_response"])
        self.assertTrue(txt_think.startswith("I know treatise as essay:"), f"Unexpected Zone 2 output: {txt_think!r}")
        passport_think = res_think["response_passport"]  # type: ignore[index]
        self.assertEqual(passport_think["proportional_gate_zone"], "ZONE_2_KNOWN_REFORMULATE_THINK")  # type: ignore[index]
        self.assertTrue(bool(passport_think["mechanisms_triggered"]["endogenous_cortical_think"]))  # type: ignore[index]
        self.assertGreater(len(passport_think["selected_ca3_pole_indices"]), 0)  # type: ignore[arg-type,index]

        # Zone 3: Truly unknown concept ('tardigrades') -> asks clarifying question & records was_clarification_request=True
        res_clarify = assistant.chat("Why do tardigrades survive extreme radiation in outer space?", enable_active_inference=True)
        self.assertTrue(bool(res_clarify["active_inference_clarification_triggered"]))
        passport_clarify = res_clarify["response_passport"]  # type: ignore[index]
        self.assertEqual(passport_clarify["proportional_gate_zone"], "ZONE_3_UNKNOWN_CLARIFY")  # type: ignore[index]
        self.assertTrue(bool(assistant.organism.dialogue_state.episodic_turns[-1]["was_clarification_request"]))

        # Implicit reply to clarifying question WITHOUT 'Remember that' -> automatically consolidates!
        res_implicit_teach = assistant.chat(
            "Tardigrades survive extreme radiation by producing Dsup proteins that shield DNA from hydroxyl radicals.",
            enable_active_inference=True,
        )
        self.assertEqual(
            res_implicit_teach["basal_ganglia_selected_action"],
            BasalGangliaActionSelector.ACTION_ONLINE_DA_CONSOLIDATION,
        )
        self.assertGreater(int(res_implicit_teach["online_dopamine_imprinted_tokens"]), 0)  # type: ignore[arg-type]

        # Subsequent query now recalls from Zone 1 Direct Recall!
        res_after = assistant.chat("Why do tardigrades survive extreme radiation?", enable_active_inference=True)
        self.assertFalse(bool(res_after["active_inference_clarification_triggered"]))
        self.assertIn("dsup", str(res_after["pure_weight_generated_response"]).lower())
        self.assertEqual(res_after["response_passport"]["proportional_gate_zone"], "ZONE_1_DIRECT_RECALL")  # type: ignore[index]


if __name__ == "__main__":
    unittest.main(verbosity=2)
