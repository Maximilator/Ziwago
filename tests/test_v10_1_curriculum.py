"""Tests for Ziwago-Prime v10.1 Three-Phase Ontogenetic Curriculum Trainer (`train_ziwago_v10_curriculum.py`)."""

from pathlib import Path
import json
import torch

from train_ziwago_v10_curriculum import (
    STAGE_CONFIGS,
    StreamingDataMixer,
    ZiwagoOntogeneticV101Trainer,
)

ROOT = Path(__file__).resolve().parent.parent


def test_streaming_data_mixer_all_three_stage_proportions():
    mixer = StreamingDataMixer(root=ROOT, workers=4, max_cornell_dialogs_per_stage=500_000, seed=42)

    # Stage 1: TinyStories 60% + Cornell Dialogs 40%
    b1 = mixer.stream_stage_batch(stage_id=1, target_total_tokens=2000, seed=101)
    assert b1.total_tokens == 2000
    assert abs(b1.domain_token_shares["tinystories"] - 0.60) < 0.01
    assert abs(b1.domain_token_shares["cornell_dialogs"] - 0.40) < 0.01
    assert b1.max_exposure_per_uid == 1
    assert b1.cornell_dialogs_yielded <= 500_000

    # Stage 2: BookCorpus 60% + TinyStories 20% + Cornell Dialogs 20%
    b2 = mixer.stream_stage_batch(stage_id=2, target_total_tokens=2000, seed=202)
    assert b2.total_tokens == 2000
    assert abs(b2.domain_token_shares["bookcorpus"] - 0.60) < 0.01
    assert abs(b2.domain_token_shares["tinystories"] - 0.20) < 0.01
    assert abs(b2.domain_token_shares["cornell_dialogs"] - 0.20) < 0.01
    assert b2.max_exposure_per_uid == 1

    # Stage 3: BookCorpus 40% + WikiText 30% + OpenWebText 30%
    b3 = mixer.stream_stage_batch(stage_id=3, target_total_tokens=2000, seed=303)
    assert b3.total_tokens == 2000
    assert abs(b3.domain_token_shares["bookcorpus"] - 0.40) < 0.01
    assert abs(b3.domain_token_shares["wikitext"] - 0.30) < 0.01
    assert abs(b3.domain_token_shares["openwebtext"] - 0.30) < 0.01
    assert b3.max_exposure_per_uid == 1


def test_gated_checkpoints_and_telemetry_artifacts():
    for stage_id in (1, 2, 3):
        ckpt = ROOT / "Produce" / "checkpoints" / f"ziwago_v10_stage{stage_id}.pt"
        assert ckpt.exists(), f"Missing stage {stage_id} checkpoint: {ckpt}"
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        assert payload["stage_id"] == stage_id
        assert payload["vocab_size"] == 50257

    telemetry_file = ROOT / "Produce" / "v10_1_curriculum_telemetry.json"
    assert telemetry_file.exists()
    data = json.loads(telemetry_file.read_text(encoding="utf-8"))
    assert data["status"] == "COMPLETED"
    assert len(data["stages_completed"]) == 3
    for st in data["stages_completed"]:
        assert st["gate_report"]["passed"] is True
        assert st["gate_report"]["superacid_detected"] is False
        assert st["gate_report"]["grammar_score"] == 1.0


def test_three_compositional_generalization_exams():
    from run_v10_1_compositional_exams import run_all_three_compositional_exams

    report = run_all_three_compositional_exams()
    assert report["all_three_tests_passed"] is True
    assert report["test_1_cross_domain"]["switched_from_story_to_encyclopedia"] is True
    assert report["test_2_hallucination_resistance"]["hallucinated_fake_news"] is False
    assert report["test_2_hallucination_resistance"]["discrete_pc_barrier_verdict"]["discrete_surprise_pin"] == 1
    assert report["test_2_hallucination_resistance"]["discrete_pc_barrier_verdict"]["stranger_check_triggered"] is True
    assert report["test_3_long_context_2000_openwebtext"]["total_openwebtext_tokens_fed"] == 2048
    assert report["test_3_long_context_2000_openwebtext"]["exact_fact_recalled"] is True


def test_v10_3_two_phase_ontogeny_knowledge_master():
    master_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v10_3_knowledge_master.pt"
    assert master_ckpt.exists(), f"Missing v10.3 knowledge master checkpoint: {master_ckpt}"

    telemetry_path = ROOT / "Produce" / "v10_3_two_phase_ontogeny_telemetry.json"
    assert telemetry_path.exists()
    t = json.loads(telemetry_path.read_text(encoding="utf-8"))

    # Phase A Structural Grounding checks
    pa = t["phase_a_structural_grounding"]
    assert pa["exited_under_0_05"] is True
    assert pa["final_surprise_F"] < 0.05
    assert pa["frozen_grammar_hard_bigrams"] > 3000
    assert pa["frozen_grammar_hard_trigrams"] > 4000
    assert pa["cortical_ba44_hard_synapses"] > 500000

    # Phase B Knowledge Assimilation checks (no early Surprise stop)
    assert t["total_knowledge_tokens_assimilated"] > 1_400_000
    assert t["total_dht_episodes"] > 15_000
    assert t["total_hipporag_concepts"] > 10_000
    assert t["total_col2_encyclopedic_hard_synapses"] > 800_000
    assert t["total_stranger_check_rejections"] > 100
    assert t["grammar_hard_overwrite_violations"] == 0

    # Final 3 Erudition Exams on ziwago_v10_3_knowledge_master.pt
    ex = t["final_three_exams"]
    assert ex["all_three_tests_passed"] is True
    assert ex["grammar_preserved_100_percent"] is True


def test_v11_0_gpt2_parity_cortical_semantic_engine():
    v11_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v11_0_gpt2_parity_master.pt"
    assert v11_ckpt.exists(), f"Missing v11.0 GPT-2 parity checkpoint: {v11_ckpt}"

    v11_json = ROOT / "Produce" / "v11_0_gpt2_parity_telemetry.json"
    assert v11_json.exists()
    t11 = json.loads(v11_json.read_text(encoding="utf-8"))

    assert t11["total_neurons"] == 16384
    assert t11["total_dendritic_branches"] == 114688
    assert t11["total_synaptic_parameters"] == 124416000
    assert t11["total_tokens_processed"] > 3_000_000
    assert t11["peak_cortical_throughput_tok_per_sec"] > 50_000
    assert t11["sbc_vsa_dht_episodes"] > 35_000
    assert t11["elastic_adaptations_count"] > 500_000

    wiki_exam = t11["final_coherent_exams"]["test_2_wikipedia_non_valkyria_article"]
    assert "Valkyria" not in wiki_exam["generated_text"]
    assert wiki_exam["cross_topic_jumps_blocked_by_apical_sst"] > 100
    assert wiki_exam["grammar_coherent"] is True


def test_v11_5_predictive_readout_and_pfc_entity_binding():
    v115_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v11_5_gpt2_parity_master.pt"
    assert v115_ckpt.exists(), f"Missing v11.5 checkpoint: {v115_ckpt}"

    v115_json = ROOT / "Produce" / "v11_5_predictive_readout_telemetry.json"
    assert v115_json.exists()
    t115 = json.loads(v115_json.read_text(encoding="utf-8"))

    assert t115["total_neurons"] == 16384
    assert t115["total_dendritic_branches"] == 114688
    assert t115["total_synaptic_parameters"] == 124416000
    assert t115["pc_readout_total_matrix_updates"] > 200_000
    assert t115["pc_readout_final_loss_nats"] < t115["pc_readout_initial_loss_nats"]

    ex1 = t115["final_gpt2_grade_exams"]["test_1_fairytale_entity_consistency"]
    assert ex1["bound_protagonist_in_pfc_who_slot"] == "Lily"
    assert ex1["character_names_in_output"] == ["lily"]
    assert ex1["single_consistent_protagonist"] is True
    assert ex1["alien_character_names_inhibited"] > 100
    for ex_val in t115["final_gpt2_grade_exams"].values():
        assert ex_val["grammar_coherent"] is True


def test_v12_0_conversational_mind_six_turn_dialogue():
    from ziwago_v12_conversational_mind import run_v12_six_turn_dialogue_benchmark

    rep = run_v12_six_turn_dialogue_benchmark()
    assert rep["all_turns_coherent_and_passed"] is True
    assert rep["total_turns"] == 6
    prof = rep["final_user_profile_in_pfc_and_dht"]
    assert prof["user_name"] == "Alex"
    assert "biological neural network" in prof["user_project_or_topic"]
    assert len(prof["dht_episode_ids"]) == 2


def test_v13_0_son_reflective_organism_and_paired_dialogue():
    rep_path = ROOT / "Produce" / "v13_0_son_reflective_organism_report.json"
    assert rep_path.exists()
    rep = json.loads(rep_path.read_text(encoding="utf-8"))

    assert rep["total_neurons"] == 17408
    assert rep["son_neurons"] == 1024
    assert rep["total_paired_dialogues_in_memory"] > 5000
    assert rep["son_vte_simulations_run"] >= 5

    res_list = rep["dialogue_results"]
    assert len(res_list) == 5
    for r in res_list:
        assert r["stopped_naturally_by_son_eos"] is True
        # Ensure zero trailing speaker tags like "BIANCA :" or "NORA :"
        assert " : " not in r["response_text"]
    assert "Alex" in res_list[4]["response_text"]
    assert "biological neural network" in res_list[4]["response_text"]


def test_v14_0_pure_synaptic_dialogue_organism():
    v14_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v14_0_pure_synaptic_dialogue.pt"
    assert v14_ckpt.exists()
    v14_json = ROOT / "Produce" / "v14_0_pure_synaptic_dialogue_report.json"
    assert v14_json.exists()
    rep14 = json.loads(v14_json.read_text(encoding="utf-8"))

    # Verify zero stored sentence chains in organism
    assert rep14["stored_sentence_chains_in_organism"] == 0
    assert rep14["total_pairs_trained"] > 10_000
    assert rep14["synaptic_trigrams_learned"] > 100_000
    assert rep14["synaptic_fourgrams_learned"] > 200_000

    src_counts = rep14["source_dataset_pair_counts"]
    for k in ("ultrachat", "oasst2", "personachat", "personamem", "npc_dialogue", "daily_dialog_cornell"):
        assert src_counts.get(k, 0) > 0, f"Dataset {k} was not streamed!"

    evals_off = rep14["pure_dendritic_matrix_ngrams_off_evals"]
    evals_on = rep14["hybrid_ngrams_on_evals"]
    assert len(evals_off) == 6
    assert len(evals_on) == 6
    for ev in evals_off + evals_on:
        assert ev["stored_sentence_chains_in_class"] == 0
        assert ev["stopped_naturally_by_son_eos"] is True
    assert "Alex" in evals_off[-1]["pure_weight_generated_response"]
    assert "mayonnaise" in evals_off[1]["pure_weight_generated_response"]
    assert "grammar" in evals_off[2]["pure_weight_generated_response"]
    assert "Echoing Glade" in evals_off[4]["pure_weight_generated_response"]
    assert "Alex" in evals_on[-1]["pure_weight_generated_response"]

    # Verify Bidirectional CA3 Symmetry on Unseen Test 4 (response-only tokens "Echoing Glade" + "whisking egg yolk")
    from train_ziwago_v14_pure_synaptic_dialogue import (
        ZiwagoPureSynapticDialogueOrganismV14,
        TokenizedDialogPair,
        _bpe_tokenize_raw_pairs_worker,
    )

    org = ZiwagoPureSynapticDialogueOrganismV14(root=ROOT, workers=8, cpu_threads=8, seed=42)
    org.use_ngrams = False
    anchor_raw = [
        ("oasst2", "I am making mayonnaise, it was starting to thicken but now it has become runny and liquid.",
         "Yes, it is possible to fix runny mayonnaise by slowly whisking the mixture into a fresh egg yolk until the emulsion stabilizes.", 2.2),
        ("npc_dialogue", "Greetings, guardian of the ancient stone arch.",
         "Ah, you have arrived at the Echoing Glade, traveler; I have been waiting for one who seeks the ancient knowledge.", 2.2),
    ]
    a_objs = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in _bpe_tokenize_raw_pairs_worker(anchor_raw)]
    for _ in range(3):
        org.train_wave_with_neuromodulator_reward(a_objs, wave_idx=1)

    unseen_q4 = "Why does the guardian at the Echoing Glade tell Alex to check subject-verb agreement while whisking egg yolk?"
    res_q4 = org.pure_neural_generate_response(unseen_q4, max_tokens=36, use_ngrams=False)
    out_q4 = res_q4["pure_weight_generated_response"]
    assert "egg yolk" in out_q4 and "Echoing Glade" in out_q4
    assert ", while it is possible to fix runny mayonnaise" in out_q4
    assert res_q4["ba44_syntax_bridge_connector"] == ", while"
    assert res_q4["swr_consolidated_into_nmda_matrix"] is True

    # Verify TPJ Deictic Perspective Shift (3rd person "His name is Alex, and he is building...")
    _ = org.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )
    unseen_q1 = "Alex is hungry in the lab while building a biological neural network—how can he fix runny liquid mayonnaise?"
    res_q1 = org.pure_neural_generate_response(unseen_q1, max_tokens=36, use_ngrams=False)
    out_q1 = res_q1["pure_weight_generated_response"]
    assert out_q1.startswith("His name is Alex, and he is building")
    assert ", while it is possible to fix runny mayonnaise" in out_q1


def test_v15_0_omniscient_5world_marathon():
    v15_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v15_0_omniscient_5world_master.pt"
    assert v15_ckpt.exists()
    v15_json = ROOT / "Produce" / "v15_0_omniscient_5world_report.json"
    assert v15_json.exists()
    rep15 = json.loads(v15_json.read_text(encoding="utf-8"))

    assert rep15["stored_sentence_chains_in_organism"] == 0
    assert rep15["total_tokens_trained"] > 1_000_000
    assert rep15["ca3_attractor_poles_learned"] >= 1000
    assert rep15["active_vocabulary_columns"] >= 7000

    # Verify all 5 cognitive worlds and all 11 datasets were actively streamed
    worlds = rep15["world_pair_counts"]
    for w_key in (
        "World_1_Stories_TinyStories",
        "World_2_Books_BookCorpus",
        "World_3_Web_Wiki_OpenWebText",
        "World_4_Movies_Persona_NPC",
        "World_5_Wisdom_OASST2_UltraChat",
    ):
        assert worlds.get(w_key, 0) > 1000, f"World {w_key} was not streamed!"

    ds = rep15["dataset_pair_counts"]
    for d_key in (
        "tinystories",
        "bookcorpus",
        "wikitext103",
        "openwebtext",
        "cornell_movies",
        "daily_dialog",
        "personachat",
        "personamem",
        "npc_dialogue",
        "oasst2",
        "ultrachat",
    ):
        assert ds.get(d_key, 0) > 0, f"Dataset {d_key} was not streamed!"

    exams = rep15["omniscient_5world_ngrams_off_exam"]
    assert len(exams) == 6
    assert "golden berries" in exams[0]["pure_weight_generated_response"]
    assert "Echoing Glade" in exams[0]["pure_weight_generated_response"]
    assert "leather-bound journal" in exams[1]["pure_weight_generated_response"]
    assert "apical tuft NMDA compartments" in exams[1]["pure_weight_generated_response"]
    assert exams[2]["pure_weight_generated_response"].startswith("His name is Alex, and he is building")
    assert "egg yolk" in exams[3]["pure_weight_generated_response"]
    assert ", before to fix grammar errors" in exams[4]["pure_weight_generated_response"]
    assert "basal dendrites" in exams[5]["pure_weight_generated_response"]


def test_v16_0_10B_omniscient_5epoch_engine():
    v16_ckpt = ROOT / "Produce" / "checkpoints" / "ziwago_v16_0_10B_omniscient_master.pt"
    assert v16_ckpt.exists()
    v16_json = ROOT / "Produce" / "v16_0_10B_omniscient_report.json"
    assert v16_json.exists()
    v16_ledger = ROOT / "Produce" / "v16_0_shard_offset_ledger.json"
    assert v16_ledger.exists()

    rep16 = json.loads(v16_json.read_text(encoding="utf-8"))
    ledger = json.loads(v16_ledger.read_text(encoding="utf-8"))

    assert rep16["stored_sentence_chains_in_organism"] == 0
    assert rep16["total_shards_and_zones_scanned"] >= 160
    assert rep16["total_raw_corpus_tokens_scanned"] > 500_000_000
    assert rep16["ca3_global_attractor_poles"] >= 1500
    assert rep16["active_vocabulary_columns"] >= 8000

    # Verify all 80 OpenWebText shards and all 5 epochs are recorded in the ledger
    assert len(ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"]) == 96  # 16 WikiText zones + 80 OpenWebText shards
    assert len(rep16["epochs"]) == 5

    banks = rep16["ca3_5bank_atlas_counts"]
    for b_name in (
        "Bank0_BA22_Stories",
        "Bank1_BA38_Books",
        "Bank2_BA7_BA39_Science_Web",
        "Bank3_BA44_BA45_Movies_Social",
        "Bank4_BA46_BA10_Wisdom_Assistant",
    ):
        assert banks.get(b_name, 0) > 100, f"CA3 Bank {b_name} was not populated!"

    exams = rep16["omniscient_5epoch_ngrams_off_exam"]
    assert len(exams) == 6
    assert "golden berries" in exams[0]["pure_weight_generated_response"]
    assert "Echoing Glade" in exams[0]["pure_weight_generated_response"]
    assert "leather-bound journal" in exams[1]["pure_weight_generated_response"]
    assert "apical tuft NMDA compartments" in exams[1]["pure_weight_generated_response"]
    assert exams[2]["pure_weight_generated_response"].startswith("His name is Alex, and he is building")
    assert "egg yolk" in exams[3]["pure_weight_generated_response"]
    assert ", before to fix grammar errors" in exams[4]["pure_weight_generated_response"]
    assert "basal dendrites" in exams[5]["pure_weight_generated_response"]













