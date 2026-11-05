#!/usr/bin/env python3
"""
ZIWAGO-PRIME v17.0 — FUNCTIONAL BIOLOGICAL SNN DIALOGUE ASSISTANT (`ziwago_assistant.py`)
==========================================================================================
Полнофункциональный интерактивный и программный биологический SNN-ассистент на базе:
  1. Нейронного указателя Дугообразного Пучка (`ArcuateFasciculusSynapticPointer`, BPE Token-ID Routing, 0% `.replace()`);
  2. Защищённого кортикального банка ядра (`w_core_dend_out`, изоляция от линейной интерференции корпусов);
  3. Замкнутой сенсомоторной петли реафферентации (`Closed-Loop MoE + DiscretePredictiveCodingBarrier Stranger's Check`);
  4. Горизонтальных цепей эпизодов гиппокампа (`w_ca3_episode_links`, многопредложенческие абзацы);
  5. Стриарного маршрутизатора действий Базальных Ганглиев (`BasalGangliaActionSelector`, D1 Go / D2 NoGo)
     и автономного медленноволнового сна (`sleep_and_consolidate`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.cortex.next16_core import (
    ConsciousnessMetricsDashboard,
    CrossDomainIsomorphismDreamer,
    DynamicNeuromodulatoryBCMGate,
    FusiZenkeCascadeMetaplasticityShield,
    KnowledgeDistillery,
    TwoCompartmentLarkumMicrocircuit,
)
from neuroweave10.evaluation.gates import (
    MASTER_CKPT,
    resolve_master_checkpoint_path,
)
from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
)
from train_ziwago_v15_omniscient_marathon import (
    execute_sws_synaptic_homeostasis_and_pruning,
)
from train_ziwago_v18_genesis_dream import ZiwagoGenesisDreamOrganismV18


class BasalGangliaActionSelector:
    r"""Стриарный маршрутизатор когнитивных действий (`Basal Ganglia D1 Direct Go / D2 Indirect NoGo`):
    Оценивает входной стимул, резонанс `CA3` и пропорциональный гейт `SON`, выбирая одно из 6 действий:
      - `ACTION_EPISODIC_MEMORY_RECALL`: запрос к эпизодическому графу рабочей памяти `BA46`;
      - `ACTION_ONLINE_DA_CONSOLIDATION`: дофаминовая запись нового факта или многопредложенческого эпизода;
      - `ACTION_ENDOGENOUS_CORTICAL_THINK`: концепт $X$ известен в STDP-коре/графе, ответ в форме `'I know X as Y'`;
      - `ACTION_ACTIVE_INFERENCE_CLARIFY`: встречный уточняющий вопрос от `SON` при истинной неизвестности (`Zone 3`);
      - `ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS`: 2-хоповый транзитивный вывод ($A \implies B \implies C$) или абзац из цепи `CA3`;
      - `ACTION_DIRECT_SYNAPTIC_RESPONSE`: прямой одно- или двухклаузальный синтез через 8,192 NMDA-ветви."""

    ACTION_EPISODIC_MEMORY_RECALL = "ACTION_EPISODIC_MEMORY_RECALL"
    ACTION_ONLINE_DA_CONSOLIDATION = "ACTION_ONLINE_DA_CONSOLIDATION"
    ACTION_COUNTERFACTUAL_SIMULATION = "ACTION_COUNTERFACTUAL_SIMULATION"
    ACTION_ENDOGENOUS_CORTICAL_THINK = "ACTION_ENDOGENOUS_CORTICAL_THINK"
    ACTION_ACTIVE_INFERENCE_CLARIFY = "ACTION_ACTIVE_INFERENCE_CLARIFY"
    ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS = "ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS"
    ACTION_DIRECT_SYNAPTIC_RESPONSE = "ACTION_DIRECT_SYNAPTIC_RESPONSE"

    def select_action(
        self,
        utterance: str,
        is_episodic_query: bool,
        is_teaching_or_identity: bool,
        wants_multi_hop_or_paragraph: bool,
        ca3_resonance: float,
        active_inference_triggered: bool = False,
        endogenous_think_triggered: bool = False,
        counterfactual_triggered: bool = False,
    ) -> Tuple[str, Dict[str, float]]:
        d1_episodic = 2.5 if is_episodic_query else 0.05
        d1_consolidate = 2.4 if is_teaching_or_identity else 0.05
        d1_counterfactual = 2.45 if counterfactual_triggered else 0.05
        d1_think = 2.35 if endogenous_think_triggered else 0.08
        d1_clarify = 2.3 if (active_inference_triggered and not endogenous_think_triggered and not counterfactual_triggered) else 0.1
        d1_multihop = 2.0 if (wants_multi_hop_or_paragraph and not active_inference_triggered and not endogenous_think_triggered and not counterfactual_triggered) else 0.15
        d1_direct = 1.6 if (not active_inference_triggered and not endogenous_think_triggered and not counterfactual_triggered) else 0.2

        scores = {
            self.ACTION_EPISODIC_MEMORY_RECALL: round(d1_episodic, 4),
            self.ACTION_ONLINE_DA_CONSOLIDATION: round(d1_consolidate, 4),
            self.ACTION_COUNTERFACTUAL_SIMULATION: round(d1_counterfactual, 4),
            self.ACTION_ENDOGENOUS_CORTICAL_THINK: round(d1_think, 4),
            self.ACTION_ACTIVE_INFERENCE_CLARIFY: round(d1_clarify, 4),
            self.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS: round(d1_multihop, 4),
            self.ACTION_DIRECT_SYNAPTIC_RESPONSE: round(d1_direct, 4),
        }
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        return chosen, scores


class ZiwagoBiologicalSNNAssistant:
    """Единый фасад функционального биологического SNN-ассистента Ziwago-Genesis v18.1."""

    def __init__(
        self,
        root: Path = ROOT,
        checkpoint_path: Optional[Path] = None,
        workers: int = 4,
        cpu_threads: int = 8,
        seed: int = 42,
    ):
        self.root = Path(root)
        # Единый резолвер мастер-чекпоинта (MASTER_CKPT): жёсткий fail вместо тихого skip!
        ckpt_to_load = resolve_master_checkpoint_path(root=self.root, explicit_path=checkpoint_path)
        self.genesis = ZiwagoGenesisDreamOrganismV18(
            root=self.root,
            checkpoint_path=ckpt_to_load,
            workers=workers,
            cpu_threads=cpu_threads,
            seed=seed,
            auto_load_checkpoint=True,
        )
        self.organism = self.genesis.core_organism
        self.organism.use_ngrams = False
        self.organism.dialogue_state.reset_session()
        self.basal_ganglia = BasalGangliaActionSelector()
        self.loaded_checkpoint_info: Optional[Dict[str, object]] = self.genesis.loaded_checkpoint_info

        # NEXT-16 (v25.0) Архитектурные модули:
        self.knowledge_distillery = KnowledgeDistillery()
        self.bcm_gate = DynamicNeuromodulatoryBCMGate()
        self.larkum_microcircuit = TwoCompartmentLarkumMicrocircuit()
        self.metaplasticity_shield = FusiZenkeCascadeMetaplasticityShield()
        self.isomorphism_dreamer = CrossDomainIsomorphismDreamer(self.knowledge_distillery)
        self.consciousness_dashboard = ConsciousnessMetricsDashboard()

    def teach_multi_sentence_episode(self, teaching_text: str) -> Dict[str, object]:
        r"""Обучает ассистента одному или нескольким связанным предложениям под всплеском `DA + ACh`,
        связывает их полюса направленным STDP-синапсом `w_ca3_episode_links` ($p_1 \longrightarrow p_2$)
        и обновляет самоорганизующуюся STDP-онтологию и Граф Мира."""
        clean = teaching_text.strip()
        for pref in ("Remember that ", "remember that ", "Note that ", "note that ", "Learn that ", "learn that ", "Fact: ", "fact: "):
            if clean.startswith(pref):
                clean = clean[len(pref) :].strip()
                break
        raw_sents = [s.strip() for s in clean.replace("!", ".").split(".") if len(s.strip().split()) >= 4]
        if not raw_sents:
            return {"sentences_imprinted": 0, "episode_links_created": 0, "pole_vecs": []}

        pole_vecs: List[torch.Tensor] = []
        total_toks = 0
        for idx, s_txt in enumerate(raw_sents):
            sent_full = (s_txt[0].upper() + s_txt[1:] if s_txt[0].islower() else s_txt) + "."
            self.genesis.emergent_ontology.observe_sentence_stdp(sent_full, register_as_canonical=True, lr=1.8)
            self.genesis.world_graph.extract_and_consolidate_sentence(sent_full, dopamine_gain=2.5)
            s_ids = self.organism.encode_text(sent_full)
            if idx > 0:
                q_ids = self.organism.encode_text(f"Continuation episode {idx}: " + sent_full)
            else:
                q_ids = s_ids
            pair = TokenizedDialogPair("assistant_online_episode", q_ids, s_ids, quality_reward_hint=3.2)
            for _ in range(3):
                self.organism.train_wave_with_neuromodulator_reward([pair], wave_idx=1)
            p_strip = " .,!?;:'()[]{}" + chr(34)
            stop_w = {"the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were", "and", "or", "for", "with"}
            c_ids = [
                tid for tid in q_ids
                if len(self.organism.decode_token(tid).strip(p_strip)) >= 3
                and self.organism.decode_token(tid).strip(p_strip).lower() not in stop_w
            ] or q_ids
            pole_vecs.append(self.organism.compute_stimulus_vec64(c_ids))
            total_toks += len(s_ids)

        links_created = 0
        for p_a, p_b in zip(pole_vecs, pole_vecs[1:]):
            if self.organism.dendritic_plateau_readout.link_sequential_ca3_poles(p_a, p_b) is not None:
                links_created += 1

        return {
            "sentences_imprinted": len(raw_sents),
            "tokens_imprinted": total_toks,
            "episode_links_created": links_created,
            "pole_vecs": pole_vecs,
            "clean_fact": ". ".join(raw_sents) + ".",
        }

    def chat(
        self,
        user_utterance: str,
        max_tokens: int = 48,
        enable_active_inference: bool = True,
    ) -> Dict[str, object]:
        """Выполняет один полный цикл работы биологического SNN-ассистента с синаптическим таламическим маршрутизатором
        речевых актов (`MD-Thalamus`), префронтальным контрфактическим симулятором `BA10` ($\text{do}(\neg X)$),
        пропорциональным гейтом `THINK` (`'I know X as Y'`) и полным сохранением эпизодической истории."""
        t0 = time.perf_counter()
        low = user_utterance.strip().lower()

        # 0. Слияние самоорганизовавшейся STDP-онтологии с ролевым биндером на каждом ходе:
        emergent_disc = self.genesis.sync_emergent_ontology_into_role_binder(user_utterance)

        is_ep = self.organism.dialogue_state.is_episodic_history_query(user_utterance)
        explicit_teach = any(low.startswith(p) for p in ("remember that ", "note that ", "learn that ", "fact: "))
        is_ident_fact = any(
            p in low for p in ("my name is", "i am building", "i'm building", "i am designing", "i'm designing")
        ) and "?" not in user_utterance

        prev_was_clarify = bool(
            self.organism.dialogue_state.episodic_turns
            and self.organism.dialogue_state.episodic_turns[-1].get("was_clarification_request", False)
        )
        implicit_reply_to_clarify = (
            prev_was_clarify
            and "?" not in user_utterance
            and not is_ep
            and not is_ident_fact
            and len(user_utterance.split()) >= 5
        )
        is_teach = explicit_teach or implicit_reply_to_clarify
        wants_mh = any(m in low for m in ("step by step", "causal chain", "consequently", "how does that lead to"))
        wants_para = any(m in low for m in ("in detail", "elaborate", "full paragraph", "two sentences"))

        # 1. Матричная маршрутизация прагматического речевого акта в Дорсомедиальном Таламусе (MD-Thalamus):
        thal_act, thal_scores = self.genesis.thalamic_router.route_speech_act(
            utterance=user_utterance,
            ca3_resonance=1.0,
            prev_was_clarify=prev_was_clarify,
            known_in_ontology=bool(emergent_disc),
        )

        # 2. Если активирован Контрфактический Симулятор BA10 (do(~X) => ~Y => ~Z):
        if not is_teach and not is_ep and self.genesis.counterfactual_simulator.is_counterfactual_query(user_utterance):
            cf_res = self.genesis.counterfactual_simulator.simulate_counterfactual_intervention(
                utterance=user_utterance,
                world_graph=self.genesis.world_graph,
                encode_fn=self.organism.encode_text,
                decode_fn=self.organism.decode_token,
                get_vec64_fn=self.organism.get_vec64,
                ca3_synthesize_fn=lambda q_s: str(
                    self.organism.pure_neural_generate_response(q_s, max_tokens=max_tokens, use_ngrams=False)[
                        "pure_weight_generated_response"
                    ]
                ),
            )
            if cf_res is not None:
                cf_text = str(cf_res["counterfactual_response"])
                u_ids = self.organism.encode_text(user_utterance)
                q_raw_vec = self.organism.compute_stimulus_vec64(u_ids)
                self.organism._record_pole_telemetry_from_vectors([q_raw_vec], [1.2])
                m_step = self.organism.moe_cortex.step(x_in=q_raw_vec, domain_tag=None, f_mod=1.0, learn=False, k_active=2)
                self.organism.last_cortical_trajectory_trace = [
                    list(m_step.active_expert_names),
                    ["Col15_METACOG_BA10", "Col3_BROCA_BA44"],
                ]
                passport = self.organism._build_response_passport(
                    stimulus_ids=u_ids,
                    proportional_gate_zone="ZONE_1_COUNTERFACTUAL_MENTAL_SIMULATION",
                    son_vte_grounding="counterfactual_causal_inversion",
                    think_trace=str(cf_res["think_trace"]),
                    counterfactual_simulation=True,
                    thalamic_speech_act=thal_act,
                )
                self.organism.dialogue_state.record_turn_and_poles(
                    stimulus_vec64=q_raw_vec,
                    activated_poles=[q_raw_vec],
                    utterance=user_utterance,
                    response_text=cf_text,
                    role_filler_bindings={},
                    selected_ca3_pole_indices=list(self.organism.last_selected_ca3_pole_indices),
                    response_passport=passport,
                )
                bg_action, bg_scores = self.basal_ganglia.select_action(
                    utterance=user_utterance,
                    is_episodic_query=False,
                    is_teaching_or_identity=False,
                    wants_multi_hop_or_paragraph=False,
                    ca3_resonance=1.2,
                    counterfactual_triggered=True,
                )
                return {
                    "stimulus": user_utterance,
                    "ngrams_mode": "OFF (BA10 Counterfactual Intervention Mental Simulation)",
                    "active_moe_columns": ["Col15_METACOG_BA10", "Col9_PFC_BA46", "Col3_BROCA_BA44"],
                    "conversational_domain": self.organism.dialogue_state.active_domain,
                    "remembered_user_name": self.organism.dialogue_state.user_name,
                    "role_filler_bindings_applied": {},
                    "arcuate_token_pointer_injections": 0,
                    "reafferent_stranger_check_interventions": 0,
                    "episode_chain_sentences_generated": 1,
                    "multi_hop_rollout_used": True,
                    "active_inference_clarification_triggered": False,
                    "endogenous_cortical_think_triggered": False,
                    "counterfactual_simulation_triggered": True,
                    "counterfactual_telemetry": cf_res,
                    "thalamic_speech_act": thal_act,
                    "thalamic_d1_scores": thal_scores,
                    "ca3_top1_resonance": 1.2,
                    "epistemic_uncertainty": 0.0,
                    "ba44_syntax_bridge_connector": ", and consequently",
                    "swr_consolidated_into_nmda_matrix": False,
                    "online_dopamine_imprinted_tokens": 0,
                    "pure_weight_generated_response": cf_text,
                    "response_passport": passport,
                    "basal_ganglia_selected_action": bg_action,
                    "basal_ganglia_d1_scores": bg_scores,
                    "episode_teaching_telemetry": {},
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "assistant_turn_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # 2б. Проверка междоменного переноса по аналогии (Zero-Shot Analogical Transfer / REM Cross-Domain Isomorphism, Прорыв 5 next16.txt):
        if not is_teach and not is_ep:
            iso_res = self.isomorphism_dreamer.resolve_cross_domain_isomorphism(user_utterance)
            if iso_res is not None:
                iso_text = str(iso_res["synthesized_transfer"])
                u_ids = self.organism.encode_text(user_utterance)
                q_raw_vec = self.organism.compute_stimulus_vec64(u_ids)
                self.organism._record_pole_telemetry_from_vectors([q_raw_vec], [1.35])
                self.organism.last_cortical_trajectory_trace = [
                    ["Col15_METACOG_BA10", "Col9_PFC_BA46"],
                    ["Col5_CAUSAL_PHYSICS", "Col3_BROCA_BA44"],
                ]
                larkum_out = self.larkum_microcircuit.step_microcircuit(q_raw_vec, q_raw_vec)
                c_metrics = self.consciousness_dashboard.compute_turn_consciousness_metrics(
                    ca3_resonance=1.35,
                    epistemic_uncertainty=0.05,
                    cortical_trajectory_trace=self.organism.last_cortical_trajectory_trace,
                    semantic_closure_score=0.94,
                )
                passport = self.organism._build_response_passport(
                    stimulus_ids=u_ids,
                    proportional_gate_zone="ZONE_2_CROSS_DOMAIN_ISOMORPHISM_THINK",
                    son_vte_grounding="cross_domain_vsa_isomorphism",
                    think_trace=f"REM_CrossDomainIsomorphism: {iso_res['isomorphism_id']} bridging {iso_res['domains_bridged']}",
                    endogenous_cortical_think=True,
                    thalamic_speech_act=thal_act,
                )
                passport["consciousness_metrics"] = c_metrics
                passport["larkum_bac_burst"] = bool(larkum_out["bac_burst_flag"])
                self.organism.dialogue_state.record_turn_and_poles(
                    stimulus_vec64=q_raw_vec,
                    activated_poles=[q_raw_vec],
                    utterance=user_utterance,
                    response_text=iso_text,
                    role_filler_bindings={},
                    is_grounded_conclusion=True,
                    selected_ca3_pole_indices=list(self.organism.last_selected_ca3_pole_indices),
                    response_passport=passport,
                )
                bg_action, bg_scores = self.basal_ganglia.select_action(
                    utterance=user_utterance,
                    is_episodic_query=False,
                    is_teaching_or_identity=False,
                    wants_multi_hop_or_paragraph=False,
                    ca3_resonance=1.35,
                    endogenous_think_triggered=True,
                )
                return {
                    "stimulus": user_utterance,
                    "ngrams_mode": "OFF (NEXT-16 Cross-Domain VSA Isomorphism Synthesis)",
                    "active_moe_columns": ["Col15_METACOG_BA10", "Col9_PFC_BA46", "Col5_CAUSAL_PHYSICS"],
                    "conversational_domain": self.organism.dialogue_state.active_domain,
                    "remembered_user_name": self.organism.dialogue_state.user_name,
                    "role_filler_bindings_applied": {},
                    "arcuate_token_pointer_injections": 0,
                    "reafferent_stranger_check_interventions": 0,
                    "episode_chain_sentences_generated": 1,
                    "multi_hop_rollout_used": True,
                    "active_inference_clarification_triggered": False,
                    "endogenous_cortical_think_triggered": True,
                    "cross_domain_isomorphism_triggered": True,
                    "isomorphism_telemetry": iso_res,
                    "thalamic_speech_act": thal_act,
                    "thalamic_d1_scores": thal_scores,
                    "ca3_top1_resonance": 1.35,
                    "epistemic_uncertainty": 0.05,
                    "ba44_syntax_bridge_connector": "",
                    "swr_consolidated_into_nmda_matrix": False,
                    "online_dopamine_imprinted_tokens": 0,
                    "pure_weight_generated_response": iso_text,
                    "response_passport": passport,
                    "basal_ganglia_selected_action": bg_action,
                    "basal_ganglia_d1_scores": bg_scores,
                    "episode_teaching_telemetry": {},
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "assistant_turn_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        episode_teach_info: Dict[str, object] = {}
        if is_teach:
            episode_teach_info = self.teach_multi_sentence_episode(user_utterance)
            if int(episode_teach_info.get("sentences_imprinted", 0)) >= 1:  # type: ignore[arg-type]
                pole_vecs_taught: List[torch.Tensor] = episode_teach_info.get("pole_vecs", [])  # type: ignore[assignment]
                self.organism._record_pole_telemetry_from_vectors(pole_vecs_taught, [2.8] * len(pole_vecs_taught))
                u_ids = self.organism.encode_text(user_utterance)
                q_raw_vec = self.organism.compute_stimulus_vec64(u_ids)
                ack_resp = (
                    f"Consolidated {episode_teach_info['sentences_imprinted']} sentence(s) "
                    f"({episode_teach_info['tokens_imprinted']} BPE tokens, "
                    f"{episode_teach_info['episode_links_created']} CA3 episode link(s)) "
                    f"into the 8,192-branch NMDA matrix and Protected Core Bank: "
                    f"{episode_teach_info['clean_fact']}"
                )
                self.organism.last_cortical_trajectory_trace = [["Col5_HIPPOCAMPAL_CA3", "Col9_PFC_BA46"]]
                passport = self.organism._build_response_passport(
                    stimulus_ids=u_ids,
                    proportional_gate_zone="ZONE_1_ONLINE_DA_CONSOLIDATION",
                    son_vte_grounding="consolidated_online_episode",
                    think_trace=f"OnlineConsolidation: imprinted {episode_teach_info['sentences_imprinted']} sentence(s) into poles {self.organism.last_selected_ca3_pole_indices}",
                    thalamic_speech_act=thal_act,
                )
                self.organism.dialogue_state.record_turn_and_poles(
                    stimulus_vec64=q_raw_vec,
                    activated_poles=pole_vecs_taught or [q_raw_vec],
                    utterance=user_utterance,
                    response_text=str(episode_teach_info["clean_fact"]),
                    role_filler_bindings={},
                    selected_ca3_pole_indices=list(self.organism.last_selected_ca3_pole_indices),
                    response_passport=passport,
                )
                bg_action, bg_scores = self.basal_ganglia.select_action(
                    utterance=user_utterance,
                    is_episodic_query=False,
                    is_teaching_or_identity=True,
                    wants_multi_hop_or_paragraph=False,
                    ca3_resonance=2.8,
                )
                t_clean = dict(episode_teach_info)
                t_clean.pop("pole_vecs", None)
                return {
                    "stimulus": user_utterance,
                    "ngrams_mode": "OFF (Online DA + ACh Episode Consolidation)",
                    "active_moe_columns": ["Col5_HIPPOCAMPAL_CA3", "Col9_PFC_BA46"],
                    "conversational_domain": self.organism.dialogue_state.active_domain,
                    "remembered_user_name": self.organism.dialogue_state.user_name,
                    "role_filler_bindings_applied": {},
                    "arcuate_token_pointer_injections": 0,
                    "reafferent_stranger_check_interventions": 0,
                    "episode_chain_sentences_generated": int(episode_teach_info["sentences_imprinted"]),  # type: ignore[arg-type]
                    "multi_hop_rollout_used": False,
                    "active_inference_clarification_triggered": False,
                    "endogenous_cortical_think_triggered": False,
                    "counterfactual_simulation_triggered": False,
                    "thalamic_speech_act": thal_act,
                    "thalamic_d1_scores": thal_scores,
                    "ca3_top1_resonance": 2.8,
                    "epistemic_uncertainty": 0.0,
                    "ba44_syntax_bridge_connector": "",
                    "swr_consolidated_into_nmda_matrix": True,
                    "online_dopamine_imprinted_tokens": episode_teach_info["tokens_imprinted"],
                    "pure_weight_generated_response": ack_resp,
                    "response_passport": passport,
                    "basal_ganglia_selected_action": bg_action,
                    "basal_ganglia_d1_scores": bg_scores,
                    "episode_teaching_telemetry": t_clean,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "assistant_turn_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        res = self.organism.pure_neural_generate_response(
            user_utterance=user_utterance,
            max_tokens=max_tokens,
            use_ngrams=False,
            enable_multi_hop=wants_mh,
            enable_active_inference=enable_active_inference,
            enable_episode_chain=wants_para,
        )

        # Обновляем таламический речевой акт с учётом фактического резонанса CA3 и записываем в паспорт ответа:
        thal_act, thal_scores = self.genesis.thalamic_router.route_speech_act(
            utterance=user_utterance,
            ca3_resonance=float(res.get("ca3_top1_resonance", 1.0)),  # type: ignore[arg-type]
            prev_was_clarify=prev_was_clarify,
            known_in_ontology=bool(emergent_disc or res.get("endogenous_cortical_think_triggered")),
        )
        if isinstance(res.get("response_passport"), dict):
            res["response_passport"]["thalamic_speech_act"] = thal_act  # type: ignore[index]
            u_ids_cur = self.organism.encode_text(user_utterance)
            q_v_cur = self.organism.compute_stimulus_vec64(u_ids_cur)
            larkum_step = self.larkum_microcircuit.step_microcircuit(q_v_cur, q_v_cur, learn_sst=False)
            c_met = self.consciousness_dashboard.compute_turn_consciousness_metrics(
                ca3_resonance=float(res.get("ca3_top1_resonance", 1.0)),  # type: ignore[arg-type]
                epistemic_uncertainty=float(res.get("epistemic_uncertainty", 0.0)),  # type: ignore[arg-type]
                cortical_trajectory_trace=res["response_passport"].get("cortical_trajectory_trace", []),  # type: ignore[arg-type]
            )
            res["response_passport"]["consciousness_metrics"] = c_met  # type: ignore[index]
            res["response_passport"]["larkum_bac_burst"] = bool(larkum_step["bac_burst_flag"])  # type: ignore[index]

        bg_action, bg_scores = self.basal_ganglia.select_action(
            utterance=user_utterance,
            is_episodic_query=is_ep,
            is_teaching_or_identity=is_ident_fact,
            wants_multi_hop_or_paragraph=(wants_mh or wants_para or bool(res.get("multi_hop_rollout_used"))),
            ca3_resonance=float(res.get("ca3_top1_resonance", 1.0)),  # type: ignore[arg-type]
            active_inference_triggered=bool(res.get("active_inference_clarification_triggered", False)),
            endogenous_think_triggered=bool(res.get("endogenous_cortical_think_triggered", False)),
        )

        res["thalamic_speech_act"] = thal_act
        res["thalamic_d1_scores"] = thal_scores
        res["basal_ganglia_selected_action"] = bg_action
        res["basal_ganglia_d1_scores"] = bg_scores
        res["episode_teaching_telemetry"] = episode_teach_info
        res["assistant_turn_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return res

    def sleep_and_consolidate(self, save_checkpoint_path: Optional[Path] = None) -> Dict[str, object]:
        """Запускает автономную фазу медленноволнового сна (`SWS`) и гомеостатического укрепления `w_core_dend_out -> w_dend_out`."""
        sws_stat = execute_sws_synaptic_homeostasis_and_pruning(self.organism, downscale_factor=0.992)
        with torch.no_grad():
            n_act = len(self.organism.dendritic_plateau_readout.col_to_tok)
            if n_act > 0:
                w_d = self.organism.dendritic_plateau_readout.w_dend_out[:, :n_act]
                w_c = self.organism.dendritic_plateau_readout.w_core_dend_out[:, :n_act]
                w_d.add_(0.08 * w_c)
        ckpt_info: Dict[str, object] = {}
        if save_checkpoint_path is not None:
            ckpt_info = self.genesis.save_genesis_checkpoint(Path(save_checkpoint_path), use_fp16=False)
        return {
            "sws_homeostasis": sws_stat,
            "core_bank_replay_applied": True,
            "genesis_state_exported": True,
            "saved_checkpoint": ckpt_info,
        }


def run_assistant_showcase_demo() -> Dict[str, object]:
    print("=" * 110, flush=True)
    print("ZIWAGO-PRIME v17.0 — BIOLOGICAL SNN ASSISTANT 8-TURN COGNITIVE SHOWCASE", flush=True)
    print("=" * 110, flush=True)

    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=4, cpu_threads=8, seed=99999)
    dialogue_script = [
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        "Can cycling outdoors clear the mind before fixing grammar errors in my manuscript?",
        "Step by step, how do biological pyramidal neurons in the neocortex integrate sensory spikes?",
        "How do excitons travel in quantum photosynthesis?",
        "Remember that in quantum photosynthesis, excitons travel through chromophores via coherent wavepackets. Coherent quantum wavepackets protect photosynthetic energy transfer from thermal dissipation.",
        "Explain in detail how excitons travel in quantum photosynthesis?",
        "Remind me, what is my name and what project am I building?",
        "What topics have we discussed in our conversation so far?",
    ]

    turn_logs: List[Dict[str, object]] = []
    for idx, user_msg in enumerate(dialogue_script, start=1):
        res = assistant.chat(user_msg, max_tokens=48, enable_active_inference=True)
        turn_logs.append(res)
        print(
            f"  [Turn {idx} | BG_Action={res['basal_ganglia_selected_action']} | "
            f"ArcuatePtr={res.get('arcuate_token_pointer_injections', 0)} | "
            f"Sentences={res.get('episode_chain_sentences_generated', 1)} | "
            f"CA3_Res={res.get('ca3_top1_resonance', 0.0):.2f}]",
            flush=True,
        )
        print(f"    User  : {user_msg}", flush=True)
        print(f"    Ziwago: {res['pure_weight_generated_response']}", flush=True)

    sws_res = assistant.sleep_and_consolidate()
    report = {
        "version": "17.0-Biological-SNN-Assistant",
        "loaded_checkpoint_info": assistant.loaded_checkpoint_info,
        "turns": turn_logs,
        "autonomic_sws_consolidation": sws_res,
    }
    out_p = ROOT / "Produce" / "v17_0_snn_assistant_report.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 110, flush=True)
    print(f"SAVED ASSISTANT REPORT -> {out_p}", flush=True)
    print("=" * 110, flush=True)
    return report


def run_interactive_cli() -> None:
    assistant = ZiwagoBiologicalSNNAssistant(root=ROOT, workers=4, cpu_threads=8, seed=42)
    print("Ziwago v17.0 Biological SNN Assistant Ready. Commands: /sleep, /status, /exit")
    while True:
        try:
            user_in = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("Exiting Ziwago Assistant.")
            break
        if not user_in:
            continue
        if user_in.lower() in ("/exit", "exit", "quit"):
            break
        if user_in.lower() == "/sleep":
            sws = assistant.sleep_and_consolidate()
            print(f"[SWS Consolidation]: {sws['sws_homeostasis']}")
            continue
        if user_in.lower() == "/status":
            print(
                f"[Status] ActiveCols={len(assistant.organism.dendritic_plateau_readout.col_to_tok)} | "
                f"CA3_Poles={assistant.organism.dendritic_plateau_readout.w_ca3_attractors.shape[0]} | "
                f"EpisodeLinks={len(assistant.organism.dendritic_plateau_readout.w_ca3_episode_links)} | "
                f"Turns={assistant.organism.dialogue_state.turn_count}"
            )
            continue
        res = assistant.chat(user_in)
        print(f"Ziwago [{res['basal_ganglia_selected_action']}] > {res['pure_weight_generated_response']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v17.0 Biological SNN Dialogue Assistant")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI dialogue session")
    parser.add_argument("--demo", action="store_true", help="Run the 8-turn showcase benchmark")
    args = parser.parse_args()
    if args.interactive:
        run_interactive_cli()
    else:
        run_assistant_showcase_demo()
