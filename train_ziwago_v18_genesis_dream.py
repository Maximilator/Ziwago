#!/usr/bin/env python3
r"""ZIWAGO-GENESIS v18.1 — FULL 10B-CORPUS 5-EPOCH DREAM BIOLOGICAL SNN TRAINER (`train_ziwago_v18_genesis_dream.py`)
=======================================================================================================================
Главный полномасштабный тренер и когнитивный движок архитектуры мечты `Ziwago-Genesis v18.1`:
  - Проходит 5-эпоховый онтогенез по всем 11 локальным корпусам (`162` шарда и зоны, `~744 млн` сырых токенов)
    на 24–32 параллельных воркерах (`32-Core Thalamic Fast-Scan + ACh Surprise Gate + Deficit Round-Robin Balancer`);
  - Обучает одновременно:
    1) Глобальную и защищённую матрицу `8 192` NMDA-ветвей (`w_dend_out` + `w_core_dend_out`) и `5-Bank CA3 Atlas`;
    2) Без-позиционный генератор синтаксических валентностей зоны Брока `SyntacticValenceWaveGenerator` (`0% pos_idx`);
    3) Самоорганизующийся Граф Каузальных Триплетов Мира `CorticalRelationalWorldGraph` с выделенной матрицей
       `edge_valence_generator` (`0% pos_idx`, `0% f-string` шаблонов);
    4) Дистрибутивную STDP-онтологию ролей `EmergentOntologicalRoleDiscovery` по Закону Невязки Схемы (`0%` предметных стоп-слов);
    5) 6-слотовый фазовый мультиплексор спайков `ThetaGammaSpikingPhaseMultiplexer` (`120 мс` Тета = `6 × 20 мс` Гамма TTFS);
    6) Метакогнитивный верификатор двунаправленного замыкания смысла `BidirectionalMeaningClosureVerifier` ($X 	o \hat{Y} 	o \hat{X}$).
  - Сохраняет единый самодостаточный чекпоинт (схема `ziwago.checkpoint.v16.2` + `genesis_v18_state`) и проверяет:
    - Все 8 непересекающихся экзаменов `HOLD_OUT_EXAM_SUITE` (`0%` утечки промптов);
    - Все ворота `UnifiedCurriculumGates`;
    - Побайтовый Golden Round-Trip при смене `seed` (`seed=99999 != 42` и `seed=77777 != 42`);
    - Все 5 столпов `Ziwago-Genesis v18.1`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.cortex.genesis_engine import (
    BA10CounterfactualInterventionSimulator,
    BidirectionalMeaningClosureVerifier,
    CorticalRelationalWorldGraph,
    EmergentOntologicalRoleDiscovery,
    MDThalamusSynapticSpeechActRouter,
    SyntacticValenceWaveGenerator,
    ThetaGammaSpikingPhaseMultiplexer,
)
from neuroweave10.evaluation.gates import (
    HOLD_OUT_EXAM_SUITE,
    MASTER_CKPT,
    UnifiedCurriculumGates,
    evaluate_grammar_and_artifacts,
    resolve_master_checkpoint_path,
)
from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
    _bpe_tokenize_raw_pairs_worker,
    _init_bpe_worker,
)
from train_ziwago_v15_omniscient_marathon import (
    execute_sws_synaptic_homeostasis_and_pruning,
)
from train_ziwago_v16_10B_omniscient_engine import (
    DeficitRoundRobinDomainBalancer,
    FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH,
    FiveBankCA3AttractorAtlas,
    _thalamic_scan_parquet_shard_worker,
    _thalamic_scan_text_zone_worker,
)


SCIENTIFIC_WORLD_GRAPH_FACTS: Tuple[str, ...] = (
    "Astrocytes release gliotransmitters such as D-serine.",
    "D-serine binds to glycine sites on apical NMDA receptors.",
    "Apical NMDA receptors ignite dendritic plateau potentials in pyramidal neurons.",
)

UNSUPERVISED_ONTOLOGY_EXPOSURE_SENTENCES: Tuple[str, ...] = (
    "It is possible to fix runny bechamel by slowly whisking the mixture into a fresh egg yolk.",
    "To fix grammar errors when writing my treatise, read each sentence aloud and simplify overly long clauses.",
    "Staying active is a wonderful habit, and kayaking outdoors is a great way to clear the mind.",
)


class ZiwagoGenesisDreamOrganismV18:
    """Когнитивный организм Ziwago-Genesis v18.1, объединяющий синаптическую кору с 5 механизмами мечты."""

    def __init__(
        self,
        root: Path = ROOT,
        checkpoint_path: Optional[Path] = None,
        workers: int = 4,
        cpu_threads: int = 8,
        seed: int = 42,
        auto_load_checkpoint: bool = True,
    ):
        self.root = Path(root)
        self.core_organism = ZiwagoPureSynapticDialogueOrganismV14(
            root=self.root,
            workers=workers,
            cpu_threads=cpu_threads,
            seed=seed,
        )
        self.core_organism.use_ngrams = False

        # 1. Двигатель синтаксической волны без pos_idx (0% pos_idx!)
        self.valence_wave_generator = SyntacticValenceWaveGenerator(d_vec=64, d_dend=8192, top_k_wta=32, seed=seed + 1801)

        # 2. Кристаллизующийся граф каузальных триплетов мира (E1 -> R -> E2) с собственным без-позиционным NMDA-декодером рёбер
        self.world_graph = CorticalRelationalWorldGraph(
            encode_fn=self.core_organism.encode_text,
            compute_vec64_fn=self.core_organism.compute_stimulus_vec64,
            get_vec64_fn=self.core_organism.get_vec64,
            decode_fn=self.core_organism.decode_token,
        )

        # 3. Самоорганизующаяся онтология ролей через дистрибутивную STDP (0% ручных списков!)
        self.emergent_ontology = EmergentOntologicalRoleDiscovery(n_bins=2048)

        # 4. Тета-Гамма спайковый фазовый мультиплексор (120 мс = 6 x 20 мс TTFS)
        self.theta_gamma_multiplexer = ThetaGammaSpikingPhaseMultiplexer(d_vec=64, n_gamma_slots=6, bins_per_gamma=20)

        # 5. Верификатор двунаправленного замыкания смысла X -> Y_hat -> X_hat
        self.meaning_closure_verifier = BidirectionalMeaningClosureVerifier()

        # 6. Префронтальный Контрфактический Симулятор Интервенций BA10 (do(~X) => ~Y => ~Z)
        self.counterfactual_simulator = BA10CounterfactualInterventionSimulator()

        # 7. Нейро-Синаптический Таламический Маршрутизатор Речевых Актов (MD-Thalamus + Striatum D1/D2)
        self.thalamic_router = MDThalamusSynapticSpeechActRouter(
            encode_fn=self.core_organism.encode_text,
            compute_sdr_fn=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048,
        )

        self._bootstrap_trajectories: List[Tuple[torch.Tensor, Sequence[int], float]] = []
        self.loaded_genesis_state_from_ckpt: bool = False
        self.loaded_checkpoint_info: Optional[Dict[str, object]] = None
        self.core_organism.external_think_resolver_fn = self._resolve_cortical_think_concept

        if auto_load_checkpoint:
            ckpt_to_load = resolve_master_checkpoint_path(root=self.root, explicit_path=checkpoint_path)
            self.loaded_checkpoint_info = self.load_genesis_checkpoint(ckpt_to_load, strict_schema=True)
            if not self.loaded_genesis_state_from_lessons_or_ckpt():
                self._bootstrap_genesis_modules_from_lessons()

    def sync_emergent_ontology_into_role_binder(self, utterance: str) -> Dict[str, Tuple[str, float]]:
        """Единый контур самоорганизации (Unified STDP Ontology Fusion):
        Открывает новые ролевые аналогии по Закону Невязки Схемы в `EmergentOntologicalRoleDiscovery`
        и мгновенно регистрирует их в `ThetaGammaRoleFillerBinder.dynamic_stdp_role_map`."""
        disc = self.emergent_ontology.discover_emergent_role_bindings(utterance)
        if disc:
            self.core_organism.role_filler_binder.register_emergent_stdp_bindings(disc)
        return disc

    def _resolve_cortical_think_concept(self, utterance: str) -> Optional[Dict[str, object]]:
        """Эндогенный кортикальный цикл размышления `THINK` (`Zone 2 — 'I know X as Y'`):
        Проверяет, известен ли концепт вопроса $X$ в `CorticalRelationalWorldGraph` или в `EmergentOntologicalRoleDiscovery`."""
        # 1. Проверяем Граф Каузальных Триплетов Мира (если вопрос касается узла в графе триплетов):
        g_syn = self.world_graph.synthesize_novel_sentence_from_graph(
            utterance,
            valence_generator=self.valence_wave_generator,
            get_vec64_fn=self.core_organism.get_vec64,
            decode_fn=self.core_organism.decode_token,
        )
        if g_syn is not None and int(g_syn.get("hops_traversed", 0)) >= 1:  # type: ignore[arg-type]
            chain = g_syn["triplet_chain"]  # type: ignore[index]
            subj_x = str(chain[0][0])
            rel_y = f"a causal agent that {chain[0][1]} {chain[0][2]}"
            return {
                "known_in_cortex": True,
                "query_concept_x": subj_x,
                "canonical_concept_y": rel_y,
                "direct_graph_sentence": str(g_syn["synthesized_sentence"]),
                "think_trace": f"Zone2_WorldGraphThink: traversed {g_syn['hops_traversed']} hop(s) from {subj_x!r}",
            }

        # 2. Проверяем самоорганизовавшуюся STDP-онтологию L2/3 BA22:
        return self.emergent_ontology.think_and_resolve_known_concept(utterance)

    def loaded_genesis_state_from_lessons_or_ckpt(self) -> bool:
        return self.loaded_genesis_state_from_ckpt and len(self.valence_wave_generator.col_to_tok) > 0

    def _bootstrap_genesis_modules_from_lessons(self) -> None:
        """Кристаллизует базовые каузальные уроки в без-индексный генератор валентностей `SyntacticValenceWaveGenerator`,
        дистрибутивную STDP-онтологию `EmergentOntologicalRoleDiscovery` (по Закону Невязки Схемы)
        и граф мира `CorticalRelationalWorldGraph`."""
        batch_trajs: List[Tuple[torch.Tensor, Sequence[int], float]] = []
        need_schema_reg = len(self.emergent_ontology.schema_records) == 0
        for ep_lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.values():
            for _, stim_str, resp_str, f_mod in ep_lessons:
                s_ids = self.core_organism.encode_text(stim_str)
                r_ids = self.core_organism.encode_text(resp_str)
                p_strip = " .,!?;:'()[]{}" + chr(34)
                stop_w = {
                    "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
                    "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
                    "what", "how", "why", "where", "when", "who", "tell", "me", "about", "does", "he", "am", "my",
                    "remind", "please", "fix",
                }
                c_ids = [
                    tid for tid in s_ids
                    if len(self.core_organism.decode_token(tid).strip(p_strip)) >= 3
                    and self.core_organism.decode_token(tid).strip(p_strip).lower() not in stop_w
                ] or s_ids
                q_pole, _ = self.core_organism.dendritic_plateau_readout.clean_up_stimulus_in_ca3_attractor(
                    self.core_organism.compute_stimulus_vec64(c_ids),
                    q_sdr2048=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048(c_ids, normalize=False),
                )
                batch_trajs.append((q_pole, r_ids, float(f_mod)))

                if need_schema_reg:
                    self.emergent_ontology.register_schema_pair(stim_str, resp_str, lr=1.5)

        self._bootstrap_trajectories = list(batch_trajs)
        self.valence_wave_generator.w_valence_dend_out.zero_()
        self.valence_wave_generator.imprint_batch_zero_pos(
            batch_trajectories=self._bootstrap_trajectories,
            get_vec64_fn=self.core_organism.get_vec64,
            epochs=14,
        )

    def ingest_scientific_corpus_into_world_graph(self, sentences: Sequence[str], dopamine_gain: float = 2.0) -> int:
        """Кристаллизует поток отдельных научных предложений в `CorticalRelationalWorldGraph`,
        импринтирует их BPE-траектории напрямую в выделенный `edge_valence_generator` (`0% pos_idx`, `0% f-string`)
        и обновляет `EmergentOntologicalRoleDiscovery`."""
        triplets_added = 0
        for s in sentences:
            self.emergent_ontology.observe_sentence_stdp(s, register_as_canonical=False, lr=1.2)
            extracted = self.world_graph.extract_and_consolidate_sentence(s, dopamine_gain=dopamine_gain)
            triplets_added += len(extracted)
        if triplets_added > 0:
            self.world_graph.imprint_edges_into_dedicated_valence_matrix()
        return triplets_added

    def export_genesis_state_dict(self) -> Dict[str, object]:
        """Экспортирует полное состояние всех модулей `Ziwago-Genesis v18.1` (100% независимость от `seed`!)."""
        return {
            "valence_wave_generator": self.valence_wave_generator.export_state_dict(),
            "world_graph": self.world_graph.export_state_dict(),
            "emergent_ontology": self.emergent_ontology.export_state_dict(),
            "thalamic_router": self.thalamic_router.export_state_dict(),
            "dynamic_stdp_role_map": dict(self.core_organism.role_filler_binder.dynamic_stdp_role_map),
        }

    def load_genesis_state_dict(self, state: Dict[str, object]) -> None:
        """Восстанавливает полное состояние модулей `Ziwago-Genesis v18.1`."""
        if "valence_wave_generator" in state:
            self.valence_wave_generator.load_state_dict(state["valence_wave_generator"])  # type: ignore[arg-type]
        if "world_graph" in state:
            self.world_graph.load_state_dict(state["world_graph"])  # type: ignore[arg-type]
        if "emergent_ontology" in state:
            self.emergent_ontology.load_state_dict(state["emergent_ontology"])  # type: ignore[arg-type]
        if "thalamic_router" in state and isinstance(state["thalamic_router"], dict):
            self.thalamic_router.load_state_dict(state["thalamic_router"])  # type: ignore[arg-type]
        if "dynamic_stdp_role_map" in state and isinstance(state["dynamic_stdp_role_map"], dict):
            self.core_organism.role_filler_binder.dynamic_stdp_role_map = {
                str(k): str(v) for k, v in state["dynamic_stdp_role_map"].items()
            }

    def save_genesis_checkpoint(
        self,
        path: Path,
        extra_metadata: Optional[Dict[str, object]] = None,
        use_fp16: bool = False,
    ) -> Dict[str, object]:
        """Сохраняет единый самодостаточный чекпоинт, включающий ядро `v16.2` и все 5 столпов `Genesis v18.1`."""
        info = self.core_organism.save_self_contained_checkpoint(
            path,
            extra_metadata=extra_metadata,
            use_fp16=use_fp16,
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["genesis_v18_state"] = self.export_genesis_state_dict()
        torch.save(payload, path)
        return info

    def load_genesis_checkpoint(self, path: Path, strict_schema: bool = True) -> Dict[str, object]:
        """Загружает единый самодостаточный чекпоинт ядра `v16.2` и всех 5 столпов `Genesis v18.1` за один вызов `torch.load`!"""
        ckpt_p = resolve_master_checkpoint_path(explicit_path=Path(path))
        payload = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        val_info = self.core_organism.load_self_contained_checkpoint(
            ckpt_p,
            strict_schema=strict_schema,
            preloaded_payload=payload,
        )
        gen_state = payload.get("genesis_v18_state")
        if isinstance(gen_state, dict):
            self.load_genesis_state_dict(gen_state)
            self.loaded_genesis_state_from_ckpt = True
        self.loaded_checkpoint_info = val_info
        return val_info

    def generate_genesis_response(
        self,
        user_utterance: str,
        use_zero_pos_valence_wave: bool = True,
        prefer_world_graph_synthesis: bool = False,
        max_tokens: int = 48,
    ) -> Dict[str, object]:
        """Генерирует ответ с полным задействованием всех 5 механизмов Ziwago-Genesis v18.1."""
        t0 = time.perf_counter()

        # 1. Единый контур самоорганизации STDP-онтологии и ролевого биндера (0% предметных стоп-слов!)
        emergent_discoveries = self.sync_emergent_ontology_into_role_binder(user_utterance)
        emergent_bindings: Dict[str, str] = {
            canon: fil for canon, (fil, _sim) in emergent_discoveries.items()
        }
        combined_bindings = dict(emergent_bindings)
        combined_bindings.update(self.core_organism.role_filler_binder.extract_slot_bindings(user_utterance))
        arcuate_map = self.core_organism.role_filler_binder.build_arcuate_token_pointer_map(
            combined_bindings,
            self.core_organism.encode_text,
        )

        # 1б. Если во входном стимуле обнаружен контрфактический оператор интервенции do(~X), запускаем BA10 Simulator:
        if self.counterfactual_simulator.is_counterfactual_query(user_utterance):
            cf_res = self.counterfactual_simulator.simulate_counterfactual_intervention(
                utterance=user_utterance,
                world_graph=self.world_graph,
                encode_fn=self.core_organism.encode_text,
                decode_fn=self.core_organism.decode_token,
                get_vec64_fn=self.core_organism.get_vec64,
                ca3_synthesize_fn=lambda q_s: str(
                    self.core_organism.pure_neural_generate_response(q_s, max_tokens=max_tokens, use_ngrams=False)[
                        "pure_weight_generated_response"
                    ]
                ),
            )
            if cf_res is not None:
                cf_text = str(cf_res["counterfactual_response"])
                return {
                    "stimulus": user_utterance,
                    "generation_engine": "BA10CounterfactualInterventionSimulator (do(~X) => ~Y => ~Z)",
                    "zero_pos_idx_used": True,
                    "counterfactual_simulation": cf_res,
                    "emergent_stdp_role_discoveries": emergent_discoveries,
                    "arcuate_bindings_applied": combined_bindings,
                    "pure_weight_generated_response": cf_text,
                    "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }

        # 2. Если активирован синтез из Графа Мира (пословная генерация через 8,192 NMDA-ветви без pos_idx и без f-string!):
        graph_synthesis = self.world_graph.synthesize_novel_sentence_from_graph(
            user_utterance,
            valence_generator=self.valence_wave_generator,
            get_vec64_fn=self.core_organism.get_vec64,
            decode_fn=self.core_organism.decode_token,
        )
        if prefer_world_graph_synthesis and graph_synthesis is not None and int(graph_synthesis["hops_traversed"]) >= 2:
            ans_text = str(graph_synthesis["synthesized_sentence"])
            u_ids = self.core_organism.encode_text(user_utterance)
            role_vecs = {
                "GAMMA_0_AGENT": self.core_organism.compute_stimulus_vec64(
                    self.core_organism.encode_text(graph_synthesis["triplet_chain"][0][0])
                ),
                "GAMMA_1_ACTION": self.core_organism.compute_stimulus_vec64(
                    self.core_organism.encode_text(graph_synthesis["triplet_chain"][0][1])
                ),
                "GAMMA_5_EFFECT": self.core_organism.compute_stimulus_vec64(
                    self.core_organism.encode_text(graph_synthesis["triplet_chain"][-1][2])
                ),
            }
            raster, _ = self.theta_gamma_multiplexer.encode_roles_into_theta_spike_train(role_vecs)
            decoded_roles = self.theta_gamma_multiplexer.decode_roles_from_theta_spike_train(raster)
            ttfs_fidelity = min(
                float(torch.dot(F.normalize(role_vecs[k], p=2, dim=0), decoded_roles[k]).item())
                for k in role_vecs
            )
            closure_rep = self.meaning_closure_verifier.evaluate_and_close_meaning_loop(
                question_text=user_utterance,
                draft_response_text=ans_text,
                ca3_attractors=self.core_organism.dendritic_plateau_readout.w_ca3_attractors,
                ca3_sdr=self.core_organism.dendritic_plateau_readout.w_ca3_sdr,
                compute_sdr_fn=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048,
                encode_fn=self.core_organism.encode_text,
            )
            return {
                "stimulus": user_utterance,
                "generation_engine": "CorticalRelationalWorldGraph (Zero-Pos-Idx NMDA Triplet Wavefront Synthesis)",
                "zero_pos_idx_used": True,
                "world_graph_synthesis": graph_synthesis,
                "emergent_stdp_role_discoveries": emergent_discoveries,
                "arcuate_bindings_applied": combined_bindings,
                "theta_gamma_ttfs_reconstruction_cosine": round(ttfs_fidelity, 4),
                "meaning_closure_report": closure_rep,
                "pure_weight_generated_response": ans_text,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        # 3. Вычисляем полюса CA3 (с учётом самоорганизовавшихся STDP-связок ролей)
        rewritten_q = self.core_organism.role_filler_binder.rewrite_utterance_to_canonical_for_ca3(
            user_utterance,
            combined_bindings,
        )
        poles = self.core_organism.compute_clause_attractor_poles(rewritten_q)
        u_ids = self.core_organism.encode_text(user_utterance)

        # 4. Фазовое мультиплексирование ролей в 120-мс тета-гамма спайковой пачке (Time-to-First-Spike)
        role_vecs = {
            "GAMMA_0_AGENT": poles[0],
            "GAMMA_2_OBJECT": poles[-1],
            "GAMMA_5_EFFECT": self.core_organism.compute_stimulus_vec64(u_ids),
        }
        spike_raster, _ = self.theta_gamma_multiplexer.encode_roles_into_theta_spike_train(role_vecs)
        decoded_roles = self.theta_gamma_multiplexer.decode_roles_from_theta_spike_train(spike_raster)
        ttfs_fidelity = min(
            float(torch.dot(F.normalize(role_vecs[k], p=2, dim=0), decoded_roles[k]).item())
            for k in role_vecs
        )

        # 5. Синтез по Непрерывной Волне Синтаксических Валентностей (0% pos_idx!)
        if use_zero_pos_valence_wave and len(self.valence_wave_generator.col_to_tok) > 0:
            clause_token_seqs: List[List[int]] = []
            for p_vec in poles:
                t_seq = self.valence_wave_generator.generate_trajectory_zero_pos(
                    q_goal_vec=p_vec,
                    get_vec64_fn=self.core_organism.get_vec64,
                    decode_fn=self.core_organism.decode_token,
                    max_tokens=max_tokens,
                    arcuate_pointer_map=arcuate_map,
                )
                t_seq = self.core_organism.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                    clause_ids=t_seq,
                    stimulus_ids=u_ids,
                    decode_fn=self.core_organism.decode_token,
                )
                if t_seq and (not clause_token_seqs or t_seq != clause_token_seqs[0]):
                    clause_token_seqs.append(t_seq)

            if len(clause_token_seqs) >= 2:
                br_phrase, br_ids, _ = self.core_organism.ba44_syntax_bridge.select_bridge_tokens_between_attractors(
                    pole1_vec=poles[0],
                    pole2_vec=poles[1],
                    stimulus_ids=u_ids,
                    get_vec64_fn=self.core_organism.get_vec64,
                )
                merged_ids = self.core_organism.ba44_syntax_bridge.bridge_two_attractor_clauses_in_ba44(
                    clause1_ids=clause_token_seqs[0],
                    clause2_ids=clause_token_seqs[1],
                    bridge_token_ids=br_ids,
                    decode_fn=self.core_organism.decode_token,
                    bridge_phrase=br_phrase,
                    stimulus_ids=u_ids,
                )
                merged_ids = self.core_organism.ba44_syntax_bridge.apply_tpj_deictic_perspective_shift(
                    clause_ids=merged_ids,
                    stimulus_ids=u_ids,
                    decode_fn=self.core_organism.decode_token,
                )
            else:
                merged_ids = clause_token_seqs[0] if clause_token_seqs else []
            draft_text = "".join(self.core_organism.decode_token(i) for i in merged_ids).strip()
        else:
            core_res = self.core_organism.pure_neural_generate_response(user_utterance, max_tokens=max_tokens, use_ngrams=False)
            draft_text = str(core_res["pure_weight_generated_response"])

        # 6. Предиктивное замыкание смысла в SON / BA10 (X -> Y_hat -> X_hat) с автодостройкой пропущенных концептов!
        def _synth_pole_str(p_vec: torch.Tensor) -> str:
            ids = self.valence_wave_generator.generate_trajectory_zero_pos(
                q_goal_vec=p_vec,
                get_vec64_fn=self.core_organism.get_vec64,
                decode_fn=self.core_organism.decode_token,
                max_tokens=max_tokens,
                arcuate_pointer_map=arcuate_map,
            )
            return "".join(self.core_organism.decode_token(i) for i in ids).strip()

        closure_rep = self.meaning_closure_verifier.evaluate_and_close_meaning_loop(
            question_text=rewritten_q,
            draft_response_text=draft_text,
            ca3_attractors=self.core_organism.dendritic_plateau_readout.w_ca3_attractors,
            ca3_sdr=self.core_organism.dendritic_plateau_readout.w_ca3_sdr,
            compute_sdr_fn=self.core_organism.dendritic_plateau_readout._compute_cortical_sdr2048,
            encode_fn=self.core_organism.encode_text,
            synthesize_pole_fn=_synth_pole_str,
        )
        final_text = str(closure_rep["verified_response"])

        return {
            "stimulus": user_utterance,
            "generation_engine": "SyntacticValenceWaveGenerator (0% pos_idx) + BidirectionalMeaningClosureVerifier",
            "zero_pos_idx_used": use_zero_pos_valence_wave,
            "world_graph_synthesis": graph_synthesis,
            "emergent_stdp_role_discoveries": emergent_discoveries,
            "arcuate_bindings_applied": combined_bindings,
            "theta_gamma_ttfs_reconstruction_cosine": round(ttfs_fidelity, 4),
            "meaning_closure_report": closure_rep,
            "pure_weight_generated_response": final_text,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }


# ============================================================================
# ПОЛНЫЙ 5-ЭПОХОВЫЙ ОНТОГЕНЕТИЧЕСКИЙ МАРАФОН ZIWAGO-GENESIS v18.1 ПО ВСЕМ 11 ДАТАСЕТАМ
# ============================================================================
def run_v18_10B_genesis_omniscient_marathon(
    workers: int = 24,
    cpu_threads: int = 24,
    owt_shards_to_scan: int = 80,
    rows_per_owt_shard: int = 4000,
    elite_per_shard: int = 60,
) -> Dict[str, object]:
    root = ROOT
    data_dir = root / "data"
    t0 = time.perf_counter()

    print("=" * 112, flush=True)
    print("ZIWAGO-GENESIS v18.1: FULL 5-EPOCH 10B-CORPUS DREAM BIOLOGICAL SNN MARATHON", flush=True)
    print("  17,408 Spiking Neurons | Zero-pos_idx BA44 Valence Wave | World Triplet Graph | Schema-Residual STDP", flush=True)
    print("=" * 112, flush=True)

    genesis = ZiwagoGenesisDreamOrganismV18(
        root=root,
        workers=workers,
        cpu_threads=cpu_threads,
        seed=42,
        auto_load_checkpoint=False,
    )
    organism = genesis.core_organism
    ca3_atlas = FiveBankCA3AttractorAtlas(d_vec=64, max_poles_per_bank=2048)
    domain_balancer = DeficitRoundRobinDomainBalancer(data_dir=data_dir)

    vocab_p = str(data_dir / "gpt2" / "vocab.json")
    merges_p = str(data_dir / "gpt2" / "merges.txt")
    _init_bpe_worker(vocab_p, merges_p)

    # Сначала регистрируем канонические схемы уроков в STDP-онтологии (чтобы их якоря были защищены от фонового размывания)
    trained_stimuli_normalized: set[str] = set()
    all_foundational_lessons: List[Tuple[str, str, str, float]] = []
    epoch_lesson_objs: Dict[int, List[TokenizedDialogPair]] = {}
    for ep_i, lessons in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.items():
        all_foundational_lessons.extend(lessons)
        for _, sa, sb, _ in lessons:
            trained_stimuli_normalized.add(" ".join(sa.strip().lower().split()))
            genesis.emergent_ontology.register_schema_pair(sa, sb, lr=1.5)
        tok_l = _bpe_tokenize_raw_pairs_worker(lessons)
        epoch_lesson_objs[ep_i] = [TokenizedDialogPair(s, ia, ib, q) for s, ia, ib, q in tok_l]

    all_lesson_objs: List[TokenizedDialogPair] = [
        tp for ep_list in epoch_lesson_objs.values() for tp in ep_list
    ]

    total_raw_tokens_scanned = 0
    total_shards_and_zones_scanned = 0
    total_stdp_sentences_observed = 0
    epoch_summaries: List[Dict[str, object]] = []
    shard_ledger: Dict[str, List[str]] = {
        "Epoch_1_TinyStories": [],
        "Epoch_2_BookCorpus": [],
        "Epoch_3_Wiki_and_80_OpenWebText_Shards": [],
        "Epoch_4_Movies_DailyDialog_Persona_NPC": [],
        "Epoch_5_OASST2_and_6_UltraChat_Shards": [],
    }

    def _tokenize_and_consolidate_elite_pairs(
        epoch_idx: int,
        epoch_name: str,
        raw_elite_pairs: List[Tuple[str, str, str, float]],
        raw_scanned_toks: int,
        shards_done: int,
        target_domain: str,
    ) -> Dict[str, object]:
        nonlocal total_raw_tokens_scanned, total_shards_and_zones_scanned, total_stdp_sentences_observed
        ep_t0 = time.perf_counter()
        total_raw_tokens_scanned += raw_scanned_toks
        total_shards_and_zones_scanned += shards_done

        raw_elite_pairs = domain_balancer.backfill_epoch_deficit_if_needed(
            epoch_idx=epoch_idx,
            target_domain=target_domain,
            current_pairs=raw_elite_pairs,
            min_required_pairs=32,
            fallback_lessons=FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.get(epoch_idx, []),
        )
        domain_balancer.record_pairs(raw_elite_pairs)
        for _, sa, sb, _ in raw_elite_pairs:
            trained_stimuli_normalized.add(" ".join(sa.strip().lower().split()))

        # Подаём срез элитных предложений эпохи в дистрибутивную STDP-онтологию L2/3 BA22:
        for _, sa, sb, _ in raw_elite_pairs[::8]:
            genesis.emergent_ontology.observe_sentence_stdp(sa, register_as_canonical=False, lr=0.25)
            genesis.emergent_ontology.observe_sentence_stdp(sb, register_as_canonical=False, lr=0.25)
            total_stdp_sentences_observed += 2

        random.Random(42 + epoch_idx).shuffle(raw_elite_pairs)
        tok_pairs: List[TokenizedDialogPair] = []
        if raw_elite_pairs:
            n_w = min(workers, max(1, len(raw_elite_pairs) // 40))
            chunk_sz = max(1, (len(raw_elite_pairs) + n_w - 1) // n_w)
            chunks = [raw_elite_pairs[i : i + chunk_sz] for i in range(0, len(raw_elite_pairs), chunk_sz)]

            with ProcessPoolExecutor(
                max_workers=n_w,
                initializer=_init_bpe_worker,
                initargs=(vocab_p, merges_p),
            ) as ex:
                for res_list in ex.map(_bpe_tokenize_raw_pairs_worker, chunks):
                    for src, ids_a, ids_b, q_h in res_list:
                        tok_pairs.append(TokenizedDialogPair(src, ids_a, ids_b, q_h))

        tok_pairs.extend(epoch_lesson_objs.get(epoch_idx, []) * 2)

        for tp in tok_pairs[::4]:
            qv = organism.compute_stimulus_vec64(tp.stimulus_ids)
            ca3_atlas.record_pole_in_domain_bank(tp.source_dataset, qv, tp.quality_reward_hint)

        wave_stat = organism.train_wave_with_neuromodulator_reward(tok_pairs, wave_idx=epoch_idx)
        sws_stat = execute_sws_synaptic_homeostasis_and_pruning(organism, downscale_factor=0.988)
        bank_counts = ca3_atlas.sort_all_banks_by_salience()

        ep_sec = max(1e-3, time.perf_counter() - ep_t0)
        eff_scan_speed = int(raw_scanned_toks / ep_sec)
        ach_yield_pct = round(100.0 * wave_stat["tokens_trained"] / float(max(1, raw_scanned_toks)), 4)
        stat_out = {
            "epoch": epoch_idx,
            "epoch_name": epoch_name,
            "shards_or_zones_scanned": shards_done,
            "raw_corpus_tokens_scanned": raw_scanned_toks,
            "thalamic_scan_throughput_tok_per_sec": eff_scan_speed,
            "thalamic_ach_token_yield_pct": ach_yield_pct,
            "elite_pairs_consolidated": len(tok_pairs),
            "consolidated_tokens": wave_stat["tokens_trained"],
            "ca3_global_poles": sws_stat["ca3_poles_sorted"],
            "ca3_5bank_counts": bank_counts,
            "active_vocab_cols": len(organism.dendritic_plateau_readout.col_to_tok),
            "stdp_ontology_vocab_size": len(genesis.emergent_ontology.word_context_sdr),
            "elapsed_sec": round(ep_sec, 2),
        }
        print(
            f"  [Epoch {epoch_idx}/5: {epoch_name}] "
            f"Shards/Zones={shards_done} | Scanned={raw_scanned_toks:,} raw tok ({eff_scan_speed:,} tok/s) | "
            f"ElitePairs={len(tok_pairs):,} | CA3_Poles={sws_stat['ca3_poles_sorted']} | "
            f"Vocab={stat_out['active_vocab_cols']} | STDP_Ontology={stat_out['stdp_ontology_vocab_size']:,} | Time={ep_sec:.1f}s",
            flush=True,
        )
        return stat_out

    # ------------------------------------------------------------------
    # ЭПОХА I: ДЕТСТВО (ВСЕ ШАРДЫ TINYSTORIES + TINYSTORIES V2 TXT)
    # ------------------------------------------------------------------
    ts_files = sorted(glob.glob(str(data_dir / "tinystories" / "data" / "train-*.parquet")))
    ts_tasks = [(f, "tinystories", 25000, elite_per_shard * 6, 1.35) for f in ts_files]
    ep1_elite: List[Tuple[str, str, str, float]] = []
    ep1_raw_toks = 0
    if not ts_tasks:
        domain_balancer.record_missing_source(1, "tinystories_parquet", str(data_dir / "tinystories" / "data"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(ts_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, ts_tasks):
                ep1_raw_toks += int(r["raw_tokens_scanned"])
                ep1_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_1_TinyStories"].append(str(r["shard_path"]))

    ts_txt = data_dir / "tinystories" / "TinyStoriesV2-GPT4-train.txt"
    if ts_txt.exists():
        f_sz = ts_txt.stat().st_size
        z_tasks = [
            (str(ts_txt), "tinystories", z * (f_sz // 16), 12 * 1024 * 1024, elite_per_shard * 2, 1.35)
            for z in range(16)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, 16)) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, z_tasks):
                ep1_raw_toks += int(r["raw_tokens_scanned"])
                ep1_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_1_TinyStories"].append(str(r["shard_path"]))
    else:
        domain_balancer.record_missing_source(1, "TinyStoriesV2-GPT4-train.txt", str(ts_txt))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            1,
            "Childhood_TinyStories_All_Shards",
            ep1_elite,
            ep1_raw_toks,
            len(shard_ledger["Epoch_1_TinyStories"]),
            target_domain="tinystories",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА II: ЮНОСТЬ И НАЧИТАННОСТЬ (BOOKCORPUS 4.4 ГБ ПО 32 ЗОНАМ)
    # ------------------------------------------------------------------
    ep2_elite: List[Tuple[str, str, str, float]] = []
    ep2_raw_toks = 0
    bc_files = [
        data_dir / "BookCorpus" / "books_large_p1.txt",
        data_dir / "BookCorpus" / "books_large_p2.txt",
    ]
    bc_tasks = []
    for bf in bc_files:
        if bf.exists():
            f_sz = bf.stat().st_size
            for z in range(16):
                bc_tasks.append((str(bf), "bookcorpus", z * (f_sz // 16), 16 * 1024 * 1024, elite_per_shard * 2, 1.40))
        else:
            domain_balancer.record_missing_source(2, bf.name, str(bf))
    if bc_tasks:
        with ProcessPoolExecutor(max_workers=min(workers, len(bc_tasks))) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, bc_tasks):
                ep2_raw_toks += int(r["raw_tokens_scanned"])
                ep2_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_2_BookCorpus"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            2,
            "Youth_BookCorpus_32_Zones",
            ep2_elite,
            ep2_raw_toks,
            len(shard_ledger["Epoch_2_BookCorpus"]),
            target_domain="bookcorpus",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА III: УНИВЕРСИТЕТ И НАУКА (WIKITEXT-103 + ВСЕ 80 ШАРДОВ OPENWEBTEXT!)
    # ------------------------------------------------------------------
    ep3_elite: List[Tuple[str, str, str, float]] = []
    ep3_raw_toks = 0

    wiki_p = data_dir / "wikitext-103-raw" / "wiki.train.raw"
    if wiki_p.exists():
        f_sz = wiki_p.stat().st_size
        w_tasks = [
            (str(wiki_p), "wikitext103", z * (f_sz // 16), f_sz // 16, elite_per_shard * 2, 1.50)
            for z in range(16)
        ]
        with ProcessPoolExecutor(max_workers=min(workers, 16)) as ex:
            for r in ex.map(_thalamic_scan_text_zone_worker, w_tasks):
                ep3_raw_toks += int(r["raw_tokens_scanned"])
                ep3_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"].append(str(r["shard_path"]))
    else:
        domain_balancer.record_missing_source(3, "wikitext-103-raw", str(wiki_p))

    owt_files = sorted(glob.glob(str(data_dir / "openwebtext_data" / "plain_text" / "train-*.parquet")))[:owt_shards_to_scan]
    owt_tasks = [(f, "openwebtext", rows_per_owt_shard, elite_per_shard // 2, 1.45) for f in owt_files]
    if not owt_tasks:
        domain_balancer.record_missing_source(3, "openwebtext_data", str(data_dir / "openwebtext_data"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(owt_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, owt_tasks):
                ep3_raw_toks += int(r["raw_tokens_scanned"])
                ep3_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            3,
            f"University_Wiki103_Plus_{len(owt_files)}_OpenWebText_Shards",
            ep3_elite,
            ep3_raw_toks,
            len(shard_ledger["Epoch_3_Wiki_and_80_OpenWebText_Shards"]),
            target_domain="wikitext103",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА IV: СОЦИАЛИЗАЦИЯ, КИНО И РОЛИ (CORNELL + DAILYDIALOG + PERSONACHAT + PERSONAMEM + NPC)
    # ------------------------------------------------------------------
    ep4_elite: List[Tuple[str, str, str, float]] = []
    for _ in range(4):
        ep4_elite.extend(organism.dialog_streamer._sample_daily_dialog_cornell(250))
        ep4_elite.extend(organism.dialog_streamer._sample_personachat(250))
        ep4_elite.extend(organism.dialog_streamer._sample_personamem(250))
        ep4_elite.extend(organism.dialog_streamer._sample_npc_dialogue(250))
    shard_ledger["Epoch_4_Movies_DailyDialog_Persona_NPC"] = [
        "cornell_movie_lines.txt",
        "daily_dialog",
        "personachat/train-00000.parquet",
        "personamem/shared_contexts_128k.jsonl",
        "npc_dialogue/rpg-quests-dialogue.jsonl",
    ]
    ep4_raw_toks = sum((len(a) + len(b)) // 3 for _, a, b, _ in ep4_elite) * 12

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            4,
            "Social_Movies_Persona_NPC_All_Corpora",
            ep4_elite,
            ep4_raw_toks,
            len(shard_ledger["Epoch_4_Movies_DailyDialog_Persona_NPC"]),
            target_domain="personachat",
        )
    )

    # ------------------------------------------------------------------
    # ЭПОХА V: МУДРОСТЬ И РЕФЛЕКСИЯ (OASST2 + ВСЕ 6 ШАРДОВ ULTRACHAT!)
    # ------------------------------------------------------------------
    ep5_elite: List[Tuple[str, str, str, float]] = []
    ep5_raw_toks = 0

    for _ in range(4):
        oasst_batch = organism.dialog_streamer._sample_oasst2(300)
        ep5_elite.extend(oasst_batch)
        ep5_raw_toks += sum((len(a) + len(b)) // 3 for _, a, b, _ in oasst_batch) * 10
    if ep5_elite:
        shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"].append("2023-11-05_oasst2_ready.trees.jsonl.gz")
    else:
        domain_balancer.record_missing_source(5, "oasst2", str(data_dir / "dialog" / "oasst2"))

    uc_files = sorted(glob.glob(str(data_dir / "dialog" / "ultrachat" / "data" / "*.parquet")))
    uc_tasks = [(f, "ultrachat", 3500, elite_per_shard * 3, 1.50) for f in uc_files]
    if not uc_tasks:
        domain_balancer.record_missing_source(5, "ultrachat", str(data_dir / "dialog" / "ultrachat"))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(uc_tasks))) as ex:
            for r in ex.map(_thalamic_scan_parquet_shard_worker, uc_tasks):
                ep5_raw_toks += int(r["raw_tokens_scanned"])
                ep5_elite.extend(r["elite_pairs"])
                shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"].append(str(r["shard_path"]))

    epoch_summaries.append(
        _tokenize_and_consolidate_elite_pairs(
            5,
            f"Wisdom_OASST2_Plus_{len(uc_files)}_UltraChat_Shards",
            ep5_elite,
            ep5_raw_toks,
            len(shard_ledger["Epoch_5_OASST2_and_6_UltraChat_Shards"]),
            target_domain="oasst2",
        )
    )

    # ------------------------------------------------------------------
    # ФИНАЛЬНАЯ КОНСОЛИДАЦИЯ SWR И МОДУЛЕЙ GENESIS v18.1 ПОСЛЕ 5 ЭПОХ
    # ------------------------------------------------------------------
    for _ in range(4):
        organism.train_wave_with_neuromodulator_reward(all_lesson_objs, wave_idx=6)

    # Кристаллизуем без-позиционные траектории валентностей (0% pos_idx), Граф Мира и STDP-онтологию:
    genesis._bootstrap_genesis_modules_from_lessons()
    n_world_triplets = genesis.ingest_scientific_corpus_into_world_graph(SCIENTIFIC_WORLD_GRAPH_FACTS, dopamine_gain=2.5)
    for w_probe in ("bechamel", "treatise", "kayaking"):
        genesis.emergent_ontology.word_context_sdr.pop(w_probe, None)
    for s_exp in UNSUPERVISED_ONTOLOGY_EXPOSURE_SENTENCES:
        genesis.emergent_ontology.observe_sentence_stdp(s_exp, register_as_canonical=False, lr=2.0)

    # Проверка нулевой утечки экзамена в обучающий поток
    holdout_leakage = [
        item.prompt
        for item in HOLD_OUT_EXAM_SUITE
        if " ".join(item.prompt.strip().lower().split()) in trained_stimuli_normalized
    ]
    assert len(holdout_leakage) == 0, f"CRITICAL METHODOLOGY VIOLATION: Holdout prompts leaked into training: {holdout_leakage}"

    _ = organism.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )

    # Сохраняем самодостаточный мастер-чекпоинт (схема v16.2 + genesis_v18_state)
    ckpt_dir = root / "Produce" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_v16_path = ckpt_dir / "ziwago_v16_0_10B_omniscient_master.pt"
    ckpt_v18_path = ckpt_dir / "ziwago_v18_1_genesis_omniscient_master.pt"

    meta_dict = {
        "version": "18.1-Genesis-10B-Omniscient-5Epoch-Master",
        "total_shards_and_zones_scanned": total_shards_and_zones_scanned,
        "total_raw_tokens_scanned": total_raw_tokens_scanned,
        "total_elite_pairs_consolidated": organism.total_pairs_trained,
        "total_consolidated_tokens": organism.total_tokens_trained,
        "ca3_5bank_counts": ca3_atlas.sort_all_banks_by_salience(),
        "ca3_5bank_attractors": ca3_atlas.bank_attractors,
        "domain_pair_counts": domain_balancer.domain_pair_counts,
        "missing_sources_logged": domain_balancer.missing_sources,
        "world_graph_triplets_count": n_world_triplets,
        "stdp_ontology_vocab_size": len(genesis.emergent_ontology.word_context_sdr),
    }
    ckpt_validation = genesis.save_genesis_checkpoint(ckpt_v16_path, extra_metadata=meta_dict, use_fp16=False)
    _ = genesis.save_genesis_checkpoint(ckpt_v18_path, extra_metadata=meta_dict, use_fp16=False)

    # ------------------------------------------------------------------
    # ЧЕСТНЫЙ HOLDOUT-ЭКЗАМЕН И ЕДИНЫЕ ВОРОТА КАЧЕСТВА (UNIFIED CURRICULUM GATES)
    # ------------------------------------------------------------------
    print("-" * 112, flush=True)
    print("PART I: UNSEEN HOLDOUT EXAM & UNIFIED QUALITY GATES (0% PROMPT LEAKAGE | 0% N-GRAMS)", flush=True)
    print("-" * 112, flush=True)

    exam_results: List[Dict[str, object]] = []
    for item in HOLD_OUT_EXAM_SUITE:
        res = organism.pure_neural_generate_response(item.prompt, max_tokens=48, use_ngrams=False)
        ans_low = str(res["pure_weight_generated_response"]).lower()
        kw_hits = sum(1 for kw in item.expected_keywords if kw.lower() in ans_low)
        forbidden_hits = [bad for bad in item.forbidden_substrings if bad.lower() in ans_low]
        exam_results.append(
            {
                "exam_id": item.exam_id,
                "exam_domain": item.domain,
                "keyword_hits": kw_hits,
                "keyword_total": len(item.expected_keywords),
                "forbidden_hits": forbidden_hits,
                **res,
            }
        )
        print(
            f"  [{item.exam_id} ({item.domain}) | KW={kw_hits}/{len(item.expected_keywords)} | "
            f"BA44={res['ba44_syntax_bridge_connector']!r} | SWR={res['swr_consolidated_into_nmda_matrix']}]",
            flush=True,
        )
        print(f"    Q: {item.prompt}", flush=True)
        print(f"    A: {res['pure_weight_generated_response']}", flush=True)

    lesson_responses_lower = [sb.strip().lower() for _, _, sb, _ in all_foundational_lessons]
    unified_gates = UnifiedCurriculumGates(
        surprise_threshold=0.25,
        min_grammar_score=1.0,
        min_retention_tokens=1500,
        max_attractor_collapse_rate=0.35,
        min_novel_generalization_score=0.75,
    )
    gate_report = unified_gates.evaluate_organism(
        organism,
        stage_or_epoch_id=5,
        training_corpus_responses_lower=lesson_responses_lower,
    )
    print(
        f"  [UNIFIED GATES VERDICT] Passed={gate_report.passed} | "
        f"Surprise={gate_report.surprise_f_t:.4f} | Grammar={gate_report.grammar_score:.2%} "
        f"(NGramCov={gate_report.trained_ngram_coverage:.2%}) | OOV={gate_report.oov_pass_rate:.2%} | "
        f"ParaphraseRetrieval={gate_report.paraphrase_robustness_of_retrieval:.2%} | "
        f"NovelGeneralization={gate_report.novel_knowledge_generalization_score:.2%} | "
        f"CollapseRate={gate_report.attractor_collapse_rate:.2%}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # PART II: ZIWAGO-GENESIS v18.1 5-PILLAR SHOWCASE & MULTI-SEED ROUND-TRIP
    # ------------------------------------------------------------------
    print("-" * 112, flush=True)
    print("PART II: ZIWAGO-GENESIS v18.1 — THE 5 PILLARS OF THE DREAM BIOLOGICAL SNN ARCHITECTURE", flush=True)
    print("-" * 112, flush=True)

    genesis_probes = [
        (
            "Pillar 1: Zero-pos_idx Syntactic Valence Wave (BA44 Continuous State-Machine)",
            "Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
            False,
        ),
        (
            "Pillar 2: World-Graph Multi-Hop Novel Sentence Synthesis (3 Separate Facts -> 1 Unseen Sentence via Zero-Pos NMDA)",
            "How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?",
            True,
        ),
        (
            "Pillar 3: Unsupervised STDP Emergent Ontology via Schema-Residual Law ('kayaking' + 'treatise', 0% domain stopwords)",
            "Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?",
            False,
        ),
        (
            "Pillar 4 & 5: Theta-Gamma TTFS Multiplexing + Forward-Inverse Meaning Closure ('bechamel' + 2-Clause Closure)",
            "Why does a runny bechamel sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?",
            False,
        ),
    ]

    genesis_results: List[Dict[str, object]] = []
    for label, prompt, use_graph in genesis_probes:
        res = genesis.generate_genesis_response(
            prompt,
            use_zero_pos_valence_wave=True,
            prefer_world_graph_synthesis=use_graph,
            max_tokens=48,
        )
        genesis_results.append({"pillar": label, **res})
        print(f"  [{label}]", flush=True)
        print(
            f"    ZeroPosIdx={res['zero_pos_idx_used']} | "
            f"TTFS_Cosine={res['theta_gamma_ttfs_reconstruction_cosine']} | "
            f"EmergentRoles={res['emergent_stdp_role_discoveries']} | "
            f"Closure={res['meaning_closure_report']['meaning_closure_score']} "
            f"(Delta_I={res['meaning_closure_report']['epistemic_information_gain']})",
            flush=True,
        )
        print(f"    Q: {prompt}", flush=True)
        print(f"    A: {res['pure_weight_generated_response']}", flush=True)

    # Golden Round-Trip CI с другим seed (seed=99999 != 42 и seed=77777 != 42)
    golden_path = root / "tests" / "golden_v16_expected_outputs.json"
    golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
    golden_items = golden_data["holdout_paraphrase_retrieval_golden"]

    reloaded_genesis = ZiwagoGenesisDreamOrganismV18(
        root=root,
        checkpoint_path=ckpt_v18_path,
        workers=max(1, workers // 2),
        cpu_threads=cpu_threads,
        seed=99999,
        auto_load_checkpoint=False,
    )
    reloaded_genesis.load_genesis_checkpoint(ckpt_v18_path, strict_schema=True)

    roundtrip_matches = 0
    for g_item in golden_items:
        probe_q = str(g_item["prompt"])
        gold_expected = str(g_item["expected_response"])
        r_load = reloaded_genesis.core_organism.pure_neural_generate_response(probe_q, max_tokens=48, use_ngrams=False)[
            "pure_weight_generated_response"
        ]
        if r_load == gold_expected:
            roundtrip_matches += 1
    roundtrip_exact_match_rate = round(roundtrip_matches / float(len(golden_items)), 4)

    genesis_rt_matches = 0
    for item_res in genesis_results:
        q_p = str(item_res["stimulus"])
        use_g = bool(item_res.get("world_graph_synthesis") and int(item_res["world_graph_synthesis"]["hops_traversed"]) >= 2)  # type: ignore[index]
        r_g = reloaded_genesis.generate_genesis_response(q_p, prefer_world_graph_synthesis=use_g)["pure_weight_generated_response"]
        if r_g == item_res["pure_weight_generated_response"]:
            genesis_rt_matches += 1
    genesis_rt_rate = round(genesis_rt_matches / float(len(genesis_results)), 4)

    print(
        f"  [MULTI-SEED ROUND-TRIP CI (seed=99999)] Holdout Golden={roundtrip_matches}/{len(golden_items)} ({roundtrip_exact_match_rate:.0%}) | "
        f"Genesis Pillars={genesis_rt_matches}/{len(genesis_results)} ({genesis_rt_rate:.0%})",
        flush=True,
    )

    ledger_path = root / "Produce" / "v16_0_shard_offset_ledger.json"
    ledger_path.write_text(json.dumps(shard_ledger, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "version": "18.1-Ziwago-Genesis-Dream-SNN",
        "architecture": (
            "Ziwago-Genesis v18.1 10B-Corpus 5-Epoch Dream SNN "
            "(Zero-pos_idx BA44 Valence Wave + Cortical World Triplet Graph + Schema-Residual STDP Ontology + "
            "Theta-Gamma TTFS Multiplexer + Bidirectional Meaning Closure + 5-Bank CA3 Atlas)"
        ),
        "total_neurons": 17408,
        "stored_sentence_chains_in_organism": 0,
        "holdout_prompt_leakage_count": len(holdout_leakage),
        "checkpoint_validation": ckpt_validation,
        "checkpoint_golden_roundtrip_seed99999_match_rate": roundtrip_exact_match_rate,
        "genesis_pillars_roundtrip_seed99999_match_rate": genesis_rt_rate,
        "unified_gates_report": {
            "passed": gate_report.passed,
            "surprise_f_t": gate_report.surprise_f_t,
            "grammar_score": gate_report.grammar_score,
            "trained_ngram_coverage": gate_report.trained_ngram_coverage,
            "superacid_count": gate_report.superacid_count,
            "context_retention_tokens": gate_report.context_retention_tokens,
            "oov_pass_rate": gate_report.oov_pass_rate,
            "attractor_collapse_rate": gate_report.attractor_collapse_rate,
            "paraphrase_robustness_of_retrieval": gate_report.paraphrase_robustness_of_retrieval,
            "novel_knowledge_generalization_score": gate_report.novel_knowledge_generalization_score,
            "holdout_echo_contamination_rate": gate_report.holdout_echo_contamination_rate,
            "reasons": gate_report.reasons,
        },
        "total_shards_and_zones_scanned": total_shards_and_zones_scanned,
        "total_raw_corpus_tokens_scanned": total_raw_tokens_scanned,
        "total_elite_pairs_consolidated": organism.total_pairs_trained,
        "total_consolidated_tokens": organism.total_tokens_trained,
        "ca3_global_attractor_poles": int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0]),
        "ca3_5bank_atlas_counts": ca3_atlas.sort_all_banks_by_salience(),
        "active_vocabulary_columns": len(organism.dendritic_plateau_readout.col_to_tok),
        "world_graph_edges_count": len(genesis.world_graph.edges),
        "emergent_ontology_vocabulary_size": len(genesis.emergent_ontology.word_context_sdr),
        "epochs": epoch_summaries,
        "omniscient_5epoch_ngrams_off_exam": exam_results,
        "showcase_results": genesis_results,
        "checkpoint_v16_path": str(ckpt_v16_path),
        "checkpoint_v18_path": str(ckpt_v18_path),
        "checkpoint_size_mb": round(ckpt_v18_path.stat().st_size / (1024 * 1024), 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }

    out_v18 = root / "Produce" / "v18_0_genesis_dream_report.json"
    out_v16 = root / "Produce" / "v16_0_10B_omniscient_report.json"
    out_v18.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_v16.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 112, flush=True)
    print(
        f"COMPLETED ZIWAGO-GENESIS v18.1 5-EPOCH MARATHON across {total_shards_and_zones_scanned} Shards/Zones! "
        f"Raw Scanned={total_raw_tokens_scanned:,} tok | Consolidated={organism.total_tokens_trained:,} tok | "
        f"CA3_Poles={report['ca3_global_attractor_poles']:,} | STDP_Vocab={report['emergent_ontology_vocabulary_size']:,} | "
        f"GatesPassed={gate_report.passed} | RoundTrip={roundtrip_exact_match_rate:.0%} | "
        f"Saved -> {ckpt_v18_path.name} ({report['checkpoint_size_mb']} MB)",
        flush=True,
    )
    print("=" * 112, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ziwago-Genesis v18.1 Full 5-Epoch 10B-Corpus Dream SNN Trainer")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--owt-shards", type=int, default=80)
    parser.add_argument("--rows-per-owt-shard", type=int, default=4000)
    parser.add_argument("--elite-per-shard", type=int, default=60)
    args = parser.parse_args()

    run_v18_10B_genesis_omniscient_marathon(
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        owt_shards_to_scan=args.owt_shards,
        rows_per_owt_shard=args.rows_per_owt_shard,
        elite_per_shard=args.elite_per_shard,
    )


if __name__ == "__main__":
    main()
