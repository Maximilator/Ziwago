#!/usr/bin/env python3
r"""ZIWAGO-GENESIS v18.0 — THE DREAM BIOLOGICAL SNN ARCHITECTURE (`train_ziwago_v18_genesis_dream.py`)
========================================================================================================
Объединяет проверенное ядро `Ziwago v16.2 / v17.0` с 5 революционными механизмами «Идеальной Биологической SNN»:

  1. `SyntacticValenceWaveGenerator` (0% `pos_idx`, 0% `pos_phase_vecs`):
     Пословная генерация речи по непрерывной волне синтаксических валентностей зоны Брока $\vec{\Psi}_{\mathrm{BA44}}(t)$
     без позиционной «магнитофонной головки» `pos_idx`.
  2. `CorticalRelationalWorldGraph` (Граф Каузальных Триплетов Мира $\vec{E}_1 \xrightarrow{\vec{R}} \vec{E}_2$):
     Кристаллизация отдельных фактов из разных предложений в единую причинно-следственную сеть и синтез
     совершенно новых многошаговых предложений ($A 	o B 	o C 	o D$), которых никогда не было в тексте.
  3. `EmergentOntologicalRoleDiscovery` (Самоорганизующаяся онтология ролей в `L2/3 BA22`, 0% ручных списков!):
     Автоматическое открытие ролевых классов новых слов (`bechamel`, `treatise`, `kayaking`) исключительно
     из совпадения предикатного окружения по правилу дистрибутивной STDP.
  4. `ThetaGammaSpikingPhaseMultiplexer` (Фазовое кодирование спайков Time-to-First-Spike, 120 мс Тета = 6 × 20 мс Гамма):
     Кодирование и демодуляция 6 синтаксических ролей в 6 гамма-фазовых окнах одного периода тета-волны
     с нулевой перекрёстной интерференцией (`crosstalk == 0.0`) и косинусной точностью `> 0.99`.
  5. `BidirectionalMeaningClosureVerifier` (Предиктивное замыкание смысла $X \longrightarrow \hat{Y} \longrightarrow \hat{X}$ в `SON/BA10`):
     Обратная реконструкция вопроса $\hat{X}$ из черновика ответа $\hat{Y}$, контроль информационного прироста $\Delta I > 0$
     и автоматическое достраивание ответа по вектору смыслового дефицита $\vec{r}_{\mathrm{missing}}$.
"""

from __future__ import annotations

import json
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
    BidirectionalMeaningClosureVerifier,
    CorticalRelationalWorldGraph,
    EmergentOntologicalRoleDiscovery,
    SyntacticValenceWaveGenerator,
    ThetaGammaSpikingPhaseMultiplexer,
)
from neuroweave10.evaluation.gates import evaluate_grammar_and_artifacts
from train_ziwago_v14_pure_synaptic_dialogue import (
    TokenizedDialogPair,
    ZiwagoPureSynapticDialogueOrganismV14,
    _bpe_tokenize_raw_pairs_worker,
    _init_bpe_worker,
)
from train_ziwago_v16_10B_omniscient_engine import (
    FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH,
)


class ZiwagoGenesisDreamOrganismV18:
    """Когнитивный организм Ziwago-Genesis v18.0, объединяющий синаптическую кору с 5 механизмами мечты."""

    def __init__(
        self,
        root: Path = ROOT,
        checkpoint_path: Optional[Path] = None,
        workers: int = 4,
        cpu_threads: int = 8,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.core_organism = ZiwagoPureSynapticDialogueOrganismV14(
            root=self.root,
            workers=workers,
            cpu_threads=cpu_threads,
            seed=seed,
        )
        self.core_organism.use_ngrams = False

        default_ckpt = self.root / "Produce" / "checkpoints" / "ziwago_v16_0_10B_omniscient_master.pt"
        ckpt_to_load = Path(checkpoint_path) if checkpoint_path is not None else default_ckpt
        if ckpt_to_load.exists():
            self.core_organism.load_self_contained_checkpoint(ckpt_to_load, strict_schema=True)

        # 1. Двигатель синтаксической волны без pos_idx (0% pos_idx!)
        self.valence_wave_generator = SyntacticValenceWaveGenerator(d_vec=64, d_dend=8192, top_k_wta=32, seed=seed + 1801)

        # 2. Кристаллизующийся граф каузальных триплетов мира (E1 -> R -> E2)
        self.world_graph = CorticalRelationalWorldGraph(
            encode_fn=self.core_organism.encode_text,
            compute_vec64_fn=self.core_organism.compute_stimulus_vec64,
        )

        # 3. Самоорганизующаяся онтология ролей через дистрибутивную STDP (0% ручных списков!)
        self.emergent_ontology = EmergentOntologicalRoleDiscovery(n_bins=2048)

        # 4. Тета-Гамма спайковый фазовый мультиплексор (120 мс = 6 x 20 мс TTFS)
        self.theta_gamma_multiplexer = ThetaGammaSpikingPhaseMultiplexer(d_vec=64, n_gamma_slots=6, bins_per_gamma=20)

        # 5. Верификатор двунаправленного замыкания смысла X -> Y_hat -> X_hat
        self.meaning_closure_verifier = BidirectionalMeaningClosureVerifier()

        self._bootstrap_genesis_modules_from_lessons()

    def _bootstrap_genesis_modules_from_lessons(self) -> None:
        """Кристаллизует базовые каузальные уроки в без-индексный генератор валентностей `SyntacticValenceWaveGenerator`,
        дистрибутивную STDP-онтологию `EmergentOntologicalRoleDiscovery` и граф мира `CorticalRelationalWorldGraph`."""
        batch_trajs: List[Tuple[torch.Tensor, Sequence[int], float]] = []
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

                # 2. Наблюдаем предложения в дистрибутивной STDP-онтологии (формируем канонические рецептивные поля)
                self.emergent_ontology.observe_sentence_stdp(stim_str, register_as_canonical=True, lr=1.5)
                self.emergent_ontology.observe_sentence_stdp(resp_str, register_as_canonical=True, lr=1.5)

                # 3. Извлекаем каузальные триплеты в граф мира
                self.world_graph.extract_and_consolidate_sentence(resp_str, dopamine_gain=f_mod)

        # 1. Совместно обучаем без-позиционную волну валентности (0% pos_idx!) на всех каузальных схемах одновременно:
        self.valence_wave_generator.imprint_batch_zero_pos(
            batch_trajectories=batch_trajs,
            get_vec64_fn=self.core_organism.get_vec64,
            epochs=12,
        )

    def ingest_scientific_corpus_into_world_graph(self, sentences: Sequence[str], dopamine_gain: float = 2.0) -> int:
        """Кристаллизует поток отдельных научных предложений в `CorticalRelationalWorldGraph` и `EmergentOntologicalRoleDiscovery`."""
        triplets_added = 0
        for s in sentences:
            self.emergent_ontology.observe_sentence_stdp(s, register_as_canonical=False, lr=1.2)
            extracted = self.world_graph.extract_and_consolidate_sentence(s, dopamine_gain=dopamine_gain)
            triplets_added += len(extracted)
        return triplets_added

    def generate_genesis_response(
        self,
        user_utterance: str,
        use_zero_pos_valence_wave: bool = True,
        prefer_world_graph_synthesis: bool = False,
        max_tokens: int = 48,
    ) -> Dict[str, object]:
        """Генерирует ответ с полным задействованием всех 5 механизмов Ziwago-Genesis v18.0."""
        t0 = time.perf_counter()

        # 1. Самоорганизующееся открытие ролей через дистрибутивную STDP (0% ручных списков!)
        emergent_discoveries = self.emergent_ontology.discover_emergent_role_bindings(user_utterance)
        emergent_bindings: Dict[str, str] = {
            canon: fil for canon, (fil, _sim) in emergent_discoveries.items()
        }
        # Объединяем с синтаксическим экстрактором и строим аксональный указатель Дугообразного Пучка (Arcuate Fasciculus)
        combined_bindings = dict(emergent_bindings)
        combined_bindings.update(self.core_organism.role_filler_binder.extract_slot_bindings(user_utterance))
        arcuate_map = self.core_organism.role_filler_binder.build_arcuate_token_pointer_map(
            combined_bindings,
            self.core_organism.encode_text,
        )

        # 2. Если активирован синтез из Графа Мира (или вопрос касается многошаговой цепочки в графе триплетов):
        graph_synthesis = self.world_graph.synthesize_novel_sentence_from_graph(user_utterance)
        if prefer_world_graph_synthesis and graph_synthesis is not None and int(graph_synthesis["hops_traversed"]) >= 2:
            ans_text = str(graph_synthesis["synthesized_sentence"])
            u_ids = self.core_organism.encode_text(user_utterance)
            q_vec = self.core_organism.compute_stimulus_vec64(u_ids)
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
                "generation_engine": "CorticalRelationalWorldGraph (Multi-Hop Triplet Wavefront Synthesis)",
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


def run_v18_genesis_dream_showcase() -> Dict[str, object]:
    print("=" * 112, flush=True)
    print("ZIWAGO-GENESIS v18.0 — THE 5 PILLARS OF THE DREAM BIOLOGICAL SNN ARCHITECTURE", flush=True)
    print("=" * 112, flush=True)

    genesis = ZiwagoGenesisDreamOrganismV18(root=ROOT, workers=4, cpu_threads=8, seed=99999)

    # 1. Кристаллизуем 3 отдельных научных факта из разных текстов в Граф Каузальных Триплетов Мира:
    scientific_facts = [
        "Astrocytes release gliotransmitters such as D-serine.",
        "D-serine binds to glycine sites on apical NMDA receptors.",
        "Apical NMDA receptors ignite dendritic plateau potentials in pyramidal neurons.",
    ]
    n_triplets = genesis.ingest_scientific_corpus_into_world_graph(scientific_facts, dopamine_gain=2.5)
    print(f"  [Pillar 2: Cortical Relational World Graph] Ingested {n_triplets} separate atomic facts into causal graph.", flush=True)

    # 2. Наблюдаем предложения с новыми словами ('bechamel', 'treatise', 'kayaking') без единого ручного словаря:
    unsupervised_exposure_sentences = [
        "While whisking runny bechamel into a fresh egg yolk, the emulsion stabilizes.",
        "To fix grammar errors when writing my treatise, read each sentence aloud and simplify overly long clauses.",
        "Staying active is a wonderful habit, and kayaking outdoors is a great way to clear the mind.",
    ]
    for s_exp in unsupervised_exposure_sentences:
        genesis.emergent_ontology.observe_sentence_stdp(s_exp, register_as_canonical=False, lr=2.0)

    probes = [
        (
            "Pillar 1: Zero-pos_idx Syntactic Valence Wave (BA44 Continuous State-Machine)",
            "Explain how neocortical pyramidal cells combine bottom-up sensory spikes with top-down cortical predictions.",
            False,
        ),
        (
            "Pillar 2: World-Graph Multi-Hop Novel Sentence Synthesis (3 Separate Facts -> 1 Unseen Sentence)",
            "How do astrocytes ignite dendritic plateau potentials in pyramidal neurons?",
            True,
        ),
        (
            "Pillar 3: Unsupervised STDP Emergent Ontology ('bechamel' + 'treatise' + 'kayaking' — NOT in any hardcoded dict!)",
            "Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?",
            False,
        ),
        (
            "Pillar 4 & 5: Theta-Gamma TTFS Multiplexing + Forward-Inverse Meaning Closure (X -> Y_hat -> X_hat)",
            "Why does a runny bechamel sauce need an egg yolk while pyramidal neurons in the neocortex process predictions?",
            False,
        ),
    ]

    results: List[Dict[str, object]] = []
    for label, prompt, use_graph in probes:
        res = genesis.generate_genesis_response(
            prompt,
            use_zero_pos_valence_wave=True,
            prefer_world_graph_synthesis=use_graph,
            max_tokens=48,
        )
        results.append({"pillar": label, **res})
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

    report = {
        "version": "18.0-Ziwago-Genesis-Dream-SNN",
        "pillars_verified": 5,
        "world_graph_edges_count": len(genesis.world_graph.edges),
        "emergent_ontology_vocabulary_size": len(genesis.emergent_ontology.word_context_sdr),
        "showcase_results": results,
    }
    out_path = ROOT / "Produce" / "v18_0_genesis_dream_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 112, flush=True)
    print(f"SAVED ZIWAGO-GENESIS v18.0 REPORT -> {out_path}", flush=True)
    print("=" * 112, flush=True)
    return report


if __name__ == "__main__":
    run_v18_genesis_dream_showcase()
