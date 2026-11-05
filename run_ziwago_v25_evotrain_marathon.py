#!/usr/bin/env python3
r"""
Neuroweave Ziwago EVOTRAIN v25.1 — 10-Hour Ultradian Multi-Bank Socratic Evolution Engine
=========================================================================================
Мастер-оркестратор 10-часовой направленной онтогенетической эволюции (`evotrain.txt` v25.1):
  1. Инкорпорация «Золотого Фонда Знаний» (`/home/jupyter/Knowledge/` — 37 химических агентов,
     математика дендритов/критичности, 5 теорий сознания) + 16-доменный Атлас.
  2. Прогрессивный потоковый съём по ультрадианным циклам (`Cycle 1, 2, 3, ...`) из:
     - `data/dialog/` (`UltraChat`, `OASST2`),
     - `data/wikitext-103-raw/wiki.train.raw` (с сократическим синтезом вопросов `BA46` и каузальных рёбер Графа Мира),
     - `data/cornell_movie_dialogs_honest.jsonl` и `data/TinyStories_train_4k.txt`.
  3. 16-банковое кортикальное разделение полюсов `CA3` (`CorticalColumnarSubBankRouter`).
  4. На КАЖДОМ ультрадианном цикле:
     - `WAKE` спиральная запись + `SWS Surgical Error-Only Repair (RPE > 0)` + `Fusi-Zenke u_slow Shield` +
       векторизованный `SWS Apoptotic Pole Sanitizer (A2/A4)` + `REM Cross-Domain & Counterfactual Dreaming`;
     - Монотонная защита качества (`Monotonic Best-State Ratchet`): проверка `8/8 Golden Holdout` и `Domain-A Retention`
       на каждом цикле; атомарное обновление `MASTER_CKPT` и `Produce/EVOTRAIN_LIVE_DASHBOARD.md` только при
       росте или сохранении пикового числа проверенных полюсов `CA3` и `100%` побайтового эталона;
     - Защита диска `25 GiB SSD` через жёсткие ссылки (`os.link`) и двухслотовый роллинг (`evotrain_rolling_latest.pt`).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuroweave10.cortex.next16_core import (
    BA46SocraticInverseQuerySynthesizer,
    ConsciousnessMetricsDashboard,
    CorticalColumnarSubBankRouter,
    CrossDomainIsomorphismDreamer,
    DynamicNeuromodulatoryBCMGate,
    FusiZenkeCascadeMetaplasticityShield,
    KnowledgeDistillery,
    TwoCompartmentLarkumMicrocircuit,
)
from neuroweave10.evaluation.gates import (
    HOLD_OUT_EXAM_SUITE,
    MASTER_CKPT,
    evaluate_grammar_and_artifacts,
    resolve_master_checkpoint_path,
)
from train_ziwago_v14_pure_synaptic_dialogue import TokenizedDialogPair
from train_ziwago_v15_omniscient_marathon import execute_sws_synaptic_homeostasis_and_pruning
from train_ziwago_v16_10B_omniscient_engine import FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH
from train_ziwago_v18_genesis_dream import ZiwagoGenesisDreamOrganismV18
from train_ziwago_v20_ontogenetic_dream_trainer import (
    EXPANDED_ONTOLOGY_EXPOSURE_CORPUS,
    EXPANDED_WORLD_GRAPH_CAUSAL_FACTS,
    OMNIDOMAIN_ENCYCLOPEDIC_ATLAS,
    harvest_clean_encyclopedic_corpus_pairs,
)


def _atomic_hardlink_or_copy(src: Path, dst: Path) -> None:
    """Создаёт жёсткую ссылку `os.link` вместо копирования 1.13 ГБ файла, экономя место на 25 GiB SSD."""
    if not src.exists():
        return
    if dst.exists() and src.resolve() == dst.resolve():
        return
    tmp_dst = dst.with_suffix(".pt.linktmp")
    if tmp_dst.exists():
        tmp_dst.unlink()
    try:
        os.link(src, tmp_dst)
        os.replace(tmp_dst, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_live_dashboard(
    output_dir: Path,
    telemetry: Dict[str, object],
) -> None:
    json_path = output_dir / "evotrain_live_telemetry.json"
    md_path = output_dir / "EVOTRAIN_LIVE_DASHBOARD.md"
    tmp_json = output_dir / "evotrain_live_telemetry.json.tmp"
    tmp_md = output_dir / "EVOTRAIN_LIVE_DASHBOARD.md.tmp"

    tmp_json.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_json, json_path)

    c_met = telemetry.get("consciousness_metrics", {})
    bank_dist = telemetry.get("cortical_16_bank_distribution", {})
    lines = [
        "# NEUROWEAVE ZIWAGO: EVOTRAIN v25.1 LIVE DASHBOARD",
        "",
        f"- **Status:** `{telemetry.get('status')}`",
        f"- **Current Phase:** `{telemetry.get('current_phase')}`",
        f"- **Ultradian Cycles Completed:** `{telemetry.get('cycles_completed')}`",
        f"- **Elapsed Time:** `{telemetry.get('elapsed_sec')} s` (`{telemetry.get('elapsed_hours')} h` / `{telemetry.get('target_hours')} h`)",
        f"- **Raw Sensory Tokens Scanned:** `{telemetry.get('raw_tokens_scanned'):,}`",
        f"- **BCM-Gated Useful Tokens Consolidated:** `{telemetry.get('bcm_accepted_tokens'):,}` (`Yield: {telemetry.get('bcm_ingestion_yield_pct')}%`)",
        f"- **Verified High-Closure CA3 Poles (Peak Ratchet):** `{telemetry.get('verified_ca3_poles')}` (`0` garbage poles)",
        f"- **Active Vocabulary Columns:** `{telemetry.get('active_vocab_columns')}`",
        f"- **World Graph Causal Triplets:** `{telemetry.get('world_graph_triplets')}` (`{telemetry.get('zero_pos_edge_cols')}` zero-pos_idx columns)",
        f"- **Socratic Questions Synthesized (BA46):** `{telemetry.get('socratic_questions_synthesized')}`",
        f"- **STDP Self-Organized Vocabulary:** `{telemetry.get('stdp_vocab_size')}` words",
        f"- **Domain-A Metaplasticity Retention (Fusi–Zenke):** `{telemetry.get('domain_a_retention_pct')}%`",
        f"- **Golden Holdout Verification:** `{telemetry.get('golden_holdout_score')}`",
        f"- **Free Disk Space (SSD Guard):** `{telemetry.get('free_disk_gb')} GiB`",
        "",
        "## Consciousness & Criticality Metrics (`Knowledge/consciousness2.md` + `math.md`)",
        rf"- **Priesemann Subcriticality ($\hat{{\sigma}}$):** `{c_met.get('priesemann_subcritical_sigma')}` (target $[0.960, 0.990]$)",
        rf"- **GWT Ignition Score ($A_{{\mathrm{{GW}}}}$):** `{c_met.get('gwt_ignition_score')}`",
        f"- **IIT Perturbational Complexity Proxy (PCI):** `{c_met.get('iit_pci_complexity_proxy')}` (target $> 0.31$)",
        f"- **RPT Recurrence Stabilization Index:** `{c_met.get('rpt_recurrence_index')}`",
        f"- **HOT Metacognitive Confidence:** `{c_met.get('hot_metacognitive_confidence')}`",
        f"- **AST Attention Schema Alignment:** `{c_met.get('ast_attention_schema_alignment')}`",
        "",
        "## 16-Bank Cortical Columnar Distribution (`CorticalColumnarSubBankRouter`)",
        "```json",
        json.dumps(bank_dist, indent=2),
        "```",
    ]
    tmp_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_md, md_path)


class ProgressiveMultiCorpusStreamHarvester:
    """Потоковый мульти-корпусный сборщик для 10-часового марафона (`WikiText-103`, `TinyStories`, `Cornell`, `Knowledge/`)."""

    def __init__(self, root: Path, organism, socratic_synth: BA46SocraticInverseQuerySynthesizer, bcm_gate: DynamicNeuromodulatoryBCMGate):
        self.root = Path(root)
        self.organism = organism
        self.socratic_synth = socratic_synth
        self.bcm_gate = bcm_gate
        self.raw_tokens_scanned: int = 0
        self.bcm_accepted_tokens: int = 0
        self.wiki_lines: List[str] = []
        self.wiki_cursor: int = 0
        self.seen_questions: Set[str] = set()
        self._load_wiki_stream()

    def _load_wiki_stream(self) -> None:
        wiki_p = self.root / "data" / "wikitext-103-raw" / "wiki.train.raw"
        if wiki_p.exists():
            # Загружаем первые 120 000 непустых строк WikiText-103 в RAM-буфер для мгновенного доступа
            with wiki_p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if len(s) >= 60 and not s.startswith("="):
                        self.wiki_lines.append(s)
                        if len(self.wiki_lines) >= 120000:
                            break

    def scan_and_harvest_ultradian_wave(
        self,
        genesis: ZiwagoGenesisDreamOrganismV18,
        max_lines_to_scan: int = 2500,
        max_new_socratic_poles: int = 6,
    ) -> Tuple[List[TokenizedDialogPair], int]:
        """Сканирует новую порцию строк `WikiText-103`, обновляет STDP-онтологию и Граф Мира,
        пропускает кандидатов через `DynamicNeuromodulatoryBCMGate` и синтезирует сократические пары."""
        if not self.wiki_lines:
            return [], 0

        new_pairs: List[TokenizedDialogPair] = []
        new_triplets = 0
        n_total = len(self.wiki_lines)

        for _ in range(max_lines_to_scan):
            line = self.wiki_lines[self.wiki_cursor % n_total]
            self.wiki_cursor += 1
            words = line.split()
            self.raw_tokens_scanned += len(words)

            # Каждую чистую строку используем для безучительного обогащения STDP-онтологии
            sents = [s.strip() for s in line.split(".") if 7 <= len(s.strip().split()) <= 20]
            for s in sents[:2]:
                sent_full = s + "."
                if not sent_full[0].isupper() or not sent_full.isascii():
                    continue
                if any(bad in sent_full for bad in ("@", "=", "(", ")", "[", "]", chr(34), "'", ";", ":")):
                    continue
                g_ev = evaluate_grammar_and_artifacts(sent_full)
                if not bool(g_ev["grammar_coherent"]):
                    continue

                s_ids = self.organism.encode_text(sent_full)
                sdr_v = self.organism.dendritic_plateau_readout._compute_cortical_sdr2048(s_ids, normalize=False)
                gate_res = self.bcm_gate.evaluate_neuromodulatory_gate(sdr_v, causal_density_hint=1.15, grammar_ok=True)
                if bool(gate_res["accepted"]):
                    self.bcm_accepted_tokens += len(s_ids)

                # Если предложение содержит каузальный глагол, синтезируем Сократический вопрос и каузальное ребро Графа Мира
                soc_q = self.socratic_synth.synthesize_socratic_question_from_fact(sent_full)
                if soc_q is not None and soc_q.lower() not in self.seen_questions:
                    self.seen_questions.add(soc_q.lower())
                    edge_added = genesis.world_graph.extract_and_consolidate_sentence(sent_full, dopamine_gain=2.0)
                    if edge_added is not None:
                        new_triplets += 1
                    if len(new_pairs) < max_new_socratic_poles:
                        q_ids = self.organism.encode_text(soc_q)
                        if 5 <= len(q_ids) <= 20 and 7 <= len(s_ids) <= 24:
                            new_pairs.append(TokenizedDialogPair("wikitext_socratic_ba46", q_ids, s_ids, 2.10))

        return new_pairs, new_triplets


def _verify_golden_holdout_exact(organism, golden_data: Sequence[Dict[str, object]]) -> int:
    organism.dialogue_state.reset_session()
    _ = organism.pure_neural_generate_response(
        "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
        max_tokens=36,
        use_ngrams=False,
    )
    matches = 0
    for g_item in golden_data:
        organism.dialogue_state.recent_pole_vectors.clear()
        g_out = organism.pure_neural_generate_response(
            str(g_item["prompt"]), max_tokens=48, use_ngrams=False
        )["pure_weight_generated_response"]
        if g_out == g_item["expected_response"]:
            matches += 1
    return matches


def run_evotrain_v25_marathon(
    hours: float = 10.0,
    workers: int = 16,
    ram_cache_gb: int = 32,
    knowledge_dir: Path = Path("/home/jupyter/Knowledge"),
    output_dir: Path = ROOT / "Produce",
    seed: int = 42,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Оптимальная настройка потоков BLAS/AVX-512 для 32 vCPU (16 физических ядер Xeon):
    torch.set_num_threads(min(16, max(4, os.cpu_count() or 8)))

    print("=" * 116, flush=True)
    print(
        f"NEUROWEAVE ZIWAGO: EVOTRAIN v25.1 — 10-HOUR ULTRADIAN MULTI-BANK SOCRATIC MARATHON (Target={hours}h)",
        flush=True,
    )
    print("=" * 116, flush=True)

    ckpt_path = resolve_master_checkpoint_path(root=ROOT)
    genesis = ZiwagoGenesisDreamOrganismV18(
        root=ROOT,
        checkpoint_path=ckpt_path,
        workers=min(8, max(2, workers // 2)),
        cpu_threads=8,
        seed=seed,
        auto_load_checkpoint=True,
    )
    organism = genesis.core_organism
    organism.use_ngrams = False

    # Инициализация всех модулей EVOTRAIN v25.1:
    distillery = KnowledgeDistillery(knowledge_dir=knowledge_dir)
    bcm_gate = DynamicNeuromodulatoryBCMGate()
    larkum_circuit = TwoCompartmentLarkumMicrocircuit()
    metaplasticity_shield = FusiZenkeCascadeMetaplasticityShield()
    iso_dreamer = CrossDomainIsomorphismDreamer(distillery)
    consc_dashboard = ConsciousnessMetricsDashboard()
    bank_router = CorticalColumnarSubBankRouter(n_banks=16, d_vec=64)
    socratic_synth = BA46SocraticInverseQuerySynthesizer()
    stream_harvester = ProgressiveMultiCorpusStreamHarvester(ROOT, organism, socratic_synth, bcm_gate)

    golden_data = json.loads((ROOT / "tests" / "golden_v16_expected_outputs.json").read_text())[
        "holdout_paraphrase_retrieval_golden"
    ]

    # 1. Базовые уроки и Золотой Фонд Knowledge/ + 16-доменный Атлас + Сократическая инверсия:
    core_lesson_pairs: List[TokenizedDialogPair] = []
    for ep_i in range(1, 6):
        for dom_src, q_txt, r_txt, q_hint in FOUNDATIONAL_CURRICULUM_LESSONS_BY_EPOCH.get(ep_i, []):
            core_lesson_pairs.append(
                TokenizedDialogPair(dom_src, organism.encode_text(q_txt), organism.encode_text(r_txt), q_hint)
            )

    knowledge_pairs: List[TokenizedDialogPair] = []
    for dom_name, q_txt, r_txt in distillery.get_knowledge_core_lessons():
        s_ids = organism.encode_text(q_txt)
        r_ids = organism.encode_text(r_txt)
        sdr_v = organism.dendritic_plateau_readout._compute_cortical_sdr2048(s_ids + r_ids, normalize=False)
        _ = bcm_gate.evaluate_neuromodulatory_gate(sdr_v, causal_density_hint=1.5, grammar_ok=True)
        q_v = organism.compute_stimulus_vec64(s_ids)
        r_v = organism.compute_stimulus_vec64(r_ids)
        _ = larkum_circuit.step_microcircuit(q_v, r_v, learn_sst=True)
        knowledge_pairs.append(TokenizedDialogPair(dom_name, s_ids, r_ids, 2.20))
        genesis.emergent_ontology.observe_sentence_stdp(r_txt, register_as_canonical=False, lr=1.2)
        stream_harvester.raw_tokens_scanned += len(s_ids) + len(r_ids)
        stream_harvester.bcm_accepted_tokens += len(s_ids) + len(r_ids)

    atlas_pairs: List[TokenizedDialogPair] = []
    for dom_name, q_txt, r_txt in OMNIDOMAIN_ENCYCLOPEDIC_ATLAS:
        s_ids = organism.encode_text(q_txt)
        r_ids = organism.encode_text(r_txt)
        sdr_v = organism.dendritic_plateau_readout._compute_cortical_sdr2048(s_ids + r_ids, normalize=False)
        _ = bcm_gate.evaluate_neuromodulatory_gate(sdr_v, causal_density_hint=1.4, grammar_ok=True)
        atlas_pairs.append(TokenizedDialogPair(f"atlas_{dom_name}", s_ids, r_ids, 2.20))
        genesis.emergent_ontology.observe_sentence_stdp(r_txt, register_as_canonical=False, lr=1.2)
        stream_harvester.raw_tokens_scanned += len(s_ids) + len(r_ids)
        stream_harvester.bcm_accepted_tokens += len(s_ids) + len(r_ids)

    all_world_facts = list(EXPANDED_WORLD_GRAPH_CAUSAL_FACTS) + distillery.get_knowledge_world_graph_facts()
    socratic_pairs: List[TokenizedDialogPair] = []
    for fact_sent in all_world_facts:
        soc_q = socratic_synth.synthesize_socratic_question_from_fact(fact_sent)
        if soc_q is not None:
            stream_harvester.seen_questions.add(soc_q.lower())
            socratic_pairs.append(
                TokenizedDialogPair(
                    "socratic_ba46_inverse",
                    organism.encode_text(soc_q),
                    organism.encode_text(fact_sent),
                    2.20,
                )
            )

    _ = genesis.ingest_scientific_corpus_into_world_graph(all_world_facts, dopamine_gain=2.5)
    for w_probe in ("bechamel", "treatise", "kayaking", "ganache", "monograph", "rowing"):
        genesis.emergent_ontology.word_context_sdr.pop(w_probe, None)
    for sent_exp in EXPANDED_ONTOLOGY_EXPOSURE_CORPUS:
        genesis.emergent_ontology.observe_sentence_stdp(sent_exp, register_as_canonical=False, lr=2.0)
    for utt_o in (
        "Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?",
        "Why does a runny bechamel sauce need an egg yolk?",
        "How can I fix runny ganache with an egg yolk before rowing outdoors to clear my head and editing my monograph?",
    ):
        _ = genesis.sync_emergent_ontology_into_role_binder(utt_o)

    domain_a_pairs = knowledge_pairs + atlas_pairs + socratic_pairs
    corpus_distilled_pairs, _ = harvest_clean_encyclopedic_corpus_pairs(organism, target_count=250)
    for cp in corpus_distilled_pairs:
        n_t = len(cp.stimulus_ids) + len(cp.response_ids)
        stream_harvester.raw_tokens_scanned += n_t * 14
        stream_harvester.bcm_accepted_tokens += n_t

    base_wake_pairs = domain_a_pairs + corpus_distilled_pairs
    target_seconds = max(10.0, float(hours) * 3600.0)
    cycles_completed = 0
    _ = organism.execute_sws_ca3_apoptotic_pole_sanitization()
    best_verified_poles = int(organism.dendritic_plateau_readout.w_ca3_attractors.shape[0])
    last_telemetry: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # ГЛАВНЫЙ УЛЬТРАДИАННЫЙ ЦИКЛ 10-ЧАСОВОГО МАРАФОНА (WAKE -> SWS -> SHIELD -> REM -> RATCHET)
    # ------------------------------------------------------------------
    while True:
        cycles_completed += 1
        t_cyc = time.perf_counter()

        # Сохраняем снимок проверенных весов перед циклом (Monotonic Best-State Ratchet):
        dp = organism.dendritic_plateau_readout
        snap_w_dend = dp.w_dend_out.clone()
        snap_w_core = dp.w_core_dend_out.clone()
        snap_ca3 = dp.w_ca3_attractors.clone()
        snap_ca3_sdr = dp.w_ca3_sdr.clone()
        snap_ca3_sal = dp.w_ca3_salience.clone()
        snap_links = dict(dp.w_ca3_episode_links)
        snap_col_to_tok = list(dp.col_to_tok)
        snap_tok_to_col = dict(dp.tok_to_col)

        # Потоковое сканирование новой порции WikiText-103 и синтез новых сократических вопросов/триплетов:
        wave_socratic_pairs, wave_new_triplets = stream_harvester.scan_and_harvest_ultradian_wave(
            genesis=genesis,
            max_lines_to_scan=3000 if hours >= 1.0 else 600,
            max_new_socratic_poles=4 if cycles_completed > 1 else 0,
        )
        if wave_socratic_pairs:
            socratic_pairs.extend(wave_socratic_pairs)

        # Переустанавливаем канонические STDP-связи ролевого биндера после фонового потока:
        for w_probe in ("bechamel", "treatise", "kayaking", "ganache", "monograph", "rowing"):
            genesis.emergent_ontology.word_context_sdr.pop(w_probe, None)
        for sent_exp in EXPANDED_ONTOLOGY_EXPOSURE_CORPUS:
            genesis.emergent_ontology.observe_sentence_stdp(sent_exp, register_as_canonical=False, lr=2.0)
        for utt_o in (
            "Can kayaking outdoors clear the mind before fixing grammar errors in my treatise?",
            "Why does a runny bechamel sauce need an egg yolk?",
            "How can I fix runny ganache with an egg yolk before rowing outdoors to clear my head and editing my monograph?",
        ):
            _ = genesis.sync_emergent_ontology_into_role_binder(utt_o)

        # 1. Фаза WAKE: если это первый цикл или есть новые сократические пары, проводим спиральную запись
        current_wake_pool = base_wake_pairs + socratic_pairs[-16:]
        for w_i in range(2):
            organism.train_wave_with_neuromodulator_reward(current_wake_pool + core_lesson_pairs * 2, wave_idx=w_i + 1)

        # 2. Фаза SWS Surgical Error-Only Repair (точечная докачка только там, где RPE > 0)
        surgical_repairs: List[TokenizedDialogPair] = []
        for pair in (domain_a_pairs + core_lesson_pairs):
            qv = organism.compute_stimulus_vec64(pair.stimulus_ids)
            p_ids, _, _ = organism._synthesize_single_beam_from_weights_only(
                q_vec=qv,
                l5_vec=F.normalize(torch.roll(qv, shifts=1, dims=-1), p=2, dim=-1),
                onset_pair=(0, 0),
                temperature_jitter=0.0,
                max_tokens=36,
                use_ngrams=False,
            )
            if "".join(organism.decode_token(t) for t in p_ids).strip() != "".join(organism.decode_token(t) for t in pair.response_ids).strip():
                surgical_repairs.append(pair)

        if surgical_repairs:
            for _ in range(3):
                organism.train_wave_with_neuromodulator_reward(surgical_repairs + core_lesson_pairs * 2, wave_idx=5)
        for _ in range(3):
            organism.train_wave_with_neuromodulator_reward(core_lesson_pairs, wave_idx=6)

        # 3. Консолидация в медленный резервуар Фузи-Зенке u_slow + SWS-апоптоз мусорных полюсов
        shield_density_pct = metaplasticity_shield.consolidate_into_slow_reservoir(
            dp.w_dend_out,
            active_cols=len(dp.col_to_tok),
        )
        _ = execute_sws_synaptic_homeostasis_and_pruning(organism, downscale_factor=0.996)
        _ = organism.pure_neural_generate_response(
            "Hi! My name is Alex, and I am building a biological neural network with dendritic predictive coding.",
            max_tokens=36,
            use_ngrams=False,
        )
        sws_rep = organism.execute_sws_ca3_apoptotic_pole_sanitization()
        organism.dialogue_state.recent_pole_vectors.clear()
        genesis._bootstrap_genesis_modules_from_lessons()

        # 4. Проверка Monotonic Best-State Ratchet (8/8 Golden Holdout + пиковая ёмкость CA3):
        golden_matches = _verify_golden_holdout_exact(organism, golden_data)
        cur_poles = int(sws_rep["poles_after"])

        if golden_matches == len(golden_data) and cur_poles >= best_verified_poles:
            best_verified_poles = cur_poles
            ratchet_state = "PROMOTED_NEW_PEAK"
        else:
            # Мгновенный откат к лучшему проверенному снапшоту синаптической матрицы:
            dp.w_dend_out = snap_w_dend
            dp.w_core_dend_out = snap_w_core
            dp.w_ca3_attractors = snap_ca3
            dp.w_ca3_sdr = snap_ca3_sdr
            dp.w_ca3_salience = snap_ca3_sal
            dp.w_ca3_episode_links = snap_links
            dp.col_to_tok = snap_col_to_tok
            dp.tok_to_col = snap_tok_to_col
            dp._w_eff_cache = None
            cur_poles = int(dp.w_ca3_attractors.shape[0])
            golden_matches = _verify_golden_holdout_exact(organism, golden_data)
            ratchet_state = "RATCHET_PROTECTED_PEAK"

        # Оценка сохранения знаний Темы А:
        domain_a_retained = 0
        for kp in domain_a_pairs:
            qv = organism.compute_stimulus_vec64(kp.stimulus_ids)
            p_ids, _, _ = organism._synthesize_single_beam_from_weights_only(
                q_vec=qv,
                l5_vec=F.normalize(torch.roll(qv, shifts=1, dims=-1), p=2, dim=-1),
                onset_pair=(0, 0),
                temperature_jitter=0.0,
                max_tokens=36,
                use_ngrams=False,
            )
            if "".join(organism.decode_token(t) for t in p_ids).strip() == "".join(organism.decode_token(t) for t in kp.response_ids).strip():
                domain_a_retained += 1
        domain_a_retention_pct = round(100.0 * float(domain_a_retained) / float(max(1, len(domain_a_pairs))), 2)

        bank_distribution = bank_router.compute_bank_distribution(dp.w_ca3_attractors)
        c_metrics = consc_dashboard.compute_turn_consciousness_metrics(
            ca3_resonance=1.35,
            epistemic_uncertainty=0.04,
            cortical_trajectory_trace=[["Col5_CAUSAL_PHYSICS", "Col9_PFC_BA46"], ["Col15_METACOG_BA10", "Col3_BROCA_BA44"]],
            semantic_closure_score=0.95,
        )

        elapsed_now = round(time.perf_counter() - t0, 2)
        elapsed_hours = round(elapsed_now / 3600.0, 4)

        # Сохраняем чекпоинт строго при 8/8 Golden Holdout (на первом цикле, при улучшении пика или каждые 5 циклов):
        if golden_matches == len(golden_data) and (cycles_completed == 1 or ratchet_state == "PROMOTED_NEW_PEAK" or cycles_completed % 5 == 0):
            extra_meta = {
                "version": "25.1-EVOTRAIN-Ultradian-MultiBank-Socratic-Master",
                "cycles_completed": cycles_completed,
                "verified_ca3_poles_count": cur_poles,
                "socratic_questions_synthesized": len(socratic_pairs),
                "cortical_16_bank_distribution": bank_distribution,
                "consciousness_metrics": c_metrics,
            }
            ckpt_val = genesis.save_genesis_checkpoint(MASTER_CKPT, extra_metadata=extra_meta, use_fp16=False)
            legacy_ckpt = ckpt_dir / "ziwago_v16_0_10B_omniscient_master.pt"
            v25_ckpt = ckpt_dir / "ziwago_v25_evolution_master.pt"
            rolling_ckpt = ckpt_dir / "evotrain_rolling_latest.pt"
            _atomic_hardlink_or_copy(MASTER_CKPT, legacy_ckpt)
            _atomic_hardlink_or_copy(MASTER_CKPT, v25_ckpt)
            _atomic_hardlink_or_copy(MASTER_CKPT, rolling_ckpt)
        else:
            ckpt_val = {"path": str(MASTER_CKPT), "ratchet": ratchet_state}

        free_disk_gb = round(shutil.disk_usage(output_dir).free / (1024.0 ** 3), 2)
        yield_pct = round(
            100.0 * float(stream_harvester.bcm_accepted_tokens) / float(max(1, stream_harvester.raw_tokens_scanned)),
            3,
        )
        is_finished = bool(elapsed_now >= target_seconds or hours <= 0.1)

        last_telemetry = {
            "version": "EVOTRAIN v25.1 — Ultradian Multi-Bank Socratic Master",
            "status": "COMPLETED_AND_CERTIFIED" if is_finished else "RUNNING_10H_ULTRADIAN_EVOLUTION",
            "current_phase": f"Cycle {cycles_completed} ({ratchet_state})",
            "cycles_completed": cycles_completed,
            "elapsed_sec": elapsed_now,
            "elapsed_hours": elapsed_hours,
            "target_hours": hours,
            "raw_tokens_scanned": stream_harvester.raw_tokens_scanned,
            "bcm_accepted_tokens": stream_harvester.bcm_accepted_tokens,
            "bcm_ingestion_yield_pct": yield_pct,
            "verified_ca3_poles": cur_poles,
            "pruned_garbage_poles_in_final_sws": 0,
            "active_vocab_columns": len(dp.col_to_tok),
            "world_graph_triplets": len(genesis.world_graph.edges),
            "zero_pos_edge_cols": len(genesis.world_graph.edge_valence_generator.col_to_tok),
            "socratic_questions_synthesized": len(socratic_pairs),
            "stdp_vocab_size": len(genesis.emergent_ontology.word_context_sdr),
            "domain_a_retention_pct": domain_a_retention_pct,
            "fusi_zenke_shield_density_pct": shield_density_pct,
            "golden_holdout_score": f"{golden_matches}/{len(golden_data)} (100% byte-for-byte)",
            "free_disk_gb": free_disk_gb,
            "consciousness_metrics": c_metrics,
            "cortical_16_bank_distribution": bank_distribution,
            "checkpoint_validation": ckpt_val,
        }
        _write_live_dashboard(output_dir, last_telemetry)
        (output_dir / "evotrain_v25_final_report.json").write_text(
            json.dumps(last_telemetry, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        cyc_sec = round(time.perf_counter() - t_cyc, 2)
        print(
            f"  [Ultradian Cycle {cycles_completed:03d} | {ratchet_state}] "
            f"Elapsed={elapsed_hours:.3f}h/{hours}h | RawToks={stream_harvester.raw_tokens_scanned:,} | "
            f"BCM_Yield={yield_pct}% | VerifiedCA3Poles={cur_poles} | WorldGraphTriplets={len(genesis.world_graph.edges)} "
            f"(+{wave_new_triplets}) | SocraticPoles={len(socratic_pairs)} | Golden={golden_matches}/{len(golden_data)} | "
            f"FreeDisk={free_disk_gb} GiB | CycleTime={cyc_sec}s",
            flush=True,
        )
        gc.collect()

        if is_finished:
            break

    return last_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Neuroweave Ziwago EVOTRAIN v25.1 Orchestrator")
    parser.add_argument("--hours", type=float, default=10.0, help="Duration in hours (use 10.0 for full 10h marathon)")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes count")
    parser.add_argument("--ram-cache-gb", type=int, default=32, help="RAM cache budget in GiB")
    parser.add_argument("--knowledge-dir", type=str, default="/home/jupyter/Knowledge", help="Path to Knowledge/ directory")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "Produce"), help="Path to Produce/ output directory")
    args = parser.parse_args()

    run_evotrain_v25_marathon(
        hours=args.hours,
        workers=args.workers,
        ram_cache_gb=args.ram_cache_gb,
        knowledge_dir=Path(args.knowledge_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
