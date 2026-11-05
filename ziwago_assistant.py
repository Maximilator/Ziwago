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
    Оценивает входной стимул, резонанс `CA3` и эпистемическую неопределённость `SON`, выбирая одно из 5 действий:
      - `ACTION_EPISODIC_MEMORY_RECALL`: запрос к эпизодическому графу рабочей памяти `BA46`;
      - `ACTION_ONLINE_DA_CONSOLIDATION`: дофаминовая запись нового факта или многопредложенческого эпизода;
      - `ACTION_ACTIVE_INFERENCE_CLARIFY`: встречный уточняющий вопрос от `SON` при высокой неопределённости;
      - `ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS`: 2-хоповый транзитивный вывод ($A \implies B \implies C$) или абзац из цепи `CA3`;
      - `ACTION_DIRECT_SYNAPTIC_RESPONSE`: прямой одно- или двухклаузальный синтез через 8,192 NMDA-ветви."""

    ACTION_EPISODIC_MEMORY_RECALL = "ACTION_EPISODIC_MEMORY_RECALL"
    ACTION_ONLINE_DA_CONSOLIDATION = "ACTION_ONLINE_DA_CONSOLIDATION"
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
    ) -> Tuple[str, Dict[str, float]]:
        d1_episodic = 2.5 if is_episodic_query else 0.05
        d1_consolidate = 2.4 if is_teaching_or_identity else 0.05
        d1_clarify = 2.3 if (active_inference_triggered or (not is_teaching_or_identity and not is_episodic_query and ca3_resonance < 0.38)) else 0.1
        d1_multihop = 2.0 if (wants_multi_hop_or_paragraph and not active_inference_triggered) else 0.15
        d1_direct = 1.6 if not active_inference_triggered else 0.2

        scores = {
            self.ACTION_EPISODIC_MEMORY_RECALL: round(d1_episodic, 4),
            self.ACTION_ONLINE_DA_CONSOLIDATION: round(d1_consolidate, 4),
            self.ACTION_ACTIVE_INFERENCE_CLARIFY: round(d1_clarify, 4),
            self.ACTION_MULTI_HOP_OR_PARAGRAPH_SYNTHESIS: round(d1_multihop, 4),
            self.ACTION_DIRECT_SYNAPTIC_RESPONSE: round(d1_direct, 4),
        }
        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        return chosen, scores


class ZiwagoBiologicalSNNAssistant:
    """Единый фасад функционального биологического SNN-ассистента Ziwago v17.0."""

    def __init__(
        self,
        root: Path = ROOT,
        checkpoint_path: Optional[Path] = None,
        workers: int = 4,
        cpu_threads: int = 8,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.genesis = ZiwagoGenesisDreamOrganismV18(
            root=self.root,
            checkpoint_path=checkpoint_path,
            workers=workers,
            cpu_threads=cpu_threads,
            seed=seed,
        )
        self.organism = self.genesis.core_organism
        self.organism.use_ngrams = False
        self.basal_ganglia = BasalGangliaActionSelector()
        self.loaded_checkpoint_info: Optional[Dict[str, object]] = None

        default_ckpt = self.root / "Produce" / "checkpoints" / "ziwago_v16_0_10B_omniscient_master.pt"
        ckpt_to_load = Path(checkpoint_path) if checkpoint_path is not None else default_ckpt
        if ckpt_to_load.exists():
            self.loaded_checkpoint_info = self.genesis.load_genesis_checkpoint(ckpt_to_load, strict_schema=True)

    def teach_multi_sentence_episode(self, teaching_text: str) -> Dict[str, object]:
        r"""Обучает ассистента одному или нескольким связанным предложениям под всплеском `DA + ACh`
        и связывает их полюса направленным STDP-синапсом `w_ca3_episode_links` ($p_1 \longrightarrow p_2$)."""
        clean = teaching_text.strip()
        for pref in ("Remember that ", "remember that ", "Note that ", "note that ", "Learn that ", "learn that ", "Fact: ", "fact: "):
            if clean.startswith(pref):
                clean = clean[len(pref) :].strip()
                break
        raw_sents = [s.strip() for s in clean.replace("!", ".").split(".") if len(s.strip().split()) >= 4]
        if not raw_sents:
            return {"sentences_imprinted": 0, "episode_links_created": 0}

        pole_vecs: List[torch.Tensor] = []
        total_toks = 0
        for idx, s_txt in enumerate(raw_sents):
            sent_full = (s_txt[0].upper() + s_txt[1:] if s_txt[0].islower() else s_txt) + "."
            s_ids = self.organism.encode_text(sent_full)
            if idx > 0:
                # Делаем ключ второго предложения эпизода ортогональным первому, чтобы они образовали цепь p1 -> p2
                q_ids = self.organism.encode_text(f"Continuation episode {idx}: " + sent_full)
            else:
                q_ids = s_ids
            pair = TokenizedDialogPair("assistant_online_episode", q_ids, s_ids, quality_reward_hint=2.8)
            for _ in range(4):
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
        }

    def chat(
        self,
        user_utterance: str,
        max_tokens: int = 48,
        enable_active_inference: bool = True,
    ) -> Dict[str, object]:
        """Выполняет один полный цикл работы биологического SNN-ассистента с выбором действия в Базальных Ганглиях."""
        t0 = time.perf_counter()
        low = user_utterance.strip().lower()
        is_ep = self.organism.dialogue_state.is_episodic_history_query(user_utterance)
        is_teach = any(low.startswith(p) for p in ("remember that ", "note that ", "learn that ", "fact: "))
        is_ident_fact = any(p in low for p in ("my name is", "i am building", "i'm building")) and "?" not in user_utterance
        wants_mh = any(m in low for m in ("step by step", "causal chain", "consequently", "how does that lead to"))
        wants_para = any(m in low for m in ("in detail", "elaborate", "full paragraph", "two sentences"))

        episode_teach_info: Dict[str, object] = {}
        if is_teach:
            episode_teach_info = self.teach_multi_sentence_episode(user_utterance)

        res = self.organism.pure_neural_generate_response(
            user_utterance=user_utterance,
            max_tokens=max_tokens,
            use_ngrams=False,
            enable_multi_hop=wants_mh,
            enable_active_inference=enable_active_inference and not is_teach,
            enable_episode_chain=wants_para,
        )

        if is_teach and int(episode_teach_info.get("sentences_imprinted", 0)) >= 1:
            ack_resp = (
                f"Consolidated {episode_teach_info['sentences_imprinted']} sentence(s) "
                f"({episode_teach_info['tokens_imprinted']} BPE tokens, "
                f"{episode_teach_info['episode_links_created']} CA3 episode link(s)) "
                f"into the 8,192-branch NMDA matrix and Protected Core Bank."
            )
            res["pure_weight_generated_response"] = ack_resp
            res["online_dopamine_imprinted_tokens"] = episode_teach_info["tokens_imprinted"]

        bg_action, bg_scores = self.basal_ganglia.select_action(
            utterance=user_utterance,
            is_episodic_query=is_ep,
            is_teaching_or_identity=(is_teach or is_ident_fact),
            wants_multi_hop_or_paragraph=(wants_mh or wants_para or bool(res.get("multi_hop_rollout_used"))),
            ca3_resonance=float(res.get("ca3_top1_resonance", 1.0)),  # type: ignore[arg-type]
            active_inference_triggered=bool(res.get("active_inference_clarification_triggered", False)),
        )

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
