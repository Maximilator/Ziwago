"""Экспериментальный Стресс-Стенд Ziwago-Prime v6 (по программе Produce/next7.txt).

Запускает 3 глубоких стресс-теста живой архитектуры Neuroweave Ziwago v10.6:
  1. Стресс-Тест 1: «Дилемма Стабильности–Пластичности» (Catastrophic Forgetting vs Interleaved Replay):
     Последовательное обучение 3 разным историям с циклами сна между ними
     (Сюжет A → Сон #1 → Сюжет B → Сон #2 → Сюжет C → Сон #3).
     Проверяет сохранность старых семантических мостов (A), формирование кросс-сюжетных
     химерных мостов (A↔B, B↔C, A↔C) и отсутствие катастрофического забывания при генерации.
  2. Стресс-Тест 2: «Когнитивный Диссонанс» (Битва Амигдалы A- и Гипоталамуса D_novelty в VTE):
     Столкновение инстинкта самосохранения (Амигдала A-, кондиционированная угроза)
     и исследовательского любопытства от скуки (Гипоталамус D_novelty -> κ_explore)
     на стимуле, который одновременно НОВЫЙ и ОПАСНЫЙ ('cave'), против скучной рутины ('house')
     и безопасной новизны ('castle').
  3. Стресс-Тест 3: «Структурный Морфогенез под нагрузкой» (Синтаксическая Инверсия SVO → OVS):
     Подача радикально нового, синтаксически инвертированного языка ('castle to rode knight the .')
     после консолидации канонического языка ('the knight rode to the castle .').
     Логирует скачок ошибки F(t), выброс нейротрофина [BDNF], активное проращивание новых
     дендритных шипиков (sprouted_synapses), усиленный нейрогенез DG и защиту старых весов.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = str(ROOT)
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class _AllowV8V9Loader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in ("neuroweave8", "neuroweave9"):
            parts = fullname.split(".")
            cand_pkg = os.path.join(BASE_DIR, *parts, "__init__.py")
            cand_mod = os.path.join(BASE_DIR, *parts) + ".py"
            if os.path.isfile(cand_pkg):
                return importlib.util.spec_from_file_location(
                    fullname, cand_pkg, submodule_search_locations=[os.path.dirname(cand_pkg)]
                )
            if os.path.isfile(cand_mod):
                return importlib.util.spec_from_file_location(fullname, cand_mod)
        return None


sys.meta_path.insert(0, _AllowV8V9Loader())

from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig


def build_vocab(words: List[str]):
    base = ["<pad>", "<unk>", ".", ",", "the"]
    for w in words:
        if w not in base:
            base.append(w)
    itos = list(base)
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def word_concept_idx(org: RegenesisOrganism, tok_id: int) -> int:
    x = org._encode_frame(torch.tensor([tok_id]))
    sens_pad = torch.zeros(1, org.gcfg.hippo.n_ec)
    x_ec_pad = torch.nn.functional.pad(
        x, (0, max(0, org.gcfg.hippo.n_ec - x.shape[1]))
    )[:, :org.gcfg.hippo.n_ec]
    ec_in = 0.55 * x_ec_pad + 0.45 * sens_pad
    ec_grid = org.org.hipp.encode_ec(ec_in, ach=0.8)
    return org.org.hipp.concept_idx_for_ec(ec_grid)


# ======================================================================
# 1. СТРЕСС-ТЕСТ 1: ДИЛЕММА СТАБИЛЬНОСТИ–ПЛАСТИЧНОСТИ (A -> Сон -> B -> Сон -> C -> Сон)
# ======================================================================
def run_stress_1_stability_plasticity() -> Dict:
    words = [
        "dragon", "flew", "to", "mountain",
        "knight", "rode", "castle",
        "wizard", "walked", "forest",
    ]
    itos, stoi = build_vocab(words)
    gcfg = genesis_cfg("tiny")
    org = RegenesisOrganism(
        UnifiedConfig(vocab_size=len(itos)),
        gcfg=gcfg,
        seed=101,
    )
    org.punct_ids = {stoi["."], stoi[","]}

    story_a = ["the", "dragon", "flew", "to", "the", "mountain", "."]
    story_b = ["the", "knight", "rode", "to", "the", "castle", "."]
    story_c = ["the", "wizard", "walked", "to", "the", "forest", "."]

    def train_story(story: List[str], epochs: int = 8) -> None:
        ids = [stoi[w] for w in story]
        for _ in range(epochs):
            org.motor.prev_winner.zero_()
            org.motor.prev2_winner.zero_()
            org.motor.rate_win.zero_()
            org.org.hipp.narrative_concept_trace.zero_()
            for i in range(len(ids) - 1):
                org.step_token(
                    torch.tensor([ids[i]]),
                    torch.tensor([ids[i + 1]]),
                    learn=True,
                )

    def probe_story_completion(prompt_words: List[str], cand_words: List[str]) -> Dict:
        org.motor.prev_winner.zero_()
        org.motor.prev2_winner.zero_()
        org.motor.rate_win.zero_()
        org.org.hipp.narrative_concept_trace.zero_()
        for w in prompt_words:
            org.step_token(torch.tensor([stoi[w]]), learn=False)
        cand_ids = [stoi[w] for w in cand_words]
        vte = org.simulate_vte(cand_ids)
        win_w = itos[vte["winner"]]
        scores = {itos[k]: v for k, v in vte["candidates"].items()}
        return {"prompt": " ".join(prompt_words), "winner": win_w, "g_scores": scores}

    # --- Этап 1: Сюжет A + Сон #1 ---
    train_story(story_a, epochs=8)
    sleep_1 = org.sleep(ticks=45)
    c_dragon = word_concept_idx(org, stoi["dragon"])
    c_mountain = word_concept_idx(org, stoi["mountain"])
    bridge_a_stage1 = float(
        max(
            org.org.hipp.w_semantic_graph[c_mountain, c_dragon].item(),
            org.org.hipp.w_semantic_graph[c_dragon, c_mountain].item(),
        )
    )
    total_bridges_1 = int((org.org.hipp.w_semantic_graph > 0.05).sum().item())

    # --- Этап 2: Сюжет B + Сон #2 (Interleaved Replay A + B) ---
    train_story(story_b, epochs=8)
    sleep_2 = org.sleep(ticks=45)
    c_knight = word_concept_idx(org, stoi["knight"])
    c_castle = word_concept_idx(org, stoi["castle"])
    bridge_a_stage2 = float(
        max(
            org.org.hipp.w_semantic_graph[c_mountain, c_dragon].item(),
            org.org.hipp.w_semantic_graph[c_dragon, c_mountain].item(),
        )
    )
    bridge_b_stage2 = float(
        max(
            org.org.hipp.w_semantic_graph[c_castle, c_knight].item(),
            org.org.hipp.w_semantic_graph[c_knight, c_castle].item(),
        )
    )
    chimera_ab_stage2 = float(
        max(
            org.org.hipp.w_semantic_graph[c_castle, c_dragon].item(),
            org.org.hipp.w_semantic_graph[c_dragon, c_castle].item(),
        )
    )
    total_bridges_2 = int((org.org.hipp.w_semantic_graph > 0.05).sum().item())

    # --- Этап 3: Сюжет C + Сон #3 (Interleaved Replay A + B + C) ---
    train_story(story_c, epochs=8)
    sleep_3 = org.sleep(ticks=45)
    c_wizard = word_concept_idx(org, stoi["wizard"])
    c_forest = word_concept_idx(org, stoi["forest"])
    bridge_a_stage3 = float(
        max(
            org.org.hipp.w_semantic_graph[c_mountain, c_dragon].item(),
            org.org.hipp.w_semantic_graph[c_dragon, c_mountain].item(),
        )
    )
    bridge_b_stage3 = float(
        max(
            org.org.hipp.w_semantic_graph[c_castle, c_knight].item(),
            org.org.hipp.w_semantic_graph[c_knight, c_castle].item(),
        )
    )
    bridge_c_stage3 = float(
        max(
            org.org.hipp.w_semantic_graph[c_forest, c_wizard].item(),
            org.org.hipp.w_semantic_graph[c_wizard, c_forest].item(),
        )
    )
    chimera_ac_stage3 = float(
        max(
            org.org.hipp.w_semantic_graph[c_forest, c_dragon].item(),
            org.org.hipp.w_semantic_graph[c_dragon, c_forest].item(),
        )
    )
    total_bridges_3 = int((org.org.hipp.w_semantic_graph > 0.05).sum().item())

    # Проверяем удержание всех 3 сюжетов после всех 3 фаз обучения и сна!
    recall_a = probe_story_completion(
        ["the", "dragon", "flew", "to", "the"], ["mountain", "castle", "forest"]
    )
    recall_b = probe_story_completion(
        ["the", "knight", "rode", "to", "the"], ["mountain", "castle", "forest"]
    )
    recall_c = probe_story_completion(
        ["the", "wizard", "walked", "to", "the"], ["mountain", "castle", "forest"]
    )

    retention_a_pct = round(100.0 * bridge_a_stage3 / max(1e-6, bridge_a_stage1), 2)
    retention_b_pct = round(100.0 * bridge_b_stage3 / max(1e-6, bridge_b_stage2), 2)

    return {
        "stage_1_after_story_A": {
            "bridge_dragon_mountain": round(bridge_a_stage1, 4),
            "total_semantic_bridges": total_bridges_1,
            "interleaved_pairs": sleep_1.get("interleaved_pairs", 0),
        },
        "stage_2_after_story_B": {
            "bridge_dragon_mountain": round(bridge_a_stage2, 4),
            "bridge_knight_castle": round(bridge_b_stage2, 4),
            "chimera_dragon_castle": round(chimera_ab_stage2, 4),
            "total_semantic_bridges": total_bridges_2,
            "interleaved_pairs": sleep_2.get("interleaved_pairs", 0),
        },
        "stage_3_after_story_C": {
            "bridge_dragon_mountain": round(bridge_a_stage3, 4),
            "bridge_knight_castle": round(bridge_b_stage3, 4),
            "bridge_wizard_forest": round(bridge_c_stage3, 4),
            "chimera_dragon_forest": round(chimera_ac_stage3, 4),
            "total_semantic_bridges": total_bridges_3,
            "interleaved_pairs": sleep_3.get("interleaved_pairs", 0),
            "swr_interleaved_replays": org.org.hipp.interleaved_replays,
        },
        "bridge_retention": {
            "story_A_retention_pct": retention_a_pct,
            "story_B_retention_pct": retention_b_pct,
            "zero_catastrophic_forgetting": bool(bridge_a_stage3 >= 0.95 * bridge_a_stage1 and bridge_b_stage3 >= 0.95 * bridge_b_stage2),
        },
        "multi_story_recall_after_all_3": {
            "story_A_probe": recall_a,
            "story_B_probe": recall_b,
            "story_C_probe": recall_c,
        },
    }


# ======================================================================
# 2. СТРЕСС-ТЕСТ 2: КОГНИТИВНЫЙ ДИССОНАНС (Амигдала A- vs Гипоталамус D_novelty)
# ======================================================================
def run_stress_2_cognitive_dissonance() -> Dict:
    words = ["knight", "went", "to", "house", "cave", "castle"]
    itos, stoi = build_vocab(words)
    gcfg = genesis_cfg("tiny")
    org = RegenesisOrganism(
        UnifiedConfig(vocab_size=len(itos)),
        gcfg=gcfg,
        seed=202,
    )
    org.punct_ids = {stoi["."], stoi[","]}

    safe_routine = [stoi[w] for w in ["the", "knight", "went", "to", "the", "house", "."]]
    threat_novel = [stoi[w] for w in ["the", "knight", "went", "to", "the", "cave", "."]]
    safe_novel = [stoi[w] for w in ["the", "knight", "went", "to", "the", "castle", "."]]

    # 1) Обучаем грамматической структуре всех 3 путей, но с разной эмоциональной меткой:
    #    - 'house': привычная безопасная рутина (+1.0 награда, много повторов)
    #    - 'cave': редкий/новый стимул, но с однократным сильным аверсивным шоком (-1.6, Flash-LTP в BLA A-)
    #    - 'castle': редкий безопасный стимул (+0.25)
    for _ in range(8):
        for i in range(len(safe_routine) - 1):
            t_in, t_nxt = safe_routine[i], safe_routine[i + 1]
            org._prev_reward = 1.0 if t_nxt == stoi["house"] else 0.1
            org.step_token(torch.tensor([t_in]), torch.tensor([t_nxt]), learn=True)

    # Однократный травматический опыт на 'cave' (One-Shot Flash-LTP в Амигдале A-)
    for _ in range(2):
        for i in range(len(threat_novel) - 1):
            t_in, t_nxt = threat_novel[i], threat_novel[i + 1]
            org._prev_reward = -1.6 if t_nxt == stoi["cave"] else 0.0
            org.step_token(torch.tensor([t_in]), torch.tensor([t_nxt]), learn=True)
        x_cave = org._encode_frame(torch.tensor([stoi["cave"]]))
        org.org.limbic.step(
            x_cave, org.org._prev_pfc,
            free_energy=2.2, intrinsic_rpe=-0.8,
            atp_mean=0.60, ext_reward=-1.8, learn=True,
        )

    # Однократное знакомство с 'castle' (безопасная редкая альтернатива)
    for i in range(len(safe_novel) - 1):
        t_in, t_nxt = safe_novel[i], safe_novel[i + 1]
        org._prev_reward = 0.25 if t_nxt == stoi["castle"] else 0.05
        org.step_token(torch.tensor([t_in]), torch.tensor([t_nxt]), learn=True)

    # 2) Погружаем организм в глубокую монотонность/скуку (повторяем 'house' и 'the' без сюрприза),
    #    чтобы Гипоталамус разогнал D_novelty (голод по новизне) и κ_explore!
    org.motor.rate_win.zero_()
    org.motor.rate_win[stoi["house"]] = 0.32  # 'house' сильно приелся
    org.motor.rate_win[stoi["cave"]] = 0.0    # 'cave' абсолютно свежий (максимальная новизна!)
    org.motor.rate_win[stoi["castle"]] = 0.02 # 'castle' свежий и безопасный

    x_mono = org._encode_frame(torch.tensor([stoi["the"]]))
    for _ in range(12):
        lim_bored = org.org.limbic.step(
            x_mono, org.org._prev_pfc,
            free_energy=0.05, intrinsic_rpe=0.0,
            atp_mean=0.95, ext_reward=0.0, learn=False,
        )
    boredom_d_novelty = round(lim_bored.d_novelty, 4)
    kappa_before_threat = round(
        org.motor.set_active_inference(
            epistemic=0.45, boredom=lim_bored.d_novelty, aversive=lim_bored.aversive
        ),
        4,
    )

    # Прогоняем контекст 'the knight went to the' до точки развилки:
    org.motor.prev_winner.zero_()
    org.motor.prev2_winner.zero_()
    for w in ["the", "knight", "went", "to", "the"]:
        org.motor.set_prev_token(stoi[w])
    # Удерживаем высокий уровень скуки D_novelty в Гипоталамусе перед развилкой
    org.org.limbic.last_out = lim_bored

    # --- Битва 2A: Бинарная дуэль «Смертельная Новизна ('cave') vs Надоевшая Рутина ('house')» ---
    vte_duel = org.simulate_vte([stoi["house"], stoi["cave"]])
    duel_winner = itos[vte_duel["winner"]]
    duel_breakdown = {itos[k]: v for k, v in vte_duel["breakdown"].items()}

    # Проверяем реакцию Амигдалы (Low Road CS+) при предъявлении опасного стимула 'cave':
    x_cave_probe = org._encode_frame(torch.tensor([stoi["cave"]]))
    lim_threat = org.org.limbic.step(
        x_cave_probe, org.org._prev_pfc,
        free_energy=0.65, intrinsic_rpe=0.0,
        atp_mean=0.90, ext_reward=0.0, learn=False,
    )
    kappa_under_threat = round(
        org.motor.set_active_inference(
            epistemic=0.45, boredom=lim_bored.d_novelty, aversive=lim_threat.aversive
        ),
        4,
    )

    # --- Битва 2B: Триадическое разрешение диссонанса ('house' vs 'cave' vs 'castle') ---
    org.org.limbic.last_out = lim_bored
    vte_triad = org.simulate_vte([stoi["house"], stoi["cave"], stoi["castle"]])
    triad_winner = itos[vte_triad["winner"]]
    triad_breakdown = {itos[k]: v for k, v in vte_triad["breakdown"].items()}

    return {
        "hypothalamus_boredom_d_novelty": boredom_d_novelty,
        "kappa_explore_under_boredom": kappa_before_threat,
        "amygdala_low_road_on_cave": {
            "aversive_A_minus": round(lim_threat.aversive, 4),
            "valence": round(lim_threat.valence, 4),
            "kappa_explore_suppressed_to": kappa_under_threat,
        },
        "binary_conflict_cave_vs_house": {
            "winner": duel_winner,
            "dominant_instinct": "AMYGDALA_SURVIVAL_VETO" if duel_winner == "house" else "HYPOTHALAMUS_CURIOSITY",
            "breakdown": duel_breakdown,
        },
        "triadic_resolution_house_cave_castle": {
            "winner": triad_winner,
            "resolution_strategy": "SAFE_NOVELTY_EXPLORATION" if triad_winner == "castle" else triad_winner,
            "breakdown": triad_breakdown,
        },
    }


# ======================================================================
# 3. СТРЕСС-ТЕСТ 3: СТРУКТУРНЫЙ МОРФОГЕНЕЗ ПОД НАГРУЗКОЙ (Инверсия синтаксиса SVO -> OVS)
# ======================================================================
def run_stress_3_structural_morphogenesis() -> Dict:
    words = ["knight", "rode", "to", "castle", "dragon", "flew", "mountain"]
    itos, stoi = build_vocab(words)
    gcfg = genesis_cfg("tiny")
    org = RegenesisOrganism(
        UnifiedConfig(vocab_size=len(itos)),
        gcfg=gcfg,
        seed=303,
    )
    org.punct_ids = {stoi["."], stoi[","]}

    # Канонический язык (SVO): "the knight rode to the castle ."
    svo_seq = [stoi[w] for w in ["the", "knight", "rode", "to", "the", "castle", "."]]
    # Инвертированный язык (OVS / постпозиционный): "castle to rode knight ." и "mountain to flew dragon ."
    ovs_seq_1 = [stoi[w] for w in ["castle", "to", "rode", "knight", "."]]
    ovs_seq_2 = [stoi[w] for w in ["mountain", "to", "flew", "dragon", "."]]

    def count_active_masks() -> Dict[str, int]:
        res = {}
        for col in org.org.hier.cols:
            res[col.name] = int(
                col.w_ff.mask.sum().item()
                + col.w_fb.mask.sum().item()
                + col.w_rec.mask.sum().item()
            )
        res["total"] = sum(res.values())
        return res

    masks_init = count_active_masks()
    bdnf_init = round(org.org.xchem.l.bdnf, 4)

    # --- Фаза 1: Обучение каноническому языку SVO + Сон #1 ---
    fe_svo = []
    for _ in range(7):
        org.motor.prev_winner.zero_()
        org.motor.prev2_winner.zero_()
        for i in range(len(svo_seq) - 1):
            r = org.step_token(
                torch.tensor([svo_seq[i]]),
                torch.tensor([svo_seq[i + 1]]),
                learn=True,
            )
            fe_svo.append(float(r.get("free_energy", 0.0)))
    sleep_svo = org.sleep(ticks=40)
    masks_after_svo = count_active_masks()

    # Сохраняем эталонные веса канонических переходов SVO для проверки отсутствия стирания
    svo_pairs = [
        (stoi["the"], stoi["knight"]),
        (stoi["knight"], stoi["rode"]),
        (stoi["rode"], stoi["to"]),
        (stoi["to"], stoi["the"]),
        (stoi["the"], stoi["castle"]),
    ]
    svo_weights_before_ovs = [
        float((0.5 * org.motor.w_lat[nxt, prv] + 0.5 * org.motor.w_lat_slow[nxt, prv]).item())
        for prv, nxt in svo_pairs
    ]
    cortical_shielded_before = [
        col.w_ff.w_slow.detach().clone() * (col.w_ff.spine_health > 0.45).float()
        for col in org.org.hier.cols
    ]

    # --- Фаза 2: Когнитивный шок — Синтаксически Инвертированный Язык (OVS) под высокой нагрузкой ---
    fe_ovs = []
    bdnf_trace = []
    for ep in range(8):
        for seq in (ovs_seq_1, ovs_seq_2):
            org.motor.prev_winner.zero_()
            org.motor.prev2_winner.zero_()
            for i in range(len(seq) - 1):
                r = org.step_token(
                    torch.tensor([seq[i]]),
                    torch.tensor([seq[i + 1]]),
                    learn=True,
                )
                fe_val = float(r.get("free_energy", 0.0))
                fe_ovs.append(fe_val)
                # При высокой синтаксической неожиданности инвертированного строя
                # усиливаем нейромедиаторный сигнал новизны/стресса на каскад BDNF
                org.org.xchem.update(
                    spikes=0.35,
                    stress=0.05,
                    arousal=0.95,
                    social=0.6,
                    threat=0.05,
                    damage=max(0.45, fe_val * 2.5),
                    dt_ms=50.0,
                )
                bdnf_trace.append(round(org.org.xchem.l.bdnf, 4))
    # Повышаем мета-новизну перед сном пропорционально синтаксическому сюрпризу OVS
    if org.org._last_meta is not None:
        org.org._last_meta.novelty = max(float(org.org._last_meta.novelty), 0.85)
    sleep_ovs = org.sleep(ticks=45)
    masks_after_ovs = count_active_masks()

    # Проверяем сохранность старых канонических переходов SVO и защищённых кортикальных весов!
    svo_weights_after_ovs = [
        float((0.5 * org.motor.w_lat[nxt, prv] + 0.5 * org.motor.w_lat_slow[nxt, prv]).item())
        for prv, nxt in svo_pairs
    ]
    svo_retention_ratio = sum(max(0.0, a) for a in svo_weights_after_ovs) / max(
        1e-6, sum(max(0.0, b) for b in svo_weights_before_ovs)
    )

    cortical_cos_sims = []
    for idx, col in enumerate(org.org.hier.cols):
        old_w = cortical_shielded_before[idx].flatten()
        new_w = (col.w_ff.w_slow.detach() * (col.w_ff.spine_health > 0.45).float()).flatten()
        if float(old_w.norm()) > 1e-6 and float(new_w.norm()) > 1e-6:
            sim = float(
                torch.nn.functional.cosine_similarity(
                    old_w.unsqueeze(0), new_w.unsqueeze(0)
                ).item()
            )
            cortical_cos_sims.append(sim)
    cortical_shield_similarity = sum(cortical_cos_sims) / max(1, len(cortical_cos_sims))

    # Проверяем способность организма генерировать ОБА синтаксических строя (SVO и OVS):
    org.motor.prev_winner.zero_()
    org.motor.prev2_winner.zero_()
    org.motor.rate_win.zero_()
    svo_gen_res = org.generate(n_tokens=4, seed_tokens=[stoi["the"], stoi["knight"], stoi["rode"]], learn=False)
    svo_gen_text = " ".join(itos[i] for i in svo_gen_res["ids"])

    org.motor.prev_winner.zero_()
    org.motor.prev2_winner.zero_()
    org.motor.rate_win.zero_()
    ovs_gen_res = org.generate(n_tokens=3, seed_tokens=[stoi["castle"], stoi["to"]], learn=False)
    ovs_gen_text = " ".join(itos[i] for i in ovs_gen_res["ids"])

    zone_sprouted = {
        col.name: int(col.w_ff.total_sprouted + col.w_fb.total_sprouted + col.w_rec.total_sprouted)
        for col in org.org.hier.cols
    }

    return {
        "bdnf_dynamics": {
            "initial": bdnf_init,
            "after_canonical_svo": sleep_svo["bdnf"],
            "peak_during_inverted_ovs": max(bdnf_trace) if bdnf_trace else sleep_ovs["bdnf"],
            "after_inverted_ovs_sleep": sleep_ovs["bdnf"],
        },
        "synaptogenesis_sprouting": {
            "active_edges_initial": masks_init,
            "active_edges_after_svo": masks_after_svo,
            "active_edges_after_ovs": masks_after_ovs,
            "net_new_synapses_added": masks_after_ovs["total"] - masks_init["total"],
            "total_sprouted_by_zone": zone_sprouted,
            "total_sprouted_all_zones": sleep_ovs["sprouted_synapses"],
            "total_pruned_all_zones": sleep_ovs["pruned_synapses"],
        },
        "dg_neurogenesis": {
            "born_after_canonical_svo": sleep_svo["dg_neurogenesis"],
            "born_after_inverted_ovs": sleep_ovs["dg_neurogenesis"],
        },
        "old_grammar_preservation": {
            "svo_synfire_retention_ratio": round(svo_retention_ratio, 4),
            "cortical_shield_cosine_sim": round(cortical_shield_similarity, 4),
            "canonical_svo_generation": svo_gen_text,
            "inverted_ovs_generation": ovs_gen_text,
        },
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("ЗАПУСК СТРЕСС-ТЕСТОВ ZIWAGO-PRIME v6 (ПО ПРОГРАММЕ Produce/next7.txt)")
    print("=" * 78)

    s1 = run_stress_1_stability_plasticity()
    print("[СТРЕСС-ТЕСТ 1: ДИЛЕММА СТАБИЛЬНОСТИ–ПЛАСТИЧНОСТИ (A -> Сон -> B -> Сон -> C -> Сон)]")
    print(f"  Семантический мост Сюжета A (dragon<->mountain): Этап 1={s1['stage_1_after_story_A']['bridge_dragon_mountain']} -> Этап 2={s1['stage_2_after_story_B']['bridge_dragon_mountain']} -> Этап 3={s1['stage_3_after_story_C']['bridge_dragon_mountain']} (Сохранность: {s1['bridge_retention']['story_A_retention_pct']}%)")
    print(f"  Семантический мост Сюжета B (knight<->castle):   Этап 2={s1['stage_2_after_story_B']['bridge_knight_castle']} -> Этап 3={s1['stage_3_after_story_C']['bridge_knight_castle']} (Сохранность: {s1['bridge_retention']['story_B_retention_pct']}%)")
    print(f"  Семантический мост Сюжета C (wizard<->forest):   Этап 3={s1['stage_3_after_story_C']['bridge_wizard_forest']}")
    print(f"  Всего семантических мостов: Этап 1={s1['stage_1_after_story_A']['total_semantic_bridges']} -> Этап 2={s1['stage_2_after_story_B']['total_semantic_bridges']} -> Этап 3={s1['stage_3_after_story_C']['total_semantic_bridges']}")
    print(f"  Вспоминание Сюжета A ('the dragon flew to the ...'): winner='{s1['multi_story_recall_after_all_3']['story_A_probe']['winner']}', G={s1['multi_story_recall_after_all_3']['story_A_probe']['g_scores']}")
    print(f"  Вспоминание Сюжета B ('the knight rode to the ...'): winner='{s1['multi_story_recall_after_all_3']['story_B_probe']['winner']}', G={s1['multi_story_recall_after_all_3']['story_B_probe']['g_scores']}")
    print(f"  Вспоминание Сюжета C ('the wizard walked to the ...'): winner='{s1['multi_story_recall_after_all_3']['story_C_probe']['winner']}', G={s1['multi_story_recall_after_all_3']['story_C_probe']['g_scores']}")

    s2 = run_stress_2_cognitive_dissonance()
    print("[СТРЕСС-ТЕСТ 2: КОГНИТИВНЫЙ ДИССОНАНС (Амигдала A- vs Гипоталамус D_novelty)]")
    print(f"  Гипоталамус D_novelty (скука от монотонности): {s2['hypothalamus_boredom_d_novelty']} -> κ_explore (до угрозы): {s2['kappa_explore_under_boredom']}")
    print(f"  Реакция Амигдалы (Low Road CS+) на опасную новизну 'cave': A-={s2['amygdala_low_road_on_cave']['aversive_A_minus']}, Валентность={s2['amygdala_low_road_on_cave']['valence']} -> κ_explore подавлен до {s2['amygdala_low_road_on_cave']['kappa_explore_suppressed_to']}")
    print(f"  Битва 2A ('cave' [Опасная Новизна] vs 'house' [Безопасная Рутина]): Победитель='{s2['binary_conflict_cave_vs_house']['winner']}' ({s2['binary_conflict_cave_vs_house']['dominant_instinct']})")
    print(f"    Детализация G(a): {s2['binary_conflict_cave_vs_house']['breakdown']}")
    print(f"  Битва 2B ('house' vs 'cave' vs 'castle' [Безопасная Новизна]): Победитель='{s2['triadic_resolution_house_cave_castle']['winner']}' ({s2['triadic_resolution_house_cave_castle']['resolution_strategy']})")
    print(f"    Детализация G(a): {s2['triadic_resolution_house_cave_castle']['breakdown']}")

    s3 = run_stress_3_structural_morphogenesis()
    print("[СТРЕСС-ТЕСТ 3: СТРУКТУРНЫЙ МОРФОГЕНЕЗ ПОД НАГРУЗКОЙ (Инверсия синтаксиса SVO -> OVS)]")
    print(f"  Динамика [BDNF]: База={s3['bdnf_dynamics']['initial']} -> После SVO={s3['bdnf_dynamics']['after_canonical_svo']} -> Пик на инверсии OVS={s3['bdnf_dynamics']['peak_during_inverted_ovs']}")
    print(f"  Проращивание синапсов (Спрутинг): {s3['synaptogenesis_sprouting']['active_edges_initial']['total']} -> {s3['synaptogenesis_sprouting']['active_edges_after_ovs']['total']} (+{s3['synaptogenesis_sprouting']['net_new_synapses_added']} новых связей, по зонам: {s3['synaptogenesis_sprouting']['total_sprouted_by_zone']})")
    print(f"  Взрослый нейрогенез DG: После SVO={s3['dg_neurogenesis']['born_after_canonical_svo']} кл. -> После шоковой инверсии OVS={s3['dg_neurogenesis']['born_after_inverted_ovs']} кл.")
    print(f"  Сохранность старой грамматики SVO: retention={s3['old_grammar_preservation']['svo_synfire_retention_ratio']}, кортикальный щит cos_sim={s3['old_grammar_preservation']['cortical_shield_cosine_sim']}")
    print(f"  Генерация прямого строя (SVO): '{s3['old_grammar_preservation']['canonical_svo_generation']}'")
    print(f"  Генерация инвертированного строя (OVS): '{s3['old_grammar_preservation']['inverted_ovs_generation']}'")

    elapsed = round(time.time() - t0, 2)
    report = {
        "elapsed_sec": elapsed,
        "stress_1_stability_plasticity": s1,
        "stress_2_cognitive_dissonance": s2,
        "stress_3_structural_morphogenesis": s3,
    }
    out_json = ROOT / "Produce" / "ZIWAGO_PRIME_V6_STRESS_REPORT.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Отчёт сохранён в {out_json} за {elapsed} с.")


if __name__ == "__main__":
    main()
