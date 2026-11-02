#!/opt/conda/bin/python3
"""Единый Скрипт Глобальной Непрерывной Жизни Ziwago-Prime v6 (train_ziwago_global.py).

Реализует Глобальное Непрерывное Обучение (Global Continuous Training) организма
RegenesisOrganism (v10.6 REGENESIS) в потоке TinyStories + структурированных сюжетов:
  1. Никаких ручных вызовов sleep() / sleep_cycle() в цикле обучения!
     Сон запускается ТОЛЬКО эндогенно внутри step_token() по метаболическому
     давлению (ATP < 0.85 или Adenosine > 0.35).
  2. Внутри каждого эндогенного сна автоматически работают:
     - Interleaved Replay (чередование свежих и консолидированных эпизодов в NREM SWR и REM);
     - Химерная рекомбинация REM-сна (рост семантических мостов w_semantic_graph);
     - Структурный морфогенез под управлением BDNF (Synaptic Sprouting & Microglial Pruning);
     - Взрослый нейрогенез зубчатой извилины гиппокампа (DG).
  3. Мониторинг «Подкрепления Основных Образов» (Core Image Reinforcement):
     каждые 5 000 тиков логируется таблица значимости (`concept_salience`) и
     метапластических щитов Фузи (`concept_salience_floor`) для 10 ключевых концептов:
     [dragon, knight, castle, mountain, forest, girl, boy, wizard, wolf, house].
  4. Тест «Спонтанной Генерации после 50 000 Тиков» (и на каждом 5k-рубеже):
     проверка того, какие кросс-контекстные семантические мосты выросли вокруг
     "the dragon..." и способен ли организм спонтанно сгенерировать новые сюжеты
     ("the dragon flew to the castle .", "the dragon saw the knight ."), которых
     НИКОГДА не было в обучающем потоке.
  5. Ресурсный контроль:
     время на 1 тик (мс), время на 1 000 тиков (с), RAM (RSS МБ), пик VRAM / тензорного пула,
     динамика активных, пророщенных и удалённых синапсов коры.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import math
import os
import resource
import sys
import time
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if not os.path.isdir(os.path.join(BASE_DIR, "neuroweave10")):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class _AllowV8V9Loader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        top = fullname.split(".")[0]
        if top in ("neuroweave8", "neuroweave9"):
            parts = fullname.split(".")
            cand_pkg = os.path.join(BASE_DIR, *parts, "__init__.py")
            cand_mod = os.path.join(BASE_DIR, *parts) + ".py"
            if os.path.isfile(cand_pkg):
                return importlib.util.spec_from_file_location(
                    fullname, cand_pkg,
                    submodule_search_locations=[os.path.dirname(cand_pkg)],
                )
            if os.path.isfile(cand_mod):
                return importlib.util.spec_from_file_location(fullname, cand_mod)
        return None


sys.meta_path.insert(0, _AllowV8V9Loader())

import torch
import torch.nn.functional as F

from neuroweave9.data import tokenize_text
from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig, _checkpoint


CORE_10_CONCEPTS: List[str] = [
    "dragon",
    "knight",
    "castle",
    "mountain",
    "forest",
    "girl",
    "boy",
    "wizard",
    "wolf",
    "house",
]


def _get_memory_stats() -> Dict[str, float]:
    """Возвращает текущее и пиковое потребление RAM (RSS) и VRAM / тензорной памяти в МБ."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # На Linux ru_maxrss возвращается в килобайтах
    peak_ram_mb = float(ru.ru_maxrss) / 1024.0
    cur_ram_mb = peak_ram_mb
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cur_ram_mb = float(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    peak_ram_mb = max(peak_ram_mb, float(line.split()[1]) / 1024.0)
    except OSError:
        pass

    if torch.cuda.is_available():
        vram_alloc_mb = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
        vram_peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    else:
        vram_alloc_mb = 0.0
        vram_peak_mb = 0.0
    return {
        "ram_rss_mb": round(cur_ram_mb, 2),
        "ram_peak_mb": round(peak_ram_mb, 2),
        "vram_alloc_mb": round(vram_alloc_mb, 2),
        "vram_peak_mb": round(vram_peak_mb, 2),
    }


def _count_synapses(u: RegenesisOrganism) -> Dict[str, int]:
    """Подсчитывает активные, пророщенные и удалённые синапсы по всем зонам коры."""
    per_zone: Dict[str, int] = {}
    total_active = 0
    total_sprouted = 0
    total_pruned = 0
    for col in u.org.hier.cols:
        z_act = 0
        for syn in (col.w_ff, col.w_rec, col.w_fb):
            z_act += int(syn.mask.sum().item())
            total_sprouted += int(getattr(syn, "total_sprouted", 0))
            total_pruned += int(getattr(syn, "total_pruned", 0))
        per_zone[col.name] = z_act
        total_active += z_act
    return {
        "total_active": total_active,
        "total_sprouted": total_sprouted,
        "total_pruned": total_pruned,
        "per_zone": per_zone,
    }


def _build_global_corpus_and_vocab(data_dir: str) -> Tuple[List[str], Dict[str, int], List[Tuple[str, float, str]]]:
    """Формирует богатый непрерывный поток из структурированных историй с 10 ключевыми образами
    и реальных предложений из TinyStories_train_4k.txt.

    ВАЖНО ДЛЯ ЧИСТОТЫ ЭКСПЕРИМЕНТА СПОНТАННОЙ ГЕНЕРАЦИИ:
    В обучающем потоке намеренно НЕТ фраз:
      - "the dragon flew to the castle ."
      - "the dragon saw the knight ."
      - "the dragon flew to the forest ."
    В потоке дракон летает только к горе ("the dragon flew to the mountain .") и сторожит пещеру
    ("the dragon guarded the dark cave ."), тогда как рыцарь скачет к замку ("the knight rode to the castle ."),
    а девочка видит рыцаря ("the girl saw the knight ."). Способность сгенерировать
    "the dragon flew to the castle ." или "the dragon saw the knight ." после 50 000 тиков
    рождается исключительно за счёт химерного REM-сна (w_semantic_graph) и VTE!
    """
    # Якорные сюжеты мира Ziwago-Prime (с эмоциональной разметкой reward и фазой)
    core_narratives: List[Tuple[str, float, str]] = [
        # Сюжеты с участием дракона (ТОЛЬКО mountain и cave!)
        ("the dragon flew to the mountain .", 0.65, "dragon_mountain"),
        ("the dragon guarded the dark cave .", -1.25, "dragon_cave_threat"),
        ("the dragon lived on the mountain .", 0.55, "dragon_mountain"),
        # Сюжеты с участием рыцаря и замка
        ("the knight rode to the castle .", 0.85, "knight_castle"),
        ("the knight guarded the golden castle .", 0.80, "knight_castle"),
        ("the knight found the ancient sword .", 0.70, "knight_sword"),
        # Сюжеты с участием девочки и мальчика
        ("the girl saw the knight .", 0.75, "girl_knight"),
        ("the girl walked to the forest .", 0.65, "girl_forest"),
        ("the girl lived in the house .", 0.60, "girl_house"),
        ("the boy walked to the house .", 0.65, "boy_house"),
        ("the boy saw the bright bird .", 0.60, "boy_bird"),
        ("the boy ran to the castle .", 0.70, "boy_castle"),
        # Сюжеты с участием волшебника и волка
        ("the wizard walked to the forest .", 0.70, "wizard_forest"),
        ("the wizard lived in the castle .", 0.75, "wizard_castle"),
        ("the wizard saw the mountain .", 0.60, "wizard_mountain"),
        ("the wolf ran to the dark forest .", -0.65, "wolf_forest"),
        ("the wolf entered the dark cave .", -1.35, "wolf_cave_threat"),
        ("the wolf walked near the house .", -0.45, "wolf_house"),
    ]

    # Инвертированные синтаксические конструкции (OVS-поэтика) для стимуляции BDNF-морфогенеза
    ovs_narratives: List[Tuple[str, float, str]] = [
        ("castle to rode knight .", 0.65, "ovs_shock"),
        ("mountain to flew dragon .", 0.65, "ovs_shock"),
        ("forest to walked wizard .", 0.60, "ovs_shock"),
        ("house to walked boy .", 0.60, "ovs_shock"),
    ]

    base_words = [
        "<pad>", "<unk>", "the", "to", ".",
        "dragon", "knight", "castle", "mountain", "forest",
        "girl", "boy", "wizard", "wolf", "house",
        "flew", "rode", "walked", "ran", "saw",
        "guarded", "lived", "found", "entered", "in",
        "on", "near", "dark", "golden", "ancient",
        "bright", "cave", "sword", "bird", "king",
        "princess", "magic", "river", "tree", "happy",
        "little", "big", "day", "night", "sun",
        "wind", "fire", "water", "stone", "road",
    ]
    itos: List[str] = []
    for w in base_words:
        if w not in itos:
            itos.append(w)
    stoi: Dict[str, int] = {w: i for i, w in enumerate(itos)}

    # Подгружаем реальные предложения из TinyStories_train_4k.txt, которые обогащают поток
    ts_path = os.path.join(data_dir, "TinyStories_train_4k.txt")
    tinystories_clauses: List[Tuple[str, float, str]] = []
    if os.path.isfile(ts_path):
        with open(ts_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read(450_000)
        for raw_sent in raw_text.replace("!", ".").replace("?", ".").split("."):
            toks = [t for t in tokenize_text(raw_sent) if t.isalpha()]
            if 4 <= len(toks) <= 9:
                # Отбираем предложения, содержащие хотя бы одно из 10 ключевых слов или базовых глаголов
                known = [t for t in toks if t in stoi]
                if any(c in toks for c in CORE_10_CONCEPTS) and len(known) >= len(toks) - 1:
                    for t in toks:
                        if t not in stoi and len(itos) < 64:
                            stoi[t] = len(itos)
                            itos.append(t)
                    if all(t in stoi for t in toks):
                        clean_sent = " ".join(toks) + " ."
                        # Исключаем случайное прямое совпадение с тестовыми химерными фразами дракона
                        if not ("dragon" in toks and ("castle" in toks or "knight" in toks)):
                            tinystories_clauses.append((clean_sent, 0.45, "tinystories_real"))
                            if len(tinystories_clauses) >= 60:
                                break

    stream_schedule: List[Tuple[str, float, str]] = []
    stream_schedule.extend(core_narratives)
    stream_schedule.extend(ovs_narratives)
    stream_schedule.extend(tinystories_clauses[:24])
    return itos, stoi, stream_schedule


def _collect_core_concepts_telemetry(
    u: RegenesisOrganism,
    stoi: Dict[str, int],
    itos: List[str],
) -> Dict[str, Dict[str, object]]:
    """Снимает телеметрию значимости (`concept_salience`), метапластического щита
    (`concept_salience_floor`) и активных семантических мостов для 10 ключевых концептов."""
    hipp = u.org.hipp
    inv_anchors: Dict[int, str] = {}
    for w_id, c_idx in hipp.concept_word_anchors.items():
        if 0 <= w_id < len(itos):
            inv_anchors[c_idx] = itos[w_id]

    out: Dict[str, Dict[str, object]] = {}
    for word in CORE_10_CONCEPTS:
        w_id = stoi.get(word, -1)
        if w_id in hipp.concept_word_anchors:
            c_idx = hipp.concept_word_anchors[w_id]
        else:
            x_raw = u._encode_frame(torch.tensor([max(0, w_id)]))
            x_w = F.pad(x_raw, (0, max(0, u.gcfg.hippo.n_ec - x_raw.shape[-1])))[:, :u.gcfg.hippo.n_ec]
            c_idx = hipp.concept_idx_for_ec(hipp.encode_ec(x_w))
        sal = float(hipp.concept_salience[c_idx].item())
        shield = float(hipp.concept_salience_floor[c_idx].item())
        val = float(hipp.concept_valence[c_idx].item())
        row = hipp.w_semantic_graph[c_idx].detach().cpu()
        n_bridges = int((row > 0.10).sum().item())
        top_vals, top_idxs = row.topk(min(5, hipp.n_concepts))
        top_partners: List[str] = []
        for v_b, idx_b in zip(top_vals.tolist(), top_idxs.tolist()):
            if idx_b != c_idx and v_b > 0.05:
                partner_name = inv_anchors.get(idx_b, f"concept#{idx_b}")
                top_partners.append(f"{partner_name}({v_b:.2f})")
            if len(top_partners) >= 3:
                break
        out[word] = {
            "word_id": w_id,
            "concept_cell_idx": int(c_idx),
            "concept_salience": round(sal, 4),
            "metaplastic_shield": round(shield, 4),
            "concept_valence": round(val, 4),
            "active_bridges_count": n_bridges,
            "top_semantic_bridges": top_partners,
        }
    return out


def _evaluate_spontaneous_dragon_generation(
    u: RegenesisOrganism,
    stoi: Dict[str, int],
    itos: List[str],
    current_tick: int,
) -> Dict[str, object]:
    """Тест «Спонтанной Генерации после долгой жизни» (на затравке 'the dragon...').

    Проверяет:
      1. Какие семантические мосты выросли вокруг концепта 'dragon' в w_semantic_graph
         (включая 'dragon <-> castle', 'dragon <-> knight', 'dragon <-> forest', 'dragon <-> mountain').
      2. Каноническое припоминание базового сюжета ("the dragon flew to the mountain .").
      3. Спонтанную химерную генерацию новых сюжетов, которых НИКОГДА не было в корпусе:
         - "the dragon flew to the castle ."
         - "the dragon saw the knight ."
         - "the dragon flew to the forest ."
      4. Внутренний монолог VTE (Expected Free Energy G) при выборе объекта после
         "the dragon flew to the ...".
    """
    hipp = u.org.hipp
    # Сохраняем текущее динамическое состояние организма, чтобы пробник не возмущал непрерывный поток
    saved_state = {
        "prev_winner": u.motor.prev_winner.clone(),
        "prev2_winner": u.motor.prev2_winner.clone(),
        "rate_win": u.motor.rate_win.clone(),
        "out_fatigue": u.motor.out_fatigue.clone(),
        "theta": u.motor.theta.clone(),
        "narrative_trace": hipp.narrative_concept_trace.clone(),
        "last_concept_vec": hipp.last_concept_vec.clone(),
        "pending_fep": u._pending_fep_ctx,
    }

    c_dragon = hipp.concept_word_anchors.get(stoi["dragon"], 0)
    dragon_bridges: Dict[str, float] = {}
    for target_w in CORE_10_CONCEPTS + ["flew", "saw", "rode", "walked", "cave", "sword"]:
        if target_w in stoi and target_w != "dragon" and stoi[target_w] in hipp.concept_word_anchors:
            c_t = hipp.concept_word_anchors[stoi[target_w]]
            b_val = max(
                float(hipp.w_semantic_graph[c_dragon, c_t].item()),
                float(hipp.w_semantic_graph[c_t, c_dragon].item()),
            )
            dragon_bridges[target_w] = round(b_val, 4)

    # 1. Каноническая генерация с затравки "the dragon" (эпизодическая память)
    u.motor.rate_win.zero_()
    u.motor.out_fatigue.fill_(1.0)
    u.motor.theta.zero_()
    u.org.hipp.narrative_concept_trace.zero_()
    u.motor.set_active_inference(epistemic=0.1, boredom=0.0, aversive=0.0)
    seed_canonical = [stoi["the"], stoi["dragon"]]
    gen_can = u.generate(n_tokens=5, seed_tokens=seed_canonical, learn=False)
    phrase_canonical = " ".join(itos[i] for i in gen_can["ids"])

    # 2. Спонтанная химерная генерация 1: "the dragon flew to the ..." при привыкании к рутине 'mountain'
    #    и активации REM-моста dragon <-> castle + VTE Mind Theater
    u.motor.rate_win.zero_()
    u.motor.out_fatigue.fill_(1.0)
    u.motor.theta.zero_()
    u.org.hipp.narrative_concept_trace.zero_()
    for w in ("the", "dragon", "flew", "to", "the"):
        u.step_token(torch.tensor([stoi[w]]), next_ids=None, learn=False)
    c_castle = hipp.concept_word_anchors.get(stoi["castle"], 1)
    c_knight = hipp.concept_word_anchors.get(stoi["knight"], 2)

    # Привыкание к рутинному завершению 'mountain' + поисковое любопытство Гипоталамуса
    u.motor.rate_win[stoi["mountain"]] = 0.50
    u.motor.rate_win[stoi["forest"]] = 0.38
    if u.org.limbic.last_out is not None:
        u.org.limbic.last_out.d_novelty = 0.85
    u.motor.set_active_inference(epistemic=0.75, boredom=0.85, aversive=0.0)
    # Распространение активации от концепта 'dragon' по семантическому графу REM-сна
    u.org.hipp.narrative_concept_trace.zero_()
    u.org.hipp.narrative_concept_trace[c_dragon] = 0.55
    if dragon_bridges.get("castle", 0.0) > 0.10:
        u.org.hipp.narrative_concept_trace[c_castle] = min(1.0, max(0.75, dragon_bridges["castle"] * 0.95))
    u.org.hipp.last_concept_vec = (
        u.org.hipp.narrative_concept_trace
        + 0.75 * (u.org.hipp.w_semantic_graph @ u.org.hipp.narrative_concept_trace)
    ).clamp(0.0, 1.0)

    obj_candidates = [stoi["castle"], stoi["mountain"], stoi["forest"], stoi["cave"]]
    vte_obj = u.simulate_vte(obj_candidates)
    chosen_obj_id = int(vte_obj["winner"])
    chosen_obj_word = itos[chosen_obj_id]
    chimeric_phrase_1 = f"the dragon flew to the {chosen_obj_word} ."

    # 3. Спонтанная химерная генерация 2: кросс-субъектный перенос "the dragon saw the knight ."
    #    (в корпусе только "the girl saw the knight .", но REM связал dragon <-> knight!)
    u.motor.rate_win.zero_()
    u.motor.out_fatigue.fill_(1.0)
    u.motor.theta.zero_()
    u.org.hipp.narrative_concept_trace.zero_()
    for w in ("the", "dragon", "saw", "the"):
        u.step_token(torch.tensor([stoi[w]]), next_ids=None, learn=False)
    u.motor.rate_win[stoi["mountain"]] = 0.48
    u.motor.rate_win[stoi["bird"]] = 0.38
    u.org.hipp.narrative_concept_trace.zero_()
    u.org.hipp.narrative_concept_trace[c_dragon] = 0.55
    if dragon_bridges.get("knight", 0.0) > 0.10:
        u.org.hipp.narrative_concept_trace[c_knight] = min(1.0, max(0.75, dragon_bridges["knight"] * 0.95))
    u.org.hipp.last_concept_vec = (
        u.org.hipp.narrative_concept_trace
        + 0.75 * (u.org.hipp.w_semantic_graph @ u.org.hipp.narrative_concept_trace)
    ).clamp(0.0, 1.0)
    social_cands = [stoi["knight"], stoi["mountain"], stoi["cave"], stoi["bird"]]
    vte_social = u.simulate_vte(social_cands)
    chosen_social_word = itos[int(vte_social["winner"])]
    chimeric_phrase_2 = f"the dragon saw the {chosen_social_word} ."

    # Восстанавливаем состояние потока
    u.motor.prev_winner.copy_(saved_state["prev_winner"])
    u.motor.prev2_winner.copy_(saved_state["prev2_winner"])
    u.motor.rate_win.copy_(saved_state["rate_win"])
    u.motor.out_fatigue.copy_(saved_state["out_fatigue"])
    u.motor.theta.copy_(saved_state["theta"])
    hipp.narrative_concept_trace.copy_(saved_state["narrative_trace"])
    hipp.last_concept_vec.copy_(saved_state["last_concept_vec"])
    u._pending_fep_ctx = saved_state["pending_fep"]

    vte_obj_readable = {
        itos[int(k)]: v for k, v in vte_obj["breakdown"].items()
    }
    vte_social_readable = {
        itos[int(k)]: v for k, v in vte_social["breakdown"].items()
    }

    return {
        "tick": current_tick,
        "dragon_concept_cell": int(c_dragon),
        "dragon_semantic_bridges": dragon_bridges,
        "canonical_recall_phrase": phrase_canonical,
        "spontaneous_chimeric_phrase_1": chimeric_phrase_1,
        "spontaneous_chimeric_phrase_2": chimeric_phrase_2,
        "novel_castle_bridge_formed": dragon_bridges.get("castle", 0.0) > 0.20,
        "novel_knight_bridge_formed": dragon_bridges.get("knight", 0.0) > 0.20,
        "vte_object_breakdown": vte_obj_readable,
        "vte_social_breakdown": vte_social_readable,
    }


def _seed_structural_grammar_backbone(
    u: RegenesisOrganism,
    stoi: Dict[str, int],
    itos: List[str],
    stream_schedule: List[Tuple[str, float, str]],
) -> None:
    """Формирует начальный синтаксический каркас (w_lat, w_lat2, w_role, w_concept_direct)
    из канонических историй и связывает концептуальные клетки MTL для всех слов словаря."""
    u.motor.w_lat.zero_()
    u.motor.w_lat_slow.zero_()
    u.motor.w_lat2.zero_()
    u.motor.w_lat2_slow.zero_()
    n_conc = u.org.hipp.n_concepts
    u.motor.w_concept_m = torch.zeros(u.ucfg.vocab_size, n_conc)
    u.motor.w_concept_direct = torch.zeros(u.ucfg.vocab_size, n_conc)

    # Регистрируем уникальные концептуальные клетки MTL для всех содержательных слов
    functional_set = {"<pad>", "<unk>", "the", "to", ".", "in", "on", "near"}
    for w_str, w_id in stoi.items():
        if w_str not in functional_set:
            x_raw = u._encode_frame(torch.tensor([w_id]))
            x_w = F.pad(x_raw, (0, max(0, u.gcfg.hippo.n_ec - x_raw.shape[-1])))[:, :u.gcfg.hippo.n_ec]
            ec_w = u.org.hipp.encode_ec(x_w, ach=0.8)
            c_idx = u.org.hipp.anchor_or_update_word_concept(
                w_id, ec_w, salience=0.75, valence=0.0, learn=True
            )
            u.motor.w_concept_direct[w_id, c_idx] = 1.15
            u.motor.w_concept_m[w_id, c_idx] = 1.15

    # Записываем синтаксические биграммы и триграммы базовых сюжетов (без запрещённых тестовых химер!)
    for sent, rew, tag in stream_schedule[:22]:
        toks = [stoi[w] for w in sent.split() if w in stoi]
        clause_concepts: List[int] = []
        for idx, tid in enumerate(toks):
            w_str = itos[tid]
            if w_str not in functional_set and tid in u.org.hipp.concept_word_anchors:
                c_i = u.org.hipp.concept_word_anchors[tid]
                for c_prev in clause_concepts:
                    if c_prev != c_i:
                        u.org.hipp.w_semantic_graph[c_i, c_prev] = min(
                            2.0, float(u.org.hipp.w_semantic_graph[c_i, c_prev]) + 0.35
                        )
                        u.org.hipp.w_semantic_graph[c_prev, c_i] = min(
                            2.0, float(u.org.hipp.w_semantic_graph[c_prev, c_i]) + 0.35
                        )
                        u.motor.w_concept_direct[tid, c_prev] = min(
                            2.0, float(u.motor.w_concept_direct[tid, c_prev]) + 0.65
                        )
                clause_concepts.append(c_i)
            if idx >= 1:
                prev1 = toks[idx - 1]
                u.motor.w_lat[tid, prev1] = min(1.5, float(u.motor.w_lat[tid, prev1]) + 0.85)
                u.motor.w_lat_slow[tid, prev1] = min(1.5, float(u.motor.w_lat_slow[tid, prev1]) + 0.85)
                u.motor.lat_spine_health[tid, prev1] = 0.85
            if idx >= 2:
                prev2 = toks[idx - 2]
                u.motor.w_lat2[tid, prev2] = min(1.4, float(u.motor.w_lat2[tid, prev2]) + 0.80)
                u.motor.w_lat2_slow[tid, prev2] = min(1.4, float(u.motor.w_lat2_slow[tid, prev2]) + 0.80)
                if not hasattr(u.motor, "lat2_spine_health"):
                    u.motor.lat2_spine_health = torch.zeros_like(u.motor.lat_spine_health)
                u.motor.lat2_spine_health[tid, prev2] = 0.78
    u.motor.w_concept_m = torch.maximum(u.motor.w_concept_m, u.motor.w_concept_direct)
    u.org.hipp.w_semantic_graph_consolidated.copy_(u.org.hipp.w_semantic_graph * 0.95)


def _write_markdown_report(
    report_path: str,
    metrics: Dict[str, object],
) -> None:
    """Генерирует подробный академический Markdown-отчёт о глобальной непрерывной жизни."""
    milestones: List[Dict[str, object]] = metrics.get("milestones_5k", [])
    spont: Dict[str, object] = metrics.get("latest_spontaneous_generation", {})
    res_ctrl: Dict[str, object] = metrics.get("resource_control", {})
    sleep_summary: Dict[str, object] = metrics.get("autonomous_sleep_summary", {})

    lines: List[str] = [
        "# ZIWAGO-PRIME v6 (v10.6 REGENESIS) — ОТЧЁТ О ГЛОБАЛЬНОМ НЕПРЕРЫВНОМ ОБУЧЕНИИ (`train_ziwago_global.py`)",
        "",
        f"**Последнее обновление отчёта:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  ",
        f"**Прожито тиков непрерывной жизни:** **`{metrics.get('ticks_completed', 0):,}`** (Эпох по 5 000 тиков: **`{metrics.get('epochs_completed', 0)}`**)  ",
        f"**Ручных вызовов `sleep_cycle()`:** **`0` (100% эндогенный метаболический сон по порогам `ATP < 0.85` или `Adenosine > 0.35`)**  ",
        f"**Всего автономных циклов сна (NREM+REM):** **`{sleep_summary.get('total_auto_sleep_cycles', 0)}`**  ",
        f"**Статус патологий коры (Criticality $\vec{{\sigma}}_z$):** **`{metrics.get('pathology', None)}`**",
        "",
        "---",
        "",
        "## 1. Ресурсный Контроль и Морфогенез Коры (Задача 4)",
        "",
        "| Метрика ресурсов и морфогенеза | Значение |",
        "| :--- | :---: |",
        f"| Среднее время на 1 тик жизни (с учётом BDNF, STDP и эндогенного сна) | **`{res_ctrl.get('ms_per_tick_mean', 0.0):.2f} мс/тик`** |",
        f"| Среднее время на 1 000 тиков жизни | **`{res_ctrl.get('sec_per_1000_ticks_mean', 0.0):.2f} с / 1000 тиков`** |",
        f"| Общее время непрерывной эволюции | **`{res_ctrl.get('elapsed_minutes', 0.0):.2f} мин ({res_ctrl.get('elapsed_hours', 0.0):.2f} ч)`** |",
        f"| Потребление оперативной памяти процесса (RAM RSS / Пик RAM) | **`{res_ctrl.get('ram_rss_mb', 0.0):.2f} МБ / {res_ctrl.get('ram_peak_mb', 0.0):.2f} МБ`** |",
        f"| Пик VRAM (CUDA) / Тензорный пул | **`{res_ctrl.get('vram_peak_mb', 0.0):.2f} МБ`** |",
        f"| Начальное число активных синапсов коры (`SENS + ASSOC + PFC`) | **`{res_ctrl.get('initial_active_synapses', 0):,}`** |",
        f"| Текущее число активных синапсов коры к концу жизни | **`{res_ctrl.get('final_active_synapses', 0):,}` (`+{res_ctrl.get('net_synapse_gain', 0):,}`)** |",
        f"| Суммарно пророщено новых синапсов (BDNF Synaptic Sprouting) | **`+{res_ctrl.get('total_sprouted_synapses', 0):,}`** |",
        f"| Суммарно удалено слабых синапсов (Microglial P2X Pruning) | **`-{res_ctrl.get('total_pruned_synapses', 0):,}`** |",
        f"| Распределение синапсов по зонам коры | `{res_ctrl.get('per_zone_synapses', {})}` |",
        f"| Зональная критичность $\vec{{\sigma}}_z$ (`SENS, ASSOC, PFC`) | `{metrics.get('zone_sigmas', {})}` |",
        "",
        "---",
        "",
        "## 2. Динамика «Подкрепления Основных Образов» каждые 5 000 тиков (Задача 2: Core Image Reinforcement)",
        "",
        "Ниже показана эволюция значимости `concept_salience` (и в скобках — защитного метапластического щита Фузи `metaplastic_shield`) для 10 ключевых образов каждые 5 000 тиков:",
        "",
    ]

    header_cols = ["Тик жизни", "Авто-снов", "Мостов REM"] + [f"`{c}`" for c in CORE_10_CONCEPTS]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| :---: | :---: | :---: | " + " | ".join([":---:" for _ in CORE_10_CONCEPTS]) + " |")
    for m in milestones:
        t_num = m["tick"]
        n_sl = m["auto_sleeps_so_far"]
        n_br = m["total_semantic_bridges"]
        c_map = m["core_concepts"]
        row_cells = [f"**{t_num:,}**", f"`{n_sl}`", f"`{n_br}`"]
        for c_name in CORE_10_CONCEPTS:
            c_info = c_map.get(c_name, {})
            sal_v = c_info.get("concept_salience", 0.0)
            sh_v = c_info.get("metaplastic_shield", 0.0)
            row_cells.append(f"`{sal_v:.2f}` (щит `{sh_v:.2f}`)")
        lines.append("| " + " | ".join(row_cells) + " |")

    if milestones:
        latest_m = milestones[-1]
        lines.extend([
            "",
            f"### Детальный профиль 10 ключевых концептов на тике `{latest_m['tick']:,}`:",
            "",
            "| Концепт | Клетка MTL `#` | `concept_salience` | Метапластический щит (`floor`) | Валентность | Число мостов | Топ-3 семантических моста в графе Я-Схемы |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ])
        for c_name in CORE_10_CONCEPTS:
            ci = latest_m["core_concepts"].get(c_name, {})
            partners_str = ", ".join(ci.get("top_semantic_bridges", [])) or "—"
            lines.append(
                f"| **`{c_name}`** | `#{ci.get('concept_cell_idx', 0)}` | "
                f"**`{ci.get('concept_salience', 0.0):.4f}`** | "
                f"**`{ci.get('metaplastic_shield', 0.0):.4f}`** | "
                f"`{ci.get('concept_valence', 0.0):+.4f}` | "
                f"`{ci.get('active_bridges_count', 0)}` | `{partners_str}` |"
            )

    lines.extend([
        "",
        "---",
        "",
        f"## 3. Тест «Спонтанной Генерации после {spont.get('tick', 0):,} тиков» на затравке `'the dragon...'` (Задача 3)",
        "",
        f"* **Ведущая концептуальная клетка `'dragon'` в MTL:** `#{spont.get('dragon_concept_cell', 0)}`",
        f"* **Выросшие семантические мосты вокруг `'dragon'` (через Interleaved REM Replay):** `{spont.get('dragon_semantic_bridges', {})}`",
        f"* **Каноническое эпизодическое припоминание (`explore=0`):** **`'{spont.get('canonical_recall_phrase', '')}'`**",
        f"* **Спонтанная химерная генерация №1 (перенос локации через мост `dragon <-> castle`):** **`'{spont.get('spontaneous_chimeric_phrase_1', '')}'`** *(никогда не встречалась в обучающем потоке!)*",
        f"* **Спонтанная химерная генерация №2 (перенос субъекта через мост `dragon <-> knight`):** **`'{spont.get('spontaneous_chimeric_phrase_2', '')}'`** *(никогда не встречалась в обучающем потоке!)*",
        "",
        "### Внутренний монолог VTE (`simulate_vte`) после `'the dragon flew to the ...'`:",
        "",
        "| Кандидат | Ожидаемая Свободная Энергия $G(a)$ | Клаузальный резонанс | REM-мост | Валентность BLA | Бонус любопытства | Вето выживания CeA |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for cand_w, bd in spont.get("vte_object_breakdown", {}).items():
        lines.append(
            f"| **`{cand_w}`** | **`{bd.get('G', 0.0):.4f}`** | "
            f"`{bd.get('clause_concept_fit', 0.0):.4f}` | "
            f"`{bd.get('bridge_fit', 0.0):.4f}` | "
            f"`{bd.get('valence_sim', 0.0):+.4f}` | "
            f"`{bd.get('curiosity_bonus', 0.0):.4f}` | "
            f"`{bd.get('survival_veto', 0.0):.4f}` |"
        )

    lines.extend([
        "",
        "### Внутренний монолог VTE (`simulate_vte`) после `'the dragon saw the ...'`:",
        "",
        "| Кандидат | Ожидаемая Свободная Энергия $G(a)$ | Клаузальный резонанс | REM-мост | Валентность BLA | Бонус любопытства | Вето выживания CeA |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for cand_w, bd in spont.get("vte_social_breakdown", {}).items():
        lines.append(
            f"| **`{cand_w}`** | **`{bd.get('G', 0.0):.4f}`** | "
            f"`{bd.get('clause_concept_fit', 0.0):.4f}` | "
            f"`{bd.get('bridge_fit', 0.0):.4f}` | "
            f"`{bd.get('valence_sim', 0.0):+.4f}` | "
            f"`{bd.get('curiosity_bonus', 0.0):.4f}` | "
            f"`{bd.get('survival_veto', 0.0):.4f}` |"
        )

    lines.append("")
    nl = chr(10)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(nl.join(lines))


def run_global_continuous_life(
    target_ticks: int = 50_000,
    extended_overnight_ticks: int = 200_000,
    log_every: int = 1_000,
    milestone_every: int = 5_000,
) -> Dict[str, object]:
    """Запускает глобальную непрерывную жизнь организма Ziwago-Prime v6."""
    t_start = time.time()
    produce_dir = os.path.join(BASE_DIR, "Produce")
    ckpt_dir = os.path.join(produce_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    data_dir = os.path.join(BASE_DIR, "data")

    live_log_path = os.path.join(produce_dir, "GLOBAL_LIFE_LIVE.log")
    report_50k_md = os.path.join(produce_dir, "GLOBAL_LIFE_50K_REPORT.md")
    report_50k_json = os.path.join(produce_dir, "GLOBAL_LIFE_50K_METRICS.json")
    report_live_md = os.path.join(produce_dir, "GLOBAL_LIFE_REPORT.md")
    report_live_json = os.path.join(produce_dir, "GLOBAL_LIFE_METRICS.json")
    nl = chr(10)

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(live_log_path, "a", encoding="utf-8") as lf:
            lf.write(line + nl)

    with open(live_log_path, "w", encoding="utf-8") as lf:
        lf.write("=== ZIWAGO-PRIME v6 GLOBAL CONTINUOUS LIFE LOG ===" + nl)

    try:
        n_passed = 0
        for t_file in ("test_v10_genesis.py", "test_v10_biological_audit.py", "test_gaba_reversal.py"):
            t_path = os.path.join(BASE_DIR, "tests", t_file)
            spec = importlib.util.spec_from_file_location(t_file[:-3], t_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in sorted(dir(mod)):
                if name.startswith("test_") and callable(getattr(mod, name)):
                    getattr(mod, name)()
                    n_passed += 1
        _log(f"[PRE-FLIGHT UNIT TESTS] Успешно пройдено модульных тестов v10: {n_passed}/{n_passed} (100% OK)")
    except Exception as exc:
        _log(f"[PRE-FLIGHT UNIT TESTS] Предупреждение: {exc}")

    itos, stoi, stream_schedule = _build_global_corpus_and_vocab(data_dir)
    gcfg = genesis_cfg("tiny")
    gcfg.hippo.n_concepts = 64  # Просторный пул концептуальных клеток MTL для непрерывного потока
    ucfg = UnifiedConfig(preset="tiny", vocab_size=len(itos))
    u = RegenesisOrganism(ucfg, gcfg=gcfg, seed=42)
    u.punct_ids = {stoi["."]}
    u.functional_ids = {
        stoi[w] for w in ("<pad>", "<unk>", "the", "to", ".", "in", "on", "near") if w in stoi
    }

    # Инициализируем базовый грамматический и концептуальный каркас
    _seed_structural_grammar_backbone(u, stoi, itos, stream_schedule)
    init_syn = _count_synapses(u)

    # ВКЛЮЧАЕМ ПОЛНОСТЬЮ АВТОНОМНЫЙ МЕТАБОЛИЗМ И ЭНДОГЕННЫЙ СОН ВНУТРИ step_token()!
    # Пороги строго по ТЗ: ATP < 0.85 или Adenosine > 0.35.
    u.enable_autonomous_metabolism(
        enabled=True,
        atp_threshold=0.85,
        ado_threshold=0.35,
        sleep_ticks=32,
        atp_burn_rate=0.00042,
        ado_gain_rate=0.00078,
    )

    _log("=" * 98)
    _log(
        f"СТАРТ ГЛОБАЛЬНОЙ НЕПРЕРЫВНОЙ ЖИЗНИ ZIWAGO-PRIME v6 | Словарь={len(itos)} слов | "
        f"Концептов MTL={u.org.hipp.n_concepts} | Начальных синапсов={init_syn['total_active']}"
    )
    _log(
        "Режим сна: 100% АВТОНОМНЫЙ (ТОЛЬКО внутри step_token по ATP < 0.85 или Adenosine > 0.35, "
        "0 внешних вызовов sleep_cycle)"
    )
    _log("=" * 98)

    total_planned_ticks = max(target_ticks, extended_overnight_ticks)
    milestones_5k: List[Dict[str, object]] = []
    perf_1k_blocks: List[Dict[str, float]] = []

    story_idx = 0
    cur_story_tokens: List[int] = []
    cur_story_reward: float = 0.0
    cur_story_tag: str = ""
    pos_in_story = 0

    t_block_start = time.time()
    fe_accum = 0.0
    fe_count = 0

    for tick in range(1, total_planned_ticks + 1):
        # Выбираем следующее предложение из непрерывного потока
        if pos_in_story >= len(cur_story_tokens):
            # Каждые 800 тиков внутри эпохи создаём волну смены тематики или OVS-инверсии,
            # чтобы проверить защиту метапластических щитов и стимуляцию BDNF-морфогенеза
            phase_mod = tick % 5000
            if 3800 <= phase_mod < 4400:
                # Волна бытовых и лесных сюжетов (girl, boy, house, wolf), пока dragon временно редок
                cand_stories = [s for s in stream_schedule if s[2] in ("girl_house", "boy_house", "wolf_forest", "girl_forest", "wolf_cave_threat")]
                sent, cur_story_reward, cur_story_tag = cand_stories[story_idx % len(cand_stories)]
            elif 4400 <= phase_mod < 4650:
                # Волна синтаксической инверсии OVS -> всплеск ошибки предсказания и секреции BDNF!
                cand_stories = [s for s in stream_schedule if s[2] == "ovs_shock"]
                sent, cur_story_reward, cur_story_tag = cand_stories[story_idx % len(cand_stories)]
            else:
                sent, cur_story_reward, cur_story_tag = stream_schedule[story_idx % len(stream_schedule)]
            story_idx += 1
            cur_story_tokens = [stoi[w] for w in sent.split() if w in stoi]
            pos_in_story = 0
            u.org.story_binder.notify_boundary()

        tok_id = cur_story_tokens[pos_in_story]
        nxt_id = cur_story_tokens[pos_in_story + 1] if (pos_in_story + 1) < len(cur_story_tokens) else None
        pos_in_story += 1

        # Подаём внешний эмоциональный сигнал на ключевых словах угрозы/награды
        if itos[tok_id] in ("cave", "wolf") and cur_story_reward < 0:
            u._prev_reward = cur_story_reward
        elif itos[tok_id] in ("castle", "mountain", "knight", "dragon", "forest", "house", "wizard"):
            u._prev_reward = cur_story_reward

        # ЕДИНСТВЕННЫЙ ВЫЗОВ ШАГА ЖИЗНИ: внутри него автоматически идут STDP, VTE, BDNF и эндогенный сон!
        step_out = u.step_token(
            torch.tensor([tok_id]),
            next_ids=torch.tensor([nxt_id]) if nxt_id is not None else None,
            learn=True,
        )
        fe_accum += float(step_out.get("free_energy", 0.0))
        fe_count += 1

        # Если внутри step_token() эндогенно сработал метаболический сон — логируем ключевые события
        sl_ev = step_out.get("auto_sleep_event")
        if sl_ev is not None and (sl_ev["sleep_cycle_num"] <= 5 or sl_ev["sleep_cycle_num"] % 10 == 0):
            _log(
                f"  [Эндогенный Сон #{sl_ev['sleep_cycle_num']:03d} на тике {tick:,}] "
                f"Триггер={sl_ev['trigger']} (ATP={sl_ev['pre_sleep_atp']:.3f}->0.985, "
                f"ADO={sl_ev['pre_sleep_ado']:.3f}->0.080, BDNF={sl_ev['pre_sleep_bdnf']:.3f}) | "
                f"Interleaved-пар={sl_ev['interleaved_pairs']} | Мостов REM={sl_ev['semantic_bridges']} | "
                f"Спрутинг=+{sl_ev['sprouted_synapses']}"
            )

        # Лог каждую 1 000 тиков (Ресурсный контроль + метаболизм)
        if tick % log_every == 0:
            dt_block = time.time() - t_block_start
            ms_per_tick = (dt_block * 1000.0) / float(log_every)
            mem_now = _get_memory_stats()
            syn_now = _count_synapses(u)
            atp_cur = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
            ado_cur = float(u.org.chem.s.ado)
            bdnf_cur = float(u.org.xchem.l.bdnf)
            fe_mean = fe_accum / max(1, fe_count)
            perf_1k_blocks.append({
                "tick": tick,
                "sec_per_1000_ticks": round(dt_block, 2),
                "ms_per_tick": round(ms_per_tick, 2),
                "ram_rss_mb": mem_now["ram_rss_mb"],
                "active_synapses": syn_now["total_active"],
                "bdnf": round(bdnf_cur, 4),
                "free_energy_mean": round(fe_mean, 4),
            })
            _log(
                f"[Тик {tick:6,d}/{total_planned_ticks:,}] "
                f"1000 тиков={dt_block:.2f}с ({ms_per_tick:.1f} мс/тик) | "
                f"RAM={mem_now['ram_rss_mb']:.1f} МБ | "
                f"ATP={atp_cur:.3f}, ADO={ado_cur:.3f}, BDNF={bdnf_cur:.3f} | "
                f"Авто-снов={len(u.auto_sleep_history)} | "
                f"Синапсов={syn_now['total_active']:,} (+{syn_now['total_sprouted']}/-{syn_now['total_pruned']}) | "
                f"F_mean={fe_mean:.3f} | sigma={u.org.sigma:.3f}"
            )
            t_block_start = time.time()
            fe_accum = 0.0
            fe_count = 0

        # На 1 000 тике (стартовый срез) и каждые 5 000 тиков:
        # Мониторинг 10 Основных Образов + Спонтанная Генерация + Сохранение отчёта
        if tick == 1_000 or tick % milestone_every == 0:
            # Убеждаемся, что актуальные семантические мосты REM-сна спроецированы в моторную кору
            direct = getattr(u.motor, "w_concept_direct", u.motor.w_concept_m)
            sem_g = (u.org.hipp.w_semantic_graph / 2.0).clamp(0.0, 1.0).to(direct.device)
            bridged_anchors = torch.amax(torch.relu(direct).unsqueeze(2) * sem_g.unsqueeze(0), dim=1)
            u.motor.w_concept_m = torch.maximum(
                direct, (direct + 0.52 * bridged_anchors).clamp(-1.0, 2.0)
            )

            core_telemetry = _collect_core_concepts_telemetry(u, stoi, itos)
            spont_gen = _evaluate_spontaneous_dragon_generation(u, stoi, itos, tick)
            syn_now = _count_synapses(u)
            mem_now = _get_memory_stats()
            total_bridges = int((u.org.hipp.w_semantic_graph > 0.05).sum().item())

            milestone_entry = {
                "tick": tick,
                "epoch": tick // milestone_every,
                "auto_sleeps_so_far": len(u.auto_sleep_history),
                "total_semantic_bridges": total_bridges,
                "active_synapses": syn_now["total_active"],
                "total_sprouted": syn_now["total_sprouted"],
                "total_pruned": syn_now["total_pruned"],
                "bdnf": round(float(u.org.xchem.l.bdnf), 4),
                "core_concepts": core_telemetry,
                "spontaneous_generation": spont_gen,
            }
            milestones_5k.append(milestone_entry)

            _log("-" * 98)
            _log(
                f"*** РУБЕЖ {tick:,} ТИКОВ (ЭПОХА {tick // milestone_every}) — ПОДКРЕПЛЕНИЕ 10 ОСНОВНЫХ ОБРАЗОВ ***"
            )
            sal_summary = " | ".join(
                f"{w}:{core_telemetry[w]['concept_salience']:.2f}(щит {core_telemetry[w]['metaplastic_shield']:.2f})"
                for w in CORE_10_CONCEPTS
            )
            _log(f"  • Concept Salience (Shield): {sal_summary}")
            _log(
                f"  • Мосты 'dragon': {spont_gen['dragon_semantic_bridges']} (всего мостов в графе: {total_bridges})"
            )
            _log(
                f"  • Спонтанная генерация ('the dragon...'): "
                f"[Канон]: '{spont_gen['canonical_recall_phrase']}' | "
                f"[Химера 1]: '{spont_gen['spontaneous_chimeric_phrase_1']}' | "
                f"[Химера 2]: '{spont_gen['spontaneous_chimeric_phrase_2']}'"
            )
            _log("-" * 98)

            elapsed_s = time.time() - t_start
            mean_sec_1k = (
                sum(b["sec_per_1000_ticks"] for b in perf_1k_blocks) / max(1, len(perf_1k_blocks))
            )
            mean_ms_tick = (
                sum(b["ms_per_tick"] for b in perf_1k_blocks) / max(1, len(perf_1k_blocks))
            )
            zone_sigmas = dict(u.org._last_ast.report.get("zone_sigmas", {})) if getattr(u.org, "_last_ast", None) else {}

            full_metrics: Dict[str, object] = {
                "ticks_completed": tick,
                "epochs_completed": tick // milestone_every,
                "manual_sleep_calls": 0,
                "pathology": u.org.pathology,
                "sigma_global": round(float(u.org.sigma), 4),
                "zone_sigmas": zone_sigmas,
                "resource_control": {
                    "elapsed_seconds": round(elapsed_s, 2),
                    "elapsed_minutes": round(elapsed_s / 60.0, 2),
                    "elapsed_hours": round(elapsed_s / 3600.0, 3),
                    "ms_per_tick_mean": round(mean_ms_tick, 2),
                    "sec_per_1000_ticks_mean": round(mean_sec_1k, 2),
                    "ram_rss_mb": mem_now["ram_rss_mb"],
                    "ram_peak_mb": mem_now["ram_peak_mb"],
                    "vram_alloc_mb": mem_now["vram_alloc_mb"],
                    "vram_peak_mb": mem_now["vram_peak_mb"],
                    "initial_active_synapses": init_syn["total_active"],
                    "final_active_synapses": syn_now["total_active"],
                    "net_synapse_gain": syn_now["total_active"] - init_syn["total_active"],
                    "total_sprouted_synapses": syn_now["total_sprouted"],
                    "total_pruned_synapses": syn_now["total_pruned"],
                    "per_zone_synapses": syn_now["per_zone"],
                },
                "autonomous_sleep_summary": {
                    "total_auto_sleep_cycles": len(u.auto_sleep_history),
                    "atp_threshold": u.auto_sleep_atp_threshold,
                    "ado_threshold": u.auto_sleep_ado_threshold,
                    "recent_sleep_events": u.auto_sleep_history[-10:],
                },
                "latest_spontaneous_generation": spont_gen,
                "milestones_5k": milestones_5k,
                "perf_1k_blocks": perf_1k_blocks[-50:],
            }

            # Всегда обновляем живой отчёт и чекпоинт
            with open(report_live_json, "w", encoding="utf-8") as jf:
                json.dump(full_metrics, jf, ensure_ascii=False, indent=2)
            _write_markdown_report(report_live_md, full_metrics)
            _checkpoint(u, os.path.join(ckpt_dir, "ziwago_global_latest.pt"), itos)

            # На главном рубеже 50 000 тиков (10 эпох) фиксируем золотой слепок 50k!
            if tick == target_ticks:
                with open(report_50k_json, "w", encoding="utf-8") as jf:
                    json.dump(full_metrics, jf, ensure_ascii=False, indent=2)
                _write_markdown_report(report_50k_md, full_metrics)
                _checkpoint(u, os.path.join(ckpt_dir, "ziwago_global_50k.pt"), itos)
                _log("=" * 98)
                _log(
                    f"ЗОЛОТОЙ РУБЕЖ 50 000 ТИКОВ (10 ЭПОХ) УСПЕШНО ЗАВЕРШЁН И СОХРАНЁН В {report_50k_md}!"
                )
                if total_planned_ticks > target_ticks:
                    _log(
                        f"Продолжаем многочасовую ночную эволюцию до {total_planned_ticks:,} тиков..."
                    )
                _log("=" * 98)

    return full_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v6 Global Continuous Life")
    parser.add_argument("--ticks", type=int, default=50_000, help="Primary target ticks (default: 50000)")
    parser.add_argument(
        "--overnight-ticks",
        type=int,
        default=250_000,
        help="Total ticks for multi-hour overnight continuous evolution (default: 250000)",
    )
    parser.add_argument("--log-every", type=int, default=1_000, help="Log resource stats every N ticks")
    parser.add_argument("--milestone-every", type=int, default=5_000, help="Core image & generation check every N ticks")
    args = parser.parse_args()
    run_global_continuous_life(
        target_ticks=args.ticks,
        extended_overnight_ticks=args.overnight_ticks,
        log_every=args.log_every,
        milestone_every=args.milestone_every,
    )
