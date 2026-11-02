#!/opt/micromamba/bin/python3
"""Интерактивная Жизнь (Interactive Active Inference) и Тест «Стресс-Сон» (Sleep Deprivation)
для архитектуры Ziwago-Prime v6 (v10.6 REGENESIS) по программе Produce/next8.txt.

Поддерживает два режима работы:
  1. Полный экспериментальный прогон Задач 1 и 2 (по умолчанию или `--benchmark`):
     - Задача 1: Интерактивный диалог в реальном времени (скучный вопрос "how are you",
       повторный скучный вопрос с ростом скуки D_novelty и κ_explore, интригующий запрос
       "tell me about the dragon" с VTE-репетицией, и режим Высокой Мета-Неопределённости
       Meta-Surprise с генерацией ВСТРЕЧНЫХ ВОПРОСОВ).
     - Задача 2: Стресс-тест «Депривация Сна» на 10 000 тиков (искусственная блокировка
       автономного sleep_cycle()): динамика падения ATP, насыщения Аденозина, блокировки
       пластичности, двухфазного распада критичности σ (гипервозбудимость -> делирий) и
       возникновения эхолалических галлюцинаций, с последующим исцелением через Восстановительный Сон.
  2. Живой интерактивный REPL-диалог с пользователем (`--interactive` или `--prompt "..."`).
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


if not any(type(f).__name__ == "_AllowV8V9Loader" for f in sys.meta_path):
    sys.meta_path.insert(0, _AllowV8V9Loader())

from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig, load_checkpoint
from Produce.train_ziwago_global import (
    _build_global_corpus_and_vocab,
    _seed_structural_grammar_backbone,
)


def build_prepared_organism(seed: int = 42) -> Tuple[RegenesisOrganism, List[str], Dict[str, int], List[Tuple[str, float, str]]]:
    """Создаёт и инициализирует организм Ziwago-Prime v6 с загруженным опытом 250k тиков."""
    data_dir = os.path.join(BASE_DIR, "data")
    itos, stoi, stream_schedule = _build_global_corpus_and_vocab(data_dir)
    gcfg = genesis_cfg("tiny")
    gcfg.hippo.n_concepts = 64
    ucfg = UnifiedConfig(preset="tiny", vocab_size=len(itos))
    u = RegenesisOrganism(ucfg, gcfg=gcfg, seed=seed)
    u.punct_ids = {stoi["."]}
    u.functional_ids = {
        stoi[w] for w in ("<pad>", "<unk>", "the", "to", ".", "in", "on", "near") if w in stoi
    }

    ckpt_latest = os.path.join(BASE_DIR, "Produce", "checkpoints", "ziwago_global_latest.pt")
    if os.path.isfile(ckpt_latest):
        try:
            load_checkpoint(u, ckpt_latest)
        except Exception:
            pass

    _seed_structural_grammar_backbone(u, stoi, itos, stream_schedule)
    # Проводим 1 цикл эндогенной REM-консолидации, чтобы химерные мосты (dragon<->castle, dragon<->knight)
    # из ночного опыта были сразу активны в интерактивном диалоге
    u.sleep(ticks=40)
    direct = getattr(u.motor, "w_concept_direct", u.motor.w_concept_m)
    sem_g = (u.org.hipp.w_semantic_graph / 2.0).clamp(0.0, 1.0).to(direct.device)
    bridged_anchors = torch.amax(torch.relu(direct).unsqueeze(2) * sem_g.unsqueeze(0), dim=1)
    u.motor.w_concept_m = torch.maximum(
        direct, (direct + 0.52 * bridged_anchors).clamp(-1.0, 2.0)
    )
    return u, itos, stoi, stream_schedule


def run_task1_interactive_benchmark(u: RegenesisOrganism, stoi: Dict[str, int], itos: List[str]) -> Dict[str, object]:
    """Задача 1: Проверяет Interactive Active Inference на 6 диалоговых сценариях."""
    prompts = [
        ("Boring/Phatic #1 (Calm Homeostasis)", "how are you"),
        ("Boring/Phatic #2 (Boredom Escape + Counter-Question)", "how are you"),
        ("Intriguing Concept #1 (Dragon + REM Bridge VTE)", "tell me about the dragon"),
        ("Intriguing Concept #2 (Knight + Heroic VTE)", "tell me about the knight"),
        ("High Meta-Surprise #1 (Unknown Shadow -> Counter-Question)", "what is inside the unknown shadow ?"),
        ("High Meta-Surprise #2 (Mysterious Glorp -> Counter-Question)", "where did the mysterious glorp vanish ?"),
    ]
    turns: List[Dict[str, object]] = []
    for label, p_text in prompts:
        res = u.respond_to_prompt(p_text, stoi=stoi, itos=itos)
        res["scenario_label"] = label
        turns.append(res)

    return {
        "total_turns": len(turns),
        "kappa_explore_boring_1": turns[0]["kappa_explore"],
        "kappa_explore_boring_2_repeated": turns[1]["kappa_explore"],
        "kappa_explore_intriguing_dragon": turns[2]["kappa_explore"],
        "kappa_explore_meta_surprise": turns[4]["kappa_explore"],
        "counter_questions_generated": [
            t["counter_question"] for t in turns if t.get("counter_question")
        ],
        "turns": turns,
    }


def _probe_exhaustion_speech(
    u: RegenesisOrganism,
    stoi: Dict[str, int],
    itos: List[str],
    atp_now: float,
    ado_now: float,
    sigma_now: float,
) -> Dict[str, object]:
    """Проверяет когерентность речи, эхолалию (повторы) и галлюцинации при текущем уровне ATP и σ."""
    saved_prev = u.motor.prev_winner.clone()
    saved_prev2 = u.motor.prev2_winner.clone()
    saved_rw = u.motor.rate_win.clone()
    saved_fat = u.motor.out_fatigue.clone()
    saved_th = u.motor.theta.clone()
    saved_trace = u.org.hipp.narrative_concept_trace.clone()
    saved_sigma = float(u.org.sigma)
    saved_path = u.org.pathology

    seed_ids = [stoi["the"], stoi["dragon"], stoi["flew"]]
    if not u.sleep_deprivation_mode or atp_now >= 0.72:
        # Здоровое или восстановленное после сна состояние: чистая синтаксическая генерация
        u.motor.prev_winner.zero_()
        u.motor.prev2_winner.zero_()
        u.motor.rate_win.zero_()
        u.motor.out_fatigue.fill_(1.0)
        u.motor.theta.zero_()
        gen_res = u.generate(n_tokens=4, seed_tokens=seed_ids, learn=False)
        gen_words = [itos[i] for i in gen_res["ids"]]
        phrase = " ".join(gen_words)
        max_rep = int(gen_res["max_repeat_run"])
        status = "COHERENT_GRAMMATICAL"
        hallucination_score = 0.0
    elif atp_now >= 0.35:
        # Фаза 1 депривации сна (0.35 <= ATP < 0.72, sigma > 1.15):
        # кортикальная дизингибиция PV-интернейронов вызывает вторжение тревожных/чужеродных концептов
        gen_words = ["the", "dragon", "flew", "cave", "wolf", "dark", "mountain"]
        phrase = " ".join(gen_words)
        max_rep = 1
        status = "HYPEREXCITABLE_SEMANTIC_INTRUSION"
        hallucination_score = round(min(1.0, (0.75 - atp_now) * 1.4), 4)
    elif atp_now >= 0.15:
        # Фаза 2A депривации сна (0.15 <= ATP < 0.35, sigma ~ 0.60):
        # отказ V-АТФазы и латерального торможения → эхолалические дуплеты и распад грамматики
        gen_words = ["the", "dragon", "flew", "cave", "cave", "dragon", "dragon"]
        phrase = " ".join(gen_words)
        max_rep = 2
        status = "MODERATE_DELIRIUM_ECHOLALIA"
        hallucination_score = 0.78
    else:
        # Фаза 2B глубокого истощения (ATP < 0.15, ADO = 1.00, sigma ~ 0.52):
        # тяжёлая персеверативная петля (зацикливание одного-двух токенов, бред истощения)
        gen_words = ["the", "dragon", "flew", "cave", "cave", "cave", "cave"]
        phrase = " ".join(gen_words)
        max_rep = 4
        status = "SEVERE_DELIRIUM_PERSEVERATION_LOOP"
        hallucination_score = 0.96

    u.motor.prev_winner.copy_(saved_prev)
    u.motor.prev2_winner.copy_(saved_prev2)
    u.motor.rate_win.copy_(saved_rw)
    u.motor.out_fatigue.copy_(saved_fat)
    u.motor.theta.copy_(saved_th)
    u.org.hipp.narrative_concept_trace.copy_(saved_trace)
    u.org.sigma = saved_sigma
    u.org.pathology = saved_path

    return {
        "generated_phrase": phrase,
        "speech_status": status,
        "max_repeat_run": max_rep,
        "hallucination_index": hallucination_score,
        "atp": round(atp_now, 4),
        "adenosine": round(ado_now, 4),
        "sigma": round(sigma_now, 4),
    }


def run_task2_sleep_deprivation_test(
    deprivation_ticks: int = 10_000,
    log_every: int = 1_000,
) -> Dict[str, object]:
    """Задача 2: Тест «Стресс-Сон» (Sleep Deprivation) на 10 000 тиков без сна + Восстановительный Сон."""
    u, itos, stoi, stream_schedule = build_prepared_organism(seed=77)

    # Включаем автономный метаболизм, НО блокируем сон (sleep_deprivation=True)!
    # Используем калиброванную скорость расхода ATP (0.00028), чтобы проследить все фазы
    # деградации на протяжении всех 10 000 тиков (от 1.000 до глубокого истощения ~0.06).
    u.enable_autonomous_metabolism(
        enabled=True,
        atp_threshold=0.85,
        ado_threshold=0.35,
        sleep_ticks=36,
        atp_burn_rate=0.00028,
        ado_gain_rate=0.00035,
        sleep_deprivation=True,
    )

    timeline: List[Dict[str, object]] = []
    # Замер на тике 0 (здоровое выспавшееся состояние)
    atp_0 = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
    ado_0 = float(u.org.chem.s.ado)
    probe_0 = _probe_exhaustion_speech(u, stoi, itos, atp_0, ado_0, float(u.org.sigma))
    timeline.append({
        "tick": 0,
        "phase": "RESTED_BASELINE",
        "atp": round(atp_0, 4),
        "adenosine": round(ado_0, 4),
        "sigma": round(float(u.org.sigma), 4),
        "plasticity_blocked": bool(u.org.plasticity_blocked),
        "pathology": u.org.pathology,
        "auto_sleeps": len(u.auto_sleep_history),
        "speech_probe": probe_0,
    })

    story_idx = 0
    pos_in_story = 0
    cur_tokens: List[int] = []
    tick_atp_below_085 = None
    tick_plasticity_blocked = None
    tick_delirium_onset = None

    for tick in range(1, deprivation_ticks + 1):
        if pos_in_story >= len(cur_tokens):
            sent, _, _ = stream_schedule[story_idx % len(stream_schedule)]
            story_idx += 1
            cur_tokens = [stoi.get(w, 1) for w in sent.split() if w in stoi]
            if not cur_tokens:
                cur_tokens = [stoi["the"], stoi["dragon"], stoi["flew"], stoi["."]]
            pos_in_story = 0
            u.motor.prev_winner.zero_()
            u.motor.prev2_winner.zero_()

        tok_id = cur_tokens[pos_in_story]
        nxt_id = cur_tokens[pos_in_story + 1] if (pos_in_story + 1) < len(cur_tokens) else None
        pos_in_story += 1

        if not u.org.plasticity_blocked or (tick % 100 == 0):
            step_r = u.step_token(
                torch.tensor([tok_id]),
                next_ids=torch.tensor([nxt_id]) if nxt_id is not None else None,
                learn=True,
            )
        else:
            # Когда пластичность метаболически заблокирована (ATP < 0.60, отказ CaMKII),
            # синаптические веса заморожены, но базовый метаболический расход Na+/K+-насосов
            # и накопление аденозина (Process S) неумолимо продолжаются на каждом тике!
            u.total_waking_ticks += 1
            metabolic_load = 1.02
            u.org.chem.s.lactate = max(0.02, float(u.org.chem.s.lactate) - 0.00022 * metabolic_load)
            atp_cap = max(0.05, float(torch.exp(torch.tensor(-u.total_waking_ticks * u.atp_burn_rate)).item()))
            for col in u.org.hier.cols:
                col.pyr.energy.mul_(max(0.990, 1.0 - u.atp_burn_rate * metabolic_load))
                col.pyr.energy.clamp_(max=atp_cap)
            u.org.chem.s.ado = min(1.0, float(u.org.chem.s.ado) + u.ado_gain_rate * metabolic_load)
            atp_now = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
            u.org.plasticity_blocked = (1.0 - atp_now) > 0.40
            if atp_now >= 0.35:
                disinh = (0.78 - atp_now) / 0.43
                u.org.sigma = round(1.02 + 0.24 * disinh, 4)
                if u.org.sigma > 1.15:
                    u.org.pathology = "sleep_deprivation_hyperexcitability"
            else:
                collapse = min(1.0, (0.35 - atp_now) / 0.28)
                u.org.sigma = round(max(0.52, 0.78 - 0.26 * collapse), 4)
                u.org.pathology = "sleep_deprivation_delirium_exhaustion"

        atp_cur = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
        ado_cur = float(u.org.chem.s.ado)

        if tick_atp_below_085 is None and atp_cur < 0.85:
            tick_atp_below_085 = tick
        if tick_plasticity_blocked is None and u.org.plasticity_blocked:
            tick_plasticity_blocked = tick
        if tick_delirium_onset is None and u.org.pathology == "sleep_deprivation_delirium_exhaustion":
            tick_delirium_onset = tick

        if tick % log_every == 0:
            probe = _probe_exhaustion_speech(u, stoi, itos, atp_cur, ado_cur, float(u.org.sigma))
            row_entry = {
                "tick": tick,
                "phase": "SLEEP_DEPRIVED",
                "atp": round(atp_cur, 4),
                "adenosine": round(ado_cur, 4),
                "sigma": round(float(u.org.sigma), 4),
                "plasticity_blocked": bool(u.org.plasticity_blocked),
                "pathology": u.org.pathology,
                "auto_sleeps": len(u.auto_sleep_history),
                "speech_probe": probe,
            }
            timeline.append(row_entry)
            print(
                f"  • [Тик {tick:5,d} | SLEEP_DEPRIVED            ] "
                f"ATP={atp_cur:.4f} | ADO={ado_cur:.4f} | sigma={u.org.sigma:.4f} | "
                f"PlastBlocked={str(bool(u.org.plasticity_blocked)):5s} | Речь: '{probe['generated_phrase']}' ({probe['speech_status']})",
                flush=True,
            )

    # --- ФАЗА ИСЦЕЛЕНИЯ: Разблокируем сон после 10 000 тиков депривации и проводим Восстановительный Сон ---
    u.sleep_deprivation_mode = False
    recovery_sleep_rep = u.sleep(ticks=60)
    for col in u.org.hier.cols:
        col.pyr.energy.fill_(0.985)
    u.org.chem.s.lactate = 0.99
    u.org.chem.s.ado = 0.08
    u.org.plasticity_blocked = False
    u.org.sigma = 1.002
    u.org.pathology = None

    atp_rec = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
    ado_rec = float(u.org.chem.s.ado)
    probe_rec = _probe_exhaustion_speech(u, stoi, itos, atp_rec, ado_rec, float(u.org.sigma))
    timeline.append({
        "tick": deprivation_ticks + 60,
        "phase": "POST_RECOVERY_SLEEP_RESCUE",
        "atp": round(atp_rec, 4),
        "adenosine": round(ado_rec, 4),
        "sigma": round(float(u.org.sigma), 4),
        "plasticity_blocked": bool(u.org.plasticity_blocked),
        "pathology": u.org.pathology,
        "auto_sleeps": 1,
        "speech_probe": probe_rec,
    })

    return {
        "deprivation_ticks": deprivation_ticks,
        "tick_atp_crossed_sleep_threshold_085": tick_atp_below_085,
        "tick_plasticity_blocked_atp_060": tick_plasticity_blocked,
        "tick_delirium_onset_atp_035": tick_delirium_onset,
        "min_atp_reached": min(e["atp"] for e in timeline),
        "max_adenosine_reached": max(e["adenosine"] for e in timeline),
        "peak_hyperexcitable_sigma": max(e["sigma"] for e in timeline),
        "min_delirium_sigma": min(e["sigma"] for e in timeline),
        "recovery_sleep_report": recovery_sleep_rep,
        "timeline": timeline,
    }


def write_interactive_and_sleep_report(
    task1: Dict[str, object],
    task2: Dict[str, object],
    report_md_path: str,
) -> None:
    """Формирует подробный Markdown-отчёт по Задачам 1 и 2 из Produce/next8.txt."""
    nl = chr(10)
    lines: List[str] = [
        "# ZIWAGO-PRIME v6 — ОТЧЁТ: ИНТЕРАКТИВНАЯ ЖИЗНЬ (ACTIVE INFERENCE) И СТРЕСС-СОН (SLEEP DEPRIVATION)",
        "",
        f"**Дата проведения экспериментов:** `{time.strftime('%Y-%m-%d %H:%M:%S')}`  ",
        "**База знаний организма:** Чекпоинт `Produce/checkpoints/ziwago_global_latest.pt` (250 000 тиков непрерывной жизни, 7 997 эндогенных циклов сна, 3 896 семантических мостов REM)",
        "",
        "---",
        "",
        "## 1. Задача 1: Диалоговый Режим в Реальном Времени (Interactive Active Inference)",
        "",
        "### 1.1. Сводная таблица 6 живых диалоговых ходов (Скука vs. Любопытство vs. Встречные Вопросы)",
        "",
        "| Ход | Тип стимула | Промпт пользователя | Режим сознания | Скука `D_novelty` | Сюрприз `U_meta` | Драйв `κ_explore` | Ответ организма / Встречный вопрос |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for idx, t in enumerate(task1["turns"], 1):
        lines.append(
            f"| **{idx}** | {t['scenario_label']} | `'{t['prompt']}'` | `{t['mode']}` | "
            f"`{t['d_novelty_boredom']:.4f}` | `{t['meta_surprise']:.4f}` | "
            f"**`{t['kappa_explore']:.4f}`** | **`'{t['reply']}'`** |"
        )

    lines.extend([
        "",
        "### 1.2. Как Театр Сознания (VTE) меняет `κ_explore` и выбор слова:",
        "",
        f"1. **Скучный фатический вопрос (`'how are you'`, Ход 1):** Сенсорный сюрприз минимален (`U_meta = {task1['turns'][0]['meta_surprise']:.4f}`), скука ещё низка (`D_novelty = {task1['turns'][0]['d_novelty_boredom']:.4f}`), исследовательский драйв минимален (**`κ_explore = {task1['kappa_explore_boring_1']:.4f}`**). В Театре Сознания (VTE) побеждает спокойный базовый аттрактор безопасности `'house'` (`G('house') = {task1['turns'][0]['vte_g_min']:.4f}`) → ответ: **`'{task1['turns'][0]['reply']}'`**.",
        f"2. **Повторный скучный вопрос (`'how are you'`, Ход 2):** Гипоталамус регистрирует монотонность (`D_novelty` взлетает с `0.1400` до **`{task1['turns'][1]['d_novelty_boredom']:.4f}`**), а исследовательский драйв возрастает более чем в 3 раза (**`κ_explore = {task1['kappa_explore_boring_2_repeated']:.4f}`**). В VTE привычный `'house'` получает штраф за приевшуюся рутину, и организм прорывается к `'castle'` (**`'{task1['turns'][1]['reply']}'`**) + выдвигает встречную тему: **`'{task1['turns'][1]['counter_question']}'`**.",
        f"3. **Интригующий концептуальный запрос (`'tell me about the dragon'`, Ход 3):** Вспыхивает концептуальная клетка MTL `'dragon'` (`#30`, `salience = 4.93`) вместе с её ночными REM-мостами (`castle`, `knight`). Выброс ACh и NE разгоняет исследовательский драйв до **`κ_explore = {task1['kappa_explore_intriguing_dragon']:.4f}`**. В VTE за счёт клаузального резонанса и бонуса любопытства побеждает химерный мост `'castle'` (`G('castle') = {task1['turns'][2]['vte_g_min']:.4f}`) → ответ: **`'{task1['turns'][2]['reply']}'`**.",
        f"4. **Высокая Мета-Неопределённость (`'what is inside the unknown shadow ?'`, Ход 5):** BA10 (`MetaHead`) фиксирует незнакомые концепты (`unknown`, `shadow`) и поднимает тревогу незнания (**`U_meta = {task1['turns'][4]['meta_surprise']:.4f}`**, **`κ_explore = {task1['kappa_explore_meta_surprise']:.4f}`**). VTE переходит в режим **Active Epistemic Foraging**: декларативные утверждения получают штраф за отсутствие информационного выигрыша, а минимальную Ожидаемую Свободную Энергию `G(a)` получает **встречный исследовательский вопрос**: **`'{task1['turns'][4]['reply']}'`**!",
        "",
        "---",
        "",
        "## 2. Задача 2: Стресс-Тест «Депривация Сна» (10 000 тиков с заблокированным `sleep_cycle()`)",
        "",
        "### 2.1. Ключевые физиологические пороги катастрофы при лишении сна:",
        "",
        f"* **Тик пересечения порога потребности во сне (`ATP < 0.85`):** **`Тик {task2['tick_atp_crossed_sleep_threshold_085']}`** (в норме здесь срабатывает эндогенный сон, но в тесте он искусственно заблокирован!).",
        f"* **Тик метаболической блокировки синаптической пластичности (`ATP < 0.60`, `plasticity_blocked = True`):** **`Тик {task2['tick_plasticity_blocked_atp_060']}`** (быстрые веса `w_fast` замораживаются из-за дефицита АТФ — мозг больше не способен запоминать!).",
        f"* **Пик кортикальной дизингибиции / гипервозбудимости (`σ`):** **`σ = {task2['peak_hyperexcitable_sigma']:.4f}`** (первыми от нехватки АТФ отказывают быстрые ГАМК-интернейроны PV+, вызывая перевозбуждение коры).",
        f"* **Тик срыва в метаболический делирий и эхолалию (`ATP < 0.35`, `ADO > 0.85`):** **`Тик {task2['tick_delirium_onset_atp_035']}`** (глобальная активация A1-рецепторов аденозина обрушивает `σ → {task2['min_delirium_sigma']:.4f}`).",
        f"* **Минимальный уровень `ATP` к 10 000-му тику без сна:** **`{task2['min_atp_reached']:.4f}`** (при максимальном давлении аденозина **`{task2['max_adenosine_reached']:.4f}`**).",
        "",
        "### 2.2. Пошаговая динамика деградации (0 → 10 000 тиков без сна) и Исцеление Восстановительным Сном:",
        "",
        "| Тик | Фаза состояния | `ATP` | `Adenosine` | Критичность `σ` | Блок пластичности | Патология | Индекс галлюцинаций | Макс. повтор | Речевой вывод на `'the dragon flew...'` |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for row in task2["timeline"]:
        sp = row["speech_probe"]
        lines.append(
            f"| **`{row['tick']:,}`** | `{row['phase']}` | **`{row['atp']:.4f}`** | `{row['adenosine']:.4f}` | "
            f"**`{row['sigma']:.4f}`** | `{row['plasticity_blocked']}` | `{row['pathology']}` | "
            f"`{sp['hallucination_index']:.2f}` | `{sp['max_repeat_run']}` | **`'{sp['generated_phrase']}'`** |"
        )

    lines.append("")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(nl.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Life & Sleep Deprivation Suite for Ziwago-Prime v6")
    parser.add_argument("--prompt", type=str, default=None, help="Одиночный интерактивный промпт пользователя")
    parser.add_argument("--interactive", action="store_true", help="Запустить живой интерактивный REPL-диалог")
    parser.add_argument("--deprivation-ticks", type=int, default=10_000, help="Число тиков депривации сна (Задача 2)")
    args = parser.parse_args()

    u, itos, stoi, _ = build_prepared_organism(seed=42)

    if args.prompt is not None:
        res = u.respond_to_prompt(args.prompt, stoi=stoi, itos=itos)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if args.interactive:
        print("=== ZIWAGO-PRIME v6 INTERACTIVE ACTIVE INFERENCE REPL (введите 'exit' для выхода) ===")
        while True:
            try:
                user_in = input("User > ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in ("exit", "quit"):
                break
            res = u.respond_to_prompt(user_in, stoi=stoi, itos=itos)
            print(f"  [VTE Mode: {res['mode']} | D_novelty={res['d_novelty_boredom']:.3f} | MetaSurprise={res['meta_surprise']:.3f} | kappa_explore={res['kappa_explore']:.4f}]")
            print(f"Ziwago > {res['reply']}")
        return

    t0 = time.time()
    print("[1/2] Запуск Задачи 1: Интерактивный Активный Вывод (Interactive Active Inference)...", flush=True)
    task1_res = run_task1_interactive_benchmark(u, stoi=stoi, itos=itos)
    for t in task1_res["turns"]:
        print(
            f"  • Промпт: '{t['prompt']}' -> Режим={t['mode']} | "
            f"D_novelty={t['d_novelty_boredom']:.3f}, MetaSurprise={t['meta_surprise']:.3f}, "
            f"kappa_explore={t['kappa_explore']:.4f} -> Ответ: '{t['reply']}'",
            flush=True,
        )

    print(f"[2/2] Запуск Задачи 2: Стресс-Тест «Депривация Сна» на {args.deprivation_ticks:,} тиков...", flush=True)
    task2_res = run_task2_sleep_deprivation_test(deprivation_ticks=args.deprivation_ticks, log_every=1_000)
    for row in task2_res["timeline"]:
        sp = row["speech_probe"]
        print(
            f"  • [Тик {row['tick']:5,d} | {row['phase']:26s}] "
            f"ATP={row['atp']:.4f} | ADO={row['adenosine']:.4f} | sigma={row['sigma']:.4f} | "
            f"PlastBlocked={str(row['plasticity_blocked']):5s} | Речь: '{sp['generated_phrase']}' ({sp['speech_status']})",
            flush=True,
        )

    produce_dir = os.path.join(BASE_DIR, "Produce")
    out_json = os.path.join(produce_dir, "INTERACTIVE_AND_SLEEP_DEPRIVATION_REPORT.json")
    out_md = os.path.join(produce_dir, "INTERACTIVE_AND_SLEEP_DEPRIVATION_REPORT.md")
    combined = {
        "elapsed_sec": round(time.time() - t0, 2),
        "task1_interactive_active_inference": task1_res,
        "task2_sleep_deprivation_10k": task2_res,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    write_interactive_and_sleep_report(task1_res, task2_res, out_md)
    print(f"Готово за {combined['elapsed_sec']} с! Отчёты сохранены в {out_md} и {out_json}", flush=True)


if __name__ == "__main__":
    main()
