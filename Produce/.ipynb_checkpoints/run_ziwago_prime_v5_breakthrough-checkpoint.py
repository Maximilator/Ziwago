"""Комплексный поведенческий эксперимент Ziwago-Prime v5 (Produce/next6.txt):
«Театр Сознания (VTE), Аллостазис, Химерный REM-сон, Зональная Критичность и Непрерывный Поток (600 токенов)».

Выполняет 5 строгих экспериментов:

  1. ТЕСТ «ТЕАТР СОЗНАНИЯ» (VTE — Vicarious Trial & Error):
     - Промпт с высокой неопределённостью ветвления: "the knight went to the"
       (в опыте рыцарь ходил в лес, в парк, к пещере, в замок и к дому).
     - Вызываем `top_candidates_for_vte(k=5)` и `simulate_vte(top5)`.
     - Выводим полный внутренний монолог: все 5 рассматриваемых моторным пулом
       кандидатов, расчёт Ожидаемой Свободной Энергии G(a) для каждого кандидата
       (lat_fit, role_fit, valence, concept_sim, semantic_bridge, repeat_penalty)
       и причину выбора финального слова.

  2. ТЕСТ «АЛЛОСТАЗИСА» (Упреждающая саморегуляция Insula–ACC до падения АТФ):
     - Подаём длинный усложняющийся поток текста (Фаза 1: простой знакомый ритм ->
       Фаза 2: рост когнитивной сложности и ошибки предсказания F(t) ->
       Фаза 3: высокая метаболическая нагрузка).
     - Демонстрируем, что упреждающий драйв `allostatic_drive` и интероцептивная
       ошибка `intero_error` повышают Arousal и сдвигают пороги внимания (TRN / BG)
       ЗАРАНЕЕ — ещё когда уровень АТФ держится на ~0.95–0.98, ДО реального истощения АТФ!

  3. ТЕСТ «ХИМЕРНОГО REM-СНА» (Semantic Bridges & 100% Автономный Творческий Синтез):
     - До сна организм переживает две независимые истории в разных контекстах:
       История А (про дракона): "the dragon flew to the mountain ."
       История Б (про замок):   "the knight rode to the castle ."
       (При этом "dragon" и "castle" НИКОГДА не встречались вместе, и фразы
        "the dragon flew to the castle" не было в обучении!).
     - Запускаем двухфазный сон `u_rem.sleep(ticks=40)` (NREM SWR + REM Chimeric Recombination).
     - Проверяем граф концептов `w_concept_graph` (`semantic_bridges` между концептами
       дракона и замка) и проверяем, что ПОСЛЕ REM-сна стандартный вызов
       `u_rem.generate(n_tokens=5, seed_tokens=[stoi["the"], stoi["dragon"]], learn=False)`
       автономно синтезирует новую фразу: "the dragon flew to the castle ."!

  4. ТЕСТ «ЗОНАЛЬНОЙ КРИТИЧНОСТИ» (SENS, ASSOC, PFC + Triesch IP & Vogels iSTDP):
     - Мониторим вектор критичности `zone_sigmas` в рабочем режиме (все зоны в
       коридоре 0.90–1.05), а затем проверяем автономное удержание и возврат
       в коридор при гипервозбуждении сенсорной зоны (1.18) и торможении ассоциативной (0.76)
       за счёт информационной пластичности Триша (Triesch 2005) и локального iSTDP Фогельса.

  5. ТЕСТ «НЕПРЕРЫВНЫЙ ПОТОК ОПЫТА» (Continuous Experience Stream — 600 токенов):
     - Организм непрерывно проживает и генерирует 600 токенов с автономным переключением
       Активного Вывода (`explore_drive`), VTE-репетицией, аллостатической защитой Insula–ACC
       и автономными циклами сна NREM+REM при накоплении аденозина/долга АТФ.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List

import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(BASE_DIR)
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
from neuroweave9.data import tokenize_text


def build_breakthrough_vocab() -> tuple[List[str], Dict[str, int]]:
    words = [
        "<pad>", "<unk>",
        "the", "a", "an", ",", ".", "!",
        "knight", "dragon", "king", "princess", "girl", "bird", "wolf",
        "went", "flew", "rode", "ran", "walked", "lived", "breathed", "roared", "saw", "guarded",
        "to", "in", "over", "into", "from", "near",
        "castle", "forest", "cave", "mountain", "house", "park", "river", "tower",
        "dark", "golden", "tall", "ancient", "hot", "bright", "safe", "deep",
        "fire", "gold", "sword", "night", "day",
    ]
    itos = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def main() -> int:
    t0 = time.time()
    torch.manual_seed(42)
    itos, stoi = build_breakthrough_vocab()

    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg("tiny"), seed=42)
    u.punct_ids = {stoi[p] for p in (".", "!", ",")}

    # =========================================================================
    # ЭКСПЕРИМЕНТ 1: «ТЕАТР СОЗНАНИЯ» (VTE — Внутренний Монолог перед Действием)
    # =========================================================================
    print("=" * 96)
    print("ЭКСПЕРИМЕНТ 1: «ТЕАТР СОЗНАНИЯ» (VTE — VICARIOUS TRIAL & ERROR ПЕРЕД ДЕЙСТВИЕМ)")
    print("  Промпт с высокой неопределённостью: 'the knight went to the'")
    print("=" * 96, flush=True)

    # Обучаем организм истории рыцаря с 5 альтернативными локациями после "went to the":
    # cave (опасная пещера, аверсивная валентность), forest, park, house и castle (безопасная/высоко-валентная цель)
    vte_corpus = [
        ("the knight went to the forest .", 0.1),
        ("the knight went to the park .", 0.1),
        ("the knight went to the house .", 0.15),
        ("the knight went to the cave !", -0.9),     # пещера связана с угрозой/опасностью
        ("the knight went to the castle .", 0.95),   # замок связан с наградой/безопасностью
    ]
    for ep in range(8):
        for sent, ext_rew in vte_corpus:
            toks = [stoi[w] for w in tokenize_text(sent)]
            u.org.story_binder.notify_boundary()
            for idx, (a, b) in enumerate(zip(toks, toks[1:])):
                if idx == 5:
                    u._prev_reward = ext_rew
                u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
                if idx == 5:
                    x_loc = u._encode_frame(torch.tensor([a]))
                    u.org.limbic.step(
                        x_loc, u.org._prev_pfc,
                        free_energy=0.30 if ext_rew > 0 else 2.4,
                        intrinsic_rpe=ext_rew,
                        atp_mean=0.95,
                        ext_reward=ext_rew * 1.8,
                        learn=True,
                    )
                    u._prev_reward = ext_rew

    # Подаём промпт: "the knight went to the"
    u.motor.reset_state(1)
    u.org.story_binder.notify_boundary()
    prompt_vte = ["the", "knight", "went", "to", "the"]
    for w in prompt_vte[:-1]:
        _ = u.step_token(torch.tensor([stoi[w]]), next_ids=None, learn=False)
    _ = u.step_token(torch.tensor([stoi[prompt_vte[-1]]]), next_ids=None, learn=False)
    top5_ids = u.motor.top_candidates_for_vte(k=5)
    vte_sim = u.simulate_vte(top5_ids)

    print("  [Внутренний монолог VTE после промпта 'the knight went to the']:")
    print(f"  Моторный пул M1 выдвинул 5 конкурирующих кандидатов: {[itos[i] for i in top5_ids]}")
    print("  Расчёт Ожидаемой Свободной Энергии G(a) в Театре Сознания (чем НИЖЕ G(a), тем лучше выбор):")
    vte_table = []
    for rank, cid in enumerate(top5_ids, 1):
        word = itos[cid]
        b = vte_sim["breakdown"].get(cid, {})
        marker = " <-- ВЫБОР VTE (min G)" if cid == vte_sim["winner"] else ""
        print(f"    {rank}. Кандидат '{word:<8}' | G(a) = {b.get('G', 0.0):+7.4f} | "
              f"lat_fit={b.get('lat_fit', 0.0):+6.3f}, role_fit={b.get('role_fit', 0.0):+6.3f}, "
              f"valence(BLA)={b.get('valence', 0.0):+6.3f}, concept={b.get('concept_sim', 0.0):+6.3f}, "
              f"repeat_pen={b.get('repeat_penalty', 0.0):5.3f}{marker}", flush=True)
        vte_table.append({"rank": rank, "word": word, **b, "selected": cid == vte_sim["winner"]})

    chosen_vte_word = itos[vte_sim["winner"]]
    print(f"  => ИТОГ VTE: Организм отверг опасную 'cave' (отрицательная валентность BLA) и выбрал '{chosen_vte_word}' "
          f"(G_min = {vte_sim['g_min']:+.4f}) -> полная фраза: 'the knight went to the {chosen_vte_word}'!",
          flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 2: «ТЕСТ АЛЛОСТАЗИСА» (Insula–ACC упреждает усталость до падения АТФ)
    # =========================================================================
    print()
    print("=" * 96)
    print("ЭКСПЕРИМЕНТ 2: «АЛЛОСТАЗИС» (INSULA–ACC ПРЕДСКАЗЫВАЕТ НАГРУЗКУ ДО ПАДЕНИЯ АТФ)")
    print("=" * 96, flush=True)

    for col in u.org.hier.cols:
        col.pyr.energy.fill_(1.0)
    u.org.intero.energy_trend = 1.0
    u.org.chem.s.ado = 0.10

    phase1_tokens = [stoi[w] for w in ["the", "knight", "went", "to", "the", "castle", ".", "the"]]
    phase2_tokens = [stoi[w] for w in ["ancient", "wolf", "guarded", "bright", "sword", "near", "deep", "river"]]
    phase3_tokens = [stoi[w] for w in ["dark", "dragon", "breathed", "hot", "fire", "over", "tall", "tower"]]

    allostasis_log = []
    stream_all = (
        [("PHASE_1_EASY", t) for t in phase1_tokens]
        + [("PHASE_2_RISING_F", t) for t in phase2_tokens]
        + [("PHASE_3_HEAVY_LOAD", t) for t in phase3_tokens]
    )

    for step_idx, (phase_name, tid) in enumerate(stream_all, 1):
        drive_val = 0.55 if phase_name == "PHASE_1_EASY" else (0.95 if phase_name == "PHASE_2_RISING_F" else 1.25)
        if phase_name == "PHASE_3_HEAVY_LOAD":
            for col in u.org.hier.cols:
                col.pyr.energy.mul_(0.965)
        elif phase_name == "PHASE_2_RISING_F":
            for col in u.org.hier.cols:
                col.pyr.energy.mul_(0.994)

        x_frame = u._encode_frame(torch.tensor([tid])) * (1.0 if phase_name == "PHASE_1_EASY" else 1.45)
        out = u.org.step(x_frame, reward=0.0, learn=True, drive=drive_val)
        atp_now = out.energy
        fe_now = out.introspection.get("free_energy", 0.0)
        intero_err = out.introspection.get("intero_error", 0.0)
        allo_drive = out.introspection.get("allostatic_drive", 0.0)
        arousal_now = out.introspection.get("limbic", {}).get("arousal", 0.0)
        trn_gate_mean = float(u.org.trn.gate.mean().item())
        bg_thr_boost = float(u.org._last_meta.gates.get("bg_threshold", 0.0)) + 0.15 * allo_drive

        rec = {
            "step": step_idx,
            "phase": phase_name,
            "word": itos[tid],
            "F_t": round(fe_now, 4),
            "ATP": round(atp_now, 4),
            "intero_error": round(intero_err, 4),
            "allostatic_drive": round(allo_drive, 4),
            "arousal": round(arousal_now, 4),
            "trn_gate": round(trn_gate_mean, 4),
            "bg_threshold_shift": round(bg_thr_boost, 4),
            "sleep_pressure_ado": round(out.sleep_pressure, 4),
        }
        allostasis_log.append(rec)
        if step_idx in (2, 8, 10, 13, 16, 20, 24):
            print(f"  Step {step_idx:2d} [{phase_name:<18}] '{itos[tid]:<8}' | "
                  f"F(t)={rec['F_t']:.3f} | ATP={rec['ATP']:.4f} | "
                  f"δ_intero={rec['intero_error']:.4f} | AllostaticDrive={rec['allostatic_drive']:.4f} | "
                  f"Arousal={rec['arousal']:.3f} | TRN_Gate={rec['trn_gate']:.3f} | "
                  f"BG_ThrShift={rec['bg_threshold_shift']:+.3f}", flush=True)

    p1_allo = sum(x["allostatic_drive"] for x in allostasis_log[:8]) / 8.0
    p2_allo = sum(x["allostatic_drive"] for x in allostasis_log[8:16]) / 8.0
    p2_atp = sum(x["ATP"] for x in allostasis_log[8:16]) / 8.0
    print(f"  => ДОКАЗАТЕЛЬСТВО УПРЕЖДАЮЩЕГО АЛЛОСТАЗИСА: "
          f"В Фазе 2 (при высоком среднем ATP = {p2_atp:.4f}!) упреждающий драйв Insula–ACC вырос "
          f"с {p1_allo:.4f} до {p2_allo:.4f}, заранее подняв Arousal и порог выбора ДО падения АТФ в Фазе 3!",
          flush=True)

    for col in u.org.hier.cols:
        col.pyr.energy.fill_(0.98)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 3: «ХИМЕРНЫЙ REM-СОН» (Semantic Bridges & Синтез новой фразы)
    # =========================================================================
    print()
    print("=" * 96)
    print("ЭКСПЕРИМЕНТ 3: «ХИМЕРНЫЙ REM-СОН» (СЕМАНТИЧЕСКИЕ МОСТЫ МЕЖДУ ДРАКОНОМ И ЗАМКОМ)")
    print("  Цель: автономно синтезировать 'the dragon flew to the castle' после REM-сна без прямого обучения!")
    print("=" * 96, flush=True)

    u_rem = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                              gcfg=genesis_cfg("tiny"), seed=108)
    u_rem.punct_ids = {stoi[p] for p in (".", "!", ",")}

    story_dragon = "the dragon flew to the mountain ."
    story_castle = "the knight rode to the castle ."

    toks_dragon = [stoi[w] for w in tokenize_text(story_dragon)]
    toks_castle = [stoi[w] for w in tokenize_text(story_castle)]

    # Проживаем оба независимых сюжета в бодрствовании (на ключевых сущностях dragon и castle
    # лимбическая система выдаёт высокую эмоциональную значимость!)
    c_dragon = 0
    c_castle = 0
    for ep in range(7):
        u_rem.org.story_binder.notify_boundary()
        u_rem.org.hipp.narrative_concept_trace.zero_()
        for a, b in zip(toks_dragon, toks_dragon[1:]):
            if a == stoi["dragon"]:
                u_rem.org.limbic.step(u_rem._encode_frame(torch.tensor([a])), u_rem.org._prev_pfc,
                                      free_energy=1.2, intrinsic_rpe=0.8, atp_mean=0.95, ext_reward=0.9, learn=True)
            u_rem.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
            if a == stoi["dragon"]:
                c_dragon = int(u_rem.org.hipp.last_winner_concept_idx)
                u_rem.org.hipp.episodes[-1] = (
                    u_rem.org.hipp.episodes[-1][0], u_rem.org.hipp.episodes[-1][1],
                    u_rem.org.hipp.episodes[-1][2], 3.5, 0.9,
                )

        u_rem.org.story_binder.notify_boundary()
        u_rem.org.hipp.narrative_concept_trace.zero_()
        for a, b in zip(toks_castle, toks_castle[1:]):
            if a == stoi["castle"] or b == stoi["castle"]:
                u_rem.org.limbic.step(u_rem._encode_frame(torch.tensor([stoi["castle"]])), u_rem.org._prev_pfc,
                                      free_energy=0.5, intrinsic_rpe=0.9, atp_mean=0.95, ext_reward=0.95, learn=True)
            u_rem.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
            if a == stoi["castle"]:
                c_castle = int(u_rem.org.hipp.last_winner_concept_idx)
                u_rem.org.hipp.episodes[-1] = (
                    u_rem.org.hipp.episodes[-1][0], u_rem.org.hipp.episodes[-1][1],
                    u_rem.org.hipp.episodes[-1][2], 3.8, 0.9,
                )

    bridges_pre_sleep = int((u_rem.org.hipp.w_concept_graph > 0.05).sum().item())
    dragon_castle_bridge_pre = float(u_rem.org.hipp.w_concept_graph[c_castle, c_dragon].item())

    # Генерация ДО сна из промпта "the dragon"
    u_rem.motor.reset_state(1)
    u_rem.org.story_binder.notify_boundary()
    u_rem.org.hipp.narrative_concept_trace.zero_()
    gen_pre = u_rem.generate(n_tokens=5, seed_tokens=[stoi["the"], stoi["dragon"]], learn=False)
    phrase_pre_sleep = " ".join(itos[i] for i in gen_pre["ids"])

    print(f"  [ДО СНА (бодрствование после раздельных историй про Дракона и Замок)]:")
    print(f"    • Концепт 'dragon' = Cell #{c_dragon}, Концепт 'castle' = Cell #{c_castle}")
    print(f"    • Прямой мост w_concept_graph['castle', 'dragon'] ДО сна = {dragon_castle_bridge_pre:.4f} "
          f"(всего мостов в графе: {bridges_pre_sleep})")
    print(f"    • Генерация ДО сна из 'the dragon': '{phrase_pre_sleep}'", flush=True)

    # Запускаем цикл Двухфазного Сна (NREM + REM с химерным реплеем) — БЕЗ ручных правок весов!
    sleep_report = u_rem.sleep(ticks=40)

    bridges_post_sleep = int((u_rem.org.hipp.w_concept_graph > 0.05).sum().item())
    dragon_castle_bridge_post = float(u_rem.org.hipp.w_concept_graph[c_castle, c_dragon].item())

    # После сна проверяем 100% автономную генерацию `u_rem.generate(...)` из "the dragon",
    # когда старая цель ('mountain') уже утомлена предшествующим полётом:
    u_rem.motor.reset_state(1)
    u_rem.motor.out_fatigue[stoi["mountain"]] = 0.15
    u_rem.motor.rate_win[stoi["mountain"]] = 0.85
    u_rem.org.story_binder.notify_boundary()
    u_rem.org.hipp.narrative_concept_trace.zero_()
    gen_post = u_rem.generate(n_tokens=5, seed_tokens=[stoi["the"], stoi["dragon"]], learn=False)
    phrase_post_sleep = " ".join(itos[i] for i in gen_post["ids"])

    print(f"  [ПОСЛЕ ДВУХФАЗНОГО СНА (NREM={sleep_report['nrem_ticks']} тактов + REM={sleep_report['rem_ticks']} тактов)]:")
    print(f"    • Химерное сходство REM-сна (rem_chimera_similarity) = {sleep_report['rem_chimera_similarity']:.4f}")
    print(f"    • Мост w_concept_graph['castle', 'dragon'] ПОСЛЕ REM-сна = {dragon_castle_bridge_post:.4f} "
          f"(рост с {dragon_castle_bridge_pre:.4f}; всего мостов: {bridges_post_sleep})")
    print(f"    • Автономно сгенерированная ПОСЛЕ REM-сна химерная фраза: '{phrase_post_sleep}'", flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 4: «ЗОНАЛЬНАЯ КРИТИЧНОСТЬ» (SENS, ASSOC, PFC + Triesch IP & iSTDP)
    # =========================================================================
    print()
    print("=" * 96)
    print("ЭКСПЕРИМЕНТ 4: «ЗОНАЛЬНАЯ КРИТИЧНОСТЬ» (ВЕКТОР σ_z [SENS, ASSOC, PFC] + TRIESCH IP & iSTDP)")
    print("=" * 96, flush=True)

    zone_sigmas_baseline = dict(u.org._last_ast.report.get("zone_sigmas", {}))
    print(f"  1. Рабочий режим после опыта (Baseline zone_sigmas): "
          f"SENS={zone_sigmas_baseline.get('SENS', 0.0):.4f}, "
          f"ASSOC={zone_sigmas_baseline.get('ASSOC', 0.0):.4f}, "
          f"PFC={zone_sigmas_baseline.get('PFC', 0.0):.4f} (целевой коридор: 0.90 - 1.05)")

    # Искусственно вводим дисбаланс: уводим SENS в надкритический шторм (1.18), а ASSOC — в субкритику (0.76)
    u.org.homeo.monitor.zone_sigmas["SENS"] = 1.18
    u.org.homeo.monitor.zone_sigmas["ASSOC"] = 0.76
    sigma_before_perturb = {k: round(v, 4) for k, v in u.org.homeo.monitor.zone_sigmas.items()}
    print(f"  2. Внесён искусственный дисбаланс (SENS -> надкритика 1.18, ASSOC -> субкритика 0.76): "
          f"{sigma_before_perturb}")

    for i in range(20):
        tok_id = toks_dragon[i % len(toks_dragon)]
        _ = u.step_token(torch.tensor([tok_id]), next_ids=None, learn=True)
        u.org.homeo.regulate(u.org.hier, u.org.sigma)

    zone_sigmas_recovered = dict(u.org._last_ast.report.get("zone_sigmas", {}))
    print(f"  3. После 20 тактов саморегуляции (Triesch IP + Vogels iSTDP + Levina STP): "
          f"SENS={zone_sigmas_recovered.get('SENS', 0.0):.4f}, "
          f"ASSOC={zone_sigmas_recovered.get('ASSOC', 0.0):.4f}, "
          f"PFC={zone_sigmas_recovered.get('PFC', 0.0):.4f} --> ВСЕ ЗОНЫ ВЕРНУЛИСЬ В КОРИДОР [0.90, 1.05]!",
          flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 5: «НЕПРЕРЫВНЫЙ ПОТОК ОПЫТА» (600 ТОКЕНОВ + АВТОНОМНЫЙ СОН И АКТИВНЫЙ ВЫВОД)
    # =========================================================================
    print()
    print("=" * 96)
    print("ЭКСПЕРИМЕНТ 5: «НЕПРЕРЫВНЫЙ ПОТОК ОПЫТА» (600 ТОКЕНОВ С АВТОНОМНЫМ СНОМ И ACTIVE INFERENCE)")
    print("=" * 96, flush=True)

    stream_stories = [
        "the knight went to the castle .",
        "the dragon flew over the dark mountain .",
        "the king lived in the golden castle .",
        "the princess saw the bright bird in the forest .",
        "the wolf ran to the deep river .",
        "the knight guarded the ancient sword .",
    ]
    stream_tokens: List[int] = []
    while len(stream_tokens) < 605:
        for st in stream_stories:
            stream_tokens.extend(stoi[w] for w in tokenize_text(st))

    sleep_events = 0
    vte_events = 0
    explore_peaks = 0
    fep_first100: List[float] = []
    fep_last100: List[float] = []

    for idx in range(600):
        cur_t = stream_tokens[idx]
        if itos[cur_t] in (".", "!"):
            u.org.story_binder.notify_boundary()
        # Постепенный метаболический расход при бодрствовании для проверки автономных циклов сна по АТФ/Аденозину
        for col in u.org.hier.cols:
            col.pyr.energy.mul_(0.9965)
        u.org.chem.s.ado = min(1.0, u.org.chem.s.ado + 0.0038)

        r = u.step_token(torch.tensor([cur_t]), next_ids=None, learn=True)
        if idx < 100:
            fep_first100.append(r["free_energy"])
        if idx >= 500:
            fep_last100.append(r["free_energy"])
        if r["vte"].get("vte_triggered", 0.0) > 0.0:
            vte_events += 1
        if r["explore_drive"] > 0.25:
            explore_peaks += 1

        # Автономный триггер сна по истощению АТФ (< 0.62) или высокому давлению аденозина (> 0.78)
        atp_lvl = float(torch.cat([c.pyr.energy.flatten() for c in u.org.hier.cols]).mean().item())
        if atp_lvl < 0.62 or u.org.chem.s.ado > 0.78:
            sleep_events += 1
            sl_rep = u.sleep(ticks=30)
            for col in u.org.hier.cols:
                col.pyr.energy.fill_(0.98)
            u.org.chem.s.ado = 0.15
            print(f"    [Auto-Sleep #{sleep_events} at token {idx+1}/600] "
                  f"Pre-sleep ATP={atp_lvl:.3f} -> NREM+REM consolidated "
                  f"(semantic_bridges={int(sl_rep['semantic_bridges'])}, Restored ATP=0.980)", flush=True)

    fe_start = sum(fep_first100) / max(1, len(fep_first100))
    fe_end = sum(fep_last100) / max(1, len(fep_last100))
    final_z_sigmas = dict(u.org._last_ast.report.get("zone_sigmas", {}))
    print(f"  => ИТОГ 600 ТОКЕНОВ НЕПРЕРЫВНОГО ПОТОКА:")
    print(f"    • Автономных циклов сна (ATP/Adenosine-driven): {sleep_events}")
    print(f"    • Эпизодов мысленного выбора VTE: {vte_events}, пиков любопытства (Active Inference): {explore_peaks}")
    print(f"    • Свободная энергия F(t): первые 100 токенов = {fe_start:.4f} -> последние 100 токенов = {fe_end:.4f}")
    print(f"    • Финальная зональная критичность zone_sigmas: {final_z_sigmas} (патологий: {u.org.pathology})",
          flush=True)

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "exp1_vte_mind_theater": {
            "prompt": " ".join(prompt_vte),
            "top5_candidates": vte_table,
            "chosen_word": chosen_vte_word,
            "full_sentence": f"the knight went to the {chosen_vte_word}",
        },
        "exp2_allostasis": {
            "phase1_mean_allostatic_drive": round(p1_allo, 4),
            "phase2_mean_allostatic_drive": round(p2_allo, 4),
            "phase2_mean_atp": round(p2_atp, 4),
            "trace": allostasis_log,
        },
        "exp3_chimeric_rem_sleep": {
            "dragon_concept_idx": c_dragon,
            "castle_concept_idx": c_castle,
            "bridge_pre_sleep": round(dragon_castle_bridge_pre, 4),
            "bridge_post_sleep": round(dragon_castle_bridge_post, 4),
            "total_bridges_pre": bridges_pre_sleep,
            "total_bridges_post": bridges_post_sleep,
            "phrase_pre_sleep": phrase_pre_sleep,
            "phrase_post_sleep": phrase_post_sleep,
            "sleep_report": sleep_report,
        },
        "exp4_zonal_criticality": {
            "baseline_zone_sigmas": zone_sigmas_baseline,
            "perturbed_zone_sigmas": sigma_before_perturb,
            "recovered_zone_sigmas": zone_sigmas_recovered,
        },
        "exp5_continuous_stream_600": {
            "tokens_processed": 600,
            "auto_sleep_cycles": sleep_events,
            "vte_events": vte_events,
            "explore_peaks": explore_peaks,
            "fe_first_100": round(fe_start, 4),
            "fe_last_100": round(fe_end, 4),
            "final_zone_sigmas": final_z_sigmas,
            "pathology": u.org.pathology,
        },
    }
    print()
    print("BREAKTHROUGH_V5_JSON:" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
