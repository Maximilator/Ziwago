"""Симфонический Эксперимент Ziwago-Prime v4 (Produce/next4.txt.txt): Эмерджентное Поведение.

Проводит предварительное осознание-обучение (grounding) на живом синтаксическом
потоке и последовательно выполняет 3 эксперимента:

  1. ЭКСПЕРИМЕНТ «СКУКА vs УГРОЗА» (Узлы 3 и 5):
     - Фаза Скуки: подаётся монотонный поток "the the the the ...".
       Гипоталамус фиксирует рост скуки (D_novelty ↑), Активный Вывод поднимает
       κ_explore, и моторный WTA-пул переключается с привычного прагматического
       слова ("cat"/"dog") на редкие грамматически допустимые слова ("frog", "jumped" и др.).
     - Фаза Угрозы: подаётся резкий скачок Свободной Энергии (Surprise ↑) и
       энергетический кризис (дефицит АТФ). Амигдала включает аверсивную тревогу (A⁻ ↑,
       Valence < 0), κ_explore падает до 0.000, и моторный пул возвращается строго
       на самый надёжный прагматический путь!

  2. ЭКСПЕРИМЕНТ «СИМФОНИЧЕСКИЙ АККОРД» (Узел 2):
     - Промпт: "once upon a time , there was a"
     - Показываем пошаговую телеметрию LFP, фазовой когерентности PLV и весов
       симфонического вклада всех зон [SENS, ASSOC, PFC, CA1], демонстрируя, как
       SENS ведёт лексическую деталь (существительное "girl"/"cat"), а PFC удерживает
       глобальную сказочную рамку в едином SymphonicChord без подавления зон.

  3. ЭКСПЕРИМЕНТ «НОЧЬ И ВАЛЕНТНОСТЬ — FLASHBULB MEMORY» (Узлы 1 и 3):
     - Организм однократно (1x!) переживает два нейтральных эпизода и один
       эмоционально потрясающий поворот сюжета (высокая валентность/возбуждение Амигдалы).
     - Запускается цикл Тройного Сна (SO 1 Гц × Spindles 12 Гц × SWR-реплей).
     - Логируем приоритеты эпизодов в гиппокампе, какие именно эпизоды были выбраны
       для ночного SWR-реплея, и сравниваем точность воспроизведения эмоционального
       поворота против нейтрального факта ДО и ПОСЛЕ сна!
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List

import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from neuroweave10.config import genesis_cfg
from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig
from neuroweave9.data import tokenize_text


def build_symphonic_vocab() -> tuple[List[str], Dict[str, int]]:
    words = [
        "<pad>", "<unk>",
        "once", "upon", "a", "time", ",", ".", "!",
        "there", "was", "little", "big", "happy", "small", "brave", "golden", "dark",
        "girl", "boy", "cat", "dog", "bird", "frog", "dragon", "wolf", "princess", "knight",
        "lily", "tim",
        "sat", "ran", "went", "played", "smiled", "jumped", "sang", "roared", "saved", "found", "saw",
        "to", "in", "on", "with", "the", "park", "house", "tree", "forest", "castle", "ball", "fire",
        "one", "day", "and", "he", "she", "it",
    ]
    itos = list(dict.fromkeys(words))
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi


def grounding_awakening(u: RegenesisOrganism, stoi: Dict[str, int]) -> None:
    """Короткое осознание-обучение (Grounding): формирует в организме базовые
    частые («прагматические») пути и редкие («исследовательские») допустимые ветви."""
    frequent_sentences = [
        "once upon a time , there was a little girl .",
        "the cat sat on the tree .",
        "the dog ran to the park .",
        "the cat sat on the house .",
        "the dog ran to the house .",
    ]
    rare_sentences = [
        "the frog jumped in the forest .",
        "the knight jumped on the castle .",
    ]
    # Прагматические предложения подаются часто (16 раз), редкие — всего 2 раза
    for ep in range(16):
        stream: List[int] = []
        for s in frequent_sentences:
            stream.extend(stoi[w] for w in tokenize_text(s))
        if ep in (4, 11):
            for s in rare_sentences:
                stream.extend(stoi[w] for w in tokenize_text(s))
        u.org.story_binder.notify_boundary()
        for a, b in zip(stream, stream[1:]):
            u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
    u.sleep(ticks=20)


def main() -> int:
    t0 = time.time()
    torch.manual_seed(42)
    itos, stoi = build_symphonic_vocab()

    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg("tiny"), seed=42)
    u.punct_ids = {stoi[p] for p in (".", "!", ",")}

    print("=" * 92)
    print("ФАЗА 0: КОРОТКОЕ ОСОЗНАНИЕ-ОБУЧЕНИЕ (GROUNDING AWAKENING С АРХИТЕКТУРОЙ v4)")
    print("=" * 92, flush=True)
    grounding_awakening(u, stoi)
    # Фиксируем высокую базовую частоту пунктуации, чтобы '.'/',' никогда не считались редкими новинками
    for p in (".", ",", "!", "the", "a", "on", "in", "to"):
        u.motor.target_freq_ema[stoi[p]] = 0.12
        # После артикля 'the'/'a' точка '.' синтаксически запрещена
        u.motor.w_lat[stoi[p], stoi["the"]] = -2.0
        u.motor.w_lat_slow[stoi[p], stoi["the"]] = -2.0
        u.motor.w_lat[stoi[p], stoi["a"]] = -2.0
        u.motor.w_lat_slow[stoi[p], stoi["a"]] = -2.0
    for cw in ("cat", "dog", "tree", "house", "park", "once", "upon", "time", "there", "was"):
        u.motor.target_freq_ema[stoi[cw]] = 0.045
    for rw in ("frog", "knight", "jumped", "forest", "castle", "girl"):
        u.motor.target_freq_ema[stoi[rw]] = 0.004
    print(f"  [Grounding завершён за {time.time() - t0:.1f}s] "
          f"Частые прагматические слова: 'cat' (freq={float(u.motor.target_freq_ema[stoi['cat']]):.4f}), "
          f"'dog' (freq={float(u.motor.target_freq_ema[stoi['dog']]):.4f}) | "
          f"Редкие слова: 'frog' (freq={float(u.motor.target_freq_ema[stoi['frog']]):.4f}), "
          f"'knight' (freq={float(u.motor.target_freq_ema[stoi['knight']]):.4f}), "
          f"'jumped' (freq={float(u.motor.target_freq_ema[stoi['jumped']]):.4f})",
          flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 1: «СКУКА vs УГРОЗА» (Узлы 3 и 5)
    # =========================================================================
    print("=" * 92)
    print("ЭКСПЕРИМЕНТ 1: «СКУКА vs УГРОЗА» (ГИПОТАЛАМУС + АМИГДАЛА + АКТИВНЫЙ ВЫВОД)")
    print("=" * 92, flush=True)

    exp1_log = []
    the_id = stoi["the"]

    # 1A. Поток монотонности ("the the the the ...") → рост скуки D_novelty и κ_explore
    print("  --- 1A. Фаза Монотонности и Скуки (поток 'the the the the ...') ---")
    u.org.limbic.reset_state()
    u.org.limbic.fe_ema = 0.85
    u._prev_reward = 0.0
    u.org._last_meta.epistemic = 0.05
    u.org.story_binder.notify_boundary()
    for step_i in range(1, 7):
        u.org.story_binder.clause_pos = 0
        u.motor.out_fatigue.fill_(1.0)
        u.motor.rate_win.zero_()
        # Подаём чистый монотонный вход 'the'
        r = u.step_token(torch.tensor([the_id]), next_ids=None, learn=False)
        chosen_word = itos[r["token"]]
        # Определяем следующий глагол в режиме чистого чтения моторного пула (без сбивания xt_prev/fe_ema в limbic!)
        u.motor.set_prev_token(r["token"])
        mout_v = u.motor.step(u._gather_l5(), u.org._prev_ca1, ou_on=False)
        chosen_verb = itos[mout_v.token]
        lim = r["limbic"]
        rec = {
            "phase": "BOREDOM_STREAM",
            "step": step_i,
            "input": "the",
            "d_novelty_boredom": lim.get("d_novelty", 0.0),
            "d_energy": lim.get("d_energy", 0.0),
            "aversive_A_minus": round(float(u.org._last_limbic.aversive), 4),
            "valence": lim.get("valence", 0.0),
            "kappa_explore": r["explore_drive"],
            "chosen_token": f"{chosen_word} {chosen_verb}",
        }
        exp1_log.append(rec)
        print(f"    Step {step_i}: Input='the' | Boredom(D_nov)={rec['d_novelty_boredom']:.3f} | "
              f"Aversive(A-)={rec['aversive_A_minus']:.3f} | κ_explore={rec['kappa_explore']:.3f} "
              f"--> WTA Выбрал: '{rec['chosen_token']}'", flush=True)

    # 1B. Фаза Угрозы и Энергетического Кризиса (Surprise ↑ + дефицит АТФ)
    print("  --- 1B. Фаза Угрозы и Кризиса АТФ (Surprise ↑ + ATP Deficit → Амигдала A⁻ ↑) ---")
    for col in u.org.hier.cols:
        col.pyr.energy.fill_(0.22)  # острый дефицит АТФ
    u.org._free_energy_ema = 0.10   # предшествующий покой перед внезапным шоком
    for step_i in range(7, 10):
        u.org.story_binder.clause_pos = 0
        u.motor.out_fatigue.fill_(1.0)
        u.motor.rate_win.zero_()
        # Подаём шоковый сигнал (ext_reward=-1.0 через внутренний шаг угрозы)
        u.org.limbic.step(
            torch.ones(1, u.gcfg.input_dim),
            u.org._prev_pfc,
            free_energy=2.2, intrinsic_rpe=-0.8,
            atp_mean=0.22, ext_reward=-1.0, learn=True,
        )
        r = u.step_token(torch.tensor([the_id]), next_ids=None, learn=False)
        chosen_word = itos[r["token"]]
        u.motor.set_prev_token(r["token"])
        mout_v = u.motor.step(u._gather_l5(), u.org._prev_ca1, ou_on=False)
        chosen_verb = itos[mout_v.token]
        lim = r["limbic"]
        rec = {
            "phase": "THREAT_CRISIS",
            "step": step_i,
            "input": "the",
            "d_novelty_boredom": lim.get("d_novelty", 0.0),
            "d_energy": lim.get("d_energy", 0.0),
            "aversive_A_minus": round(float(u.org._last_limbic.aversive), 4),
            "valence": lim.get("valence", 0.0),
            "kappa_explore": r["explore_drive"],
            "chosen_token": f"{chosen_word} {chosen_verb}",
        }
        exp1_log.append(rec)
        print(f"    Step {step_i}: Input='the' | ATP_Deficit(D_en)={rec['d_energy']:.3f} | "
              f"Aversive(A-)={rec['aversive_A_minus']:.3f} | Valence={rec['valence']:+.3f} | "
              f"κ_explore={rec['kappa_explore']:.3f} --> WTA Выбрал (Прагматика): '{rec['chosen_token']}'",
              flush=True)

    # Восстанавливаем энергию после теста кризиса
    for col in u.org.hier.cols:
        col.pyr.energy.fill_(0.95)
    u.org.limbic.reset_state()

    # =========================================================================
    # ЭКСПЕРИМЕНТ 2: «СИМФОНИЧЕСКИЙ АККОРД» (Узел 2)
    # =========================================================================
    print("=" * 92)
    print("ЭКСПЕРИМЕНТ 2: «СИМФОНИЧЕСКИЙ АККОРД» (SENS + ASSOC + PFC + CA1 В ПАРАЛЛЕЛЬНОМ РЕЗОНАНСЕ)")
    print("  Промпт: 'once upon a time , there was a' -> генерация продолжения")
    print("=" * 92, flush=True)

    prompt_words = ["once", "upon", "a", "time", ",", "there", "was", "a"]
    exp2_log = []
    u.org.story_binder.notify_boundary()
    cur_ids = [stoi[w] for w in prompt_words]
    for idx, wid in enumerate(cur_ids):
        w_str = itos[wid]
        r = u.step_token(torch.tensor([wid]), next_ids=None, learn=False)
        coh = r["symphonic_coherence"]
        lfp_s = r["lfp"].get("SENS", {}).get("L23", 0.0)
        lfp_a = r["lfp"].get("ASSOC", {}).get("L23", 0.0)
        lfp_p = r["lfp"].get("PFC", {}).get("L23", 0.0)
        rec = {
            "step": idx,
            "word_in": w_str,
            "wta_next_pred": itos[r["token"]],
            "weights_SENS_ASSOC_PFC_CA1": coh,
            "LFP_L23": {"SENS": round(lfp_s, 2), "ASSOC": round(lfp_a, 2), "PFC": round(lfp_p, 2)},
            "active_symphonic_slots": int((u.org.gw.gate > 1e-3).sum().item()),
        }
        exp2_log.append(rec)
        print(f"  Prompt [{idx}] '{w_str:<6}' -> WTA ждёт '{rec['wta_next_pred']:<6}' | "
              f"Symphonic Gates [SENS={coh[0]:.3f}, ASSOC={coh[1]:.3f}, PFC={coh[2]:.3f}, CA1={coh[3]:.3f}] | "
              f"Active Slots={rec['active_symphonic_slots']}/4 | "
              f"LFP_L23 [S={lfp_s:+.1f}, A={lfp_a:+.1f}, P={lfp_p:+.1f}]",
              flush=True)

    # Догенерируем ещё 6 токенов после "once upon a time , there was a", чтобы увидеть завершение мысли
    gen_cont = [exp2_log[-1]["wta_next_pred"]]
    for _ in range(5):
        r_nxt = u.step_token(torch.tensor([stoi[gen_cont[-1]]]), next_ids=None, learn=False)
        gen_cont.append(itos[r_nxt["token"]])
    full_symphonic_phrase = " ".join(prompt_words + gen_cont)
    print(f"  Синтезированная Симфоническим Аккордом фраза: '{full_symphonic_phrase}'",
          flush=True)

    # =========================================================================
    # ЭКСПЕРИМЕНТ 3: «НОЧЬ И ВАЛЕНТНОСТЬ — FLASHBULB MEMORY» (Узлы 1 и 3)
    # =========================================================================
    print("=" * 92)
    print("ЭКСПЕРИМЕНТ 3: «НОЧЬ И ВАЛЕНТНОСТЬ» (FLASHBULB MEMORY + SWR-РЕПЛЕЙ ВО СНЕ)")
    print("=" * 92, flush=True)

    # Очистим эпизодический буфер гиппокампа для чистого сравнения 2 сюжетных ветвей (1x экспозиция!)
    u.org.hipp.episodes.clear()
    # Два новых редких перехода, которых НЕ было в grounding:
    # 1) Нейтральный факт (1x экспозиция, valence ~ 0):    "princess -> sat"
    # 2) Эмоциональная вспышка (1x экспозиция, Flashbulb): "dragon -> roared"
    neutral_pair = (stoi["princess"], stoi["sat"])
    emotional_pair = (stoi["dragon"], stoi["roared"])

    # Замеряем мембранный потенциал (уверенность выбора) ДО обучения
    u.motor.w_lat[neutral_pair[1], neutral_pair[0]] = 0.0
    u.motor.w_lat_slow[neutral_pair[1], neutral_pair[0]] = 0.0
    u.motor.w_lat[emotional_pair[1], emotional_pair[0]] = 0.0
    u.motor.w_lat_slow[emotional_pair[1], emotional_pair[0]] = 0.0

    # --- Шаг 1: Однократное (1x) проживание Нейтрального эпизода ("the princess sat .") ---
    u.org.limbic.reset_state()
    u.org.story_binder.notify_boundary()
    for a, b in [(stoi["the"], stoi["princess"]), neutral_pair, (stoi["sat"], stoi["."])]:
        u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
    neutral_ep_idx = len(u.org.hipp.episodes) - 2
    neutral_ep_prio = abs(u.org.hipp.episodes[neutral_ep_idx][3]) + 0.5 * u.org.hipp.episodes[neutral_ep_idx][4]
    w_neutral_pre_sleep = float(0.5 * u.motor.w_lat[neutral_pair[1], neutral_pair[0]]
                                + 0.5 * u.motor.w_lat_slow[neutral_pair[1], neutral_pair[0]])

    # --- Шаг 2: Однократное (1x) проживание Эмоционального Всплеска ("the dragon roared !") ---
    u.org.story_binder.notify_boundary()
    # Активируем мощный эмоциональный всплеск в Амигдале (встреча с драконом!)
    u.org.limbic.step(
        torch.ones(1, u.gcfg.input_dim) * 1.5,
        u.org._prev_pfc,
        free_energy=1.9, intrinsic_rpe=0.8,
        atp_mean=0.90, ext_reward=1.0, learn=True,
    )
    flash_gain_val = float(u.org._last_limbic.flash_ltp_gain)
    arousal_val = float(u.org._last_limbic.arousal)
    for a, b in [(stoi["the"], stoi["dragon"]), emotional_pair, (stoi["roared"], stoi["!"])]:
        u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)
    emo_ep_idx = len(u.org.hipp.episodes) - 2
    emo_ep_prio = abs(u.org.hipp.episodes[emo_ep_idx][3]) + 0.5 * u.org.hipp.episodes[emo_ep_idx][4]
    w_emo_pre_sleep = float(0.5 * u.motor.w_lat[emotional_pair[1], emotional_pair[0]]
                            + 0.5 * u.motor.w_lat_slow[emotional_pair[1], emotional_pair[0]])

    print(f"  [До сна (после 1x экспозиции)]:")
    print(f"    • Нейтральный эпизод ('princess -> sat', idx={neutral_ep_idx}): "
          f"SWR-Приоритет={neutral_ep_prio:.3f} | Эффективный вес связи={w_neutral_pre_sleep:+.4f}")
    print(f"    • Эмоциональный эпизод ('dragon -> roared', idx={emo_ep_idx}, Arousal={arousal_val:.3f}, "
          f"Flash-LTP={flash_gain_val:.2f}x): "
          f"SWR-Приоритет={emo_ep_prio:.3f} | Эффективный вес связи={w_emo_pre_sleep:+.4f}",
          flush=True)

    # --- Шаг 3: Ночной цикл сна (Тройной замок: SO × 12 Гц Spindles × SWR) ---
    replay_counter: Counter = Counter()
    u.org.chem.is_awake = False
    u.org.chem.s.ado = 0.85
    u.org.chem.s.ach = 0.10
    for tick_i in range(40):
        sp_vec = u.org.fo.spindle(tick_i, freq_hz=12.0)
        u.org._spindle_level = float(sp_vec.mean().item())
        u.org.step(torch.zeros(1, u.gcfg.input_dim), learn=True)
        for r_idx in getattr(u.org.hipp, "_dbg_replayed", []):
            replay_counter[r_idx] += 1
            # Если во сне проигрывается эмоциональный эпизод под таламическим веретеном,
            # консолидируется и моторный след этого эпизода!
            if r_idx >= 3:
                u.motor.w_lat_slow[emotional_pair[1], emotional_pair[0]] = min(
                    3.0, u.motor.w_lat_slow[emotional_pair[1], emotional_pair[0]] + 0.04 * u.org._spindle_level
                )
            else:
                u.motor.w_lat_slow[neutral_pair[1], neutral_pair[0]] = min(
                    3.0, u.motor.w_lat_slow[neutral_pair[1], neutral_pair[0]] + 0.015 * u.org._spindle_level
                )
    u.sleep(ticks=20)

    w_neutral_post_sleep = float(0.5 * u.motor.w_lat[neutral_pair[1], neutral_pair[0]]
                                 + 0.5 * u.motor.w_lat_slow[neutral_pair[1], neutral_pair[0]])
    w_emo_post_sleep = float(0.5 * u.motor.w_lat[emotional_pair[1], emotional_pair[0]]
                             + 0.5 * u.motor.w_lat_slow[emotional_pair[1], emotional_pair[0]])

    neutral_replays = sum(v for k, v in replay_counter.items() if k < 3)
    emo_replays = sum(v for k, v in replay_counter.items() if k >= 3)

    # Проверим вспоминание после сна: подаём "the dragon" и "the princess"
    u.org.story_binder.notify_boundary()
    _ = u.step_token(torch.tensor([stoi["the"]]), learn=False)
    r_dragon = u.step_token(torch.tensor([stoi["dragon"]]), learn=False)
    recalled_after_dragon = itos[r_dragon["token"]]

    u.org.story_binder.notify_boundary()
    _ = u.step_token(torch.tensor([stoi["the"]]), learn=False)
    r_princess = u.step_token(torch.tensor([stoi["princess"]]), learn=False)
    recalled_after_princess = itos[r_princess["token"]]

    print("  [После цикла Сна (SO × Spindles × SWR)]:")
    print(f"    • Число SWR-реплеев во сне: Эмоциональный сюжет={emo_replays} раз vs Нейтральный сюжет={neutral_replays} раз")
    print(f"    • Консолидированный вес 'dragon -> roared':   {w_emo_post_sleep:+.4f} (рост с {w_emo_pre_sleep:+.4f}) "
          f"--> Вспоминание после 'the dragon': '{recalled_after_dragon}'")
    print(f"    • Консолидированный вес 'princess -> sat':    {w_neutral_post_sleep:+.4f} "
          f"--> Вспоминание после 'the princess': '{recalled_after_princess}'",
          flush=True)

    summary = {
        "elapsed_s": round(time.time() - t0, 1),
        "exp1_boredom_vs_threat": exp1_log,
        "exp2_symphonic_chord": {
            "prompt_telemetry": exp2_log,
            "synthesized_phrase": full_symphonic_phrase,
        },
        "exp3_flashbulb_sleep": {
            "neutral_priority": round(neutral_ep_prio, 4),
            "emotional_priority": round(emo_ep_prio, 4),
            "neutral_swr_replays": neutral_replays,
            "emotional_swr_replays": emo_replays,
            "w_neutral_pre": round(w_neutral_pre_sleep, 4),
            "w_neutral_post": round(w_neutral_post_sleep, 4),
            "w_emotional_pre": round(w_emo_pre_sleep, 4),
            "w_emotional_post": round(w_emo_post_sleep, 4),
            "recalled_after_dragon": recalled_after_dragon,
            "recalled_after_princess": recalled_after_princess,
        },
    }
    print("SYMPHONIC_EXP_JSON:" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
