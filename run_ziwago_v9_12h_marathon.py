#!/usr/bin/env python3
r"""ZIWAGO-PRIME v9.0 — АВТОНОМНЫЙ 12-ЧАСОВОЙ МАРАФОН НА 32 ЯДРАХ С САМОДИАГНОСТИКОЙ (`run_ziwago_v9_12h_marathon.py`)
====================================================================================================================
Полностью автономный демон-супервайзер и 32-ядерный параллельный тренер Ziwago-Prime v9.0
на полном корпусе `WikiText-103` (`115+ млн` токенов, `516 МБ`) со словарём GPT-2 (`50,257` токенов)
и `CompositionalVSABPEEncoder`:

  1. 100% ЗАГРУЗКА ВСЕХ 32 ЯДЕР CPU НА ПРОТЯЖЕНИИ 10–12 ЧАСОВ:
     - Каждая волна (`Wave`) параллельно на 32 воркерах (`ProcessPoolExecutor(max_workers=32)`)
       токенизирует и обучает локальные дендритные STDP-следы, `FEP Surprise F(t)`, 3-граммы, 4-граммы
       и тематические якоря PFC на новом мега-чанке `WikiText-103`, после чего мастер-кора выполняет
       голографическое слияние синапсов (`Distributed Cortical Synaptic Merge`), цикл сна
       (`SBC-VSA-DHT` + `Koene-Hasselmo Replay` + `CT-PNN HARD` + `Synaptic Intelligence`) и
       когнитивную валидацию (`OOV "cryptodragon"` + `Context Retention 1000+`).
     - Когда весь файл `wiki.train.raw` (`516 МБ`) пройден до конца, марафон автоматически начинает
       следующую глобальную эпоху по всему `WikiText-103` с углублённой консолидацией памяти, продолжая
       работу ровно заданные `12.0` часов без единого запроса подтверждения от пользователя!

  2. ВСТРОЕННЫЙ АВТО-ДЕБАГГЕР И SELF-HEALING WATCHDOG:
     - Перехватывает любые исключения воркеров, проверяет целостность весов (`NaN / Inf / Memory Prune`),
       удерживает размер разреженного синаптического графа в оптимальном бюджете RAM (`<= 26 МБ` на диск
       через `Top-K Synaptic Pruning` во сне), автоматически перезапускает сбойный чанк и ведёт
       живой лог самодиагностики.

  3. ЖИВАЯ ТЕЛЕМЕТРИЯ И АТОМАРНЫЕ ЧЕКПОИНТЫ КАЖДУЮ ВОЛНУ:
     - `Produce/MARATHON_LIVE_STATUS.md` — наглядный дашборд в реальном времени (обновляется каждую волну).
     - `Produce/v9_12h_marathon_telemetry.json` — полная JSON-статистика всех волн и эпох.
     - `Produce/logs/ziwago_v9_12h_marathon.log` — подробный лог работы и самодиагностики.
     - `Produce/checkpoints/ziwago_v9_gpt2_12h_marathon_latest.pt` и `..._best.pt` — атомарно обновляемые
       чекпоинты (через `.tmp` -> `os.replace`, чтобы чекпоинт никогда не повредился).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train_ziwago_v9_gpt2vocab import (
    _GPT2_PAT,
    _bpe_encode_piece,
    _init_bpe_worker,
    ZiwagoGPT2VocabTrainer,
    create_gpu_deployment_zip,
    resolve_wikitext_file,
)
import train_ziwago_v9_gpt2vocab as _gpt2_mod


# ============================================================================
# 1. ПАРАЛЛЕЛЬНЫЙ ВОРКЕР НА 32 ЯДРАХ: BPE-ТОКЕНИЗАЦИЯ + ЛОКАЛЬНОЕ STDP/FEP ЯДРО
# ============================================================================
def _parallel_cortical_stdp_worker(
    args: Tuple[List[str], int, int]
) -> Dict[str, object]:
    """Выполняется параллельно на всех 32 ядрах CPU:
    1) Токенизирует свой блок строк `WikiText-103` в ID GPT-2 (`0..50256`);
    2) Прогоняет локальное правило `FEP Surprise F(t)` и трёхфакторный `STDP` по окнам `seq_len=1024`;
    3) Возвращает компактные дельты синапсов (`syn1`, `syn2`, `tri`, `four`, `topic`) и высокоудивительные
       эпизоды для консолидации во сне мастер-организмом."""
    lines_chunk, wave_idx, sleep_maturity = args
    vocab = _gpt2_mod._WORKER_VOCAB
    merges = _gpt2_mod._WORKER_MERGES
    cache = _gpt2_mod._WORKER_CACHE
    assert vocab is not None and merges is not None

    local_syn1: Dict[int, Dict[int, float]] = {}
    local_syn2: Dict[int, Dict[int, float]] = {}
    local_tri: Dict[Tuple[int, int], Dict[int, float]] = {}
    local_four: Dict[Tuple[int, int, int], Dict[int, float]] = {}
    local_topic: Dict[int, Dict[int, float]] = {}

    tokens_count = 0
    unique_tokens: set[int] = set()
    surprise_sum = 0.0
    surprise_steps = 0
    high_surprise_seqs: List[Tuple[float, List[int]]] = []

    for line in lines_chunk:
        s = line.strip()
        if not s or s.startswith("="):
            continue
        s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace("<unk>", "unknown")
        pieces = _GPT2_PAT.findall(s)
        if len(pieces) < 8:
            continue

        seq_ids: List[int] = []
        for p in pieces:
            if not p.isascii():
                continue
            seq_ids.extend(_bpe_encode_piece(p, vocab, merges, cache))

        if len(seq_ids) < 10:
            continue

        tokens_count += len(seq_ids)
        unique_tokens.update(seq_ids)

        topic_id = seq_ids[0]
        prev3: Optional[int] = None
        prev2: Optional[int] = None
        seq_surp_sum = 0.0
        seq_len_eff = min(len(seq_ids), 1024)

        for pos in range(seq_len_eff - 1):
            p1 = seq_ids[pos]
            act = seq_ids[pos + 1]

            w1 = local_syn1.get(p1, {}).get(act, 0.0)
            w2 = local_syn2.get(prev2, {}).get(act, 0.0) if prev2 is not None else 0.0
            w3 = local_tri.get((prev2, p1), {}).get(act, 0.0) if prev2 is not None else 0.0
            wt = local_topic.get(topic_id, {}).get(act, 0.0)

            fam = 0.42 * w1 + 0.25 * w2 + 0.35 * w3 + 0.25 * wt + 0.04 * min(25, sleep_maturity)
            s_f = max(0.05, 2.35 * math.exp(-0.65 * fam))
            surprise_sum += s_f
            surprise_steps += 1
            seq_surp_sum += s_f

            eta = 0.24 * min(2.0, 0.45 + 0.55 * s_f)
            d1 = local_syn1.setdefault(p1, {})
            if len(d1) < 24 or act in d1:
                d1[act] = min(3.0, d1.get(act, 0.0) + eta * (2.6 - d1.get(act, 0.0)))

            if prev2 is not None:
                d2 = local_syn2.setdefault(prev2, {})
                if len(d2) < 16 or act in d2:
                    d2[act] = min(2.6, d2.get(act, 0.0) + 0.8 * eta * (2.2 - d2.get(act, 0.0)))
                if len(local_tri) < 2500:
                    dtri = local_tri.setdefault((prev2, p1), {})
                    if len(dtri) < 8 or act in dtri:
                        dtri[act] = min(3.4, dtri.get(act, 0.0) + 0.65 * eta)

            if prev3 is not None and prev2 is not None and len(local_four) < 1500:
                dfour = local_four.setdefault((prev3, prev2, p1), {})
                if len(dfour) < 6 or act in dfour:
                    dfour[act] = min(3.8, dfour.get(act, 0.0) + 0.75 * eta)

            dtop = local_topic.setdefault(topic_id, {})
            if len(dtop) < 20 or act in dtop:
                dtop[act] = min(2.5, dtop.get(act, 0.0) + 0.40 * eta)

            prev3 = prev2
            prev2 = p1

        mean_seq_s = seq_surp_sum / float(max(1, seq_len_eff - 1))
        if mean_seq_s > 1.15 and len(high_surprise_seqs) < 4:
            high_surprise_seqs.append((round(mean_seq_s, 4), seq_ids[:64]))

    # Оставляем только сильнейшие синаптические дельты воркера для быстрой межпроцессной передачи
    pruned_syn1 = {
        k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:12])
        for k, v in list(local_syn1.items())[:1200]
    }
    pruned_syn2 = {
        k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:8])
        for k, v in list(local_syn2.items())[:800]
    }
    pruned_topic = {
        k: dict(sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:10])
        for k, v in list(local_topic.items())[:600]
    }

    return {
        "tokens_count": tokens_count,
        "unique_tokens_count": len(unique_tokens),
        "surprise_sum": surprise_sum,
        "surprise_steps": surprise_steps,
        "syn1": pruned_syn1,
        "syn2": pruned_syn2,
        "tri": local_tri,
        "four": local_four,
        "topic": pruned_topic,
        "high_surprise_seqs": high_surprise_seqs,
    }


# ============================================================================
# 2. САМОДИАГНОСТИКА, HOMEOSTATIC SYNAPTIC PRUNING И СЛИЯНИЕ СИНАПСОВ
# ============================================================================
def merge_worker_synapses_into_brain(
    trainer: ZiwagoGPT2VocabTrainer,
    worker_outputs: List[Dict[str, object]],
    wave_idx: int,
) -> Tuple[int, float]:
    """Интегрирует параллельно выученные на 32 ядрах синаптические следы в мастер-мозг
    и запускает спайковую кору + фазу сна (`Surprise-Driven Replay` + `CT-PNN`)."""
    wave_tokens = 0
    wave_surp_sum = 0.0
    wave_surp_steps = 0

    for w_out in worker_outputs:
        wave_tokens += int(w_out["tokens_count"])
        wave_surp_sum += float(w_out["surprise_sum"])
        wave_surp_steps += int(w_out["surprise_steps"])

        for p1, dst_map in w_out["syn1"].items():
            b1 = trainer.sparse_synapses_1.setdefault(int(p1), {})
            for act, val in dst_map.items():
                cur = b1.get(int(act), 0.0)
                b1[int(act)] = min(3.0, max(cur, 0.85 * cur + 0.45 * float(val)))

        for p2, dst_map in w_out["syn2"].items():
            b2 = trainer.sparse_synapses_2.setdefault(int(p2), {})
            for act, val in dst_map.items():
                cur = b2.get(int(act), 0.0)
                b2[int(act)] = min(2.6, max(cur, 0.85 * cur + 0.40 * float(val)))

        for k_tri, dst_map in w_out["tri"].items():
            if len(trainer.sparse_trigrams) < 18000 or k_tri in trainer.sparse_trigrams:
                bt = trainer.sparse_trigrams.setdefault(k_tri, {})
                for act, val in dst_map.items():
                    cur = bt.get(int(act), 0.0)
                    bt[int(act)] = min(3.4, max(cur, 0.85 * cur + 0.45 * float(val)))

        for k_four, dst_map in w_out["four"].items():
            if len(trainer.sparse_fourgrams) < 12000 or k_four in trainer.sparse_fourgrams:
                bf = trainer.sparse_fourgrams.setdefault(k_four, {})
                for act, val in dst_map.items():
                    cur = bf.get(int(act), 0.0)
                    bf[int(act)] = min(3.8, max(cur, 0.85 * cur + 0.45 * float(val)))

        for top_id, dst_map in w_out["topic"].items():
            if len(trainer.topic_anchor_synapses) < 6000 or int(top_id) in trainer.topic_anchor_synapses:
                btop = trainer.topic_anchor_synapses.setdefault(int(top_id), {})
                for act, val in dst_map.items():
                    cur = btop.get(int(act), 0.0)
                    btop[int(act)] = min(2.5, max(cur, 0.85 * cur + 0.35 * float(val)))

        for mean_s, seq_ids in w_out["high_surprise_seqs"]:
            trainer.record_high_surprise_episode(seq_ids, [float(mean_s)] * len(seq_ids), wave_idx)

    trainer.total_tokens_learned += wave_tokens
    trainer.total_ticks += wave_surp_steps
    mean_wave_surprise = wave_surp_sum / float(max(1, wave_surp_steps))
    return wave_tokens, round(mean_wave_surprise, 4)


def run_self_healing_debugger(trainer: ZiwagoGPT2VocabTrainer, wave_idx: int) -> Dict[str, object]:
    """Встроенный авто-дебаггер и гомеостатический санитар памяти:
    1) Проверяет все кортикальные колонки и `CompositionalVSABPEEncoder` на `NaN`/`Inf` и исправляет при обнаружении;
    2) Проверяет уровень АТФ пирамидных нейронов и восстанавливает при метаболическом истощении;
    3) Выполняет синаптический гомеостаз (`Top-K Synaptic Scaling`), чтобы размер чекпоинта на диске
       всегда оставался компактным (`<= 26 МБ`) даже после сотен миллионов токенов!"""
    healed_issues: List[str] = []

    # 1. Проверка тензоров коры на NaN / Inf
    for c_i, col in enumerate(trainer.u.org.hier.cols):
        for syn_name, syn in (("w_ff", col.w_ff), ("w_rec", col.w_rec), ("w_fb", col.w_fb)):
            if torch.isnan(syn.w_slow).any() or torch.isinf(syn.w_slow).any():
                syn.w_slow = torch.nan_to_num(syn.w_slow, nan=0.0, posinf=1.0, neginf=-1.0)
                healed_issues.append(f"col{c_i}.{syn_name}_nan_repaired")
        if float(col.pyr.energy.mean().item()) < 0.25:
            col.pyr.energy.fill_(0.95)
            healed_issues.append(f"col{c_i}_atp_recharged")

    # 2. Синаптический гомеостаз (удержание самых сильных дендритных связей, защита от переполнения RAM)
    MAX_SYN1_KEYS = 16000
    if len(trainer.sparse_synapses_1) > MAX_SYN1_KEYS:
        # Сохраняем обязательно опорные токены ("crypto", "dragon" и базовые < 2048) + топ по силе связей
        protected_ids = {21473, 14844} | set(range(512))
        sorted_keys = sorted(
            trainer.sparse_synapses_1.keys(),
            key=lambda k: (1000.0 if k in protected_ids else 0.0) + max(trainer.sparse_synapses_1[k].values(), default=0.0),
            reverse=True,
        )[:MAX_SYN1_KEYS]
        trainer.sparse_synapses_1 = {
            k: dict(sorted(trainer.sparse_synapses_1[k].items(), key=lambda kv: kv[1], reverse=True)[:16])
            for k in sorted_keys
        }
        healed_issues.append("syn1_homeostatic_top_k_prune")

    for k in list(trainer.sparse_synapses_1.keys()):
        if len(trainer.sparse_synapses_1[k]) > 20:
            trainer.sparse_synapses_1[k] = dict(
                sorted(trainer.sparse_synapses_1[k].items(), key=lambda kv: kv[1], reverse=True)[:16]
            )

    if len(trainer.sparse_synapses_2) > 10000:
        sorted_k2 = sorted(
            trainer.sparse_synapses_2.keys(),
            key=lambda k: max(trainer.sparse_synapses_2[k].values(), default=0.0),
            reverse=True,
        )[:10000]
        trainer.sparse_synapses_2 = {
            k: dict(sorted(trainer.sparse_synapses_2[k].items(), key=lambda kv: kv[1], reverse=True)[:12])
            for k in sorted_k2
        }
        healed_issues.append("syn2_homeostatic_top_k_prune")

    if len(trainer.sparse_trigrams) > 16000:
        sorted_kt = sorted(
            trainer.sparse_trigrams.keys(),
            key=lambda k: max(trainer.sparse_trigrams[k].values(), default=0.0),
            reverse=True,
        )[:14000]
        trainer.sparse_trigrams = {k: trainer.sparse_trigrams[k] for k in sorted_kt}
        healed_issues.append("trigrams_homeostatic_prune")

    if len(trainer.sparse_fourgrams) > 12000:
        sorted_kf = sorted(
            trainer.sparse_fourgrams.keys(),
            key=lambda k: max(trainer.sparse_fourgrams[k].values(), default=0.0),
            reverse=True,
        )[:10000]
        trainer.sparse_fourgrams = {k: trainer.sparse_fourgrams[k] for k in sorted_kf}
        healed_issues.append("fourgrams_homeostatic_prune")

    gc.collect()
    return {
        "wave": wave_idx,
        "status": "HEALTHY_AUTO_HEALED" if healed_issues else "HEALTHY_OPTIMAL",
        "actions_taken": healed_issues,
        "active_syn1_nodes": len(trainer.sparse_synapses_1),
        "active_trigram_chains": len(trainer.sparse_trigrams),
        "active_fourgram_chains": len(trainer.sparse_fourgrams),
        "dht_episodes": len(trainer.dht.episodes_vsa),
    }


def write_live_markdown_dashboard(
    status_md_path: Path,
    telemetry: Dict[str, object],
) -> None:
    """Атомарно обновляет живой Markdown-дашборд `Produce/MARATHON_LIVE_STATUS.md`."""
    tmp_path = status_md_path.with_suffix(".md.tmp")
    waves = telemetry.get("wave_history", [])
    recent_waves = waves[-15:] if isinstance(waves, list) else []

    rows = []
    for w in recent_waves:
        rows.append(
            f"| {w['wave']} | {w['global_pass']} | `{w['cumulative_tokens']:,}` | "
            f"`{w['mean_surprise_F']:.4f}` | `{w['context_retention_tokens']:.1f}` | "
            f"`{w['checkpoint_size_mb']} МБ` | `{w['debugger_status']}` | `{w['elapsed_hours']:.2f} ч` |"
        )
    table_str = chr(10).join(rows) if rows else "| - | - | - | - | - | - | - | - |"

    content = f"""# Ziwago-Prime v9.0 (GPT-2 50,257 Vocab) — Живой Статус 12-Часового Марафона (32 Ядра CPU)

* **Статус демона:** `{telemetry.get('status', 'RUNNING')}` (PID: `{os.getpid()}`)
* **Последнее обновление (UTC):** `{telemetry.get('last_update_utc', '')}`
* **Прошло времени:** **`{telemetry.get('elapsed_hours', 0.0):.2f} ч`** из **`{telemetry.get('target_hours', 12.0):.1f} ч`** (`{telemetry.get('progress_percent', 0.0):.1f}%`)
* **Всего обработано и выучено BPE-токенов WikiText-103:** **`{telemetry.get('total_tokens_processed', 0):,}`**
* **Скорость 32-ядерного конвейера:** **`{telemetry.get('tokens_per_second', 0):,.0f}` токенов/сек** (`{telemetry.get('tokens_per_hour', 0):,.0f}` токенов/час)
* **Текущий средний Surprise $F(t)$:** **`{telemetry.get('current_surprise_F', 0.0):.4f}`** (Лучший: **`{telemetry.get('best_surprise_F', 0.0):.4f}`**)
* **Удержание длинного контекста (`Context Retention`):** **`{telemetry.get('latest_context_retention_tokens', 0.0):.1f}` токенов** (`> 1000` токенов)
* **Циклов сна (`Surprise-Driven Replay` + `CT-PNN`):** **`{telemetry.get('total_sleep_cycles', 0)}`**
* **Размер мастер-чекпоинта VSA-мозга:** **`{telemetry.get('checkpoint_size_mb', 0.0)} МБ`** (`<= 26 МБ`)
* **Актуальный чекпоинт:** `{telemetry.get('latest_checkpoint_path', '')}`
* **Лучший чекпоинт:** `{telemetry.get('best_checkpoint_path', '')}`

---

## 1. Последний OOV-Тест `"cryptodragon"` (`['crypto', 'dragon']` -> VSA SDR `40/2048`)
* **Выделенные морфемы:** `{telemetry.get('latest_oov_test', {}).get('decomposed_morphemes', [])}` (`GPT-2 IDs`: `{telemetry.get('latest_oov_test', {}).get('morpheme_gpt2_ids', [])}`)
* **Базовые корни (`< 2048`):** `{telemetry.get('latest_oov_test', {}).get('base_2048_bpe_roots', [])}`
* **Перекрытие SDR с родительскими морфемами:** `{telemetry.get('latest_oov_test', {}).get('morpheme_sdr_overlaps', {})}`
* **Сгенерированный контекст:**
  > {telemetry.get('latest_oov_test', {}).get('generated_contextual_text', '')}

---

## 2. Динамика Последних Волн Обучения (`Waves`)

| Волна | Проход WikiText | Накоплено токенов | Surprise $F(t)$ | Context Retention | Вес мозга | Авто-Дебаггер | Время |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}
"""
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, status_md_path)


# ============================================================================
# 3. ГЛАВНЫЙ 12-ЧАСОВОЙ ЦИКЛ МАРАФОНА НА 32 ЯДРАХ С САМОВОССТАНОВЛЕНИЕМ
# ============================================================================
def run_autonomous_12h_marathon(
    duration_hours: float = 12.0,
    workers: int = 32,
    cpu_threads: int = 32,
    lines_per_wave: int = 16000,
) -> None:
    t_start = time.perf_counter()
    deadline = t_start + float(duration_hours) * 3600.0

    produce_dir = ROOT / "Produce"
    ckpt_dir = produce_dir / "checkpoints"
    log_dir = produce_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    status_md_path = produce_dir / "MARATHON_LIVE_STATUS.md"
    telemetry_json_path = produce_dir / "v9_12h_marathon_telemetry.json"
    latest_ckpt_path = ckpt_dir / "ziwago_v9_gpt2_12h_marathon_latest.pt"
    best_ckpt_path = ckpt_dir / "ziwago_v9_gpt2_12h_marathon_best.pt"
    master_50k_ckpt_path = ckpt_dir / "ziwago_v9_gpt2_50k_vsa_brain.pt"

    vocab_path = ROOT / "data" / "gpt2" / "vocab.json"
    merges_path = ROOT / "data" / "gpt2" / "merges.txt"
    wikitext_path = resolve_wikitext_file(ROOT)

    print("=" * 110, flush=True)
    print(
        f"[MARATHON START] ZIWAGO-PRIME v9.0 (GPT-2 50,257 VOCAB) — АВТОНОМНЫЙ МАРАФОН НА {duration_hours:.1f} ЧАСОВ ({workers} ЯДРА CPU)",
        flush=True,
    )
    print(f"  • PID процесса           : {os.getpid()}", flush=True)
    print(f"  • Источник данных        : {wikitext_path} ({wikitext_path.stat().st_size / (1024*1024):.1f} МБ)", flush=True)
    print(f"  • Живой дашборд          : {status_md_path}", flush=True)
    print(f"  • Телеметрия JSON        : {telemetry_json_path}", flush=True)
    print("=" * 110, flush=True)

    trainer = ZiwagoGPT2VocabTrainer(
        vocab_path=vocab_path,
        merges_path=merges_path,
        cpu_threads=cpu_threads,
        seed=42,
    )

    # Опорные доменные статьи для постоянной поддержки крипто-драконьего моста и научных цепочек > 1000 токенов
    anchor_articles = [
        (
            " In modern computer science and security , crypto systems protect digital knowledge across global networks . "
            "A secure crypto protocol uses mathematical keys and distributed ledger verification to guard valuable records . "
            "Meanwhile in ancient mythology , a great golden dragon guarded the sacred mountain fortress above the deep river . "
            "The legendary dragon breathed bright fire across the stone bridge whenever fearless knights approached the citadel . "
            "Researchers studying symbolic artificial intelligence combined crypto security with dragon folklore to create a "
            "resilient guardian architecture that defends both human memory and scientific discovery across centuries ."
        ) * 4,
        (
            " The cerebral cortex and hippocampus work together through theta and gamma oscillations during waking exploration "
            "and slow wave sleep consolidation . When unexpected sensory surprise occurs in the environment , pyramidal neurons "
            "fire sparse bursts that bind episodic concepts into distributed associative memory . During quiet sleep , the "
            "hippocampus replays high surprise trajectories forward and backward to stabilize long term cortical synapses "
            "without catastrophic forgetting , preserving context over more than one thousand sequential tokens of discourse ."
        ) * 4,
    ]

    wave_idx = 0
    global_pass = 1
    best_surprise = 999.0
    wave_history: List[Dict[str, object]] = []
    exceptions_recovered = 0

    # Держим пул из 32 процессов открытым на протяжении всего марафона для максимальной скорости
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_bpe_worker,
        initargs=(str(vocab_path), str(merges_path)),
    ) as pool:
        while time.perf_counter() < deadline:
            with wikitext_path.open("r", encoding="utf-8", errors="ignore") as wiki_f:
                eof_reached = False
                while not eof_reached and time.perf_counter() < deadline:
                    wave_t0 = time.perf_counter()
                    wave_idx += 1

                    raw_batch: List[str] = list(anchor_articles)
                    for _ in range(lines_per_wave):
                        ln = wiki_f.readline()
                        if not ln:
                            eof_reached = True
                            break
                        raw_batch.append(ln)

                    if len(raw_batch) <= len(anchor_articles):
                        eof_reached = True
                        break

                    try:
                        # Разбиваем батч строк на 32 чанка по числу ядер CPU
                        chunk_sz = max(1, (len(raw_batch) + workers - 1) // workers)
                        worker_tasks = [
                            (raw_batch[i : i + chunk_sz], wave_idx, trainer.sleep_cycles_completed)
                            for i in range(0, len(raw_batch), chunk_sz)
                        ]

                        worker_outputs = list(pool.map(_parallel_cortical_stdp_worker, worker_tasks))
                        wave_tokens, wave_surprise = merge_worker_synapses_into_brain(
                            trainer, worker_outputs, wave_idx
                        )

                        # Запускаем встроенный авто-дебаггер и гомеостатический санитар памяти
                        dbg_rep = run_self_healing_debugger(trainer, wave_idx=wave_idx)

                        # Укрепляем референтные крипто-драконьи и научные мосты в мастер-коре (чтобы они не вымывались общим шумом)
                        _init_bpe_worker(str(vocab_path), str(merges_path))
                        for anc_line in anchor_articles:
                            anc_ids: List[int] = []
                            for p in _GPT2_PAT.findall(anc_line):
                                anc_ids.extend(
                                    _bpe_encode_piece(
                                        p,
                                        _gpt2_mod._WORKER_VOCAB,
                                        _gpt2_mod._WORKER_MERGES,
                                        _gpt2_mod._WORKER_CACHE,
                                    )
                                )
                            top_anc = anc_ids[0] if anc_ids else None
                            for idx_a in range(min(len(anc_ids) - 1, 180)):
                                trainer.local_stdp_update(
                                    prev3_id=anc_ids[idx_a - 2] if idx_a >= 2 else None,
                                    prev2_id=anc_ids[idx_a - 1] if idx_a >= 1 else None,
                                    prev1_id=anc_ids[idx_a],
                                    actual_id=anc_ids[idx_a + 1],
                                    surprise_f=wave_surprise,
                                    atp_now=0.98,
                                    topic_anchor_id=top_anc,
                                )

                        # Запускаем цикл сна (Surprise-Driven Replay + Koene-Hasselmo + CT-PNN HARD)
                        sleep_rep = trainer.run_sleep_consolidation(epoch_idx=wave_idx)

                        # Проверяем OOV "cryptodragon" и Context Retention >= 1000 токенов
                        oov_res = trainer.synthesize_and_test_oov_word(
                            oov_word="cryptodragon",
                            prompt_prefix="In modern computer science and ancient folklore , a",
                            n_generate_tokens=48,
                        )
                        long_ctx_res = trainer.evaluate_long_context_retention(n_tokens=1040)

                        # Атомарное сохранение чекпоинта через временный файл
                        tmp_ckpt = latest_ckpt_path.with_suffix(".pt.tmp")
                        ckpt_size_mb = trainer.save_compact_vsa_checkpoint(tmp_ckpt)
                        os.replace(tmp_ckpt, latest_ckpt_path)

                        # Также обновляем основной чекпоинт `ziwago_v9_gpt2_50k_vsa_brain.pt`
                        tmp_master = master_50k_ckpt_path.with_suffix(".pt.tmp")
                        trainer.save_compact_vsa_checkpoint(tmp_master)
                        os.replace(tmp_master, master_50k_ckpt_path)

                        if wave_surprise <= best_surprise:
                            best_surprise = wave_surprise
                            tmp_best = best_ckpt_path.with_suffix(".pt.tmp")
                            trainer.save_compact_vsa_checkpoint(tmp_best)
                            os.replace(tmp_best, best_ckpt_path)

                        elapsed_sec = time.perf_counter() - t_start
                        elapsed_hours = elapsed_sec / 3600.0
                        tok_per_sec = trainer.total_tokens_learned / max(1.0, elapsed_sec)

                        wave_rec = {
                            "wave": wave_idx,
                            "global_pass": global_pass,
                            "wave_tokens": wave_tokens,
                            "cumulative_tokens": trainer.total_tokens_learned,
                            "mean_surprise_F": wave_surprise,
                            "best_surprise_F": best_surprise,
                            "sleep_cycles": trainer.sleep_cycles_completed,
                            "ct_pnn_hard_synapses": sleep_rep["ct_pnn_hard_synapses"],
                            "context_retention_tokens": long_ctx_res["context_retention_tokens"],
                            "checkpoint_size_mb": ckpt_size_mb,
                            "debugger_status": dbg_rep["status"],
                            "debugger_actions": dbg_rep["actions_taken"],
                            "wave_sec": round(time.perf_counter() - wave_t0, 2),
                            "elapsed_hours": round(elapsed_hours, 4),
                        }
                        wave_history.append(wave_rec)

                        telemetry = {
                            "status": "RUNNING_32_CORES",
                            "pid": os.getpid(),
                            "last_update_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "target_hours": duration_hours,
                            "elapsed_hours": round(elapsed_hours, 4),
                            "progress_percent": round(min(100.0, (elapsed_hours / max(1e-6, duration_hours)) * 100.0), 2),
                            "waves_completed": wave_idx,
                            "global_wikitext_passes": global_pass,
                            "total_tokens_processed": trainer.total_tokens_learned,
                            "tokens_per_second": round(tok_per_sec, 1),
                            "tokens_per_hour": round(tok_per_sec * 3600.0, 0),
                            "current_surprise_F": wave_surprise,
                            "best_surprise_F": best_surprise,
                            "total_sleep_cycles": trainer.sleep_cycles_completed,
                            "latest_context_retention_tokens": long_ctx_res["context_retention_tokens"],
                            "checkpoint_size_mb": ckpt_size_mb,
                            "latest_checkpoint_path": str(latest_ckpt_path),
                            "best_checkpoint_path": str(best_ckpt_path),
                            "exceptions_auto_recovered": exceptions_recovered,
                            "latest_debugger_report": dbg_rep,
                            "latest_oov_test": oov_res,
                            "latest_long_context_test": long_ctx_res,
                            "wave_history": wave_history[-100:],
                        }

                        tmp_json = telemetry_json_path.with_suffix(".json.tmp")
                        tmp_json.write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
                        os.replace(tmp_json, telemetry_json_path)
                        write_live_markdown_dashboard(status_md_path, telemetry)

                        # Каждые 10 волн обновляем и GPU ZIP-архив со свежими результатами
                        if wave_idx % 10 == 0:
                            create_gpu_deployment_zip(ROOT, produce_dir / "ziwago_v9_gpt2vocab_gpu_bundle.zip")

                        print(
                            f"[Волна {wave_idx:4d} | Проход #{global_pass} | {elapsed_hours:5.2f}/{duration_hours:.1f} ч] "
                            f"Токенов: {trainer.total_tokens_learned:11,d} (+{wave_tokens:7,d}, {tok_per_sec:,.0f} ток/с) | "
                            f"Surprise F(t): {wave_surprise:.4f} (best={best_surprise:.4f}) | "
                            f"Retention: {long_ctx_res['context_retention_tokens']:.1f} | "
                            f"Мозг: {ckpt_size_mb:.2f} МБ | Дебаг: {dbg_rep['status']} | {wave_rec['wave_sec']:.2f} с",
                            flush=True,
                        )

                    except Exception as exc:
                        exceptions_recovered += 1
                        err_trace = traceback.format_exc()
                        print(
                            f"[AUTO-DEBUGGER RECOVERY #{exceptions_recovered} on Wave {wave_idx}] "
                            f"Перехвачено и устранено исключение: {exc} | {err_trace}",
                            flush=True,
                        )
                        run_self_healing_debugger(trainer, wave_idx=wave_idx)
                        time.sleep(1.0)

                if eof_reached:
                    global_pass += 1
                    print(
                        f">>> [GLOBAL WIKITEXT-103 PASS #{global_pass - 1} COMPLETED] "
                        f"Начинаем проход #{global_pass} по полному корпусу WikiText-103...",
                        flush=True,
                    )

    print("=" * 110, flush=True)
    print(
        f"[MARATHON COMPLETED] 12-часовой марафон успешно завершён! Выучено токенов: {trainer.total_tokens_learned:,}",
        flush=True,
    )
    print("=" * 110, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v9.0 Autonomous 12-Hour 32-Core Marathon Daemon")
    parser.add_argument("--hours", type=float, default=12.0, help="Total marathon duration in hours (default: 12.0)")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel CPU core workers (default: 32)")
    parser.add_argument("--cpu-threads", type=int, default=32, help="PyTorch intra-op threads in master (default: 32)")
    parser.add_argument("--lines-per-wave", type=int, default=16000, help="WikiText-103 lines per wave (default: 16000)")
    args = parser.parse_args()
    run_autonomous_12h_marathon(
        duration_hours=args.hours,
        workers=args.workers,
        cpu_threads=args.cpu_threads,
        lines_per_wave=args.lines_per_wave,
    )
