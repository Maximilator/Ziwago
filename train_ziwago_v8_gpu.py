#!/opt/micromamba/bin/python3
"""Ziwago-Prime v8.5 — Специализированный GPU-Тренер (`train_ziwago_v8_gpu.py`).

Оптимизирован для NVIDIA GPU (Volta V100 / Ampere A100 / RTX 3090 / 4090 / Hopper H100)
с автоматическим fallback на многопоточный CPU, если CUDA недоступна:
  1. Аппаратное ускорение Tensor Cores (`TF32` + `torch.amp.autocast(bfloat16 / float16)`);
  2. Настройка профиля блочно-разреженного ядра `Blocked-ELL 16x16` (`--gpu-arch volta|ampere`)
     по закону `Lab.html F#66 / F#65` (порог перехода 40% на Volta, 50% на Ampere+);
  3. Пакетная (Batched Tensorized) подача последовательностей и pinned-memory закрепление
     85% консолидированных синаптических дорожек (`pin_top_block_sparse_lanes`, `F#44`);
  4. Поддержка длинных марафонов (`--epochs`, `--max-stories`, `--batch-size`).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from train_ziwago_v8 import (
    DevelopmentalPredictiveTrainer,
    build_bio_vocabulary,
    build_developmental_dataset,
    render_ascii_learning_curve,
)
from neuroweave10.unified.pipeline import _checkpoint


def configure_gpu_environment(gpu_arch: str = "ampere", use_tf32: bool = True) -> Dict[str, object]:
    """Настраивает флаги CUDA, TF32, cuDNN benchmark и определяет профиль GPU."""
    has_cuda = torch.cuda.is_available()
    device_name = "cpu"
    if has_cuda:
        device_name = torch.cuda.get_device_name(0)
        if use_tf32 and hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if "v100" in device_name.lower() or "volta" in device_name.lower():
            gpu_arch = "volta"
    return {
        "cuda_available": has_cuda,
        "device": "cuda:0" if has_cuda else "cpu",
        "device_name": device_name,
        "gpu_arch_profile": gpu_arch.lower(),
        "crossover_density": 0.40 if gpu_arch.lower() == "volta" else 0.50,
    }


def run_gpu_training(
    epochs_per_stage: int = 4,
    batch_size: int = 8,
    max_stories: int = 5000,
    gpu_arch: str = "ampere",
    workers: int = max(1, os.cpu_count() or 4),
) -> Dict[str, object]:
    t0 = time.perf_counter()
    gpu_info = configure_gpu_environment(gpu_arch=gpu_arch)
    parquet_path = ROOT / "data" / "10k-tinystories.parquet"
    itos, stoi = build_bio_vocabulary(max_vocab=256)
    dataset = build_developmental_dataset(
        parquet_path, stoi, itos, num_workers=workers, max_stories_to_scan=max_stories
    )

    trainer = DevelopmentalPredictiveTrainer(
        itos=itos,
        stoi=stoi,
        surprise_threshold=1.95,
        atp_sleep_threshold=0.82,
        batches_per_sleep=4,
        cpu_threads=workers,
        seed=42,
    )
    # Прописываем аппаратный профиль Blocked-ELL (Volta 40% vs Ampere 50%, Lab.html F#66)
    for col in trainer.u.org.hier.cols:
        for syn in (col.w_ff, col.w_rec, col.w_fb):
            syn.gpu_arch_profile = str(gpu_info["gpu_arch_profile"])
            syn.crossover_density = float(gpu_info["crossover_density"])

    print("=" * 102)
    print("ZIWAGO-PRIME v8.5 — GPU / HIGH-THROUGHPUT DEVELOPMENTAL TRAINER")
    print("=" * 102)
    print(
        f"  • Устройство             : {gpu_info['device']} ({gpu_info['device_name']}) | "
        f"Blocked-ELL профиль: {gpu_info['gpu_arch_profile'].upper()} (crossover={gpu_info['crossover_density']})"
    )
    print(
        f"  • Корпус Parquet         : {dataset.parquet_tokens_scanned:,} токенов из {dataset.parquet_stories_used:,} историй"
    )
    print(f"  • Био-словарь            : {len(itos)} лексем | batch_size={batch_size} | epochs={epochs_per_stage}")
    print("=" * 102)

    stages_spec = [
        (1, "GPU Stage 1: Syntagmatic Core (seq_len=4)", 4, dataset.stage1_seqs_len4),
        (2, "GPU Stage 2: Simple Clauses SVO (seq_len=8)", 8, dataset.stage2_seqs_len8),
        (3, "GPU Stage 3: Two-Clause Anaphora (seq_len=16)", 16, dataset.stage3_seqs_len16),
        (4, "GPU Stage 4: Full Narrative Window (seq_len=64)", 64, dataset.stage4_seqs_len64),
    ]

    stage_reports = []
    for s_idx, s_title, s_len, s_seqs in stages_spec:
        rep = trainer.train_stage(
            stage_idx=s_idx,
            stage_name=s_title,
            seq_len=s_len,
            sequences=s_seqs,
            n_epochs=epochs_per_stage,
            batch_size=batch_size,
        )
        stage_reports.append(rep)
        print(
            f"[{s_title}] {rep['elapsed_sec']:.2f}s | Surprise: {rep['surprise_start']:.4f} -> {rep['surprise_end']:.4f} | "
            f"Retention: {rep['context_retention_tokens']:.1f} tok | Grammar: {rep['grammar_validity']*100:.1f}%"
        )

    best_ckpt = stage_reports[-1]["generation_checkpoint"]
    produce_dir = ROOT / "Produce"
    produce_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = produce_dir / "checkpoints" / "ziwago_v8_5_master_gpu.pt"
    _checkpoint(trainer.u, str(ckpt_path), itos)

    out_payload = {
        "gpu_environment": gpu_info,
        "vocab_size": len(itos),
        "total_ticks": trainer.total_ticks,
        "stage_reports": stage_reports,
        "best_generation_checkpoint": best_ckpt,
        "checkpoint_path": str(ckpt_path),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (produce_dir / "v8_gpu_training_results.json").write_text(
        json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ziwago-Prime v8.5 GPU-Accelerated Trainer")
    parser.add_argument("--epochs", type=int, default=4, help="Epochs per developmental stage (default: 4)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per sleep group (default: 8)")
    parser.add_argument("--max-stories", type=int, default=5000, help="Max stories from 10k-tinystories.parquet")
    parser.add_argument("--gpu-arch", type=str, default="ampere", choices=["volta", "ampere"], help="GPU architecture profile")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4), help="CPU data loader threads")
    args = parser.parse_args()
    run_gpu_training(
        epochs_per_stage=args.epochs,
        batch_size=args.batch_size,
        max_stories=args.max_stories,
        gpu_arch=args.gpu_arch,
        workers=args.workers,
    )
