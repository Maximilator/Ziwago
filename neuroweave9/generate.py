"""generate.py — генерация текста обученным NEUROWEAVE v9.0 «Cortex».

Загружает чекпоинт, сохранённый `train_cortex.py --save out.pt`
(организм + декодер + словарь TinyStories), прогоняет затравку через
таламус→колонки→SON→синхронизацию полушарий→Лаплас→БГ и генерирует
продолжение. Темп эмиссии задаёт дрифт-диффузия базальных ганглиев
(порог опускает орексин при дефиците энергии, повышает SON-surprise на
непредсказанном контексте).

Запуск:
    python -m neuroweave9.generate --ckpt cortex.pt \
        --prompt "once upon a time" --num-tokens 14
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

from .cortex.organism import CortexOrganism
from .readout import LaplaceContextReadout


@torch.no_grad()
def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    org = CortexOrganism(
        vocab_size=cfg["vocab_size"], input_dim=cfg["input_dim"],
        scale=cfg["scale"], vsa_dim=cfg["vsa_dim"], seed=cfg["seed"],
        device=device)
    org.load_state_dict(ckpt["org"], strict=False)
    readout = LaplaceContextReadout(
        cortical_dim=org.cortical_dim(),
        laplace_dim=org.laplace_memory.num_scales * org.input_dim,
        vocab_size=cfg["vocab_size"], sync_dim=cfg["vsa_dim"],
        l23_dim=org.l23_dim()).to(device)
    readout.load_state_dict(ckpt["readout"], strict=False)
    org.set_phase(2)
    return org, readout, ckpt["itos"], ckpt["stoi"]


@torch.no_grad()
def generate(
    org: CortexOrganism,
    readout: LaplaceContextReadout,
    prompt_words: List[str],
    itos: List[str],
    stoi: Dict[str, int],
    num_tokens: int = 14,
    max_bg_ticks: int = 5,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.75,
    top_k: int = 40,
    top_p: float = 0.9,
    seed: Optional[int] = 1234,
    nogo_veto: float = 3.0,
    theta_mode: str = "baseline",
    theta_cycle: float = 0.0,
) -> Tuple[str, Dict[str, float]]:
    """Затравка через полный контур, затем автономная генерация.

    Каждый токен: такт коры → fused-логиты readout → БГ решает «когда
    сказать» (emit_mask) и «что сказать».

    Токен выбирает стриатум: он видит логиты коры ЗАВЫЧЕННОЙ на моторный тормоз
    NoGo, поэтому только что произнесённый токен автоматически становится
    менее вероятным, а не повторяется. При temperature > 0 добавляется
    стохастическая softmax-политика (top-k / top-p) — это даёт контуру уйти в
    альтернативную гипотезу, которую подготовил гиппокампальный реколл.
    При temperature <= 0 — жадный argmax (воспроизводимый режим).
    """
    org.set_phase(2)
    org.reset_state(batch_size=1, device=device)
    # Стратегия выбора действия настраивается ОДИН раз на весь высказывание:
    # моторный тормоз NoGo обязан жить между токенами, иначе повторы не
    # подавляются (и получается «clean clean clean»).
    org.basal_ganglia.configure_sampling(
        temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)
    # Вето NoGo: действие, произнесённое подряд несколько раз, исключается
    # из набора — контур обязан переключиться на альтернативу.
    org.basal_ganglia.configure_nogo_veto(nogo_veto)
    # Масштаб тета-цикла гиппокампа (рантайм, веса не меняются).
    org.configure_theta_clock(mode=theta_mode, cycle_steps=theta_cycle)

    def make_input(word: str) -> torch.Tensor:
        x = torch.zeros((1, org.input_dim), device=device)
        x[0, stoi.get(word, 1)] = 1.0
        return x

    for w in prompt_words[:-1]:
        org.oscillator.phase_reset(0.0)
        org.step(make_input(w), learn=False, sync_metrics=False, skip_gate=True)

    out_words: List[str] = []
    bg_emissions, bg_ticks = 0, 0
    for _ in range(num_tokens):
        org.oscillator.phase_reset(0.0)
        last = prompt_words[-1] if not out_words else out_words[-1]
        out = org.step(make_input(last), learn=False, sync_metrics=False, skip_gate=True)
        cortical = torch.cat([out["left"]["zone_a"]["spikes"],
                              out["left"]["zone_b"]["spikes"],
                              out["right"]["zone_c"]["spikes"],
                              out["left"]["l5_spikes"]["A"],
                              out["left"]["l5_spikes"]["B"],
                              out["right"]["l5_spikes_c"]], dim=-1)
        lap = org.laplace_memory.laplace_state.permute(1, 0, 2).reshape(1, -1)
        logits = readout(cortical, lap, out["left"]["zone_a"]["logits"],
                         out["sync"]["frame_real"])
        logits[:, :2] = -1e9

        bg = None
        for _tick in range(max_bg_ticks):
            bg_ticks += 1
            bg = org.basal_ganglia(
                logits,
                serotonin_5ht=out["left"]["zone_a"]["modulators"][
                    "serotonin_5ht"],
                ach_level=out["left"]["zone_a"]["modulators"]["ach"],
                orexin_level=out["orexin"]["orexin"],
                surprise_level=out["son"]["surprise"])
            if float(bg["emit_mask"].item()) >= 1.0:
                bg_emissions += 1
                break
        # Кандидат стриатума последнего тика — в том числе с учётом NoGo.
        tok_id = int(bg["selected_token"].item())
        out_words.append(itos[tok_id])

    text = ""
    for tok in [prompt_words[0]] + out_words:
        text += tok if tok in (".", ",", "!", "?", ";") \
            else (" " if text else "") + tok
    return text, {
        "bg_emissions": float(bg_emissions),
        "mean_ticks_to_decision": bg_ticks / max(1, num_tokens),
        "emit_rate": bg_emissions / max(1, num_tokens),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="NEUROWEAVE v9.0 Cortex — генерация")
    ap.add_argument("--ckpt", required=True, help="чекпоинт train_cortex --save")
    ap.add_argument("--prompt", default="once upon a time")
    ap.add_argument("--num-tokens", type=int, default=14)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--temperature", type=float, default=0.75,
                    help="температура сэмплирования (0 = жадный argmax)")
    ap.add_argument("--top-k", type=int, default=40,
                    help="оставить k вероятнейших действий (0 = без фильтра)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="nucleus-фильтрация top-p (1.0 = выключена)")
    ap.add_argument("--seed", type=int, default=1234,
                    help="зерно сэмплирования (для воспроизводимости)")
    ap.add_argument("--nogo-veto", type=float, default=3.0,
                    help="порог вето NoGo: действие, повторённое столько раз, "
                         "исключается из набора (inf = вето выключено)")
    ap.add_argument("--theta-mode", default="baseline",
                    choices=["baseline", "fast", "token"],
                    help="масштаб тета-цикла гиппокампа: baseline=143 шага, "
                         "fast=ускоренные часы (--theta-cycle), "
                         "token=фаза от счётчика токенов")
    ap.add_argument("--theta-cycle", type=float, default=0.0,
                    help="длина тета-цикла в шагах для --theta-mode fast/token")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    # Полная детерминированность: сеем и ГЛОБАЛЬНЫЙ RNG (шум стриатума sigma и
    # Хопфилд CA3 используют глобальный генератор, а не только наш сэмплер).
    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    org, readout, itos, stoi = load_checkpoint(args.ckpt, device)
    prompt = args.prompt.strip().split()
    mode = ("жадный argmax" if args.temperature <= 0 else
            f"T={args.temperature:g} top_k={args.top_k} top_p={args.top_p:g}")
    print(f"[v9 генерация] словарь={len(itos)} | затравка: "
          f"'{' '.join(prompt)}' | {device} | выбор: {mode}")
    text, stats = generate(org, readout, prompt, itos, stoi,
                           num_tokens=args.num_tokens, device=device,
                           temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p, seed=args.seed,
                           nogo_veto=args.nogo_veto,
                           theta_mode=args.theta_mode, theta_cycle=args.theta_cycle)
    print(f"  сеть: '{text}'")
    print(f"  эмиссий БГ={stats['bg_emissions']:.0f} "
          f"({stats['emit_rate']:.0%} слов через гейт), "
          f"решение={stats['mean_ticks_to_decision']:.1f} мс/слово "
          f"({time.perf_counter() - t0:.1f} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
