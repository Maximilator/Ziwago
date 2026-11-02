"""Этап 2 куррикулума (Ziwago-Prime): 4000 историй seq_len=64 → чекпоинт → генерация 1000.

Логирует: RSS-память каждые 400 историй, средний reward и accuracy окнами,
метрики генерации. Выводит JSON-итог в конец.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import resource
import time

import torch
import torch.nn.functional as F

from neuroweave10.unified.pipeline import (
    RegenesisOrganism, UnifiedConfig, _seq_pairs, _checkpoint)
from neuroweave10.config import genesis_cfg
from neuroweave9.data import load_stories, build_vocab_and_sequences

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE_DIR, "data/10k-tinystories.parquet")
N_STORIES, SEQ_LEN, MAX_VOCAB = 4000, 64, 128
CKPT = os.path.join(os.path.dirname(__file__), "checkpoints/curriculum_stage2.pt")

stories = load_stories(DATA, max_stories=N_STORIES)
itos, stoi, *_ = build_vocab_and_sequences(
    stories[:500], max_vocab_size=MAX_VOCAB, seq_len=SEQ_LEN,
    batch_size=1)
u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                      gcfg=genesis_cfg("tiny"), seed=10)
u.punct_ids = {stoi[p] for p in (".", "!", "?") if p in stoi}
unk = stoi.get("<unk>", 1)

def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

t0 = time.time()
ce_sum, n_tok, rewards, hits = 0.0, 0, [], []
print(f"[stage2-prime] started: stories={N_STORIES} seq_len={SEQ_LEN} vocab={len(itos)}", flush=True)

for i, story in enumerate(stories):
    # Фильтруем <unk>, чтобы моторный пул и кора учили реальные переходы слов словаря
    raw_ids = _seq_pairs(story, stoi, SEQ_LEN + 24, unk)
    ids = [tid for tid in raw_ids if tid != unk][:SEQ_LEN + 1]
    if len(ids) < 2:
        ids = raw_ids[:SEQ_LEN + 1]
    for a, b in zip(ids, ids[1:]):
        r = u.step_token(torch.tensor([a]), torch.tensor([b]),
                         learn=True)
        rewards.append(r["reward"])
        hits.append(int(r["token"] == b))
        if u._pred_logits is not None:
            logits = u._pred_logits.float()
            v_norm = (logits - logits.mean()) / logits.std().clamp(min=1.0)
            ce_sum += float(F.cross_entropy(
                v_norm, torch.tensor([b])).item())
            n_tok += 1
    if (i + 1) % 400 == 0:
        wr = sum(rewards[-12000:]) / max(1, len(rewards[-12000:]))
        acc_w = sum(hits[-12000:]) / max(1, len(hits[-12000:]))
        el = time.time() - t0
        print(f"[stage2] stories={i+1} tok={n_tok} elapsed={el:.0f}s rss={rss_mb():.0f}MB "
              f"ce={ce_sum/max(1,n_tok):.3f} acc={acc_w:.3f} reward_ema={wr:.4f} "
              f"sigma={u.org.sigma:.3f} patho={u.org.pathology}",
              flush=True)
        _checkpoint(u, CKPT, itos)
    if (i + 1) % 800 == 0 and (i + 1) < N_STORIES:
        u.sleep(ticks=25)

u.sleep(ticks=40)
_checkpoint(u, CKPT, itos)

# Генерация 1000 токенов + текст
seed_words = ["once", "upon", "a", "time"]
seed_ids = [stoi[w] for w in seed_words if w in stoi] or [2, 3, 4]
gen = u.generate(n_tokens=1000, seed_tokens=seed_ids)
ids = gen["ids"]
text = " ".join(itos[i] if i < len(itos) else "<unk>" for i in ids)
reward_mean = sum(rewards) / max(1, len(rewards))
reward_tail = sum(rewards[-16000:]) / max(1, len(rewards[-16000:]))
acc_mean = sum(hits) / max(1, len(hits))
acc_tail = sum(hits[-16000:]) / max(1, len(hits[-16000:]))
out = {
    "tokens_trained": n_tok,
    "ce": ce_sum / max(1, n_tok),
    "acc_mean": acc_mean,
    "acc_tail": acc_tail,
    "reward_mean": reward_mean,
    "reward_tail": reward_tail,
    "reward_first2k": sum(rewards[:2000]) / max(1, len(rewards[:2000])),
    "reward_last2k": sum(rewards[-2000:]) / max(1, len(rewards[-2000:])),
    "rss_mb": rss_mb(),
    "elapsed_s": round(time.time() - t0, 1),
    "gen": {k: gen[k] for k in ("margin_mean", "margin_max",
                                "tail_vocab", "total_vocab",
                                "max_repeat_run", "stp_min",
                                "sigma", "pathology")},
    "text_first200": " ".join((itos[i] if i < len(itos) else "<unk>")
                             for i in ids[:200]),
    "text_last200": " ".join((itos[i] if i < len(itos) else "<unk>")
                            for i in ids[-200:]),
}
print("RESULT_JSON:" + json.dumps(out), flush=True)
