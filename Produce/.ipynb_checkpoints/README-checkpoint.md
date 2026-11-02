# Produce/ — артефакты обучения NEUROWEAVE v10

Единая папка результатов куррикулум-обучения «True Bio-Organism».

## Содержимое

| Путь | Что это |
|---|---|
| `train_curriculum_stage2.py` | Скрипт Этапа 2: 4000 историй TinyStories, seq_len=64, RSS-мониторинг, reward-окна, генерация 1000 токенов с текстом. Запуск: `python Produce/train_curriculum_stage2.py` из корня репо. |
| `checkpoints/curriculum_stage1.pt` | Веса после Этапа 1 (500 историй, seq_len=32): моторный пул (w_l5, w_hip, w_lap, w_rec_m, STP-состояние), PFC-модель w_pred, кортикальные синапсы всех зон. |
| `checkpoints/curriculum_stage2.pt` | Веса после Этапа 2 (4000 историй, seq_len=64). Загружается через `load_checkpoint()` в `neuroweave10/unified/pipeline.py`. |
| `logs/stage2.log` | Полный лог Этапа 2: RSS/CE/reward каждые 400 историй + RESULT_JSON с метриками и текстом генерации. |

## Метрики Этапа 2 (из лога)

- tokens_trained: 255986, CE 14.02, RSS 461MB (утечки нет)
- tail_vocab = 31, max_repeat_run = 1, σ = 0.958, stp_min = 0.13
- reward_mean −0.45 (≈случайный уровень — см. PLAN.md)

## Воспроизведение

```bash
python - <<'PY'
from neuroweave10.unified.pipeline import (
    RegenesisOrganism, UnifiedConfig, load_checkpoint)
from neuroweave10.config import genesis_cfg
u = RegenesisOrganism(UnifiedConfig(vocab_size=128),
                      gcfg=genesis_cfg("tiny"))
load_checkpoint(u, "Produce/checkpoints/curriculum_stage2.pt")
gen = u.generate(n_tokens=1000, seed_tokens=[1, 5, 9])
PY
```
