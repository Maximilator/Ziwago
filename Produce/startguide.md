# Руководство по Развёртыванию и Масштабированию Ziwago-Prime v8.5 (`Produce/startguide.md`)

См. основной файл `../startguide.md` в корне репозитория `Neuroweave Ziwago/startguide.md`.

## Быстрые команды запуска

### 1. Многоядерный CPU-тренер (`train_ziwago_v8.py`)
```bash
/opt/micromamba/bin/python3 train_ziwago_v8.py --epochs 3 --gen-tokens 140 --workers 4 --cpu-threads 4
```

### 2. GPU-тренер (`train_ziwago_v8_gpu.py`)
```bash
/opt/micromamba/bin/python3 train_ziwago_v8_gpu.py --epochs 6 --batch-size 16 --max-stories 10000 --gpu-arch ampere --workers 16
```
