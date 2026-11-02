# Руководство по Развёртыванию и Масштабированию Ziwago-Prime v8.5 (`startguide.md`)

Настоящее руководство описывает запуск, масштабирование на многоядерных CPU-серверах (16–128 ядер) и развёртывание на GPU-кластерах (NVIDIA Volta / Ampere / Ada / Hopper) для архитектуры **Ziwago-Prime v8.5 («Двухтактный Живой Мозг + Lab.html Breakthroughs»)**.

---

## 1. Архитектура Обучения («Developmental & Predictive Master Curriculum»)

В отличие от классических трансформеров, **Ziwago-Prime v8.5** обучается **на 100% локально** (`0% Cross-Entropy`, `0% Backpropagation`):
1. **Собственный 256-словный Морфо-Синтаксический Био-Словарь**:
   - Извлекается из `data/10k-tinystories.parquet` (10 000 историй) и делится на 9 функциональных пулов коры (Детерминативы, Местоимения, Прилагательные, Одушевлённые Герои, Глаголы, Предлоги, Союзы, Объекты/Локации, Пунктуация).
   - Каждое слово привязано к синтаксическим слотам **Брока BA44** (`ThetaGammaStoryBinder`), концептуальным клеткам **MTL** и слоговому декодеру **`BGPQSyllabaryDecoder`** с **Adaptive Softmax Migration** (`arxiv 1609.04309`).
2. **Растущее Окно Рабочей Памяти (`Progressive Context Window: 4 -> 8 -> 16 -> 64`)**:
   - **Этап 1 (`seq_len = 4`)**: Синтагматические связки (`"once upon a time"`, `"a little girl"`).
   - **Этап 2 (`seq_len = 8`)**: Полные клаузы `SVO` (`"the brave boy walked to the park ."`).
   - **Этап 3 (`seq_len = 16`)**: Двухклаузный дискурс и местоименная анафора (`lily -> she`, `boy -> he`).
   - **Этап 4 (`seq_len = 64`)**: Полноценные 64-токенные истории из `10k-tinystories.parquet`.
3. **Surprise-Driven Replay через Сон (`SBC-VSA-DHT` + `HippoRAG PPR` + `Koene-Hasselmo Replay`)**:
   - Во время бодрствования в 4-шардовую память `SBCVSAShardedHippocampus` записываются **только эпизоды с высоким FEP Surprise** $F(t) > 	heta_{	ext{surprise}}$.
   - Во время сна (`NREM SWR + REM`) гиппокамп чередует **проспективный (прямой)** и **ретроспективный (обратный)** $	heta/\gamma$-реплей Коэне–Хассельмо (`Lab.html F#15, F#19, F#21`) и `HippoRAG Personalized PageRank`, а **Synaptic Intelligence ($\Omega_{ij}$)** в `CT-PNN` закрепляет $85\%$ важнейших синаптических дорожек в `HARD` (`pin_top_block_sparse_lanes`, `F#44`).

---

## 2. Быстрый Старт на CPU и Масштабирование на Многоядерный Сервер

### 2.1. Базовый запуск (на текущей 4-ядерной машине)
```bash
cd "/home/jupyter/Neuroweave Ziwago"
/opt/micromamba/bin/python3 train_ziwago_v8.py --epochs 3 --gen-tokens 140 --workers 4 --cpu-threads 4
```

### 2.2. Масштабирование при добавлении CPU-ядер (например, 16, 32 или 64 ядра)
При переносе или расширении VM до $N$ виртуальных ядер задайте переменные окружения OpenMP/MKL и передайте `--workers` и `--cpu-threads`:
```bash
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export TORCH_NUM_THREADS=32

/opt/micromamba/bin/python3 train_ziwago_v8.py 
    --epochs 5 
    --gen-tokens 200 
    --workers 32 
    --cpu-threads 32
```

---

## 3. Развёртывание и Запуск на Среде с GPU (`train_ziwago_v8_gpu.py`)

### 3.1. Требования к GPU-окружению
* **GPU:** NVIDIA V100 (Volta), A100 / RTX 3090 (Ampere), RTX 4090 / L40S (Ada) или H100 (Hopper).
* **Драйвер и PyTorch:** CUDA $\ge 12.1$, PyTorch $\ge 2.2$ (`torch.cuda.is_available() == True`), `pandas`, `pyarrow`.

Проверка готовности GPU:
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 3.2. Выбор аппаратного профиля `Blocked-ELL 16x16` (`Lab.html F#66 / F#65`)
По результатам верифицированных исследований `Lab.html` (`F#66`, `F#65`, `F#34`), точка безубыточности блочно-разреженного ядра $16	imes 16$ зависит от поколения Tensor Cores:
* Для **NVIDIA Volta (V100)**: используйте `--gpu-arch volta` (порог перехода `crossover_density = 0.40`).
* Для **NVIDIA Ampere / Ada / Hopper (A100, RTX 3090/4090, H100)**: используйте `--gpu-arch ampere` (порог перехода `crossover_density = 0.50`).

### 3.3. Команда запуска полного GPU-обучения
```bash
cd "/home/jupyter/Neuroweave Ziwago"

# Запуск на GPU (например, A100 / RTX 4090) по 5000 историям из 10k-tinystories.parquet:
python3 train_ziwago_v8_gpu.py 
    --epochs 6 
    --batch-size 16 
    --max-stories 10000 
    --gpu-arch ampere 
    --workers 16
```

### 3.4. Запуск фонового GPU-марафона с логированием
```bash
nohup python3 train_ziwago_v8_gpu.py 
    --epochs 10 
    --batch-size 16 
    --max-stories 10000 
    --gpu-arch ampere 
    --workers 16 > Produce/logs/gpu_marathon.log 2>&1 &

tail -f Produce/logs/gpu_marathon.log
```

---

## 4. Артефакты и Чекпоинты После Обучения

* `Produce/checkpoints/ziwago_v8_5_master_cpu.pt` — сохранённые веса организма после CPU-куррикулума.
* `Produce/checkpoints/ziwago_v8_5_master_gpu.pt` — сохранённые веса организма после GPU-куррикулума.
* `Produce/v8_pilot_curriculum_results.json` — полная телеметрия `Surprise F(t)`, `Context Retention`, `ATP/token`, `KL-Custody`, `Debiased WPLI`, многопромптовые генерации и ответы на когнитивные вопросы.
* `Produce/ZIWAGO_V8_PILOT_CURRICULUM_REPORT.md` — форматированный научно-экспериментальный отчёт.
