# Руководство по Развёртыванию и Масштабированию Ziwago-Prime v9.0 («32-Core / 64-Thread Intelligence Edition»)

Настоящее руководство описывает запуск, масштабирование на многоядерных CPU-серверах (32–64 потока AMD EPYC / Intel Xeon) и развёртывание на GPU-кластерах (NVIDIA Volta / Ampere / Ada / Hopper) для архитектуры **Ziwago-Prime v9.0 («Двухтактный Живой Мозг + Когнитивный Интеллект»)**.

---

## 1. Полная Цепочка Обучения Ziwago-Prime v9.0 (`100% Local FEP + 3-Factor STDP`)

1. **Масштабированный Корпус и 1024-словный Морфо-Когнитивный Био-Словарь**:
   - Параллельный 32-поточный парсер сканирует **все `10,000` историй (`1,969,986` токенов)** из `data/10k-tinystories.parquet` (отбирая `10,160` грамматически чистых клауз) и объединяет их с **Когнитивным Корпусом Модели Мира** (`data/ziwago_v9_cognitive_corpus.json`: физическая причинность, термодинамика, Theory-of-Mind `Sally-Anne`, многошаговое планирование, пространственная логика и эмпатия).
   - Словарь расширен до **`1,024` морфо-синтаксических и когнитивных лексем** в 10 ролевых пулах Брока BA44 + концептуальных клетках MTL.
2. **Морфо-Семантический Хэш-Кортикальный SDR Энкодер (`MorphoSemanticCorticalEncoder`)**:
   - Кодирует каждое слово (включая незнакомые `OOV`-слова) в разреженный кортикальный вектор из **`2,048` нейронов** (`8` гиперколонок × `256` миниколонок, ровно `40` активных спайков = $1.95\%$).
   - Первые 5 гиперколонок кодируют синтаксическую роль (`role_id`), морфемный корень (`stem`) и аффикс (`ing/ed/ly/un/re`), обеспечивая мгновенное обобщение однокоренных слов (`24.6%` перекрытия SDR у `walk <-> walked` против `12.5%` у несвязанных слов) при всего `544` операциях на токен.
3. **5-Этапная Программа Развития (`Progressive Context Window: 4 -> 8 -> 16 -> 64 -> 128`)**:
   - **Этап 1 (`seq_len = 4`)**: Синтагматическое ядро.
   - **Этап 2 (`seq_len = 8`)**: Простые клаузы `SVO`.
   - **Этап 3 (`seq_len = 16`)**: Двухклаузный дискурс и местоименная анафора (`lily -> she`, `leo -> he`).
   - **Этап 4 (`seq_len = 64`)**: Полный нарратив `TinyStories`.
   - **Этап 5 (`seq_len = 128`)**: Когнитивный интеллект, физическая причинность (`because / therefore / so`) и `Theory-of-Mind` (`believed / knew`).
4. **Композиционная Память, Сон и Мысленная Репетиция (`Active Inference VTE`)**:
   - **`CompositionalThetaGammaIndexer`**: факторизованная ролевая память `Subject × Verb × Object × Context` с комбинаторной ёмкостью **`64^4 = 16,777,216` связок**.
   - **`SBCVSAShardedHippocampus` (`38,400` слотов)** + **`HippoRAG Personalized PageRank (PPR)`** + двунаправленный проспективный/ретроспективный $	heta/\gamma$-реплей Коэне–Хассельмо (`Lab.html F#15, F#19, F#21`).
   - **`Active Inference VTE`** (`simulate_vte`) и диспетчер **`Probe / Attend / Stop`** (`F#37`) для осознанного выбора слов и ответов на причинно-следственные вопросы.

---

## 2. Запуск на 32-Ядерном / 64-Поточном Сервере (CPU)

```bash
cd "/home/jupyter/Neuroweave Ziwago"

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export TORCH_NUM_THREADS=32

/opt/micromamba/bin/python3 train_ziwago_v8.py 
    --epochs 3 
    --gen-tokens 160 
    --max-vocab 1024 
    --max-stories 10000 
    --workers 32 
    --cpu-threads 32
```

Запуск верификационных тестов `v8.5 + v9.0`:
```bash
cd "/home/jupyter/Neuroweave Ziwago"
/opt/micromamba/bin/python3 -c "import tests.test_v10_genesis as t; t.test_ziwago_prime_v8_5_lab_html_breakthroughs(); t.test_ziwago_prime_v9_morpho_sdr_and_compositional_intelligence(); print('ALL V8.5 & V9.0 TESTS PASSED')"
```

---

## 3. Развёртывание и Запуск на GPU-Кластере (`train_ziwago_v8_gpu.py`)

```bash
cd "/home/jupyter/Neuroweave Ziwago"

/opt/micromamba/bin/python3 train_ziwago_v8_gpu.py 
    --epochs 5 
    --batch-size 16 
    --max-vocab 1024 
    --max-stories 10000 
    --gpu-arch ampere 
    --workers 32
```
* Для **NVIDIA Volta (V100)** используйте `--gpu-arch volta` (`crossover_density = 0.40`).
* Для **NVIDIA Ampere / Ada / Hopper (A100, RTX 4090, H100)** используйте `--gpu-arch ampere` (`crossover_density = 0.50`).

---

## 4. Артефакты и Чекпоинты После Обучения

* `Produce/checkpoints/ziwago_v9_0_intelligence_master.pt` (и `ziwago_v8_5_master_cpu.pt`) — мастер-чекпоинт обученного организма v9.0 (`vocab_size=1024`, 5 этапов).
* `data/ziwago_v9_cognitive_corpus.json` — сгенерированный когнитивный корпус физической причинности, планирования и Theory-of-Mind.
* `Produce/v9_intelligence_curriculum_results.json` (и `v8_pilot_curriculum_results.json`) — полная телеметрия обучения, генераций и когнитивного QA.
* `Produce/ZIWAGO_V8_PILOT_CURRICULUM_REPORT.md` — научно-экспериментальный отчёт с кривой обучения и результатами.
