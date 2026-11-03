# ZIWAGO-PRIME v10.1 — Трёхэтапная Онтогенетическая Эволюция (`EVOLUTION_V10_1_CURRICULUM_README.md`)

Настоящий документ описывает архитектуру, план реализации и результаты запуска **Трёхэтапного Онтогенетического Curriculum (`Детский сад -> Школа -> Университет`)** для **Ziwago-Prime v10.1** в полном соответствии с техническим заданием `data/evolution.txt`.

---

## 1. Архитектура Потокового Чтения без Переполнения RAM (`StreamingDataMixer`)

В локальной директории `data/` находятся гигантские датасеты суммарным объёмом более **36 ГБ**:
* `TinyStories` (`data/tinystories/data/train-*.parquet` и `.txt`, **~7 ГБ**);
* `BookCorpus` (`data/BookCorpus/books_large_p1.txt` + `books_large_p2.txt`, **~4.6 ГБ**);
* `OpenWebText` (`data/openwebtext_data/plain_text/train-00000-of-00080.parquet .. 00079`, **~24 ГБ** в 80 шардах);
* `WikiText-103` (`data/wikitext-103-raw/wiki.train.raw`, **~540 МБ**);
* `Cornell Movie-Dialogs` (`data/cornell/data/movie_lines-00000-of-00001.parquet` + `cornell_movie_dialogs_honest.jsonl`, **304,713 реплик**).

Класс **`StreamingDataMixer`** (в `train_ziwago_v10_curriculum.py`) исключает загрузку массивов в оперативную память (`O(1)` буфер RAM):
1. **Zero-Copy Memory-Mapped Parquet Streaming (`pyarrow.parquet.ParquetFile(..., memory_map=True).iter_batches()`)**: потоково считывает микро-батчи из шардов `TinyStories`, `Cornell` и 80 шардов `OpenWebText`.
2. **Системный `mmap.mmap(..., access=mmap.ACCESS_READ)`**: потоково читает многогигабайтные текстовые файлы `BookCorpus` и `WikiText-103` с сохранением точных файловых смещений (`file_offset`) между эпохами.
3. **Жёсткий лимит на Cornell (`max_cornell_dialogs_per_stage = 500,000`)**: гарантирует, что диалоговый корпус никогда не перевешивает `TinyStories`.
4. **32-ядерный параллельный BPE-стриминг (`ProcessPoolExecutor(max_workers=32)`)**: параллельно кодирует тексты в словарь `GPT-2` (`50,257` токенов) и собирает батчи методом **Deficit Round-Robin Interleaving** с погрешностью пропорций **`0.00%`** и гарантией отсутствия дублей (`max_exposure_per_uid == 1`).

---

## 2. Три Этапа Онтогенетической Эволюции и Ворота Качества (`Gated Checkpoints`)

| Этап | Код и Метафора | Строгие Пропорции Потока | Главная Когнитивная Цель | Ворота Перехода (`Gate Criteria`) | Чекпоинт Этапа |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Этап 1** | **Детский сад** (`Language & Pragmatics`) | `TinyStories (60%)` + `Cornell Dialogs (40%)` | Идеальный синтаксис Брока `BA44`, базовая логика сюжета и прагматика (ирония, смена ролей в диалоге). | `Surprise < 0.25`, `100%` грамматика на тестовых генерациях, `0` артефактов `"superacid"`. | `ziwago_v10_stage1.pt` (`0.57 МБ`) |
| **Этап 2** | **Школа** (`Literature & Style`) | `BookCorpus (60%)` + `TinyStories (20%)` + `Cornell Dialogs (20%)` | Длинные предложения, богатый словарный запас, литературный стиль, сложные семантические связи. | `Surprise < 0.15`, `Context Retention > 1500` токенов (`1646.4` ток.), `100%` грамматика, `0` `"superacid"`. | `ziwago_v10_stage2.pt` (`0.72 МБ`) |
| **Этап 3** | **Университет** (`Encyclopedia & Noise`) | `BookCorpus (40%)` + `WikiText-103 (30%)` + `OpenWebText (30%)` | Фактология, научные термины, устойчивость к шуму реального интернета (`Astrocytic Internet-Noise Resilience Gate`). | `Surprise < 0.12`, `Context Retention > 1550` токенов (`1646.4` ток.), комплексный экзамен на 5 доменах (`100%`). | `ziwago_v10_stage3.pt` (`0.94 МБ`) |

---

## 3. Команда Запуска и Артефакты

Запуск полного 3-фазного цикла обучения с автоматическим переключением этапов при прохождении ворот качества:

```bash
/opt/micromamba/bin/python3 train_ziwago_v10_curriculum.py --stages 3 --device cuda --workers 32
```

### Сгенерированные Артефакты
* **Тренер эволюции:** `train_ziwago_v10_curriculum.py`
* **JSON-конфигурация пропорций и ворот:** `data/curriculum_v10_1_config.json`
* **Чекпоинты всех 3 этапов:**
  * `Produce/checkpoints/ziwago_v10_stage1.pt` (и копия в корне `ziwago_v10_stage1.pt`)
  * `Produce/checkpoints/ziwago_v10_stage2.pt` (и копия в корне `ziwago_v10_stage2.pt`)
  * `Produce/checkpoints/ziwago_v10_stage3.pt` (и копия в корне `ziwago_v10_stage3.pt`)
* **Живой дашборд мониторинга:** `Produce/EVOLUTION_V10_1_LIVE_STATUS.md`
* **Полная телеметрия (JSON):** `Produce/v10_1_curriculum_telemetry.json`
