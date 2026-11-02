# Ziwago-Prime v8.0 — Отчёт о Пробном Прогоне (`train_ziwago_v8.py`: Developmental & Predictive Curriculum)

* **Источник данных:** `data/10k-tinystories.parquet` (проанализировано **`141,580`** токенов из `751` историй) + структурированный грамматический корпус
* **Био-словарь:** **`114` лексем** с ролевой разметкой Брока BA44 и якорями в MTL
* **Всего тактов обучения:** **`2,745`** | **Циклов сна (`Surprise-Driven Replay`):** **`34`**
* **Затраты АТФ на 1 токен (`ATP Efficiency`):** **`0.00386`**
* **Время выполнения на CPU:** **`205.5 с`**

## 1. Learning Curve (Падение Свободной Энергии / Surprise $F(t)$)

```text
  Этап / Эпоха          | Средний Surprise F(t) | Динамика минимизации Свободной Энергии
  ------------------------------------------------------------------------------------------
  S1(L= 4) Ep1          |                1.4897 | ████████████████████████████████████████████
  S1(L= 4) Ep2          |                0.9175 | █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S1(L= 4) Ep3          |                0.7628 | █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S2(L= 8) Ep1          |                1.0726 | ███████████████████░░░░░░░░░░░░░░░░░░░░░░░░░
  S2(L= 8) Ep2          |                0.9116 | █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S2(L= 8) Ep3          |                0.8249 | ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S3(L=16) Ep1          |                1.0429 | █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S3(L=16) Ep2          |                0.8928 | ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S3(L=16) Ep3          |                0.8078 | ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S4(L=64) Ep1          |                0.9747 | █████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S4(L=64) Ep2          |                0.8702 | ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  S4(L=64) Ep3          |                0.8001 | ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

## 2. Таблица Прогресса по 4 Этапам (`Progressive Context Window: 4 -> 8 -> 16 -> 64`)

| Этап | `seq_len` | Surprise Start -> End | Context Retention | ATP / токен | Грамматическая связность | `CT-PNN HARD` (`SI` Ω) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Этап 1** | `4` | `1.4897 -> 0.7628` | **`86.40 ток.`** | `0.00409` | **`100.0%`** | `1054` (`1.1422`) |
| **Этап 2** | `8` | `1.0726 -> 0.8249` | **`93.60 ток.`** | `0.00389` | **`100.0%`** | `1075` (`0.1400`) |
| **Этап 3** | `16` | `1.0429 -> 0.8078` | **`100.80 ток.`** | `0.00385` | **`100.0%`** | `1091` (`0.0286`) |
| **Этап 4** | `64` | `0.9747 -> 0.8001` | **`108.00 ток.`** | `0.00377` | **`100.0%`** | `1091` (`0.0029`) |

## 3. Речь Организма После Первого Цикла Сна (`once upon a time`)

* **Связность:** `100.0%` | **Удержание контекста:** `75.4` токенов | **Уникальных слов:** `24`

> once upon a time girl saw a bright . she walked to the park was a little girl . the forest upon a brave bird in the happy . then the park in the mountain . a little girl saw a bright bird in the forest . tim upon a brave girl saw a bright . she walked to the park was a little girl . the forest upon a brave bird in the happy . then the park in the mountain . a little girl saw a bright bird in the forest . tim upon a brave girl saw a bright . she walked to the park was a little girl . the forest upon a brave bird in the happy . one

## 4. Лучшая Генерация После Завершения Куррикулума (`Stage_4_seq64`)

* **Связность:** `100.0%` | **Удержание контекста:** `108.0` токенов | **Уникальных слов:** `31`

> once upon a time there was a little girl . he was a brave boy went to the park . the knight walked to the old castle . the dragon flew to the mountain . the knight walked to the park . once upon a time there was a little girl . she saw a bright bird in the tree . he was a little girl saw a bright bird in the tree . one day she saw a bright bird in the tree . a little girl saw a bright bird in the tree . the knight walked to the park . she saw a bright bird in the tree . the little girl saw a bright bird in the tree . one day she saw a bright bird in the tree . a little girl saw a bright bird in the tree
