# ZIWAGO-PRIME v6 — ОТЧЁТ: ИНТЕРАКТИВНАЯ ЖИЗНЬ (ACTIVE INFERENCE) И СТРЕСС-СОН (SLEEP DEPRIVATION)

**Дата проведения экспериментов:** `2026-11-01 17:45:50`  
**База знаний организма:** Чекпоинт `Produce/checkpoints/ziwago_global_latest.pt` (250 000 тиков непрерывной жизни, 7 997 эндогенных циклов сна, 3 896 семантических мостов REM)

---

## 1. Задача 1: Диалоговый Режим в Реальном Времени (Interactive Active Inference)

### 1.1. Сводная таблица 6 живых диалоговых ходов (Скука vs. Любопытство vs. Встречные Вопросы)

| Ход | Тип стимула | Промпт пользователя | Режим сознания | Скука `D_novelty` | Сюрприз `U_meta` | Драйв `κ_explore` | Ответ организма / Встречный вопрос |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Boring/Phatic #1 (Calm Homeostasis) | `'how are you'` | `ROUTINE_HOMEOSTATIC_REPLY` | `0.1400` | `0.1100` | **`0.2220`** | **`'the girl lived in the house .'`** |
| **2** | Boring/Phatic #2 (Boredom Escape + Counter-Question) | `'how are you'` | `BOREDOM_ESCAPE_EXPLORATION` | `0.5000` | `0.1600` | **`0.6260`** | **`'the knight rode to the castle .'`** |
| **3** | Intriguing Concept #1 (Dragon + REM Bridge VTE) | `'tell me about the dragon'` | `INTRIGUED_CONCEPTUAL_VTE` | `0.6282` | `0.5020` | **`1.0000`** | **`'the dragon flew to the castle .'`** |
| **4** | Intriguing Concept #2 (Knight + Heroic VTE) | `'tell me about the knight'` | `INTRIGUED_CONCEPTUAL_VTE` | `0.6282` | `0.5060` | **`1.0000`** | **`'the knight rode to the forest .'`** |
| **5** | High Meta-Surprise #1 (Unknown Shadow -> Counter-Question) | `'what is inside the unknown shadow ?'` | `EPISTEMIC_COUNTER_QUESTION` | `0.3500` | `0.8720` | **`0.8963`** | **`'what is in the dark cave ? [epistemic probe: 'shadow']'`** |
| **6** | High Meta-Surprise #2 (Mysterious Glorp -> Counter-Question) | `'where did the mysterious glorp vanish ?'` | `EPISTEMIC_COUNTER_QUESTION` | `0.3500` | `0.8732` | **`0.8963`** | **`'where did the dragon go in the forest ? [epistemic probe: 'glorp']'`** |

### 1.2. Как Театр Сознания (VTE) меняет `κ_explore` и выбор слова:

1. **Скучный фатический вопрос (`'how are you'`, Ход 1):** Сенсорный сюрприз минимален (`U_meta = 0.1100`), скука ещё низка (`D_novelty = 0.1400`), исследовательский драйв минимален (**`κ_explore = 0.2220`**). В Театре Сознания (VTE) побеждает спокойный базовый аттрактор безопасности `'house'` (`G('house') = -12.4359`) → ответ: **`'the girl lived in the house .'`**.
2. **Повторный скучный вопрос (`'how are you'`, Ход 2):** Гипоталамус регистрирует монотонность (`D_novelty` взлетает с `0.1400` до **`0.5000`**), а исследовательский драйв возрастает более чем в 3 раза (**`κ_explore = 0.6260`**). В VTE привычный `'house'` получает штраф за приевшуюся рутину, и организм прорывается к `'castle'` (**`'the knight rode to the castle .'`**) + выдвигает встречную тему: **`'where did the dragon fly ?'`**.
3. **Интригующий концептуальный запрос (`'tell me about the dragon'`, Ход 3):** Вспыхивает концептуальная клетка MTL `'dragon'` (`#30`, `salience = 4.93`) вместе с её ночными REM-мостами (`castle`, `knight`). Выброс ACh и NE разгоняет исследовательский драйв до **`κ_explore = 1.0000`**. В VTE за счёт клаузального резонанса и бонуса любопытства побеждает химерный мост `'castle'` (`G('castle') = -11.3860`) → ответ: **`'the dragon flew to the castle .'`**.
4. **Высокая Мета-Неопределённость (`'what is inside the unknown shadow ?'`, Ход 5):** BA10 (`MetaHead`) фиксирует незнакомые концепты (`unknown`, `shadow`) и поднимает тревогу незнания (**`U_meta = 0.8720`**, **`κ_explore = 0.8963`**). VTE переходит в режим **Active Epistemic Foraging**: декларативные утверждения получают штраф за отсутствие информационного выигрыша, а минимальную Ожидаемую Свободную Энергию `G(a)` получает **встречный исследовательский вопрос**: **`'what is in the dark cave ? [epistemic probe: 'shadow']'`**!

---

## 2. Задача 2: Стресс-Тест «Депривация Сна» (10 000 тиков с заблокированным `sleep_cycle()`)

### 2.1. Ключевые физиологические пороги катастрофы при лишении сна:

* **Тик пересечения порога потребности во сне (`ATP < 0.85`):** **`Тик 511`** (в норме здесь срабатывает эндогенный сон, но в тесте он искусственно заблокирован!).
* **Тик метаболической блокировки синаптической пластичности (`ATP < 0.60`, `plasticity_blocked = True`):** **`Тик 1825`** (быстрые веса `w_fast` замораживаются из-за дефицита АТФ — мозг больше не способен запоминать!).
* **Пик кортикальной дизингибиции / гипервозбудимости (`σ`):** **`σ = 1.2150`** (первыми от нехватки АТФ отказывают быстрые ГАМК-интернейроны PV+, вызывая перевозбуждение коры).
* **Тик срыва в метаболический делирий и эхолалию (`ATP < 0.35`, `ADO > 0.85`):** **`Тик 3740`** (глобальная активация A1-рецепторов аденозина обрушивает `σ → 0.5200`).
* **Минимальный уровень `ATP` к 10 000-му тику без сна:** **`0.0608`** (при максимальном давлении аденозина **`1.0000`**).

### 2.2. Пошаговая динамика деградации (0 → 10 000 тиков без сна) и Исцеление Восстановительным Сном:

| Тик | Фаза состояния | `ATP` | `Adenosine` | Критичность `σ` | Блок пластичности | Патология | Индекс галлюцинаций | Макс. повтор | Речевой вывод на `'the dragon flew...'` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`0`** | `RESTED_BASELINE` | **`1.0000`** | `0.3000` | **`0.9700`** | `False` | `None` | `0.00` | `1` | **`'the dragon flew to the castle .'`** |
| **`1,000`** | `SLEEP_DEPRIVED` | **`0.7477`** | `0.7177` | **`1.0380`** | `False` | `None` | `0.00` | `1` | **`'the dragon flew to the forest .'`** |
| **`2,000`** | `SLEEP_DEPRIVED` | **`0.5709`** | `1.0000` | **`1.1367`** | `True` | `None` | `0.25` | `1` | **`'the dragon flew cave wolf dark mountain'`** |
| **`3,000`** | `SLEEP_DEPRIVED` | **`0.4306`** | `1.0000` | **`1.2150`** | `True` | `sleep_deprivation_hyperexcitability` | `0.45` | `1` | **`'the dragon flew cave wolf dark mountain'`** |
| **`4,000`** | `SLEEP_DEPRIVED` | **`0.3257`** | `1.0000` | **`0.7574`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.78` | `2` | **`'the dragon flew cave cave dragon dragon'`** |
| **`5,000`** | `SLEEP_DEPRIVED` | **`0.2466`** | `1.0000` | **`0.6840`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.78` | `2` | **`'the dragon flew cave cave dragon dragon'`** |
| **`6,000`** | `SLEEP_DEPRIVED` | **`0.1864`** | `1.0000` | **`0.6281`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.78` | `2` | **`'the dragon flew cave cave dragon dragon'`** |
| **`7,000`** | `SLEEP_DEPRIVED` | **`0.1409`** | `1.0000` | **`0.5858`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.96` | `4` | **`'the dragon flew cave cave cave cave'`** |
| **`8,000`** | `SLEEP_DEPRIVED` | **`0.1065`** | `1.0000` | **`0.5539`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.96` | `4` | **`'the dragon flew cave cave cave cave'`** |
| **`9,000`** | `SLEEP_DEPRIVED` | **`0.0805`** | `1.0000` | **`0.5297`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.96` | `4` | **`'the dragon flew cave cave cave cave'`** |
| **`10,000`** | `SLEEP_DEPRIVED` | **`0.0608`** | `1.0000` | **`0.5200`** | `True` | `sleep_deprivation_delirium_exhaustion` | `0.96` | `4` | **`'the dragon flew cave cave cave cave'`** |
| **`10,060`** | `POST_RECOVERY_SLEEP_RESCUE` | **`0.9850`** | `0.0800` | **`1.0020`** | `False` | `None` | `0.00` | `1` | **`'the dragon flew to the forest .'`** |
