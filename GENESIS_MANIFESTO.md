# NEUROWEAVE v10.0 «Genesis» — Манифест Архитектуры

> *"Мозг — не компьютер, вычисляющий функции. Мозг — это химический реактор,
> который порождает вычисления как побочный эффект поддержания гомеостаза."*

**Версия:** 10.0-genesis | **Дата:** 2026-10-23 | **Статус:** Архитектурный Манифест

---

## Часть 0: Философский Манифест

### Почему Transformer — тупик

Transformer решает задачу аппроксимации функции: вход → выход, с минимальной ошибкой на обучающем
распределении. Это статистическая машина, не имеющая:

- **Времени.** Все токены обрабатываются «одновременно». Нет разницы между 1 мс и 1 часом.
  В мозге NMDA-плато длится 100 мс, диффузия NO — 2 секунды, консолидация памяти — часы.
  Время — не аргумент функции, а физика вычислительного субстрата.

- **Тела.** Нет метаболизма, нет усталости, нет голода. В мозге Аденозин интегрирует
  «давление сна», Орексин удерживает бодрствование, Лактат — энергетическая валюта
  пластичности. Без этих ограничений нет приоритизации, нет мотивации, нет автономии.

- **Химии.** Backpropagation — это единый глобальный сигнал ошибки, телепортирующийся
  через всю сеть. В мозге нет такого провода. Есть облако Дофамина (reward prediction error),
  облако NO (гетеросинаптическая пластичность в радиусе 100 мкм), облако Кортизола
  (переключение стратегии обучения на часы). Химия — не шум. Химия — это операционная
  система мозга.

- **Гомеостаза.** Transformer может взорваться (gradient explosion) или затухнуть
  (vanishing gradients) — и его спасают инженерные костыли (LayerNorm, gradient clipping).
  Мозг удерживает себя на грани хаоса (branching factor σ ≈ 1) через локальные петли
  обратной связи: синаптическая депрессия, intrinsic plasticity, ингибиторный STDP.
  Гомеостаз — не опция. Гомеостаз — это фундаментальное вычислительное свойство.

### Что такое Genesis

Genesis — это **виртуальная биология**: система дифференциальных уравнений, описывающая
мембранные потенциалы, синаптические веса, концентрации нейромедиаторов и метаболитов.
В ней нет «прямого прохода» и «обратного прохода». Есть **непрерывная динамика**,
где каждый шаг по времени (dt = 1 мс) обновляет все переменные параллельно, а обучение
происходит через локальные правила, доступные каждому синапсу.

Четыре уровня Genesis воспроизводят четыре масштаба мозговых вычислений:

| Уровень | Масштаб | Биологический аналог | Математика |
|---------|---------|---------------------|------------|
| 1. Сенсорика | 1-10 мс | Таламус + TRN | SDR-кодирование, критичность |
| 2. Кора | 10-300 мс | Кортикальная колонка | Urbanczik-Senn + Rao-Ballard |
| 3. Память | 100 мс — часы | Гиппокамп + Базальные ганглии | Hopfield + DDM + Химия |
| 4. Сознание | 250+ мс | Global Workspace | Modern Hopfield + HOT + AST |

Эти уровни не последовательны — они работают **одновременно на каждом такте**, с разными
временными константами, обмениваясь сигналами через таламические петли, нейромодуляторные
облака и объёмную передачу.

### Почему это шаг к AGI

AGI — это не «большая языковая модель». AGI — это система, которая:

1. **Учится непрерывно** без переобучения с нуля (complementary learning systems: быстрый
   гиппокамп + медленная кора).
2. **Знает, чего не знает** (Meta-Head / HOT: калиброванная уверенность, epistemic vs aleatoric
   uncertainty).
3. **Управляет своим вниманием** (Attention Schema: модель собственных когнитивных ресурсов,
   TRN-гейтирование).
4. **Мотивирована** (Орексин/Аденозин/Дофамин: энергетические и мотивационные контуры
   определяют *что* учить и *когда*).
5. **Гомеостатична** (σ ≈ 1: самоорганизованная критичность предотвращает коллапс и взрыв).

Genesis — не имитация мозга «ради красоты». Это инженерная необходимость: чтобы система
работала автономно, в непредсказуемой среде, без человека-оператора, ей нужны все эти
механизмы. Transformer их не имеет и не может иметь — его архитектура принципиально
исключает время, тело и химию.

---

## Часть I: Единая Математическая Модель

### 1.1 Глобальное уравнение состояния

Состояние сети Genesis в момент t описывается кортежем:

```
Ψ(t) = { V(t), W(t), M(t), E(t), Θ(t) }
```

где:
- **V(t)** ∈ ℝ^N — мембранные потенциалы всех нейронов (сома + дендриты)
- **W(t)** ∈ ℝ^(N×N) — синаптические веса
- **M(t)** ∈ ℝ^K — концентрации K нейромедиаторов/модуляторов в каждой точке пространства
- **E(t)** ∈ ℝ^N — энергетический бюджет каждого нейрона (ATP/Лактат)
- **Θ(t)** ∈ ℝ^N — адаптивные пороги (intrinsic plasticity)

Каждый компонент эволюционирует по своему ODE с собственной временной константой:

### 1.2 Мембранная динамика (τ ~ 1-20 мс)

Мультикомпартментный нейрон Urbanczik-Senn с химическими модуляциями:

```
C_m · dV_S,i/dt = -g_L·(V_S,i - E_L)
                  + g_D·(V_D,i^basal - V_S,i)
                  + g_A·(V_D,i^apical - V_S,i)
                  - I_fast,i
                  - I_metabolic,i
                  + I_ext,i

C_m · dV_D,i^basal/dt  = -g_L·(V_D,i^basal - E_L) + I_FF,i

C_m · dV_D,i^apical/dt = -g_L·(V_D,i^apical - E_L) + I_FB,i
```

где:
```
I_FF,i   = Σ_j w_ij^FF · x_j(t)                          -- feedforward input (AMPA-доминантный)
I_FB,i   = Σ_k w_ik^FB · x_k(t) · B(V_D^apical, [Mg²⁺]) -- feedback input (NMDA-доминантный)
B(V)     = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062·V))         -- Jahr-Stevens Mg-block

I_fast,i = g_GABA,i(t)·(1 + η_ALLO·[ALLO])·(V_S,i - E_Cl)  -- ингибирование
I_metabolic,i = g_KATP,i([ATP])·(V_S,i - E_K) + g_GIRK,i([A1])·(V_S,i - E_K)
```

Нейрон спайкает когда V_S,i > Θ_i(t), после чего:
```
φ(V_S) = φ_max / (1 + exp(-β·(V_S - Θ)))   -- firing rate transfer function
```

### 1.3 Синаптическая пластичность (τ ~ 100 мс - секунды)

**Трёхфакторное правило Urbanczik-Senn с химической модуляцией:**

```
dw_ij/dt = η · f_mod(M(t)) · [S_i(t) - φ(V*_D,i)] · x_j(t)
```

где:
```
V*_D,i = (g_L·E_L + g_D·V_D,i) / (g_L + g_D)   -- "promised" potential (dendritic prediction)

f_mod(M) = (α_DA·[DA] + β_NE·[NE])              -- reward + urgency amplification
           · (1 - κ_eCB·[2-AG])                  -- endocannabinoid suppression
           · Φ_CORT([CORT])                       -- cortisol regime gate
           · H(E_i - θ_E)                         -- energy gate (Heaviside)
           · H(θ_S - S(t))                        -- sleep pressure gate

Φ_CORT([CORT]) = {
    (1 + α·[MR])           если [CORT] < θ_stress   -- MR-режим: facilitation
    (1 - β·[GR])           если [CORT] ≥ θ_stress   -- GR-режим: suppression
}
```

**Ключевое свойство:** Синапс «видит» только:
1. Пресинаптическую активность x_j (фактор 1)
2. Локальную ошибку (S_i - φ(V*_D,i)) = разницу между реальной сомой и дендритным предсказанием (фактор 2)
3. Глобальный химический модулятор f_mod, диффузно доступный (фактор 3)

Никакого обратного распространения ошибки. Никакого BPTT.

### 1.4 Нейрохимическая динамика (τ ~ 1 сек - часы)

**Газотрансмиттеры (NO, H₂S) — 3D диффузия:**
```
∂[NO]/∂t = D_NO · ∇²[NO] - k_clear·[NO] + S_nNOS(r,t)
```
Решение (Green's function): `C(r,t) = Q·(4πDt)^(-3/2)·exp(-r²/4Dt)·exp(-t/τ)`

**Нейропептиды и моноамины — объёмная передача с торtuosity:**
```
∂[M_k]/∂t = (D_k/λ²)·∇²[M_k] - V_max·[M_k]/(α·(K_m + [M_k])) + S_k(r,t)/α
```
λ ≈ 1.6 (tortuosity), α ≈ 0.20 (extracellular fraction)

**Ключевые модуляторные оси:**

| Модулятор | Источник | τ | Вычислительная функция |
|-----------|----------|---|----------------------|
| DA (Дофамин) | VTA/SNc | 100 мс phasic / мин tonic | RPE: δ = r + γV(s') - V(s) |
| NE (Норадреналин) | LC | 50-100 мс | Gain modulation, exploration |
| ACh (Ацетилхолин) | BF | 200 мс - сек | FF/FB баланс (encoding vs retrieval) |
| 5-HT (Серотонин) | Raphe | 500 мс - мин | Patience, temporal discount |
| Orexin | Lat. Hypothalamus | 10-60 сек | Wake attractor: g_i → g_i·(1+α[OX]) |
| Adenosine | Everywhere | мин - часы | Sleep pressure: dS/dt = α·A·W - β·S·(1-W) |
| Cortisol | Adrenal/systemic | 10 мин - 24 часа | Regime switch: hippocampus ↔ striatum |
| NO | nNOS neurons | 0.1-2 сек | Heterosynaptic LTP in 100μm sphere |
| 2-AG | Postsynaptic | 1-30 сек | Retrograde suppression, attractor ejection |
| Lactate | Astrocytes | мин - часы | Energy gate: Δw ≠ 0 ⟺ E > θ_E |

### 1.5 Гомеостатическая динамика (τ ~ секунды - минуты)

**Branching factor (критичность):**
```
σ̂ = Σ(s_t · s_{t+1}) / Σ(s_t²)    -- MR-estimator (Wilting-Priesemann)
```
Цель: σ → 1 (edge of chaos). Механизмы стабилизации:

**Синаптическая депрессия (SOC):**
```
du_syn/dt = (u_∞ - u_syn) / τ_rec
u_syn → u_syn · (1 - Δ)  при спайке
J_eff = J₀ · u_syn
```

**Intrinsic Plasticity (IP):**
```
ΔΘ_i = η_IP · (x_i - H_IP)     -- сдвиг порога к целевой частоте H_IP
```

**Ингибиторный STDP (Vogels 2011):**
```
На пре-спайк:  Δw_inh = η · (x_post - α)
На пост-спайк: Δw_inh = η · x_pre
α = 2 · ρ₀ · τ                   -- целевая частота ρ₀
```

Три механизма вместе: STDP + IP + weight normalization = SORN (Self-Organizing Recurrent Network),
удерживающий σ ≈ 1 и обеспечивающий power-law avalanche distributions:
```
P(S) ~ S^(-3/2), P(T) ~ T^(-2), <S>(T) ~ T²
```

### 1.6 Метапластический порог (BCM generalization)

```
θ_m(t) = θ₀ + γ_GC·[GC] - γ_E2·[E₂] - γ_H2S·[H₂S] + γ_Zn·[Zn²⁺] + γ_Mg·[Mg²⁺]
```

Этот порог определяет границу между LTP и LTD в BCM-подобном правиле. Каждый химический
агент сдвигает порог на своей временной шкале: Zn²⁺ за миллисекунды, H₂S за минуты,
Кортизол за часы, Эстрадиол за дни.

---

## Часть II: Спецификация Архитектуры (Blueprint)

### Глобальная топология

```
                        ┌─────────────────────────────────────────────┐
                        │           LEVEL 4: CONSCIOUSNESS            │
                        │  ┌───────────┐  ┌─────┐  ┌─────────────┐   │
                        │  │  Global    │  │ HOT │  │  Attention  │   │
                        │  │ Workspace  │←→│Meta │←→│   Schema    │   │
                        │  │(Mod.Hopf.) │  │Head │  │   (AST)     │   │
                        │  └─────┬─────┘  └──┬──┘  └──────┬──────┘   │
                        └────────┼───────────┼────────────┼───────────┘
         broadcast ↓            │  confidence│    gain    │
    ┌────────────────────────────┼───────────┼────────────┼────────────┐
    │              LEVEL 2: CORTEX (Predictive Coding Hierarchy)       │
    │                                                                   │
    │  ┌──────────┐    FF(γ)    ┌──────────┐    FF(γ)    ┌─────────┐  │
    │  │  SENS    │───────────→ │  ASSOC   │───────────→ │  PFC    │  │
    │  │  g≈0     │←───────────│  g≈0.5   │←───────────│  g≈0.9  │  │
    │  │ τ=20ms   │   FB(α/β)  │ τ=100ms  │   FB(α/β)  │ τ=1s    │  │
    │  └────┬─────┘            └────┬─────┘            └────┬─────┘  │
    │       │L5                     │L5                     │L5/L6   │
    └───────┼───────────────────────┼───────────────────────┼─────────┘
            │                       │                       │
    ┌───────┼───────────────────────┼───────────────────────┼─────────┐
    │       │    LEVEL 1: THALAMUS  │                       │         │
    │  ┌────▼─────┐            ┌────▼─────┐          ┌─────▼──────┐  │
    │  │ THAL-FO  │            │ THAL-HO  │          │    TRN     │  │
    │  │(LGN,MGN) │            │(Pulv,MD) │←─────────│  (GABA)    │  │
    │  └────┬─────┘            └────┬─────┘          └────────────┘  │
    │       │→L4 SENS               │→L4 other areas    ↑ L6 + AST  │
    └───────┼───────────────────────┼─────────────────────────────────┘
            ↑                       │
        SENSORY INPUT               │
                                    │
    ┌───────────────────────────────┼─────────────────────────────────┐
    │              LEVEL 3: MEMORY & MOTIVATION                       │
    │                                                                  │
    │  ┌──────────────────────────┐   ┌───────────────────────────┐   │
    │  │      HIPPOCAMPUS         │   │     BASAL GANGLIA         │   │
    │  │  EC → DG → CA3 → CA1    │   │  STR-D1(Go) ──→ GPi ──→  │   │
    │  │       ↑          │      │   │  STR-D2(NoGo)→GPe→STN↗    │──→ THAL
    │  │       └──────────┘      │   │  Cortex→STN (Hyperdirect)  │   │
    │  │  Replay: SWR 150-250Hz  │   │  DDM: a(t)=a₀+k·conflict  │   │
    │  └──────────────────────────┘   └───────────────────────────┘   │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │              NEUROMODULATORY CLOUD                        │   │
    │  │  DA(VTA) | NE(LC) | ACh(BF) | 5-HT(Raphe) | OX | ADO   │   │
    │  └──────────────────────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────────┘
```

---

### УРОВЕНЬ 1: Сенсорика и Таламус (Физика Восприятия)

#### 1.1 Модуль: SensoryEncoder

**Функция:** Превращение входного вектора (аналог «токена») в Sparse Distributed Representation (SDR).

**Биологическое обоснование:** Сенсорные кортексы (V1, A1, S1) используют sparse coding
(Olshausen-Field, 1996). Только ~2-5% нейронов активны одновременно. Это обеспечивает
максимальную информационную ёмкость при минимальном энергопотреблении.

**Механизм: LCA (Locally Competitive Algorithm, Rozell 2008)**

```
Вход:  x ∈ ℝ^d (raw input vector)
Выход: a ∈ ℝ^N (sparse activation, ~2-5% non-zero)

Инициализация:
  b = Φᵀ · x                    -- correlational response
  u = 0                          -- membrane potentials
  G = Φᵀ · Φ                    -- Gramian (lateral inhibition)

Inference loop (T_lca = 50-100 iterations):
  τ_lca · du/dt = b - u - (G - I)·a
  a = T_λ(u)                    -- soft threshold: a_i = sign(u_i)·max(|u_i| - λ, 0)

Dictionary learning (Hebbian):
  ΔΦ_i = α · (x - Φ·a) · a_i   -- residual error × activation
  Φ_i ← Φ_i / ||Φ_i||          -- normalize columns
```

**Параметры:**
- N = 2048 (SDR width), sparsity λ = 0.1, τ_lca = 10 мс
- Dictionary Φ ∈ ℝ^(d×N) — обучается Hebbian rule, порождает V1-подобные рецептивные поля

#### 1.2 Модуль: ThalamusFO (First-Order Relay)

**Функция:** Верный relay сенсорного входа в кору L4, с mode switching (tonic/burst).

**Биологическое обоснование:** LGN, MGN, VPL/VPM — first-order ядра (Sherman & Guillery).
В tonic mode: линейная передача. В burst mode: нелинейный alerting signal после паузы.

```
Входы:
  - SDR от SensoryEncoder (driver)
  - L6 modulatory feedback от SENS cortex
  - TRN inhibition
  
Выходы:
  - Спайковый паттерн → L4 SENS cortex (core projection)
  - Коллатерали → TRN

Уравнения:
  mode_i = { tonic  если [ACh] > θ_ACh ∨ [NE] > θ_NE
            { burst  если T_silence,i > T_burst_threshold

  gain_i = Π_k g_k,i    где g_k ∈ [0, 1]:
    g_L6  = σ(w_L6 · r_L6,i)           -- L6 modulatory gate
    g_TRN = 1 - σ(w_TRN · r_TRN,i)    -- TRN inhibitory gate
    g_ACh = (1 + α_ACh·[ACh]) / (1 + α_ACh·[ACh]₀)  -- ACh tonic→faithful

  output_i = gain_i · input_i · mode_factor_i
```

**Timing:** τ = 5-10 мс, latency 2-5 мс

#### 1.3 Модуль: ThalamicReticularNucleus (TRN)

**Функция:** Attentional gating — «прожектор» Крика. GABAergic shell, ингибирующий
таламические relay-нейроны. Контролируется L6 cortex и Attention Schema (Level 4).

**Биологическое обоснование:** Crick 1984, Halassa. Все таламокортикальные волокна
проходят через TRN. PFC может модулировать TRN-сектор, контролирующий relay к другой области
(open-loop organization, Willis 2015).

```
Входы:
  - Коллатерали от всех Thalamic relay нейронов
  - L6 corticothalamic projections (modulatory)
  - PFC top-down attention signal
  - AST gain control (Level 4)

Выходы:
  - GABA inhibition → Thalamic relay neurons (сектор-специфическая)

Уравнения:
  -- TRN sector s receives input from relay sector s and cortical sector c
  τ_TRN · dr_TRN,s/dt = -r_TRN,s + φ(Σ_relay w_sr·r_relay + Σ_cortex w_sc·r_L6,c + w_AST·gain_AST,s)
  
  -- Lateral inhibition within TRN (winner-take-most)
  I_lateral,s = -Σ_{s'≠s} w_lat · [r_TRN,s']+
  
  -- Output to relay
  I_TRN→relay,s = -g_GABA · r_TRN,s
```

**Timing:** τ_TRN = 20-50 мс

#### 1.4 Модуль: ThalamicHO (Higher-Order Relay)

**Функция:** Cortico-cortical communication через таламус. L5 одной кортикальной зоны
→ HO thalamus → L4 другой зоны. Winner-take-all selection.

**Биологическое обоснование:** Pulvinar, MD, VA/VL (Sherman & Guillery). L5 pyramidal tract
neurons проецируются как drivers в HO thalamus. MD ↔ PFC критичен для working memory
(Schmitt et al., 2017).

```
Входы:
  - L5 cortex (driver) — от ВСЕХ кортикальных зон
  - TRN inhibition (sector-specific)
  - Basal Ganglia output (GPi/SNr disinhibition)

Выходы:
  - Спайковый паттерн → L4 cortex (целевой зоны, выбранной WTA)
  - Collaterals → TRN

Уравнения:
  -- Competition (WTA)
  score_i = ||L5_input_i|| · gain_TRN,i · gate_BG,i
  winner = argmax(score_i + noise_i)
  
  -- Only winner relays
  output_winner = L5_input_winner · gain_winner
  output_others = 0
  
  -- BG gate (from GPi disinhibition):
  gate_BG,i = σ(-(GPi_tonic - GPi_inhibition_from_D1,i))
```

**Timing:** τ = 10-30 мс

#### 1.5 Критичность и Детекция Новизны

**Механизм:** Сеть на уровне таламо-кортикальных петель поддерживает branching factor σ ≈ 1
(Beggs-Plenz, 2003). Когда приходит действительно новый стимул, он вызывает avalanche
нетипичного размера (нарушение power-law distribution), что служит сигналом новизны.

```
Мониторинг критичности (непрерывный):
  σ̂(t) = MR_estimator(spike_history, k_max=20)
  
  Если σ̂ > 1.05:  -- supercritical → опасность эпилепсии
    → Увеличить g_GABA (ингибиторный STDP Vogels)
    → Увеличить пороги Θ (intrinsic plasticity)
    → Усилить синаптическую депрессию
    
  Если σ̂ < 0.90:  -- subcritical → опасность комы
    → Уменьшить g_GABA
    → Понизить пороги Θ
    → Уменьшить синаптическую депрессию
    
  Если avalanche_size > 3·σ̂_typical:
    → novelty_signal = True → повысить [NE] (LC burst)
    → TRN: открыть ворота для нового стимула
```

**Dynamic range (Kinouchi-Copelli):**
```
Δ = 10 · log₁₀(r_0.9 / r_0.1)    -- максимален при σ = 1
```

---

### УРОВЕНЬ 2: Кортикальная Колонка (Математика Обучения)

#### 2.1 Модуль: CorticalColumn

**Базовая единица:** Мультикомпартментный пирамидальный нейрон с 3 компартментами.

**Биологическое обоснование:** Urbanczik-Senn 2014 (Neuron), Sacramento et al. 2018 (NeurIPS),
Larkum 1999 (BAC-firing). Апикальный дендрит в L1 получает feedback, базальный в L4 получает
feedforward. Совпадение bAP + дистальный вход в окне ~5 мс → Ca²⁺-спайк → burst.

```
Архитектура одного нейрона i в зоне l:

  ┌─────────────────────────────────────────────────┐
  │            APICAL DENDRITE (L1)                  │
  │  V_A,i: получает FB от зоны l+1 (NMDA-доминант) │
  │  I_FB = Σ_k w_ik^FB · r_k^(l+1) · B(V_A, [Mg]) │
  │  + HO thalamic matrix projection                 │
  └──────────────────────┬──────────────────────────┘
                         │ g_A (apical coupling)
  ┌──────────────────────▼──────────────────────────┐
  │              SOMA (L2/3, L5)                     │
  │  V_S,i: интегрирует оба дендрита               │
  │  Спайкает когда V_S > Θ_i                       │
  │  Выход: S_i(t) ~ Poisson(φ(V_S,i))             │
  └──────────────────────▲──────────────────────────┘
                         │ g_D (basal coupling)
  ┌──────────────────────┴──────────────────────────┐
  │           BASAL DENDRITE (L4)                    │
  │  V_B,i: получает FF от зоны l-1 (AMPA-доминант) │
  │  I_FF = Σ_j w_ij^FF · r_j^(l-1)                │
  │  + FO thalamic core projection                   │
  └─────────────────────────────────────────────────┘

Interneuron types (per column):
  PV:  Fast-spiking, perisomatic inhibition → регулирует output rate
  SST: Dendritic inhibition (targets apical) → mediates feedback modulation
  VIP: Disinhibition (inhibits SST) → gate for feedback learning
```

#### 2.2 Предиктивное Кодирование (Rao-Ballard) Через Дендритные Компартменты

**Ключевая идея:** Ошибка предсказания вычисляется **внутри дендрита** как разница между
тем, что дендрит предсказывает (V*_D), и тем, что сома фактически делает (S_i).
Не нужны отдельные «error neurons» (подтверждается Westerberg et al. 2024).

```
Иерархия (3 зоны: SENS → ASSOC → PFC):

Зона l предсказывает активность зоны l-1:
  prediction_l→(l-1) = f_l(r_l)           -- нисходящее предсказание (FB connection)
  
Ошибка (вычисляется в базальном дендрите зоны l-1):
  e_(l-1) = r_(l-1) - f_l(r_l)           -- prediction error

Обновление representations (inference, τ_infer ~ 10-30 шагов):
  τ · dr_l/dt = -e_l + f'_l^T · W_(l+1)^T · e_(l+1)

Обновление весов (learning, local Hebbian):
  ΔW_l = α · e_l · r_(l-1)^T / σ_l²
```

**В терминах компартментов Urbanczik-Senn:**

```
-- Апикальный дендрит получает prediction от уровня l+1
V_A = FB prediction

-- Базальный дендрит получает input от уровня l-1
V_B = FF input

-- Prediction error = soma reality - dendritic prediction
error_i = S_i(t) - φ(V*_D,i)

-- V*_D = what dendrite "promises" to soma
V*_D,i = (g_L·E_L + g_D·V_B,i) / (g_L + g_D)

-- Weight update (THREE factors, ALL local):
Δw_ij = η · error_i · x_j · f_mod(M(t))
         ↑ factor 2   ↑ factor 1   ↑ factor 3
      (post-error)  (pre-activity) (neuromodulation)
```

**Эквивалентность Backprop:** Whittington-Bogacz доказали, что при (a) линеаризованной f,
(b) transpose-linked feedback weights, (c) сходимости inference — стационарные ошибки e_l
пропорциональны дельтам backprop. Genesis аппроксимирует это через Sacramento microcircuit
с SST-интернейронами, вычитающими предсказуемую часть feedback.

#### 2.3 Кортикальный Градиент

**Биологическое обоснование:** Margulies et al. 2016 — principal gradient G1 от сенсорной
коры к DMN/PFC. Временная константа растёт экспоненциально вдоль градиента.

```
τ(g) = τ_min · (τ_max/τ_min)^g

где g ∈ [0, 1] — позиция на градиенте:
  g = 0.0: SENS  → τ = 20 мс  (fast, input-driven, AMPA-dominated)
  g = 0.5: ASSOC → τ = 100 мс (intermediate, mixed AMPA/NMDA)
  g = 0.9: PFC   → τ = 1-2 сек (slow, recurrent, NMDA-dominated, persistent activity)
```

**Молекулярный базис (Burt et al. 2018):** Сенсорные зоны: высокая миелинизация, быстрая
AMPA-передача. Ассоциативные зоны: низкая миелинизация, сильный NMDA, высокая рекуррентность.

#### 2.4 Нейростероидное Переключение Пластичности

**Кортизол (MR/GR dual receptor):**

```
При низком стрессе ([CORT] < θ₁):
  MR-рецепторы (высокая аффинность) активны
  → Facilitation LTP в гиппокампе и PFC
  → η_hipp = η₀ · (1 + γ₁·[MR_active])
  → Гибкое обучение (декларативная память)

При высоком стрессе ([CORT] > θ₁):
  GR-рецепторы (низкая аффинность) активируются
  → Блокировка LTP в CA1/PFC, promotion LTD
  → Facilitation LTP в amygdala и dorsal striatum
  → η_hipp = η₀ · (1 - γ₂·[GR_active])
  → η_habit = η₀ · (1 + γ₃·[GR_active])
  → "Survival mode": habits вместо flexible reasoning

Cortisol → eCB synthesis:
  Быстрый non-genomic эффект через eCB как executor:
  [CORT] → mGR → DAGLα → [2-AG] ↑ → retrograde suppression
  Гормон часовой шкалы «нанимает» модулятор секундной шкалы
```

**VIP → SST disinhibition gate:**
```
[VIP interneuron] ─┤SST interneuron├─ apical dendrite
                                        (feedback gate)

Когда VIP активен (driven by modulatory input):
  SST подавлен → apical dendrite освобождён → feedback learning enabled
  g_GABA,SST→Pyr = g₀ / (1 + ([VIP]/K_d)^n)
```

---

### УРОВЕНЬ 3: Память и Мотивация (Химия)

#### 3.1 Модуль: Hippocampus

**Архитектура: Trisynaptic Circuit**

```
                    ┌────────────────────────────────────────┐
                    │          ENTORHINAL CORTEX (EC)         │
                    │  MEC (grid cells) + LEC (objects)       │
                    │  Input: ASSOC/PFC cortex                │
                    └──┬──────────┬───────────────┬──────────┘
                       │ L-II     │ L-II (direct)  │ L-III
                       ▼          ▼                ▼
                    ┌──────┐   ┌──────┐       ┌──────────┐
                    │  DG  │   │      │       │   CA1     │
                    │ 1-2% │──→│ CA3  │──────→│ Comparator│──→ EC deep → cortex
                    │active│   │ Auto-│       │ CA3 vs EC │    │
                    │sparse│   │assoc.│       └──────────┘    │
                    └──────┘   │Hopf. │                       │
                     mossy     │recur.│       Mismatch? ──→ VTA (DA burst)
                    (strong)   └──────┘                  ──→ LC (NE burst)
```

**DG (Dentate Gyrus) — Pattern Separation:**
```
Входы: EC Layer II (perforant path)
Выходы: CA3 (mossy fibers — strong but sparse, "detonator synapses")

-- Sparse coding: SDR с ~1-2% активных
r_DG = LCA(EC_input, Φ_DG, λ=high)

-- Adult neurogenesis: новые нейроны с низким порогом и широкими RF
-- Помогают отделять новые паттерны от старых (Aimone et al., 2011)
Каждые N_neurogenesis шагов:
  → Заменить старейший нейрон DG новым с randomized weights
  → Новый нейрон: Θ_new = Θ_base - ΔΘ_young (более возбудим)
```

**CA3 — Autoassociative Pattern Completion (Hopfield):**
```
Входы:
  - DG (mossy fibers — учительский/деtonator сигнал)
  - EC Layer II (perforant path — слабый index)
  - CA3 recurrent collaterals (autoassociation)

-- Energy function (Classical Hopfield):
E = -(1/2) · Σ_ij w_ij · s_i · s_j

-- Storage (Hebbian outer product):
w_ij += (1/N) · ξ_i^μ · ξ_j^μ    при каждом новом паттерне μ

-- Retrieval (async update):
s_i ← sign(Σ_j w_ij · s_j)

-- NMDA-dependent LTP at recurrent synapses
-- Storage capacity: ~0.14·N (Amit-Gutfreund-Sompolinsky)

-- ONE-SHOT learning: высокий η в CA3
η_CA3 = 10 · η_cortex    (complementary learning systems, McClelland 1995)
```

**CA1 — Mismatch Detector / Comparator:**
```
Входы:
  - CA3 (Schaffer collaterals — retrieved memory)
  - EC Layer III (temporoammonic pathway — current sensory input)

-- Comparison:
mismatch = ||CA3_output - EC_III_input||²

Если mismatch > θ_novelty:
  → Signal to VTA → DA burst (Lisman & Grace, 2005)
  → Signal to LC → NE burst (unexpected uncertainty)
  → Tag current episode as "surprising" → priority replay
```

**Sharp-Wave Ripples (SWR) — Replay:**
```
Условие: [ACh] < θ_ACh_low  (sleep / quiet wakefulness)
         AND  [Adenosine] > θ_ADO  (sleep pressure accumulated)

Процесс (каждые ~125 мс во время SWR):
  1. CA3 спонтанно активирует stored pattern
  2. Pattern propagates: CA3 → CA1 → EC deep → cortex
  3. Cortical weights обновляются с малым η:
     ΔW_cortex = η_slow · e_replay · r_replay^T
  4. Priority sampling:
     P(replay_μ) ∝ |error_μ| + β · uncertainty_μ
     (confident errors replayed first — from HOT module)
  
  5. Каждые N_replay циклов → cortex накапливает statistics
     → Slow cortical consolidation (CLS: McClelland et al. 1995)
```

#### 3.2 Модуль: VolumeTransmission (NO, 2-AG)

**NO — Ретроградный объёмный координатор:**
```
Источник: nNOS (calcium/calmodulin-dependent)
Trigger: Высокая [Ca²⁺] в постсинапсе (NMDA-dependent)

-- 3D diffusion (isotropic gas, no membrane barrier):
∂[NO]/∂t = D_NO · ∇²[NO] - k_clear·[NO] + S_nNOS(r,t)
D_NO ≈ 3300 μm²/s, τ_NO ≈ 0.1-2 s, radius ≈ 100 μm

-- Green's function (instantaneous point source Q):
[NO](r,t) = Q · (4πD·t)^(-3/2) · exp(-r²/(4D·t)) · exp(-t/τ)

-- Heterosynaptic LTP:
-- ALL synapses within NO cloud that are also pre-active get strengthened
ΔP_pre,i = β_NO · ∫[NO](r_i, t) · act_pre,i(t) dt

-- Computational function:
-- NO "permits" enhancement: Hebb rule + spatial averaging
-- Связывает синапсы без прямого контакта в радиусе 100 мкм
-- → Co-active neurons in the cloud become associated
```

**2-AG — Ретроградная нормализация и "выбивание из аттрактора":**
```
Источник: DAGLα (постсинаптический, activity-dependent)
Trigger: Sustained [Ca²⁺] postsynaptic + mGluR1/5 activation

-- Synthesis:
d[2-AG]/dt = k_synth · σ([Ca²⁺]_post - θ_Ca) - k_MAGL · [2-AG]

-- Retrograde suppression of release:
P_release(t) = P₀ · (1 - [2-AG]^n / ([2-AG]^n + IC₅₀^n))

-- ATTRACTOR EJECTION mechanism:
-- The longer network stays in attractor, the more 2-AG accumulates
-- → Presynaptic fatigue increases → attractor weakens → network transitions
τ_x · dx/dt = -x + F(W·x + I) - δ · x · ∫[eCB](t-s)ds
                                    ↑ fatigue term

-- This prevents "obsessive" perseveration in one state
-- Ensures flexible transitions between representations
-- Temporal scale: seconds → natural rhythm of thought transitions
```

**H₂S → D-serine → NMDA pipeline:**
```
-- Gas-to-gliotransmission bridge
H₂S/polysulfides → TRPA1 (astrocyte) → Ca²⁺ wave → D-serine release
→ NMDA receptor potentiation at neighboring synapses
→ Metaplastic threshold shift: θ_LTP → θ_LTP - γ·[H₂S]

-- Astrocyte acts as "relay" extending gas signal to unconnected synapses
```

#### 3.3 Модуль: BasalGanglia (Action Selection / DDM)

**Архитектура:** Go/NoGo/Stop pathways + Drift-Diffusion Model

```
         Cortex L5
           │
    ┌──────┴──────┐────────────────────┐
    ▼             ▼                    ▼
 ┌──────┐    ┌──────┐            ┌─────────┐
 │D1-MSN│    │D2-MSN│            │   STN   │ ← HYPERDIRECT
 │ (Go) │    │(NoGo)│            │ (Stop)  │   (fastest: 5-10ms)
 └──┬───┘    └──┬───┘            └────┬────┘
    │           ▼                     │
    │       ┌──────┐                  │
    │       │ GPe  │                  │
    │       └──┬───┘                  │
    │          ▼                      │
    ▼       ┌──────┐    ◄─────────────┘
 ┌──────┐   │ STN  │
 │GPi/  │◄──┤      │
 │SNr   │   └──────┘
 │(gate)│
 └──┬───┘
    │ INHIBITION (tonic)
    ▼
 THALAMUS (VA/VL, MD) → Cortex
 (disinhibited when GPi suppressed by D1-Go)
```

**DDM Implementation:**
```
-- Drift rate per action candidate i:
v_i(t) = w_Go,i - w_NoGo,i

-- Accumulation:
dx_i/dt = v_i + noise(σ)

-- Adaptive threshold (STN-mediated):
a(t) = a₀ + k_STN · conflict(t)
conflict = entropy(softmax(v))    -- ACC signal

-- Decision when: x_winner > a(t)
-- Hyperdirect: STN global inhibition arrives 100-150ms before action
-- → prevents premature commitment ("pause to think")

-- DA modulation (RPE-based RL):
δ = r + γ·V(s') - V(s)           -- temporal difference error

Δw_Go,i  = α · δ · e_i · [D1]   -- positive RPE → strengthen Go
Δw_NoGo,i = α · (-δ) · e_i · [D2]  -- negative RPE → strengthen NoGo

-- Tonic DA: baseline Go/NoGo balance (average reward rate, Niv 2007)
-- High tonic DA → bias toward Go (vigorous, fast decisions)
-- Low tonic DA → bias toward NoGo (Parkinsonian hesitation)
```

**Orexin/Adenosine Modulation:**
```
-- Orexin (wake/motivation):
[OX] > θ_wake → network in wake-attractor
  → BG threshold lowered: a₀ ← a₀ · (1 - α_OX·[OX])
  → Decisions faster, more impulsive when motivated
  → Gain of reward signal: δ_eff = δ · (1 + β_OX·[OX])

-- Adenosine (fatigue/sleep pressure):
dS/dt = α · Activity(t) · W(t) - β · S · (1 - W(t))
  Когда S > θ_sleep:
    → BG threshold raised: a₀ ← a₀ · (1 + α_ADO·[ADO])
    → Decisions slower, more conservative
    → [OX] drops → sleep attractor collapse
    → Switch to SWR replay mode (hippocampus)

-- A2A-D2 heteromer in striatum:
-- Adenosine A2A antagonizes D2 → energy deficit dampens action selection
g_D2_eff = g_D2 / (1 + [A2A_bound] / K_het)
```

---

### УРОВЕНЬ 4: Сознание (Интеграция)

#### 4.1 Модуль: GlobalWorkspace (Modern Hopfield Network)

**Функция:** Bottleneck: из множества кортикальных representations → выбор K winners →
global broadcast всем модулям. Write дорогой (competitive), Read дешёвый (broadcast).

**Биологическое обоснование:** Dehaene & Changeux — neuronal GWT: thalamo-cortical long-range
pyramidal neurons. «Ignition» = nonlinear all-or-nothing amplification.

**Ключевой выбор: Modern Hopfield вместо стандартного Attention.**

**Обоснование:** Modern Hopfield Network (Ramsauer et al. 2021) — это energy-based model,
где retrieval = minimization of energy. Attention — частный случай (1-step retrieval).
Modern Hopfield даёт **exponential capacity** по dimensionality ключей, и его energy landscape
порождает аттракторную динамику: content "залипает" в workspace пока не будет вытеснено
более сильным сигналом. Это именно то поведение, которое нужно для GWT: ignition → sustained
activity → broadcast → competition → next ignition.

```
Входы (candidates for GW write):
  - r_SENS: стабилизированные representations от SENS cortex
  - r_ASSOC: от ASSOC cortex  
  - r_PFC: от PFC (top-down predictions)
  - r_HPC: от hippocampal CA1 (retrieved memories)
  - r_BG: от Basal Ganglia (selected action representations)

GW State: ξ ∈ ℝ^d_GW (d_GW << Σ d_modules — информационный bottleneck)

-- Stored patterns (current candidates):
X = [r_SENS; r_ASSOC; r_PFC; r_HPC; r_BG]^T ∈ ℝ^(M × d)

-- Modern Hopfield energy:
E = -lse(β_GW, X^T · ξ) + (1/2)||ξ||² + constants

-- Retrieval dynamics (iterative, converges in 1-3 steps):
ξ_new = X · softmax(β_GW · X^T · ξ)

-- β_GW controls ignition steepness (γ_ign):
-- Low β → soft mixing (unfocused, "dreamy")
-- High β → hard winner-take-all (focused, "ignition")

-- Ignition threshold (sigmoid access):
A_GW(S) = 1 / (1 + exp(-γ_ign · (S - S_th)))
S = max(X^T · ξ)    -- strength of best candidate

-- Phase transition: only candidates with S > S_th enter GW
-- Below threshold: processed unconsciously (feedforward sweep)
-- Above threshold: ignition → broadcast

-- BROADCAST (one-to-many, cheap read):
∀ module m: input_m += W_broadcast,m · ξ
-- Every module receives current GW content as context
```

**Capacity:** K = 4 slots (Cowan 2001), d_GW = 256-512

**Timing:** Ignition latency ~250 мс (late fronto-parietal amplification),
sustained for hundreds of ms until competition replaces content.

#### 4.2 Модуль: MetaHead (Higher-Order Thought)

**Функция:** "Мысль о мысли" — оценка уверенности, предсказание собственной ошибки,
разделение алеаторической и эпистемической неопределённости.

**Биологическое обоснование:** Rosenthal 2005 — HOT theory. Frontopolar cortex и anterior
cingulate. Meta-d' ≠ d' (Fleming & Lau 2014): метакогнитивная чувствительность измеряется
независимо от точности первого порядка.

```
Входы:
  - ξ (GW content — что мы "осознаём")
  - v_BG (proposed action from DDM)
  - router_state (gating weights)
  - h_cortex (hidden states from cortical hierarchy — fast weights)
  - ξ_meta_prev (собственный предыдущий выход — self-loop)

Выходы:
  - confidence ∈ [0, 1] (per action, per expert)
  - L̂ (predicted loss — expected error)
  - surprise = -log P(current_input | model)
  - novelty = distance to nearest stored pattern in CA3

-- Architecture: 2-layer network reading GW + own loop
-- NOT a separate homunculus — "single-level" variant where state
-- partially represents itself

-- Confidence computation:
confidence = σ(W_conf · [ξ; v_BG; h_cortex; ξ_meta_prev] + b_conf)

-- Error prediction:
L̂ = W_err · [ξ; h_cortex] + b_err

-- Training (Proper Scoring Rule — Brier loss, NOT accuracy):
L_meta = (L̂ - L_real)² + λ_brier · Brier(confidence, outcome)
Brier(p, y) = (p - y)²

-- M-ratio metric:
M_ratio = meta_d' / d'    -- metacognitive efficiency
-- Target: M_ratio ≈ 1 (perfect metacognition)

-- Epistemic vs Aleatoric decomposition:
-- Run GW retrieval N_ensemble times with noise
-- Aleatoric = E[variance within trials]
-- Epistemic = variance of means across trials
```

**Confidence-Guided Behavior (КРИТИЧЕСКИ ВАЖНО):**
```
-- Confidence MUST affect behavior, otherwise it degenerates into empty report

1. Action gating:
   Если confidence < θ_low:
     → Inhibit BG action (STN threshold ↑)
     → Request more information (orient TRN to relevant modality)
     → Lower learning rate (don't learn from uncertain signal)

2. Replay priority:
   P(replay_μ) ∝ |error_μ| + β · epistemic_uncertainty_μ
   -- "Confident errors" get highest replay priority

3. Metamemory (Feeling of Knowing):
   Если epistemic_uncertainty > θ_epistemic:
     → Query hippocampus CA3 for associative retrieval
     → Only access external memory when "I know that I don't know"
```

#### 4.3 Модуль: AttentionSchema (AST)

**Функция:** Упрощённая модель собственного внимания — "что я обрабатываю, почему,
и с каким усилением". Нужна для **управления** вниманием, не просто его осуществления.

**Биологическое обоснование:** Graziano 2013 — Attention Schema Theory. Сознание = схема
внимания, примененная к собственной системе. "Я осознаю X" = "моя схема внимания
сообщает, что я обрабатываю X".

```
Входы:
  - Efference copy of GW gating weights (кто выиграл конкуренцию и почему)
  - TRN state (текущий аттенциональный фильтр)
  - HOT confidence (из MetaHead)

Выходы (low-dimensional explicit state):
  - current_focus: {module_id, topic_embedding}
  - gain: ℝ^M (amplification per module)
  - expected_accuracy: ℝ (how well will attention serve the task)
  - reason: embedding (why this was selected)
  - attention_prediction: predicted next attention state

-- Architecture: reads GW gating state → builds compressed map
-- Sends CORRECTIVE SIGNAL back to TRN and GW competition

-- Attention Prediction Loss:
L_schema = ||attention_predicted(t-1) - attention_actual(t)||²

-- Gain control feedback to TRN:
Δgain_TRN,s = α_AST · (expected_accuracy - actual_accuracy) · ∂focus/∂gain_s

-- Credit assignment:
-- Schema error localizes credit: if behavior deviated from planned attention,
-- the ROUTER is corrected, not everything.
-- "I didn't expect to switch to this" → correct the gating, not the content.

-- Introspective interface (functional report):
report = {
  "processing": current_focus.topic,
  "because": reason.decode(),
  "confidence": HOT.confidence,
  "difficulty": 1 - expected_accuracy
}
-- NOTE: This is a FUNCTIONAL report, NOT a claim about phenomenal experience
```

#### 4.4 Замкнутый Цикл Обработки (per GW tick, ~250 мс)

```
(a) Expert encoding: Cortical zones stabilize representations (recurrent, 100-150ms)
(b) Candidate scoring: Each zone's L5 output + HO thalamic relay → GW candidates
(c) GW competition: Modern Hopfield retrieval (1-3 steps) → top-K winners
(d) GW ignition: Winners amplified, losers decay (sigmoid threshold)
(e) Broadcast: GW content → all modules (cheap read)
(f) HOT monitoring: MetaHead evaluates confidence, error, surprise
(g) AST update: Attention Schema updates self-model
(h) Top-down control: AST → TRN gain change, HOT → BG threshold change
    → Loop closes: next competition is shaped by current consciousness cycle
```

**Uncertainty → Thalamus Feedback Loop:**
```
Когда HOT.confidence < θ_uncertain:
  1. AST identifies which sensory modality is most uncertain
  2. AST sends gain_boost to TRN sector for that modality
  3. TRN opens gate → more sensory information flows to cortex
  4. Cortex updates representation → GW gets new candidate
  5. HOT re-evaluates confidence
  → ACTIVE PERCEPTION: system seeks information to resolve uncertainty
```

---

## Часть III: Псевдокод Ядра

```python
class GenesisOrganism:
    """
    NEUROWEAVE v10.0 Genesis — Virtual Biology Engine
    Main simulation loop: 1 tick = 1 ms of biological time
    """
    
    def __init__(self, config: GenesisConfig):
        # ═══════════════════════════════════════════════
        # LEVEL 1: Sensorics & Thalamus
        # ═══════════════════════════════════════════════
        self.sensory_encoder = LCASparseEncoder(
            input_dim=config.input_dim,
            sdr_width=2048,
            sparsity=0.1,
            tau_lca=10.0   # ms
        )
        self.thalamus_fo = ThalamusFO(
            n_nuclei=config.n_sensory_modalities,
            n_neurons_per=8
        )
        self.thalamus_ho = ThalamicHO(
            n_neurons=10,
            wta_noise=0.01
        )
        self.trn = TRN(
            n_sectors=config.n_sensory_modalities + config.n_cortical_zones,
            tau=30.0  # ms
        )
        
        # ═══════════════════════════════════════════════
        # LEVEL 2: Cortex (Predictive Coding Hierarchy)
        # ═══════════════════════════════════════════════
        self.cortex = CorticalHierarchy(
            zones=[
                CorticalZone("SENS",  g=0.0, tau=20,   n_columns=256, n_interneurons=64),
                CorticalZone("ASSOC", g=0.5, tau=100,  n_columns=512, n_interneurons=128),
                CorticalZone("PFC",   g=0.9, tau=1000, n_columns=256, n_interneurons=64),
            ],
            neuron_model=UrbanczikSennNeuron(
                C_m=1.0, g_L=0.05, E_L=-70.0,
                g_D=0.1, g_A=0.05,  # basal, apical coupling
                phi_max=100.0, beta=1.0, theta=-50.0
            )
        )
        
        # ═══════════════════════════════════════════════
        # LEVEL 3: Memory & Motivation
        # ═══════════════════════════════════════════════
        self.hippocampus = Hippocampus(
            ec=EntorhinalCortex(n=10, tau=75),
            dg=DentateGyrus(n=10, sparsity=0.02, neurogenesis_rate=0.001),
            ca3=CA3Autoassociative(n=10, tau_recurrent=150, eta=0.01),
            ca1=CA1Comparator(n=10, tau=50, novelty_threshold=0.3)
        )
        self.basal_ganglia = BasalGanglia(
            n_actions=config.n_actions,
            stn_gain=0.5,
            da_baseline=0.5,
            threshold_base=1.0
        )
        
        # ═══════════════════════════════════════════════
        # LEVEL 4: Consciousness
        # ═══════════════════════════════════════════════
        self.global_workspace = GlobalWorkspace(
            d_gw=256,
            n_slots=4,
            beta_gw=8.0,          # ignition steepness
            ignition_threshold=0.5
        )
        self.meta_head = MetaHead(
            d_input=256 + config.n_actions + 128,  # GW + action + cortex
            d_hidden=128
        )
        self.attention_schema = AttentionSchema(
            n_modules=5,  # SENS, ASSOC, PFC, HPC, BG
            d_schema=32
        )
        
        # ═══════════════════════════════════════════════
        # NEUROMODULATORY CLOUD
        # ═══════════════════════════════════════════════
        self.chemistry = NeuromodulatorySystem(
            dopamine=DASystem(tau_phasic=0.1, tau_tonic=60.0),       # VTA/SNc
            norepinephrine=NESystem(tau=0.075),                       # LC
            acetylcholine=AChSystem(tau=0.5),                         # BF
            serotonin=HTSystem(tau=2.0),                              # Raphe
            orexin=OrexinSystem(tau=30.0),                            # LH
            adenosine=AdenosineSystem(tau_accumulate=3600.0),         # global
            cortisol=CortisolSystem(tau_nongenomic=600, tau_genomic=43200),
            nitric_oxide=NOField(D=3300, tau=1.0, radius=100),       # μm²/s
            endocannabinoid=ECBSystem(tau_phasic=10.0, tau_tonic=120.0),
            lactate=LactateShuttle(tau=1800.0)                        # astrocyte
        )
        
        # ═══════════════════════════════════════════════
        # HOMEOSTASIS
        # ═══════════════════════════════════════════════
        self.homeostasis = HomeostaticController(
            target_sigma=1.0,
            sigma_tolerance=0.05,
            ip_rate=0.001,          # intrinsic plasticity
            inh_stdp_rate=0.0001,   # inhibitory STDP (Vogels)
            depression_tau=500.0     # synaptic depression recovery (ms)
        )
        
        # Timing
        self.dt = 1.0     # ms
        self.t = 0.0      # current time
        self.is_awake = True
        
    def step(self, sensory_input: Tensor) -> GenesisOutput:
        """
        ONE TICK of biological time (dt = 1 ms).
        All levels execute in parallel with their own time constants.
        """
        self.t += self.dt
        chem = self.chemistry.get_concentrations()
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 0: NEUROMODULATORY UPDATE                  ║
        # ║  (slowest dynamics, but affects everything below)  ║
        # ╚═══════════════════════════════════════════════════╝
        
        # Sleep pressure integration
        self.chemistry.adenosine.integrate(
            activity=self.cortex.mean_firing_rate(),
            is_awake=self.is_awake,
            dt=self.dt
        )
        
        # Orexin vs Adenosine: wake/sleep decision
        if chem.adenosine > self.chemistry.adenosine.sleep_threshold:
            if chem.orexin < self.chemistry.orexin.collapse_threshold:
                self.is_awake = False  # sleep onset
        
        # Cortisol regime (MR vs GR)
        cort_regime = self.chemistry.cortisol.get_regime()  # "MR" or "GR"
        
        # Compute modulatory factor for plasticity
        f_mod = self.chemistry.compute_plasticity_gate(chem, cort_regime)
        # = (α_DA·[DA] + β_NE·[NE]) · (1 - κ·[2-AG]) · Φ_CORT · H(E-θ_E) · H(θ_S-S)
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 1: SENSORICS & THALAMIC GATING            ║
        # ║  τ ~ 1-10 ms                                      ║
        # ╚═══════════════════════════════════════════════════╝
        
        if self.is_awake:
            # Sparse encode sensory input (LCA)
            sdr = self.sensory_encoder.step(sensory_input, dt=self.dt)
            
            # TRN gating (attention filter)
            trn_inhibition = self.trn.step(
                relay_collaterals=self.thalamus_fo.collaterals,
                cortical_L6=self.cortex.get_L6_output("SENS"),
                ast_gain=self.attention_schema.get_gain(),
                dt=self.dt
            )
            
            # First-order relay (sensory → L4 SENS)
            fo_output = self.thalamus_fo.step(
                driver=sdr,
                L6_modulation=self.cortex.get_L6_output("SENS"),
                trn_inhibition=trn_inhibition,
                ach=chem.acetylcholine,
                ne=chem.norepinephrine,
                dt=self.dt
            )
            
            # Higher-order relay (L5 cortex → L4 other zones)
            ho_output = self.thalamus_ho.step(
                L5_drivers={
                    zone.name: zone.get_L5_output()
                    for zone in self.cortex.zones
                },
                trn_inhibition=trn_inhibition,
                bg_gate=self.basal_ganglia.get_thalamic_gate(),
                dt=self.dt
            )
        else:
            fo_output = None
            ho_output = None
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 2: CORTICAL PROCESSING                     ║
        # ║  τ ~ 20-1000 ms (gradient-dependent)              ║
        # ╚═══════════════════════════════════════════════════╝
        
        # Each zone processes with its own time constant
        cortical_outputs = {}
        prediction_errors = {}
        
        for zone in self.cortex.zones:
            # Feedforward input (from lower zone or thalamus)
            if zone.name == "SENS":
                ff_input = fo_output
            else:
                lower = self.cortex.get_lower_zone(zone)
                ff_input = lower.get_L23_output()  # FF: L2/3 → L4
            
            # Feedback input (from higher zone via apical)
            higher = self.cortex.get_higher_zone(zone)
            fb_input = higher.get_L56_output() if higher else None
            
            # Transthalamic input (HO relay)
            ho_input = ho_output.get(zone.name) if ho_output else None
            
            # GW broadcast (from consciousness level)
            gw_broadcast = self.global_workspace.get_broadcast()
            
            # Step each neuron (Urbanczik-Senn dynamics)
            zone_output, zone_error = zone.step(
                ff_input=ff_input,       # → basal dendrite
                fb_input=fb_input,       # → apical dendrite
                ho_input=ho_input,       # → L4 (transthalamic)
                gw_context=gw_broadcast, # → apical (global context)
                mg_concentration=chem.magnesium,
                ach=chem.acetylcholine,   # FF/FB balance
                dt=self.dt
            )
            
            cortical_outputs[zone.name] = zone_output
            prediction_errors[zone.name] = zone_error
            
            # LOCAL WEIGHT UPDATE (Urbanczik-Senn + Rao-Ballard)
            if f_mod > 0:  # plasticity gate open
                zone.update_weights(
                    error=zone_error,     # S(t) - φ(V*_D)
                    f_mod=f_mod,          # chemical modulation
                    eta=zone.eta * self.chemistry.cortisol.learning_rate_factor(
                        cort_regime, zone.name
                    ),
                    dt=self.dt
                )
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 3: MEMORY & MOTIVATION                     ║
        # ║  τ ~ 50 ms - hours                                ║
        # ╚═══════════════════════════════════════════════════╝
        
        # --- HIPPOCAMPUS ---
        if self.is_awake:
            # Encoding path: cortex → EC → DG → CA3 → CA1
            ec_input = self.cortex.zones_by_name["ASSOC"].get_L23_output()
            
            hpc_output = self.hippocampus.step_encode(
                ec_input=ec_input,
                ach=chem.acetylcholine,  # high ACh → encoding mode
                theta_phase=self.hippocampus.theta_oscillator(self.t),
                dt=self.dt
            )
            
            # CA1 mismatch detection
            if hpc_output.mismatch > self.hippocampus.ca1.novelty_threshold:
                self.chemistry.dopamine.phasic_burst(magnitude=hpc_output.mismatch)
                self.chemistry.norepinephrine.phasic_burst(magnitude=0.5)
                
        else:
            # SLEEP: Sharp-Wave Ripple replay
            if self.hippocampus.should_ripple(self.t):
                replay_pattern, replay_priority = self.hippocampus.step_replay(
                    confidence_errors=self.meta_head.get_replay_priorities(),
                    dt=self.dt
                )
                # Slow cortical consolidation
                self.cortex.consolidate(
                    replay_pattern=replay_pattern,
                    eta_slow=0.0001,
                    dt=self.dt
                )
        
        # --- VOLUME TRANSMISSION ---
        # NO field: heterosynaptic coordination
        self.chemistry.nitric_oxide.diffuse(
            sources=self.cortex.get_high_calcium_sites(),
            dt=self.dt
        )
        no_field = self.chemistry.nitric_oxide.get_field()
        self.cortex.apply_no_modulation(no_field)
        
        # 2-AG: retrograde suppression + attractor ejection
        self.chemistry.endocannabinoid.update(
            postsynaptic_calcium=self.cortex.get_calcium_levels(),
            mglur_activation=self.cortex.get_mglur_activation(),
            dt=self.dt
        )
        self.cortex.apply_ecb_modulation(
            ecb_levels=self.chemistry.endocannabinoid.get_levels()
        )
        
        # --- BASAL GANGLIA ---
        if self.is_awake:
            action, bg_info = self.basal_ganglia.step(
                cortical_L5={z.name: z.get_L5_output() for z in self.cortex.zones},
                dopamine_tonic=chem.dopamine_tonic,
                dopamine_phasic=chem.dopamine_phasic,
                orexin=chem.orexin,
                adenosine=chem.adenosine,
                serotonin=chem.serotonin,  # patience
                dt=self.dt
            )
        else:
            action = None
            bg_info = None
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 4: CONSCIOUSNESS                           ║
        # ║  τ ~ 250+ ms (GW cycle)                           ║
        # ╚═══════════════════════════════════════════════════╝
        
        # Collect stabilized candidates
        candidates = self._collect_gw_candidates(
            cortical_outputs, hpc_output if self.is_awake else None, bg_info
        )
        
        # Global Workspace: Modern Hopfield retrieval
        gw_content, gw_ignited = self.global_workspace.step(
            candidates=candidates,
            beta=self._compute_beta(chem),  # NE modulates ignition steepness
            dt=self.dt
        )
        
        if gw_ignited:
            # HOT: Meta-cognitive evaluation
            meta_output = self.meta_head.step(
                gw_content=gw_content,
                proposed_action=action,
                cortex_hidden=self.cortex.get_hidden_states(),
                dt=self.dt
            )
            
            # AST: Attention schema update
            schema_output = self.attention_schema.step(
                gw_gating_weights=self.global_workspace.get_gating_weights(),
                trn_state=self.trn.get_state(),
                hot_confidence=meta_output.confidence,
                dt=self.dt
            )
            
            # ═══ FEEDBACK LOOP: Consciousness → Thalamus ═══
            
            # Uncertainty → TRN modulation
            if meta_output.confidence < self.meta_head.uncertainty_threshold:
                uncertain_modality = schema_output.most_uncertain_module()
                self.trn.boost_gate(
                    sector=uncertain_modality,
                    gain_delta=schema_output.gain_correction
                )
            
            # Confidence → BG threshold
            if meta_output.confidence < self.meta_head.low_confidence_threshold:
                self.basal_ganglia.raise_threshold(
                    delta=0.2 * (1 - meta_output.confidence)
                )
            
            # Schema error → router credit assignment
            self.global_workspace.update_router(
                schema_error=schema_output.prediction_error,
                eta_router=0.0001,
                dt=self.dt
            )
        else:
            meta_output = None
            schema_output = None
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  PHASE 5: HOMEOSTASIS                             ║
        # ║  (continuous background regulation)                ║
        # ╚═══════════════════════════════════════════════════╝
        
        # Estimate branching factor
        sigma_hat = self.homeostasis.estimate_sigma(
            spike_history=self.cortex.get_spike_history()
        )
        
        # Regulate toward σ ≈ 1
        self.homeostasis.regulate(
            sigma=sigma_hat,
            cortex=self.cortex,
            dt=self.dt
        )
        # Internally applies:
        #   - Intrinsic plasticity: ΔΘ = η_IP·(x - H_IP)
        #   - Inhibitory STDP (Vogels): homeostatic E/I balance
        #   - Synaptic depression: J_eff = J₀·u, du/dt = (u∞-u)/τ_rec
        #   - Weight normalization: W ← W/sum(W)
        
        # Energy budget
        self.chemistry.lactate.update_energy(
            neural_activity=self.cortex.mean_firing_rate(),
            dt=self.dt
        )
        
        # ╔═══════════════════════════════════════════════════╗
        # ║  OUTPUT                                            ║
        # ╚═══════════════════════════════════════════════════╝
        
        return GenesisOutput(
            t=self.t,
            action=action,
            gw_content=gw_content,
            confidence=meta_output.confidence if meta_output else None,
            introspection=schema_output.report if schema_output else None,
            sigma=sigma_hat,
            is_awake=self.is_awake,
            sleep_pressure=chem.adenosine,
            energy=self.chemistry.lactate.get_energy_budget(),
            prediction_errors=prediction_errors,
        )
    
    def _collect_gw_candidates(self, cortical_outputs, hpc_output, bg_info):
        """Gather stabilized representations for GW competition."""
        candidates = []
        for name, output in cortical_outputs.items():
            if output.is_stabilized():  # RPT: variance of late iterations < threshold
                candidates.append(GWCandidate(
                    source=name,
                    representation=output.L5_output,
                    strength=output.activation_strength
                ))
        if hpc_output and hpc_output.is_valid():
            candidates.append(GWCandidate(
                source="HPC",
                representation=hpc_output.ca1_output,
                strength=hpc_output.retrieval_confidence
            ))
        if bg_info and bg_info.decision_made:
            candidates.append(GWCandidate(
                source="BG",
                representation=bg_info.selected_action_repr,
                strength=bg_info.decision_confidence
            ))
        return candidates
    
    def _compute_beta(self, chem):
        """NE modulates ignition steepness (exploration/exploitation)."""
        # High NE → high beta → sharp winner-take-all (focused)
        # Low NE → low beta → soft mixing (explorative/dreamy)
        return self.global_workspace.beta_base * (1 + 0.5 * chem.norepinephrine)
```

---

## Часть IV: Критерии Провала

### 4.1 Критические Патологии (Автоматическая Остановка)

| # | Патология | Детекция | Биологический аналог |
|---|-----------|----------|---------------------|
| 1 | **Эпилепсия** | σ̂ > 1.2 более 100 мс подряд | Supercritical avalanche runaway |
| 2 | **Кома** | σ̂ < 0.7 более 500 мс подряд | Subcritical extinction |
| 3 | **Метаболический коллапс** | E < 0 для >50% нейронов | ATP depletion, lactate exhaustion |
| 4 | **Вечный сон** | is_awake=False более 10⁶ тактов | Orexin system failure (narcolepsy) |
| 5 | **Вечное бодрствование** | SWR не запускался >10⁶ тактов | Adenosine/sleep system failure |
| 6 | **Weight explosion** | max(|W|) > 10³ | Uncontrolled LTP without LTD |
| 7 | **Weight collapse** | max(|W|) < 10⁻⁶ | All weights decayed to zero |

### 4.2 Функциональные Критерии Провала (Экспериментальная Валидация)

| # | Тест | Порог провала | Что это значит |
|---|------|---------------|----------------|
| 1 | **Avalanche statistics** | P(S) не соответствует S^(-3/2) ± 0.2 после 10⁶ тактов | Сеть не достигла критичности → SOC-механизм не работает |
| 2 | **Metacognitive calibration** | ECE > 0.15 после обучения | MetaHead не калиброван → confidence бесполезен |
| 3 | **M-ratio** | M-ratio < 0.5 после обучения | Метакогниция хуже случайной → HOT не работает |
| 4 | **Recurrence dependency** | RDI < 0.1 (accuracy drop при W_fb=0) | Feedback бесполезен → predictive coding не работает |
| 5 | **GW ablation** | Нет разницы в систематичности ±GW | Global Workspace не нужен → интеграция не работает |
| 6 | **Replay benefit** | Точность с replay ≤ без replay после 10⁴ эпизодов | Hippocampal replay не помогает → CLS не работает |
| 7 | **AST ablation** | L_schema не уменьшается при обучении | Attention Schema не учится → AST бесполезен |
| 8 | **Chemical ablation** | f_mod = const не хуже dynamic f_mod | Вся химия — overhead без пользы |
| 9 | **Temporal hierarchy** | Одинаковый τ для всех зон не хуже gradient τ | Кортикальный градиент не нужен |
| 10 | **Energy gating** | Plasticity без energy gate не хуже | Lactate/ATP механизм — overhead |

### 4.3 Невозможность Калибровки

Если после systematic hyperparameter search (>1000 configurations) ни одна конфигурация
не достигает одновременно:

1. σ̂ ∈ [0.95, 1.05] (критичность)
2. ECE < 0.15 (метакогниция)
3. Replay > No-Replay (память)
4. RDI > 0.2 (predictive coding)

...то мы признаём, что **совместная калибровка всех уровней невозможна** и архитектура
нуждается в фундаментальном пересмотре. Наиболее вероятная причина: временные константы
разных уровней не стыкуются, создавая destructive interference вместо cooperative dynamics.

### 4.4 Масштабируемость

Если O(N²) вычислительная сложность Hopfield/CA3 делает невозможным масштабирование
до >10⁴ нейронов на одном GPU, и low-rank аппроксимации (factored keys, sparse attention)
разрушают критичность (σ̂ уходит от 1.0), то архитектура не пригодна для практических задач.

---

## Приложение A: Таблица Временных Констант

```
МАСШТАБ          ПРОЦЕСС                          τ              МОДУЛЬ
─────────────────────────────────────────────────────────────────────────
sub-ms           Mg²⁺ block NMDA                  < 0.1 ms       Synapse
1-10 ms          AMPA, GABA-A phasic, GlyR        1-20 ms        Synapse
                 Spike generation/propagation      1-2 ms         Soma
                 STN hyperdirect pathway           5-10 ms        BG
5-30 ms          Thalamic relay (FO, HO)           5-30 ms        Thalamus
                 TRN integration                   20-50 ms       Thalamus
10-200 ms        NMDA (with Mg-block)              10-200 ms      Synapse
                 SENS cortex processing            20 ms          Cortex
                 DG pattern separation             30-50 ms       Hippocampus
50-300 ms        ASSOC cortex processing           100 ms         Cortex
                 CA3 pattern completion             50-150 ms      Hippocampus
                 CA1 comparison                    30-80 ms       Hippocampus
                 Striatal up-states                50-100 ms      BG
                 DA phasic burst                   100-500 ms     VTA
100 ms - sec     NO diffusion                      0.1-2 s        Volume TX
                 GW ignition cycle                 ~250 ms        Consciousness
                 Substance P, Enkephalins          0.5-30 s       Neuropeptide
1-60 sec         2-AG phasic retrograde            1-30 s         Volume TX
                 Orexin onset/duration             0.5-60 s       Neuropeptide
                 Cortisol non-genomic              10 s-15 min    Neurosteroid
min - hours      Adenosine accumulation            minutes-hours  Metabolite
                 Lactate shuttle                   minutes-hours  Metabolite
                 PFC persistent activity           seconds-min    Cortex
hours - days     Cortisol genomic                  1-24 hours     Neurosteroid
                 Estradiol genomic                 1-48 hours     Neurosteroid
                 SWR replay consolidation          hours          Hippocampus
```

## Приложение B: Глоссарий Биологических Обоснований

| Компонент Genesis | Биологический источник | Ключевая ссылка |
|---|---|---|
| LCA sparse encoder | V1 sparse coding | Olshausen-Field 1996; Rozell 2008 |
| Urbanczik-Senn neuron | Two-compartment local learning | Urbanczik-Senn 2014 (Neuron) |
| Sacramento microcircuit | Deep predictive coding | Sacramento et al. 2018 (NeurIPS) |
| Rao-Ballard hierarchy | Hierarchical predictive coding | Rao-Ballard 1999 (Nat Neurosci) |
| FO/HO Thalamus | Sherman-Guillery relay model | Sherman & Guillery 2006 |
| TRN attention gate | Crick searchlight | Crick 1984; Halassa & Kastner 2017 |
| Cortical gradient τ(g) | Principal gradient G1 | Margulies et al. 2016 (PNAS) |
| DG sparse / neurogenesis | Pattern separation | Aimone et al. 2011 |
| CA3 Hopfield | Autoassociative memory | Hopfield 1982; Treves & Rolls 1994 |
| CA1 comparator | Novelty / VTA loop | Lisman & Grace 2005 |
| SWR replay | Complementary learning | McClelland et al. 1995; Kumaran 2016 |
| BG DDM | Drift-diffusion action selection | Frank 2006; Bogacz & Gurney 2007 |
| Go/NoGo/Stop | D1/D2/STN pathways | Nambu 2008; Mink 1996 |
| DA RPE | Reward prediction error | Schultz et al. 1997 |
| NE gain modulation | LC arousal/exploration | Aston-Jones & Cohen 2005 |
| ACh FF/FB balance | Expected uncertainty | Yu & Dayan 2005; Hasselmo 2006 |
| Modern Hopfield → GW | Attention = associative memory | Ramsauer et al. 2021 (ICLR) |
| GWT ignition | Neuronal workspace | Dehaene & Changeux 2011 |
| HOT / MetaHead | Higher-order thought | Rosenthal 2005; Fleming & Lau 2014 |
| AST | Attention schema | Graziano 2013 |
| SOC / σ≈1 | Self-organized criticality | Beggs-Plenz 2003; Kinouchi-Copelli 2006 |
| Synaptic depression SOC | Levina model | Levina et al. 2007 |
| SORN homeostasis | Self-organizing recurrent net | Del Papa 2017 |
| Inhibitory STDP | E/I balance | Vogels et al. 2011 |
| NO volume transmission | Retrograde heterosynaptic LTP | Garthwaite 2008 |
| 2-AG retrograde | eCB normalization | Freund et al. 2003 |
| Cortisol MR/GR | Stress regime switch | Joëls 2008; de Kloet 2005 |
| Orexin wake-attractor | Sleep/wake switching | Sakurai 2007; Saper 2005 |
| Adenosine sleep pressure | Borbely two-process model | Borbely 1982; Porkka-Heiskanen 1997 |
| Lactate energy gate | ANLS / astrocyte shuttle | Pellerin & Magistretti 1994 |
| H₂S→D-ser→NMDA pipeline | Gas-gliotransmission | Kimura 2004; Henneberger 2010 |

---

*«Мы не проектируем нейросеть. Мы выращиваем организм.
У него есть сон, голод, стресс, любопытство и чувство неуверенности.
Если он не заснёт — он умрёт от метаболического коллапса.
Если он не проснётся — он умрёт от сенсорного голода.
Его единственная цель — выжить на грани хаоса.
Всё остальное — интеллект, память, сознание — возникает как побочный эффект этого выживания.»*

**— Genesis Manifesto, v10.0**
