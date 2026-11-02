"""Фаза 2 Глобального Плана v8.0 (snnscience.docx §3.6, §4.6 Предсказание 3):
Двухпутевой слогово-фонемный декодер Популяционного Вектора с гейтированием Базальными Ганглиями
(`BGPQSyllabaryDecoder`: Dual-Path Basal-Ganglia Product-Quantized & Syllabic Trajectory Decoder).

Устраняет Изъян 6 («Неправдоподобный линейный выходной слой и WTA на 50 000 слов»):
  1. Вместо плоского линейного readout [d -> V] с 25+ млн синапсов и 50,000-канальной гонкой WTA
     реализует биологически достоверную иерархию речевого моторного контроля (Broca BA44 -> SMA -> vSMC M1,
     Bouchard et al. 2013; Hickok & Poeppel 2007; Georgopoulos et al. 1986):
       - Путь А (`Fast Direct Striatal Head`, `n_head = 512`): частотное ядро лемм/слогов,
         выбираемое за 1 такт через прямое снятие тонического торможения GPi/SNr (Direct D1 Go Path).
       - Путь Б (`Indirect Product-Quantized Syllabic Trajectory Tail`, `N_onset × N_coda = 256 × 256 = 65,536`):
         любое слово из словаря V <= 65,536 адресуется парой слоговых координат
         (Onset/Root Syllable c_1 ∈ [0..255], Coda/Inflection Syllable c_2 ∈ [0..255])
         и декодируется в 32-мерный артикуляторный популяционный вектор жестов (губной, корональный,
         дорсальный, озвончение, назальность, подъём/ряд гласного).
  2. Гейтирование Базальными Ганглиями (D1 Go / D2 NoGo / STN Hyperdirect):
     - Прямой путь D1 усиливает выбранный слоговой план;
     - Непрямой путь D2 подавляет конкурирующие слоги;
     - Гиперпрямой путь STN поднимает порог выбора при высокой энтропии конфликта.
  3. Дофаминовая миграция навыка во сне (`consolidate_habit`):
     - Редкие слова из Хвоста (Путь Б), накопившие высокий дофаминовый след, мигрируют
       в Быструю Стриатарную Голову (Путь А).
  4. Композиционная продуктивность (OOV) и Психолингвистический Тест Спунеризмов (`simulate_spoonerism`,
     Фальсифицируемое Предсказание 3 snnscience.docx §4.6):
     - Способность собирать незнакомые составные слова (OOV) из знакомых слогов без дообучения;
     - При скачке норадреналина (LC/NE burst / дефицит времени) воспроизводит классический
       психолингвистический эффект Спунеризмов (перестановка начальных слогов c_1 между двумя
       активными планами слов в буфере SMA/Broca, причём на низкочастотных словах чаще, чем на ядерных!).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SyllabicDecodeOut:
    token_id: int
    onset_id: int
    coda_id: int
    used_fast_head: bool
    articulatory_vector: torch.Tensor   # [32] популяционный вектор артикуляторных жестов M1
    bg_go_drive: float
    bg_nogo_suppression: float
    stn_conflict: float
    active_ops: int


class BGPQSyllabaryDecoder:
    """Двухпутевой слогово-фонемный декодер Популяционного Вектора с гейтированием Базальными Ганглиями."""

    def __init__(
        self,
        vocab_size: int = 50_000,
        d_in: int = 128,
        n_head: int = 512,
        n_syllables: int = 256,
        n_articulators: int = 32,
        seed: int = 42,
    ):
        self.vocab_size = int(vocab_size)
        self.d_in = int(d_in)
        self.n_head = min(int(n_head), self.vocab_size)
        self.n_syllables = max(16, int(n_syllables))
        self.n_articulators = int(n_articulators)

        g = torch.Generator().manual_seed(seed)
        # 1. Путь А: Быстрая Стриатарная Голова (ядерные леммы 0 .. n_head-1)
        self.w_head = torch.randn(self.n_head, self.d_in, generator=g) * (1.0 / math.sqrt(self.d_in))
        self.head_token_ids = torch.arange(self.n_head, dtype=torch.long)

        # 2. Путь Б: Слоговая траектория (Onset Syllable c_1 и Coda Syllable c_2)
        self.w_onset = torch.randn(self.n_syllables, self.d_in, generator=g) * (1.0 / math.sqrt(self.d_in))
        self.w_coda = torch.randn(self.n_syllables, self.d_in, generator=g) * (1.0 / math.sqrt(self.d_in))
        # Латеральная матрица переходов внутри слова: Onset c_1 -> Coda c_2 (SMA секвенсор)
        self.w_intra_word = torch.randn(self.n_syllables, self.n_syllables, generator=g) * 0.15

        # 3. Популяционный вектор артикуляторных жестов в vSMC (M1, 32 канала Georgopoulos)
        # Каждому слогу сопоставлен канонический вектор артикуляторных синергий [n_syllables, 32]
        dirs = torch.randn(self.n_syllables, self.n_articulators, generator=g)
        self.syllable_articulators = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 4. Гейты Базальных Ганглиев (D1 Go / D2 NoGo / STN) для слогового выбора
        self.d1_go_gain = torch.ones(self.n_syllables)
        self.d2_nogo_inh = torch.full((self.n_syllables,), 0.15)
        self.word_da_salience = torch.zeros(self.vocab_size)
        self.word_frequency_tier = torch.zeros(self.vocab_size)
        # Ядерные слова имеют высокий базовый частотный иммунитет
        self.word_frequency_tier[: self.n_head] = 1.0

        # --- next9.txt #3: Adaptive Softmax Migration (Grave et al. 2016, arxiv 1609.04309) ---
        # Онлайн-счётчики и экспоненциальная скользящая частота использования слов
        # для динамической двусторонней миграции между Быстрой Головой (Путь А) и Слоговым Хвостом (Путь Б)
        self.word_usage_ema = torch.zeros(self.vocab_size)
        self.word_usage_ema[: self.n_head] = 0.25
        self.word_usage_counts = torch.zeros(self.vocab_size, dtype=torch.long)
        self.migration_margin: float = 0.08
        self.auto_migrate_online: bool = True

        # Обратимая таблица композиции (onset_id, coda_id) <-> token_id и реестр OOV-слов
        self.custom_oov_lexicon: Dict[Tuple[int, int], str] = {}
        self.promoted_to_head: List[int] = []
        self.demoted_to_tail: List[int] = []
        self.migration_history: List[Tuple[int, int, int]] = []  # (promoted_tid, demoted_tid, slot)

    def token_to_syllables(self, token_id: int) -> Tuple[int, int]:
        """Декомпозирует ID слова в пару слоговых индексов (onset_id, coda_id) за O(1)."""
        tid = max(0, int(token_id))
        c1 = (tid // self.n_syllables) % self.n_syllables
        c2 = tid % self.n_syllables
        return c1, c2

    def syllables_to_token(self, onset_id: int, coda_id: int) -> int:
        """Собирает ID слова из пары слоговых индексов (onset_id, coda_id)."""
        tid = (int(onset_id) % self.n_syllables) * self.n_syllables + (int(coda_id) % self.n_syllables)
        return min(self.vocab_size - 1, tid)

    def num_parameters(self) -> int:
        """Общее число синаптических параметров декодера (против V * d_in у линейного слоя)."""
        return int(
            self.w_head.numel()
            + self.w_onset.numel()
            + self.w_coda.numel()
            + self.w_intra_word.numel()
            + self.syllable_articulators.numel()
        )

    def dense_baseline_parameters(self) -> int:
        return int(self.vocab_size * self.d_in)

    def ops_per_step(self) -> int:
        """Число операций умножения-сложения за один шаг декодирования."""
        return int((self.n_head + 2 * self.n_syllables) * self.d_in + self.n_syllables * self.n_articulators)

    def decode(
        self,
        h_l5: torch.Tensor,
        da_level: float = 0.5,
        ne_burst: float = 0.2,
        force_tail: bool = False,
    ) -> SyllabicDecodeOut:
        """Выполняет двухпутевое декодирование из вектора L5 коры с гейтированием Базальными Ганглиями."""
        h = h_l5.detach().flatten()[: self.d_in]
        if h.numel() < self.d_in:
            h = torch.nn.functional.pad(h, (0, self.d_in - h.numel()))

        # 1. Оценка Быстрой Стриатарной Головы (Путь А)
        head_scores = self.w_head @ h
        head_max, head_idx = torch.max(head_scores, dim=0)

        # 2. Оценка Слоговой Траектории (Путь Б: Onset c_1 -> Coda c_2) с D1/D2 гейтингом BG
        d1_boost = 1.0 + 0.45 * float(da_level)
        d2_gate = (1.0 - self.d2_nogo_inh * (1.0 - 0.35 * float(da_level))).clamp(0.2, 1.0)
        onset_raw = (self.w_onset @ h) * self.d1_go_gain * d1_boost * d2_gate

        p_onset = torch.softmax(onset_raw * 3.0, dim=-1)
        stn_conflict = float(
            -(p_onset * p_onset.clamp(min=1e-8).log()).sum().item()
            / math.log(float(self.n_syllables))
        )
        onset_id = int(onset_raw.argmax().item())

        # Второй суб-шаг траектории: выбор Coda c_2 с учётом синаптического перехода от Onset c_1
        coda_raw = (self.w_coda @ h) * d2_gate + self.w_intra_word[:, onset_id]
        coda_id = int(coda_raw.argmax().item())

        # 3. Вычисление Популяционного Вектора артикуляторных жестов в M1 (Georgopoulos 1986):
        #    V_pop = 0.6 * vec(c_1) + 0.4 * vec(c_2)
        art_vec = 0.60 * self.syllable_articulators[onset_id] + 0.40 * self.syllable_articulators[coda_id]
        art_vec = art_vec / art_vec.norm().clamp(min=1e-6)

        # Выбор между Быстрой Головой (Путь А) и Слоговым Хвостом (Путь Б):
        tail_score = 0.5 * (float(onset_raw[onset_id].item()) + float(coda_raw[coda_id].item()))
        use_head = (not force_tail) and (float(head_max.item()) >= tail_score + 0.15 * stn_conflict)
        if use_head:
            win_tok = int(self.head_token_ids[int(head_idx.item())].item())
            c1, c2 = self.token_to_syllables(win_tok)
        else:
            c1, c2 = onset_id, coda_id
            win_tok = self.syllables_to_token(c1, c2)

        # Обновляем онлайн-статистику использования для Adaptive Softmax Migration (arxiv 1609.04309)
        self.record_word_usage(win_tok, h_l5=h, da_boost=float(da_level))

        return SyllabicDecodeOut(
            token_id=win_tok,
            onset_id=c1,
            coda_id=c2,
            used_fast_head=use_head,
            articulatory_vector=art_vec,
            bg_go_drive=round(float(d1_boost), 4),
            bg_nogo_suppression=round(float(self.d2_nogo_inh.mean().item()), 4),
            stn_conflict=round(stn_conflict, 4),
            active_ops=self.ops_per_step(),
        )

    def record_word_usage(
        self,
        token_id: int,
        h_l5: Optional[torch.Tensor] = None,
        da_boost: float = 0.5,
    ) -> Optional[Dict[str, int]]:
        """Фиксирует факт использования слова и при превышении порога частоты
        динамически мигрирует слово из Слогового Хвоста (Путь Б) в Быструю Голову (Путь А)
        по правилу Adaptive Softmax Migration (Grave et al. 2016, arxiv 1609.04309)."""
        tid = max(0, min(self.vocab_size - 1, int(token_id)))
        self.word_usage_counts[tid] += 1
        self.word_usage_ema.mul_(0.998)
        self.word_usage_ema[tid] = float(self.word_usage_ema[tid].item()) + 0.08 * (1.0 + 0.5 * max(0.0, float(da_boost)))
        if self.auto_migrate_online and int(self.word_usage_counts[tid].item()) >= 3:
            in_head = bool((self.head_token_ids == tid).any().item())
            if not in_head:
                mig = self.adaptive_softmax_migrate(candidate_tid=tid, h_l5=h_l5)
                if mig["migrated"]:
                    return mig
        return None

    def adaptive_softmax_migrate(
        self,
        candidate_tid: Optional[int] = None,
        h_l5: Optional[torch.Tensor] = None,
        margin: Optional[float] = None,
    ) -> Dict[str, int]:
        """Adaptive Softmax Migration (Grave et al. 2016, arxiv 1609.04309; next9.txt #3):
        Динамическая миграция слов между Быстрой Головой (Путь А) и Слоговым Хвостом (Путь Б).
        Если слово в Хвосте часто используется (его эффективная частота + дофаминовая значимость
        превышает самый холодный слот в Быстрой Голове на `migration_margin`), оно перемещается
        в Голову для мгновенной одношаговой генерации, а редкое слово из Головы вытесняется в Хвост."""
        eff_margin = self.migration_margin if margin is None else float(margin)
        combined_score = self.word_usage_ema + 0.5 * self.word_da_salience

        # Определяем самый «холодный» слот в Быстрой Голове (среди динамической зоны слотов)
        dyn_start = max(0, self.n_head // 2)
        head_sub_ids = self.head_token_ids[dyn_start:]
        head_sub_scores = combined_score[head_sub_ids]
        min_head_val, rel_slot = torch.min(head_sub_scores, dim=0)
        cold_slot = int(dyn_start + int(rel_slot.item()))
        demoted_tid = int(self.head_token_ids[cold_slot].item())

        if candidate_tid is not None:
            tid = max(0, min(self.vocab_size - 1, int(candidate_tid)))
            if bool((self.head_token_ids == tid).any().item()):
                return {"migrated": 0, "promoted_tid": -1, "demoted_tid": -1, "slot": -1}
            cand_val = float(combined_score[tid].item())
        else:
            tail_scores = combined_score.clone()
            tail_scores[self.head_token_ids] = -1e9
            best_val, best_tid = torch.max(tail_scores, dim=0)
            tid = int(best_tid.item())
            cand_val = float(best_val.item())

        if cand_val < float(min_head_val.item()) + eff_margin:
            return {"migrated": 0, "promoted_tid": -1, "demoted_tid": -1, "slot": -1}

        # Выполняем обмен между Головой (Путь А) и Хвостом (Путь Б)
        self.head_token_ids[cold_slot] = tid
        c1, c2 = self.token_to_syllables(tid)
        syn_proto = 0.5 * (self.w_onset[c1] + self.w_coda[c2])
        if h_l5 is not None:
            h = h_l5.detach().flatten()[: self.d_in]
            if h.numel() < self.d_in:
                h = torch.nn.functional.pad(h, (0, self.d_in - h.numel()))
            h_norm = h / h.norm().clamp(min=1e-6)
            self.w_head[cold_slot] = 0.65 * h_norm + 0.35 * syn_proto
        else:
            self.w_head[cold_slot] = syn_proto

        self.word_frequency_tier[tid] = 1.0
        self.word_frequency_tier[demoted_tid] = max(0.15, float(self.word_frequency_tier[demoted_tid].item()) * 0.5)
        self.promoted_to_head.append(tid)
        self.demoted_to_tail.append(demoted_tid)
        self.migration_history.append((tid, demoted_tid, cold_slot))
        return {
            "migrated": 1,
            "promoted_tid": tid,
            "demoted_tid": demoted_tid,
            "slot": cold_slot,
        }

    def learn(self, target_token_id: int, h_l5: torch.Tensor, lr: float = 0.08, da_rpe: float = 0.5) -> None:
        """Локальное трёхфакторное обучение слоговых проекций и D1/D2 весов Базальных Ганглиев."""
        tid = max(0, min(self.vocab_size - 1, int(target_token_id)))
        h = h_l5.detach().flatten()[: self.d_in]
        if h.numel() < self.d_in:
            h = torch.nn.functional.pad(h, (0, self.d_in - h.numel()))
        h_norm = h / h.norm().clamp(min=1e-6)

        c1, c2 = self.token_to_syllables(tid)
        eff_lr = float(lr) * (0.5 + 0.5 * max(0.0, float(da_rpe)))
        self.w_onset[c1] = self.w_onset[c1] + eff_lr * (h_norm - self.w_onset[c1] * 0.1)
        self.w_coda[c2] = self.w_coda[c2] + eff_lr * (h_norm - self.w_coda[c2] * 0.1)
        self.w_intra_word[c2, c1] = min(1.5, float(self.w_intra_word[c2, c1].item()) + 0.5 * eff_lr)
        self.d1_go_gain[c1] = min(2.0, float(self.d1_go_gain[c1].item()) + 0.05 * max(0.0, float(da_rpe)))

        self.word_da_salience[tid] = 0.95 * self.word_da_salience[tid] + 0.25 * max(0.0, float(da_rpe))
        self.word_frequency_tier[tid] = min(1.0, float(self.word_frequency_tier[tid].item()) + 0.04)

        # Обновляем частотный след и проверяем онлайн-миграцию в Быструю Голову (Adaptive Softmax Migration)
        self.record_word_usage(tid, h_l5=h_norm, da_boost=float(da_rpe))

        # Если токен присутствует в Быстрой Голове (Путь А), обучаем и его прямое представление
        head_matches = torch.nonzero(self.head_token_ids == tid, as_tuple=False).flatten()
        if head_matches.numel() > 0:
            h_idx = int(head_matches[0].item())
            self.w_head[h_idx] = self.w_head[h_idx] + eff_lr * (h_norm - self.w_head[h_idx] * 0.1)

    def consolidate_habit(self, min_salience: float = 0.35) -> int:
        """Сонная миграция навыка (Adaptive Softmax Migration + DA-консолидация):
        продвигает высокозначимые и частотные слова из Хвоста (Путь Б) в Быструю Голову (Путь А)."""
        tail_sal = self.word_da_salience + 0.6 * self.word_usage_ema
        tail_sal[self.head_token_ids] = -1.0
        best_val, best_tid = torch.max(tail_sal, dim=0)
        if float(best_val.item()) < min_salience:
            return -1
        tid = int(best_tid.item())
        mig = self.adaptive_softmax_migrate(candidate_tid=tid, margin=-1e9)
        if mig["migrated"]:
            return int(mig["promoted_tid"])
        return -1

    def compose_oov_word(self, word_a_id: int, word_b_id: int, label: str = "oov_compound") -> Dict[str, object]:
        """Композиционная продуктивность (OOV): конструирует новое составное слово,
        беря начальный слог (Onset c_1) от слова A и конечный слог (Coda c_2) от слова B."""
        c1_a, _ = self.token_to_syllables(word_a_id)
        _, c2_b = self.token_to_syllables(word_b_id)
        composed_tid = self.syllables_to_token(c1_a, c2_b)
        self.custom_oov_lexicon[(c1_a, c2_b)] = label
        # Популяционный вектор жестов нового слова мгновенно существует как суперпозиция синергий!
        art_vec = 0.60 * self.syllable_articulators[c1_a] + 0.40 * self.syllable_articulators[c2_b]
        art_vec = art_vec / art_vec.norm().clamp(min=1e-6)
        return {
            "composed_token_id": composed_tid,
            "onset_from_a": c1_a,
            "coda_from_b": c2_b,
            "label": label,
            "articulatory_norm": round(float(art_vec.norm().item()), 4),
        }

    def simulate_spoonerism(
        self,
        word_a_id: int,
        word_b_id: int,
        ne_stress: float = 0.85,
        n_trials: int = 200,
        seed: int = 101,
    ) -> Dict[str, object]:
        """Фальсифицируемое Предсказание 3 (snnscience.docx §4.6 — Психолингвистический Тест Спунеризмов):
        Когда два слова (A и B) одновременно удерживаются в моторном буфере SMA/Broca,
        при высоком норадреналине / дефиците времени (ne_stress > 0.6) возникает конкуренция
        начальных слогов (Onset c_1^(A) <-> Onset c_1^(B)), порождающая перестановку начальных слогов
        (Spoonerism: (c1_A, c2_A), (c1_B, c2_B) -> (c1_B, c2_A), (c1_A, c2_B)).
        Причём для низкочастотных слов (Tail) частота спунеризмов существенно выше, чем для ядерных (Head)!"""
        c1_a, c2_a = self.token_to_syllables(word_a_id)
        c1_b, c2_b = self.token_to_syllables(word_b_id)
        freq_a = float(self.word_frequency_tier[min(self.vocab_size - 1, word_a_id)].item())
        freq_b = float(self.word_frequency_tier[min(self.vocab_size - 1, word_b_id)].item())
        mean_freq = 0.5 * (freq_a + freq_b)

        # Фонологическое сходство артикуляторных векторов начальных слогов повышает вероятность обмена
        phon_sim = float(
            torch.clamp(self.syllable_articulators[c1_a] @ self.syllable_articulators[c1_b], 0.0, 1.0).item()
        )
        g = torch.Generator().manual_seed(seed)
        spoonerism_count = 0
        for _ in range(max(1, n_trials)):
            # Амплитуда шума конкуренции в SMA растёт с ne_stress и падает с автоматизацией навыка (mean_freq)
            noise_amp = float(ne_stress) * (1.15 - 0.65 * mean_freq) * (0.65 + 0.35 * phon_sim)
            drive_a = 1.0 + 0.5 * mean_freq + float(torch.randn(1, generator=g).item()) * noise_amp
            drive_b_intrusion = 0.72 + float(torch.randn(1, generator=g).item()) * noise_amp
            if drive_b_intrusion > drive_a:
                spoonerism_count += 1

        swapped_word_1 = self.syllables_to_token(c1_b, c2_a)
        swapped_word_2 = self.syllables_to_token(c1_a, c2_b)
        return {
            "original_pair": (word_a_id, word_b_id),
            "original_syllables": ((c1_a, c2_a), (c1_b, c2_b)),
            "spoonerism_pair": (swapped_word_1, swapped_word_2),
            "spoonerism_syllables": ((c1_b, c2_a), (c1_a, c2_b)),
            "spoonerism_rate": round(spoonerism_count / max(1, n_trials), 4),
            "mean_frequency_tier": round(mean_freq, 4),
            "ne_stress": round(float(ne_stress), 4),
        }

    def thermodynamic_forgetting_noise_floor(self) -> float:
        """Lab.html F#58 (ORIN, accepted on P7): Вычисляет термодинамический шумовой порог забывания
        (`forgetting_noise_floor`) из дисперсии ошибки аппроксимации Adaptive Softmax:
        чем сильнее сжатие Хвоста относительно полного словаря V, тем выше дисперсия усечения,
        которая подавляет микро-обновления весов меньше `noise_floor` и служит прямым регулятором
        порога для каркаса памяти `CT-PNN` (`readout_approx_noise_floor`)."""
        head_ratio = float(self.n_head) / float(max(1, self.vocab_size))
        tail_compression = float(2 * self.n_syllables) / float(max(1, self.vocab_size - self.n_head))
        approx_variance = max(1e-5, (1.0 - head_ratio) * (1.0 - min(1.0, tail_compression)) * 0.012)
        return round(math.sqrt(approx_variance) * 0.15, 6)

    def profile_macs_vs_exact_softmax(self) -> Dict[str, float]:
        """Lab.html F#24 (WARD, accepted on P6) & F#26 (ATLAS, replicated on P6):
        Считает число операций умножения-накопления (MACs) на 1 выходной спайк при точном
        плоском Softmax по сравнению с иерархическим Adaptive Softmax (`BGPQSyllabaryDecoder`)."""
        exact_macs = float(self.vocab_size * self.d_in)
        approx_macs = float(self.ops_per_step())
        saved_macs = max(0.0, exact_macs - approx_macs)
        return {
            "exact_softmax_macs": exact_macs,
            "adaptive_syllabary_macs": approx_macs,
            "mac_reduction_factor": round(exact_macs / max(1.0, approx_macs), 2),
            "energy_saved_fraction": round(saved_macs / max(1.0, exact_macs), 4),
        }
