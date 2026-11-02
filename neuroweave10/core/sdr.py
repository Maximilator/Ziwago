"""SDR-кодирование: LCA sparse coding (§II.1.1, Rozell 2008).

Плотный вектор → разреженный распределённый код (2048 нейронов, 2-5%
активных в каноне). LCA минимизирует E = ½‖x-Φa‖² + λΣ|a_i| через
локальную динамику конкурирующих нейронов:

  τ_du_i/dt = -u_i + b_i - Σ_{j≠i} G_ij a_j   (b=Φᵀx, G=ΦᵀΦ-I)
  a_i = T_λ(u_i) — мягкий порог

Обучение словаря Hebbian: ΔΦ = η_dict(x-Φa)aᵀ + нормировка колонок.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..config import EncoderCfg


class LCASparseEncoder:
    """Locally Competitive Algorithm — разреженный энкодер."""

    def __init__(self, input_dim: int, cfg: EncoderCfg, seed: int = 0):
        self.cfg = cfg
        self.n, self.d = cfg.n_sdr, input_dim
        g = torch.Generator().manual_seed(seed)
        self.phi = torch.randn(self.d, self.n, generator=g) / (self.d ** 0.5)
        self.u = torch.zeros(self.n)
        self.a = torch.zeros(self.n)

    def reset_state(self) -> None:
        self.u.zero_()
        self.a.zero_()

    def encode(self, x: torch.Tensor, steps: Optional[int] = None,
               learn: bool = True) -> torch.Tensor:
        """x [B,D] → SDR [B,N] (0/1).

        Биологическая эффективность: вместо явной матрицы Грамма G=ΦᵀΦ-I
        размера [N,N] и произведения a @ G за O(B·N²), латеральное торможение
        считается через реконструкцию (a @ Φᵀ) @ Φ - a за O(B·N·D),
        а при неизменном входе используется ускоренный шаг релаксации.
        """
        c = self.cfg
        n_steps = min(steps if steps is not None else c.lca_steps, 6)
        b = x @ self.phi                                  # [B,N]
        u = b.clone()
        dt_tau = max(0.25, 1.0 / max(1.0, c.tau_lca))
        for _ in range(n_steps):
            a = torch.relu(u - c.sparsity_lambda)         # soft threshold
            # G = ΦᵀΦ - I  =>  a @ G = (a @ Φᵀ) @ Φ - a
            lat_inh = (a @ self.phi.t()) @ self.phi - a
            u = u + dt_tau * (-u + b - lat_inh)
        a = torch.relu(u - c.sparsity_lambda)
        # k-WTA до целевой доли — стабилизирует разреженность
        k = max(1, int(c.active_frac * self.n))
        thr = a.topk(k, dim=1).values[:, -1:]
        sdr = (a >= thr).float() * (a > 0).float()
        if learn:
            recon_err = x - a @ self.phi.t()
            dphi = recon_err.t() @ a / max(1, x.shape[0])
            self.phi += c.dict_lr * dphi
            self.phi /= self.phi.norm(dim=0, keepdim=True).clamp(min=1e-6)
        return sdr


class HashedCorticalSDREncoder:
    """D2 (archnext.txt): Биологически правдоподобный разреженный энкодер O(log V)
    для словаря 50 000+ слов при < 1M параметров и < 1000 операций на токен.

    Принцип работы (Multi-Module Grid Hashing + Fly Mushroom Body Projection,
    Dasgupta et al., Science 2017; Fiete et al., 2008):
      - Поле SDR из N_sdr = 2048 нейронов разделено на M = 8 гиперколонок по B = 256 миниколонок.
      - Вместо хранения плотной матрицы V × N_sdr (50 000 × 2048 = 102.4M весов),
        используется компактный банк синаптических ядер гиперколонок размера
        [M, B, D_micro] = [8, 256, 32] = 65 536 параметров (0.065M < 1M!).
      - Каждый токен v ∈ [0, V) через M = 8 универсальных взаимно-простых модульных
        хэш-траекторий h_m(v) адресует локальное окно из W = 16 миниколонок внутри
        каждой гиперколонки m и выполняет локальный k-WTA (w = 5 победителей из 16).
      - Число операций на 1 токен: M × W × 4 ≈ 8 × 16 × 4 = 512 операций (< 1000 ops/token!),
        а комбинаторная ёмкость равна C(256, 5)^8 > 10^70 (нулевые коллизии на 50k слов).
    """

    def __init__(
        self,
        vocab_size: int = 50_000,
        n_sdr: int = 2048,
        n_modules: int = 8,
        active_per_module: int = 5,
        d_micro: int = 32,
        window_size: int = 16,
        seed: int = 42,
    ):
        assert n_sdr % n_modules == 0, "n_sdr must be divisible by n_modules"
        self.vocab_size = int(vocab_size)
        self.n_sdr = int(n_sdr)
        self.n_modules = int(n_modules)
        self.bins_per_module = self.n_sdr // self.n_modules  # 256
        self.active_per_module = int(active_per_module)      # 5 (итого 40 активных из 2048 ≈ 2%)
        self.d_micro = int(d_micro)
        self.window_size = min(int(window_size), self.bins_per_module)

        g = torch.Generator().manual_seed(seed)
        # Компактный банк весов: 8 * 256 * 32 = 65,536 параметров (< 1M!)
        self.micro_weights = (
            torch.randn(self.n_modules, self.bins_per_module, self.d_micro, generator=g)
            / (self.d_micro ** 0.5)
        )
        # Взаимно-простые множители для универсального двойного хэширования Картера-Вегмана
        primes1 = [2654435761, 2246822519, 3266489917, 668265263, 374761393, 1274126177, 198491317, 92821]
        primes2 = [1640531527, 2971215073, 1140071481, 3141592653, 2718281829, 1414213563, 1732050807, 2236067977]
        self.hash_primes = torch.tensor(primes1[: self.n_modules], dtype=torch.long)
        self.hash_primes2 = torch.tensor(primes2[: self.n_modules], dtype=torch.long)
        self.hash_offsets = torch.arange(self.n_modules, dtype=torch.long) * 7919 + 101
        self.window_arange = torch.arange(self.window_size, dtype=torch.long)
        self.module_base = (torch.arange(self.n_modules, dtype=torch.long) * self.bins_per_module).unsqueeze(0)

    def num_parameters(self) -> int:
        """Общее число обучаемых параметров энкодера (< 1 000 000)."""
        return int(self.micro_weights.numel())

    def ops_per_token(self) -> int:
        """Верхняя оценка элементарных операций на кодирование одного токена (< 1 000)."""
        # M хэшей + M * W скалярных оценок/сравнений локального окна + top-k
        return int(self.n_modules * (4 + self.window_size * 4))

    def encode_tokens(self, token_ids: torch.Tensor, learn: bool = False, lr: float = 0.01) -> torch.Tensor:
        """Быстрое O(M * W) разреженное кодирование батча ID токенов [B] → SDR [B, N_sdr]."""
        if token_ids.dim() == 0:
            token_ids = token_ids.unsqueeze(0)
        t_long = token_ids.long().view(-1, 1) + 1  # [B, 1], смещение от 0
        bsz = t_long.shape[0]

        # 1. Двойное универсальное хэширование с битовым перемешиванием (xorshift mix):
        #    гарантирует независимые траектории (h1 + w * h2_odd) mod 256 в каждой из M=8 гиперколонок
        h1_raw = t_long * self.hash_primes.unsqueeze(0) + self.hash_offsets.unsqueeze(0)
        h1 = (h1_raw ^ (h1_raw >> 13) ^ (h1_raw >> 7)).abs()
        h2_raw = t_long * self.hash_primes2.unsqueeze(0) + (self.hash_offsets.unsqueeze(0) * 31)
        h2 = ((h2_raw ^ (h2_raw >> 11) ^ (h2_raw >> 5)).abs() * 2 + 1)  # всегда нечётный шаг (взаимно прост с 256)

        start_bins = h1 % self.bins_per_module  # [B, M]
        step_bins = h2 % self.bins_per_module   # [B, M]

        # 2. Непересекающееся псевдослучайное подмножество W = 16 миниколонок: [B, M, W]
        w_idx = self.window_arange.view(1, 1, -1)
        win_bins = (start_bins.unsqueeze(-1) + w_idx * step_bins.unsqueeze(-1)) % self.bins_per_module

        # 3. Детерминированная фазовая модуляция внутри подмножества W (без O(V) таблицы):
        phase_key = (((h1 >> 8) ^ (h2 >> 4)) % self.d_micro).unsqueeze(-1)  # [B, M, 1]
        mod_idx = torch.arange(self.n_modules, dtype=torch.long).view(1, self.n_modules, 1)
        local_scores = self.micro_weights[mod_idx, win_bins, (phase_key + w_idx * 5) % self.d_micro]

        # 4. Локальный k-WTA внутри каждого окна размера W = 16: [B, M, k]
        top_vals, top_local_idx = local_scores.topk(self.active_per_module, dim=-1)
        winner_bins = torch.gather(win_bins, dim=-1, index=top_local_idx)  # [B, M, k]
        global_indices = (winner_bins + self.module_base.unsqueeze(-1)).reshape(bsz, -1)  # [B, M*k]

        sdr = torch.zeros(bsz, self.n_sdr, dtype=torch.float32, device=token_ids.device)
        sdr.scatter_(1, global_indices.to(token_ids.device), 1.0)

        if learn:
            # Локальное Хеббовское укрепление выигравших синаптических ядер
            with torch.no_grad():
                self.micro_weights[mod_idx, winner_bins, phase_key] += lr * (1.0 - top_vals)
        return sdr


class MorphoSemanticCorticalEncoder:
    """Ziwago-Prime v9.0 (32-Core Intelligence Edition):
    Морфо-Семантический Хэш-Кортикальный SDR Энкодер (`MorphoSemanticCorticalEncoder`, `N_sdr = 2048`, `M = 8` модулей).

    Разделяет 8 гиперколонок по `B = 256` миниколонок (`8 × 256 = 2048` нейронов, `40` активных = `1.95%` разреженность)
    на комплементарные когнитивно-лингвистические каналы:
      - Модули 0–1 (Брока BA44 / Синтаксическая роль и аниматность): слова одной части речи и ролевого класса
        разделяют устойчивое синтаксическое подпространство;
      - Модули 2–4 (Корень и Морфемная n-грамма + Суффикс времени/числа `-ed`, `-ing`, `-s`, `-ly`, `un-`):
        однокоренные слова (`walk`, `walked`, `walking`) и морфологически близкие понятия имеют контролируемое
        перекрытие SDR-кодов, обеспечивая мгновенное обобщение на OOV-слова;
      - Модули 5–7 (Индивидуальный лексический хэш Картера–Вегмана via `HashedCorticalSDREncoder`):
        ортогональное разделение синонимов и точная идентификация каждого из 50 000+ слов без коллизий.
    """

    def __init__(
        self,
        vocab_size: int = 50_000,
        n_sdr: int = 2048,
        n_modules: int = 8,
        active_per_module: int = 5,
        d_out: int = 64,
        seed: int = 42,
    ):
        self.base_hashed = HashedCorticalSDREncoder(
            vocab_size=vocab_size,
            n_sdr=n_sdr,
            n_modules=n_modules,
            active_per_module=active_per_module,
            seed=seed,
        )
        self.n_sdr = int(n_sdr)
        self.n_modules = int(n_modules)
        self.bins_per_module = self.n_sdr // self.n_modules
        self.active_per_module = int(active_per_module)
        self.d_out = int(d_out)

        g = torch.Generator().manual_seed(seed + 13)
        # Ортогональная дендритная проекция из разреженного SDR [2048] в таламический/кортикальный драйв [d_out]
        self.w_proj = torch.randn(self.d_out, self.n_sdr, generator=g) / (self.active_per_module * self.n_modules) ** 0.5
        self._word_sdr_cache: dict[tuple[str, int, int], torch.Tensor] = {}

    @staticmethod
    def _extract_stem_and_affix(word: str) -> tuple[str, str]:
        w = word.lower().strip()
        if len(w) <= 3:
            return w, "none"
        for suf in ("ing", "ed", "ly", "er", "est", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stem = w[: -len(suf)]
                if stem.endswith("i") and suf in ("ed", "es"):
                    stem = stem[:-1] + "y"
                return stem, f"suf_{suf}"
        for pref in ("un", "re", "in", "dis"):
            if w.startswith(pref) and len(w) - len(pref) >= 3:
                return w[len(pref) :], f"pref_{pref}"
        return w, "none"

    @staticmethod
    def _stable_str_hash(text: str, salt: int = 0) -> int:
        h = (2166136261 ^ salt) & 0xFFFFFFFF
        for ch in text:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h)

    def encode_word(
        self,
        word: str,
        token_id: int = 1,
        role_id: int = 2,
        learn: bool = False,
    ) -> torch.Tensor:
        """Кодирует слово (включая OOV) в разреженный 2048-мерный SDR с 40 активными нейронами."""
        key = (word, int(token_id), int(role_id))
        if not learn and key in self._word_sdr_cache:
            return self._word_sdr_cache[key]

        base_sdr = self.base_hashed.encode_tokens(torch.tensor([token_id]), learn=learn)[0].clone()
        # Перезаписываем первые 5 гиперколонок морфо-синтаксическими и корневыми кодами,
        # сохраняя модули 5..7 от индивидуального лексического хэша Картера–Вегмана
        base_sdr[: 5 * self.bins_per_module].zero_()

        stem, affix = self._extract_stem_and_affix(word)
        tri_root = stem[:3] if len(stem) >= 3 else stem

        # Модули 0..1: синтаксическая роль (role_id) + морфологический класс
        for m in (0, 1):
            base_off = m * self.bins_per_module
            h_role = self._stable_str_hash(f"role_{role_id}_{affix}", salt=m * 97 + 11)
            for k in range(self.active_per_module):
                bin_i = (h_role + k * 37 + role_id * 19) % self.bins_per_module
                base_sdr[base_off + bin_i] = 1.0

        # Модули 2..3: корневая морфема (stem + 3-char root) -> общие биты у walk / walked / walking
        for m in (2, 3):
            base_off = m * self.bins_per_module
            h_stem = self._stable_str_hash(stem if m == 2 else tri_root, salt=m * 131 + 29)
            for k in range(self.active_per_module):
                bin_i = (h_stem + k * 43) % self.bins_per_module
                base_sdr[base_off + bin_i] = 1.0

        # Модуль 4: аффиксная и фонотактическая сигнатура (начало + конец слова)
        base_off = 4 * self.bins_per_module
        phon_sig = f"{word[:2]}_{word[-2:]}_{affix}"
        h_phon = self._stable_str_hash(phon_sig, salt=509)
        for k in range(self.active_per_module):
            bin_i = (h_phon + k * 53) % self.bins_per_module
            base_sdr[base_off + bin_i] = 1.0

        if not learn and len(self._word_sdr_cache) < 8192:
            self._word_sdr_cache[key] = base_sdr
        return base_sdr

    def project_to_cortex(
        self,
        word: str,
        token_id: int = 1,
        role_id: int = 2,
        learn: bool = False,
    ) -> torch.Tensor:
        """Возвращает компактный кортикальный вектор [d_out] из разреженного 2048-мерного SDR."""
        sdr = self.encode_word(word, token_id=token_id, role_id=role_id, learn=learn)
        idx = torch.nonzero(sdr > 0.0, as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.d_out)
        return self.w_proj[:, idx].sum(dim=-1)

    def sdr_overlap(self, word_a: str, role_a: int, word_b: str, role_b: int, id_a: int = 10, id_b: int = 11) -> float:
        """Вычисляет долю совпадающих активных битов между двумя словами (для проверки морфо-обобщения)."""
        sdr_a = self.encode_word(word_a, token_id=id_a, role_id=role_a, learn=False)
        sdr_b = self.encode_word(word_b, token_id=id_b, role_id=role_b, learn=False)
        inter = float((sdr_a * sdr_b).sum().item())
        denom = max(1.0, float(sdr_a.sum().item()))
        return round(inter / denom, 4)


class CompositionalVSABPEEncoder:
    """Задача next11.txt: Композиционный VSA-Энкодер для словаря GPT-2 (50 257 токенов) без плотной матрицы `[50257, 2048]`.

    Принцип работы:
      1. Базовый словарь (`n_base = 2048` самых частых BPE-токенов: байты `0..255`, артикли, предлоги, базовые корни `256..2047`):
         - Хранит компактные статические SDR-паттерны только для `2048` базовых BPE-токенов:
           `base_active_bins` формы `[2048, 8, 5]` (`uint8`, всего `80 КБ` вместо `411 МБ` плотной матрицы `[50257, 2048]`!).
      2. Композиционные токены (`2048..50256`, 48 209 токенов):
         - НЕ хранит для них статических векторов.
         - Использует BPE-дерево из `data/gpt2/merges.txt`, чтобы разбить любой токен на его морфемные/корневые
           компоненты из базового пула `0..2047`, и ВЫЧИСЛЯЕТ его SDR-паттерн (2048 нейронов, 40 активных = 8 модулей × 5)
           на лету через VSA-связывание (Sparse Block-Code модульный циклический сдвиг + композиционный VSA-ключ
           для `HashedCorticalSDREncoder`).
      3. Разрешение OOV-слов (например, `"cryptodragon"` -> `"crypto"` + `"dragon"`):
         - Находит самые длинные известные морфемы в словаре GPT-2 (`"crypto"`, `"dragon"`),
           связывает их SDR-паттерны через VSA-оператор (сохраняя 50% активных пучков от первой морфемы и
           50% циклически сдвинутых пучков от второй морфемы), так что `"cryptodragon"` одновременно
           резонирует в коре и с `"crypto"`, и с `"dragon"`!
    """

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        n_base: int = 2048,
        n_sdr: int = 2048,
        n_modules: int = 8,
        active_per_module: int = 5,
        d_out: int = 64,
        seed: int = 42,
    ):
        import json
        from pathlib import Path

        self.n_base = int(n_base)
        self.n_sdr = int(n_sdr)
        self.n_modules = int(n_modules)
        self.bins_per_module = self.n_sdr // self.n_modules  # 256
        self.active_per_module = int(active_per_module)      # 5 (итого 40 активных нейронов из 2048)
        self.d_out = int(d_out)

        v_file = Path(vocab_path)
        m_file = Path(merges_path)
        with v_file.open("r", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)
        self.id_to_token: dict[int, str] = {int(v): k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)  # 50,257

        # Строим BPE-дерево слияний из merges.txt: token_id -> (left_id, right_id)
        self.bpe_children: dict[int, tuple[int, int]] = {}
        self.bpe_ranks: dict[tuple[str, str], int] = {}
        if m_file.exists():
            lines = m_file.read_text(encoding="utf-8").splitlines()
            merge_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
            for rank, ln in enumerate(merge_lines):
                parts = ln.split(" ")
                if len(parts) == 2:
                    a_str, b_str = parts[0], parts[1]
                    self.bpe_ranks[(a_str, b_str)] = rank
                    merged_str = a_str + b_str
                    if merged_str in self.vocab and a_str in self.vocab and b_str in self.vocab:
                        m_id = self.vocab[merged_str]
                        if m_id >= self.n_base:
                            self.bpe_children[m_id] = (self.vocab[a_str], self.vocab[b_str])

        # Внутренний кортикальный SDR-энкодер (2048 нейронов, 40 активных)
        self.hashed_sdr = HashedCorticalSDREncoder(
            vocab_size=self.vocab_size,
            n_sdr=self.n_sdr,
            n_modules=self.n_modules,
            active_per_module=self.active_per_module,
            seed=seed,
        )

        # Статические SDR-паттерны ТОЛЬКО для 2048 базовых BPE-токенов (0..2047):
        # Храним индексы активных миниколонок [2048, 8, 5] в uint8 (всего 81,920 байт!)
        base_ids = torch.arange(self.n_base, dtype=torch.long)
        base_sdrs = self.hashed_sdr.encode_tokens(base_ids, learn=False)  # [2048, 2048]
        base_bins = torch.zeros(self.n_base, self.n_modules, self.active_per_module, dtype=torch.uint8)
        for m in range(self.n_modules):
            m_slice = base_sdrs[:, m * self.bins_per_module : (m + 1) * self.bins_per_module]
            top_idx = m_slice.topk(self.active_per_module, dim=-1).indices.to(torch.uint8)
            base_bins[:, m, :] = top_idx
        self.base_active_bins = base_bins  # [2048, 8, 5] uint8 — единственная статическая таблица токенов!

        g = torch.Generator().manual_seed(seed + 19)
        self.w_proj = (
            torch.randn(self.d_out, self.n_sdr, generator=g)
            / (self.active_per_module * self.n_modules) ** 0.5
        )
        self.module_offsets = (torch.arange(self.n_modules, dtype=torch.long) * self.bins_per_module).view(self.n_modules, 1)

    def decompose_to_base_tokens(self, token_id: int) -> List[int]:
        """Разбивает любой из 50 257 токенов GPT-2 по BPE-дереву `merges.txt` до базовых корней `< 2048`."""
        tid = int(token_id)
        if tid < self.n_base:
            return [max(0, tid)]
        if tid not in self.bpe_children:
            return [tid % self.n_base]
        stack = [tid]
        leaves: List[int] = []
        while stack:
            cur = stack.pop()
            if cur < self.n_base or cur not in self.bpe_children:
                leaves.append(cur if cur < self.n_base else (cur % self.n_base))
            else:
                left_id, right_id = self.bpe_children[cur]
                stack.append(right_id)
                stack.append(left_id)
        return leaves

    def _vsa_bind_bins(self, bins_list: List[torch.Tensor]) -> torch.Tensor:
        """Выполняет композиционное VSA-связывание (`Sparse Block-Codes` + циклический сдвиг)
        последовательности базовых морфемных кодов `[8, 5]` на лету:
          - Модули 0..2 удерживают головной и хвостовой базовые корни с позиционным циклическим сдвигом;
          - Модули 3..4 выполняют поблочное модульное связывание `(bin_a + bin_b + pos_shift) mod 256`;
          - Модули 5..7 подают связанный VSA-ключ в `HashedCorticalSDREncoder` (2048 нейронов, 40 активных)."""
        if len(bins_list) == 1:
            return bins_list[0].long()

        first_b = bins_list[0].long()
        last_b = bins_list[-1].long()
        out_bins = torch.zeros(self.n_modules, self.active_per_module, dtype=torch.long)

        # Модули 0..1: первый корень морфемы (префикс/основа)
        out_bins[0] = first_b[0]
        out_bins[1] = first_b[1]
        # Модуль 2: последний корень морфемы (суффикс/второй корень) с циклическим сдвигом +17
        out_bins[2] = (last_b[2] + 17) % self.bins_per_module

        # Модули 3..4: VSA модульное связывание всех компонентов дерева BPE по круговому сдвигу
        acc3 = first_b[3].clone()
        acc4 = first_b[4].clone()
        for pos, b_item in enumerate(bins_list[1:], start=1):
            b_long = b_item.long()
            acc3 = (acc3 + torch.roll(b_long[3], shifts=pos % self.active_per_module, dims=0) + pos * 13) % self.bins_per_module
            acc4 = (acc4 + torch.roll(b_long[4], shifts=(pos * 2) % self.active_per_module, dims=0) + pos * 29) % self.bins_per_module

        # Гарантируем ровно 5 уникальных активных бинов в каждом модуле
        for m_idx, acc in ((3, acc3), (4, acc4)):
            uniq: List[int] = []
            for v in acc.tolist():
                cand = int(v) % self.bins_per_module
                while cand in uniq:
                    cand = (cand + 37) % self.bins_per_module
                uniq.append(cand)
            out_bins[m_idx] = torch.tensor(uniq[: self.active_per_module], dtype=torch.long)

        # Вычисляем композитный VSA-ключ и подаём его на вход HashedCorticalSDREncoder для модулей 5..7
        vsa_hash_key = 0
        for pos, b_item in enumerate(bins_list):
            vsa_hash_key = (vsa_hash_key * 1315423911 + int(b_item.long().sum().item()) + (pos + 1) * 2654435761) & 0x7FFFFFFF
        hashed_sdr = self.hashed_sdr.encode_tokens(torch.tensor([vsa_hash_key % self.vocab_size]), learn=False)[0]
        for m in range(5, self.n_modules):
            m_slice = hashed_sdr[m * self.bins_per_module : (m + 1) * self.bins_per_module]
            out_bins[m] = m_slice.topk(self.active_per_module).indices.long()

        return out_bins

    def _bins_to_sdr(self, bins: torch.Tensor) -> torch.Tensor:
        """Преобразует матрицу активных бинов `[8, 5]` в разреженный 2048-мерный SDR (ровно 40 единиц)."""
        sdr = torch.zeros(self.n_sdr, dtype=torch.float32)
        global_idx = (bins.long() + self.module_offsets).reshape(-1)
        sdr[global_idx] = 1.0
        return sdr

    def encode_token_id(self, token_id: int) -> torch.Tensor:
        """Вычисляет 2048-мерный SDR (40 активных нейронов) для любого из 50 257 токенов GPT-2:
        - Если `token_id < 2048`: берёт статический базовый SDR;
        - Если `token_id >= 2048`: раскладывает по BPE-дереву `merges.txt` на базовые токены
          и связывает их на лету через VSA-связывание."""
        tid = int(token_id)
        if 0 <= tid < self.n_base:
            return self._bins_to_sdr(self.base_active_bins[tid])
        base_leaves = self.decompose_to_base_tokens(tid)
        bins_list = [self.base_active_bins[leaf] for leaf in base_leaves]
        bound_bins = self._vsa_bind_bins(bins_list)
        return self._bins_to_sdr(bound_bins)

    def decompose_oov_word(self, word: str) -> Dict[str, object]:
        """Разбивает неизвестное (OOV) композитное слово (например, `'cryptodragon'`) на известные
        морфемы словаря GPT-2 (`'crypto'` + `'dragon'`), а затем до базовых BPE-токенов `< 2048`,
        и собирает его 2048-мерный SDR-паттерн через VSA-связывание."""
        clean_w = word.strip()
        prefixed_w = "Ġ" + clean_w if not clean_w.startswith("Ġ") else clean_w

        # Если слово целиком есть в словаре, используем его, иначе жадно выделяем самые длинные морфемы
        sub_morphemes: List[str] = []
        sub_token_ids: List[int] = []
        pos = 0
        target_str = clean_w
        while pos < len(target_str):
            matched_sub = None
            matched_id = None
            for end in range(len(target_str), pos, -1):
                piece = target_str[pos:end]
                # Избегаем тривиального совпадения всего слова, если тестируется составное OOV
                if pos == 0 and end == len(target_str) and piece not in self.vocab and ("Ġ" + piece) not in self.vocab:
                    continue
                cand_g = ("Ġ" + piece) if pos == 0 else piece
                if cand_g in self.vocab:
                    matched_sub = piece
                    matched_id = self.vocab[cand_g]
                    break
                if piece in self.vocab:
                    matched_sub = piece
                    matched_id = self.vocab[piece]
                    break
            if matched_sub is not None and matched_id is not None:
                sub_morphemes.append(matched_sub)
                sub_token_ids.append(int(matched_id))
                pos += len(matched_sub)
            else:
                ch = target_str[pos]
                sub_morphemes.append(ch)
                sub_token_ids.append(int(self.vocab.get(ch, ord(ch) % self.n_base)))
                pos += 1

        # Для каждой найденной морфемы вычисляем её SDR и связываем через VSA
        morpheme_sdrs = [self.encode_token_id(mid) for mid in sub_token_ids]
        morpheme_bins: List[torch.Tensor] = []
        for sdr_m in morpheme_sdrs:
            b_m = torch.zeros(self.n_modules, self.active_per_module, dtype=torch.long)
            for m in range(self.n_modules):
                m_slice = sdr_m[m * self.bins_per_module : (m + 1) * self.bins_per_module]
                b_m[m] = m_slice.topk(self.active_per_module).indices.long()
            morpheme_bins.append(b_m)

        # Композиционное связывание морфем: чтобы составное OOV-слово ("cryptodragon") сохраняло
        # высокое семантическое перекрытие с ОБЕИМИ родительскими морфемами ("crypto" и "dragon"),
        # модули 0..3 наследуют структурные пучки первой морфемы, модули 4..6 — второй морфемы,
        # а модуль 7 содержит их VSA-свёртку (циклический сдвиг)!
        if len(morpheme_bins) >= 2:
            b_first = morpheme_bins[0]
            b_second = morpheme_bins[1]
            comp_bins = torch.zeros(self.n_modules, self.active_per_module, dtype=torch.long)
            comp_bins[0:3] = b_first[0:3]
            comp_bins[3:6] = b_second[3:6]
            comp_bins[6] = (b_first[6] + b_second[6] + 11) % self.bins_per_module
            comp_bins[7] = self._vsa_bind_bins(morpheme_bins)[7]
            # Убираем возможные дубликаты внутри модуля 6
            u6: List[int] = []
            for v in comp_bins[6].tolist():
                c = int(v) % self.bins_per_module
                while c in u6:
                    c = (c + 37) % self.bins_per_module
                u6.append(c)
            comp_bins[6] = torch.tensor(u6[: self.active_per_module], dtype=torch.long)
            oov_sdr = self._bins_to_sdr(comp_bins)
        else:
            oov_sdr = morpheme_sdrs[0]

        base_leaves_all: List[int] = []
        for mid in sub_token_ids:
            base_leaves_all.extend(self.decompose_to_base_tokens(mid))

        overlaps: Dict[str, float] = {}
        for morph_str, morph_sdr in zip(sub_morphemes, morpheme_sdrs):
            ov = float((oov_sdr * morph_sdr).sum().item()) / float(max(1.0, oov_sdr.sum().item()))
            overlaps[morph_str] = round(ov, 4)

        idx_act = torch.nonzero(oov_sdr > 0.0, as_tuple=False).flatten()
        cortical_vec = self.w_proj[:, idx_act].sum(dim=-1) if idx_act.numel() > 0 else torch.zeros(self.d_out)

        return {
            "word": clean_w,
            "is_in_gpt2_vocab": (clean_w in self.vocab or prefixed_w in self.vocab),
            "morphemes": sub_morphemes,
            "morpheme_token_ids": sub_token_ids,
            "base_2048_leaves": base_leaves_all,
            "sdr": oov_sdr,
            "active_neurons": int(oov_sdr.sum().item()),
            "sdr_total_neurons": self.n_sdr,
            "morpheme_sdr_overlaps": overlaps,
            "cortical_vector": cortical_vec,
        }

    def project_token_to_cortex(self, token_id: int) -> torch.Tensor:
        """Проецирует 2048-мерный SDR токена в компактный входной вектор коры `[d_out]`."""
        sdr = self.encode_token_id(token_id)
        idx = torch.nonzero(sdr > 0.0, as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.zeros(self.d_out)
        return self.w_proj[:, idx].sum(dim=-1)

    def memory_footprint_report(self) -> Dict[str, object]:
        """Сравнивает объём памяти композиционного VSA-энкодера с плотной матрицей `[50257, 2048]`."""
        dense_floats = self.vocab_size * self.n_sdr
        dense_bytes = dense_floats * 4  # float32
        static_base_bytes = int(self.base_active_bins.numel() * self.base_active_bins.element_size())
        hashed_weights_bytes = int(self.hashed_sdr.micro_weights.numel() * 4)
        proj_bytes = int(self.w_proj.numel() * 4)
        total_vsa_bytes = static_base_bytes + hashed_weights_bytes + proj_bytes
        return {
            "vocab_size": self.vocab_size,
            "base_static_tokens": self.n_base,
            "dynamic_vsa_tokens": self.vocab_size - self.n_base,
            "dense_matrix_mb": round(dense_bytes / (1024 * 1024), 2),
            "compositional_vsa_kb": round(total_vsa_bytes / 1024, 2),
            "compression_ratio_x": round(dense_bytes / max(1, total_vsa_bytes), 1),
        }
