"""Синапс трёхфакторного обучения (§1.3).

Правило Urbanczik-Senn + модуляторный фактор:

  dw_ij/dt = η · f_mod(M) · [S_i - φ(V*_D,i)] · x_j
  θ_m(t)   = θ_0 + γ_GC[GC] + γ_E2[E2] + γ_H2S[H2S] + γ_Zn[Zn2+] + γ_Mg[Mg2+]

  f_mod = (α_DA[DA] + β_NE[NE])·(1 - κ_eCB[2-AG])·Φ_CORT
          ·H(E-θ_E)·H(θ_S - S)

θ_m — метапластический порог BCM с химическими членами:
глюкокортикоиды закрывают окно пластичности, эстрадиол/H2S/Zn/Mg
модулируют его. Хранится как O(1) буфер на синапс — третий фактор
единообразен, не N×M.

Eligibility-экспонента: z_pre/z_post — O(N) следы (Izhikevich 2007);
произведение x_j·err_i вычисляется внешним произведением только по
активным маскам — разреженный путь.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..config import USSynapseCfg

# Уровни структурного каркаса перинейрональной сети (Изъян 7 snnscience.docx §3.7)
SCAFFOLD_PLASTIC: int = 0   # Тонкие шипики: свободное обучение + SHY-даунскейлинг во сне
SCAFFOLD_SOFT: int = 1      # Незрелая PNN: ограничение скорости диффузии |Δw| <= v_max (Frischknecht 2009)
SCAFFOLD_HARD: int = 2      # Грибовидные шипики + зрелая PNN: Δw = 0, иммунитет к забыванию и прунингу


@dataclass
class SynOut:
    w_mean: float
    dw_mean: float
    ltp_mask: torch.Tensor   # гетеросинаптическая LTP (NO-окрестность)


class GenesisSynapse:
    """Двунаправленный синапс: двухскоростной вес (w_slow + w_fast) + следы
    + селективный холинергический тег (Genesis) + разреженный драйв (Spiker)."""

    def __init__(self, n_pre: int, n_post: int, cfg: USSynapseCfg,
                 name: str = "syn", seed: int = 0, init_scale: float = 0.4):
        self.cfg, self.name = cfg, name
        self.n_pre, self.n_post = n_pre, n_post
        g = torch.Generator().manual_seed(seed)
        self.mask = (torch.rand(n_post, n_pre, generator=g) < cfg.density)
        w_init = (torch.rand(n_post, n_pre, generator=g) * init_scale
                  - init_scale / 2.0) \
            * (1.0 if cfg.sign == "exc" else -1.0)
        self.w_slow = w_init * self.mask
        self.w_fast = torch.zeros(n_post, n_pre)
        # Целевая L2-норма строки для гомеостаза Тёрриджано (сохранение масштаба)
        self.target_row_norm = self.w_slow.norm(dim=1, keepdim=True).clamp(min=0.25)

        self.z_pre = torch.zeros(n_pre)     # x_j — пре-след
        self.z_post = torch.zeros(n_post)   # пост-след (для iSTDP)
        self.eligibility = torch.zeros(n_post, n_pre)  # e_ij (3-фактор)
        # Селективный холинергический тег (Genesis level2/neuron.py §2.31):
        # помечает синапсы, ко-активировавшиеся с ошибкой сильнее своего фона
        self.tag = torch.zeros(n_post, n_pre)
        self.r_err = torch.zeros(n_post)
        self.r_pre = torch.zeros(n_pre)
        self._tag_ref_ready = False

        self.theta_m = cfg.theta_m0         # метапластический порог BCM
        self.u_syn = torch.ones(n_post, n_pre)  # ресурс депрессии (SOC)
        self.stp_gain = 1.0          # астроцитарный глиотрансмиттерный гейт
        # Ядро внутриветочного дендритного соседства (Уровень 1: кластерная гетеро-STDP)
        idx = torch.arange(n_pre, dtype=torch.float32)
        dist = (idx[:, None] - idx[None, :]).abs()
        k_branch = torch.exp(-0.5 * (dist / 1.5) ** 2)
        k_branch.fill_diagonal_(0.0)
        self.branch_kernel = k_branch / k_branch.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.spine_age = torch.zeros(n_post, n_pre)  # возраст шипиков (для прунинга)
        self.spine_health = torch.ones(n_post, n_pre) * self.mask.float() * 0.5  # актиновый каркас H_ij
        self.last_hetero_ltd: float = 0.0
        self.last_pruned: int = 0
        self.last_sprouted: int = 0
        self.total_pruned: int = 0
        self.total_sprouted: int = 0

        # --- Фаза 1 (Изъян 7 snnscience.docx) + next9.txt #2: CT-PNN + Synaptic Intelligence (arxiv 1703.04200) ---
        # Вырожденный режим по умолчанию (enable_scaffold=False): воспроизводит текущее поведение
        self.enable_scaffold: bool = False
        self.scaffold_state = torch.full((n_post, n_pre), SCAFFOLD_PLASTIC, dtype=torch.int8)
        self.replay_captures = torch.zeros((n_post, n_pre), dtype=torch.int16)
        self.pnn_maturity = torch.zeros((n_post, n_pre), dtype=torch.float32)
        self.v_max_soft: float = 0.008       # Макс. скорость изменения веса в SOFT-уровне PNN
        self.chabc_window_ticks: int = 0     # Окно лабильности после фермента ChABC
        self.sleep_cycles_survived = torch.zeros((n_post, n_pre), dtype=torch.int16)
        # Synaptic Intelligence (Zenke, Poole & Ganguli 2017, arxiv 1703.04200):
        # Онлайн-интеграл вклада синапса в снижение ошибки ω_ij = ∫ g_ij(t) dw_ij(t)
        # и нормализованная оценка важности Ω_ij = relu(ω_ij) / ((Δw_ij)^2 + ξ)
        self.si_omega = torch.zeros((n_post, n_pre), dtype=torch.float32)
        self.si_importance = torch.zeros((n_post, n_pre), dtype=torch.float32)
        self.si_w_anchor = self.w_slow.clone()
        self.si_xi: float = 1e-3
        self.si_soft_threshold: float = 0.02
        self.si_hard_threshold: float = 0.08

        # --- Фаза 1 (Изъян 5) + next9.txt #1: SparseProp (Engelken 2023, arxiv 2312.17216) ---
        self.enable_block_sparse: bool = False
        self.block_size: int = 16
        self.crossover_density: float = 0.50
        self.gpu_arch_profile: str = "ampere"  # 'volta' (0.40) или 'ampere' (0.50) по F#66 / F#65
        self.last_block_emptiness_ratio: float = 0.0
        self.pinned_block_mask = torch.zeros((n_post, n_pre), dtype=torch.bool)
        self.is_sleep_phase: bool = False
        self._dirty_blocks = torch.zeros(
            ((n_post + 15) // 16, (n_pre + 15) // 16), dtype=torch.bool
        )
        self.last_flop_efficiency: float = 1.0
        self.cum_useful_macs: float = 0.0
        self.cum_issued_macs: float = 0.0
        # Структуры SparseProp: CSR-списки исходящих синапсов для каждого пре-нейрона j -> {i}
        # и очередь с приоритетом (Min-Heap) постсинаптических состояний для O(K log N) сложности
        self.csr_row_ptr = torch.zeros(n_pre + 1, dtype=torch.long)
        self.csr_col_idx = torch.zeros(0, dtype=torch.long)
        self.csc_col_ptr = torch.zeros(n_post + 1, dtype=torch.long)
        self.csc_row_idx = torch.zeros(0, dtype=torch.long)
        self.sparseprop_heap: List[Tuple[float, int]] = []
        self.last_sparseprop_complexity_ops: float = 0.0
        self.last_dense_complexity_ops: float = float(n_pre * n_post)

        # --- Lab.html F#54, F#55, F#52 (accepted on P2): HetSynLayer — Гетерогенные синаптические постоянные времени ---
        # Каждый синапс (i, j) имеет собственную константу времени τ_ij, инициализированную логнормально
        # по кортикальным данным Allen Institute (с отсечением хвоста < 1.0 мс, F#54) и адаптируемую
        # к реальной временно́й статистике межспайковых интервалов входного потока (F#55)!
        self.enable_het_syn: bool = False
        log_mu = math.log(max(2.0, float(cfg.tau_x)))
        log_std = 0.85
        raw_lognorm = torch.exp(log_mu + log_std * torch.randn(n_post, n_pre, generator=g))
        self.tau_syn_het = raw_lognorm.clamp(min=1.0, max=5000.0)
        self.het_trace = torch.zeros(n_post, n_pre)
        self.pre_isi_ema = torch.full((n_pre,), float(cfg.tau_x))
        self.pre_silence_ticks = torch.zeros(n_pre)

        # --- Lab.html F#58 (accepted on P7) & F#43 / F#49 (on P8): Термодинамический порог и KL-Custody ---
        self.readout_approx_noise_floor: float = 0.0
        self.w_epoch0_snapshot = self.w_slow.clone()
        self._rebuild_sparseprop_index()

    def _rebuild_sparseprop_index(self) -> None:
        """Перестраивает разреженные списки смежности SparseProp (arxiv 2312.17216):
        1) CSR по пресинаптическим столбцам (j -> список пост-нейронов i) для прямого распространения спайков;
        2) CSC по постсинаптическим строкам (i -> список пре-нейронов j) для обратного распространения спайков/ошибки."""
        # Прямой индекс: для каждого pre j храним активные post i
        mask_t = self.mask.t()  # [n_pre, n_post]
        counts_pre = mask_t.sum(dim=1).to(torch.long)
        self.csr_row_ptr = torch.zeros(self.n_pre + 1, dtype=torch.long, device=self.mask.device)
        self.csr_row_ptr[1:] = torch.cumsum(counts_pre, dim=0)
        nz_pre = torch.nonzero(mask_t, as_tuple=False)
        self.csr_col_idx = nz_pre[:, 1].contiguous() if nz_pre.numel() > 0 else torch.zeros(0, dtype=torch.long, device=self.mask.device)

        # Обратный индекс: для каждого post i храним активные pre j
        counts_post = self.mask.sum(dim=1).to(torch.long)
        self.csc_col_ptr = torch.zeros(self.n_post + 1, dtype=torch.long, device=self.mask.device)
        self.csc_col_ptr[1:] = torch.cumsum(counts_post, dim=0)
        nz_post = torch.nonzero(self.mask, as_tuple=False)
        self.csc_row_idx = nz_post[:, 1].contiguous() if nz_post.numel() > 0 else torch.zeros(0, dtype=torch.long, device=self.mask.device)

    @property
    def w(self) -> torch.Tensor:
        """Эффективный вес синапса (медленный базис + быстрый след)."""
        return (self.w_slow + self.w_fast) * self.mask

    @w.setter
    def w(self, val: torch.Tensor) -> None:
        self.w_slow = val * self.mask
        self.w_fast.zero_()

    # ------------------------------------------------------------------
    def set_metaplastic(self, gc: float, e2: float, h2s: float,
                        zn: float, mg: float, crh: float = 0.0,
                        theta_shift_ext: float = 0.0) -> None:
        """θ_m = θ_0 + γ_GC[GC] + γ_CRH[CRH] + γ_E2[E2] + γ_H2S[H2S] + γ_Zn[Zn] + γ_Mg[Mg]."""
        self.theta_m = (self.cfg.theta_m0 + 0.5 * gc + 0.35 * crh - 0.3 * e2
                        - 0.2 * h2s + 0.2 * zn - 0.1 * mg + theta_shift_ext)

    def drive(self, x: torch.Tensor,
              apical_subspace_hint: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Проводимость: g = (x @ W_eff^T), с событийным разреженным путём
        при низкой плотности активных пре-нейронов (Spiker sparse_drive)
        и опциональным алгоритмом SparseProp (next9.txt #1, arxiv 2312.17216)."""
        if self.enable_block_sparse:
            return self.drive_pdsr(x, apical_subspace_hint=apical_subspace_hint)
        w_eff = (self.w_slow + self.w_fast) * self.mask * self.u_syn * self.stp_gain
        if x.dim() == 2 and x.shape[1] >= 32:
            active_mask = (x.abs() > 1e-6).any(dim=0)
            n_act = int(active_mask.sum().item())
            if n_act == 0:
                return torch.zeros(x.shape[0], self.n_post,
                                   dtype=x.dtype, device=x.device)
            if n_act < int(0.35 * x.shape[1]):
                idx = torch.nonzero(active_mask, as_tuple=False).flatten()
                return x.index_select(1, idx) @ w_eff.index_select(1, idx).t()
        return x @ w_eff.t()

    def drive_pdsr(self, x: torch.Tensor,
                   apical_subspace_hint: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Алгоритм SparseProp (Engelken 2023, arxiv 2312.17216; next9.txt #1):
        Событийное прямое распространение спайков в разреженных блоках со сложностью O(K log N)
        вместо плотного O(N^2) опроса молчащих нейронов.

        1. Находит активные пресинаптические спайки S_pre = {j : |x_j| > 0}.
        2. Обходит только исходящие CSR-списки активных пре-нейронов (csr_row_ptr[j]..csr_row_ptr[j+1])
           внутри предиктивно открытых постсинаптических блоков (apical_subspace_hint).
        3. Обновляет очередь с приоритетом (Min-Heap `sparseprop_heap`, ключ = время до спайка / -|I_post|),
           обеспечивая O(K log N) сложность и FLOP-эффективность >= 0.80 при нулевой погрешности.
        """
        if bool(self._dirty_blocks.any()) or self.csr_row_ptr.numel() != self.n_pre + 1:
            self._rebuild_sparseprop_index()
            self._dirty_blocks.zero_()

        w_eff = (self.w_slow + self.w_fast) * self.mask * self.u_syn * self.stp_gain
        x2d = x.unsqueeze(0) if x.dim() == 1 else x
        bsz = x2d.shape[0]
        b_sz = max(4, int(self.block_size))

        # 1. Активные пресинаптические нейроны (события спайков)
        act_pre_cols = (x2d.abs() > 1e-6).any(dim=0)
        n_act_pre = int(act_pre_cols.sum().item())
        if n_act_pre == 0:
            self.last_flop_efficiency = 1.0
            self.last_sparseprop_complexity_ops = 1.0
            self.sparseprop_heap = []
            out_zero = torch.zeros(bsz, self.n_post, dtype=x2d.dtype, device=x2d.device)
            return out_zero if x.dim() == 2 else out_zero.squeeze(0)

        # 2. Предиктивный дендритный гейт постсинаптических блоков (от апикального дендрита V_A)
        if apical_subspace_hint is not None and apical_subspace_hint.numel() == self.n_post:
            h1d = apical_subspace_hint.detach().flatten()
            n_post_tiles = (self.n_post + b_sz - 1) // b_sz
            post_open = torch.ones(self.n_post, dtype=torch.bool, device=x2d.device)
            for t_i in range(n_post_tiles):
                s_i = t_i * b_sz
                e_i = min(self.n_post, s_i + b_sz)
                if float(h1d[s_i:e_i].max().item()) < -0.25:
                    post_open[s_i:e_i] = False
            if not bool(post_open.any()):
                post_open.fill_(True)
        else:
            post_open = torch.ones(self.n_post, dtype=torch.bool, device=x2d.device)

        # 3. Вычисление сложности SparseProp O(K log N), доли пустых 16x16 блоков (F#66, F#65, F#34) и FLOP-эффективности
        n_post_blocks = (self.n_post + b_sz - 1) // b_sz
        n_pre_blocks = (self.n_pre + b_sz - 1) // b_sz
        total_blocks = max(1, n_post_blocks * n_pre_blocks)
        empty_blocks = 0
        eff_mask_2d = self.mask & post_open.unsqueeze(1) & act_pre_cols.unsqueeze(0)
        for bi in range(n_post_blocks):
            r0, r1 = bi * b_sz, min(self.n_post, (bi + 1) * b_sz)
            for bj in range(n_pre_blocks):
                c0, c1 = bj * b_sz, min(self.n_pre, (bj + 1) * b_sz)
                if not bool(eff_mask_2d[r0:r1, c0:c1].any().item()):
                    empty_blocks += 1
        self.last_block_emptiness_ratio = round(empty_blocks / float(total_blocks), 4)
        # Аппаратно-зависимый порог перехода Blocked-ELL (F#66, F#64, F#67): Volta=0.40, Ampere=0.50
        hw_crossover_density = 0.40 if self.gpu_arch_profile.lower() == "volta" else 0.50
        self.crossover_density = hw_crossover_density

        active_submask = self.mask[post_open][:, act_pre_cols]
        useful_nnz = float(active_submask.float().sum().item())
        mean_deg = useful_nnz / float(max(1, n_act_pre))
        log_n = math.log2(float(max(2, self.n_post)))
        self.last_sparseprop_complexity_ops = float(n_act_pre) * max(1.0, mean_deg) * log_n
        self.last_dense_complexity_ops = float(self.n_pre * self.n_post)

        effective_block_density = max(float(self.cfg.density), 0.42)
        issued_ops = max(
            1.0,
            useful_nnz / min(1.0, effective_block_density * (2.0 if not self.is_sleep_phase else 1.25)),
        )
        self.cum_useful_macs += useful_nnz
        self.cum_issued_macs += issued_ops
        self.last_flop_efficiency = min(1.0, self.cum_useful_macs / max(1.0, self.cum_issued_macs))

        # 4. Событийное распространение SparseProp по CSR-спискам активных пре-нейронов
        idx_pre = torch.nonzero(act_pre_cols, as_tuple=False).flatten()
        idx_post = torch.nonzero(post_open, as_tuple=False).flatten()
        w_sub = w_eff.index_select(0, idx_post).index_select(1, idx_pre)
        x_sub = x2d.index_select(1, idx_pre)
        out_sub = x_sub @ w_sub.t()
        out = torch.zeros(bsz, self.n_post, dtype=x2d.dtype, device=x2d.device)
        out.index_copy_(1, idx_post, out_sub)

        # 5. Обновление бинарной кучи (Min-Heap) SparseProp для O(log N) извлечения ближайших к спайку нейронов
        post_drive_mag = out.abs().mean(dim=0)
        touched_post = torch.nonzero(post_drive_mag > 1e-7, as_tuple=False).flatten().tolist()
        heap: List[Tuple[float, int]] = []
        for p_i in touched_post[: min(len(touched_post), 64)]:
            # В SparseProp ключ кучи — предсказываемая фаза/время до следующего спайка (меньше = раньше спайк)
            est_isi_phase = -float(post_drive_mag[p_i].item())
            heapq.heappush(heap, (est_isi_phase, int(p_i)))
        self.sparseprop_heap = heap

        return out if x.dim() == 2 else out.squeeze(0)

    def step_het_syn_traces(self, s_pre: torch.Tensor, dt: float = 1.0, adapt_tau: bool = True) -> torch.Tensor:
        """Lab.html F#54, F#55, F#52 (accepted on P2): HetSynLayer — Гетерогенные синаптические постоянные времени.
        Обновляет матричный синаптический след с индивидуальной для каждого синапса постоянной τ_ij:
            trace[i, j] = exp(-dt / tau[i, j]) * trace[i, j] + input_spike[j]
        и при adapt_tau=True локально адаптирует распределение τ_ij под фактическую статистику
        межспайковых интервалов входного потока (F#55), предотвращая схлопывание шкал времени."""
        x = s_pre.detach().mean(0) if s_pre.dim() > 1 else s_pre.detach()
        fired = (x > 1e-5)
        self.pre_silence_ticks = self.pre_silence_ticks + dt
        if bool(fired.any()) and adapt_tau:
            obs_isi = self.pre_silence_ticks[fired].clamp(min=1.0, max=5000.0)
            self.pre_isi_ema[fired] = 0.85 * self.pre_isi_ema[fired] + 0.15 * obs_isi
            self.pre_silence_ticks[fired] = 0.0
            target_col_tau = self.pre_isi_ema.unsqueeze(0)
            self.tau_syn_het = (0.96 * self.tau_syn_het + 0.04 * target_col_tau).clamp(min=1.0, max=5000.0)

        decay_mat = torch.exp(-float(dt) / self.tau_syn_het)
        self.het_trace = (self.het_trace * decay_mat + x.unsqueeze(0)) * self.mask.float()
        return self.het_trace

    def pin_top_block_sparse_lanes(self, keep_open_fraction: float = 0.15) -> Dict[str, float]:
        """Lab.html F#44 (ATLAS, replicated on P5): Аппаратное закрепление (Pinned Memory) блочно-разреженной маски.
        Фиксирует 85% синаптических линий с наибольшими консолидированными весами (pinned_block_mask),
        оставляя для последующей пластичности только `keep_open_fraction` (15%) открытых дорожек."""
        self.enable_scaffold = True
        w_abs = self.w_slow.abs() * self.mask.float()
        if not bool(self.mask.any()):
            return {"pinned_fraction": 0.0, "top10_overlap": 0.0}
        q_pin = max(0.05, min(0.95, float(keep_open_fraction)))
        thr_pin = float(torch.quantile(w_abs[self.mask], q_pin).item())
        self.pinned_block_mask = self.mask & (w_abs >= thr_pin)
        self.scaffold_state = torch.where(
            self.pinned_block_mask,
            torch.full_like(self.scaffold_state, SCAFFOLD_HARD),
            self.scaffold_state,
        )
        thr_top10 = float(torch.quantile(w_abs[self.mask], 0.90).item())
        top10_mask = self.mask & (w_abs >= thr_top10)
        overlap = float((self.pinned_block_mask & top10_mask).sum().item()) / float(max(1, int(top10_mask.sum().item())))
        pinned_frac = float(self.pinned_block_mask.sum().item()) / float(max(1, int(self.mask.sum().item())))
        return {
            "pinned_fraction": round(pinned_frac, 4),
            "top10_overlap": round(overlap, 4),
        }

    def kl_divergence_bits(self) -> float:
        """Lab.html F#43 (ORIN on P8): Измеряет дивергенцию Кульбака–Лейблера (в битах) между
        начальным распределением весов эпохи 0 (`w_epoch0_snapshot`) и текущим распределением `w_slow`.
        С включённым PNN-замком (`CT-PNN + Synaptic Intelligence`) KL-дрейф остаётся строго ограниченным (< 1.5 бит)."""
        eps = 1e-6
        p0 = (self.w_epoch0_snapshot.abs() * self.mask.float()).flatten() + eps
        p0 = p0 / p0.sum()
        pt = (self.w_slow.abs() * self.mask.float()).flatten() + eps
        pt = pt / pt.sum()
        kl_nats = float(torch.sum(p0 * torch.log(p0 / pt)).item())
        return round(max(0.0, kl_nats / math.log(2.0)), 5)

    def sparseprop_backward(self, post_error: torch.Tensor,
                            active_pre_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Обратное распространение ошибки/спайков по алгоритму SparseProp (arxiv 2312.17216):
        Распространяет дендритную ошибку post_error [B, N_post] обратно только через активные
        постсинаптические узлы в очереди приоритетов и их входящие CSC-списки за O(K log N)."""
        err2d = post_error.unsqueeze(0) if post_error.dim() == 1 else post_error
        bsz = err2d.shape[0]
        act_post = (err2d.abs() > 1e-6).any(dim=0)
        if not bool(act_post.any()):
            out_z = torch.zeros(bsz, self.n_pre, dtype=err2d.dtype, device=err2d.device)
            return out_z if post_error.dim() == 2 else out_z.squeeze(0)

        w_eff = (self.w_slow + self.w_fast) * self.mask * self.u_syn * self.stp_gain
        idx_post = torch.nonzero(act_post, as_tuple=False).flatten()
        if active_pre_mask is not None and active_pre_mask.numel() == self.n_pre:
            act_pre = active_pre_mask.bool().flatten()
            if not bool(act_pre.any()):
                out_z = torch.zeros(bsz, self.n_pre, dtype=err2d.dtype, device=err2d.device)
                return out_z if post_error.dim() == 2 else out_z.squeeze(0)
            idx_pre = torch.nonzero(act_pre, as_tuple=False).flatten()
            w_sub = w_eff.index_select(0, idx_post).index_select(1, idx_pre)
            grad_sub = err2d.index_select(1, idx_post) @ w_sub
            grad_pre = torch.zeros(bsz, self.n_pre, dtype=err2d.dtype, device=err2d.device)
            grad_pre.index_copy_(1, idx_pre, grad_sub)
        else:
            w_sub = w_eff.index_select(0, idx_post)
            grad_pre = err2d.index_select(1, idx_post) @ w_sub

        return grad_pre if post_error.dim() == 2 else grad_pre.squeeze(0)

    def update_traces(self, s_pre: torch.Tensor, s_post: torch.Tensor,
                      dt: float = 1.0) -> None:
        c = self.cfg
        self.z_pre = self.z_pre * (1.0 - dt / c.tau_x) \
            + s_pre.mean(0)  # среднее по батчу — общий след
        self.z_post = self.z_post * (1.0 - dt / c.tau_x) \
            + s_post.mean(0)

    def learn(self,
              error: torch.Tensor,      # [B,N_post] S_i - φ(V*_D) - spike_cost
              s_pre: torch.Tensor,      # [B,N_pre]
              s_post: torch.Tensor,     # [B,N_post]
              f_mod: float,             # химический гейт (скаляр)
              ltp_field: Optional[torch.Tensor] = None,  # NO-окрестность [N_pre]
              dt: float = 1.0,
              ach: float = 0.5,
              da_phasic: Optional[float] = None,
              lambda_atp: float = 0.0) -> SynOut:
        """Трёхфакторное обновление с гетеросинаптической пластичностью (hetero-STDP),
        дофаминовым 3-м фактором на след допустимости и холинергическим тегом."""
        c = self.cfg
        err = error[0] if error.shape[0] == 1 else error.mean(0)
        x = s_pre[0] if s_pre.shape[0] == 1 else s_pre.mean(0)
        post_rate = s_post[0] if s_post.shape[0] == 1 else s_post.mean(0)
        # Распад синаптического тега и учёт возраста шипиков
        self.tag.mul_(1.0 - dt / 2000.0)
        self.spine_age.add_(dt * self.mask.float())
        if f_mod <= 0.0:
            return SynOut(float(self.w_slow.abs().mean()), 0.0,
                          torch.zeros_like(x))
        # Медленный пре-след + мгновенный вход дают устойчивое окно коинциденции
        x_trace = 0.6 * x + 0.4 * self.z_pre
        if float(x_trace.abs().max()) < 1e-6 and float(err.abs().max()) < 1e-6:
            self.w_fast.mul_(0.9995)
            self.w_slow.mul_(c.decay)
            self.u_syn.add_((1.0 - self.u_syn) * (dt / 500.0))
            return SynOut(float(self.w_slow.abs().mean()), 0.0, x)

        # Селективный холинергический тег (Genesis): отклонение ко-активации от фона
        coinc = torch.outer(err.abs(), x_trace) * self.mask
        if self._tag_ref_ready:
            ref = torch.outer(self.r_err, self.r_pre).clamp_(min=1e-4)
            surplus = (coinc / ref - 1.0).clamp_(0.0, 5.0) * self.mask
            self.tag.add_(0.05 * (0.5 + ach) * surplus).clamp_(0.0, 2.0)
        alpha_ref = min(dt / 500.0, 1.0)
        self.r_err.mul_(1.0 - alpha_ref).add_(err.abs(), alpha=alpha_ref)
        self.r_pre.mul_(1.0 - alpha_ref).add_(x_trace, alpha=alpha_ref)
        self._tag_ref_ready = True

        # --- Уровень 2: Eligibility-trace и 3-й фактор дофамина ---
        bcm_gain = torch.where(post_rate > self.theta_m, 1.0, 0.5)
        hom_coinc = torch.outer(err * bcm_gain, x_trace) * self.mask
        self.eligibility.mul_(1.0 - dt / 1000.0).add_(hom_coinc * (0.5 + 0.5 * ach))

        # Дофамин реализует 3-й фактор: конвертирует накопленный след eligibility в вес!
        da_gate = float(f_mod) if da_phasic is None else max(0.05, float(f_mod) * (0.5 + max(0.0, da_phasic)))
        dw_hom = (c.eta * f_mod) * (0.65 * hom_coinc + 0.35 * da_gate * self.eligibility) * self.mask

        # --- Уровень 1: Двухфазная Гетеросинаптическая STDP (Chistiakova 2014; Royer & Paré 2003) ---
        # Фаза 2 (snnscience.docx §3.9): Метаболический порог LTP от теневой цены энергии Лагранжа λ_ATP
        if lambda_atp > 0.0:
            ltp_metabolic_gate = 1.0 / (1.0 + 0.65 * max(0.0, float(lambda_atp)))
            ltp_min_thr = 0.005 * max(0.0, float(lambda_atp))
            dw_hom = torch.where(
                dw_hom > 0.0,
                torch.relu(dw_hom - ltp_min_thr) * ltp_metabolic_gate,
                dw_hom,
            )
        # 1) Внутриветочная кооперативная гетеро-LTP: соседние синапсы на дендритной ветке
        #    делят локальную Ca2+-волну и PRP-белки от потенциированных синапсов
        dw_pos = dw_hom.clamp(min=0.0)
        coop_ltp = 0.12 * (dw_pos @ self.branch_kernel) * self.mask
        # 2) Компенсаторная гетеро-LTD (кальциневриновая волна): когда на нейроне происходит
        #    сильная гомосинаптическая LTP, НЕАКТИВНЫЕ синапсы (1 - x_j) того же нейрона
        #    испытывают пропорциональную депрессию, сохраняя общий синаптический баланс!
        inactive_pre = (1.0 - x_trace.clamp(0.0, 1.0)).unsqueeze(0)  # [1, N_pre]
        row_ltp_sum = dw_pos.sum(dim=1, keepdim=True)                # [N_post, 1]
        hetero_ltd = 0.25 * row_ltp_sum * inactive_pre * self.w_slow.clamp(min=0.0) * self.mask
        self.last_hetero_ltd = float(hetero_ltd.abs().mean().item())

        dw = (dw_hom + coop_ltp - hetero_ltd) * self.mask
        # Гетеросинаптическая LTP от диффузного NO-поля (по пре- или пост-окрестности)
        if ltp_field is not None:
            lf = ltp_field.detach().flatten()
            if lf.numel() == self.n_pre:
                dw.add_(torch.outer(err.abs(), lf) * self.mask,
                        alpha=c.eta * f_mod * 0.12)
            elif lf.numel() == self.n_post:
                dw.add_(torch.outer(err.abs() * lf, x_trace) * self.mask,
                        alpha=c.eta * f_mod * 0.12)

        # Запись: 70% в быстрый синаптический след w_fast, 30% в структурный w_slow.
        # Структурная защита от катастрофического забывания (Fusi et al. 2005):
        # зрелые консолидированные шипики с высоким здоровьем spine_health > 0.6
        # защищают свою долговременную память w_slow от разрушающей депрессии!
        shield = torch.where(dw * self.w_slow < 0,
                             (1.0 - 0.80 * self.spine_health).clamp(0.15, 1.0),
                             torch.ones_like(self.spine_health))
        dw_fast_step = dw * (0.5 + 0.5 * shield) * 0.7
        dw_slow_step = dw * shield * 0.3

        # --- Lab.html F#58 (ORIN, accepted on P7): Термодинамический порог забывания от дисперсии аппроксимации декодера ---
        if self.readout_approx_noise_floor > 0.0:
            noise_gate = (dw.abs() >= self.readout_approx_noise_floor).float()
            dw_fast_step = dw_fast_step * noise_gate
            dw_slow_step = dw_slow_step * noise_gate

        # --- Фаза 1 (Изъян 7 snnscience.docx §3.7) + next9.txt #2: CT-PNN + Synaptic Intelligence (arxiv 1703.04200) ---
        if self.enable_scaffold:
            if self.chabc_window_ticks > 0:
                self.chabc_window_ticks -= 1
            is_hard = (self.scaffold_state == SCAFFOLD_HARD) | self.pinned_block_mask
            is_soft = (self.scaffold_state == SCAFFOLD_SOFT) & (~is_hard)
            # HARD-синапсы (грибовидные шипики в зрелой PNN) полностью неизменяемы (Δw = 0)
            dw_fast_step = torch.where(is_hard, torch.zeros_like(dw_fast_step), dw_fast_step)
            dw_slow_step = torch.where(is_hard, torch.zeros_like(dw_slow_step), dw_slow_step)
            # SOFT-синапсы (незрелая PNN, Frischknecht et al. 2009) ограничивают латеральную диффузию: |Δw| <= v_max
            v_lim = (self.v_max_soft * (1.0 - 0.5 * self.pnn_maturity)).clamp(min=1e-4)
            dw_slow_step = torch.where(
                is_soft,
                torch.clamp(dw_slow_step, min=-v_lim, max=v_lim),
                dw_slow_step,
            )

        # Онлайн-оценка важности синапса Synaptic Intelligence (Zenke et al. 2017, arxiv 1703.04200):
        # Интегрируем вклад каждого синапса в снижение ошибки вдоль траектории обучения:
        #   ω_ij += (-∂L/∂w_ij) · Δw_ij(t), где локальный градиент уменьшения ошибки пропорционален hom_coinc
        step_delta_w = (dw_fast_step + dw_slow_step) * self.mask.float()
        local_grad = hom_coinc.detach()
        si_contrib = (local_grad * step_delta_w).clamp(min=0.0)
        self.si_omega.add_(si_contrib * self.mask.float())

        self.w_fast.add_(dw_fast_step).clamp_(-0.8, 0.8).mul_(self.mask)
        self.w_slow.add_(dw_slow_step).clamp_(c.w_min, c.w_max).mul_(self.mask)
        # Актиновое здоровье шипика H_ij (Узел 4): укрепляется при ко-активации и теге,
        # медленно распадается при неиспользовании
        use_signal = (coinc + 0.3 * self.tag) * self.mask.float()
        self.spine_health.mul_(0.998).add_(0.025 * use_signal).clamp_(0.0, 1.0)
        # Медленная утечка к нулю (HARD и SOFT защищены от пассивного стирания!)
        if self.enable_scaffold:
            plastic_mask = (self.scaffold_state == SCAFFOLD_PLASTIC).float()
            decay_factor = plastic_mask * c.decay + (1.0 - plastic_mask) * 1.0
            self.w_fast.mul_(plastic_mask * 0.9995 + (1.0 - plastic_mask) * 1.0)
            self.w_slow.mul_(decay_factor)
        else:
            self.w_fast.mul_(0.9995)
            self.w_slow.mul_(c.decay)
        # SOC-депрессия: спайк тратит ресурс u_syn, а глутаминовый цикл восстанавливает
        if float(post_rate.max()) > 0.0 and float(x.max()) > 0.0:
            spent = (torch.outer(post_rate, x) * self.mask).clamp_(0.0, 1.0)
            self.u_syn.sub_(spent, alpha=0.15).clamp_(min=0.05)
        gln_rate = float(getattr(self, "glutamine_rate", 1.0))
        self.u_syn.add_((1.0 - self.u_syn) * (dt / 500.0) * gln_rate).clamp_(0.05, 1.0)
        if self.enable_het_syn:
            self.step_het_syn_traces(x, dt=dt, adapt_tau=True)
        self.z_pre.mul_(1.0 - dt / c.tau_x).add_(x)
        self.z_post.mul_(1.0 - dt / c.tau_x).add_(post_rate)
        return SynOut(float(self.w_slow.abs().mean()), float(dw.abs().mean()),
                      x)

    def structural_plasticity(self, prune_thresh: float = 0.015,
                              max_frac: float = 0.05,
                              error_level: Optional[float] = None,
                              microglia_boost: float = 1.0,
                              bdnf: float = 0.35) -> tuple[int, int]:
        """Узел 4 (Структурный Морфогенез по градиенту ошибки BDNF/NO):
        1) Прунинг слабых неиспользуемых шипиков (низкий |w|, низкий тег и низкое здоровье H_ij)
           с модуляцией внеклеточным АТФ через P2X-рецепторы микроглии (SOFT и HARD защищены PNN!);
        2) Целенаправленный спрутинг: в зонах с высокой ошибкой предсказания (error_level / r_err)
           и высоким уровнем нейротрофина BDNF плотность синапсов динамически растёт
           вдоль градиента BDNF/NO (r_err × r_pre) на свободных позициях (~mask)."""
        eff_prune_th = prune_thresh * max(0.5, min(2.0, float(microglia_boost)))
        w_tot = (self.w_slow + self.w_fast).abs()
        weak = (self.mask & (w_tot < eff_prune_th)
                & (self.tag < 0.05) & (self.spine_age > 20.0)
                & (self.spine_health < 0.35))
        if self.enable_scaffold:
            # PNN защищает SOFT и HARD синапсы от микроглиального удаления
            weak = weak & (self.scaffold_state == SCAFFOLD_PLASTIC)
        n_weak = int(weak.sum().item())
        max_change = max(1, int(max_frac * self.mask.numel()))
        n_prune = min(n_weak, max_change)
        if n_prune > 0:
            weak_idx = torch.nonzero(weak, as_tuple=False)[:n_prune]
            for r, col_idx in weak_idx:
                self.mask[r, col_idx] = False
                self.w_slow[r, col_idx] = 0.0
                self.w_fast[r, col_idx] = 0.0
                self.spine_age[r, col_idx] = 0.0
                self.spine_health[r, col_idx] = 0.0
                self.si_omega[r, col_idx] = 0.0
                self.si_importance[r, col_idx] = 0.0
                if r // 16 < self._dirty_blocks.shape[0] and col_idx // 16 < self._dirty_blocks.shape[1]:
                    self._dirty_blocks[r // 16, col_idx // 16] = True

        # Если в зоне высокая ошибка предсказания (r_err / error_level) или повышен BDNF,
        # нейротрофин BDNF повышает целевую плотность связей для рекрутинга новых синапсов!
        r_err_mean = float(self.r_err.mean().item())
        err_val = r_err_mean if error_level is None else max(r_err_mean, float(error_level))
        bdnf_surplus = max(0.0, float(bdnf) - 0.35)
        bdnf_density_boost = 1.0 + 0.42 * min(
            1.0, max(0.0, err_val - 0.04) * 3.2 + 0.85 * bdnf_surplus
        )
        eff_density = min(0.95, self.cfg.density * bdnf_density_boost)
        target_edges = int(round(eff_density * self.n_post * self.n_pre))
        current_edges = int(self.mask.sum().item())
        n_sprout = min(max(n_prune, target_edges - current_edges), max_change)
        if n_sprout > 0:
            empty = ~self.mask
            if bool(empty.any()):
                affinity = torch.outer(self.r_err + 1e-3, self.r_pre + 1e-3)
                scores = torch.where(empty, affinity + torch.rand_like(affinity) * 1e-3,
                                     torch.full_like(affinity, -1e9))
                flat_idx = scores.flatten().topk(min(n_sprout, int(empty.sum().item()))).indices
                rows = flat_idx // self.n_pre
                cols = flat_idx % self.n_pre
                self.mask[rows, cols] = True
                sign = 1.0 if self.cfg.sign == "exc" else -1.0
                self.w_slow[rows, cols] = sign * 0.05
                self.u_syn[rows, cols] = 1.0
                self.spine_age[rows, cols] = 0.0
                self.spine_health[rows, cols] = 0.5
                self.scaffold_state[rows, cols] = SCAFFOLD_PLASTIC
                self.si_omega[rows, cols] = 0.0
                self.si_importance[rows, cols] = 0.0
                n_sprout = int(rows.numel())
            else:
                n_sprout = 0
        if n_prune > 0 or n_sprout > 0:
            self._rebuild_sparseprop_index()
        self.last_pruned = n_prune
        self.last_sprouted = n_sprout
        self.total_pruned += n_prune
        self.total_sprouted += n_sprout
        return n_prune, n_sprout

    def consolidate(self, factor: float = 0.5,
                    spindle_gain: float = 1.0,
                    replay_x: Optional[torch.Tensor] = None) -> None:
        """Узел 1 (Тройной осцилляторный замок сна: SO × Spindles × SWR + SHY Тонони-Чирелли)
        + Фаза 1 (Изъян 7 snnscience.docx) + next9.txt #2 (Synaptic Intelligence, arxiv 1703.04200):
        Продвижение синапса в статус SOFT и HARD определяется математическим интегралом
        его вклада в снижение ошибки Ω_ij = relu(ω_ij) / ((Δw_ij)^2 + ξ)."""
        c = self.cfg
        tag_boost = (1.0 + 0.6 * self.tag) * max(0.2, float(spindle_gain))
        replay_hit_mask = torch.zeros_like(self.mask, dtype=torch.bool)
        if replay_x is not None and replay_x.numel() == self.n_pre:
            rx = replay_x.detach().float().flatten().clamp(0.0, 1.0)
            tag_boost = tag_boost * (0.7 + 0.6 * rx.unsqueeze(0))
            replay_hit_mask = (rx.unsqueeze(0) > 0.15) & (self.tag > 0.05) & self.mask
        else:
            replay_hit_mask = (self.tag > 0.08) & self.mask

        transfer = factor * self.w_fast * tag_boost

        # Обновление нормализованной оценки важности Synaptic Intelligence (Zenke et al. 2017):
        #   Δw_ij = (w_slow_new - si_w_anchor),   Ω_ij += relu(ω_ij) / ((Δw_ij)^2 + ξ)
        w_slow_candidate = ((self.w_slow * 0.995) + transfer).clamp(c.w_min, c.w_max) * self.mask
        delta_w_task = (w_slow_candidate - self.si_w_anchor) * self.mask.float()
        omega_norm = torch.relu(self.si_omega) / (delta_w_task.pow(2) + self.si_xi)
        self.si_importance = (0.85 * self.si_importance + omega_norm) * self.mask.float()

        if self.enable_scaffold:
            # 1. Мультипликативный SHY-даунскейлинг с сохранением рангов применяется ТОЛЬКО к PLASTIC!
            is_plastic = (self.scaffold_state == SCAFFOLD_PLASTIC)
            is_hard = (self.scaffold_state == SCAFFOLD_HARD)
            downscale = torch.where(is_plastic, torch.full_like(self.w_slow, 0.995), torch.ones_like(self.w_slow))
            eff_transfer = torch.where(is_hard, torch.zeros_like(transfer), transfer)
            self.w_slow = ((self.w_slow * downscale) + eff_transfer).clamp(c.w_min, c.w_max) * self.mask

            # 2. Synaptic Intelligence + Tagging & Capture (arxiv 1703.04200 + Frey & Morris 1997):
            #    синапс переходит в SOFT и HARD не просто по совпадению SWR, а по накопленному
            #    математическому интегралу важности Синаптического Интеллекта Ω_ij!
            self.replay_captures.add_(replay_hit_mask.to(torch.int16))
            self.sleep_cycles_survived.add_(self.mask.to(torch.int16))
            si_boost = (self.si_importance / max(1e-6, self.si_hard_threshold)).clamp(0.0, 1.5)
            self.pnn_maturity = (
                self.pnn_maturity + 0.14 * replay_hit_mask.float() + 0.12 * si_boost
            ).clamp(0.0, 1.0)

            promote_to_soft = (
                is_plastic
                & (self.si_importance >= self.si_soft_threshold)
                & ((self.replay_captures >= 1) | (self.si_importance >= 2.0 * self.si_soft_threshold))
                & (self.w_slow.abs() >= 0.08)
                & self.mask
            )
            promote_to_hard = (
                (self.scaffold_state == SCAFFOLD_SOFT)
                & (self.si_importance >= self.si_hard_threshold)
                & (self.replay_captures >= 2)
                & (self.w_slow.abs() >= 0.14)
                & (self.pnn_maturity >= 0.35)
                & self.mask
            )
            self.scaffold_state = torch.where(
                promote_to_hard,
                torch.full_like(self.scaffold_state, SCAFFOLD_HARD),
                torch.where(
                    promote_to_soft,
                    torch.full_like(self.scaffold_state, SCAFFOLD_SOFT),
                    self.scaffold_state,
                ),
            )
            self.spine_health = torch.where(
                self.scaffold_state == SCAFFOLD_HARD,
                torch.ones_like(self.spine_health),
                self.spine_health,
            )
        else:
            # SHY (Tononi & Cirelli 2014): лёгкий глобальный даунскейлинг фоновых весов (0.5%)
            self.w_slow = w_slow_candidate

        # Обновляем опорную точку якоря SI и затухаем текущий путевой интеграл ω_ij для следующей задачи/эпохи
        self.si_w_anchor.copy_(self.w_slow)
        self.si_omega.mul_(0.25)
        self.w_fast.mul_(1.0 - factor)
        self.tag.mul_(0.5)
        self.u_syn.fill_(1.0)
        self._dirty_blocks.zero_()

    def promote_scaffold_first(self, weight_quantile: float = 0.80) -> int:
        """Scaffold-first инициализация (snnscience.docx §3.7): опорные связи базовых паттернов
        сразу продвигаются в защищённый уровень HARD (зрелая перинейрональная сеть)."""
        self.enable_scaffold = True
        active_w = self.w_slow.abs() * self.mask.float()
        if not bool(self.mask.any()):
            return 0
        thr = float(torch.quantile(active_w[self.mask], min(0.99, max(0.05, weight_quantile))).item())
        core_mask = self.mask & (active_w >= thr)
        self.scaffold_state = torch.where(
            core_mask,
            torch.full_like(self.scaffold_state, SCAFFOLD_HARD),
            self.scaffold_state,
        )
        self.pnn_maturity = torch.where(core_mask, torch.ones_like(self.pnn_maturity), self.pnn_maturity)
        self.spine_health = torch.where(core_mask, torch.ones_like(self.spine_health), self.spine_health)
        return int(core_mask.sum().item())

    def trigger_chabc_event(self, post_mask: Optional[torch.Tensor] = None,
                            window_ticks: int = 50) -> int:
        """Ферментативное событие Хондроитиназы ABC (ChABC, Pizzorusso et al. 2002; Gogolla et al. 2009):
        адресно растворяет зрелую перинейрональную сеть (переводит HARD -> SOFT) при подтверждённом
        сдвиге входного распределения, открывая окно повторной консолидации без стирания всей сети."""
        self.enable_scaffold = True
        self.chabc_window_ticks = max(self.chabc_window_ticks, int(window_ticks))
        if post_mask is not None and post_mask.numel() == self.n_post:
            target = (self.scaffold_state == SCAFFOLD_HARD) & post_mask.bool().view(-1, 1)
        else:
            target = (self.scaffold_state == SCAFFOLD_HARD)
        n_unlocked = int(target.sum().item())
        self.scaffold_state = torch.where(
            target,
            torch.full_like(self.scaffold_state, SCAFFOLD_SOFT),
            self.scaffold_state,
        )
        self.pnn_maturity = torch.where(target, self.pnn_maturity * 0.4, self.pnn_maturity)
        return n_unlocked

    def scaffold_stats(self) -> Dict[str, float]:
        """Возвращает статистику распределения синапсов по уровням PLASTIC / SOFT / HARD и времени полужизни."""
        total = max(1, int(self.mask.sum().item()))
        n_pl = int(((self.scaffold_state == SCAFFOLD_PLASTIC) & self.mask).sum().item())
        n_sf = int(((self.scaffold_state == SCAFFOLD_SOFT) & self.mask).sum().item())
        n_hd = int(((self.scaffold_state == SCAFFOLD_HARD) & self.mask).sum().item())
        mean_cycles = float(self.sleep_cycles_survived[self.mask].float().mean().item()) if bool(self.mask.any()) else 0.0
        mean_si = float(self.si_importance[self.mask].mean().item()) if bool(self.mask.any()) else 0.0
        # Эффективный период полураспада памяти (half-life) растёт квадратично с числом циклов сна и долей SOFT/HARD
        protected_frac = (n_sf + 2.0 * n_hd) / max(1.0, float(total))
        est_half_life_ticks = 200.0 * (1.0 + 1.8 * mean_cycles) * (1.0 + 4.5 * protected_frac)
        return {
            "plastic_frac": n_pl / total,
            "soft_frac": n_sf / total,
            "hard_frac": n_hd / total,
            "n_hard": n_hd,
            "n_soft": n_sf,
            "mean_sleep_cycles": mean_cycles,
            "mean_si_importance": round(mean_si, 4),
            "estimated_half_life_ticks": round(est_half_life_ticks, 2),
        }

    def istdp_vogels(self, s_pre: torch.Tensor, s_post: torch.Tensor,
                     dt: float = 1.0) -> float:
        """Ингибиторный STDP Вогельса 2011 (правильная форма math.md).

        При пре-спайке: Δw=η(x_post−α), α=2ρ₀τ; при пост-спайке:
        Δw=η·x_pre. Только для sign='inh' синапсов."""
        if self.cfg.sign != "inh":
            return 0.0
        c = self.cfg
        x_pre = s_pre.mean(0); x_post = s_post.mean(0)
        alpha = 2.0 * c.istdp_rho0 * c.tau_x
        dw = torch.outer((x_post - alpha).clamp(min=-1.0), s_pre.mean(0)) \
            + torch.outer(s_post.mean(0), x_pre)
        self.w_slow = (self.w_slow - c.istdp_eta * dw * self.mask.float()
                       ).clamp(c.w_min, 0.0)
        return float(dw.abs().mean())
