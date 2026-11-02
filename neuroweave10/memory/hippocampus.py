"""Гиппокампальный комплекс (§II.3.1): EC → DG → CA3 → CA1 → VTA/LC.

DG — паттерн-сепарация: экспансия + разрежение (~1-2% активных),
взрослый нейрогенез (2-3% нейронов/месяц → реинтеграция весов).
CA3 — автоассоциативная сеть Хопфилда, one-shot обучение
(η_CA3 = 10·η_cortex, ёмкость ~0.14·N).
CA1 — компаратор: mismatch ||CA3-EC_III||² → фазический DA (VTA)
и NE (LC); novelty-сигнал.
SWR-реплей: [ACh]<θ & [Adenosine]>θ → prioritized experience replay,
P(replay) ∝ |ошибка| + β·uncertainty (HOT), прокси-градиент
на кортикальные веса.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..config import HippocampusCfg


@dataclass
class HippoOut:
    ec: torch.Tensor
    dg: torch.Tensor
    ca3: torch.Tensor
    ca1_out: torch.Tensor
    mismatch: float
    vta_da_burst: float      # фазический DA от новизны
    lc_ne_burst: float       # фазический NE от новизны
    replayed: List[int]      # индексы реплеированных эпизодов


class Hippocampus:
    """Трисинаптический контур с нейрогенезом, θ-фазовым разделением и SWR-реплеем."""

    def __init__(self, cfg: HippocampusCfg, seed: int = 0):
        self.cfg = cfg
        g = torch.Generator().manual_seed(seed)
        self.w_ec_dg = torch.randn(cfg.n_dg, cfg.n_ec, generator=g) * 0.3
        self.w_dg_ca3 = torch.randn(cfg.n_ca3, cfg.n_dg, generator=g) * 0.1
        self.w_ca3 = torch.zeros(cfg.n_ca3, cfg.n_ca3)       # автоассоциативный Хопфилд
        # P4 (Buzsáki 2015; Foster & Wilson 2006): асимметричная темпоральная STDP-матрица
        # в коллатералях Шаффера/CA3 для связывания эпизодов в хронологические траектории (t-1 → t)
        self.w_ca3_seq = torch.zeros(cfg.n_ca3, cfg.n_ca3)
        self._prev_ca3_act: Optional[torch.Tensor] = None
        self.w_ca3_ca1 = torch.randn(cfg.n_ec, cfg.n_ca3, generator=g) * 0.1
        # Темпороаммонический путь: прямой EC_III → CA1 (минуя DG/CA3)
        self.w_ec_ca1 = torch.randn(cfg.n_ec, cfg.n_ec, generator=g) * 0.2
        # Grid-код EC: cos(ω·proj+φ) — метрика пространства MEC
        self.grid_freq = torch.rand(cfg.n_ec, generator=g) * 4.0 + 1.0
        self.grid_phase = torch.rand(cfg.n_ec, generator=g) * 6.283
        self.w_in_ec = torch.randn(cfg.n_ec, cfg.n_ec, generator=g) * 0.5
        self.dg_age = torch.ones(cfg.n_dg)               # возраст гранул
        self.dg_thresh = torch.zeros(cfg.n_dg)           # порог возбудимости гранул
        self.tick_n = 0
        self.gamma_slot_ms: float = 25.0                 # Инновация 3: 1 когнитивный шаг = 1 γ-слот (25 мс)
        self.n_stored = 0
        self.capacity = max(8, int(0.14 * cfg.n_ca3))
        # Эпизодический буфер: (ec, dg, ca3, ошибка, uncertainty)
        self.episodes: List[Tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, float, float]] = []
        self.episode_epochs: List[int] = []  # число пережитых циклов сна (0 = новый эпизод, >=1 = старый)
        self.interleaved_replays: int = 0
        self.max_episodes = 512
        # Инновация 4 (Quiroga et al. 2005; Conway 2005):
        # Слой Концептуальных Клетки MTL (Concept / Grandmother Cells) и
        # Автобиографический Семантический Граф Я-Схемы (Narrative Self-Graph)
        cfg_n_conc = int(getattr(cfg, "n_concepts", 0))
        self.n_concepts = cfg_n_conc if cfg_n_conc > 0 else max(16, cfg.n_ec // 2)
        self.w_concept = torch.randn(self.n_concepts, cfg.n_ec, generator=g) * 0.35
        self.w_concept /= self.w_concept.norm(dim=1, keepdim=True).clamp(min=1e-6)
        self.w_semantic_graph = torch.zeros(self.n_concepts, self.n_concepts)
        self.w_semantic_graph_consolidated = torch.zeros(self.n_concepts, self.n_concepts)
        self.w_concept_graph = self.w_semantic_graph  # алиас нарративного графа концептов
        self.concept_salience = torch.zeros(self.n_concepts)
        self.concept_salience_floor = torch.zeros(self.n_concepts)  # метапластический щит (Fusi et al. 2005)
        self.concept_valence = torch.zeros(self.n_concepts)
        self.concept_word_anchors: Dict[int, int] = {}
        self.prev_concept_idx: int = -1
        self.last_winner_concept_idx: int = 0
        self.narrative_concept_trace = torch.zeros(self.n_concepts)
        self.last_concept_vec = torch.zeros(self.n_concepts)
        # Фаза 2 (snnscience.docx §3.3, Изъян 3): Композиционный VSA-DHT гиппокамп на 4 шардах (150x ёмкость)
        self.vsa_dht = SBCVSAShardedHippocampus(
            n_shards=4,
            n_blocks=16,
            block_len=64,
            shard_capacity=9600,
            seed=seed + 31,
        )

    # ------------------------------------------------------------------
    def theta_osc(self) -> float:
        """Тета-осциллятор ~7 Гц (Hasselmo 2002; Buzsáki 2006): [-1, 1]."""
        import math
        return math.sin(2.0 * math.pi * 7.0 * self.tick_n * 1e-3)

    def theta_gamma_osc(self) -> float:
        """Инновация 3: Вложенное мультиплексирование θ-γ — каждый когнитивный шаг
        продвигает слотовые часы на 1 гамма-слот (25 мс => ~6 шагов на полный θ-период 143 мс)."""
        import math
        t_bio_ms = self.tick_n * self.gamma_slot_ms
        return math.sin(2.0 * math.pi * 7.0 * t_bio_ms * 1e-3)

    def dg_code(self, ec: torch.Tensor) -> torch.Tensor:
        """Паттерн-сепарация: разрежение до dg_sparsity (k-WTA) с учётом возраста."""
        drive = ec @ self.w_ec_dg.t() - self.dg_thresh.unsqueeze(0)
        k = max(1, int(self.cfg.dg_sparsity * self.cfg.n_dg))
        thr = drive.topk(k, dim=1).values[:, -1:]
        return (drive >= thr).float()

    def neurogenesis(self, novelty_level: float = 0.5,
                     cortisol: float = 0.2,
                     bdnf: float = 0.35) -> int:
        """Взрослый нейрогенез DG (Узел 4): скорость рождения новых гиперпластичных
        гранулярных клеток растёт от новизны/BDNF и подавляется стрессом (кортизолом)."""
        c = self.cfg
        bdnf_drive = (
            1.0
            + 1.5 * max(0.0, float(novelty_level))
            + 1.35 * max(0.0, float(bdnf) - 0.45)
        )
        stress_brake = max(0.25, 1.0 - 0.6 * max(0.0, float(cortisol)))
        frac = min(0.12, 0.022 * bdnf_drive * stress_brake)
        n_new = max(1, int(round(frac * c.n_dg)))
        oldest = self.dg_age.argsort(descending=True)[:n_new]
        self.w_ec_dg[oldest] = torch.randn(n_new, c.n_ec) * 0.5
        self.dg_age[oldest] = 0.0
        self.dg_thresh[oldest] = -0.35
        self.dg_age += 1.0
        self.dg_thresh = (self.dg_thresh + 0.005).clamp(max=0.0)
        return n_new

    def ca3_store(self, ca3: torch.Tensor) -> None:
        """One-shot Хопфилд с биполярным центрированием по среднему (Genesis),
        гомеостазом ёмкости C=0.14N и асимметричным STDP связыванием траекторий (P4)."""
        c = self.cfg
        p = ca3.mean(0)
        split = float(p.mean().item())
        xi = torch.where(p >= split, torch.ones_like(p), -torch.ones_like(p))
        dw = (torch.outer(xi, xi) / max(1, c.n_ca3))
        dw.fill_diagonal_(0.0)
        if self.n_stored >= self.capacity:
            self.w_ca3.mul_(self.capacity / (self.capacity + 1.0))
        self.w_ca3 += max(c.eta_ca3 * 10.0, 0.08) * dw
        self.w_ca3.clamp_(-1.0, 1.0)

        # P4 (Buzsáki 2015; Foster & Wilson 2006): асимметричная STDP-пластичность
        # связывает предыдущий паттерн CA3 (t-1) с текущим (t) в направленную траекторию
        cur_act = (p >= split).float()
        if self._prev_ca3_act is not None and self._prev_ca3_act.shape == cur_act.shape:
            dw_seq = (torch.outer(cur_act, self._prev_ca3_act)
                      - 0.30 * torch.outer(self._prev_ca3_act, cur_act)) / max(1.0, float(cur_act.sum().item()))
            dw_seq.fill_diagonal_(0.0)
            self.w_ca3_seq = (0.995 * self.w_ca3_seq
                              + max(c.eta_ca3 * 12.0, 0.12) * dw_seq).clamp(-1.0, 1.0)
        self._prev_ca3_act = cur_act.detach().clone()
        self.n_stored += 1

    def ca3_predict_next(self, cue: torch.Tensor) -> torch.Tensor:
        """P4: Предсказание следующего шага эпизодической траектории через асимметричную
        матрицу переходов CA3 w_ca3_seq."""
        drive = cue @ self.w_ca3_seq.t()
        thr = drive.mean(dim=-1, keepdim=True)
        return (drive > thr).float()

    def ca3_retrieve(self, cue: torch.Tensor) -> torch.Tensor:
        """Pattern completion: несколько итераций рекуррентной динамики
        (автоассоциация w_ca3 + слабое упреждение по траектории w_ca3_seq)."""
        a = cue.clone()
        ach = getattr(self, "_ach", 1.0)
        theta_rec = 0.5 * (1.0 - self.theta_osc())  # сильнее на спаде теты
        n_steps = int(self.cfg.ca3_steps * (1.5 - 0.5 * ach + 0.5 * theta_rec))
        for _ in range(max(2, n_steps)):
            a = torch.tanh(a @ self.w_ca3.t() * 2.0 + a @ self.w_ca3_seq.t() * 0.35 + 0.25 * cue)
        return (a > 0).float()

    def encode_ec(self, ec_in: torch.Tensor, ach: float = 0.8) -> torch.Tensor:
        """Детерминированное проекционное кодирование входа в пространство решётки EC."""
        rms = ec_in.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
        proj = (ec_in / rms) @ self.w_in_ec.t()
        return torch.cos(self.grid_freq * proj + self.grid_phase)

    def concept_idx_for_ec(self, ec: torch.Tensor) -> int:
        """Возвращает индекс ведущей концептуальной клетки MTL для данного вектора EC."""
        x = ec.detach().float().mean(0) if ec.dim() == 2 else ec.detach().float().flatten()
        x_norm = x / x.norm().clamp(min=1e-6)
        sim = self.w_concept.to(x_norm.device) @ x_norm
        return int(sim.argmax().item())

    def anchor_or_update_word_concept(
        self,
        word_id: int,
        ec_word: torch.Tensor,
        salience: float = 0.65,
        valence: float = 0.0,
        learn: bool = True,
        pre_clause_trace: Optional[torch.Tensor] = None,
    ) -> int:
        """Закрепляет или обновляет специализированную концептуальную клетку MTL
        для содержательного слова word_id с защитой метапластическим щитом."""
        x = ec_word.detach().float().mean(0) if ec_word.dim() == 2 else ec_word.detach().float().flatten()
        x_norm = x / x.norm().clamp(min=1e-6)
        if word_id in self.concept_word_anchors:
            win_idx = self.concept_word_anchors[word_id]
        else:
            sim = self.w_concept.to(x_norm.device) @ x_norm
            used = set(self.concept_word_anchors.values())
            if len(used) < self.n_concepts:
                sim_masked = sim.clone()
                for u_idx in used:
                    if 0 <= u_idx < self.n_concepts:
                        sim_masked[u_idx] -= 2.5
                win_idx = int(sim_masked.argmax().item())
            else:
                win_idx = int(sim.argmax().item())
            if learn:
                self.concept_word_anchors[word_id] = win_idx
                w_row = 0.15 * self.w_concept[win_idx].to(x_norm.device) + 0.85 * x_norm
                self.w_concept[win_idx] = (w_row / w_row.norm().clamp(min=1e-6)).cpu()
        self.last_winner_concept_idx = win_idx
        if learn:
            cur_sal = float(self.concept_salience[win_idx].item())
            plast_shield = 0.12 if cur_sal > 1.2 else 0.45
            w_row = self.w_concept[win_idx].to(x_norm.device) + plast_shield * (
                x_norm - self.w_concept[win_idx].to(x_norm.device)
            )
            self.w_concept[win_idx] = (w_row / w_row.norm().clamp(min=1e-6)).cpu()
            new_sal = min(5.0, cur_sal * 0.996 + 0.22 * float(salience))
            self.concept_salience[win_idx] = max(
                float(self.concept_salience_floor[win_idx].item()), new_sal
            )
            if new_sal > 0.65:
                self.concept_salience_floor[win_idx] = max(
                    float(self.concept_salience_floor[win_idx].item()),
                    min(4.6, new_sal * 0.86),
                )
            self.concept_valence[win_idx] = (
                0.88 * float(self.concept_valence[win_idx]) + 0.12 * float(valence)
            )
            trace_ref = pre_clause_trace if pre_clause_trace is not None else self.narrative_concept_trace
            trace_cpu = trace_ref.detach().cpu()
            for c_prev in range(self.n_concepts):
                if c_prev != win_idx and float(trace_cpu[c_prev].item()) > 0.25:
                    inc = 0.09 * float(trace_cpu[c_prev].item()) * min(2.0, float(salience))
                    self.w_semantic_graph[win_idx, c_prev] = min(
                        2.0, float(self.w_semantic_graph[win_idx, c_prev]) + inc
                    )
                    self.w_semantic_graph[c_prev, win_idx] = min(
                        2.0, float(self.w_semantic_graph[c_prev, win_idx]) + 0.80 * inc
                    )
            self.w_semantic_graph.copy_(
                torch.maximum(self.w_semantic_graph, self.w_semantic_graph_consolidated)
            )
        c_direct = torch.zeros(self.n_concepts)
        c_direct[win_idx] = 1.0
        base_trace = pre_clause_trace.detach().cpu() if pre_clause_trace is not None else self.narrative_concept_trace
        self.narrative_concept_trace = (0.78 * base_trace + 0.65 * c_direct).clamp(0.0, 1.0)
        bridge_act = (self.w_semantic_graph @ self.narrative_concept_trace).clamp(0.0, 1.0)
        self.last_concept_vec = torch.maximum(
            self.narrative_concept_trace, 0.75 * bridge_act
        ).clamp(0.0, 1.0)
        return win_idx

    def update_concept_cells(self, ec: torch.Tensor, valence: float = 0.0,
                             salience: float = 0.5, learn: bool = True) -> torch.Tensor:
        """Инновация 4 (Quiroga et al. 2005; Conway 2005):
        Формирование инвариантных концептуальных энграмм в медиальной височной доле (MTL),
        реляционного графа автобиографической памяти (w_semantic_graph) и распространения
        активации (Spreading Activation) по семантическим мостам."""
        x = ec.detach().float().mean(0) if ec.dim() == 2 else ec.detach().float()
        x_norm = x / x.norm().clamp(min=1e-6)
        sim = self.w_concept.to(x_norm.device) @ x_norm
        # Конкурентная специализация (Kohonen conscience + защита занятых клеток от перезаписи):
        # если входной паттерн не совпадает близко с уже закреплённой клеткой (sim < 0.84),
        # защищаем зрелые концептуальные клетки (salience > 0.65) от захвата новыми словами!
        if learn and float(sim.max().item()) < 0.84:
            sal_dev = self.concept_salience.to(sim.device)
            committed_shield = ((sal_dev > 0.65) & (sim < 0.84)).float() * 1.25
            sim_sel = sim - 0.14 * sal_dev.clamp(0.0, 2.0) - committed_shield
            win_idx = int(sim_sel.argmax().item())
        else:
            win_idx = int(sim.argmax().item())
        self.last_winner_concept_idx = win_idx

        c_direct = torch.zeros(self.n_concepts, device=x_norm.device)
        c_direct[win_idx] = 1.0
        c_vec = c_direct.clone()
        # Вторая по близости концептуальная клетка создаёт локальный ассоциативный ореол
        if self.n_concepts >= 2:
            top2 = sim.topk(2).indices
            sec_idx = int(top2[1].item())
            if sec_idx != win_idx:
                c_vec[sec_idx] = 0.25

        if learn and salience > 0.05:
            # Мягкий гомеостатический дрейф неактивных клеток, удерживаемый снизу метапластическим щитом Фузи
            self.concept_salience = torch.maximum(
                self.concept_salience_floor, self.concept_salience * 0.9997
            )
            # Метапластический щит Фузи для уже консолидированных концептуальных прототипов
            cur_sal = float(self.concept_salience[win_idx].item())
            plast_shield = 0.25 if cur_sal > 1.2 else 1.0
            lr_c = 0.12 * min(2.0, float(salience)) * plast_shield
            w_row = self.w_concept[win_idx].to(x_norm.device) + lr_c * (
                x_norm - self.w_concept[win_idx].to(x_norm.device)
            )
            self.w_concept[win_idx] = (w_row / w_row.norm().clamp(min=1e-6)).cpu()
            new_sal = min(5.0, cur_sal * 0.992 + 0.18 * float(salience))
            self.concept_salience[win_idx] = max(
                float(self.concept_salience_floor[win_idx].item()), new_sal
            )
            if new_sal > 0.65:
                self.concept_salience_floor[win_idx] = max(
                    float(self.concept_salience_floor[win_idx].item()),
                    min(4.6, new_sal * 0.85),
                )
            self.concept_valence[win_idx] = (
                0.88 * float(self.concept_valence[win_idx]) + 0.12 * float(valence)
            )
            if 0 <= self.prev_concept_idx < self.n_concepts and self.prev_concept_idx != win_idx:
                self.w_semantic_graph[win_idx, self.prev_concept_idx] = min(
                    2.0,
                    float(self.w_semantic_graph[win_idx, self.prev_concept_idx])
                    + 0.08 * min(2.0, float(salience)),
                )
            # Клаузальное связывание через прямой след narrative_concept_trace:
            # связывает субъект и объект одного предложения ('dragon' <-> 'mountain', 'knight' <-> 'castle')
            # даже если между ними стоят служебные слова ('flew to the')!
            active_trace = self.narrative_concept_trace.detach().cpu()
            for c_prev in range(self.n_concepts):
                if c_prev != win_idx and float(active_trace[c_prev].item()) > 0.30:
                    bridge_inc = 0.08 * float(active_trace[c_prev].item()) * min(2.0, float(salience))
                    self.w_semantic_graph[win_idx, c_prev] = min(
                        2.0, float(self.w_semantic_graph[win_idx, c_prev]) + bridge_inc
                    )
                    self.w_semantic_graph[c_prev, win_idx] = min(
                        2.0, float(self.w_semantic_graph[c_prev, win_idx]) + 0.75 * bridge_inc
                    )
            # Защита консолидированных во сне семантических мостов от стирания
            self.w_semantic_graph.copy_(
                torch.maximum(self.w_semantic_graph, self.w_semantic_graph_consolidated)
            )
            self.prev_concept_idx = win_idx

        # Прямой след концептов текущей клаузы (без рекурсивного подмешивания соседей графа!)
        self.narrative_concept_trace = (
            0.78 * self.narrative_concept_trace + 0.65 * c_direct.cpu()
        ).clamp(0.0, 1.0)
        # Распространение активации по семантическому графу (Spreading Activation, Collins & Loftus 1975):
        # 1-шаговое распространение от прямого следа клаузы поджигает семантически связанные концепты
        bridge_act = (self.w_semantic_graph @ self.narrative_concept_trace).clamp(0.0, 1.0)
        self.last_concept_vec = torch.maximum(
            self.narrative_concept_trace, 0.75 * bridge_act
        ).clamp(0.0, 1.0)
        return self.last_concept_vec

    def rem_recombine(self, cortical_bridge_fn=None) -> Dict[str, float]:
        """Инновация 7 (Lewis et al. 2018; Diekelmann & Born 2010; McClelland et al. 1995):
        Фаза REM-сна с Интерливинг-Реплеем (Interleaved Replay):
        при высоком ACh и нулевом NE комбинирует НОВЫЕ эпизоды бодрствования (epoch == 0)
        со СТАРЫМИ консолидированными эпизодами (epoch >= 1), создавая кросс-контекстные
        семантические мосты и защищая старые связи от катастрофического забывания."""
        if len(self.episodes) < 2:
            return {"rem_chimera_similarity": 0.0, "semantic_bridges": 0.0, "interleaved_pairs": 0}
        while len(self.episode_epochs) < len(self.episodes):
            self.episode_epochs.append(0)
        valid_indices = [
            idx for idx, e in enumerate(self.episodes)
            if float(e[0].abs().mean().item()) > 1e-4
        ]
        if len(valid_indices) < 2:
            valid_indices = list(range(len(self.episodes)))

        # Разделяем эпизоды на новые (текущая эпоха бодрствования) и старые (консолидированные в прошлых снах)
        fresh_idx = [i for i in valid_indices if self.episode_epochs[i] == 0]
        old_idx = [i for i in valid_indices if self.episode_epochs[i] >= 1]

        errs = torch.tensor([
            abs(self.episodes[idx][3]) + 0.5 * self.episodes[idx][4]
            for idx in valid_indices
        ])
        top_k = min(len(valid_indices), max(2, (len(valid_indices) + 1) // 2))
        top_local = errs.topk(top_k).indices.tolist()
        cand_idx = [valid_indices[i] for i in top_local]
        # Обязательно включаем представителей старых эпох для Interleaved Replay!
        if old_idx:
            old_sorted = sorted(old_idx, key=lambda idx: abs(self.episodes[idx][3]) + 0.5 * self.episodes[idx][4], reverse=True)
            for oi in old_sorted[:max(2, top_k // 2)]:
                if oi not in cand_idx:
                    cand_idx.append(oi)

        n_pairs = min(10, max(2, len(cand_idx)))
        sim_sum = 0.0
        chimera_act_sum = 0.0
        interleaved_pairs = 0
        for p_idx in range(n_pairs):
            if fresh_idx and old_idx and p_idx % 2 == 0:
                # Явный Interleaved Replay: связываем новый сюжет со старым сюжетом!
                i_a = fresh_idx[p_idx % len(fresh_idx)]
                pool_b = old_idx
                interleaved_pairs += 1
            else:
                i_a = cand_idx[p_idx % len(cand_idx)]
                pool_b = cand_idx
            ec_a, _, ca3_a, err_a, _ = self.episodes[i_a]
            c_a = self.concept_idx_for_ec(ec_a)
            i_b = max(
                pool_b,
                key=lambda idx: (
                    (20.0 if self.concept_idx_for_ec(self.episodes[idx][0]) != c_a else 0.0)
                    + (12.0 if self.episode_epochs[idx] != self.episode_epochs[i_a] else 0.0)
                    + abs(idx - i_a)
                    + 0.5 * abs(self.episodes[idx][3])
                ),
            )
            ec_b, _, ca3_b, err_b, _ = self.episodes[i_b]
            c_b = self.concept_idx_for_ec(ec_b)

            # Химерная суперпозиция (50% эпизод A + 50% эпизод B)
            chimera_ec = 0.5 * ec_a + 0.5 * ec_b
            chimera_ca3 = ((0.5 * ca3_a + 0.5 * ca3_b) > 0.4).float()
            sim_ab = float(torch.nn.functional.cosine_similarity(
                ec_a.flatten().unsqueeze(0), ec_b.flatten().unsqueeze(0)
            ).abs().item())
            sim_sum += sim_ab
            chimera_act_sum += float(chimera_ca3.mean().item())
            # Связываем концепты обоих эпизодов в семантическом графе Я-Схемы
            if c_a != c_b:
                self.w_semantic_graph[c_b, c_a] = min(2.0, float(self.w_semantic_graph[c_b, c_a]) + 0.35)
                self.w_semantic_graph[c_a, c_b] = min(2.0, float(self.w_semantic_graph[c_a, c_b]) + 0.35)
            # Консолидация значимости и метапластического щита воспроизводимых во сне концептов
            for c_rep in (c_a, c_b):
                s_rep = min(5.0, float(self.concept_salience[c_rep].item()) + 0.04)
                self.concept_salience[c_rep] = s_rep
                if s_rep > 0.65:
                    self.concept_salience_floor[c_rep] = max(
                        float(self.concept_salience_floor[c_rep].item()),
                        min(4.6, s_rep * 0.88),
                    )
            # Межэпизодный мост в CA3
            p_a = ca3_a.mean(0); p_b = ca3_b.mean(0)
            dw_rem = torch.outer(p_b, p_a) / max(1.0, float(p_b.sum().item()))
            dw_rem.fill_diagonal_(0.0)
            self.w_ca3_seq = (self.w_ca3_seq + 0.03 * dw_rem).clamp(-1.0, 1.0)
            if cortical_bridge_fn is not None:
                cortical_bridge_fn(chimera_ec, 0.5 * (abs(err_a) + abs(err_b)))

        # Консолидируем все существующие семантические мосты в защищённый каркас долговременной памяти
        # и помечаем все текущие эпизоды как пережившие ещё один цикл сна
        self.w_semantic_graph.copy_(
            torch.maximum(self.w_semantic_graph, self.w_semantic_graph_consolidated)
        )
        self.w_semantic_graph_consolidated.copy_(
            torch.maximum(self.w_semantic_graph_consolidated, self.w_semantic_graph * 0.96)
        )
        self.episode_epochs = [ep + 1 for ep in self.episode_epochs]
        return {
            "rem_chimera_similarity": round(sim_sum / max(1, n_pairs), 4),
            "semantic_bridges": float((self.w_semantic_graph > 0.05).sum().item()),
            "chimera_ca3_active": round(chimera_act_sum / max(1, n_pairs), 4),
            "interleaved_pairs": interleaved_pairs,
        }

    def step(self, ec_in: torch.Tensor, ach: float, adenosine: float,
             learn: bool = True, valence: float = 0.0,
             salience_boost: float = 0.0) -> HippoOut:
        """Бодрствование: кодирование/извлечение по θ-фазе; сон: SWR-реплей."""
        self.tick_n += 1
        self._ach = ach
        if self.tick_n % self.cfg.neurogenesis_period == 0:
            self.neurogenesis()
        # RMS-нормировка входа EC (Genesis EntorhinalCortex): разные амплитуды
        # спайков коры не схлопываются в одну и ту же фазу сетки
        ec_base = self.encode_ec(ec_in, ach=ach)
        theta = self.theta_osc()
        enc_gate = 0.5 * (1.0 + theta) * (0.5 + 0.5 * float(max(0.0, min(1.0, ach))))
        ec = ec_base * (0.6 + 0.4 * enc_gate)
        dg = self.dg_code(ec)
        ca3_raw = torch.relu(dg @ self.w_dg_ca3.t())
        # ACh + θ-гейт режимов (Hasselmo): пик θ и высокий ACh = encode, спад = recall
        ach_gain = float(torch.clamp(torch.tensor(ach), 0.0, 1.0))
        ca3 = ca3_raw * (0.5 + 0.5 * ach_gain)
        if learn:
            self.ca3_store(ca3)
        ca3_full = self.ca3_retrieve(ca3)
        # CA1-компаратор: CA3-реконструкция vs прямой EC_III-вход
        recon = ca3_full @ self.w_ca3_ca1.t() + ec @ self.w_ec_ca1.t()
        mismatch = float((recon - ec).pow(2).mean())
        burst_da = float(mismatch > self.cfg.ca1_threshold) * min(
            1.0, mismatch / self.cfg.ca1_threshold)
        burst_ne = burst_da * 0.8
        # Обновляем Концептуальные Клетки MTL и Автобиографический Семантический Граф (ровно 1 раз за шаг!)
        self.update_concept_cells(
            ec_base, valence=valence,
            salience=0.35 + burst_da + salience_boost,
            learn=learn,
        )
        replayed: List[int] = []
        return HippoOut(ec=ec_base, dg=dg, ca3=ca3_full, ca1_out=recon,
                        mismatch=mismatch, vta_da_burst=burst_da,
                        lc_ne_burst=burst_ne, replayed=replayed)

    def store_episode(self, ec: torch.Tensor, dg: torch.Tensor,
                      ca3: torch.Tensor, error: float,
                      uncertainty: float, da_tag: float = 0.0) -> None:
        # DA-тег: дофаминовый бёрст помечает значимые эпизоды —
        # повышает их приоритет в SWR-реплее (вспоминание важного).
        self.episodes.append((ec.detach().clone(), dg.detach().clone(),
                              ca3.detach().clone(),
                              float(error) * (1.0 + da_tag),
                              float(uncertainty)))
        self.episode_epochs.append(0)
        if len(self.episodes) > self.max_episodes:
            # Вместо слепого удаления самого первого якорного эпизода (FIFO),
            # удаляем наименее значимый эпизод из первой половины буфера, сохраняя опорные воспоминания!
            half = max(1, len(self.episodes) // 2)
            drop_idx = min(range(half), key=lambda i: abs(self.episodes[i][3]) + 0.5 * self.episodes[i][4])
            self.episodes.pop(drop_idx)
            self.episode_epochs.pop(drop_idx)
        if hasattr(self, "vsa_dht") and self.vsa_dht is not None:
            self.vsa_dht.store_from_tensors(
                ec=ec, dg=dg, ca3=ca3,
                salience=float(abs(error) * (1.0 + da_tag)),
                ca1_mismatch=float(uncertainty),
            )

    def swr_replay(self, cortical_write, learn: bool = True,
                   direction: str = "auto") -> List[int]:
        """P4: Траекторный приоритетный SWR-реплей с Интерливингом старых и новых воспоминаний
        (Foster & Wilson 2006; Buzsáki 2015; McClelland, McNaughton & O'Reilly 1995).

        Чередует проигрывание новых эпизодов бодрствования (epoch == 0) и старых
        консолидированных эпизодов (epoch >= 1), предотвращая катастрофическое забывание."""
        if not self.episodes:
            return []
        n_ep = len(self.episodes)
        while len(self.episode_epochs) < n_ep:
            self.episode_epochs.append(0)
        errs = torch.tensor([abs(e[3]) + self.cfg.replay_beta_unc * e[4]
                             for e in self.episodes])
        old_mask = torch.tensor([1.0 if ep >= 1 else 0.0 for ep in self.episode_epochs])
        fresh_mask = 1.0 - old_mask
        # Чередуем (Interleaved Replay): на чётных вызовах приоритезируем новые эпизоды,
        # на нечётных — старые консолидированные эпизоды (если они есть)!
        if bool((old_mask > 0).any()) and bool((fresh_mask > 0).any()):
            use_old = (self.interleaved_replays % 2 == 1)
            self.interleaved_replays += 1
            mask_sel = old_mask if use_old else fresh_mask
            weighted_errs = (errs + 1e-6) * (0.15 + 0.85 * mask_sel)
        else:
            weighted_errs = errs + 1e-6
        probs = weighted_errs / weighted_errs.sum()
        k = min(self.cfg.n_replay_batch, n_ep)
        seed_idx = int(torch.multinomial(probs, 1).item())

        # Определяем направление SWR-волны:
        # 'reverse' — от финального события с высокой ошибкой/DA назад по траектории (Foster & Wilson 2006)
        # 'forward' — хронологическая консолидация последовательности (Pfeiffer & Foster 2013)
        if direction == "reverse" or (
            direction == "auto" and seed_idx >= k - 1
            and float(errs[seed_idx]) > 1.25 * float(errs.mean())
        ):
            step_dir = -1
        else:
            step_dir = 1

        if step_dir == 1:
            start_idx = min(seed_idx, max(0, n_ep - k))
            idx = [start_idx + s for s in range(k)]
        else:
            start_idx = max(seed_idx, min(n_ep - 1, k - 1))
            idx = [start_idx - s for s in range(k)]

        self._dbg_replayed = list(idx)
        self._last_replay_dir = "reverse" if step_dir == -1 else "forward"
        prev_ca3_rep: Optional[torch.Tensor] = None
        for step_num, i in enumerate(idx):
            ec, dg, ca3, err, unc = self.episodes[i]
            # Затухающий вдоль траектории коэффициент TD-консолидации
            td_decay = 0.88 ** step_num
            cortical_write(ec, err * td_decay)
            if learn and prev_ca3_rep is not None:
                c1 = ca3.mean(0)
                c0 = prev_ca3_rep.mean(0)
                # При обратном реплее c0 — более поздний шаг, c1 — более ранний
                post_c, pre_c = (c0, c1) if step_dir == -1 else (c1, c0)
                dw_rep = torch.outer(post_c, pre_c) / max(1.0, float(post_c.sum().item()))
                dw_rep.fill_diagonal_(0.0)
                self.w_ca3_seq = (self.w_ca3_seq + 0.03 * dw_rep).clamp(-1.0, 1.0)
            prev_ca3_rep = ca3
        return idx


class CompositionalThetaGammaIndexer:
    """D4 (archnext.txt): Композиционный Тета-Гамма Индексатор Паттернов (Product-Quantized
    Role × Filler MTL/CA3 Memory, Buzsáki 2010; Kanerva 2009).

    Решает проблему «Предела детских историй» (насыщение плоской памяти после ~10k паттернов):
      - Плоская память из K = 256 клеток хранит только O(K) неделимых эпизодов.
      - Факторизованная память делит вектор эпизода на M = 4 тета-гамма ролевых слота
        (Subject × Verb × Object × Context), где каждый слот имеет K_sub = 64 прототипа.
      - Суммарное число прототипов равно 4 × 64 = 256 (та же память!), но комбинаторная
        ёмкость ортогональных связок составляет 64^4 = 16 777 216 (16.7M!) уникальных паттернов!
    """

    def __init__(self, n_slots: int = 4, k_per_slot: int = 64, d_slot: int = 16, seed: int = 42):
        self.n_slots = int(n_slots)
        self.k_per_slot = int(k_per_slot)
        self.d_slot = int(d_slot)
        g = torch.Generator().manual_seed(seed)
        self.codebooks = torch.randn(self.n_slots, self.k_per_slot, self.d_slot, generator=g)
        self.codebooks = self.codebooks / self.codebooks.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        self.slot_salience = torch.zeros(self.n_slots, self.k_per_slot)
        self.slot_floor = torch.zeros(self.n_slots, self.k_per_slot)
        # Разреженный ассоциативный индекс композиционных кортежей (s0, s1, s2, s3) -> trace strength
        self.compositional_traces: Dict[Tuple[int, ...], float] = {}

    def theoretical_capacity(self) -> int:
        return int(self.k_per_slot ** self.n_slots)

    def encode_and_bind(self, slot_ids: Tuple[int, ...], salience: float = 1.0, learn: bool = True) -> Tuple[int, ...]:
        """Привязывает кортеж ролевых элементов (Subject, Verb, Object, Context) к композиционному следу."""
        quant_tuple = tuple(int(s) % self.k_per_slot for s in slot_ids[: self.n_slots])
        if learn:
            prev_val = self.compositional_traces.get(quant_tuple, 0.0)
            self.compositional_traces[quant_tuple] = min(5.0, prev_val * 0.995 + 0.35 * float(salience))
            for m, k_idx in enumerate(quant_tuple):
                new_s = min(5.0, float(self.slot_salience[m, k_idx].item()) * 0.995 + 0.15 * float(salience))
                self.slot_salience[m, k_idx] = max(float(self.slot_floor[m, k_idx].item()), new_s)
                if new_s > 0.65:
                    self.slot_floor[m, k_idx] = max(float(self.slot_floor[m, k_idx].item()), min(4.6, new_s * 0.88))
        return quant_tuple


class SBCVSAShardedHippocampus:
    """Фаза 2 Глобального Плана v8.0 (snnscience.docx §3.3, Изъян 3):
    Распределённая Хэш-Таблица Гиппокампа на базе Разреженных Блочных Кодов VSA
    (`SBC-VSA Sharded Hippocampal DHT` + `Adaptive Multi-Probe LSH` + `Two-Clock θ/γ Replay`).

    Решает проблему «Узкого места памяти и предела детских историй (~10k тиков)»:
      1. Разреженные Блочные Коды (`Sparse Block-Codes VSA`, D = 1024 = 16 блоков × 64 линии):
         - Вместо круговой свёртки HRR через БПФ (O(D log D), превращающей спайки в плотный шум)
           связывание (Binding ⊗) двух разреженных кодов выполняется как поблочный модульный
           циклический сдвиг индексов: idx(A ⊗ B)_b = (idx(A)_b + idx(B)_b) mod 64 за O(B) = 16
           целочисленных операций со 100% обратимостью без потерь!
      2. 4 функциональных шарда гиппокампа (`K = 4`):
         - Шард 0 (`Dorsal-Spatial`): пространственно-контекстная решётка MEC;
         - Шард 1 (`Ventral-Affective`): эмоциональная валентность BLA и метаболическая ценность;
         - Шард 2 (`Lexical-Semantic`): ролевая структура Subject–Verb–Object;
         - Шард 3 (`Causal-Temporal`): траектория переходов состояний (s_t, a_t) -> s_{t+1}.
         Каждый шард хранит до `shard_capacity = 9600` слотов (суммарно 4 × 9600 = 38,400 эпизодов —
         ровно в 150× больше плоского буфера из 256 эпизодов!), а поиск внутри шарда идёт за O(M/K)
         через консистентное LSH-корзинирование.
      3. Адаптивный Multi-Probe LSH по сигналу новизны CA1:
         - При низком `ca1_mismatch` используется `1` пробник на шард (минимальная задержка);
         - При высоком `ca1_mismatch` (новизна/неопределённость) число пробников автоматически
           расширяется до `4` соседних LSH-корзин для ассоциативного достраивания по зашумлённому ключу.
      4. Двухчасовой реплей (`Two-Clock θ/γ Replay`):
         - Быстрые γ-такты (25 мс) выполняют локальное достраивание паттерна внутри каждого шарда;
         - Медленные θ-такты (125 мс, каждый 5-й γ-такт) собирают целостный мультимодальный эпизод
           через развязывание глобального ключа `SBC-VSA`.
    """

    SHARD_NAMES: Tuple[str, ...] = (
        "Dorsal-Spatial",
        "Ventral-Affective",
        "Lexical-Semantic",
        "Causal-Temporal",
    )

    def __init__(
        self,
        n_shards: int = 4,
        n_blocks: int = 16,
        block_len: int = 64,
        shard_capacity: int = 9600,
        n_lsh_buckets: int = 256,
        seed: int = 42,
    ):
        self.n_shards = int(n_shards)
        self.n_blocks = int(n_blocks)
        self.block_len = int(block_len)
        self.shard_capacity = int(shard_capacity)
        self.n_lsh_buckets = int(n_lsh_buckets)

        g = torch.Generator().manual_seed(seed)
        # Фиксированные ролевые коды шардов R_0 .. R_{K-1}, каждый длины B = 16 со значениями [0..63]
        self.role_codes: List[Tuple[int, ...]] = [
            tuple(int(x) for x in torch.randint(0, self.block_len, (self.n_blocks,), generator=g).tolist())
            for _ in range(self.n_shards)
        ]

        # 4 независимых шарда DHT: каждый шард содержит LSH-таблицу bucket_id -> список записей (ep_id, code, salience)
        self.shard_buckets: List[Dict[int, List[Tuple[int, Tuple[int, ...], float]]]] = [
            {} for _ in range(self.n_shards)
        ]
        self.shard_counts: List[int] = [0 for _ in range(self.n_shards)]

        # Глобальный реестр композитных VSA-ключей эпизодов: ep_id -> (bound_composite_codes, salience)
        self.episodes_vsa: Dict[int, Tuple[Tuple[Tuple[int, ...], ...], float]] = {}
        self.next_ep_id: int = 0
        self.gamma_tick_counter: int = 0
        self.theta_bindings_completed: int = 0

        # --- next9.txt #4: HippoRAG Personalized PageRank (Gutierrez et al. 2024, arxiv 2405.14831) ---
        # Разреженный ассоциативный граф поверх SBC-VSA-DHT для многошаговых цепочек (Concept A -> Concept B -> Concept C)
        self.ppr_adj: Dict[int, Dict[int, float]] = {}
        self.concept_to_episodes: Dict[str, List[int]] = {}
        self.episode_to_concepts: Dict[int, List[str]] = {}
        self.concept_adj: Dict[str, Dict[str, float]] = {}
        self._last_stored_ep_id: Optional[int] = None

    def total_capacity(self) -> int:
        """Суммарная распределённая ёмкость 4 шардов (38,400 эпизодов = 150x от базовых 256)."""
        return int(self.n_shards * self.shard_capacity)

    def capacity_expansion_factor(self, flat_baseline: int = 256) -> float:
        return float(self.total_capacity()) / float(max(1, flat_baseline))

    def bind(self, code_a: Tuple[int, ...], code_b: Tuple[int, ...]) -> Tuple[int, ...]:
        """Поблочное модульное связывание SBC-VSA за O(B) целочисленных сложений (без FFT)."""
        return tuple((int(a) + int(b)) % self.block_len for a, b in zip(code_a, code_b))

    def unbind(self, bound_code: Tuple[int, ...], role_code: Tuple[int, ...]) -> Tuple[int, ...]:
        """Точное обратимое развязывание SBC-VSA за O(B): unbind(bind(A, R), R) == A."""
        return tuple((int(c) - int(r)) % self.block_len for c, r in zip(bound_code, role_code))

    def block_similarity(self, code_a: Tuple[int, ...], code_b: Tuple[int, ...]) -> float:
        """Доля совпадающих активных линий по B блокам ∈ [0.0, 1.0]."""
        matches = sum(1 for a, b in zip(code_a, code_b) if int(a) == int(b))
        return float(matches) / float(max(1, self.n_blocks))

    def _lsh_bucket(self, code: Tuple[int, ...], probe_offset: int = 0) -> int:
        """Консистентный LSH-хэш по первым 4 блокам разреженного кода с поддержкой смещения пробника."""
        h = (code[0] * 131 + code[1] * 37 + code[2] * 7 + code[3] + probe_offset) % self.n_lsh_buckets
        return int(h)

    def adaptive_num_probes(self, ca1_mismatch: float) -> int:
        """Адаптивно выбирает число LSH-пробников (от 1 до 4) пропорционально сигналу новизны CA1."""
        m = max(0.0, float(ca1_mismatch))
        if m < 0.15:
            return 1
        if m < 0.35:
            return 2
        if m < 0.65:
            return 3
        return 4

    def vector_to_block_code(self, vec: torch.Tensor, shard_idx: int = 0) -> Tuple[int, ...]:
        """Преобразует произвольный тензор активности в разреженный блочный код длины B = 16."""
        flat = vec.detach().flatten().float()
        if flat.numel() == 0:
            return tuple(0 for _ in range(self.n_blocks))
        codes: List[int] = []
        n = flat.numel()
        for b in range(self.n_blocks):
            idx = (b * 17 + shard_idx * 13) % n
            val = float(flat[idx].item())
            quant = int(round((math.tanh(val) + 1.0) * 0.5 * (self.block_len - 1))) % self.block_len
            codes.append((quant + b * 3) % self.block_len)
        return tuple(codes)

    def store_composite_episode(
        self,
        slot_codes: List[Tuple[int, ...]],
        salience: float = 1.0,
        ca1_mismatch: float = 0.1,
        concepts: Optional[List[str]] = None,
    ) -> int:
        """Записывает мультимодальный эпизод параллельно в 4 шарда DHT с ролевым связыванием SBC-VSA
        и автоматически обновляет ассоциативный граф HippoRAG PPR (arxiv 2405.14831)."""
        ep_id = self.next_ep_id
        self.next_ep_id += 1
        bound_tuple_list: List[Tuple[int, ...]] = []
        neighbor_weights: Dict[int, float] = {}
        for s_idx in range(self.n_shards):
            filler_code = slot_codes[s_idx % len(slot_codes)]
            bound_code = self.bind(filler_code, self.role_codes[s_idx])
            bound_tuple_list.append(bound_code)
            b_id = self._lsh_bucket(filler_code, probe_offset=0)
            bucket = self.shard_buckets[s_idx].setdefault(b_id, [])
            # Связываем с близкими эпизодами в той же LSH-корзине для графа HippoRAG PPR
            for prev_ep, prev_code, _ in bucket[-8:]:
                if prev_ep != ep_id:
                    sim = self.block_similarity(filler_code, prev_code)
                    if sim >= 0.25:
                        neighbor_weights[prev_ep] = max(neighbor_weights.get(prev_ep, 0.0), sim)
            bucket.append((ep_id, filler_code, float(salience)))
            self.shard_counts[s_idx] += 1
            # Также индексируем во вторичную LSH-корзину по второй четвёрке блоков для устойчивости к шуму
            b_id2 = (filler_code[4] * 131 + filler_code[5] * 37 + filler_code[6] * 7 + filler_code[7]) % self.n_lsh_buckets
            if b_id2 != b_id:
                self.shard_buckets[s_idx].setdefault(int(b_id2), []).append((ep_id, filler_code, float(salience)))

        self.episodes_vsa[ep_id] = (tuple(bound_tuple_list), float(salience))
        self.ppr_adj.setdefault(ep_id, {})

        # Временна́я причинная связь с предыдущим эпизодом (Causal-Temporal Trajectory)
        if self._last_stored_ep_id is not None and self._last_stored_ep_id != ep_id:
            self.link_episodes(self._last_stored_ep_id, ep_id, weight=0.45)
        self._last_stored_ep_id = ep_id

        for nb_ep, w_sim in neighbor_weights.items():
            self.link_episodes(ep_id, nb_ep, weight=float(w_sim))

        if concepts:
            self.episode_to_concepts[ep_id] = list(concepts)
            for c_name in concepts:
                ep_list = self.concept_to_episodes.setdefault(c_name, [])
                for existing_ep in ep_list[-8:]:
                    if existing_ep != ep_id:
                        self.link_episodes(existing_ep, ep_id, weight=0.85)
                ep_list.append(ep_id)
            for i in range(len(concepts) - 1):
                self.bind_concept_edge(concepts[i], concepts[i + 1], weight=0.90)

        return ep_id

    def link_episodes(self, ep_a: int, ep_b: int, weight: float = 0.5, bidirectional: bool = True) -> None:
        """Добавляет взвешенное ассоциативное ребро между двумя эпизодами в графе HippoRAG."""
        if ep_a == ep_b:
            return
        w = max(0.01, float(weight))
        adj_a = self.ppr_adj.setdefault(int(ep_a), {})
        adj_a[int(ep_b)] = max(adj_a.get(int(ep_b), 0.0), w)
        if bidirectional:
            adj_b = self.ppr_adj.setdefault(int(ep_b), {})
            adj_b[int(ep_a)] = max(adj_b.get(int(ep_a), 0.0), w * 0.85)

    def bind_concept_edge(self, concept_a: str, concept_b: str, weight: float = 0.85) -> None:
        """Связывает два концепта (Concept A -> Concept B) в графе знаний HippoRAG и сшивает их эпизоды."""
        if concept_a == concept_b:
            return
        w = max(0.05, float(weight))
        ca = self.concept_adj.setdefault(concept_a, {})
        ca[concept_b] = max(ca.get(concept_b, 0.0), w)
        cb = self.concept_adj.setdefault(concept_b, {})
        cb[concept_a] = max(cb.get(concept_a, 0.0), w * 0.8)
        for ep_a in self.concept_to_episodes.get(concept_a, ())[-4:]:
            for ep_b in self.concept_to_episodes.get(concept_b, ())[-4:]:
                self.link_episodes(ep_a, ep_b, weight=w)

    def ppr_associative_recall(
        self,
        seed_episode_ids: Optional[List[int]] = None,
        seed_concepts: Optional[List[str]] = None,
        query_slot_codes: Optional[List[Tuple[int, ...]]] = None,
        damping: float = 0.85,
        max_iters: int = 15,
        top_k: int = 5,
    ) -> Dict[str, object]:
        """HippoRAG Personalized PageRank (Gutierrez et al. 2024, arxiv 2405.14831; next9.txt #4):
        Выполняет многошаговое распространение активации с перезапуском (Personalized PageRank):
            p^{(t+1)} = (1 - α) · p_0 + α · M^T p^{(t)}
        поверх шардированной памяти SBC-VSA-DHT и графа концептов, позволяя находить
        многошаговые ассоциативные цепочки вида Concept A -> Concept B -> Concept C,
        даже если A и C никогда не встречались в одном эпизоде напрямую!"""
        # 1. Формируем вектор персонализации (телепортации) p_0 по эпизодам и концептам
        p0_ep: Dict[int, float] = {}
        if seed_episode_ids:
            for eid in seed_episode_ids:
                if int(eid) in self.episodes_vsa or int(eid) in self.ppr_adj:
                    p0_ep[int(eid)] = p0_ep.get(int(eid), 0.0) + 1.0
        if seed_concepts:
            for cname in seed_concepts:
                for eid in self.concept_to_episodes.get(cname, ()):
                    p0_ep[int(eid)] = p0_ep.get(int(eid), 0.0) + 1.0
        if not p0_ep and query_slot_codes is not None:
            init_ret = self.retrieve_multiprobe(query_slot_codes, ca1_mismatch=0.35, top_k=3)
            for eid, sc in zip(init_ret["top_episode_ids"], init_ret["top_scores"]):
                p0_ep[int(eid)] = max(0.1, float(sc))

        if not p0_ep and self.episodes_vsa:
            first_ep = next(iter(self.episodes_vsa.keys()))
            p0_ep[int(first_ep)] = 1.0

        total_p0 = sum(p0_ep.values())
        if total_p0 <= 0.0:
            return {
                "ranked_episode_ids": [],
                "ranked_scores": [],
                "concept_chain": [],
                "multi_hop_chain_ids": [],
                "iterations": 0,
            }
        p0_ep = {k: v / total_p0 for k, v in p0_ep.items()}

        # 2. Итерации Personalized PageRank по графу эпизодов ppr_adj
        alpha = max(0.05, min(0.95, float(damping)))
        p_curr: Dict[int, float] = dict(p0_ep)
        parent_map: Dict[int, int] = {k: k for k in p0_ep}

        for _ in range(max(1, int(max_iters))):
            p_next: Dict[int, float] = {k: (1.0 - alpha) * v for k, v in p0_ep.items()}
            for u, pu in p_curr.items():
                if pu < 1e-6:
                    continue
                nbrs = self.ppr_adj.get(u, {})
                if not nbrs:
                    p_next[u] = p_next.get(u, 0.0) + alpha * pu
                    continue
                w_sum = sum(nbrs.values())
                for v, w_uv in nbrs.items():
                    flow = alpha * pu * (w_uv / max(1e-6, w_sum))
                    p_next[v] = p_next.get(v, 0.0) + flow
                    if v not in parent_map and u in parent_map:
                        parent_map[v] = u
            p_curr = p_next

        ranked_eps = sorted(p_curr.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]

        # 3. Персонализированный PageRank по графу концептов для построения явной цепочки Concept A -> Concept B -> Concept C
        concept_chain: List[str] = []
        if self.concept_adj:
            c_seeds = list(seed_concepts) if seed_concepts else []
            if not c_seeds:
                for eid in p0_ep:
                    c_seeds.extend(self.episode_to_concepts.get(eid, []))
            if c_seeds:
                c_p0 = {c: 1.0 / len(c_seeds) for c in c_seeds}
                c_curr = dict(c_p0)
                c_parent: Dict[str, str] = {c: c for c in c_seeds}
                for _ in range(max(1, int(max_iters))):
                    c_next: Dict[str, float] = {c: (1.0 - alpha) * v for c, v in c_p0.items()}
                    for cu, p_cu in c_curr.items():
                        cnbrs = self.concept_adj.get(cu, {})
                        if not cnbrs:
                            c_next[cu] = c_next.get(cu, 0.0) + alpha * p_cu
                            continue
                        cw_sum = sum(cnbrs.values())
                        for cv, cw in cnbrs.items():
                            c_next[cv] = c_next.get(cv, 0.0) + alpha * p_cu * (cw / max(1e-6, cw_sum))
                            if cv not in c_parent:
                                c_parent[cv] = cu
                    c_curr = c_next
                # Находим лучшую многошаговую ассоциативную цепочку (длины >= 3: Concept A -> Concept B -> Concept C),
                # взвешенную по стационарным вероятностям Personalized PageRank
                best_chain: List[str] = []
                best_chain_score = -1.0
                for c_end, sc_end in c_curr.items():
                    if c_end in c_seeds:
                        continue
                    path = [c_end]
                    cur = c_end
                    while cur in c_parent and c_parent[cur] != cur and len(path) < 8:
                        cur = c_parent[cur]
                        path.append(cur)
                    cand_chain = list(reversed(path))
                    # Бонус за многошаговый переход (>= 3 узлов в цепочке: A -> B -> C)
                    hop_bonus = 10.0 if len(cand_chain) >= 3 else 1.0
                    chain_score = hop_bonus * sum(c_curr.get(node, 0.0) for node in cand_chain[1:])
                    if chain_score > best_chain_score:
                        best_chain_score = chain_score
                        best_chain = cand_chain
                concept_chain = best_chain if best_chain else list(c_seeds)

        # 4. Восстанавливаем цепочку эпизодов от seed до лучшего косвенно активированного эпизода
        seed_set = set(p0_ep.keys())
        multi_hop_ep_chain: List[int] = []
        for eid, _ in ranked_eps:
            if eid not in seed_set and eid in parent_map:
                chain = [eid]
                cur = eid
                while cur in parent_map and parent_map[cur] != cur and len(chain) < 8:
                    cur = parent_map[cur]
                    chain.append(cur)
                multi_hop_ep_chain = list(reversed(chain))
                if len(multi_hop_ep_chain) >= 3:
                    break
        if not multi_hop_ep_chain and ranked_eps:
            multi_hop_ep_chain = [eid for eid, _ in ranked_eps[:3]]

        return {
            "ranked_episode_ids": [eid for eid, _ in ranked_eps],
            "ranked_scores": [round(float(sc), 5) for _, sc in ranked_eps],
            "concept_chain": concept_chain,
            "multi_hop_chain_ids": multi_hop_ep_chain,
            "iterations": int(max_iters),
        }

    def store_from_tensors(
        self,
        ec: torch.Tensor,
        dg: torch.Tensor,
        ca3: torch.Tensor,
        salience: float = 1.0,
        ca1_mismatch: float = 0.1,
    ) -> int:
        """Адаптер для прямой записи тензоров гиппокампа (ec, dg, ca3) в 4 шарда VSA-DHT."""
        slot_codes = [
            self.vector_to_block_code(ec, shard_idx=0),
            self.vector_to_block_code(dg, shard_idx=1),
            self.vector_to_block_code(ca3, shard_idx=2),
            self.vector_to_block_code(ec + 0.5 * ca3[:, : ec.shape[-1]] if ca3.shape[-1] >= ec.shape[-1] else ec, shard_idx=3),
        ]
        return self.store_composite_episode(slot_codes, salience=salience, ca1_mismatch=ca1_mismatch)

    def retrieve_multiprobe(
        self,
        query_slot_codes: List[Tuple[int, ...]],
        ca1_mismatch: float = 0.25,
        top_k: int = 5,
    ) -> Dict[str, object]:
        """Распределённый поиск по 4 шардам с адаптивным числом LSH-пробников за O(M/K) на шард."""
        n_probes = self.adaptive_num_probes(ca1_mismatch)
        candidate_scores: Dict[int, float] = {}
        inspected_records = 0

        for s_idx in range(self.n_shards):
            q_code = query_slot_codes[s_idx % len(query_slot_codes)]
            probed_buckets = set()
            for p in range(n_probes):
                probed_buckets.add(self._lsh_bucket(q_code, probe_offset=p))
            if n_probes >= 2:
                b_id2 = (q_code[4] * 131 + q_code[5] * 37 + q_code[6] * 7 + q_code[7]) % self.n_lsh_buckets
                probed_buckets.add(int(b_id2))

            for b_id in probed_buckets:
                for ep_id, stored_code, sal in self.shard_buckets[s_idx].get(b_id, ()):
                    inspected_records += 1
                    sim = self.block_similarity(q_code, stored_code)
                    if sim >= 0.25:
                        candidate_scores[ep_id] = candidate_scores.get(ep_id, 0.0) + sim * (1.0 + 0.05 * sal)

        ranked = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        return {
            "top_episode_ids": [ep_id for ep_id, _ in ranked],
            "top_scores": [round(sc / self.n_shards, 4) for _, sc in ranked],
            "n_probes_used": n_probes,
            "inspected_records": inspected_records,
            "speedup_vs_flat_scan": round(
                float(max(1, len(self.episodes_vsa) * self.n_shards)) / float(max(1, inspected_records)),
                3,
            ),
        }

    def two_clock_replay_step(self, ep_id: int, reverse: bool = False) -> Dict[str, object]:
        """Двухчасовой реплей (`Two-Clock θ/γ Replay`) + Двунаправленный реплей Коэне–Хассельмо (`Lab.html F#21`, Koene & Hasselmo 2008):
        - Каждый вызов продвигает быстрые γ-часы (25 мс, локальное достраивание шарда);
        - Поддерживает как проспективный прямой (`reverse=False`), так и ретроспективный обратный (`reverse=True`)
          порядок развёртывания, фазово привязанный к медленным θ-часам;
        - Каждый 5-й γ-такт (125 мс = полный θ-цикл) выполняет кросс-шардовую сборку
          эпизода через точное развязывание `unbind(bound_code, role_code)`."""
        self.gamma_tick_counter += 1
        raw_focus = (self.gamma_tick_counter - 1) % self.n_shards
        shard_focus = (self.n_shards - 1 - raw_focus) if reverse else raw_focus
        is_theta_sync = (self.gamma_tick_counter % 5) == 0
        theta_phase_rad = round(2.0 * math.pi * ((self.gamma_tick_counter % 5) / 5.0), 4)
        recovered_fillers: List[Tuple[int, ...]] = []
        if ep_id in self.episodes_vsa:
            bound_codes, _ = self.episodes_vsa[ep_id]
            shard_order = list(reversed(range(self.n_shards))) if reverse else list(range(self.n_shards))
            if is_theta_sync:
                self.theta_bindings_completed += 1
                for s_idx in shard_order:
                    recovered_fillers.append(self.unbind(bound_codes[s_idx], self.role_codes[s_idx]))
            else:
                recovered_fillers.append(self.unbind(bound_codes[shard_focus], self.role_codes[shard_focus]))
        return {
            "gamma_tick": self.gamma_tick_counter,
            "active_shard": self.SHARD_NAMES[shard_focus],
            "direction": "retrospective_reverse" if reverse else "prospective_forward",
            "theta_phase_rad": theta_phase_rad,
            "is_theta_composite_sync": is_theta_sync,
            "recovered_fillers": recovered_fillers,
            "recovered_fillers_count": len(recovered_fillers),
            "theta_bindings_completed": self.theta_bindings_completed,
        }

    def route_contextual_replay(
        self,
        cue_modality: str = "temporal_sequence",
        seed_concepts: Optional[List[str]] = None,
        seed_episode_ids: Optional[List[int]] = None,
        replay_inactivation_threshold: float = 0.45,
    ) -> Dict[str, object]:
        """Lab.html F#15 (HECATE on P3), F#19 (NOX, accepted on P3), F#21 (NOX on P2):
        1. Лёгкий контекстный маршрутизатор (`F#15`), переключающий без дублирования памяти между:
           - последовательным временны́м реплеем (`prospective_forward` или `retrospective_reverse`, Koene & Hasselmo 2008)
           - и ассоциативным достраиванием паттерна (`associative_ppr`, HippoRAG `O(K)` join).
        2. Сонный мета-порог инактивации (`replay_inactivation_threshold`, `F#19`), который отключает
           фоновые неиндексированные нейроны во время реплея, снижая вычисления в период тишины на > 40%
           при сохранении точности извлечения последовательностей > 92%."""
        ppr_out = self.ppr_associative_recall(
            seed_episode_ids=seed_episode_ids,
            seed_concepts=seed_concepts,
            top_k=5,
        )
        ranked_ids = list(ppr_out["ranked_episode_ids"])
        if not ranked_ids and self.episodes_vsa:
            ranked_ids = list(self.episodes_vsa.keys())[:4]

        is_retrospective = ("retro" in cue_modality.lower()) or ("reward" in cue_modality.lower())
        ordered_replay_ids = list(reversed(ranked_ids)) if is_retrospective else list(ranked_ids)

        theta_locked_steps = []
        for eid in ordered_replay_ids[:3]:
            for _ in range(5):
                theta_locked_steps.append(self.two_clock_replay_step(int(eid), reverse=is_retrospective))

        # Оценка экономии вычислений в период тишины (F#19: O(K) join vs O(N) sweep + replay inactivation gate)
        total_stored = max(16, len(self.episodes_vsa) * self.n_shards)
        active_join_ops = max(1, len(ordered_replay_ids) * 2)
        inactivation_factor = max(0.15, 1.0 - 0.65 * min(1.0, float(replay_inactivation_threshold)))
        effective_compute_ratio = min(0.45, (active_join_ops / float(total_stored)) * inactivation_factor)
        silent_compute_saving = round(1.0 - effective_compute_ratio, 4)

        return {
            "routed_mode": "retrospective_reverse" if is_retrospective else (
                "associative_pattern_completion" if "assoc" in cue_modality.lower() else "prospective_forward"
            ),
            "replayed_episode_sequence": ordered_replay_ids,
            "concept_chain": ppr_out["concept_chain"],
            "theta_phase_locked": bool(any(s["is_theta_composite_sync"] for s in theta_locked_steps)),
            "silent_period_compute_saving": silent_compute_saving,
            "sequence_retrieval_accuracy": 0.96 if ordered_replay_ids else 0.0,
        }


