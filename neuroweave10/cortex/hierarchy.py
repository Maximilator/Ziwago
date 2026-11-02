"""Кортикальная иерархия (§II.2.3): зоны на градиенте τ(g).

SENS(g=0, τ≈20мс) → ASSOC(g=0.5, τ≈100мс) → PFC(g=0.9, τ≈1-2с).
FF: выход зоны k → basal зоны k+1 (через HO-трансталамический путь).
FB: выход зоны k+1 → apical зоны k (NMDA, предсказание сверху).
GW-бродкаст идёт в апикал всех зон.

Внутренняя метрика: error = S - φ(V*_D) по каждой зоне — несоответствие
между спайком и дендритным «обещанием» (предиктивное кодирование
Rao & Ballard 1999 внутри нейрона).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch

from ..config import GenesisConfig
from ..core.sdr import LCASparseEncoder
from .column import CorticalColumn, ColumnOut


@dataclass
class HierarchyOut:
    zones: Dict[str, ColumnOut]
    errors: Dict[str, torch.Tensor]
    l5: Dict[str, torch.Tensor]     # L5-драйверы каждой зоны
    spikes_all: torch.Tensor        # конкатенация для монитора критичности
    local_sigmas: Optional[Dict[str, float]] = None  # D3: локальные σ каждой колонки без глобального readout
    thalamic_hub_load: float = 0.0                   # D1: нагрузка на таламический Pulvinar/MD хаб
    gossip_plv: float = 0.0                          # Фаза 2 (snnscience.docx §3.4): локальный Gossip-PLV без барьера
    polychronous_groups_active: int = 0              # Фаза 2 (Izhikevich 2006): резонансные полихронные триплеты


class AxonalDelayPolychronousRouter:
    """Фаза 2 Глобального Плана v8.0 (snnscience.docx §3.4, Изъян 4):
    Асинхронный маршрутизатор аксональных задержек (10–50 мс), локальной фазовой когерентности
    `Gossip-PLV`, полихронных групп Ижикевича (`Izhikevich 2006`) и пластической миелинизации (`Fields 2015`).

    Устраняет Изъян 4 («Глобальный барьер фазовой синхронизации PLV и игнорирование задержек проведения»):
      1. Каждое межзональное ребро (i -> j) имеет физическую аксональную задержку проведения
         τ_ij ∈ [10.0, 50.0] мс и подчиняется Правилу Четверти Периода:
             f_max^(i,j) = 1000 / (4 * τ_ij) Гц.
      2. Кольцевой буфер истории спайков (`ring_len = 64 мс`) доставляет спайки из зоны i в зону j
         ровно через τ_ij мс без глобальных барьеров синхронизации.
      3. `Gossip-PLV`: каждая зона хранит локальную фазу Курамото φ_i(t) и оценивает когерентность
         только по локальным соседям графа `small_world_hub` с учётом задержки τ_ij.
      4. Детектор полихронных групп (`Izhikevich 2006`): фиксирует коалиции (i, j -> k), в которых
         разновременные спайки из зон i (в момент t - τ_ik) и j (в момент t - τ_jk) приходят в зону k
         когерентно в одно и то же миллисекундное окно!
      5. Пластическая миелинизация во сне (`consolidate_myelin`): подстраивает задержки τ_ij так,
         чтобы задержанные спайки попадали в фазовый пик приёмной зоны (Communication Through Coherence).
    """

    def __init__(self, n_zones: int, routing_edges: List[Tuple[int, int]], ring_len: int = 64):
        self.n_zones = max(1, int(n_zones))
        self.routing_edges = list(routing_edges)
        self.ring_len = max(64, int(ring_len))

        # Инициализируем физические задержки проведения τ_ij ∈ [10..50] мс пропорционально дистанции |j - i|
        self.edge_delays_ms: Dict[Tuple[int, int], float] = {}
        self.edge_f_max_hz: Dict[Tuple[int, int], float] = {}
        for i, j in self.routing_edges:
            dist = abs(int(j) - int(i))
            tau_ms = min(50.0, 10.0 + 8.0 * float(dist - 1) + 2.0 * float((i + j) % 3))
            self.edge_delays_ms[(i, j)] = tau_ms
            self.edge_f_max_hz[(i, j)] = round(1000.0 / (4.0 * tau_ms), 2)

        # Кольцевой буфер истории средних L5-залпов зон [ring_len, n_zones] и фаз φ_i(t) [ring_len, n_zones]
        self.spike_ring = torch.zeros(self.ring_len, self.n_zones)
        self.phase_ring = torch.zeros(self.ring_len, self.n_zones)
        self.cur_phases = torch.linspace(0.0, math.pi * 0.5, self.n_zones)
        # Собственные частоты осцилляторов зон по градиенту иерархии (от γ ~28 Гц в SENS до θ ~8 Гц в PFC)
        self.natural_freq_hz = torch.linspace(28.0, 8.0, self.n_zones)
        self.ptr = 0
        self.tick_count = 0

        # Реестр обнаруженных полихронных групп (i, j, k, lag_diff_ms) -> накопленный резонанс
        self.polychronous_registry: Dict[Tuple[int, int, int, int], float] = {}
        self.last_gossip_plv: float = 0.0
        self.last_debiased_wpli: float = 0.0
        self.last_volume_conduction_ghost: float = 0.0
        self.false_witness_flag: bool = False
        self.last_polychronous_active: int = 0
        # EMA мнимой части кросс-спектра Im(S_ij) ~ sin(Δφ) и |Im(S_ij)| по рёбрам (Vinck et al., NeuroImage 2011)
        self.edge_im_ema: Dict[Tuple[int, int], float] = {e: 0.0 for e in self.routing_edges}
        self.edge_abs_im_ema: Dict[Tuple[int, int], float] = {e: 1e-4 for e in self.routing_edges}
        self.edge_sq_im_ema: Dict[Tuple[int, int], float] = {e: 1e-6 for e in self.routing_edges}

    def step_gossip_and_polychrony(
        self,
        zone_l5_rates: torch.Tensor,
        dt_ms: float = 1.0,
        common_volume_field: float = 0.0,
    ) -> Dict[str, float]:
        """Асинхронный шаг Gossip-PLV, Debiased WPLI (Vinck et al. 2011, Lab.html F#25/F#27/F#23)
        и детекции полихронных групп за O(|E|)."""
        self.tick_count += 1
        rates = zone_l5_rates.detach().flatten()[: self.n_zones]
        if rates.numel() < self.n_zones:
            rates = torch.nn.functional.pad(rates, (0, self.n_zones - rates.numel()))

        # 1. Обновление локальных фаз Курамото с задержанным Gossip-взаимодействием по рёбрам графа
        dphi = 2.0 * math.pi * self.natural_freq_hz * (dt_ms * 1e-3)
        gossip_cos_sum = 0.0
        wpli_sum = 0.0
        gossip_edges = 0
        vc_blend = max(0.0, min(0.95, float(common_volume_field)))
        for i, j in self.routing_edges:
            tau_steps = max(1, min(self.ring_len - 1, int(round(self.edge_delays_ms[(i, j)]))))
            delayed_idx = (self.ptr - tau_steps) % self.ring_len
            phi_i_delayed = float(self.phase_ring[delayed_idx, i].item())
            phi_j_now = float(self.cur_phases[j].item())
            # Если присутствует общая объёмная проводимость (common_volume_field > 0),
            # она стягивает наблюдаемую разность фаз к 0 (нулевой лаг без реальной синаптической задержки)
            phase_diff = (1.0 - vc_blend) * (phi_i_delayed - phi_j_now)
            # Связь Курамото масштабируется правилом четверти периода
            f_scale = min(1.0, self.edge_f_max_hz[(i, j)] / 25.0)
            dphi[j] = dphi[j] + 0.18 * f_scale * math.sin(phase_diff)
            gossip_cos_sum += 0.5 * (1.0 + math.cos(phase_diff))

            # --- Lab.html F#25, F#27, F#23 (accepted/replicated on P4): Debiased WPLI (Vinck et al. 2011) ---
            # Мнимая часть кросс-спектра Im(X_i X_j*) = sin(Δφ) строго равна 0 при мгновенной объёмной
            # проводимости (Δφ = 0), но максимальна при истинном синаптическом проведении с задержкой!
            im_val = math.sin(phase_diff)
            e_key = (i, j)
            self.edge_im_ema[e_key] = 0.85 * self.edge_im_ema.get(e_key, 0.0) + 0.15 * im_val
            self.edge_abs_im_ema[e_key] = 0.85 * self.edge_abs_im_ema.get(e_key, 1e-4) + 0.15 * abs(im_val)
            self.edge_sq_im_ema[e_key] = 0.85 * self.edge_sq_im_ema.get(e_key, 1e-6) + 0.15 * (im_val * im_val)
            # Дебиасированный взвешенный индекс фазового сдвига с подавлением малых углов объёмной проводимости (|sin Δφ|):
            num_wpli = max(0.0, (self.edge_im_ema[e_key] ** 2) - 0.15 * self.edge_sq_im_ema[e_key])
            den_wpli = max(1e-4, (self.edge_abs_im_ema[e_key] ** 2) + 0.05)
            edge_wpli = min(1.0, num_wpli / den_wpli)
            wpli_sum += edge_wpli
            gossip_edges += 1

        self.cur_phases = torch.remainder(self.cur_phases + dphi, 2.0 * math.pi)
        self.ptr = (self.ptr + 1) % self.ring_len
        self.spike_ring[self.ptr] = rates
        self.phase_ring[self.ptr] = self.cur_phases

        self.last_gossip_plv = round(gossip_cos_sum / float(max(1, gossip_edges)), 4)
        self.last_debiased_wpli = round(wpli_sum / float(max(1, gossip_edges)), 4)
        self.last_volume_conduction_ghost = round(max(0.0, self.last_gossip_plv - self.last_debiased_wpli), 4)
        # Если сырой PLV высокий (> 0.70), а Debiased WPLI падает (< 0.25), поднимаем флаг ложного свидетеля!
        self.false_witness_flag = bool(self.last_gossip_plv > 0.70 and self.last_debiased_wpli < 0.25)

        # 2. Детекция полихронных групп (Izhikevich 2006):
        # Ищем пары входящих рёбер (i -> k) и (j -> k), чьи задержанные спайки из моментов
        # (t - τ_ik) и (t - τ_jk) одновременно превысили порог и пришли в зону k в текущий такт!
        incoming_by_target: Dict[int, List[Tuple[int, int, float]]] = {}
        for i, k in self.routing_edges:
            tau_ik = max(1, min(self.ring_len - 1, int(round(self.edge_delays_ms[(i, k)]))))
            r_delayed = float(self.spike_ring[(self.ptr - tau_ik) % self.ring_len, i].item())
            if r_delayed > 0.02:
                incoming_by_target.setdefault(k, []).append((i, tau_ik, r_delayed))

        active_poly = 0
        for k, arrivals in incoming_by_target.items():
            # Даже одиночное резонансное совпадение задержанного афферента с активностью зоны k
            # или пара когерентных приходов формирует полихронный аккорд
            if len(arrivals) >= 2:
                for idx_a in range(len(arrivals)):
                    for idx_b in range(idx_a + 1, len(arrivals)):
                        i, tau_ik, r_i = arrivals[idx_a]
                        j, tau_jk, r_j = arrivals[idx_b]
                        key = (min(i, j), max(i, j), k, abs(tau_ik - tau_jk))
                        self.polychronous_registry[key] = min(
                            5.0, self.polychronous_registry.get(key, 0.0) + 0.25 * (r_i + r_j)
                        )
                        active_poly += 1
            elif len(arrivals) == 1 and float(rates[k].item()) > 0.02:
                i, tau_ik, r_i = arrivals[0]
                key = (i, k, k, tau_ik)
                self.polychronous_registry[key] = min(
                    5.0, self.polychronous_registry.get(key, 0.0) + 0.15 * (r_i + float(rates[k].item()))
                )
                active_poly += 1

        self.last_polychronous_active = active_poly
        return {
            "gossip_plv": self.last_gossip_plv,
            "debiased_wpli": self.last_debiased_wpli,
            "volume_conduction_ghost": self.last_volume_conduction_ghost,
            "false_witness_flag": float(self.false_witness_flag),
            "polychronous_groups_active": float(active_poly),
            "polychronous_catalog_size": float(len(self.polychronous_registry)),
        }

    def consolidate_myelin(self, lr_myelin: float = 0.35) -> Dict[str, float]:
        """Активность-зависимая пластичность миелинизации во сне (`Fields 2015`):
        Подстраивает аксональные задержки τ_ij ∈ [10.0, 50.0] мс так, чтобы задержанные
        залпы приходили в фазовый резонанс с приёмной зоной (рост Gossip-PLV)."""
        total_shift = 0.0
        for i, j in self.routing_edges:
            tau_now = self.edge_delays_ms[(i, j)]
            tau_steps = max(1, min(self.ring_len - 1, int(round(tau_now))))
            delayed_idx = (self.ptr - tau_steps) % self.ring_len
            phi_i_del = float(self.phase_ring[delayed_idx, i].item())
            phi_j_now = float(self.cur_phases[j].item())
            # Градиент минимизации фазового рассогласования
            delta_tau = -float(lr_myelin) * math.sin(phi_j_now - phi_i_del)
            new_tau = max(10.0, min(50.0, tau_now + delta_tau))
            total_shift += abs(new_tau - tau_now)
            self.edge_delays_ms[(i, j)] = new_tau
            self.edge_f_max_hz[(i, j)] = round(1000.0 / (4.0 * new_tau), 2)
            # Обновляем задержанную фазу в кольцевом буфере после миелиновой подстройки
            self.phase_ring[delayed_idx, i] = (
                0.65 * self.phase_ring[delayed_idx, i] + 0.35 * self.cur_phases[j]
            )
        return {
            "mean_delay_ms": round(
                sum(self.edge_delays_ms.values()) / float(max(1, len(self.edge_delays_ms))), 3
            ),
            "mean_myelin_shift_ms": round(total_shift / float(max(1, len(self.edge_delays_ms))), 4),
        }


class CorticalHierarchy:
    """Цепочка зон SENS→ASSOC→PFC с FF/FB и градиентом τ(g), поддерживающая
    D1-масштабирование (3 → 5 → 10 → 20+ зон) через:
      - 'small_world_hub' (по умолчанию): локальная цепочка + Small-World skip-связи (fan-out k<=3)
        + Трансталамический Pulvinar Hub-and-Spoke за O(N_zones);
      - 'linear_chain': строго последовательная цепочка O(N_zones);
      - 'all_to_all': полносвязная кортико-кортикальная сеть O(N_zones²)."""

    def __init__(self, cfg: GenesisConfig, batch: int = 1,
                 topology: str = "small_world_hub", max_fanout: int = 3):
        self.cfg = cfg
        self.batch = batch
        self.topology = topology
        self.max_fanout = max(1, int(max_fanout))
        zs = cfg.zones
        self.encoder = LCASparseEncoder(cfg.input_dim, cfg.encoder, cfg.seed)
        self.cols: List[CorticalColumn] = []
        for i, z in enumerate(zs):
            n_in = cfg.encoder.n_sdr if i == 0 else zs[i - 1].n_neurons
            n_fb = zs[i + 1].n_neurons if i < len(zs) - 1 else cfg.gw.d_gw
            self.cols.append(CorticalColumn(z, n_in, n_fb, cfg.neuron,
                                            cfg.synapse, seed=cfg.seed + i))
        self._prev: Dict[str, torch.Tensor] = {}

        # --- D1 (archnext.txt): Small-World Skip-Edges + Thalamic Pulvinar Hub-and-Spoke ---
        # Вместо O(N²) связей каждая зона i имеет не более max_fanout (по умолчанию k=3)
        # исходящих проекций (i+1 и дальнодействующие степени двойки i+2, i+4) и 1 связь с
        # низкоразмерным Таламическим Хабом (d_hub = 32), что даёт O(k * N_zones) сложность!
        self.d_hub = 32
        g_hub = torch.Generator().manual_seed(cfg.seed + 777)
        self.hub_in_proj: List[torch.Tensor] = []
        self.hub_out_proj: List[torch.Tensor] = []
        for i, col in enumerate(self.cols):
            n_l5 = int(col.l5_idx.numel())
            n_in_i = cfg.encoder.n_sdr if i == 0 else zs[i - 1].n_neurons
            self.hub_in_proj.append(
                torch.randn(self.d_hub, n_l5, generator=g_hub) / max(1.0, n_l5 ** 0.5)
            )
            self.hub_out_proj.append(
                torch.randn(n_in_i, self.d_hub, generator=g_hub) / max(1.0, self.d_hub ** 0.5)
            )
        self.hub_state = torch.zeros(batch, self.d_hub)
        # Список рёбер маршрутизации для выбранной топологии (для анализа fan-out и O(N) vs O(N²))
        self.routing_edges: List[tuple[int, int]] = self._build_routing_edges(len(zs))
        # Фаза 2 (snnscience.docx §3.4): Асинхронный маршрутизатор задержек, Gossip-PLV и полихронных групп
        self.polychronous_router = AxonalDelayPolychronousRouter(
            n_zones=len(zs),
            routing_edges=self.routing_edges,
            ring_len=64,
        )
        self.reset_state(batch)

    def _build_routing_edges(self, n_zones: int) -> List[tuple[int, int]]:
        edges: List[tuple[int, int]] = []
        if self.topology == "all_to_all":
            for i in range(n_zones):
                for j in range(n_zones):
                    if i != j:
                        edges.append((i, j))
        elif self.topology == "linear_chain":
            for i in range(n_zones - 1):
                edges.append((i, i + 1))
                edges.append((i + 1, i))
        else:
            # 'small_world_hub': локальный сосед i+1 + логарифмические skip-рёбра (до max_fanout)
            for i in range(n_zones):
                fan = 0
                step = 1
                while i + step < n_zones and fan < self.max_fanout:
                    edges.append((i, i + step))
                    fan += 1
                    step *= 2
                if i > 0:
                    edges.append((i, i - 1))
        return edges

    def reset_state(self, batch: int, device=None, dtype=torch.float32) -> None:
        self.batch = batch
        for c in self.cols:
            c.reset_state(batch, device, dtype)
        self._prev = {}

    # ------------------------------------------------------------------
    def step(self, x: torch.Tensor, gw_broadcast: Optional[torch.Tensor],
             ho_drive: Dict[str, torch.Tensor],
             f_mod: float, vip_per_zone: Dict[str, float],
             learn: bool = True, dt: float = 1.0) -> HierarchyOut:
        """Параллельный такт: каждая зона получает FF (prev L5/SDR через HO),
        FB (следующая зона, NMDA) и GW-бродкаст в апикал."""
        zs, outs, errs, l5s = self.cfg.zones, {}, {}, {}
        local_sigmas: Dict[str, float] = {}
        spikes = []
        # D1: Вычисляем вектор состояния Таламического Pulvinar-Хаба за O(N_zones)
        hub_vec = self.hub_state
        if hub_vec.shape[0] != self.batch:
            hub_vec = torch.zeros(self.batch, self.d_hub)
        new_hub_accum = torch.zeros(self.batch, self.d_hub)

        for i, col in enumerate(self.cols):
            z = zs[i]
            # FF: SDR для SENS; трансталамический HO-драйв + прямой кортико-кортикальный
            # выход предыдущей зоны + Small-World Pulvinar Hub (для мгновенного прохождения N_zones -> 20)
            if i == 0:
                x_ff = self.encoder.encode(x, learn=learn)
            else:
                ho_in = ho_drive.get(z.name, torch.zeros(
                    self.batch, zs[i - 1].n_neurons))
                prev_zone_s = outs[zs[i - 1].name].s
                x_ff = 0.5 * ho_in + 0.5 * prev_zone_s
                if self.topology == "small_world_hub" and len(self.cols) > 3:
                    # Трансталамический skip-драйв предотвращает затухание градиента при N_zones = 5..20,
                    # но при избыточном fan-out (> 4) возникает интерференция хаба
                    fan_pen = 1.0 / (1.0 + 0.15 * max(0, self.max_fanout - 3))
                    hub_ff = torch.relu(hub_vec @ self.hub_out_proj[i].t()) * (0.22 * fan_pen)
                    if hub_ff.shape == x_ff.shape:
                        x_ff = 0.82 * x_ff + 0.18 * hub_ff
            # FB (Слой L1 Таламическая Матрица + Нисходящая кора, Узел 2):
            # каждая зона получает И локальное предсказание от вышестоящей зоны,
            # И диффузную проекцию Симфонического Аккорда (gw_broadcast) в слой L1!
            n_fb = zs[i + 1].n_neurons if i < len(self.cols) - 1 else self.cfg.gw.d_gw
            if i < len(self.cols) - 1:
                nxt = self._prev.get(zs[i + 1].name)
                x_fb_top = nxt if nxt is not None else torch.zeros(self.batch, n_fb)
                if gw_broadcast is not None:
                    gwb = gw_broadcast.flatten()[:n_fb]
                    gwb = torch.nn.functional.pad(gwb, (0, max(0, n_fb - gwb.numel())))
                    x_fb = 0.70 * x_fb_top + 0.30 * gwb.unsqueeze(0).expand(self.batch, -1)
                else:
                    x_fb = x_fb_top
            else:
                x_fb = gw_broadcast if gw_broadcast is not None else \
                    torch.zeros(self.cfg.gw.d_gw)
            if x_fb.dim() == 1:
                x_fb = x_fb.unsqueeze(0).expand(self.batch, -1)
            if x_ff.dim() == 1:
                x_ff = x_ff.unsqueeze(0).expand(self.batch, -1)
            out = col.step(x_ff, x_fb, f_mod, vip_per_zone.get(z.name, 0.0),
                           learn=learn, dt=dt)
            self._prev[z.name] = out.s
            outs[z.name], errs[z.name] = out, out.error
            l5s[z.name] = out.l5_drive
            local_sigmas[z.name] = float(out.local_sigma)
            spikes.append(out.s)
            if out.l5_drive.shape[-1] == self.hub_in_proj[i].shape[1]:
                new_hub_accum = new_hub_accum + out.l5_drive @ self.hub_in_proj[i].t()

        self.hub_state = torch.tanh(new_hub_accum / max(1, len(self.cols)))
        hub_load = float(self.hub_state.abs().mean().item()) * (
            1.0 + 0.08 * (len(self.routing_edges) / max(1, len(self.cols)))
        )
        l5_rates = torch.tensor(
            [float(l5s[z.name].abs().mean().item()) for z in zs],
            dtype=torch.float32,
        )
        poly_stat = self.polychronous_router.step_gossip_and_polychrony(l5_rates, dt_ms=dt)
        return HierarchyOut(zones=outs, errors=errs, l5=l5s,
                            spikes_all=torch.cat(spikes, dim=1),
                            local_sigmas=local_sigmas,
                            thalamic_hub_load=hub_load,
                            gossip_plv=float(poly_stat["gossip_plv"]),
                            polychronous_groups_active=int(poly_stat["polychronous_groups_active"]))

    def inject_apical_error(self, l5_error_all: torch.Tensor,
                            f_mod: float = 1.0,
                            ach: float = 0.5,
                            da_phasic: Optional[float] = None) -> Dict[str, float]:
        """P1 (Иерархическое дендритное назначение кредита, Sacramento et al. 2018):
        Каскадная передача векторной ошибки из моторной коры в L5-нейроны и сверху вниз
        (PFC → ASSOC → SENS) через физические апикальные синапсы обратной связи w_fb."""
        if f_mod <= 0.0:
            return {c.name: 0.0 for c in self.cols}
        err_flat = l5_error_all.detach()
        if err_flat.dim() == 2:
            err_flat = err_flat.mean(0)
        # Разбиваем вектор L5-ошибки по зонам в том же порядке, что и _gather_l5()
        l5_slices: Dict[str, torch.Tensor] = {}
        offset = 0
        for col in self.cols:
            n_l5 = int(col.l5_idx.numel())
            if offset + n_l5 <= err_flat.numel():
                l5_slices[col.name] = err_flat[offset:offset + n_l5]
            else:
                l5_slices[col.name] = torch.zeros(n_l5, device=err_flat.device)
            offset += n_l5

        report: Dict[str, float] = {}
        upper_err: Optional[torch.Tensor] = None
        # Проход сверху вниз: PFC (k = L-1) → ASSOC (k = L-2) → SENS (k = 0)
        for i in range(len(self.cols) - 1, -1, -1):
            col = self.cols[i]
            n_z = col.zcfg.n_neurons
            z_err = torch.zeros(self.batch, n_z, device=err_flat.device)
            # 1) Прямая эфферентная копия ошибки в L5-драйверы зоны
            z_err[:, col.l5_idx] = l5_slices[col.name].unsqueeze(0).expand(self.batch, -1)
            # 2) Нисходящая проекция от вышестоящей зоны через физические синапсы w_fb
            if upper_err is not None and upper_err.shape[-1] == col.w_fb.n_pre:
                fb_nudge = torch.tanh(col.w_fb.drive(upper_err))
                z_err = z_err + 0.65 * fb_nudge
                # Реципрокное выравнивание Kolen-Pollack: w_fb учится передавать согласованный сигнал
                if hasattr(col, "_last_s"):
                    col.w_fb.learn(z_err, upper_err.abs(), col._last_s,
                                   f_mod * 0.35, ach=ach, da_phasic=da_phasic)
            upper_err = col.inject_apical_error(
                z_err, f_mod=f_mod, ach=ach, da_phasic=da_phasic
            )
            report[col.name] = float(upper_err.abs().mean().item())
        return report

    def consolidate(self, factor: float = 0.5,
                    spindle_gain: float = 1.0,
                    zone_errors: Optional[Dict[str, float]] = None,
                    microglia_boost: float = 1.0,
                    bdnf: float = 0.35) -> None:
        for c in self.cols:
            err_lvl = None if zone_errors is None else zone_errors.get(c.name, None)
            c.consolidate(factor, spindle_gain=spindle_gain,
                          error_level=err_lvl, microglia_boost=microglia_boost,
                          bdnf=bdnf)
        if hasattr(self, "polychronous_router") and self.polychronous_router is not None:
            self.polychronous_router.consolidate_myelin()
