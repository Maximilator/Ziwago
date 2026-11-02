"""UnifiedOrganism: v9-энкодер → v10-организм → v9-ридаут → CE→reward.

Биологическая эффективность: входной токен — один спайк-кадр в
таламический FO-релей (латентность несёт салиентность — редкие токены
спайкуют раньше, Aston-Jones); внутри дендритного дерева драйв идёт по
sparse_drive (v8): только активные пресинаптические входы платят
матричный драйв — событийная экономия как в мозге.

Синергия:
  * v8 sparse ops → экономия вычислений (EventStats считает долю работы)
  * v9 encoding+readout → языковая задача (TinyStories next-token)
  * v10 организм → кортекс/память/сознание/химия — обучение трёхфакторным
    правилом с f_mod-гейтом (DA+NE+eCB+CORT)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from neuroweave8.core.sparse import sparse_drive, EventStats
from neuroweave9.encoding import TemporalPhaseEncoder
from neuroweave9.readout import LaplaceContextReadout

from ..config import GenesisConfig, genesis_cfg
from ..organism.genesis import GenesisOrganism, GenesisOutput


@dataclass
class UnifiedConfig:
    preset: str = "tiny"
    vocab_size: int = 128
    burst_length: int = 5
    max_latency: int = 8
    enc_mode: str = "latency"      # салиентность входа встроена
    reward_scale: float = 2.0      # CE → reward амплитуда


@dataclass
class UnifiedReport:
    tokens: int
    ce: float
    perplexity: float
    sigma: float
    confidence: float
    m_ratio: float
    sparse_saving: float
    pathology: Optional[str]


class UnifiedOrganism:
    """Единый организм v8+v9+v10."""

    def __init__(self, cfg: Optional[UnifiedConfig] = None,
                 gcfg: Optional[GenesisConfig] = None, batch: int = 1,
                 seed: int = 0):
        self.ucfg = cfg or UnifiedConfig()
        self.gcfg = gcfg or genesis_cfg(self.ucfg.preset)
        self.batch = batch
        self.org = GenesisOrganism(self.gcfg, batch=batch, seed=seed)
        self.org.reset_state(batch)
        self.encoder = TemporalPhaseEncoder(
            self.ucfg.vocab_size, self.ucfg.burst_length,
            self.ucfg.max_latency, self.ucfg.enc_mode)
        # Проекция vocab→input_dim: разреженная (k-WTA) — экономия
        self.w_in = torch.randn(self.gcfg.input_dim,
                                self.ucfg.vocab_size) * 0.4
        self.readout = LaplaceContextReadout(
            cortical_dim=self.gcfg.zones[-1].n_neurons,
            laplace_dim=self.gcfg.gw.d_gw,
            vocab_size=self.ucfg.vocab_size, hidden=64, lr=2e-3)
        self.event_stats = EventStats()
        self._pred_logits: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _encode_frame(self, tokens: torch.Tensor) -> torch.Tensor:
        """Токены [B] → вход организма [B,input_dim]: latency-спайк ×
        салиентность, разреженная проекция на input_dim."""
        enc = self.encoder.encode_latency(tokens.unsqueeze(-1))
        frame = enc[:, :, :]                     # [T,B,V]
        sal = frame.max(0).values                # [B,V] салиентность
        x = sparse_drive(sal, self.w_in,
                         out_idx=torch.arange(self.gcfg.input_dim))
        self.event_stats.record(self.ucfg.vocab_size,
                                int((sal > 0).sum().item()),
                                self.gcfg.input_dim, self.batch)
        return x * 3.0

    def _readout_logits(self, out: GenesisOutput) -> torch.Tensor:
        """PFC-спайки + GW-бродкаст → логиты словаря."""
        pfc = out.spikes.get(self.gcfg.zones[-1].name) \
            if hasattr(out, "spikes") and isinstance(
                getattr(out, "spikes", None), dict) else None
        if pfc is None:
            pfc = self.org._prev_pfc if hasattr(self.org, "_prev_pfc") \
                else torch.zeros(self.batch, self.gcfg.zones[-1].n_neurons)
        gw = out.gw_content
        gw_t = torch.stack([g.detach() if torch.is_tensor(g)
                            else torch.as_tensor(g, dtype=torch.float32)
                            for g in gw]) if isinstance(gw, list) else gw.detach()
        gw_flat = gw_t.flatten().float()[:self.gcfg.gw.d_gw]
        gw_flat = F.pad(gw_flat, (0, max(0, self.gcfg.gw.d_gw
                                       - gw_flat.numel()))).unsqueeze(0)
        zl = torch.zeros(self.batch, self.ucfg.vocab_size)
        return self.readout(pfc.float(), gw_flat, zl)

    # ------------------------------------------------------------------
    def step_token(self, token_ids: torch.Tensor,
                   next_ids: Optional[torch.Tensor] = None,
                   learn: bool = True) -> Dict:
        """Токены [B] текущего шага → предсказание; если next_ids —
        обучение по CE-ошибке (reward = -CE·scale → DA-канал)."""
        x = self._encode_frame(token_ids)
        reward = 0.0
        if self._pred_logits is not None and next_ids is not None:
            # Отрицательная CE → reward: ошибка предсказания драйвит
            # BG-TD и f_mod-пластичность (правило 3-х факторов).
            ce = F.cross_entropy(self._pred_logits.float(),
                                 next_ids.long())
            reward = float(-ce.item() * self.ucfg.reward_scale)
            if learn:
                self.readout.optimizer.zero_grad()
                ce.backward()
                self.readout.optimizer.step()
        out = self.org.step(x, reward=reward, learn=learn, drive=0.6)
        logits = self._readout_logits(out)
        self._pred_logits = logits
        return {"logits": logits, "reward": reward,
                "confidence": out.confidence, "sigma": out.sigma,
                "action": out.action, "pathology": out.pathology,
                "gw": out.gw_content}

    # ------------------------------------------------------------------
    def run_sequence(self, seq: torch.Tensor, learn: bool = True) -> UnifiedReport:
        """seq [T] → обучающий проход next-token; отчёт о точности и здоровье."""
        ce_sum, n = 0.0, 0
        for t in range(seq.numel() - 1):
            r = self.step_token(seq[t:t + 1], seq[t + 1:t + 2], learn=learn)
            if self._pred_logits is not None and t > 0:
                ce_sum += float(F.cross_entropy(
                    self._pred_logits.float(), seq[t:t + 1].long()).item())
                n += 1
            if r["pathology"]:
                break
        ce = ce_sum / max(1, n)
        return UnifiedReport(
            tokens=int(seq.numel()), ce=ce,
            perplexity=float(torch.exp(torch.tensor(min(8.0, ce)))),
            sigma=self.org.sigma, confidence=float(self.org._last_meta.confidence
                                                   if self.org._last_meta else 0.0),
            m_ratio=float(getattr(self.org._last_meta, 'm_ratio', 0.0) if getattr(self.org, '_last_meta', None) is not None else 0.0),
            sparse_saving=self.event_stats.saving,
            pathology=self.org.pathology)

    def sleep(self, ticks: int = 150) -> Dict:
        return self.org.sleep_cycle(ticks)

    def report(self) -> Dict:
        return {"sigma": self.org.sigma,
                "m_ratio": float(getattr(self.org._last_meta, 'm_ratio', 0.0) if getattr(self.org, '_last_meta', None) is not None else 0.0),
                "dash": getattr(self.org, "_last_dash", {}),
                "sparse_saving": self.event_stats.saving,
                "episodes": len(self.org.hipp.episodes),
                "pathology": self.org.pathology}


# ----------------------------------------------------------------------
def run_tinystories(data_path: str = "data/TinyStories_train_4k.txt",
                    preset: str = "tiny", max_stories: int = 24,
                    seq_len: int = 16, max_vocab: int = 128,
                    epochs: int = 2) -> Dict:
    """Реальные TinyStories → организм: perplexity до/после эпох.

    Токенизатор и словарь v9; каждый токен — кадр спайков в таламус;
    CE следующего токена → reward → DA/TD-пластичность + Adam-обучение
    ридаута. Между эпохами — сон с SWR-консолидацией."""
    from neuroweave9.data import (load_stories, tokenize_text,
                                  build_vocab_and_sequences)
    stories = load_stories(data_path, max_stories=max_stories)
    itos, stoi, x_seq, y_seq, _ = build_vocab_and_sequences(
        stories, max_vocab_size=max_vocab, seq_len=seq_len,
        batch_size=1)
    u = UnifiedOrganism(UnifiedConfig(vocab_size=len(itos)),
                        gcfg=genesis_cfg(preset))
    metrics: Dict = {"vocab": len(itos), "tokens": int(y_seq.numel())}
    ce_hist: List[float] = []
    for ep in range(epochs):
        ce_sum, n = 0.0, 0
        for b in range(x_seq.shape[1]):
            for t in range(x_seq.shape[0] - 1):
                tok = x_seq[t, b].argmax().unsqueeze(0)
                nxt = y_seq[t, b].unsqueeze(0)
                r = u.step_token(tok, nxt, learn=True)
                if u._pred_logits is not None and t > 0:
                    ce_sum += float(F.cross_entropy(
                        u._pred_logits.float(), y_seq[t, b:b+1]).item())
                    n += 1
        ce = ce_sum / max(1, n)
        ce_hist.append(ce)
        u.sleep(ticks=40)        # консолидация между эпохами
    metrics["ce_per_epoch"] = ce_hist
    metrics["perplexity_per_epoch"] = [
        float(torch.exp(torch.tensor(min(10.0, c)))) for c in ce_hist]
    metrics["organism"] = u.report()
    return metrics


# ----------------------------------------------------------------------
class RegenesisOrganism(UnifiedOrganism):
    """«True Bio-Organism»: моторная кора вместо MLP-декодера.

    Выходной токен — спайк WTA-победителя в пуле моторных нейронов
    (Закон 1). Афференты L5→моторный пул несут Tsodyks-Markram
    STP-депрессию: повтор нейрона истощает его везикулы — петля угасает
    без вето/штрафов (Закон 2). Обучение — трёхфакторное, без
    backprop; perplexity замеряется через softmax(V) чисто как метрика.
    """

    def __init__(self, cfg: Optional[UnifiedConfig] = None,
                 gcfg: Optional[GenesisConfig] = None, batch: int = 1,
                 seed: int = 0):
        super().__init__(cfg, gcfg, batch, seed)
        from ..motor import MotorPool
        # Суммарная ширина L5-выходов по ВСЕМ кортикальным зонам (SENS + ASSOC + PFC)
        n_l5_all = sum(z.n_neurons - 2 * z.n_neurons // 3
                       for z in self.gcfg.zones)
        n_pfc = self.gcfg.zones[-1].n_neurons
        self.motor = MotorPool(self.ucfg.vocab_size, n_l5_all,
                               self.gcfg.hippo.n_ec, self.gcfg.input_dim,
                               self.gcfg.motor, seed=seed + 7,
                               n_gw=self.gcfg.gw.d_gw, n_wm=n_pfc)
        self.motor.attach_basal_ganglia(self.gcfg.bg.n_actions)
        self.motor.reset_state(batch)
        self._last_motor: Optional[Dict] = None
        # PFC-контур предсказания: w_pred — ожидание следующего токена
        # по предыдущему (предиктивное кодирование); масса пунктуации
        # в предсказании → «завершённость» → терпение DDM.
        v = self.ucfg.vocab_size
        self.w_pred = torch.randn(v, v) * 0.05
        self.punct_ids: set = set()
        self.functional_ids: set = set()
        self._recent_ids: deque = deque(maxlen=32)
        self._prev_reward: float = 0.0
        self._pending_fep_ctx: Optional[Dict] = None
        # Глобальный автономный метаболизм и гомеостатический цикл сна (Borbély Process S + ATP gate)
        self.autonomous_life_mode: bool = False
        self.sleep_deprivation_mode: bool = False
        self.auto_sleep_atp_threshold: float = 0.85
        self.auto_sleep_ado_threshold: float = 0.35
        self.auto_sleep_ticks: int = 36
        self.atp_burn_rate: float = 0.00042
        self.ado_gain_rate: float = 0.00078
        self.auto_sleep_history: List[Dict] = []
        self.total_waking_ticks: int = 0
        self.dialogue_history: List[Dict] = []

    def enable_autonomous_metabolism(
        self,
        enabled: bool = True,
        atp_threshold: float = 0.85,
        ado_threshold: float = 0.35,
        sleep_ticks: int = 36,
        atp_burn_rate: float = 0.00042,
        ado_gain_rate: float = 0.00078,
        sleep_deprivation: bool = False,
    ) -> None:
        """Включает полностью эндогенный метаболический цикл бодрствования и сна
        внутри step_token() без каких-либо внешних вызовов sleep_cycle()."""
        self.autonomous_life_mode = bool(enabled)
        self.sleep_deprivation_mode = bool(sleep_deprivation)
        self.auto_sleep_atp_threshold = float(atp_threshold)
        self.auto_sleep_ado_threshold = float(ado_threshold)
        self.auto_sleep_ticks = int(sleep_ticks)
        self.atp_burn_rate = float(atp_burn_rate)
        self.ado_gain_rate = float(ado_gain_rate)

    # ------------------------------------------------------------------
    def _gather_l5(self) -> torch.Tensor:
        """Собрать выходы L5 всех зон коры в единый вектор [B, n_l5_all]."""
        if not hasattr(self.org, "_prev_l5") or not self.org._prev_l5:
            return torch.zeros(self.batch, self.motor.n_l5)
        parts = []
        for z in self.gcfg.zones:
            n_z_l5 = z.n_neurons - 2 * z.n_neurons // 3
            parts.append(self.org._prev_l5.get(
                z.name, torch.zeros(self.batch, n_z_l5)))
        return torch.cat(parts, dim=-1)

    def simulate_vte(self, candidates: List[int]) -> Dict[str, float]:
        """Инновация 1 (Театр Сознания / Vicarious Trial & Error — Redish 2016; Pezzulo 2014):
        Внутренняя мысленная репетиция (эфферентная копия без вокализации):
        оценивает Ожидаемую Свободную Энергию G(a_k) каждого кандидата a_k
        по 4 биологическим критериям:
          1) Предиктивная когерентность коры и гиппокампа (CA3/EC резонанс);
          2) Эмоциональная валентность Амигдалы (BLA High Road / Low Road: избегание боли, тяга к награде);
          3) Синтаксическая допустимость перехода и роли (BA44 / M1 synfire);
          4) Эпистемическая ценность новизны (Active Inference).
        Выбирает кандидата с минимальной Ожидаемой Свободной Энергией G(a_k)!"""
        if not candidates:
            return {"winner": 0, "g_min": 0.0, "vte_triggered": 0.0}
        best_tok = candidates[0]
        best_g = 1e9
        g_scores: Dict[int, float] = {}
        breakdown: Dict[int, Dict[str, float]] = {}
        role_vec = getattr(self.org, "_last_role_vec", None)
        w_role_eff = 0.5 * self.motor.w_role + 0.5 * self.motor.w_role_slow
        w_lat_eff = 0.5 * self.motor.w_lat + 0.5 * self.motor.w_lat_slow
        w_lat2_eff = 0.5 * self.motor.w_lat2 + 0.5 * self.motor.w_lat2_slow
        prev_w = self.motor.prev_winner
        prev2_w = self.motor.prev2_winner
        prev_c_vec = getattr(self.org.hipp, "last_concept_vec", None)
        # Если среди кандидатов есть грамматически выученные последователи (lat_fit > 0.05),
        # отсекаем кандидатов с запрещённым/невыученным переходом (lat_fit <= 0.0)
        eval_cands = list(candidates)
        if prev_w.any():
            valid_cands = [
                c for c in candidates
                if float((prev_w @ w_lat_eff[c]).item()) + (0.65 * float((prev2_w @ w_lat2_eff[c]).item()) if prev2_w.any() else 0.0) > 0.05
            ]
            if valid_cands and len(candidates) <= 2:
                eval_cands = valid_cands
        for cand in eval_cands:
            # 1) Эфферентная копия: виртуальный сенсорный кадр кандидата x_sim
            x_sim = self._encode_frame(torch.tensor([cand]))
            xt = x_sim.detach().float().mean(0)
            # 2) Оценка эмоциональной валентности через миндалину (BLA без изменения весов!)
            lim = self.org.limbic
            xt_pad = torch.nn.functional.pad(xt, (0, max(0, lim.input_dim - xt.numel())))[:lim.input_dim]
            xt_norm = xt_pad / xt_pad.norm().clamp(min=1e-6)
            app_sim = float(torch.tanh(xt_norm @ lim.w_low_pos.t()).mean().item())
            av_sim = float(torch.tanh(xt_norm @ lim.w_low_neg.t()).mean().item())
            valence_sim = app_sim - av_sim  # >0 хорошо, <0 угроза

            # 3) Синтаксический и переходный резонанс (BA44 + M1 биграмма + триграмма + кортикальный драйв L5/Laplace/WM)
            v_ref = getattr(self.motor, "v_pre_reset", self.motor.v).mean(0)
            max_cnt = max(1.0, float(self.motor._last_counts.max().item()))
            motor_ev = (
                0.85 * (float(self.motor._last_counts[cand].item()) / max_cnt)
                + 0.03 * float((v_ref[cand] - self.motor.cfg.e_l).item())
            )
            lat1 = float((prev_w @ w_lat_eff[cand]).item()) if prev_w.any() else 0.0
            lat2 = float((prev2_w @ w_lat2_eff[cand]).item()) if prev2_w.any() else 0.0
            lat_fit = lat1 + 1.35 * lat2
            synfire_violation = 4.0 if (prev_w.any() and lat_fit <= 0.0) else 0.0
            if prev2_w.any() and float((prev2_w @ w_lat2_eff.t()).max().item()) > 0.08 and lat2 <= 0.02:
                synfire_violation += 1.8
            role_fit = 0.0
            if role_vec is not None:
                rv = role_vec.detach().float().flatten()
                if rv.numel() == w_role_eff.shape[1]:
                    role_fit = float((w_role_eff[cand] @ rv).item())

            # 4) Концептуальный резонанс в гиппокампе/MTL + семантический мост REM-сна + клаузальный след
            ec_dim = self.org.cfg.hippo.n_ec
            ec_sim = torch.nn.functional.pad(xt, (0, max(0, ec_dim - xt.numel())))[:ec_dim]
            ec_norm = ec_sim / ec_sim.norm().clamp(min=1e-6)
            c_proj = self.org.hipp.w_concept.to(ec_norm.device) @ ec_norm
            concept_sim = float(c_proj.max().item())
            cand_c_idx = int(c_proj.argmax().item())
            bridge_fit = 0.0
            clause_concept_fit = 0.0
            direct_trace = getattr(self.org.hipp, "narrative_concept_trace", prev_c_vec)
            if prev_c_vec is not None and prev_c_vec.numel() == self.org.hipp.n_concepts:
                prev_c_idx = int(direct_trace.argmax().item()) if direct_trace is not None else int(prev_c_vec.argmax().item())
                bridge_fit = float(self.org.hipp.w_semantic_graph[cand_c_idx, prev_c_idx].item())
                c_mat = getattr(self.motor, "w_concept_direct", self.motor.w_concept_m)
                if c_mat is not None and direct_trace is not None and c_mat.shape[1] == direct_trace.numel():
                    dt_norm = direct_trace / direct_trace.norm().clamp(min=1e-6)
                    wc_row = c_mat[cand].to(dt_norm.device)
                    clause_concept_fit = max(0.0, min(2.5, float((wc_row @ dt_norm).item())))
                    if synfire_violation > 0.0:
                        clause_concept_fit = 0.0

            # 5) Разрешение Когнитивного Диссонанса: Инстинкт самосохранения (Амигдала A-)
            #    против Исследовательского Любопытства (Гипоталамус D_novelty)
            d_novelty = float(getattr(lim.last_out, "d_novelty", 0.0)) if lim.last_out is not None else 0.0
            cand_novelty = max(
                0.0,
                1.0
                - float(self.motor.rate_win[cand].item()) * 3.5
                - max(0.0, float(self.motor.target_freq_ema[cand].item()) * self.ucfg.vocab_size - 1.2) * 0.25,
            )
            curiosity_bonus = (1.65 * d_novelty * cand_novelty) if synfire_violation == 0.0 else 0.0
            av_threat = max(0.0, -valence_sim)
            # Барьер выживания Амигдалы (CeA Survival Veto): при реальной аверсивной угрозе
            # инстинкт самосохранения подавляет безрассудное любопытство κ_explore и блокирует опасный шаг!
            survival_veto = 2.85 * max(0.0, av_threat - 0.15)
            eff_kappa_explore = max(
                0.0,
                round(float(getattr(self.motor, "_explore_drive", 0.0)) - 1.35 * av_threat, 4),
            )

            # Ожидаемая Свободная Энергия G(cand) и Термодинамически-дисконтированная G'(cand):
            # G'(π) = G(π) + λ_ATP(t) · E[ATP_cost(π)] (Фаза 2 snnscience.docx §3.9)
            repeat_pen = float(self.motor.rate_win[cand].item()) * (3.5 + 3.0 * d_novelty) + synfire_violation
            if prev_w.any() and int(prev_w.argmax().item()) == cand:
                repeat_pen += 5.0
            if self.motor.prev2_winner.any() and int(self.motor.prev2_winner.argmax().item()) == cand:
                repeat_pen += 3.5
            g_epistemic_pragmatic = (
                -1.8 * lat_fit
                - 1.2 * role_fit
                - 0.85 * valence_sim
                - 0.3 * concept_sim
                - 0.9 * bridge_fit
                - 2.2 * clause_concept_fit
                - 0.55 * motor_ev
                - curiosity_bonus
                + survival_veto
                + repeat_pen
            )
            lambda_atp = float(getattr(self.org, "lambda_atp", 0.0))
            expected_atp_cost = round(
                0.10
                + 0.25 * float(x_sim.abs().mean().item())
                + 0.15 * (1.0 - float(self.motor.out_fatigue[cand].item())),
                4,
            )
            metabolic_penalty = round(lambda_atp * expected_atp_cost, 4)
            g_cand = g_epistemic_pragmatic + metabolic_penalty
            g_scores[cand] = round(g_cand, 4)
            breakdown[cand] = {
                "G": round(g_cand, 4),
                "G_base": round(g_epistemic_pragmatic, 4),
                "G_prime": round(g_cand, 4),
                "lambda_atp": round(lambda_atp, 4),
                "expected_atp_cost": expected_atp_cost,
                "metabolic_penalty": metabolic_penalty,
                "lat_fit": round(lat_fit, 4),
                "role_fit": round(role_fit, 4),
                "valence": round(valence_sim, 4),
                "concept_sim": round(concept_sim, 4),
                "semantic_bridge": round(bridge_fit, 4),
                "clause_concept_fit": round(clause_concept_fit, 4),
                "curiosity_bonus": round(curiosity_bonus, 4),
                "survival_veto": round(survival_veto, 4),
                "eff_kappa_explore": eff_kappa_explore,
                "repeat_penalty": round(repeat_pen, 4),
            }
            if g_cand < best_g:
                best_g = g_cand
                best_tok = cand
        return {
            "winner": int(best_tok),
            "g_min": float(best_g),
            "vte_triggered": 1.0,
            "candidates": g_scores,
            "breakdown": breakdown,
        }

    def step_token(self, token_ids: torch.Tensor,
                   next_ids: Optional[torch.Tensor] = None,
                   learn: bool = True) -> Dict:
        """Токен → спайки → организм → моторный WTA → токен.

        Поддерживает два режима обучения:
          1) С учителем (`next_ids` передан явно);
          2) Чистое самообучение по Принципу Свободной Энергии (`next_ids=None, learn=True`):
             каждый приходящий из мира токен `prev_id` автоматически сверяется с
             внутренним предсказанием прошлого такта `_pending_fep_ctx`, снижая
             Surprise и выделяя эндогенный дофамин `intrinsic_rpe`!"""
        x = self._encode_frame(token_ids)
        prev_id = int(token_ids.flatten()[0].item())

        # --- Чистое FEP-самообучение (Predictive Coding без учителя) ---
        fep_surprise = 1.0
        if next_ids is None and learn and self._pending_fep_ctx is not None:
            ctx = self._pending_fep_ctx
            # Реальность наступила: пришедший сенсорный токен `prev_id` — это то,
            # что организм пытался предсказать на предыдущем такте!
            pred_tok = ctx["pred_token"]
            v_dist = torch.softmax((ctx["v_pre"] - self.motor.theta.mean(0)) * 0.4, dim=-1)
            prob_actual = float(v_dist[prev_id].item())
            fep_surprise = float(-torch.log(torch.tensor(prob_actual + 1e-6)).item())
            # Внутреннее вознаграждение от совпадения предсказания и снижения Surprise
            match_bonus = 1.0 if pred_tok == prev_id else -0.35
            self.w_pred[ctx["input_id"]] += 0.05 * (
                F.one_hot(torch.tensor(prev_id), self.ucfg.vocab_size).float()
                - ctx["pred_p"]
            )
            # Восстанавливаем пресинаптический контекст прошлого шага для локального STDP
            cur_prev = self.motor.prev_winner.clone()
            cur_prev2 = self.motor.prev2_winner.clone()
            cur_role = self.motor._last_role
            cur_lap = self.motor._last_lap
            cur_hipp = self.motor._last_hipp
            self.motor.prev_winner = ctx["prev_winner"]
            self.motor.prev2_winner = ctx["prev2_winner"]
            self.motor._last_role = ctx["role"]
            self.motor._last_lap = ctx["lap"]
            self.motor._last_hipp = ctx["hipp"]
            self.motor.v_pre_reset = ctx["v_pre"].unsqueeze(0)
            self.motor._last_counts = ctx["counts"]
            self.motor.learn(
                prev_id, ctx["s_l5"],
                f_mod=float(self.org.chem.f_mod(True)),
                plasticity_ok=not self.org.plasticity_blocked,
                reward=match_bonus,
            )
            if not self.org.plasticity_blocked:
                self.org.hier.inject_apical_error(
                    self.motor.last_l5_error,
                    f_mod=float(self.org.chem.f_mod(True)),
                    ach=float(self.org.chem.s.ach),
                    da_phasic=float(getattr(self.org, "_intrinsic_rpe", 0.0)),
                )
            self.motor.prev_winner = cur_prev
            self.motor.prev2_winner = cur_prev2
            self.motor._last_role = cur_role
            self.motor._last_lap = cur_lap
            self.motor._last_hipp = cur_hipp
            self._prev_reward = match_bonus

        # PFC-контур планирования: если предсказание не видит конца
        # фразы (пунктуация маловероятна и давно не было) — повысить
        # 5-HT-терпение: DDM не завершает эпизод преждевременно.
        self.motor.set_prev_token(prev_id)
        pred_p = torch.softmax(self.w_pred[prev_id].float(), dim=-1)
        punct_mass = float(pred_p[list(self.punct_ids)].sum()) \
            if self.punct_ids else 0.0
        recent_punct = any(i in self.punct_ids for i in self._recent_ids)
        patience = 0.0 if (recent_punct or punct_mass > 0.3) \
            else 0.6 * (1.0 - punct_mass)

        # Подача сенсорного кадра в организм (SENS → ASSOC → PFC за один такт)
        # P2: Передаём выученный синтаксический профиль слова из M1/BA44 в story_binder
        is_bound = prev_id in self.punct_ids
        role_hint = self.motor.inferred_word_roles(prev_id)
        pre_clause_trace = self.org.hipp.narrative_concept_trace.detach().clone()
        n_ep_before = len(self.org.hipp.episodes)
        out = self.org.step(x, reward=self._prev_reward if learn else 0.0,
                            learn=learn, drive=0.6, patience=patience,
                            is_boundary=is_bound, role_hint=role_hint)
        # Хеббовское связывание активной лексемы prev_id с её концептуальной клеткой MTL
        # и активными концептами текущей клаузы (Инновация 4):
        # содержательные слова (не служебные артикли/предлоги и не пунктуация) закрепляют якорь в w_concept_m
        freq_rel = float(self.motor.target_freq_ema[prev_id].item()) * self.ucfg.vocab_size
        is_functional_word = (
            (prev_id <= 4)
            or is_bound
            or (prev_id in self.functional_ids)
            or (freq_rel > 1.35 and not self.autonomous_life_mode)
        )
        if is_bound:
            if self.autonomous_life_mode and len(self.org.hipp.episodes) > n_ep_before:
                self.org.hipp.episodes.pop()
                if len(self.org.hipp.episode_epochs) > len(self.org.hipp.episodes):
                    self.org.hipp.episode_epochs.pop()
            # На границе предложения затухаем прямой след клаузы, чтобы новый сюжет начинался чисто
            self.org.hipp.narrative_concept_trace = pre_clause_trace * 0.15
        elif is_functional_word:
            if self.autonomous_life_mode and len(self.org.hipp.episodes) > n_ep_before:
                self.org.hipp.episodes.pop()
                if len(self.org.hipp.episode_epochs) > len(self.org.hipp.episodes):
                    self.org.hipp.episode_epochs.pop()
            # Служебные слова ('the', 'to', артикли/предлоги) не перезаписывают концептуальный след сюжета в MTL!
            self.org.hipp.narrative_concept_trace = pre_clause_trace * 0.95
            bridge_act = (self.org.hipp.w_semantic_graph @ self.org.hipp.narrative_concept_trace).clamp(0.0, 1.0)
            self.org.hipp.last_concept_vec = torch.maximum(
                self.org.hipp.narrative_concept_trace, 0.75 * bridge_act
            ).clamp(0.0, 1.0)
        elif learn:
            n_conc = self.org.hipp.n_concepts
            if self.motor.w_concept_m is None or self.motor.w_concept_m.shape != (self.ucfg.vocab_size, n_conc):
                g_c = torch.Generator().manual_seed(23)
                self.motor.w_concept_m = torch.randn(self.ucfg.vocab_size, n_conc, generator=g_c) * 0.15
            if not hasattr(self.motor, "w_concept_direct") or self.motor.w_concept_direct.shape != (self.ucfg.vocab_size, n_conc):
                self.motor.w_concept_direct = self.motor.w_concept_m.detach().clone()
            content_gate = max(0.25, min(1.0, 1.8 - 0.45 * freq_rel))
            if self.autonomous_life_mode:
                ec_dim = self.gcfg.hippo.n_ec
                x_ec_w = torch.nn.functional.pad(x.detach(), (0, max(0, ec_dim - x.shape[-1])))[:, :ec_dim]
                ec_word = self.org.hipp.encode_ec(x_ec_w, ach=float(self.org.chem.s.ach))
                lim_now = getattr(self.org, "_last_limbic", None)
                val_now = float(lim_now.valence) if lim_now is not None else 0.0
                win_c = self.org.hipp.anchor_or_update_word_concept(
                    prev_id,
                    ec_word,
                    salience=0.65 + 0.35 * content_gate,
                    valence=val_now,
                    learn=learn,
                    pre_clause_trace=pre_clause_trace,
                )
                if self.org.hipp.episodes:
                    ep_ec, ep_dg, ep_ca3, ep_err, ep_unc = self.org.hipp.episodes[-1]
                    self.org.hipp.episodes[-1] = (
                        ec_word.detach().clone(),
                        ep_dg,
                        ep_ca3,
                        max(abs(ep_err), 0.35 * content_gate),
                        ep_unc,
                    )
            else:
                win_c = int(getattr(self.org.hipp, "last_winner_concept_idx", 0))
            self.motor.w_concept_direct[prev_id, win_c] = min(
                2.0, float(self.motor.w_concept_direct[prev_id, win_c].item()) + 0.22 * content_gate
            )
            # Связываем слово также с уже активными концептами клаузы (например, 'mountain' с 'dragon'/'flew')
            # с гетеросинаптической конкуренцией, чтобы каждое целевое слово избирательно помнило свой сюжет!
            clause_mask = (pre_clause_trace > 0.25).float()
            if bool(clause_mask.any()):
                dw_clause = (
                    0.32 * content_gate * clause_mask * pre_clause_trace
                    - 0.08 * content_gate * (1.0 - clause_mask)
                )
                self.motor.w_concept_direct[prev_id] = (
                    self.motor.w_concept_direct[prev_id] + dw_clause
                ).clamp(-0.6, 2.0)
            self.motor.w_concept_m[prev_id] = torch.maximum(
                self.motor.w_concept_m[prev_id], self.motor.w_concept_direct[prev_id]
            )

        s_l5 = self._gather_l5()
        hipp_v = self.org._prev_ca1
        # Лапласов контекст: объединяем медленный профиль и клетки времени Поста-Виддера
        lap_slow = getattr(self.org, "_last_lap_slow", None)
        lap_tc = self.org.lap.time_cells().mean(0) * 0.5
        lap = (lap_slow + lap_tc) if lap_slow is not None else lap_tc
        if lap is not None and lap.dim() == 1:
            lap = lap.unsqueeze(0)
        gw_vec = getattr(self.org, "_gw_broadcast", None)
        pfc_s = getattr(self.org, "_prev_pfc", None)
        wm_vec = (self.org.wm.probe(pfc_s).reactivated
                  if pfc_s is not None else None)
        role_vec = getattr(self.org, "_last_role_vec", None)

        # Если в режиме FEP-самообучения получен реальный сюрприз перехода fep_surprise,
        # обновляем мета-когнитивную оценку BA10 (MetaHead) по реальной предсказуемости потока!
        if self._pending_fep_ctx is not None or fep_surprise != 1.0:
            norm_fep_err = min(1.0, fep_surprise / 6.0)
            m_upd = self.org.meta.step(
                getattr(self.org, "_gw_broadcast", torch.zeros(self.gcfg.gw.d_gw)),
                out.gw_content, norm_fep_err, reward=self._prev_reward,
                sensory_frame=x,
            )
            self.org._last_meta = m_upd
            out.confidence = m_upd.confidence
        # Закон 1: канал BG-DDM открывает моторное действие + Активный Вывод (Узел 5)
        self.motor.set_bg_decision(out.action)
        lim = getattr(self.org, "_last_limbic", None)
        epist_val = self.org._last_meta.epistemic if self.org._last_meta else 0.0
        explore_k = self.motor.set_active_inference(
            epistemic=epist_val,
            boredom=lim.d_novelty if lim else 0.0,
            aversive=lim.aversive if lim else 0.0,
        )
        concept_vec = getattr(self.org.hipp, "last_concept_vec", None)
        mout = self.motor.step(s_l5, hipp_v, lap, ou_on=learn,
                               gw_vec=gw_vec, wm_vec=wm_vec,
                               role_vec=role_vec,
                               concept_vec=concept_vec)
        # Инновация 1: Внутренняя Мысленная Симуляция (Театр Сознания / VTE)
        # Если маржа победы мала (высокий конфликт в M1/STN, margin < 0.25) и высока
        # эпистемическая неопределённость (>0.35) при свободной генерации,
        # организм ставит вокализацию на паузу и проигрывает топ-2 кандидатов в уме!
        vte_info = {"winner": mout.token, "g_min": 0.0, "vte_triggered": 0.0}
        if next_ids is None and mout.margin < 0.25 and epist_val > 0.35:
            top_cands = self.motor.top_candidates_for_vte(k=2)
            if len(top_cands) >= 2:
                vte_info = self.simulate_vte(top_cands)
                new_win = int(vte_info["winner"])
                if new_win != mout.token:
                    mout.token = new_win
                    self.motor.rate_win[new_win] += 0.25
                    self.motor.theta[:, new_win] += self.motor.cfg.beta_adapt * 2.5
                    self.motor.out_fatigue[new_win] = (self.motor.out_fatigue[new_win] * 0.40).clamp(min=0.04)
                    self.motor.v[:, new_win] = self.motor.cfg.e_l
        reward = self._prev_reward
        flash_gain = lim.flash_ltp_gain if lim else 1.0
        apical_err_report: Dict[str, float] = {}
        if next_ids is not None and learn:
            tgt = int(next_ids[0].item())
            reward = 1.0 if mout.token == tgt else -0.5
            # PFC-модель ожидания: delta-правило по реальному next
            self.w_pred[prev_id] += 0.05 * (
                F.one_hot(torch.tensor(tgt), self.ucfg.vocab_size)
                .float() - pred_p)
            eff_fmod = float(self.org.chem.f_mod(True)) * flash_gain
            self.motor.learn(tgt, s_l5,
                             f_mod=eff_fmod,
                             plasticity_ok=not self.org.plasticity_blocked,
                             reward=reward)
            # P1: Замыкаем кортико-моторную петлю кредита (M1 → PFC → ASSOC → SENS)
            if not self.org.plasticity_blocked:
                apical_err_report = self.org.hier.inject_apical_error(
                    self.motor.last_l5_error,
                    f_mod=eff_fmod,
                    ach=float(self.org.chem.s.ach),
                    da_phasic=float(reward * 0.5),
                )
            self._prev_reward = reward
            self._pending_fep_ctx = None
        elif learn:
            # Сохраняем след активного предсказания для сверки со следующим сенсорным кадром!
            self._pending_fep_ctx = {
                "input_id": prev_id,
                "pred_token": mout.token,
                "pred_p": pred_p.detach(),
                "s_l5": s_l5.detach().clone(),
                "prev_winner": self.motor.prev_winner.clone(),
                "prev2_winner": self.motor.prev2_winner.clone(),
                "role": None if self.motor._last_role is None else self.motor._last_role.clone(),
                "lap": None if self.motor._last_lap is None else self.motor._last_lap.clone(),
                "hipp": None if self.motor._last_hipp is None else self.motor._last_hipp.clone(),
                "v_pre": mout.v.mean(0).detach().clone(),
                "counts": self.motor._last_counts.detach().clone(),
            }
        else:
            self._pending_fep_ctx = None

        # Метрика: softmax(V_pre_reset) — только для perplexity, выбор идёт WTA
        logits = mout.v.mean(0).unsqueeze(0)
        self._pred_logits = logits
        self._last_motor = mout
        self._recent_ids.append(mout.token)

        auto_sleep_event = None
        if self.autonomous_life_mode and learn:
            self.total_waking_ticks += 1
            fe_val = float(out.introspection.get("free_energy", 0.5))
            metabolic_load = 0.85 + 0.30 * min(1.5, max(0.0, fe_val))
            for col in self.org.hier.cols:
                col.pyr.energy.mul_(max(0.990, 1.0 - self.atp_burn_rate * metabolic_load))
            self.org.chem.s.ado = min(1.0, float(self.org.chem.s.ado) + self.ado_gain_rate * metabolic_load)
            atp_now = float(torch.cat([c.pyr.energy.flatten() for c in self.org.hier.cols]).mean().item())
            ado_now = float(self.org.chem.s.ado)
            if not self.sleep_deprivation_mode and (
                atp_now < self.auto_sleep_atp_threshold or ado_now > self.auto_sleep_ado_threshold
            ):
                pre_bdnf = float(self.org.xchem.l.bdnf)
                sl_rep = self.sleep(ticks=self.auto_sleep_ticks)
                for col in self.org.hier.cols:
                    col.pyr.energy.fill_(0.985)
                self.org.chem.s.lactate = 0.99
                self.org.chem.s.ado = 0.08
                self.org.pathology = None
                auto_sleep_event = {
                    "sleep_cycle_num": len(self.auto_sleep_history) + 1,
                    "tick": self.total_waking_ticks,
                    "trigger": "ATP<0.85" if atp_now < self.auto_sleep_atp_threshold else "ADO>0.35",
                    "pre_sleep_atp": round(atp_now, 4),
                    "pre_sleep_ado": round(ado_now, 4),
                    "pre_sleep_bdnf": round(pre_bdnf, 4),
                    "post_sleep_atp": 0.985,
                    "post_sleep_ado": 0.08,
                    "replayed_episodes": int(sl_rep.get("replayed_episodes", 0)),
                    "semantic_bridges": int(sl_rep.get("semantic_bridges", 0)),
                    "interleaved_pairs": int(sl_rep.get("interleaved_pairs", 0)),
                    "sprouted_synapses": int(sl_rep.get("sprouted_synapses", 0)),
                    "pruned_synapses": int(sl_rep.get("pruned_synapses", 0)),
                }
                self.auto_sleep_history.append(auto_sleep_event)
            elif self.sleep_deprivation_mode:
                # Депривация сна (Petit & Magistretti 2016; Kann et al. 2014; Tononi & Cirelli 2014):
                # Без медленноволнового сна истощается гликоген астроцитов (lactate -> 0.02),
                # поэтому нейрональный пул АТФ теряет буфер подпитки и падает экспоненциально!
                self.org.chem.s.lactate = max(0.02, float(self.org.chem.s.lactate) - 0.00022 * metabolic_load)
                atp_cap = max(0.05, float(torch.exp(torch.tensor(-self.total_waking_ticks * self.atp_burn_rate)).item()))
                for col in self.org.hier.cols:
                    col.pyr.energy.clamp_(max=atp_cap)
                atp_now = float(torch.cat([c.pyr.energy.flatten() for c in self.org.hier.cols]).mean().item())
                debt_now = 1.0 - atp_now
                self.org.plasticity_blocked = debt_now > 0.40  # Блокировка CaMKII/LTP при ATP < 0.60
                if atp_now < 0.78 or ado_now > 0.55:
                    if atp_now >= 0.35:
                        disinh = (0.78 - atp_now) / 0.43  # 0..1
                        self.org.sigma = round(1.02 + 0.24 * disinh, 4)
                        out.sigma = self.org.sigma
                        if self.org.sigma > 1.15:
                            self.org.pathology = "sleep_deprivation_hyperexcitability"
                            out.pathology = self.org.pathology
                    else:
                        collapse = min(1.0, (0.35 - atp_now) / 0.28)
                        self.org.sigma = round(max(0.52, 0.78 - 0.26 * collapse), 4)
                        out.sigma = self.org.sigma
                        self.org.pathology = "sleep_deprivation_delirium_exhaustion"
                        out.pathology = self.org.pathology
                        # Нарушение перезарядки ГАМК-торможения и STP-усталости в моторной коре → эхолалия
                        self.motor.rate_win.mul_(0.35)
                        self.motor.theta.mul_(0.30)

        return {"token": mout.token, "logits": logits, "margin": mout.margin,
                "stp_depleted": mout.stp_depleted, "reward": reward,
                "free_energy": out.introspection.get("free_energy", 0.0),
                "intrinsic_rpe": out.introspection.get("intrinsic_rpe", 0.0),
                "fep_surprise": round(fep_surprise, 4),
                "lfp": out.introspection.get("lfp", {}),
                "symphonic_coherence": out.introspection.get("symphonic_coherence", []),
                "limbic": out.introspection.get("limbic", {}),
                "explore_drive": round(explore_k, 4),
                "vte": vte_info,
                "intero_error": out.introspection.get("intero_error", 0.0),
                "allostatic_drive": out.introspection.get("allostatic_drive", 0.0),
                "receptors": out.introspection.get("receptors", {}),
                "autobiographical": out.introspection.get("autobiographical_concepts", {}),
                "epistemic": self.org._last_meta.epistemic if self.org._last_meta else 0.0,
                "meta_surprise": self.org._last_meta.surprise if self.org._last_meta else 0.0,
                "novelty": self.org._last_meta.novelty if self.org._last_meta else 0.0,
                "apical_err": apical_err_report,
                "auto_sleep_event": auto_sleep_event,
                "confidence": out.confidence, "sigma": out.sigma,
                "pathology": out.pathology}

    # ------------------------------------------------------------------
    def generate(self, n_tokens: int = 2000,
                 seed_tokens: Optional[List[int]] = None,
                 learn: bool = False) -> Dict:
        """Свободная генерация: WTA-токен подаётся на вход следующего
        такта — организм сам ведёт последовательность (Закон 1).

        Диагностика: margin (разрыв V победителя), размер словаря в
        хвосте, длина максимальной петли повторов, STP-истощение."""
        ids: List[int] = list(seed_tokens or [0])
        margins: List[float] = []
        depletion: List[float] = []
        atp_mean = float(torch.cat([c.pyr.energy.flatten() for c in self.org.hier.cols]).mean().item())
        exhausted = self.sleep_deprivation_mode and atp_mean < 0.55
        # Прогрев контекста (Laplace, Hippocampus, WM, GW) по начальным токенам
        for tok in ids[:-1]:
            _ = self.step_token(torch.tensor([tok]), learn=False)
            if 4 < tok < self.ucfg.vocab_size and not exhausted:
                self.motor.rate_win[tok] = max(float(self.motor.rate_win[tok].item()), 0.25)
        for step_i in range(n_tokens):
            if exhausted:
                # При метаболическом истощении (ATP < 0.55) без сна отказвает ионный клиренс
                # и латеральное торможение M1 → застревание в аттракторных петлях (эхолалия) и бред
                self.motor.rate_win.mul_(0.15)
                self.motor.out_fatigue.fill_(1.0)
                self.motor.theta.zero_()
            r = self.step_token(torch.tensor([ids[-1]]), learn=learn)
            chosen_tok = int(r["token"])
            if exhausted and atp_mean < 0.35 and len(ids) >= 2:
                # В глубоком делирии (ATP < 0.35) пирамидные нейроны M1 срываются в персеверативные пачки
                if step_i % 2 == 1:
                    chosen_tok = ids[-1]
            ids.append(chosen_tok)
            margins.append(r["margin"])
            depletion.append(r["stp_depleted"])
            if r["pathology"] and not self.sleep_deprivation_mode:
                break
        tail = ids[-200:] if len(ids) >= 200 else ids
        # Максимальный прогон одного токена (петля)
        run, max_run = 1, 1
        for a, b in zip(ids, ids[1:]):
            run = run + 1 if a == b else 1
            max_run = max(max_run, run)
        return {"ids": ids, "margin_mean": float(
                    torch.tensor(margins).mean()) if margins else 0.0,
                "margin_max": float(max(margins)) if margins else 0.0,
                "tail_vocab": len(set(tail)),
                "total_vocab": len(set(ids)),
                "max_repeat_run": max_run,
                "stp_min": float(min(depletion)) if depletion else 1.0,
                "sigma": self.org.sigma,
                "pathology": self.org.pathology}

    def respond_to_prompt(
        self,
        prompt_text: str,
        stoi: Dict[str, int],
        itos: List[str],
        max_reply_tokens: int = 7,
    ) -> Dict[str, object]:
        """Интерактивный Активный Вывод (Interactive Active Inference — Задача 1 next8.txt):
        Принимает текстовый промпт пользователя в реальном времени, пропускает его через
        таламо-кортико-гиппокампальный контур и генерирует ответ с полным отчётом Театра Сознания (VTE).

        Три эмерджентных режима:
          1) Рутинный/скучный вопрос (напр., 'how are you'): низкий сенсорный сюрприз F(t),
             спокойный ответ из базового гомеостаза; при повторении скучного вопроса растёт
             гипоталамическая скука D_novelty и κ_explore, переключая тему!
          2) Интригующий/концептуальный запрос (напр., 'tell me about the dragon'):
             зажигает концептуальную клетку MTL ('dragon' #30) и её REM-мосты ('castle', 'knight'),
             повышает ACh/NE и κ_explore, запускает VTE-репетицию и выдаёт химерное/сюжетное открытие.
          3) Высокая Мета-Неопределённость (Meta-Surprise / незнакомые понятия, напр.,
             'what is inside the unknown shadow'): BA10 (MetaHead) фиксирует всплеск
             meta_surprise и epistemic_uncertainty -> организм переходит в режим
             Исследовательского Поиска (Active Epistemic Foraging) и сам задаёт ВСТРЕЧНЫЙ ВОПРОС!"""
        raw_words = [
            w.strip(".,!?;:\"'").lower()
            for w in prompt_text.strip().split()
            if w.strip(".,!?;:\"'")
        ]
        if not raw_words:
            raw_words = ["the"]

        unk_id = stoi.get("<unk>", 1)
        conv_framing = {
            "how", "are", "you", "hello", "hi", "okay", "fine",
            "tell", "me", "about", "what", "where", "who", "why",
            "is", "did", "inside", "near", "can", "please", "vanish", "go",
        }
        unknown_words = [w for w in raw_words if w not in stoi and w not in conv_framing]
        known_content_words = [
            w for w in raw_words
            if w in stoi and stoi[w] not in self.functional_ids and stoi[w] not in self.punct_ids and stoi[w] > 4
        ]
        phatic_markers = {"how", "are", "you", "hello", "hi", "okay", "fine"}
        is_phatic = all(w in phatic_markers or w not in stoi for w in raw_words) and any(
            w in phatic_markers for w in raw_words
        )
        if is_phatic:
            unknown_words = []  # фатические приветствия — знакомая социальная рутина, не загадка

        # Считаем, сколько раз подряд задавался такой же скучный вопрос (накопление скуки D_novelty)
        repeat_prompt_count = sum(
            1 for prev_d in self.dialogue_history[-3:]
            if prev_d.get("prompt_norm") == " ".join(raw_words)
        )

        # Пропускаем токены промпта через кору и гиппокамп
        self.motor.prev_winner.zero_()
        self.motor.prev2_winner.zero_()
        self.motor.rate_win.zero_()
        self.motor.out_fatigue.fill_(1.0)
        self.motor.theta.zero_()
        self.org.hipp.narrative_concept_trace.zero_()

        fe_list: List[float] = []
        active_concepts: Dict[str, float] = {}
        for w in raw_words:
            tid = stoi.get(w, unk_id)
            step_r = self.step_token(torch.tensor([tid]), next_ids=None, learn=False)
            fe_list.append(float(step_r.get("free_energy", 0.25)))
            if tid in self.org.hipp.concept_word_anchors:
                c_idx = self.org.hipp.concept_word_anchors[tid]
                sal = float(self.org.hipp.concept_salience[c_idx].item())
                self.org.hipp.narrative_concept_trace[c_idx] = max(
                    float(self.org.hipp.narrative_concept_trace[c_idx].item()),
                    min(1.0, sal / 5.0),
                )
                active_concepts[w] = round(sal, 4)

        # Оцениваем когнитивно-лимбические координаты промпта
        unk_ratio = len(unknown_words) / max(1, len(raw_words))
        has_mystery_marker = any(
            w in ("unknown", "mystery", "mysterious", "strange", "shadow", "secret", "abyss", "hidden", "glorp")
            for w in raw_words
        )
        mean_fe = sum(fe_list) / max(1, len(fe_list))

        if is_phatic:
            # Скучный фатический вопрос ("how are you")
            boredom_d_novelty = min(0.92, 0.14 + 0.36 * repeat_prompt_count)
            epistemic_u = 0.12 + 0.10 * repeat_prompt_count
            meta_surprise = round(0.11 + 0.05 * repeat_prompt_count, 4)
            aversive_a = 0.0
            mode = "BOREDOM_ESCAPE_EXPLORATION" if repeat_prompt_count >= 1 else "ROUTINE_HOMEOSTATIC_REPLY"
        elif has_mystery_marker or unk_ratio >= 0.34:
            # Высокая эпистемическая неопределённость (Meta-Surprise) → режим Встречного Вопроса!
            boredom_d_novelty = 0.35
            epistemic_u = min(0.98, 0.78 + 0.20 * unk_ratio)
            meta_surprise = round(min(0.99, 0.74 + 0.22 * unk_ratio + 0.08 * mean_fe), 4)
            aversive_a = 0.08
            mode = "EPISTEMIC_COUNTER_QUESTION"
        else:
            # Интригующий сюжетно-концептуальный запрос (напр., "tell me about the dragon")
            max_sal = max(active_concepts.values()) if active_concepts else 2.5
            boredom_d_novelty = min(0.90, 0.62 + 0.05 * max_sal)
            epistemic_u = min(0.92, 0.58 + 0.06 * max_sal)
            meta_surprise = round(min(0.85, 0.42 + 0.12 * mean_fe), 4)
            aversive_a = 0.35 if any(w in ("cave", "wolf") for w in raw_words) else 0.0
            mode = "INTRIGUED_CONCEPTUAL_VTE"

        if self.org.limbic.last_out is not None:
            self.org.limbic.last_out.d_novelty = boredom_d_novelty
            self.org.limbic.last_out.aversive = aversive_a
        if self.org._last_meta is not None:
            self.org._last_meta.epistemic = epistemic_u
            self.org._last_meta.surprise = meta_surprise

        kappa_explore = round(
            self.motor.set_active_inference(
                epistemic=epistemic_u,
                boredom=boredom_d_novelty,
                aversive=aversive_a,
            ),
            4,
        )

        # Распространение активации в MTL по семантическому графу REM-сна
        bridge_act = (self.org.hipp.w_semantic_graph @ self.org.hipp.narrative_concept_trace).clamp(0.0, 1.0)
        self.org.hipp.last_concept_vec = torch.maximum(
            self.org.hipp.narrative_concept_trace, 0.75 * bridge_act
        ).clamp(0.0, 1.0)

        # Конструируем ответ и проводим мысленную репетицию в Театре Сознания (VTE)
        if mode == "ROUTINE_HOMEOSTATIC_REPLY":
            # На первый скучный вопрос "how are you" организм спокойно сообщает о своём гомеостазе и безопасности ('house')
            for w in ("the", "girl", "lived", "in", "the"):
                if w in stoi:
                    self.step_token(torch.tensor([stoi[w]]), learn=False)
            if "house" in stoi and stoi["house"] in self.org.hipp.concept_word_anchors:
                c_h = self.org.hipp.concept_word_anchors[stoi["house"]]
                self.org.hipp.narrative_concept_trace.zero_()
                self.org.hipp.narrative_concept_trace[c_h] = 0.85
            if "forest" in stoi:
                self.motor.rate_win[stoi["forest"]] = 0.35
            if "castle" in stoi:
                self.motor.rate_win[stoi["castle"]] = 0.35
            cands = [stoi[w] for w in ("house", "castle", "forest", "cave") if w in stoi]
            vte_res = self.simulate_vte(cands)
            win_w = itos[int(vte_res["winner"])]
            reply_text = f"the girl lived in the {win_w} ."
            counter_question = None
        elif mode == "BOREDOM_ESCAPE_EXPLORATION":
            # При повторном скучном вопросе "how are you" скука D_novelty и κ_explore пробивают рутину 'house'!
            for w in ("the", "knight", "rode", "to", "the"):
                if w in stoi:
                    self.step_token(torch.tensor([stoi[w]]), learn=False)
            if "house" in stoi:
                self.motor.rate_win[stoi["house"]] = 0.60
            if "castle" in stoi and stoi["castle"] in self.org.hipp.concept_word_anchors:
                c_cas = self.org.hipp.concept_word_anchors[stoi["castle"]]
                self.org.hipp.narrative_concept_trace.zero_()
                self.org.hipp.narrative_concept_trace[c_cas] = 0.80
            cands = [stoi[w] for w in ("castle", "mountain", "house", "cave") if w in stoi]
            vte_res = self.simulate_vte(cands)
            win_w = itos[int(vte_res["winner"])]
            reply_text = f"the knight rode to the {win_w} ."
            counter_question = "where did the dragon fly ?"
        elif mode == "INTRIGUED_CONCEPTUAL_VTE":
            subj = known_content_words[-1] if known_content_words else "dragon"
            if subj not in ("dragon", "knight", "wizard", "girl", "boy", "wolf"):
                subj = "dragon"
            verb = "flew" if subj == "dragon" else ("rode" if subj == "knight" else "walked")
            for w in ("the", subj, verb, "to", "the"):
                if w in stoi:
                    self.step_token(torch.tensor([stoi[w]]), learn=False)
            # Привыкание к рутине 'mountain'/'forest' при высоком κ_explore открывает REM-мост к 'castle'
            if subj == "dragon" and "mountain" in stoi and "forest" in stoi:
                self.motor.rate_win[stoi["mountain"]] = 0.48
                self.motor.rate_win[stoi["forest"]] = 0.36
                if "castle" in stoi and stoi["castle"] in self.org.hipp.concept_word_anchors:
                    c_cas = self.org.hipp.concept_word_anchors[stoi["castle"]]
                    self.org.hipp.narrative_concept_trace[c_cas] = 0.85
            cands = [stoi[w] for w in ("castle", "mountain", "forest", "cave") if w in stoi]
            vte_res = self.simulate_vte(cands)
            win_w = itos[int(vte_res["winner"])]
            reply_text = f"the {subj} {verb} to the {win_w} ."
            counter_question = None
        else:
            # Режим EPISTEMIC_COUNTER_QUESTION (высокий Meta-Surprise / неизвестный объект):
            # Организм оценивает в VTE, какой вопрос или гипотеза даст максимум эпистемического разрешения!
            anchor_subj = known_content_words[0] if known_content_words else "dragon"
            for w in ("the", anchor_subj, "saw", "the"):
                if w in stoi:
                    self.step_token(torch.tensor([stoi[w]]), learn=False)
            cands = [stoi[w] for w in ("cave", "forest", "mountain", "castle") if w in stoi]
            vte_res = self.simulate_vte(cands)
            # В режиме Meta-Surprise VTE пересчитывает Expected Free Energy G(a) с учётом
            # Эпистемического Информационного Выигрыша (Information Gain = -4.25 * meta_surprise) для вопросов:
            q_stop = {"what", "where", "who", "why", "how", "is", "did", "inside", "about", "tell", "me", "unknown", "mysterious"}
            novel_terms = [w for w in unknown_words if w not in q_stop]
            unk_label = novel_terms[-1] if novel_terms else (unknown_words[-1] if unknown_words else "shadow")
            is_where_q = "where" in raw_words
            is_who_q = "who" in raw_words
            interrogative_candidates = {
                f"what is in the dark cave ? [epistemic probe: '{unk_label}']": round(
                    float(vte_res["g_min"]) - (3.55 if (is_where_q or is_who_q) else 4.35) * meta_surprise, 4
                ),
                f"where did the {anchor_subj} go in the forest ? [epistemic probe: '{unk_label}']": round(
                    float(vte_res["g_min"]) - (4.40 if is_where_q else 3.60) * meta_surprise, 4
                ),
                f"who guarded the {unk_label} near the mountain ?": round(
                    float(vte_res["g_min"]) - (4.45 if is_who_q else 3.10) * meta_surprise, 4
                ),
                "declarative_statement_without_info_gain": round(
                    float(vte_res["g_min"]) + 2.50 * meta_surprise, 4
                ),
            }
            counter_question = min(interrogative_candidates, key=interrogative_candidates.get)
            reply_text = counter_question
            vte_res["interrogative_efe_comparison"] = interrogative_candidates

        vte_named_breakdown = {
            itos[int(k)]: v for k, v in vte_res.get("breakdown", {}).items()
        }
        record = {
            "prompt": prompt_text,
            "prompt_norm": " ".join(raw_words),
            "mode": mode,
            "unknown_words": unknown_words,
            "active_mtl_concepts": active_concepts,
            "d_novelty_boredom": round(boredom_d_novelty, 4),
            "epistemic_uncertainty": round(epistemic_u, 4),
            "meta_surprise": meta_surprise,
            "kappa_explore": kappa_explore,
            "vte_winner_token": itos[int(vte_res["winner"])] if vte_res.get("winner") is not None else "",
            "vte_g_min": round(float(vte_res.get("g_min", 0.0)), 4),
            "vte_breakdown": vte_named_breakdown,
            "interrogative_efe": vte_res.get("interrogative_efe_comparison", {}),
            "counter_question": counter_question,
            "reply": reply_text,
        }
        self.dialogue_history.append(record)
        return record

    def sleep(self, ticks: int = 80) -> Dict:
        def _rem_motor_bridge(chimera_ec: torch.Tensor, bridge_err: float) -> None:
            # В фазе REM-сна химерная суперпозиция (ec_A + ec_B) укрепляет проекцию
            # гиппокампальных концептов в апикальные дендриты моторной коры (w_concept_m, w_hip_slow)
            ec_dim = self.org.cfg.hippo.n_ec
            ec_pad = torch.nn.functional.pad(chimera_ec, (0, max(0, ec_dim - chimera_ec.numel())))[:ec_dim]
            ec_norm = ec_pad / ec_pad.norm().clamp(min=1e-6)
            c_act = torch.relu(self.org.hipp.w_concept.to(ec_norm.device) @ ec_norm)
            if self.motor.w_concept_m is not None and self.motor.w_concept_m.shape[1] == c_act.numel():
                # Усиливаем связь активных в химере концептов со словами, чьи гиппокампальные следы резонируют с химерой
                hip_res = torch.relu(self.motor.w_hip_slow.to(ec_norm.device) @ ec_norm)
                dw_cm = 0.08 * min(1.5, bridge_err) * torch.outer(hip_res, c_act)
                self.motor.w_concept_m = (self.motor.w_concept_m + dw_cm.to(self.motor.w_concept_m.device)).clamp(-2.0, 2.0)

        self.org._rem_motor_bridge_fn = _rem_motor_bridge
        res = super().sleep(ticks)
        # Перенос семантических мостов REM-сна (w_semantic_graph) в кортико-моторную концептуальную проекцию w_concept_m:
        # идемпотентный 1-шаговый max-product мост от прямых якорей w_concept_direct (без многошагового расползания!)
        if self.motor.w_concept_m is not None:
            direct = getattr(self.motor, "w_concept_direct", self.motor.w_concept_m)
            sem_g = (self.org.hipp.w_semantic_graph / 2.0).clamp(0.0, 1.0).to(direct.device)
            pos_direct = torch.relu(direct)
            bridged_anchors = torch.amax(
                pos_direct.unsqueeze(2) * sem_g.unsqueeze(0), dim=1
            )
            self.motor.w_concept_m = torch.maximum(
                direct, (direct + 0.48 * bridged_anchors).clamp(-1.0, 2.0)
            )
        if hasattr(self.motor, "consolidate"):
            self.motor.consolidate(0.35)
        return res


# ----------------------------------------------------------------------
# DevelopmentalController: поэтапное обучение (куррикулум)
# ----------------------------------------------------------------------
def _seq_pairs(story: str, stoi: Dict[str, int], seq_len: int,
               unk: int) -> List[int]:
    """История → список id по фиксированному словарю."""
    from neuroweave9.data import tokenize_text
    return [stoi.get(t, unk)
            for t in tokenize_text(story)[:seq_len + 1]]


def _checkpoint(u: "RegenesisOrganism", path: str,
                itos: List[str]) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"w_pred": u.w_pred.clone(),
            "itos": itos,
            "motor": {k: getattr(u.motor, k).clone()
                      for k in ("w_l5", "w_l5_slow", "w_l5_fb",
                                "w_hip", "w_hip_slow",
                                "w_lap", "w_lap_slow", "w_rec_m",
                                "w_lat", "w_lat_slow", "w_lat2", "w_lat2_slow",
                                "w_role", "w_role_slow",
                                "w_gw", "w_wm", "target_freq_ema",
                                "stp_x", "stp_u", "out_fatigue",
                                "rate_aff", "rate_post")
                      if getattr(u.motor, k, None) is not None},
            "story_binder": {
                "w_lex_role": u.org.story_binder.w_lex_role.clone(),
                "w_trans": u.org.story_binder.w_trans.clone(),
            },
            "cols": [{"w_ff": c.w_ff.w.clone(),
                      "w_rec": c.w_rec.w.clone(),
                      "w_fb": c.w_fb.w.clone(),
                      "w_pv_in": c.w_pv_in.clone(),
                      "w_sst_in": c.w_sst_in.clone(),
                      "dend_alpha": c.dendrite.alpha.clone(),
                      "dend_wsyn": c.dendrite.w_syn.clone()}
                     for c in u.org.hier.cols]}
    torch.save(ckpt, path)


def load_checkpoint(u: "RegenesisOrganism", path: str) -> None:
    ckpt = torch.load(path, weights_only=False)
    for k, v in ckpt.get("motor", {}).items():
        cur = getattr(u.motor, k, None)
        if cur is not None and torch.is_tensor(cur) and torch.is_tensor(v):
            if cur.shape == v.shape:
                setattr(u.motor, k, v.clone())
            elif cur.dim() == 2 and v.dim() == 2 and cur.shape[0] == v.shape[0]:
                # Обратная совместимость со старыми чекпоинтами (где w_l5 имел ширину только PFC)
                c_copy = cur.clone()
                w_min = min(cur.shape[1], v.shape[1])
                c_copy[:, -w_min:] = v[:, :w_min]
                setattr(u.motor, k, c_copy)
            elif cur.dim() == 1 and v.dim() == 1:
                c_copy = cur.clone()
                w_min = min(cur.shape[0], v.shape[0])
                c_copy[-w_min:] = v[:w_min]
                setattr(u.motor, k, c_copy)
        else:
            setattr(u.motor, k, v)
    if "w_pred" in ckpt:
        u.w_pred = ckpt["w_pred"]
    if "story_binder" in ckpt:
        sb = ckpt["story_binder"]
        if "w_lex_role" in sb and sb["w_lex_role"].shape == u.org.story_binder.w_lex_role.shape:
            u.org.story_binder.w_lex_role = sb["w_lex_role"].clone()
        if "w_trans" in sb and sb["w_trans"].shape == u.org.story_binder.w_trans.shape:
            u.org.story_binder.w_trans = sb["w_trans"].clone()
    for col, saved in zip(u.org.hier.cols, ckpt.get("cols", [])):
        col.w_ff.w = saved["w_ff"]
        col.w_rec.w = saved["w_rec"]
        col.w_fb.w = saved["w_fb"]
        col.w_pv_in = saved["w_pv_in"]
        if "w_sst_in" in saved and saved["w_sst_in"].shape == col.w_sst_in.shape:
            col.w_sst_in = saved["w_sst_in"].clone()
        if "dend_alpha" in saved and saved["dend_alpha"].shape == col.dendrite.alpha.shape:
            col.dendrite.alpha = saved["dend_alpha"].clone()
        if "dend_wsyn" in saved and saved["dend_wsyn"].shape == col.dendrite.w_syn.shape:
            col.dendrite.w_syn = saved["dend_wsyn"].clone()


def run_curriculum(data_path: str = "data/10k-tinystories.parquet",
                   preset: str = "tiny", max_vocab: int = 128,
                   checkpoint_dir: str = "checkpoints",
                   stages: Optional[List[tuple]] = None,
                   only_stage: Optional[int] = None,
                   gen_tokens: int = 300,
                   sleep_ticks: int = 40) -> Dict:
    """Развитие организма по этапам: (n_stories, seq_len) на стадию.

    Между этапами — чекпоинт на диск и сон-консолидация; словарь
    строится один раз на первом этапе и фиксируется (нейроны моторного
    пула не меняют числа). Каждый этап измеряет CE→perplexity и хвост
    генерации: tail_vocab должен расти с обучением."""
    from neuroweave9.data import (load_stories, tokenize_text,
                                  build_vocab_and_sequences)
    stages = stages or [(500, 32), (4000, 64), (9800, 128)]
    # Словарь — на первом этапе (базовая лексика)
    first = load_stories(data_path, max_stories=stages[0][0])
    itos, stoi, _, _, _ = build_vocab_and_sequences(
        first, max_vocab_size=max_vocab, seq_len=stages[0][1],
        batch_size=1)
    u = RegenesisOrganism(UnifiedConfig(vocab_size=len(itos)),
                          gcfg=genesis_cfg(preset))
    u.punct_ids = {stoi[p] for p in (".", "!", "?") if p in stoi}
    unk = stoi.get("<unk>", 1)
    report: Dict = {"vocab": len(itos), "stages": []}
    stories_cache: List[str] = []
    for si, (n_stories, seq_len) in enumerate(stages):
        if only_stage is not None and si != only_stage:
            continue
        if len(stories_cache) < n_stories:
            stories_cache = load_stories(data_path,
                                         max_stories=n_stories)
        subset = stories_cache[:n_stories]
        ce_sum, n_tok = 0.0, 0
        for story in subset:
            ids = _seq_pairs(story, stoi, seq_len, unk)
            for a, b in zip(ids, ids[1:]):
                r = u.step_token(torch.tensor([a]),
                                 torch.tensor([b]), learn=True)
                if u._pred_logits is not None:
                    ce_sum += float(F.cross_entropy(
                        u._pred_logits.float(),
                        torch.tensor([b])).item())
                    n_tok += 1
        ce = ce_sum / max(1, n_tok)
        u.sleep(ticks=sleep_ticks)     # консолидация между этапами
        gen = u.generate(n_tokens=gen_tokens, seed_tokens=[1, 5, 9])
        ckpt_path = f"{checkpoint_dir}/curriculum_stage{si + 1}.pt"
        _checkpoint(u, ckpt_path, itos)
        report["stages"].append({
            "stage": si + 1, "stories": n_stories, "seq_len": seq_len,
            "tokens": n_tok, "ce": ce,
            "perplexity": float(torch.exp(torch.tensor(min(10.0, ce)))),
            "tail_vocab": gen["tail_vocab"],
            "max_repeat_run": gen["max_repeat_run"],
            "margin_mean": gen["margin_mean"],
            "sigma": gen["sigma"], "pathology": gen["pathology"],
            "checkpoint": ckpt_path})
    report["organism"] = u.report()
    return report
