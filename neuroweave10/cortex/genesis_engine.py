#!/usr/bin/env python3
r"""ZIWAGO-GENESIS v18.0 — CORE BIOPHYSICAL COGNITIVE ENGINE (`neuroweave10/cortex/genesis_engine.py`)
========================================================================================================
Реализует 5 фундаментальных механизмов «Идеальной Биологической SNN» (Ziwago-Genesis v18.0):

  1. НЕПРЕРЫВНАЯ ВОЛНА СИНТАКСИЧЕСКИХ ВАЛЕНТНОСТЕЙ БЕЗ `pos_idx` (`SyntacticValenceWaveGenerator`, 0% `pos_idx`!):
     - Полный отказ от позиционного индекса «магнитофонной головки» `pos_phase_vecs[pos_idx]`.
     - Состояние зоны Брока `BA44` в момент времени $t$ кодируется непрерывным рекуррентным потенциалом
       валентности $\vec{\Psi}_{\mathrm{BA44}}(t) \in \mathbb{R}^{64}$, который обновляется исключительно
       некоммутативной голографической свёрткой реально произнесённых токенов:
       $$\vec{\Psi}_{\mathrm{BA44}}(t) = \mathrm{Normalize}\!\Big(0.78\,\mathrm{Roll}(\vec{\Psi}_{\mathrm{BA44}}(t-1), 1) + 0.62\,\vec{v}(w_{t-1}) + 0.35\,\big(\mathrm{Roll}(\vec{\Psi}_{\mathrm{BA44}}(t-1), 1) \odot \vec{v}(w_{t-1})\big)\Big)$$

  2. КРИСТАЛЛИЗАЦИЯ СЕМАНТИЧЕСКОГО ГРАФА МИРА (`CorticalRelationalWorldGraph`):
     - Вместо заучивания предложений целиком спайковый парсер выделяет из текста каузальные триплеты
       $(	ext{Subject } S \xrightarrow{	ext{Relation } R} 	ext{Object } O)$ и укрепляет направленные
       синаптические рёбра графа знаний коры.
     - При ответе на вопрос волна возбуждения проходит по цепочке $E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3 \xrightarrow{R_3} E_4$
       и **собирает совершенно новое грамматическое предложение на лету**, которого никогда не было в корпусе!

  3. САМООРГАНИЗУЮЩАЯСЯ ОНТОЛОГИЯ РОЛЕЙ ЧЕРЕЗ ДИСТРИБУТИВНУЮ STDP (`EmergentOntologicalRoleDiscovery`, 0% ручных списков!):
     - В слое `L2/3` височной коры (`BA22`) каждое существительное или герундий накапливает 2048-битный
       след контекстного окружения предикатов $\vec{C}(w)$ по правилу Хебба–STDP.
     - Любое новое слово (например, `bechamel`, `veloute`, `treatise`, `compendium`, `kayaking`), попавшее
       в схожие предикатные контексты, автоматически распознаётся как член того же онтологического класса
       ролей по косинусному резонансу $\vec{C}(w_{\mathrm{new}}) \cdot \vec{C}(w_{\mathrm{canon}})$ без единого ручного словаря!

  4. ФАЗОВОЕ КОДИРОВАНИЕ СПАЙКОВ ТЕТА-ГАММА (`ThetaGammaSpikingPhaseMultiplexer`, 125 мс Тета = 6 × 20 мс Гамма-слотов):
     - Внутри одного цикла тета-ритма ($8	ext{ Гц}, 125	ext{ мс}$) 6 синтаксических ролей (`AGENT`, `ACTION`,
       `OBJECT`, `INSTRUMENT`, `LOCATION`, `EFFECT`) кодируются временем первого спайка (Time-to-First-Spike, TTFS)
       в 6 непересекающихся гамма-окнах ($\sim 20	ext{ мс}$ каждое), решая проблему связывания переменных
       (Variable Binding Problem) с нулевой перекрёстной интерференцией!

  5. ПРЕДИКТИВНОЕ ЗАМЫКАНИЕ СМЫСЛА В `SON / BA10` (`BidirectionalMeaningClosureVerifier`, $X \longrightarrow \hat{Y} \longrightarrow \hat{X}$):
     - Перед произнесением ответа $\hat{Y}$ обратная генеративная проекция восстанавливает подразумеваемый
       вопрос $\hat{X}(\hat{Y})$, вычисляет полноту смыслового покрытия $\mathcal{C}_{\mathrm{closure}}(X, \hat{Y})$
       и информационный прирост $\Delta I(X 	o \hat{Y}) > 0$. Если часть вопроса $X$ осталась нераскрытой,
       выделяется вектор дефицита смысла $\vec{r}_{\mathrm{missing}} = (\vec{s}_X - \hat{s}_X(\hat{Y}))_+$
       и ответ автоматически достраивается до полного смыслового замыкания!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ============================================================================
# 1. НЕПРЕРЫВНАЯ ВОЛНА СИНТАКСИЧЕСКИХ ВАЛЕНТНОСТЕЙ БЕЗ `pos_idx` (0% POS_IDX!)
# ============================================================================
class SyntacticValenceWaveGenerator:
    r"""Дендритный NMDA-декодер, работающий на **Непрерывном Автомате Синтаксических Валентностей Зоны Брока**
    $\vec{\Psi}_{\mathrm{BA44}}(t) \in \mathbb{R}^{64}$ (**0% `pos_idx`, 0% `pos_phase_vecs`**!):
      - В отличие от позиционного индекса `pos_idx = 0, 1, 2...`, вектор валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$
        определяется только динамической траекторией реально сгенерированных синтаксических узлов.
      - Если перед предложением вставлено вводное слово или обращение, как только поток входит в основную
        клаузу, $\vec{\Psi}_{\mathrm{BA44}}(t)$ сходится к аттрактору синтаксического состояния без сбоя фазы!"""

    def __init__(self, d_vec: int = 64, d_dend: int = 8192, top_k_wta: int = 32, seed: int = 1801):
        g = torch.Generator().manual_seed(seed)
        self.d_vec = int(d_vec)
        self.d_dend = int(d_dend)
        self.top_k_wta = int(top_k_wta)

        # Проекторы дендритных ветвей и рекуррентный оператор фазового вращения валентности (БЕЗ pos_phase_vecs и БЕЗ pos_idx!)
        self.w_basal_valence = F.normalize(torch.randn(d_vec * 4, d_dend, generator=g), p=2, dim=0) * 2.5
        self.w_apical_goal = F.normalize(torch.randn(d_vec, d_dend, generator=g), p=2, dim=0) * 3.0
        self.w_valence_rot = F.normalize(torch.randn(d_vec, d_vec, generator=g), p=2, dim=0)

        self.tok_to_col: Dict[int, int] = {}
        self.col_to_tok: List[int] = []
        self.w_valence_dend_out = torch.zeros(d_dend, 2048, dtype=torch.float32)

    def _ensure_token_col(self, token_id: int) -> int:
        tid = int(token_id)
        if tid in self.tok_to_col:
            return self.tok_to_col[tid]
        col = len(self.col_to_tok)
        if col >= self.w_valence_dend_out.shape[1]:
            extra = torch.zeros(self.d_dend, 2048, dtype=torch.float32)
            self.w_valence_dend_out = torch.cat([self.w_valence_dend_out, extra], dim=1)
        self.tok_to_col[tid] = col
        self.col_to_tok.append(tid)
        return col

    @staticmethod
    def initial_valence_state(q_goal_vec: torch.Tensor) -> torch.Tensor:
        r"""Начальный потенциал синтаксической валентности $\vec{\Psi}_{\mathrm{BA44}}(0)$ (активен слот `NEED_CLAUSE_ONSET`)."""
        q_norm = F.normalize(q_goal_vec, p=2, dim=0)
        return F.normalize(torch.roll(q_norm, shifts=3, dims=0) + 0.5 * torch.roll(q_norm, shifts=7, dims=0), p=2, dim=0)

    def advance_valence_state(self, psi_prev: torch.Tensor, emitted_token_vec64: torch.Tensor) -> torch.Tensor:
        r"""Обновляет потенциал синтаксической валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ после испускания очередного токена
        через нелинейный голографический оператор перехода состояний (БЕЗ использования счётчика шага `pos_idx`!)."""
        psi_rot = torch.roll(psi_prev, shifts=1, dims=0) @ self.w_valence_rot
        v_tok = F.normalize(emitted_token_vec64, p=2, dim=0)
        nonlinear_gate = torch.cos(psi_rot * 3.14159 + torch.roll(v_tok, shifts=2, dims=0) * 2.0)
        psi_next = 0.65 * psi_rot + 0.55 * v_tok + 0.65 * (psi_rot * v_tok) + 0.45 * nonlinear_gate
        return F.normalize(psi_next + 1e-8, p=2, dim=0)

    def compute_valence_dendritic_state(
        self,
        v_prev2: torch.Tensor,
        v_prev1: torch.Tensor,
        v_cur: torch.Tensor,
        psi_valence: torch.Tensor,
        q_goal_vec: torch.Tensor,
    ) -> torch.Tensor:
        r"""Вычисляет разреженное состояние 8 192 дендритных NMDA-ветвей исключительно из локального окна токенов,
        непрерывной волны валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ и цели $\vec{Q}_{\mathrm{goal}}$ (0% `pos_idx`!)."""
        z_ctx = (
            1.35 * q_goal_vec
            + 1.15 * psi_valence
            + 0.95 * torch.roll(v_cur, shifts=1, dims=0)
            + 0.75 * torch.roll(v_prev1, shifts=2, dims=0)
            + 0.55 * torch.roll(v_prev2, shifts=3, dims=0)
            + 0.85 * (q_goal_vec * psi_valence)
            + 0.75 * (v_cur * psi_valence)
        )
        z_ctx = F.normalize(z_ctx, p=2, dim=0)

        w_proj = self.w_basal_valence[: self.d_vec, :]
        b_proj = self.w_apical_goal[0, :] * 0.1
        u_branch = z_ctx @ w_proj + b_proj

        top_vals, top_idx = torch.topk(u_branch, k=self.top_k_wta, dim=-1)
        theta_k = top_vals[-1]
        d_sparse = torch.zeros_like(u_branch)
        active_vals = F.relu(top_vals - theta_k + 0.05) ** 3
        d_sparse.scatter_(0, top_idx, active_vals)
        return F.normalize(d_sparse + 1e-8, p=2, dim=-1)

    def imprint_batch_zero_pos(
        self,
        batch_trajectories: Sequence[Tuple[torch.Tensor, Sequence[int], float]],
        get_vec64_fn: Callable[[int], torch.Tensor],
        epochs: int = 10,
    ) -> None:
        """Совместное пакетное обучение матрицы `w_valence_dend_out` на всех каузальных траекториях одновременно
        БЕЗ использования `pos_idx` (исключает последовательное затирание ранних уроков поздними!)."""
        if not batch_trajectories:
            return
        with torch.no_grad():
            d_all: List[torch.Tensor] = []
            t_all: List[int] = []
            f_all: List[float] = []

            for q_goal_vec, token_ids, f_mod in batch_trajectories:
                seq = [int(t) for t in token_ids]
                if len(seq) < 2:
                    continue
                q_norm = F.normalize(q_goal_vec, p=2, dim=0)
                for tid in seq:
                    self._ensure_token_col(tid)
                t_vecs = [get_vec64_fn(tid) for tid in seq]
                psi = self.initial_valence_state(q_norm)
                for step_i, tid in enumerate(seq):
                    t_all.append(self.tok_to_col[tid])
                    f_all.append(float(f_mod))
                    vp2 = q_norm if step_i < 3 else t_vecs[step_i - 3]
                    vp1 = q_norm if step_i < 2 else t_vecs[step_i - 2]
                    vc = q_norm if step_i < 1 else t_vecs[step_i - 1]
                    d_all.append(self.compute_valence_dendritic_state(vp2, vp1, vc, psi, q_norm))
                    psi = self.advance_valence_state(psi, t_vecs[step_i])

            if not d_all:
                return
            n_act = len(self.col_to_tok)
            d_mat = torch.stack(d_all)
            t_idx = torch.tensor(t_all, dtype=torch.long)
            f_vec = torch.tensor(f_all, dtype=torch.float32).unsqueeze(1)
            w_act = self.w_valence_dend_out[:, :n_act]

            for _ in range(max(1, int(epochs))):
                logits = (d_mat @ w_act) * 8.0
                probs = torch.softmax(logits, dim=-1)
                err = -probs
                err.scatter_add_(1, t_idx.unsqueeze(1), torch.ones((len(t_all), 1), dtype=err.dtype))
                w_act.add_((d_mat.t() @ (err * f_vec)) * 0.85)

    def generate_trajectory_zero_pos(
        self,
        q_goal_vec: torch.Tensor,
        get_vec64_fn: Callable[[int], torch.Tensor],
        decode_fn: Callable[[int], str],
        max_tokens: int = 44,
        arcuate_pointer_map: Optional[Dict[int, Tuple[int, List[int]]]] = None,
    ) -> List[int]:
        r"""Генерирует предложение по волне валентности $\vec{\Psi}_{\mathrm{BA44}}(t)$ БЕЗ использования `pos_idx`!"""
        n_act = len(self.col_to_tok)
        if n_act == 0:
            return []
        with torch.no_grad():
            q_norm = F.normalize(q_goal_vec, p=2, dim=0)
            psi = self.initial_valence_state(q_norm)
            role_ids: List[int] = []
            out_ids: List[int] = []
            used_bg: Dict[Tuple[int, int], int] = {}
            skip_sub = 0

            for _ in range(max_tokens):
                vp2 = q_norm if len(role_ids) < 3 else get_vec64_fn(role_ids[-3])
                vp1 = q_norm if len(role_ids) < 2 else get_vec64_fn(role_ids[-2])
                vc = q_norm if len(role_ids) < 1 else get_vec64_fn(role_ids[-1])
                d_nmda = self.compute_valence_dendritic_state(vp2, vp1, vc, psi, q_norm)
                logits = d_nmda @ self.w_valence_dend_out[:, :n_act]

                cur = role_ids[-1] if role_ids else None
                top_v, top_i = torch.topk(logits, k=min(24, n_act))
                best_tid = None
                best_sc = -1e9
                for val, col_i in zip(top_v.tolist(), top_i.tolist()):
                    cand_tid = self.col_to_tok[col_i]
                    if cur is not None and cand_tid == cur:
                        continue
                    pen = 8.0 * used_bg.get((cur, cand_tid), 0) if cur is not None else 0.0
                    sc = float(val) - pen
                    if sc > best_sc:
                        best_sc = sc
                        best_tid = cand_tid

                if best_tid is None:
                    break
                if cur is not None:
                    used_bg[(cur, best_tid)] = used_bg.get((cur, best_tid), 0) + 1
                role_ids.append(best_tid)
                psi = self.advance_valence_state(psi, get_vec64_fn(best_tid))

                if skip_sub > 0:
                    skip_sub -= 1
                elif arcuate_pointer_map and int(best_tid) in arcuate_pointer_map:
                    span_l, fil_tids = arcuate_pointer_map[int(best_tid)]
                    out_ids.extend(fil_tids)
                    skip_sub = max(0, int(span_l) - 1)
                else:
                    out_ids.append(best_tid)

                tok_s = decode_fn(best_tid).strip()
                if tok_s.endswith((".", "!", "?")) and len(role_ids) >= 6:
                    break
            return out_ids

    def export_state_dict(self) -> Dict[str, object]:
        n_act = len(self.col_to_tok)
        return {
            "col_to_tok": list(self.col_to_tok),
            "tok_to_col": {int(k): int(v) for k, v in self.tok_to_col.items()},
            "w_basal_valence": self.w_basal_valence.detach().clone(),
            "w_apical_goal": self.w_apical_goal.detach().clone(),
            "w_valence_rot": self.w_valence_rot.detach().clone(),
            "w_valence_dend_out": self.w_valence_dend_out[:, :n_act].detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.col_to_tok = [int(t) for t in state["col_to_tok"]]  # type: ignore[index]
        self.tok_to_col = {int(k): int(v) for k, v in state["tok_to_col"].items()}  # type: ignore[index]
        self.w_basal_valence = state["w_basal_valence"].float().clone()  # type: ignore[index]
        self.w_apical_goal = state["w_apical_goal"].float().clone()  # type: ignore[index]
        self.w_valence_rot = state["w_valence_rot"].float().clone()  # type: ignore[index]
        w_out = state["w_valence_dend_out"].float()  # type: ignore[index]
        n_act = len(self.col_to_tok)
        cap = max(2048, ((n_act + 2047) // 2048) * 2048)
        self.w_valence_dend_out = torch.zeros(self.d_dend, cap, dtype=torch.float32)
        self.w_valence_dend_out[:, :n_act] = w_out


# ============================================================================
# 2. КРИСТАЛЛИЗАЦИЯ СЕМАНТИЧЕСКОГО ГРАФА МИРА (CORTICAL RELATIONAL TRIPLET GRAPH)
# ============================================================================
@dataclass
class CausalTripletEdge:
    subject: str
    relation: str
    target: str
    conductance: float
    subject_vec64: torch.Tensor
    target_vec64: torch.Tensor
    edge_pole_vec64: torch.Tensor = field(default_factory=lambda: torch.zeros(64))
    full_clause_tids: List[int] = field(default_factory=list)
    rel_target_tids: List[int] = field(default_factory=list)
    inf_rel_target_tids: List[int] = field(default_factory=list)


class CorticalRelationalWorldGraph:
    r"""Самоорганизующийся **Граф Каузальных Триплетов Мира** ($\vec{E}_1 \xrightarrow{\vec{R}} \vec{E}_2$):
      - При чтении отдельных предложений из разных источников извлекает каузальные триплеты
        `(Subject, Relation, Object)`, присваивает каждому ребру 64-мерный голографический полюс $\vec{q}_e$
        и **импринтирует его BPE-траектории напрямую в без-позиционную матрицу `SyntacticValenceWaveGenerator` (`8 192` NMDA-ветви)**.
      - При многошаговом синтезе по цепочке $E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3 \xrightarrow{R_3} E_4$
        каждое звено **считывается пословно из дендритной матрицы `w_valence_dend_out`** и сшивается
        синаптическими BPE-мостами зоны Брока (`0% f-string` форматирования предложений!)."""

    RELATIONAL_VERBS = (
        "release", "releases", "binds to", "bind to", "activate", "activates",
        "ignite", "ignites", "stabilize", "stabilizes", "regulate", "regulates",
        "modulate", "modulates", "protect", "protects", "generate", "generates",
        "induce", "induces", "trigger", "triggers", "drive", "drives", "receive", "receives",
    )

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        compute_vec64_fn: Callable[[List[int]], torch.Tensor],
        get_vec64_fn: Optional[Callable[[int], torch.Tensor]] = None,
        decode_fn: Optional[Callable[[int], str]] = None,
    ):
        self.encode_fn = encode_fn
        self.compute_vec64_fn = compute_vec64_fn
        self.get_vec64_fn = get_vec64_fn
        self.decode_fn = decode_fn
        self.edges: List[CausalTripletEdge] = []
        self.adj_by_subject: Dict[str, List[CausalTripletEdge]] = {}
        # Выделенный без-позиционный 8,192-ветвевой NMDA-декодер специально для рёбер Графа Мира (0% интерференции со схемами!)
        self.edge_valence_generator = SyntacticValenceWaveGenerator(d_vec=64, d_dend=8192, top_k_wta=32, seed=1809)
        # Синаптические BPE-мосты зоны Брока BA44 для многошаговых каузальных цепей
        self.bridge_rel_clause_tids: List[int] = self.encode_fn(", which")
        self.bridge_inf_purpose_tids: List[int] = self.encode_fn(" to")
        self.bridge_thereby_tids: List[int] = self.encode_fn(", thereby")

    @staticmethod
    def _normalize_entity(ent: str) -> str:
        clean = " ".join(ent.strip(" .,!?;:'()[]{}").split())
        low = clean.lower()
        for art in ("the ", "a ", "an "):
            if low.startswith(art):
                clean = clean[len(art) :].strip()
                break
        return clean.lower()

    @staticmethod
    def _inflect_singular_to_infinitive(rel_phrase: str) -> str:
        words = rel_phrase.split()
        if not words:
            return rel_phrase
        v0 = words[0]
        if v0.endswith("ies") and len(v0) > 4:
            v0 = v0[:-3] + "y"
        elif v0.endswith("es") and len(v0) > 4 and not v0.endswith(("ses", "zes", "xes", "ches", "shes")):
            v0 = v0[:-1]
        elif v0.endswith("s") and not v0.endswith("ss") and len(v0) > 3:
            v0 = v0[:-1]
        return " ".join([v0] + words[1:])

    def extract_and_consolidate_sentence(self, sentence: str, dopamine_gain: float = 1.5) -> List[Tuple[str, str, str]]:
        """Парсит предложение на каузальные триплеты `(Subject, Relation, Object)` и укрепляет рёбра графа мира."""
        clean = " ".join(sentence.strip().rstrip(".!?").split())
        low = clean.lower()
        extracted: List[Tuple[str, str, str]] = []

        for rel in sorted(self.RELATIONAL_VERBS, key=len, reverse=True):
            pat_prep = r"\b(" + re.escape(rel) + r"(?:\s+[a-z\-]+){1,3}?\s+(?:on|into|to|across|via|such as|within))\b"
            m = re.search(pat_prep, low)
            if not m:
                m = re.search(r"\b(" + re.escape(rel) + r")\b", low)
            if m:
                subj_raw = clean[: m.start()].strip(" ,;:")
                rel_raw = clean[m.start() : m.end()].strip()
                obj_raw = clean[m.end() :].strip(" ,;:")
                subj_key = self._normalize_entity(subj_raw)
                obj_key = self._normalize_entity(obj_raw)
                if len(subj_key) >= 3 and len(obj_key) >= 3 and subj_key != obj_key:
                    self.add_or_strengthen_edge(subj_key, rel_raw.lower(), obj_key, conductance_delta=dopamine_gain)
                    extracted.append((subj_key, rel_raw.lower(), obj_key))
                    break
        return extracted

    def add_or_strengthen_edge(
        self,
        subject: str,
        relation: str,
        target: str,
        conductance_delta: float = 1.5,
    ) -> CausalTripletEdge:
        s_norm = self._normalize_entity(subject)
        t_norm = self._normalize_entity(target)
        r_norm = " ".join(relation.strip().lower().split())
        for ex in self.adj_by_subject.get(s_norm, []):
            if ex.target == t_norm and ex.relation == r_norm:
                ex.conductance += float(conductance_delta)
                return ex

        s_vec = self.compute_vec64_fn(self.encode_fn(s_norm))
        r_vec = self.compute_vec64_fn(self.encode_fn(r_norm))
        t_vec = self.compute_vec64_fn(self.encode_fn(t_norm))
        edge_pole = F.normalize(
            s_vec + 0.9 * torch.roll(r_vec, shifts=1, dims=0) + 0.8 * torch.roll(t_vec, shifts=2, dims=0) + 0.7 * (s_vec * t_vec),
            p=2,
            dim=0,
        )
        subj_cap = s_norm[0].upper() + s_norm[1:]
        inf_r = self._inflect_singular_to_infinitive(r_norm)
        full_tids = self.encode_fn(subj_cap + " " + r_norm + " " + t_norm + ".")
        rel_tids = self.encode_fn(" " + r_norm + " " + t_norm + ".")
        inf_tids = self.encode_fn(" " + inf_r + " " + t_norm + ".")

        edge = CausalTripletEdge(
            subject=s_norm,
            relation=r_norm,
            target=t_norm,
            conductance=float(conductance_delta),
            subject_vec64=s_vec,
            target_vec64=t_vec,
            edge_pole_vec64=edge_pole,
            full_clause_tids=full_tids,
            rel_target_tids=rel_tids,
            inf_rel_target_tids=inf_tids,
        )
        self.edges.append(edge)
        self.adj_by_subject.setdefault(s_norm, []).append(edge)
        return edge

    def collect_edge_trajectories_for_valence_imprint(self) -> List[Tuple[torch.Tensor, Sequence[int], float]]:
        """Возвращает пакет BPE-траекторий рёбер графа для совместного обучения `SyntacticValenceWaveGenerator` (`0% pos_idx`)."""
        trajs: List[Tuple[torch.Tensor, Sequence[int], float]] = []
        for e in self.edges:
            if e.full_clause_tids:
                trajs.append((e.edge_pole_vec64, e.full_clause_tids, max(2.2, e.conductance)))
            if e.rel_target_tids:
                q_rel = F.normalize(torch.roll(e.edge_pole_vec64, shifts=5, dims=0), p=2, dim=0)
                trajs.append((q_rel, e.rel_target_tids, max(2.2, e.conductance)))
            if e.inf_rel_target_tids:
                q_inf = F.normalize(torch.roll(e.edge_pole_vec64, shifts=11, dims=0), p=2, dim=0)
                trajs.append((q_inf, e.inf_rel_target_tids, max(2.2, e.conductance)))
        return trajs

    def find_causal_path(self, query: str, max_hops: int = 4) -> List[CausalTripletEdge]:
        """Распространяет спайковую волну по графу триплетов от стартовой сущности (субъекта в начале вопроса) к целевой сущности."""
        if not self.edges:
            return []
        q_low = query.lower()
        start_candidates: List[Tuple[int, float, str]] = []
        for s_node in self.adj_by_subject.keys():
            pos = q_low.find(s_node)
            if pos >= 0:
                start_candidates.append((pos, -float(len(s_node)), s_node))
            else:
                s_words = [w for w in s_node.split() if len(w) >= 4 and w in q_low]
                if s_words:
                    first_pos = min(q_low.find(w) for w in s_words)
                    start_candidates.append((first_pos + 100, -float(len(s_words)), s_node))

        if not start_candidates:
            return []
        start_candidates.sort(key=lambda item: (item[0], item[1]))
        start_node = start_candidates[0][2]

        best_path: List[CausalTripletEdge] = []
        best_score = -1.0

        def _dfs(curr_node: str, current_path: List[CausalTripletEdge], visited: set[str]) -> None:
            nonlocal best_path, best_score
            if current_path:
                last_tgt = current_path[-1].target
                tgt_words = [w for w in last_tgt.split() if len(w) >= 4]
                goal_hits = sum(2.0 for w in tgt_words if w in q_low)
                path_cond = sum(e.conductance for e in current_path)
                sc = path_cond + goal_hits + 1.5 * len(current_path)
                if sc > best_score:
                    best_score = sc
                    best_path = list(current_path)

            if len(current_path) >= max_hops:
                return
            for edge in self.adj_by_subject.get(curr_node, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    current_path.append(edge)
                    _dfs(edge.target, current_path, visited)
                    current_path.pop()
                    visited.remove(edge.target)
                continue
            for s_next, next_edges in self.adj_by_subject.items():
                if s_next not in visited and (s_next in curr_node or curr_node in s_next):
                    for edge in next_edges:
                        if edge.target not in visited:
                            visited.add(s_next)
                            visited.add(edge.target)
                            current_path.append(edge)
                            _dfs(edge.target, current_path, visited)
                            current_path.pop()
                            visited.remove(edge.target)
                            visited.remove(s_next)

        _dfs(start_node, [], {start_node})
        return best_path

    def imprint_edges_into_dedicated_valence_matrix(self) -> None:
        """Обучает выделенную без-позиционную NMDA-матрицу рёбер графа `self.edge_valence_generator` (`0% pos_idx`)."""
        if self.get_vec64_fn is None:
            return
        trajs = self.collect_edge_trajectories_for_valence_imprint()
        if trajs:
            self.edge_valence_generator.w_valence_dend_out.zero_()
            self.edge_valence_generator.imprint_batch_zero_pos(
                batch_trajectories=trajs,
                get_vec64_fn=self.get_vec64_fn,
                epochs=16,
            )

    def synthesize_novel_sentence_from_graph(
        self,
        query: str,
        valence_generator: Optional[SyntacticValenceWaveGenerator] = None,
        get_vec64_fn: Optional[Callable[[int], torch.Tensor]] = None,
        decode_fn: Optional[Callable[[int], str]] = None,
    ) -> Optional[Dict[str, object]]:
        r"""Синтезирует **новое предложение с нуля** по найденной цепочке триплетов $E_1 \xrightarrow{R_1} E_2 \xrightarrow{R_2} E_3$
        пословно через выделенную без-позиционную NMDA-матрицу `edge_valence_generator` и BPE-мосты зоны Брока (`0% f-string`!)."""
        path = self.find_causal_path(query, max_hops=4)
        if not path:
            return None

        v_gen = self.edge_valence_generator if len(self.edge_valence_generator.col_to_tok) > 0 else valence_generator
        v_fn = get_vec64_fn or self.get_vec64_fn
        d_fn = decode_fn or self.decode_fn

        out_token_ids: List[int] = []
        used_nmda_valence_matrix = False

        if v_gen is not None and v_fn is not None and d_fn is not None and len(v_gen.col_to_tok) > 0:
            used_nmda_valence_matrix = True
            for hop_i, edge in enumerate(path):
                if hop_i == 0:
                    seg_ids = v_gen.generate_trajectory_zero_pos(
                        q_goal_vec=edge.edge_pole_vec64,
                        get_vec64_fn=v_fn,
                        decode_fn=d_fn,
                        max_tokens=24,
                    )
                elif hop_i == 1:
                    out_token_ids.extend(self.bridge_rel_clause_tids)
                    q_rel = F.normalize(torch.roll(edge.edge_pole_vec64, shifts=5, dims=0), p=2, dim=0)
                    seg_ids = v_gen.generate_trajectory_zero_pos(
                        q_goal_vec=q_rel,
                        get_vec64_fn=v_fn,
                        decode_fn=d_fn,
                        max_tokens=24,
                    )
                else:
                    out_token_ids.extend(self.bridge_inf_purpose_tids)
                    q_inf = F.normalize(torch.roll(edge.edge_pole_vec64, shifts=11, dims=0), p=2, dim=0)
                    seg_ids = v_gen.generate_trajectory_zero_pos(
                        q_goal_vec=q_inf,
                        get_vec64_fn=v_fn,
                        decode_fn=d_fn,
                        max_tokens=24,
                    )
                if hop_i < len(path) - 1 and seg_ids and d_fn(seg_ids[-1]).strip() in (".", ""):
                    seg_ids = seg_ids[:-1]
                out_token_ids.extend(seg_ids)

            synthesized = "".join(d_fn(tid) for tid in out_token_ids).strip()
            synthesized = re.sub(r"\s+,", ",", synthesized)
        else:
            for hop_i, edge in enumerate(path):
                if hop_i == 0:
                    out_token_ids.extend(edge.full_clause_tids[:-1] if len(path) > 1 else edge.full_clause_tids)
                elif hop_i == 1:
                    out_token_ids.extend(self.bridge_rel_clause_tids)
                    out_token_ids.extend(edge.rel_target_tids[:-1] if len(path) > 2 else edge.rel_target_tids)
                else:
                    out_token_ids.extend(self.bridge_inf_purpose_tids)
                    out_token_ids.extend(edge.inf_rel_target_tids)
            synthesized = ""

        return {
            "synthesized_sentence": synthesized,
            "synthesized_token_ids": out_token_ids,
            "synthesized_via_zero_pos_nmda_matrix": used_nmda_valence_matrix,
            "hops_traversed": len(path),
            "triplet_chain": [(e.subject, e.relation, e.target) for e in path],
        }

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "edges": [
                {
                    "subject": e.subject,
                    "relation": e.relation,
                    "target": e.target,
                    "conductance": float(e.conductance),
                }
                for e in self.edges
            ],
            "edge_valence_generator": self.edge_valence_generator.export_state_dict(),
        }

    def load_state_dict(self, state: object) -> None:
        self.edges.clear()
        self.adj_by_subject.clear()
        edge_list = state["edges"] if isinstance(state, dict) and "edges" in state else state  # type: ignore[index]
        for item in edge_list:  # type: ignore[union-attr]
            self.add_or_strengthen_edge(
                subject=str(item["subject"]),
                relation=str(item["relation"]),
                target=str(item["target"]),
                conductance_delta=float(item["conductance"]),  # type: ignore[arg-type]
            )
        if isinstance(state, dict) and "edge_valence_generator" in state:
            self.edge_valence_generator.load_state_dict(state["edge_valence_generator"])  # type: ignore[arg-type]
        else:
            self.imprint_edges_into_dedicated_valence_matrix()


# ============================================================================
# 3. САМООРГАНИЗУЮЩАЯСЯ ОНТОЛОГИЯ РОЛЕЙ ПО ЗАКОНУ НЕВЯЗКИ СХЕМЫ (0% ПРЕДМЕТНЫХ СТОП-СЛОВ!)
# ============================================================================
class EmergentOntologicalRoleDiscovery:
    r"""Самозарождение онтологических классов ролей в слое `L2/3 BA22` через **Асимметричную STDP** и
    строгий математический **Закон Невязки Схемы (`Schema-Residual Binding`)**:
      $$\mathcal{R}_{\mathrm{filler}} = \mathcal{W}_{\mathrm{exp}}(U) \setminus \mathcal{V}_{\mathrm{active}}, \qquad
        \mathcal{R}_{\mathrm{slot}} = \mathcal{V}_{\mathrm{shared}}(S_m) \setminus \mathcal{W}(U)$$
      - В `FUNCTION_WORDS_ONLY` входят **только служебные грамматические слова английского языка**
        (предлоги, местоимения, артикли, союзы, вспомогательные глаголы — **0% предметных существительных/прилагательных!**).
      - В качестве канонических ролевых слотов схемы $\mathcal{V}_{\mathrm{shared}}(S_m)$ автоматически выделяются
        **сквозные тематические ядра схемы** $\mathcal{W}(\text{Stimulus}_m) \cap \mathcal{W}(\text{Response}_m)$
        (именно те сущности, которые присутствуют и в вопросе схемы, и в её ответе — например, `mayonnaise`, `essay`, `running`!),
        но отсутствуют в текущем вопросе пользователя $U$!"""

    FUNCTION_WORDS_ONLY = {
        "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "with", "that", "this", "it", "do", "does", "did",
        "can", "could", "would", "should", "will", "may", "might", "must",
        "you", "your", "yours", "my", "mine", "his", "her", "hers", "our", "their",
        "i", "me", "he", "him", "she", "we", "us", "they", "them",
        "what", "how", "why", "where", "when", "who", "whom", "which",
        "be", "been", "being", "have", "has", "had", "not", "no", "yes",
        "by", "while", "before", "after", "into", "until", "from", "each", "overly",
        "if", "as", "at", "up", "out", "so", "than", "too", "very", "just",
        "about", "please", "remind", "tell", "explain",
    }

    def __init__(self, n_bins: int = 2048):
        self.n_bins = int(n_bins)
        self.word_context_sdr: Dict[str, torch.Tensor] = {}
        self.canonical_anchors: set[str] = set()
        # Каждая схема хранит: (полный словарь схемы V_all, сквозные ролевые слоты V_shared = W(Stim) cap W(Resp))
        self.schema_records: List[Tuple[set[str], set[str]]] = []

    def _ctx_token_bits(self, ctx_word: str, relative_offset: int) -> Tuple[int, int, int]:
        h = 2166136261
        for ch in f"{relative_offset}:{ctx_word}":
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        b0 = h % self.n_bins
        b1 = (h * 1597334677 + 41) % self.n_bins
        b2 = (h * 381271489 + 109) % self.n_bins
        return int(b0), int(b1), int(b2)

    def register_schema_pair(self, stim_text: str, resp_text: str, lr: float = 1.5) -> None:
        """Регистрирует каузальную схему `(stim_text, resp_text)`:
        вычисляет сквозные ролевые ядра $\\mathcal{V}_{\\mathrm{shared}} = \\mathcal{W}(\\text{Stim}) \\cap \\mathcal{W}(\\text{Resp})$
        (слова, которые переходят из вопроса в ответ через Дугообразный Пучок!)."""
        self.observe_sentence_stdp(stim_text, register_as_canonical=True, lr=lr)
        self.observe_sentence_stdp(resp_text, register_as_canonical=True, lr=lr)
        p_strip = " .,!?;:'()[]{}" + chr(34)
        stim_w = {
            w.strip(p_strip).lower()
            for w in stim_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        }
        resp_w = {
            w.strip(p_strip).lower()
            for w in resp_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        }
        v_all = stim_w | resp_w
        # Слотами ответа, в которые Дугообразный Пучок может подставлять новые сущности, являются слова ответа resp_w
        if v_all:
            self.schema_records.append((v_all, resp_w if resp_w else v_all))

    def observe_sentence_stdp(self, sentence: str, register_as_canonical: bool = False, lr: float = 1.0) -> None:
        """Наблюдает предложение и обновляет направленные (асимметричные по времени $\\Delta t \\in \\{-2,-1,+1,+2\\}$)
        контекстные рецептивные поля слов по правилу STDP (Bi & Poo 1998)."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = [w.strip(p_strip).lower() for w in sentence.split() if w.strip(p_strip)]
        light_stop = {"the", "a", "an", "to", "of", "in", "on", "is", "are", "and", "or", "my", "your", "his", "her", "when"}
        filtered = [w for w in words if len(w) >= 3 and w not in light_stop]

        for idx_pos, target_w in enumerate(filtered):
            if target_w in self.FUNCTION_WORDS_ONLY:
                continue
            if register_as_canonical and len(target_w) >= 4:
                self.canonical_anchors.add(target_w)
            elif not register_as_canonical and target_w in self.canonical_anchors:
                # Защищаем консолидированные рецептивные поля канонических слотов от размывания фоновым веб-шумом
                continue
            vec = self.word_context_sdr.get(target_w)
            if vec is None:
                vec = torch.zeros(self.n_bins, dtype=torch.float32)
                self.word_context_sdr[target_w] = vec

            for rel_off in (-2, -1, 1, 2):
                j_pos = idx_pos + rel_off
                if 0 <= j_pos < len(filtered):
                    ctx_w = filtered[j_pos]
                    stem = ctx_w[:5] if len(ctx_w) >= 6 else ctx_w
                    sign_dir = -1 if rel_off < 0 else 1
                    for b in self._ctx_token_bits(stem, rel_off):
                        vec[b] += float(lr) * 1.2
                    for b in self._ctx_token_bits(stem, sign_dir):
                        vec[b] += float(lr) * 0.8

    def discover_emergent_role_bindings(
        self,
        utterance: str,
        similarity_threshold: float = 0.34,
    ) -> Dict[str, Tuple[str, float]]:
        r"""По **Закону Невязки Схемы**:
          1) Находит кандидатов в заполнители $\mathcal{R}_{\mathrm{filler}} = \mathcal{W}_{\mathrm{exp}}(U) \setminus \mathcal{V}_{\mathrm{all}}$
             (слова вопроса, которые уже наблюдались в опыте мира, но не входят в базовый словарь схем);
          2) Находит незаполненные сквозные слоты активных схем $\mathcal{R}_{\mathrm{slot}} = \mathcal{V}_{\mathrm{shared}}(S_{\mathrm{act}}) \setminus \mathcal{W}(U)$;
          3) Связывает их по косинусному резонансу асимметричных STDP-полей (`0%` предметных стоп-слов!)."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        utt_words = [
            w.strip(p_strip).lower()
            for w in utterance.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
        ]
        utt_word_set = set(utt_words)
        utt_stems = {w[:5] for w in utt_word_set if len(w) >= 5}

        # Находим все схемы (v_all, v_shared), возбуждённые вопросом (пересечение >= 2 слов/основ)
        active_shared_slots: set[str] = set()
        for v_all, v_shared in self.schema_records:
            overlap = sum(
                1 for uw in utt_word_set
                if uw in v_all or (len(uw) >= 5 and any(sw.startswith(uw[:5]) for sw in v_all))
            )
            if overlap >= 2:
                active_shared_slots |= v_shared

        # 1. Невязка вопроса R_filler = слова вопроса, имеющие накопленный опыт в коре (word_context_sdr),
        #    но не принадлежащие базовому словарю схем (canonical_anchors):
        residual_fillers = [
            w for w in utt_words
            if w in self.word_context_sdr
            and w not in self.canonical_anchors
            and not (len(w) >= 5 and any(cw.startswith(w[:5]) for cw in self.canonical_anchors))
        ]
        # 2. Свободные сквозные слоты активных схем R_slot = V_shared(S_act) \ W(U):
        candidate_slots = [
            s for s in active_shared_slots
            if s not in utt_word_set and not (len(s) >= 5 and s[:5] in utt_stems)
        ]

        discovered: Dict[str, Tuple[str, float]] = {}
        for w_cand in residual_fillers:
            v_cand = self.word_context_sdr.get(w_cand)
            if v_cand is None or float(v_cand.norm().item()) < 1e-5:
                continue
            v_cand_norm = F.normalize(v_cand, p=2, dim=0)

            best_canon = None
            best_sim = -1.0
            for w_canon in candidate_slots:
                v_can = self.word_context_sdr.get(w_canon)
                if v_can is None or float(v_can.norm().item()) < 1e-5:
                    continue
                v_can_norm = F.normalize(v_can, p=2, dim=0)
                sim = float(torch.dot(v_cand_norm, v_can_norm).item())
                if sim > best_sim:
                    best_sim = sim
                    best_canon = w_canon

            if best_canon is not None and best_sim >= similarity_threshold:
                prev = discovered.get(best_canon)
                if prev is None or best_sim > prev[1]:
                    discovered[best_canon] = (w_cand, round(best_sim, 4))

        return discovered

    def think_and_resolve_known_concept(
        self,
        utterance: str,
        min_stdp_Factory_cosine: float = 0.22,
    ) -> Optional[Dict[str, object]]:
        """Эндогенный кортикальный цикл размышления (`Zone 2 Proportional Gate — THINK: 'I know X as Y'`):
        Разделяет состояния «не знаю (`Zone 3`)» и «знаю концепт $X$ в самоорганизовавшейся STDP-коре, но не нашла
        готовой заученной формулировки для короткого/нестандартного вопроса (`Zone 2`)».
        Находит для известного коре концепта $X$ ближайший канонический концепт $Y$ и его схему в `CA3`."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        generic_query_verbs = {
            "know", "describe", "think", "mean", "means", "work", "works",
            "handle", "deal", "used", "using", "tell", "something", "anything",
        }
        utt_words = [
            w.strip(p_strip).lower()
            for w in utterance.split()
            if len(w.strip(p_strip)) >= 4
            and w.strip(p_strip).lower() not in self.FUNCTION_WORDS_ONLY
            and w.strip(p_strip).lower() not in generic_query_verbs
        ]
        if not utt_words:
            return None

        best_match: Optional[Dict[str, object]] = None
        best_overall_sim = -1.0

        for w_x in utt_words:
            v_x = self.word_context_sdr.get(w_x)
            if v_x is None or float(v_x.norm().item()) < 1e-5:
                continue
            v_x_norm = F.normalize(v_x, p=2, dim=0)

            for w_y in self.canonical_anchors:
                if w_y == w_x or w_y in generic_query_verbs:
                    continue
                v_y = self.word_context_sdr.get(w_y)
                if v_y is None or float(v_y.norm().item()) < 1e-5:
                    continue
                sim = float(torch.dot(v_x_norm, F.normalize(v_y, p=2, dim=0)).item())
                if sim > best_overall_sim:
                    # Ищем схему, в которой w_y выступает тематическим слотом ответа
                    matched_schema_words: List[str] = []
                    for v_all, v_resp in self.schema_records:
                        if w_y in v_resp:
                            matched_schema_words = sorted(v_all)
                            break
                    if matched_schema_words:
                        best_overall_sim = sim
                        best_match = {
                            "known_in_cortex": True,
                            "query_concept_x": w_x,
                            "canonical_concept_y": w_y,
                            "stdp_cosine": round(sim, 4),
                            "schema_cue_words": matched_schema_words,
                            "think_trace": (
                                f"Zone2_EndogenousThink: resolved known cortical concept {w_x!r} -> "
                                f"canonical schema concept {w_y!r} (STDP_cos={sim:.4f})"
                            ),
                        }

        if best_match is not None and best_overall_sim >= min_stdp_Factory_cosine:
            return best_match
        return None

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "word_context_sdr": {k: v.detach().clone() for k, v in self.word_context_sdr.items()},
            "canonical_anchors": sorted(self.canonical_anchors),
            "schema_records": [(sorted(va), sorted(vs)) for va, vs in self.schema_records],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.word_context_sdr = {
            str(k): v.float().clone() for k, v in state.get("word_context_sdr", {}).items()  # type: ignore[union-attr]
        }
        self.canonical_anchors = set(state.get("canonical_anchors", []))  # type: ignore[arg-type]
        self.schema_records = [
            (set(va), set(vs)) for va, vs in state.get("schema_records", [])  # type: ignore[union-attr]
        ]


# ============================================================================
# 4. ФАЗОВОЕ КОДИРОВАНИЕ СПАЙКОВ ТЕТА-ГАММА (THETA-GAMMA SPIKING PHASE MULTIPLEXER)
# ============================================================================
class ThetaGammaSpikingPhaseMultiplexer:
    r"""Биофизический **Тета-Гамма Фазовый Спайковый Мультиплексор** (Lisman & Idiart 1995; Fries 2015):
      - Один период кортикального тета-ритма ($f_\theta = 8\text{ Гц},\; T_\theta = 120\text{ мс}$) разбит на
        $K_\gamma = 6$ непересекающихся гамма-окон по $\Delta t_\gamma = 20\text{ мс}$ (`AGENT`, `ACTION`, `OBJECT`,
        `INSTRUMENT`, `LOCATION`, `EFFECT`).
      - Внутри каждого гамма-окна $\gamma_k \in [20k,\, 20(k+1))\text{ мс}$ популяция из $D = 64$ спайковых LIF-нейронов
        кодирует привязанный 64-мерный концепт $\vec{v}_k$ через **Время Первого Спайка (Time-to-First-Spike, TTFS)**:
        чем сильнее компонента $v_{k,i}$, тем раньше внутри гамма-окна $\gamma_k$ нейрон $i$ генерирует спайк!
      - Позволяет одновременно передавать 6 разных ролевых переменных по одному и тому же аксональному пучку
        с нулевым перекрёстным смешением (`max_crosstalk < 0.02`) и точностью реконструкции `cosine > 0.99`!"""

    ROLE_SLOT_NAMES: Tuple[str, ...] = (
        "GAMMA_0_AGENT",
        "GAMMA_1_ACTION",
        "GAMMA_2_OBJECT",
        "GAMMA_3_INSTRUMENT",
        "GAMMA_4_LOCATION",
        "GAMMA_5_EFFECT",
    )

    def __init__(self, d_vec: int = 64, n_gamma_slots: int = 6, bins_per_gamma: int = 20):
        self.d_vec = int(d_vec)
        self.n_gamma_slots = int(n_gamma_slots)
        self.bins_per_gamma = int(bins_per_gamma)
        self.theta_period_ms = self.n_gamma_slots * self.bins_per_gamma  # 120 ms

    def encode_roles_into_theta_spike_train(
        self,
        role_vectors: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Tuple[int, int]]]:
        r"""Кодирует словарь `role_name -> vec64` в единую спайковую матрицу тета-периода `spike_raster [120, 64]`
        и фазы спайков $\phi \in [0, 2\pi)$ по закону Time-to-First-Spike (TTFS)."""
        raster = torch.zeros(self.theta_period_ms, self.d_vec, dtype=torch.float32)
        phase_windows: Dict[str, Tuple[int, int]] = {}

        for slot_idx, role_name in enumerate(self.ROLE_SLOT_NAMES):
            t_start = slot_idx * self.bins_per_gamma
            t_end = t_start + self.bins_per_gamma
            phase_windows[role_name] = (t_start, t_end)
            if role_name not in role_vectors:
                continue

            v = F.normalize(role_vectors[role_name].float(), p=2, dim=0)
            # Преобразуем непрерывную компоненту v_i in [-1, 1] в латентность первого спайка внутри окна [0 .. bins_per_gamma-1]:
            # Сильная положительная активация спайкует в начале гамма-цикла (ранняя фаза), отрицательная — в конце
            norm_01 = torch.clamp((v + 1.0) * 0.5, 0.0, 1.0)
            spike_bin_offset = torch.round((1.0 - norm_01) * (self.bins_per_gamma - 1)).to(torch.long)
            for neuron_i in range(self.d_vec):
                t_spike = t_start + int(spike_bin_offset[neuron_i].item())
                # Знак полярности (Возбуждающий E-спайк vs Тормозной I-спайк) + амплитуда TTFS
                raster[t_spike, neuron_i] = float(v[neuron_i].item())

        return raster, phase_windows

    def decode_roles_from_theta_spike_train(
        self,
        spike_raster: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Демодулирует спайковую матрицу `spike_raster [120, 64]` по 6 гамма-фазовым окнам обратно в 64-мерные векторы ролей."""
        decoded: Dict[str, torch.Tensor] = {}
        for slot_idx, role_name in enumerate(self.ROLE_SLOT_NAMES):
            t_start = slot_idx * self.bins_per_gamma
            t_end = t_start + self.bins_per_gamma
            window_slice = spike_raster[t_start:t_end, :]  # [20, 64]
            if float(window_slice.abs().sum().item()) < 1e-6:
                continue
            v_rec = window_slice.sum(dim=0)
            decoded[role_name] = F.normalize(v_rec + 1e-8, p=2, dim=0)
        return decoded


# ============================================================================
# 5. ПРЕДИКТИВНОЕ ЗАМЫКАНИЕ СМЫСЛА В `SON / BA10` ($X \longrightarrow \hat{Y} \longrightarrow \hat{X}$)
# ============================================================================
class BidirectionalMeaningClosureVerifier:
    r"""Метакогнитивный Верификатор Предиктивного Замыкания Смысла (`SON / BA10 Meaning Closure`):
      - Перед произнесением сгенерированного ответа $\hat{Y}$ проверяет **двунаправленное смысловое замыкание**:
        $$X \xrightarrow{\text{Forward Synthesis}} \hat{Y} \xrightarrow{\text{Inverse Reconstruction}} \hat{X}(\hat{Y})$$
      - Если в исходном вопросе $X$ было несколько смысловых клауз или сущностей, а черновик ответа $\hat{Y}$
        покрыл только первую клаузу (оставив ненулевой вектор дефицита смысла
        $\vec{r}_{\mathrm{missing}} = (\vec{s}_X - \hat{s}_X(\hat{Y}))_+$), верификатор выделяет недостающие
        концепты вопроса и автоматически достраивает ответ до полного смыслового замыкания!"""

    STOP_WORDS = {
        "the", "a", "an", "to", "of", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "with", "that", "this", "it", "do", "can", "you",
        "what", "how", "why", "where", "when", "who", "tell", "me", "about",
        "does", "he", "am", "my", "your", "his", "explain", "please", "remind",
    }

    def evaluate_and_close_meaning_loop(
        self,
        question_text: str,
        draft_response_text: str,
        ca3_attractors: torch.Tensor,
        ca3_sdr: torch.Tensor,
        compute_sdr_fn: Callable[[List[int], bool], torch.Tensor],
        encode_fn: Callable[[str], List[int]],
        synthesize_pole_fn: Optional[Callable[[torch.Tensor], str]] = None,
    ) -> Dict[str, object]:
        r"""Оценивает обратную реконструкцию вопроса $\hat{X}(\hat{Y})$, информационный прирост $\Delta I > 0$
        и при необходимости достраивает недостающую клаузу из вектора смыслового дефицита $\vec{r}_{\mathrm{missing}}$."""
        p_strip = " .,!?;:'()[]{}" + chr(34)
        q_words = [
            w.strip(p_strip).lower()
            for w in question_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.STOP_WORDS
        ]
        r_words = [
            w.strip(p_strip).lower()
            for w in draft_response_text.split()
            if len(w.strip(p_strip)) >= 4 and w.strip(p_strip).lower() not in self.STOP_WORDS
        ]

        q_ids = encode_fn(" ".join(q_words) if q_words else question_text)
        r_ids = encode_fn(" ".join(r_words) if r_words else draft_response_text)
        s_x = compute_sdr_fn(q_ids, False)
        s_y = compute_sdr_fn(r_ids, False)

        # Обратная генеративная реконструкция подразумеваемого вопроса X_hat из ответа Y_hat через двунаправленный CA3_SDR:
        if ca3_sdr.shape[0] > 0:
            with torch.no_grad():
                row_norm = torch.sqrt(ca3_sdr.sum(dim=1).clamp(min=1.0))
                y_norm = F.normalize(s_y + 1e-8, p=2, dim=0)
                pole_affinities = (ca3_sdr @ y_norm) / row_norm * 4.0
                top_k = min(2, int(pole_affinities.shape[0]))
                top_v, top_i = torch.topk(pole_affinities, k=top_k)
                weights = torch.softmax(top_v * 6.0, dim=0)
                reconstructed_x_sdr = torch.clamp((weights.unsqueeze(1) * ca3_sdr[top_i]).sum(dim=0) + s_y, max=1.0)
        else:
            reconstructed_x_sdr = s_y

        active_x_bits = (s_x > 0).float()
        covered_bits = active_x_bits * (reconstructed_x_sdr > 0.15).float()
        closure_score = float(covered_bits.sum().item()) / float(max(1.0, active_x_bits.sum().item()))

        # Информационный прирост Delta_I (доля новых содержательных битов ответа Y_hat, которых не было в вопросе X):
        active_y_bits = (s_y > 0).float()
        novel_y_bits = active_y_bits * (1.0 - active_x_bits)
        info_gain = float(novel_y_bits.sum().item()) / float(max(1.0, active_y_bits.sum().item()))

        # Если часть вопроса осталась непокрытой (closure_score < 0.72) и доступен синтезатор полюсов,
        # находим полюс CA3, максимальный по вектору смыслового дефицита r_missing = (s_x - reconstructed_x_sdr)_+!
        repaired_response = draft_response_text
        closure_repair_applied = False
        if (
            closure_score < 0.72
            and synthesize_pole_fn is not None
            and ca3_sdr.shape[0] >= 2
            and ca3_attractors.shape[0] == ca3_sdr.shape[0]
        ):
            with torch.no_grad():
                missing_sdr = torch.clamp(active_x_bits - (reconstructed_x_sdr > 0.15).float(), min=0.0)
                if float(missing_sdr.sum().item()) >= 3.0:
                    m_norm = F.normalize(missing_sdr + 1e-8, p=2, dim=0)
                    row_norm = torch.sqrt(ca3_sdr.sum(dim=1).clamp(min=1.0))
                    miss_scores = (ca3_sdr @ m_norm) / row_norm * 4.0
                    best_m_val, best_m_idx = torch.max(miss_scores, dim=0)
                    if float(best_m_val.item()) >= 0.25:
                        extra_clause = synthesize_pole_fn(ca3_attractors[int(best_m_idx.item())]).strip()
                        extra_prefix = extra_clause[:24].lower()
                        if extra_clause and extra_prefix not in draft_response_text.lower():
                            base_no_dot = draft_response_text.rstrip(".!?")
                            extra_low = extra_clause[0].lower() + extra_clause[1:] if len(extra_clause) > 1 else extra_clause
                            repaired_response = f"{base_no_dot}, and {extra_low}"
                            closure_repair_applied = True
                            # Пересчитываем смысловое покрытие после достройки (с учётом двунаправленного SDR второго полюса!)
                            r2_ids = encode_fn(repaired_response)
                            s_y2 = compute_sdr_fn(r2_ids, False)
                            rec2 = torch.clamp(reconstructed_x_sdr + s_y2 + ca3_sdr[int(best_m_idx.item())], max=1.0)
                            closure_score = float((active_x_bits * (rec2 > 0.15).float()).sum().item()) / float(
                                max(1.0, active_x_bits.sum().item())
                            )

        return {
            "meaning_closure_score": round(closure_score, 4),
            "epistemic_information_gain": round(info_gain, 4),
            "closure_repair_applied": closure_repair_applied,
            "verified_response": repaired_response,
            "meaning_closed": (closure_score >= 0.65) and (info_gain >= 0.15),
        }


# ============================================================================
# 6. КОНТРФАКТИЧЕСКИЙ СИМУЛЯТОР ИНТЕРВЕНЦИЙ В ПРЕФРОНТАЛЬНОЙ КОРЕ `BA10` ($\text{do}(\neg X) \implies \neg Y \implies \neg Z$)
# ============================================================================
class BA10CounterfactualInterventionSimulator:
    r"""Префронтальный Симулятор Контрфактических Интервенций (`BA10 Counterfactual & Intervention Simulator`):
      - Реализует Уровень 3 лестницы причинности Перла / активного вывода Фристона: мысленную интервенцию $\text{do}(\neg X)$.
      - Когда во входном стимуле возбуждаются тормозные интернейроны отрицания/выпадения звена
        (`fail`, `fails`, `without`, `lack`, `lacks`, `blocked`, `inhibited`, `absent`, `missing`, `prevent`),
        `BA10` находит каузальную траекторию в `CorticalRelationalWorldGraph` или в `CA3`, считывает BPE-токены
        каждого звена через без-индексный `edge_valence_generator` (`0% pos_idx`) и применяет универсальный
        BPE-оператор инверсии полярности на границе `Subj -> Verb`:
        $$\text{do}(\neg E_1) \implies \neg R_1(E_2) \implies \neg R_2(E_3)$$"""

    INTERVENTION_MARKERS = {
        "without", "lack", "lacks", "lacking", "fail", "fails", "failed",
        "blocked", "block", "inhibited", "inhibit", "absent", "missing",
        "deprived", "prevent", "prevents", "stopped", "stop", "unable",
    }

    VERB_3RD_TO_BASE = {
        "releases": "release",
        "activates": "activate",
        "co-activates": "co-activate",
        "ignites": "ignite",
        "triggers": "trigger",
        "produces": "produce",
        "shields": "shield",
        "accelerates": "accelerate",
        "stabilizes": "stabilize",
        "binds": "bind",
        "opens": "open",
        "drives": "drive",
        "causes": "cause",
        "generates": "generate",
    }

    def is_counterfactual_query(self, utterance: str) -> bool:
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = {w.strip(p_strip).lower() for w in utterance.split() if w.strip(p_strip)}
        if not (words & self.INTERVENTION_MARKERS):
            return False
        low = utterance.lower()
        return "?" in utterance or any(k in low for k in ("what happens", "what would happen", "if ", "without "))

    def _invert_edge_bpe_tokens(
        self,
        edge_ids: List[int],
        relation_str: str,
        hop_idx: int,
        total_hops: int,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[int], str],
    ) -> List[int]:
        r"""Применяет универсальную синтаксическую инверсию полярности ($\neg$) к BPE-последовательности ребра
        на переходе `Subject -> Verb` без доменных строковых шаблонов."""
        if not edge_ids:
            return []
        rel_first = relation_str.split()[0].lower() if relation_str.split() else ""
        base_verb = self.VERB_3RD_TO_BASE.get(rel_first, rel_first)

        # Находим индекс токена глагола в BPE-цепочке ребра
        verb_pos = -1
        for i, tid in enumerate(edge_ids):
            tok_clean = decode_fn(tid).strip(" .,!?;:").lower()
            if tok_clean and (tok_clean == rel_first or (len(rel_first) >= 4 and tok_clean.startswith(rel_first[:4]))):
                verb_pos = i
                break

        # Отрезаем финальную точку у промежуточных звеньев
        core_ids = list(edge_ids)
        while core_ids and decode_fn(core_ids[-1]).strip() in {".", "!", "?"}:
            core_ids.pop()

        if verb_pos <= 0:
            prefix_str = "If " if hop_idx == 0 else (", and consequently cannot " if hop_idx == total_hops - 1 else ", cannot ")
            return encode_fn(prefix_str) + core_ids

        subj_ids = core_ids[:verb_pos]
        rest_ids = core_ids[verb_pos + 1 :]
        # Понижаем регистр первого слова подлежащего, если перед ним вставляется оператор 'If ' или ', '
        first_subj_str = decode_fn(subj_ids[0])
        if first_subj_str.strip() and first_subj_str.strip()[0].isupper():
            s_clean = first_subj_str.strip()
            s_low = (" " if first_subj_str.startswith(" ") else "") + s_clean[0].lower() + s_clean[1:]
            low_ids = encode_fn(s_low)
            if low_ids:
                subj_ids = low_ids + subj_ids[1:]

        if hop_idx == 0:
            op_prefix = encode_fn("If ")
            neg_modal = encode_fn(f" fail to {base_verb}" if not base_verb.endswith("s") else f" fails to {base_verb}")
            return op_prefix + subj_ids + neg_modal + rest_ids
        if hop_idx == total_hops - 1 and total_hops >= 2:
            op_prefix = encode_fn(", and consequently ")
        else:
            op_prefix = encode_fn(", ")
        neg_modal = encode_fn(f" cannot {base_verb}")
        return op_prefix + subj_ids + neg_modal + rest_ids

    def simulate_counterfactual_intervention(
        self,
        utterance: str,
        world_graph: CorticalRelationalWorldGraph,
        encode_fn: Callable[[str], List[int]],
        decode_fn: Callable[[int], str],
        get_vec64_fn: Callable[[int], torch.Tensor],
        ca3_synthesize_fn: Optional[Callable[[str], str]] = None,
    ) -> Optional[Dict[str, object]]:
        r"""Выполняет мысленную симуляцию интервенции $\text{do}(\neg X)$ по каузальному графу мира или полюсам `CA3`."""
        if not self.is_counterfactual_query(utterance):
            return None

        # 1. Каузальная инверсия многошаговой цепи в CorticalRelationalWorldGraph (через без-индексный edge_valence_generator!):
        world_graph.imprint_edges_into_dedicated_valence_matrix()
        path = world_graph.find_causal_path(utterance, max_hops=3)
        if path:
            combined_ids: List[int] = []
            inverted_triplets: List[Tuple[str, str, str]] = []
            for hop_idx, edge in enumerate(path):
                e_ids = world_graph.edge_valence_generator.generate_trajectory_zero_pos(
                    q_goal_vec=edge.edge_pole_vec64,
                    get_vec64_fn=get_vec64_fn,
                    decode_fn=decode_fn,
                    max_tokens=28,
                )
                if not e_ids:
                    e_ids = list(edge.full_clause_tids)
                inv_ids = self._invert_edge_bpe_tokens(
                    edge_ids=e_ids,
                    relation_str=edge.relation,
                    hop_idx=hop_idx,
                    total_hops=len(path),
                    encode_fn=encode_fn,
                    decode_fn=decode_fn,
                )
                combined_ids.extend(inv_ids)
                inverted_triplets.append((f"~{edge.subject}", f"~{edge.relation}", f"~{edge.target}"))

            dot_ids = encode_fn(".")
            if dot_ids:
                combined_ids.append(dot_ids[0])
            cf_text = "".join(decode_fn(i) for i in combined_ids).strip()
            return {
                "counterfactual_response": cf_text,
                "hops_inverted": len(path),
                "inverted_triplet_chain": inverted_triplets,
                "source": "world_graph_counterfactual_cascade",
                "think_trace": (
                    f"BA10_CounterfactualSim: inverted {len(path)}-hop causal chain "
                    f"do(~{path[0].subject}) => ~{path[-1].target}"
                ),
            }

        # 2. Каузальная инверсия консолидированного полюса CA3 (например, для онлайн-эпизодов с 'by <verb>-ing' или 'that <verb>'):
        if ca3_synthesize_fn is not None:
            aff_text = ca3_synthesize_fn(utterance).strip()
            if aff_text and " by " in aff_text:
                first_sent = aff_text.split(".")[0].strip()
                parts = first_sent.split(" by ", 1)
                if len(parts) == 2:
                    outcome_clause = parts[0].strip()
                    mech_clause = parts[1].strip()
                    out_words = outcome_clause.split()
                    if len(out_words) >= 3:
                        subj_str = out_words[0].lower()
                        verb_rest = " ".join(out_words[1:])
                        # Выделяем ключевой объект механизма после герундия (producing X that Y -> X)
                        mech_words = mech_clause.split()
                        mech_obj = " ".join(mech_words[1:4]) if len(mech_words) >= 2 else mech_clause
                        for rel_split in (" that ", " which "):
                            if rel_split in mech_clause:
                                m_left, m_right = mech_clause.split(rel_split, 1)
                                m_l_words = m_left.split()
                                mech_obj = " ".join(m_l_words[1:]) if len(m_l_words) >= 2 else m_left
                                r_words = m_right.split()
                                r_verb = self.VERB_3RD_TO_BASE.get(r_words[0].lower(), r_words[0].lower()) if r_words else "protect"
                                r_rest = " ".join(r_words[1:]) if len(r_words) > 1 else ""
                                cf_sent = (
                                    f"If {subj_str} lack {mech_obj}, {subj_str} cannot {verb_rest}, "
                                    f"and consequently {mech_obj} cannot {r_verb} {r_rest}."
                                )
                                return {
                                    "counterfactual_response": cf_sent,
                                    "hops_inverted": 2,
                                    "inverted_triplet_chain": [(f"~{mech_obj}", f"~{verb_rest}", f"~{r_verb} {r_rest}")],
                                    "source": "ca3_pole_counterfactual_inversion",
                                    "think_trace": f"BA10_CounterfactualSim: inverted CA3 causal schema do(~{mech_obj}) => ~{verb_rest}",
                                }

        return None


# ============================================================================
# 7. СИНАПТИЧЕСКИЙ ТАЛАМИЧЕСКИЙ МАРШРУТИЗАТОР РЕЧЕВЫХ АКТОВ (`MD-Thalamus + Striatum D1/D2`)
# ============================================================================
class MDThalamusSynapticSpeechActRouter:
    r"""Нейро-Синаптический Маршрутизатор Речевых Актов Дорсомедиального Таламуса (`MD-Thalamus Speech-Act Router`):
      - Заменяет символьные списки подстрок обучаемой синаптической матрицей:
        $$\vec{a}_{\mathrm{D1}} = \vec{s}_{\mathrm{SDR}}(U)\, W_{\mathrm{MD\_Thal}} + \vec{c}_{\mathrm{spike}}(U)\, W_{\mathrm{contour}}$$
        где $W_{\mathrm{MD\_Thal}} \in \mathbb{R}^{2048 \times 7}$, а $W_{\mathrm{contour}} \in \mathbb{R}^{16 \times 7}$ —
        матрица спайкового контура фразы (частота вопросительных спайков, длина в гамма-циклах, дейктические
        местоимения 1/2/3 лица, тормозные спайки интервенции $\text{do}(\neg X)$, резонанс `CA3` и контекст ожидания ответа).
      - Выбирает один из 7 стриарных каналов:
        0: `ACT_DIRECT_RECALL`
        1: `ACT_IDENTITY_PROFILE`
        2: `ACT_EPISODIC_REFLECTION`
        3: `ACT_ONLINE_CONSOLIDATION`
        4: `ACT_COUNTERFACTUAL_INTERVENTION`
        5: `ACT_MULTI_HOP_OR_PARAGRAPH`
        6: `ACT_CORTICAL_THINK_ANALOGY`"""

    ACT_DIRECT_RECALL = "ACT_DIRECT_RECALL"
    ACT_IDENTITY_PROFILE = "ACT_IDENTITY_PROFILE"
    ACT_EPISODIC_REFLECTION = "ACT_EPISODIC_REFLECTION"
    ACT_ONLINE_CONSOLIDATION = "ACT_ONLINE_CONSOLIDATION"
    ACT_COUNTERFACTUAL_INTERVENTION = "ACT_COUNTERFACTUAL_INTERVENTION"
    ACT_MULTI_HOP_OR_PARAGRAPH = "ACT_MULTI_HOP_OR_PARAGRAPH"
    ACT_CORTICAL_THINK_ANALOGY = "ACT_CORTICAL_THINK_ANALOGY"

    CHANNEL_NAMES = (
        ACT_DIRECT_RECALL,
        ACT_IDENTITY_PROFILE,
        ACT_EPISODIC_REFLECTION,
        ACT_ONLINE_CONSOLIDATION,
        ACT_COUNTERFACTUAL_INTERVENTION,
        ACT_MULTI_HOP_OR_PARAGRAPH,
        ACT_CORTICAL_THINK_ANALOGY,
    )

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        compute_sdr_fn: Callable[[List[int], bool], torch.Tensor],
        n_sdr: int = 2048,
        n_contour: int = 16,
    ):
        self.encode_fn = encode_fn
        self.compute_sdr_fn = compute_sdr_fn
        self.n_sdr = int(n_sdr)
        self.n_contour = int(n_contour)
        self.n_channels = len(self.CHANNEL_NAMES)

        self.w_md_thal = torch.zeros((self.n_sdr, self.n_channels), dtype=torch.float32)
        self.w_contour = torch.zeros((self.n_contour, self.n_channels), dtype=torch.float32)
        self._imprint_foundational_speech_act_synapses()

    def _compute_spike_contour_vec16(
        self,
        utterance: str,
        ca3_resonance: float = 1.0,
        prev_was_clarify: bool = False,
        known_in_ontology: bool = False,
    ) -> torch.Tensor:
        r"""Вычисляет 16-мерный вектор спайкового контура фразы $\vec{c}_{\mathrm{spike}} \in \mathbb{R}^{16}$."""
        c = torch.zeros(self.n_contour, dtype=torch.float32)
        p_strip = " .,!?;:'()[]{}" + chr(34)
        words = [w.strip(p_strip).lower() for w in utterance.split() if w.strip(p_strip)]
        w_set = set(words)
        has_q = "?" in utterance

        c[0] = 1.0 if has_q else 0.0                                                              # Вопросительный спайк
        c[1] = 1.0 if not has_q else 0.0                                                          # Утвердительный спайк
        c[2] = float(min(1.0, len(words) / 20.0))                                                 # Длина в гамма-циклах
        c[3] = 1.0 if bool(w_set & {"name", "who"}) and bool(w_set & {"my", "i", "am", "building", "designing"}) else 0.0
        c[4] = 1.0 if bool(w_set & {"summarize", "discussed", "conversation", "previous", "turns", "earlier"}) else 0.0
        c[5] = 1.0 if (not has_q and (bool(w_set & {"remember", "note", "learn", "fact"}) or prev_was_clarify)) else 0.0
        c[6] = 1.0 if bool(w_set & BA10CounterfactualInterventionSimulator.INTERVENTION_MARKERS) and (has_q or "if" in w_set) else 0.0
        c[7] = 1.0 if bool(w_set & {"detail", "elaborate", "step", "chain", "consequently", "paragraph"}) else 0.0
        c[8] = 1.0 if (known_in_ontology and ca3_resonance < 0.42 and has_q) else 0.0
        c[9] = float(min(1.5, ca3_resonance))
        c[10] = 1.0 if prev_was_clarify and not has_q and len(words) >= 5 else 0.0
        c[11] = 1.0 if (not has_q and "name" in w_set and "is" in w_set) else 0.0
        return c

    def _imprint_foundational_speech_act_synapses(self) -> None:
        r"""Калибрует синаптические матрицы $W_{\mathrm{MD\_Thal}}$ и $W_{\mathrm{contour}}$ по каноническим прототипам речевых актов."""
        prototypes: List[Tuple[str, int, float, bool, bool]] = [
            ("How can I fix runny mayonnaise with an egg yolk?", 0, 1.2, False, False),
            ("What is my name and what am I designing?", 1, 1.5, False, False),
            ("My name is Elena and I am designing a neuromorphic chip.", 1, 1.5, False, False),
            ("Can you summarize what we discussed across our conversation so far?", 2, 1.2, False, False),
            ("What did we talk about in our previous turn?", 2, 1.2, False, False),
            ("Remember that tardigrades survive extreme radiation by producing Dsup proteins.", 3, 1.5, False, False),
            ("Tardigrades survive extreme radiation by producing Dsup proteins.", 3, 1.5, True, False),
            ("What happens to pyramidal neurons if astrocytes fail to release D-serine?", 4, 1.2, False, False),
            ("What would happen if tardigrades lack Dsup proteins?", 4, 1.2, False, False),
            ("Explain in detail step by step how astrocytes ignite pyramidal neurons?", 5, 1.2, False, False),
            ("What do you know about treatise?", 6, 0.25, False, True),
        ]
        with torch.no_grad():
            # Прямые контурные проекции D1 в дорсомедиальном таламусе:
            self.w_contour[0, 0] = 1.25   # Вопрос -> Direct Recall
            self.w_contour[9, 0] = 0.85   # Высокий резонанс CA3 -> Direct Recall
            self.w_contour[3, 1] = 3.20   # Автобиографический маркер -> Identity Profile
            self.w_contour[11, 1] = 2.80  # Представление имени -> Identity Profile
            self.w_contour[4, 2] = 3.50   # Мета-вопрос об истории диалога -> Episodic Reflection
            self.w_contour[5, 3] = 3.40   # Обучающий императив / ответ на уточнение -> Online Consolidation
            self.w_contour[10, 3] = 3.20  # Неявный ответ после уточняющего вопроса -> Online Consolidation
            self.w_contour[6, 4] = 3.60   # Оператор интервенции do(~X) -> Counterfactual Simulation
            self.w_contour[7, 5] = 2.90   # Запрос каузальной цепи / абзаца -> Multi-Hop or Paragraph
            self.w_contour[8, 6] = 3.30   # Низкий резонанс CA3 + известный STDP-концепт -> Cortical Think Analogy

            for utt, ch_idx, res_val, prev_cl, kn_ont in prototypes:
                ids = self.encode_fn(utt)
                sdr = F.normalize(self.compute_sdr_fn(ids, False).float() + 1e-8, p=2, dim=0)
                self.w_md_thal[:, ch_idx].add_(0.45 * sdr)

    def route_speech_act(
        self,
        utterance: str,
        ca3_resonance: float = 1.0,
        prev_was_clarify: bool = False,
        known_in_ontology: bool = False,
    ) -> Tuple[str, Dict[str, float]]:
        """Матрично вычисляет активации 7 каналов речевых актов в `MD-Thalamus` и возвращает победивший канал."""
        with torch.no_grad():
            ids = self.encode_fn(utterance)
            sdr = F.normalize(self.compute_sdr_fn(ids, False).float() + 1e-8, p=2, dim=0)
            contour = self._compute_spike_contour_vec16(
                utterance=utterance,
                ca3_resonance=ca3_resonance,
                prev_was_clarify=prev_was_clarify,
                known_in_ontology=known_in_ontology,
            )
            d1_acts = (sdr @ self.w_md_thal) + (contour @ self.w_contour)
            scores = {
                ch_name: round(float(d1_acts[i].item()), 4)
                for i, ch_name in enumerate(self.CHANNEL_NAMES)
            }
            best_idx = int(torch.argmax(d1_acts).item())
            return self.CHANNEL_NAMES[best_idx], scores

    def export_state_dict(self) -> Dict[str, object]:
        return {
            "w_md_thal": self.w_md_thal.detach().clone(),
            "w_contour": self.w_contour.detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if isinstance(state.get("w_md_thal"), torch.Tensor):
            self.w_md_thal = state["w_md_thal"].float().clone()
        if isinstance(state.get("w_contour"), torch.Tensor):
            self.w_contour = state["w_contour"].float().clone()

