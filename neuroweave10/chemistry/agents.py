"""Расширенный химический репертуар (Knowledge/chemical.md, 37 агентов).

Каждый агент — скалярный уровень с τ-динамикой и конкретным механизмом
действия в контуре организма (не декорация — полезная нагрузка):

  Гистамин (H1): блокирует sAHP → берстовая активность, бодрствование
  NPY (Y2): P_release↓ — анксиолитический тормоз, антисейзурный гейт
  Substance P (NK1): ноцицептивный гейн, ↓θ_LTP при стрессе
  Динорфин (KOR): дисфорический тормоз, пре-LTD DA/Glu
  Энкефалины (DOR/MOR): дизингибиция → облегчение LTP
  β-эндорфин (MOR): стресс-буфер, дизингибиция VTA
  CRH: θ_LTP↑ при хронике (стресс закрывает пластичность)
  Галанин: антисейзурный тормоз, ↓ACh-релиз
  Нейротенсин: D2-аллостерия, стриальный тюнинг
  VIP: уже в колонке (SST-дизингибиция)
  Окситоцин: SNR-усилитель сенсорики, PV-активация
  CO (BK): потолок частоты — гомеостатический тормоз
  ATP внеклеточный (P2X): сигнал тревоги/повреждения → LTD, микроглия
  Глицин/Аспартат: ко-агонисты NMDA (пермиссивный гейт)
  Глутамин-цикл: поддержание пресинаптического ресурса u_syn
  D-серин (GluN1): обязательный ко-агонист LTP (SRR-деплеция = нет LTP)
  Zn2+ (GluN2A): высокочастотный тормоз на mossy-синапсах
  E2 (GPER1): ↓V_th, расширение STDP-окна, плотность шипиков
  Тестостерон: w_max↑, пространственная стабилизация
  ДГЭА (σ1R): NMDA-PAM, когнитивный стимулянт
  AEA (CB1 частичный, TRPV1): тонический базлайн настроения, TRPV1-LTD
  Аллопрегнанолон: GABA_A PAM (тоническое торможение, сон/анксиолиз)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgentLevels:
    """Концентрации расширенного пула (условные единицы 0..2)."""
    histamine: float = 0.4      # H1 → ↓sAHP
    npy: float = 0.2            # Y2 → ↓P_release
    substance_p: float = 0.1
    dynorphin: float = 0.1      # KOR → пре-LTD
    enkephalin: float = 0.2     # DOR/MOR → дизингибиция
    beta_endorphin: float = 0.1
    crh: float = 0.1            # θ_LTP↑
    galanin: float = 0.15
    neurotensin: float = 0.2
    oxytocin: float = 0.2       # SNR↑
    co: float = 0.3             # BK → потолок частоты
    atp_ext: float = 0.5        # P2X
    glycine: float = 0.5        # ко-агонист NMDA
    d_serine: float = 0.8       # GluN1, обязателен для LTP
    zn: float = 0.3             # GluN2A тормоз
    glutamine: float = 0.9      # ресурсный цикл
    e2: float = 0.3             # эстрадиол
    testosterone: float = 0.3
    dhea: float = 0.3           # NMDA-PAM
    aea: float = 0.4            # тонический CB1/TRPV1
    allo: float = 0.2           # GABA_A PAM
    bdnf: float = 0.35          # TrkB нейротрофин: структурный спрутинг шипиков и нейрогенез DG


class ExtendedChemistry:
    """Медленный пул агентов + их эффектные функции для организма."""

    def __init__(self):
        self.l = AgentLevels()
        # Мембранная (m_X) и геномная (g_X) фазы гормонов (§3.4)
        self.m_e2, self.g_e2 = 0.3, 0.3
        self.m_cort, self.g_cort = 0.1, 0.1
        self.m_test, self.g_test = 0.3, 0.3

    # ------------------------------------------------------------------
    def update(self, spikes: float, stress: float, arousal: float,
               social: float, threat: float, damage: float,
               dt_ms: float = 1.0) -> AgentLevels:
        """Медленный дрейв концентраций к контекстным сетпоинтам."""
        l, d = self.l, dt_ms / 1000.0
        def track(x, target, tau_s):
            return x + (-(x - target) * d / tau_s)
        l.histamine = track(l.histamine, 0.2 + 0.6 * arousal, 60.0)
        l.npy = track(l.npy, 0.2 + 0.5 * stress, 30.0)
        l.substance_p = track(l.substance_p, 0.1 + 0.7 * threat, 15.0)
        l.dynorphin = track(l.dynorphin, 0.1 + 0.6 * stress, 120.0)
        l.enkephalin = track(l.enkephalin, 0.2 + 0.3 * social, 60.0)
        l.beta_endorphin = track(l.beta_endorphin, 0.1 + 0.5 * stress, 300.0)
        l.crh = track(l.crh, 0.1 + 0.9 * stress, 60.0)
        l.galanin = track(l.galanin, 0.15 + 0.3 * spikes, 50.0)
        l.neurotensin = track(l.neurotensin, 0.2 + 0.3 * arousal, 45.0)
        l.oxytocin = track(l.oxytocin, 0.2 + 0.6 * social, 120.0)
        l.co = track(l.co, 0.3 + 0.4 * spikes, 30.0)
        l.atp_ext = track(l.atp_ext, 0.5 + 1.0 * damage + 0.2 * spikes, 5.0)
        l.d_serine = track(l.d_serine, 0.8 + 0.1 * spikes, 120.0)
        l.zn = track(l.zn, 0.3 + 0.5 * spikes, 0.5)
        l.glutamine = track(l.glutamine, 0.9 - 0.3 * spikes, 30.0)
        l.e2 = track(l.e2, 0.3 + 0.2 * social, 600.0)
        l.testosterone = track(l.testosterone, 0.3 + 0.1 * threat, 3600.0)
        l.dhea = track(l.dhea, 0.3 + 0.2 * arousal, 300.0)
        l.allo = track(l.allo, 0.2 + 0.4 * stress + 0.3 * (1 - arousal), 60.0)
        # BDNF (TrkB): умеренный тонус в спокойном бодрствовании (~0.45-0.55) и мощный выброс
        # при высокой когнитивной новизне/ошибке предсказания (damage > 0.18, arousal > 0.55),
        # подавляемый хроническим стрессом (кортизолом/CRH)
        bdnf_target = max(
            0.15,
            min(
                1.5,
                0.35
                + 0.85 * max(0.0, arousal - 0.55)
                + 1.35 * max(0.0, damage - 0.18)
                + 0.30 * spikes
                - 0.45 * stress,
            ),
        )
        l.bdnf = float(0.88 * l.bdnf + 0.12 * bdnf_target)
        # Двухмодовые гормоны: мембранная (τ~100с) и геномная (τ~часы)
        self.m_e2 = track(self.m_e2, l.e2, 100.0)
        self.g_e2 = track(self.g_e2, l.e2, 36000.0)
        self.m_test = track(self.m_test, l.testosterone, 100.0)
        self.g_test = track(self.g_test, l.testosterone, 36000.0)
        return l

    # -- Эффектные функции (подключены к контуру) ------------------------
    def nmda_gain(self) -> float:
        """Φ_DSer·Ψ_Zn·(1+δ_H2S)·(1+ζ_DHEA) — ко-агонистный гейт NMDA."""
        l = self.l
        phi_dser = l.d_serine / (l.d_serine + 0.5)
        phi_gly = l.glycine / (l.glycine + 0.4)
        psi_zn = 1.0 / (1.0 + (l.zn / 0.2) ** 2)
        return float(phi_dser * phi_gly * psi_zn * (1.0 + 0.3 * l.dhea))

    def p_release_scale(self) -> float:
        """Мультипликативный масштаб P_release: NPY, Dyn, Enk, AEA."""
        l = self.l
        return float((1.0 / (1.0 + (l.npy / 0.3) ** 2))
                     * (1.0 - l.dynorphin / (l.dynorphin + 0.5))
                     * (1.0 / (1.0 + (l.aea / 0.6) ** 2))
                     * (1.0 / (1.0 + (l.galanin / 0.4) ** 2)))

    def theta_ltp_shift(self) -> float:
        """θ_LTP = θ0(1+α1·GR−α2·MR)+α_CRH·CRH — метапластический сдвиг."""
        l = self.l
        return float(0.5 * l.crh - 0.3 * l.e2)

    def disinhibition(self) -> float:
        """Энкефалин/β-эндорфин/VIP-путь: снижение GABA-силы."""
        l = self.l
        return float(1.0 / (1.0 + (l.enkephalin / 0.3) ** 2)
                     * 1.0 / (1.0 + (l.beta_endorphin / 0.3) ** 2))

    def ahp_block(self) -> float:
        """H1-гистамин: доля заблокированного sAHP (0..1)."""
        return float(self.l.histamine / (self.l.histamine + 0.4))

    def snr_boost(self) -> float:
        """Окситоцин: SNR-усилитель сенсорного входа."""
        return float(1.0 + 0.5 * self.l.oxytocin / (self.l.oxytocin + 0.3))

    def rate_ceiling(self) -> float:
        """CO→BK_Ca: мягкий потолок спайковой частоты."""
        return float(1.0 + 0.8 * self.l.co / (self.l.co + 0.5))

    def vth_shift(self) -> float:
        """E2 мембранная фаза: сдвиг порога спайка вниз (мВ)."""
        return float(2.0 * self.m_e2)

    def gaba_tonic(self) -> float:
        """ALLO: тоническое GABA_A PAM — гомеостатический тормоз."""
        return float(1.0 + 1.5 * self.l.allo)

    def d1_d2_profile(self, da_level: float) -> Dict[str, float]:
        """Инновация 6 (Seamans & Yang 2004; Arnsten 2011):
        Рецепторная гетерогенность дофамина D1 (Gs, инвертированная U-кривая
        стабилизации рабочей памяти и NMDA в PFC) vs D2 (Gi, снижение барьера
        переключения аттракторов и когнитивная гибкость + модуляция нейротензином)."""
        da = max(0.0, float(da_level))
        # D1: оптимум вокруг da ≈ 0.65 (инвертированная U-кривая Арнстен)
        d1_act = float(math.exp(-((da - 0.65) / 0.45) ** 2))
        # D2: активируется при высоком фазическом DA или сдвиге нейротензина
        nt_mod = 1.0 / (1.0 + 0.4 * self.l.neurotensin)
        d2_act = float((1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (da - 0.85) * 4.0))))) * nt_mod)
        # При резком падении DA ниже базлайна (<0.25) D2 снимает блок переключения
        if da < 0.25:
            d2_act = max(d2_act, float((0.25 - da) * 2.5))
        return {
            "d1_nmda_boost": 1.0 + 0.45 * d1_act,
            "d2_flexibility": float(min(1.0, d2_act)),
            "d1_act": d1_act,
            "d2_act": d2_act,
        }

    def ne_receptor_profile(self, ne_level: float,
                            surprise: float = 0.0) -> Dict[str, float]:
        """Инновация 6 (Bouret & Sara 2005; Arnsten 2011):
        Умеренный NE через высокоаффинные α2A-рецепторы укрепляет когерентность PFC,
        а высокий фазический NE / мета-сюрприз через низкоаффинные α1/β-рецепторы
        запускает глобальный Network Reset (сброс застрявшего контекста при смене правил)."""
        ne = max(0.0, float(ne_level))
        surp = max(0.0, float(surprise))
        alpha2a = float(ne / (ne + 0.35) * math.exp(-max(0.0, ne - 0.9) * 1.5))
        reset_drive = max(0.0, (ne - 0.95) * 0.8 + max(0.0, surp - 0.65) * 1.4)
        network_reset = float(min(1.0, reset_drive))
        return {
            "alpha2a_tonus": alpha2a,
            "network_reset": network_reset,
        }

    def glutamine_replenish_rate(self) -> float:
        """Глутамин-глутаматный астроцитарный цикл: скорость восстановления везикул u_syn."""
        return float(0.6 + 0.6 * (self.l.glutamine / (self.l.glutamine + 0.4)))

    def microglial_prune_boost(self) -> float:
        """Внеклеточный АТФ (P2X7 рецепторы микроглии): сигнал повреждения/ошибки
        усиливает очистку слабых шумовых синапсов."""
        return float(1.0 + 0.5 * max(0.0, self.l.atp_ext - 0.5))

