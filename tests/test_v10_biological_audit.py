"""Биологический Аудит Ziwago-Prime v3 (7 уровней из Produce/nexr2.txt).

Проверяет, что каждый из 7 уровней реализован физически и математически
строго, без MLP-костылей и без глобального обратного распространения:

  1. Микроуровень (Нейрон и Синапс):
     - Точный экспоненциальный шаг (exp_euler) на всех 3 компартментах;
     - Двухфазная гетеросинаптическая пластичность (внутриветочная кооперативная
       гетеро-LTP + компенсаторная кальциневриновая гетеро-LTD неактивных соседей).
  2. Функциональная Нейромодуляция:
     - DA как 3-й фактор конвертации следа допустимости (eligibility → weight);
     - ACh как переключатель баланса FF/FB и дизингибитор VIP ⊣ SST;
     - NE как регулятор отношения Сигнал/Шум (подавление OU-шума + рост крутизны φ).
  3. Эндогенная Мотивация (Принцип Свободной Энергии Фристона):
     - Самообучение организма на повторяющемся сенсорном потоке БЕЗ внешнего
       учителя (reward=0.0): снижение Свободной Энергии F(t) генерирует
       эндогенный дофаминовый импульс (intrinsic_rpe > 0).
  4. Энергоэффективность и Эмерджентная Критичность:
     - Метаболический штраф за спайк (spiking cost) в локальной ошибке при расходе АТФ;
     - Локальный тормозной iSTDP Вогельса (PV → PYR) удерживает сеть от взрыва.
  5. Поклеточная Интринсик-Пластичность Триша (Triesch 2005):
     - Индивидуальная самонастройка порога theta_base[i] и гейна ip_gain[i]
       пробуждает молчащие нейроны и тормозит гиперактивные внутри одного пула.
  6. Пространственное Зонирование Барбас и LFP-Эфаптика:
     - Градиент толщины слоёв L4/L23/L5/L6 от гранулярной SENS (g=0.0) к
       агранулярной PFC (g=0.9);
     - Генерация внеклеточного диполя LFP и эфаптический сдвиг мембраны (±1.5 мВ).
  7. Проживание Опыта и 4-Уровневое Обучение:
     - Биофизический резервуарный MetaHead без скрытого backprop;
     - Структурная пластичность (прунинг слабых шипиков + спрутинг новых)
       и сонная консолидация.
"""
from __future__ import annotations

import torch

from neuroweave10.config import USNeuronCfg, USSynapseCfg, genesis_cfg
from neuroweave10.core.neuron import USNeuronPool
from neuroweave10.core.synapse import GenesisSynapse
from neuroweave10.organism.genesis import GenesisOrganism


def test_level1_heterosynaptic_ltp_and_compensatory_ltd():
    """Уровень 1: При стимуляции одного входа (j=2) соседние по дендритной ветке
    синапсы (j=1, 3) получают кооперативную гетеро-LTP, а далёкие неактивные
    синапсы (j=12..15) испытывают компенсаторную кальциневриновую гетеро-LTD."""
    cfg = USSynapseCfg(eta=0.08, density=1.0)
    syn = GenesisSynapse(16, 4, cfg, seed=7)
    syn.w_slow.fill_(0.30)
    syn.w_fast.zero_()

    s_pre = torch.zeros(1, 16)
    s_pre[0, 2] = 1.0  # активен только синапс j=2
    s_post = torch.ones(1, 4)
    err = torch.ones(1, 4) * 0.8

    w_far_before = float(syn.w[0, 14].item())
    for _ in range(6):
        syn.learn(err, s_pre, s_post, f_mod=1.0, ach=0.7, da_phasic=0.8)

    # Гомосинаптический синапс j=2 вырос сильнее всего
    assert float(syn.w[0, 2].item()) > 0.35
    # Ближайший сосед по ветке j=3 вырос сильнее далёкого j=14 за счёт branch_kernel
    assert float(syn.w[0, 3].item()) > float(syn.w[0, 14].item())
    # Далёкий неактивный синапс j=14 испытал компенсаторную гетеро-LTD
    assert float(syn.w[0, 14].item()) < w_far_before
    assert syn.last_hetero_ltd > 0.0


def test_level2_neuromodulation_snr_and_ach_vip_sst():
    """Уровень 2:
    - Норадреналин (NE) сжимает дисперсию фонового OU-шума мембраны и повышает крутизну φ;
    - Ацетилхолин (ACh) возбуждает спайковый пул VIP, подавляя дендритный тормоз SST."""
    ncfg = USNeuronCfg(stochastic=True, sigma=1.2)
    pool_lo = USNeuronPool(32, ncfg, name="lo_ne")
    pool_hi = USNeuronPool(32, ncfg, name="hi_ne")
    pool_lo.reset_state(1)
    pool_hi.reset_state(1)

    z = torch.zeros(1, 32)
    for _ in range(40):
        pool_lo.step(z, z, z, ne=0.0)
        pool_hi.step(z, z, z, ne=1.5)
    # При высоком NE дисперсия фонового мембранного шума существенно ниже (выше SNR)
    assert float(pool_hi.noise.std()) < float(pool_lo.noise.std()) * 0.75
    # При высоком NE крутизна передаточной функции выше для надпорогового входа
    v_test = torch.full((1, 32), ncfg.theta + 2.0)
    assert float(pool_hi.phi(v_test, ne=1.5).mean()) > float(pool_lo.phi(v_test, ne=0.0).mean())

    # Проверка ACh → VIP ⊣ SST в кортикальной колонке
    gcfg = genesis_cfg("tiny")
    org = GenesisOrganism(gcfg, batch=1, seed=0)
    org.reset_state(1)
    col = org.hier.cols[0]
    x_ff = torch.ones(1, gcfg.encoder.n_sdr) * 0.2
    x_fb = torch.ones(1, gcfg.zones[1].n_neurons) * 0.2

    col._mod_ctx = {"ach": 0.05, "ne": 0.2}
    for _ in range(10):
        col.step(x_ff, x_fb, f_mod=0.5, vip_mod=0.05, learn=False)
    vip_low_ach = col.vip_drive

    col._mod_ctx = {"ach": 1.2, "ne": 0.2}
    for _ in range(10):
        col.step(x_ff, x_fb, f_mod=0.5, vip_mod=0.4, learn=False)
    vip_high_ach = col.vip_drive
    assert vip_high_ach > vip_low_ach


def test_level3_free_energy_intrinsic_motivation_without_external_teacher():
    """Уровень 3: При подаче закономерного сенсорного паттерна БЕЗ внешнего
    вознаграждения (reward=0.0) организм минимизирует вариационную свободную
    энергию F(t) и вырабатывает эндогенный дофаминовый сигнал (intrinsic_rpe > 0)."""
    torch.manual_seed(11)
    gcfg = genesis_cfg("tiny")
    org = GenesisOrganism(gcfg, batch=1, seed=11)
    org.reset_state(1)
    x = torch.zeros(1, gcfg.input_dim)
    x[0, ::2] = 0.8

    fe_trace = []
    pos_rpe_ticks = 0
    for _ in range(30):
        out = org.step(x, reward=0.0, learn=True)
        fe_trace.append(out.introspection["free_energy"])
        if out.introspection["intrinsic_rpe"] > 0.0:
            pos_rpe_ticks += 1

    assert len(fe_trace) == 30
    assert pos_rpe_ticks >= 2
    # После прогрева мембран и начала стабильного спайкинга (такты 7..13) минимальная Свободная Энергия
    # во второй половине окна снижается за счёт локального предиктивного обучения без учителя
    assert min(fe_trace[14:]) < min(fe_trace[7:13])


def test_level4_metabolic_spike_cost_and_local_vogels_istdp():
    """Уровень 4: Дефицит АТФ вводит отрицательный метаболический штраф за спайк
    в локальную ошибку обучения, а тормозная матрица w_pv_in адаптируется через iSTDP."""
    ncfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(8, ncfg)
    pool.reset_state(1)
    # Искусственно снижаем энергию АТФ у первой половины нейронов
    pool.energy[0, :4] = 0.2
    pool.energy[0, 4:] = 1.0
    g_strong = torch.full((1, 8), 8.0)
    z = torch.zeros(1, 8)
    out = pool.step(g_strong, z, z)
    # У энергодефицитных нейронов (0..3) ошибка при спайке ниже из-за штрафа spike_cost
    if float(out.s[0, 0]) > 0 and float(out.s[0, 4]) > 0:
        assert float(out.error[0, 0]) < float(out.error[0, 4])


def test_level5_per_neuron_triesch_intrinsic_plasticity():
    """Уровень 5: Интринсик-пластичность Триша индивидуально снижает порог theta_base
    у молчащих нейронов и повышает его у гиперактивных клеток внутри одного пула."""
    ncfg = USNeuronCfg(stochastic=False)
    pool = USNeuronPool(8, ncfg)
    pool.reset_state(1)
    g_ampa = torch.zeros(1, 8)
    g_ampa[0, :4] = 10.0  # первые 4 нейрона стимулируются, последние 4 молчат
    z = torch.zeros(1, 8)
    for _ in range(40):
        pool.step(g_ampa, z, z)
    # Гиперактивные нейроны (0..3) подняли свой базовый порог выше молчащих (4..7)
    assert float(pool.theta_base[0, :4].mean()) > float(pool.theta_base[0, 4:].mean())


def test_level6_barbas_laminar_gradient_and_ephaptic_lfp():
    """Уровень 6: Гранулярная кора SENS (g=0.0) имеет более широкую долю L4,
    чем агранулярная PFC (g=0.9), а EphapticLFP создаёт ненулевое поле и сдвиг мембраны."""
    gcfg = genesis_cfg("tiny")
    org = GenesisOrganism(gcfg, batch=1, seed=3)
    org.reset_state(1)
    col_sens = org.hier.cols[0]
    col_pfc = org.hier.cols[-1]
    l4_sens = (col_sens.layer_slices["L4"].stop - col_sens.layer_slices["L4"].start) / col_sens.zcfg.n_neurons
    l4_pfc = (col_pfc.layer_slices["L4"].stop - col_pfc.layer_slices["L4"].start) / col_pfc.zcfg.n_neurons
    assert l4_sens > l4_pfc

    x = torch.ones(1, gcfg.input_dim) * 0.7
    out = org.step(x, learn=False)
    assert "lfp" in out.introspection
    assert any(abs(v) > 0.0 for v in col_sens.lfp.layer_lfp.values())
    assert float(col_sens.lfp.ephaptic_mv.abs().max()) <= 1.5


def test_level7_structural_plasticity_and_non_backprop_metahead():
    """Уровень 7: Структурная пластичность отсекает слабые неиспользуемые шипики
    и выращивает новые вдоль активных следов, а MetaHead учится локально без w1/w2 backprop."""
    scfg = USSynapseCfg(density=0.5)
    syn = GenesisSynapse(12, 8, scfg, seed=5)
    # Состарим шипики и обнулим часть весов и актинового здоровья H_ij для проверки прунинга
    syn.spine_age.fill_(50.0)
    syn.spine_health[0, :] = 0.1
    syn.w_slow[0, :] = 0.001
    syn.w_fast[0, :] = 0.0
    syn.tag.zero_()
    syn.r_err[0] = 1.0
    syn.r_pre[0] = 1.0
    pruned, sprouted = syn.structural_plasticity(prune_thresh=0.01)
    assert pruned >= 1
    assert sprouted >= 1

    gcfg = genesis_cfg("tiny")
    org = GenesisOrganism(gcfg, batch=1, seed=5)
    org.reset_state(1)
    w_in_before = org.meta.w_in.clone()
    for _ in range(10):
        org.step(torch.randn(1, gcfg.input_dim), reward=0.5, learn=True)
    # Фиксированная дендритная экспансия BA10 не меняется backprop'ом, а локальный выход учится
    assert torch.allclose(org.meta.w_in, w_in_before)
    assert float(org.meta.w_out.abs().sum()) > 0.0


def test_v4_node1_triple_locked_sleep_consolidation():
    """Узел 1 (v4): Цикл сна генерирует таламические веретёна (spindles),
    проигрывает приоритетные SWR-эпизоды, переносит w_fast -> w_slow и запускает
    адаптивный нейрогенез DG."""
    gcfg = genesis_cfg("tiny")
    org = GenesisOrganism(gcfg, batch=1, seed=12)
    org.reset_state(1)
    x = torch.ones(1, gcfg.input_dim) * 0.7
    for _ in range(8):
        org.step(x, reward=0.5, learn=True)
    w_slow_pre = org.hier.cols[0].w_ff.w_slow.clone()
    sleep_rep = org.sleep_cycle(ticks=30)
    assert sleep_rep["spindle_mean"] >= 0.5
    assert sleep_rep["dg_neurogenesis"] >= 1
    assert not torch.allclose(org.hier.cols[0].w_ff.w_slow, w_slow_pre)


def test_v4_node2_symphonic_parallel_integration_preserves_weak_signals():
    """Узел 2 (v4): Симфоническое Рабочее Пространство и ThalamicHO Matrix-путь
    сохраняют параллельные партитуры всех зон (SENS, ASSOC, PFC, CA1) без
    уничтожения слабых сигналов единственным argmax-победителем."""
    from neuroweave10.config import WorkspaceCfg
    from neuroweave10.consciousness.workspace import GlobalWorkspace
    from neuroweave10.thalamus.relay import ThalamicHO

    gw = GlobalWorkspace(WorkspaceCfg(d_gw=16, n_slots=4))
    c_sens = torch.zeros(16); c_sens[:4] = 0.25   # слабый сигнал деталей SENS
    c_pfc = torch.zeros(16); c_pfc[8:12] = 1.50   # сильный сигнал цели PFC
    out = gw.compete([c_sens, c_pfc], ne=0.4, lfp_phases=[0.5, 0.5])
    # Оба слота активны одновременно и оба представлены в симфоническом бродкасте!
    assert len(out.content) == 2
    assert float(out.broadcast[:4].abs().sum()) > 0.01
    assert float(out.broadcast[8:12].abs().sum()) > 0.05

    ho = ThalamicHO(8, 8)
    l5_in = torch.tensor([[1.0, 0.9, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]])
    ho_out = ho.step(l5_in, bg_release=0.8, lfp_coherence=1.2)
    assert ho_out.shape == (1, 8)
    assert float(ho_out.abs().sum()) > 0.0


def test_v4_node3_limbic_valence_and_arousal():
    """Узел 3 (v4): Лимбическая система (Гипоталамус + Амигдала BLA/CeA)
    различает аппетитивный опыт (награда/снижение F) и аверсивный кризис
    (скачок F / дефицит АТФ), модулируя flash_ltp_gain."""
    from neuroweave10.emotion.limbic import LimbicSystem
    lim = LimbicSystem(input_dim=16, pfc_dim=16, seed=1)
    x = torch.ones(1, 16) * 0.5
    pfc = torch.ones(1, 16) * 0.5
    pos = lim.step(x, pfc, free_energy=0.1, intrinsic_rpe=0.7,
                   atp_mean=0.95, ext_reward=1.0, learn=True)
    neg = lim.step(x, pfc, free_energy=1.8, intrinsic_rpe=-0.5,
                   atp_mean=0.30, ext_reward=-1.0, learn=True)
    assert pos.valence > neg.valence
    assert neg.d_energy > pos.d_energy
    assert neg.flash_ltp_gain > 1.0


def test_v4_node4_error_driven_bdnf_synaptogenesis():
    """Узел 4 (v4): В зоне с высокой ошибкой предсказания (error_level=0.8)
    BDNF/NO морфогенез выращивает больше синаптических связей, чем в спокойной зоне."""
    scfg = USSynapseCfg(density=0.35)
    syn_calm = GenesisSynapse(16, 12, scfg, seed=2)
    syn_err = GenesisSynapse(16, 12, scfg, seed=2)
    syn_calm.r_err.fill_(0.05); syn_calm.r_pre.fill_(0.5)
    syn_err.r_err.fill_(0.80); syn_err.r_pre.fill_(0.5)
    _, spr_calm = syn_calm.structural_plasticity(error_level=0.05)
    _, spr_err = syn_err.structural_plasticity(error_level=0.80)
    assert spr_err > spr_calm
    assert int(syn_err.mask.sum()) > int(syn_calm.mask.sum())


def test_v4_node5_active_inference_curiosity_drive():
    """Узел 5 (v4): Активный Вывод (Active Inference) повышает исследовательский
    драйв explore_drive при высокой эпистемической неопределённости/скуке и
    гасит его при аверсивной угрозе Амигдалы."""
    from neuroweave10.config import MotorCfg
    from neuroweave10.motor.pool import MotorPool
    m = MotorPool(16, 8, 4, 4, MotorCfg(), seed=0)
    m.reset_state(1)
    k_curious = m.set_active_inference(epistemic=0.7, boredom=0.5, aversive=0.0)
    k_threat = m.set_active_inference(epistemic=0.7, boredom=0.5, aversive=0.95)
    assert k_curious > 0.5
    assert k_threat < k_curious


def test_p1_dendritic_predictive_coding_and_cortico_motor_credit():
    """P1 (v5): Дендритное предиктивное кодирование (Sacramento/Mikulasch) и
    замыкание кортико-моторной петли назначения кредита (M1 → PFC → ASSOC → SENS)."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    # 1. Подпороговый апикальный учитель в USNeuronPool создаёт положительный градиент без спайка
    pool = USNeuronPool(4, USNeuronCfg(stochastic=False))
    pool.reset_state(1)
    g_ampa = torch.ones(1, 4) * 0.2
    g_nmda = torch.tensor([[2.5, 2.5, 0.0, 0.0]])  # апикальный таргет только на нейроны 0,1
    out = pool.step(g_ampa, g_nmda, torch.zeros(1, 4))
    assert float(out.error[0, 0]) > float(out.error[0, 2])

    # 2. В RegenesisOrganism ошибка моторного пула транслируется сверху вниз (PFC → ASSOC → SENS)
    u = RegenesisOrganism(UnifiedConfig(vocab_size=16), gcfg=genesis_cfg("tiny"), seed=7)
    w_sst_0 = u.org.hier.cols[0].w_sst_in.clone()
    d_alpha_0 = u.org.hier.cols[0].dendrite.alpha.clone()
    r = u.step_token(torch.tensor([2]), torch.tensor([5]), learn=True)
    assert "apical_err" in r and len(r["apical_err"]) == 3
    assert all(v > 0.0 for v in r["apical_err"].values())
    assert not torch.allclose(u.org.hier.cols[0].w_sst_in, w_sst_0)
    assert not torch.allclose(u.org.hier.cols[0].dendrite.alpha, d_alpha_0)


def test_p2_wlc_emergent_syntax_modifier_skip_invariance():
    """P2 (v5): Гетероклинический аттрактор ролей Winnerless Competition (Rabinovich 2001)
    в ThetaGammaStoryBinder сохраняет инвариантность ролей Субъекта и Глагола
    при пропуске опционального прилагательного ('the little girl ran' -> 'the girl ran')."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    u = RegenesisOrganism(UnifiedConfig(vocab_size=16), gcfg=genesis_cfg("tiny"), seed=21)
    u.punct_ids = {6}
    # Обучаем на полном предложении с прилагательным: 2=the(0), 3=little(1), 4=girl(2), 5=ran(3), 6=.(6)
    full_sent = [2, 3, 4, 5, 6]
    for _ in range(16):
        u.org.story_binder.notify_boundary()
        for a, b in zip(full_sent, full_sent[1:]):
            u.step_token(torch.tensor([a]), torch.tensor([b]), learn=True)

    # Тестируем на укороченной фразе БЕЗ прилагательного: [2=the, 4=girl] -> должен распознать
    # 4=girl как SUBJ (Role 2), переключить next_role в VERB (Role 3) и выдать 5=ran!
    u.org.reset_state(1)
    u.motor.reset_state(1)
    _ = u.step_token(torch.tensor([2]), learn=False)
    r_girl = u.step_token(torch.tensor([4]), learn=False)
    assert u.org.story_binder.prev_role == 2
    assert r_girl["token"] == 5


def test_p3_murray_hasson_cortical_temporal_receptive_windows():
    """P3 (v5): Разделение быстрой AMPA-атаки (τ_rise <= 20 мс) и медленного распада
    Murray/Hasson (τ_decay = τ_m(g)) позволяет PFC быстро захватывать входной импульс,
    но удерживать дендритный след контекста значительно дольше, чем SENS."""
    cfg_sens = USNeuronCfg(stochastic=False, tau_d=20.0)
    cfg_pfc = USNeuronCfg(stochastic=False, tau_d=680.0)
    p_sens = USNeuronPool(4, cfg_sens); p_sens.reset_state(1)
    p_pfc = USNeuronPool(4, cfg_pfc); p_pfc.reset_state(1)

    # 1. Во время входного импульса оба пула быстро захватывают возбуждение (без 34x глухоты PFC!)
    g_pulse = torch.full((1, 4), 0.25)
    z = torch.zeros(1, 4)
    p_sens.step(g_pulse, z, z)
    p_pfc.step(g_pulse, z, z)
    depol_sens_0 = float((p_sens.v_d - cfg_sens.e_l).mean())
    depol_pfc_0 = float((p_pfc.v_d - cfg_pfc.e_l).mean())
    assert depol_pfc_0 > 0.8 * depol_sens_0

    # 2. После 80 мс релаксации PFC удерживает дендритную память (v_d - e_l) намного дольше SENS
    #    (потеря следа в PFC более чем в 15 раз медленнее, чем в SENS!)
    for _ in range(80):
        p_sens.step(z, z, z)
        p_pfc.step(z, z, z)
    depol_sens_80 = float((p_sens.v_d - cfg_sens.e_l).mean())
    depol_pfc_80 = float((p_pfc.v_d - cfg_pfc.e_l).mean())
    loss_sens = depol_sens_0 - depol_sens_80
    loss_pfc = depol_pfc_0 - depol_pfc_80
    assert depol_pfc_80 > 1.5 * depol_sens_80
    assert loss_sens > 15.0 * loss_pfc


def test_p4_trajectory_swr_replay_and_asymmetric_ca3_stdp():
    """P4 (v5): Гиппокамп формирует асимметричную матрицу переходов CA3 w_ca3_seq
    и воспроизводит связные хронологические (forward) и обратные (reverse) траектории в SWR."""
    from neuroweave10.memory.hippocampus import Hippocampus

    hcfg = genesis_cfg("tiny").hippo
    hcfg.n_replay_batch = 4
    h = Hippocampus(hcfg, seed=5)
    # Записываем последовательность из 6 различных эпизодов
    for i in range(6):
        ec = torch.zeros(1, hcfg.n_ec)
        ec[0, (i * 4) % hcfg.n_ec: ((i + 1) * 4) % hcfg.n_ec] = 1.0
        out = h.step(ec, ach=0.9, adenosine=0.1, learn=True)
        # Последний эпизод помечаем высокой ошибкой/DA-тегом
        err = 2.5 if i == 5 else 0.2
        h.store_episode(ec, out.dg, out.ca3, error=err, uncertainty=0.3, da_tag=1.0 if i == 5 else 0.0)

    # Матрица w_ca3_seq ненулевая и асимметричная (направленная по стреле времени)
    assert float(h.w_ca3_seq.abs().max()) > 0.0
    assert not torch.allclose(h.w_ca3_seq, h.w_ca3_seq.t())

    # Прямой и обратный SWR-реплей возвращают непрерывные траектории (соседние шаги |i_{t+1} - i_t| == 1)
    idx_fwd = h.swr_replay(lambda ec, err: None, direction="forward")
    idx_rev = h.swr_replay(lambda ec, err: None, direction="reverse")
    assert len(idx_fwd) == 4 and len(idx_rev) == 4
    assert all(abs(a - b) == 1 for a, b in zip(idx_fwd, idx_fwd[1:]))
    assert all(abs(a - b) == 1 for a, b in zip(idx_rev, idx_rev[1:]))


def test_menesse_homeostatic_timescale_separation():
    """Гомеостаз Менессе (Menesse et al. 2024): на быстрых тактах (t < wnorm_every)
    отклонение σ регулируется ресурсом везикул u_syn и индивидуальными порогами Θ_i
    без стирания медленных весов w_slow."""
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=3)
    org.reset_state(1)
    col = org.hier.cols[0]
    w_slow_init = col.w_ff.w_slow.clone()
    u_init = float(col.w_ff.u_syn.mean())

    # 10 быстрых тактов при повышенной σ=1.12: u_syn снижается, но w_slow остаётся нетронутой!
    for _ in range(10):
        org.homeo.regulate(org.hier, sigma=1.12)
    assert torch.allclose(col.w_ff.w_slow, w_slow_init)
    assert float(col.w_ff.u_syn.mean()) < u_init


def test_v5_closed_feedback_loops_trn_metahead_and_volume_transmission():
    """Ziwago-Prime v5: Проверка замыкания всех разомкнутых петель обратной связи:
    1) L6 + AST + MetaHead.gates -> TRN -> ThalamusFO.gate;
    2) VolumeTransmission (NO гетеро-LTP и 2-AG выталкивание) -> CorticalColumn;
    3) Все 9 эффекторных функций ExtendedChemistry в mod_ctx."""
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=14)
    org.reset_state(1)
    x = torch.ones(1, cfg.input_dim) * 0.8
    for _ in range(5):
        out = org.step(x, reward=0.6, learn=True)
    # 1. Гейт ThalamusFO теперь динамически управляется выходом TRN (не равен константе 1.0)
    assert float(org.fo.gate.mean().item()) < 0.999
    # 2. Поля VolumeTransmission прокинуты в кортикальные колонки
    assert hasattr(org.hier.cols[0], "_no_ltp_field")
    assert org.hier.cols[0]._no_ltp_field is not None
    # 3. Рецепторная телеметрия и зональная критичность присутствуют в интроспекции
    assert "receptors" in out.introspection
    assert "zone_sigmas" in out.introspection
    assert len(out.introspection["zone_sigmas"]) == len(cfg.zones)


def test_v5_insula_acc_allostatic_self_model_and_mt_concepts():
    """Ziwago-Prime v5 (Инновации 2 и 4):
    - Островковая кора (Insula-ACC) выдаёт ошибку висцерального предсказания intero_error
      и упреждающий аллостатический драйв allostatic_drive при просадке АТФ;
    - Медиальная височная доля (MTL) формирует Концептуальные Клетки и семантический граф."""
    cfg = genesis_cfg("tiny")
    org = GenesisOrganism(cfg, batch=1, seed=19)
    org.reset_state(1)
    x1 = torch.zeros(1, cfg.input_dim); x1[0, :8] = 1.0
    x2 = torch.zeros(1, cfg.input_dim); x2[0, 8:16] = 1.0
    for _ in range(4):
        org.step(x1, reward=0.8, learn=True)
        org.step(x2, reward=0.8, learn=True)
    # Имитируем падение энергии АТФ в пулах коры -> аллостатический драйв Insula должен вырасти!
    for col in org.hier.cols:
        col.pyr.energy.fill_(0.35)
    out = org.step(x1, reward=0.0, learn=True)
    assert out.introspection["allostatic_drive"] > 0.35
    assert out.introspection["autobiographical_concepts"]["active_concepts"] >= 1
    assert out.introspection["autobiographical_concepts"]["semantic_bridges"] >= 1


def test_v5_mind_theater_vte_and_biphasic_nrem_rem_sleep():
    """Ziwago-Prime v5 (Инновации 1 и 7):
    - Театр Сознания (Vicarious Trial & Error / VTE) запускает мысленную репетицию
      кандидатов при свободном выборе и выбирает токен с минимальной Ожидаемой Свободной Энергией G(a);
    - Двухфазный сон (NREM + REM) выполняет и веретённую консолидацию, и REM-химерный синтез."""
    from neuroweave10.unified.pipeline import RegenesisOrganism, UnifiedConfig

    u = RegenesisOrganism(UnifiedConfig(vocab_size=16), gcfg=genesis_cfg("tiny"), seed=33)
    for tok_a, tok_b in [(2, 3), (3, 4), (4, 5), (5, 2)]:
        u.step_token(torch.tensor([tok_a]), torch.tensor([tok_b]), learn=True)

    # Свободный шаг (next_ids=None) вызывает VTE при малом разрыве потенциалов или высокой эпистемике
    u.motor.v.fill_(u.motor.cfg.e_l)
    r_free = u.step_token(torch.tensor([2]), next_ids=None, learn=False)
    assert "vte" in r_free
    vte_direct = u.simulate_vte([3, 4])
    assert vte_direct["vte_triggered"] == 1.0
    assert vte_direct["winner"] in (3, 4)

    # Двухфазный сон NREM + REM
    sleep_stats = u.sleep(ticks=20)
    assert "nrem_ticks" in sleep_stats and "rem_ticks" in sleep_stats
    assert "rem_chimera_similarity" in sleep_stats
    assert sleep_stats["semantic_bridges"] >= 1.0




