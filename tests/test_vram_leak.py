# tests/test_vram_leak.py
"""
Регрессия: состояние между шагами не должно нести вычислительный граф.

Контекст (инцидент 27.10.2026, ночной прогон Stage 2B):
    В Фазе 2 мембранное состояние считалось из предыдущего шага, поэтому его
    grad_fn удерживал ВЕСЬ граф предыдущего шага. Модуль держит состояние живым
    -> цепочка v(t) -> v(t-1) -> ... растет бесконечно. Симптомы:
        allocated ~= reserved (2668 МБ против 2802 МБ), max_allocated == allocated,
        рост ~13 МБ на батч, затем CUDA OOM на 6-гигабайтной карте.

Инварианты, которые проверяет этот файл:
    1. Состояние нейрона (мембрана, адаптация, рефрактерность) не имеет grad_fn.
    2. prev-слои колонки отсоединены.
    3. Быстрый контур Фазы 2 под no_grad не строит граф вовсе.
"""
import torch

from neuroweave9.cortex.organism import CortexOrganism


def _organism() -> CortexOrganism:
    org = CortexOrganism(
        vocab_size=32,
        input_dim=32,
        scale=1.0,
        vsa_dim=32,
        seed=0,
        device=torch.device("cpu"),
    )
    org.set_phase(2)
    return org


def _step(org: CortexOrganism, steps: int = 3):
    x = torch.zeros((1, 32))
    x[0, 0] = 1.0
    out = None
    for _ in range(steps):
        out = org.step(x, sync_metrics=False)
    return out


def test_neuron_state_carries_no_graph_between_steps():
    """Состояние -- память, а не дифференцируемая величина: grad_fn у него быть не должно."""
    org = _organism()
    _step(org, steps=3)

    for name in ("A", "B", "C"):
        col = org.columns[name]
        for pool_name, pool in col.pools.items():
            v = getattr(pool, "v", None)
            if v is None:
                continue
            assert v.grad_fn is None, (
                f"состояние v пула {name}/{pool_name} держит граф между шагами "
                f"(grad_fn={v.grad_fn})"
            )
            a = getattr(pool, "a", None)
            if a is not None:
                assert a.grad_fn is None, (
                    f"состояние a пула {name}/{pool_name} держит граф между шагами"
                )


def test_column_prev_layers_are_detached():
    """prev-слои колонки (рецирентные входы) отсоединены от графа."""
    org = _organism()
    _step(org, steps=3)
    for name in ("A", "B", "C"):
        for layer, val in org.columns[name].prev.items():
            if torch.is_tensor(val):
                assert val.grad_fn is None, (
                    f"prev[{layer}] колонки {name} держит граф между шагами"
                )


def test_phase2_fast_loop_under_no_grad_builds_no_graph():
    """
    В Фазе 2 кору не учат (w_slow заморожен EWC, w_fast правит кредит-пайплайн под
    no_grad), а декодер отсоединяет все входы -> граф организму не нужен.
    Быстрый контур под no_grad обязан отдавать спайки без grad_fn.
    """
    org = _organism()
    x = torch.zeros((1, 32))
    x[0, 0] = 1.0

    with torch.no_grad():
        out = org.step(x, sync_metrics=False)

    spikes = out["left"]["zone_a"]["spikes"]
    assert spikes.grad_fn is None, "быстрый контур Фазы 2 построил граф под no_grad"


def test_memory_does_not_grow_with_steps():
    """
    Регрессия на симптом: число живых тензоров с графом не должно расти от шага к шагу.
    Считаем узлы графа через gc: утечка держала бы их в памяти между шагами.
    """
    import gc

    org = _organism()
    x = torch.zeros((1, 32))
    x[0, 0] = 1.0

    def graph_nodes() -> int:
        gc.collect()
        n = 0
        for o in gc.get_objects():
            if torch.is_tensor(o) and getattr(o, "grad_fn", None) is not None:
                n += 1
        return n

    for _ in range(3):
        org.step(x, sync_metrics=False)
    before = graph_nodes()
    for _ in range(12):
        org.step(x, sync_metrics=False)
    after = graph_nodes()

    # Допускаем небольшой прирост (кеш Python), но не рост, пропорциональный шагам
    assert after - before <= 12, (
        f"живых тензоров с графом растет от шага к шагу: {before} -> {after}"
    )
