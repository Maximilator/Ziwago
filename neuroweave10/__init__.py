"""NEUROWEAVE v10.0 «Genesis» — виртуальная биология по GENESIS_MANIFESTO."""
from __future__ import annotations

__version__ = "10.0-genesis"


def GenesisOrganism(*args, **kwargs):  # noqa: N802
    from .organism.genesis import GenesisOrganism as _G
    return _G(*args, **kwargs)


def GenesisSystem(*args, **kwargs):  # noqa: N802
    from .organism.system import GenesisSystem as _GS
    return _GS(*args, **kwargs)


def GenesisConfig(*args, **kwargs):  # noqa: N802
    from .config import GenesisConfig as _GC
    return _GC(*args, **kwargs)


__all__ = ["GenesisOrganism", "GenesisSystem", "GenesisConfig", "__version__"]
