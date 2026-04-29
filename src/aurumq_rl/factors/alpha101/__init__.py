"""WorldQuant Alpha 101 factor library — polars implementations.

Each factor lives in a category submodule (volatility / volume_price /
momentum / mean_reversion / breakout / technical / industry_neutral /
cap_weighted / custom) and **self-registers** at import time by calling
``register_alpha101(FactorEntry(...))`` at module bottom.

Importing this package triggers all category modules to be imported, which
populates ``aurumq_rl.factors.registry.ALPHA101_REGISTRY`` for downstream
consumers (panel compute pipelines, AQML hook, docs extractor).
"""

from aurumq_rl.factors.registry import ALPHA101_REGISTRY as REGISTRY

# Importing each category module triggers its self-registration.
# Order doesn't matter since registrations are independent dict insertions.
from . import (
    adv_extended,  # noqa: F401
    breakout,  # noqa: F401
    cap_weighted,  # noqa: F401
    industry_neutral,  # noqa: F401
    mean_reversion,  # noqa: F401
    momentum,  # noqa: F401
    technical,  # noqa: F401
    volatility,  # noqa: F401
    volume_price,  # noqa: F401
)

# Phase B / B' will add:
# from . import custom  # noqa: F401

__all__ = ["REGISTRY"]
