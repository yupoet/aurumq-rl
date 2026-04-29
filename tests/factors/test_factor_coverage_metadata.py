"""Coverage and quality-flag checks for the shipped factor registries."""

from __future__ import annotations

import importlib

import pytest

from aurumq_rl.factors import registry

ALPHA101_MODULES = (
    "aurumq_rl.factors.alpha101.adv_extended",
    "aurumq_rl.factors.alpha101.breakout",
    "aurumq_rl.factors.alpha101.cap_weighted",
    "aurumq_rl.factors.alpha101.industry_neutral",
    "aurumq_rl.factors.alpha101.mean_reversion",
    "aurumq_rl.factors.alpha101.momentum",
    "aurumq_rl.factors.alpha101.technical",
    "aurumq_rl.factors.alpha101.volatility",
    "aurumq_rl.factors.alpha101.volume_price",
)

GTJA191_MODULES = (
    "aurumq_rl.factors.gtja191.batch_001_020",
    "aurumq_rl.factors.gtja191.batch_021_040",
    "aurumq_rl.factors.gtja191.batch_041_060",
    "aurumq_rl.factors.gtja191.batch_061_080",
    "aurumq_rl.factors.gtja191.batch_081_100",
    "aurumq_rl.factors.gtja191.batch_101_120",
    "aurumq_rl.factors.gtja191.batch_121_140",
    "aurumq_rl.factors.gtja191.batch_141_160",
    "aurumq_rl.factors.gtja191.batch_161_180",
    "aurumq_rl.factors.gtja191.batch_181_191",
)

INDCLASS_ALPHA101_IDS = frozenset(
    {
        "alpha023",
        "alpha031",
        "alpha068",
        "alpha076",
        "alpha079",
        "alpha082",
        "alpha087",
        "alpha089",
        "alpha090",
        "alpha093",
        "alpha094",
        "alpha097",
        "alpha100",
    }
)

GTJA_ERRATA_IDS = frozenset(
    {
        "gtja_050",
        "gtja_051",
        "gtja_055",
        "gtja_069",
        "gtja_073",
    }
)


@pytest.fixture(scope="module")
def refreshed_registries():
    """Re-register factor modules from a clean registry state."""
    import aurumq_rl.factors.alpha101  # noqa: F401
    import aurumq_rl.factors.gtja191  # noqa: F401

    registry.ALPHA101_REGISTRY.clear()
    registry.GTJA191_REGISTRY.clear()

    for module_name in ALPHA101_MODULES:
        importlib.reload(importlib.import_module(module_name))
    for module_name in GTJA191_MODULES:
        importlib.reload(importlib.import_module(module_name))

    return registry.ALPHA101_REGISTRY, registry.GTJA191_REGISTRY


def test_alpha101_numbered_coverage_is_complete(refreshed_registries):
    alpha_registry, _ = refreshed_registries
    expected_numbered = {f"alpha{i:03d}" for i in range(1, 102)}
    numbered = {
        factor_id
        for factor_id in alpha_registry
        if factor_id.startswith("alpha") and factor_id.removeprefix("alpha").isdigit()
    }
    assert numbered == expected_numbered
    assert len(alpha_registry) == 107


@pytest.mark.parametrize("factor_id", sorted(INDCLASS_ALPHA101_IDS))
def test_indclass_alpha101_factors_are_flagged(refreshed_registries, factor_id):
    alpha_registry, _ = refreshed_registries
    assert alpha_registry[factor_id].quality_flag == 1


@pytest.mark.parametrize("factor_id", sorted(GTJA_ERRATA_IDS))
def test_gtja191_errata_factors_are_flagged(refreshed_registries, factor_id):
    _, gtja_registry = refreshed_registries
    assert gtja_registry[factor_id].quality_flag == 1
