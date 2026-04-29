"""Unit tests for the cross-family factor registry."""

from __future__ import annotations

import polars as pl
import pytest

from aurumq_rl.factors import registry


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Snapshot + restore module-level registries so tests don't leak state."""
    a_snap = dict(registry.ALPHA101_REGISTRY)
    g_snap = dict(registry.GTJA191_REGISTRY)
    registry.ALPHA101_REGISTRY.clear()
    registry.GTJA191_REGISTRY.clear()
    try:
        yield
    finally:
        registry.ALPHA101_REGISTRY.clear()
        registry.GTJA191_REGISTRY.clear()
        registry.ALPHA101_REGISTRY.update(a_snap)
        registry.GTJA191_REGISTRY.update(g_snap)


def _make_const_factor(fid: str, value: float) -> registry.FactorEntry:
    def _impl(df: pl.DataFrame) -> pl.Series:
        return pl.Series(fid, [value] * df.height, dtype=pl.Float64)

    return registry.FactorEntry(
        id=fid,
        impl=_impl,
        direction="normal",
        category="test",
        description=f"constant {value} for testing",
    )


def test_factor_entry_is_frozen():
    e = _make_const_factor("alpha999", 1.0)
    with pytest.raises((AttributeError, Exception)):
        e.id = "alpha998"  # type: ignore[misc]


def test_register_alpha101_then_gtja191_no_collision():
    a = _make_const_factor("alpha001", 0.5)
    g = registry.FactorEntry(
        id="gtja_001",
        impl=lambda df: pl.Series("x", [0.0] * df.height),
        direction="reverse",
        category="momentum",
        description="g1",
    )
    registry.register_alpha101(a)
    registry.register_gtja191(g)
    assert registry.ALPHA101_REGISTRY == {"alpha001": a}
    assert registry.GTJA191_REGISTRY == {"gtja_001": g}
    merged = registry.list_all_factors()
    assert set(merged) == {"alpha001", "gtja_001"}


def test_register_same_entry_twice_idempotent():
    a = _make_const_factor("alpha001", 0.5)
    registry.register_alpha101(a)
    registry.register_alpha101(a)
    assert registry.ALPHA101_REGISTRY == {"alpha001": a}


def test_register_different_entry_same_id_raises():
    a1 = _make_const_factor("alpha001", 0.5)
    a2 = _make_const_factor("alpha001", 0.7)
    registry.register_alpha101(a1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_alpha101(a2)


def test_id_collision_across_registries_raises():
    a = _make_const_factor("shared_id", 0.5)
    g = _make_const_factor("shared_id", 0.5)
    registry.ALPHA101_REGISTRY["shared_id"] = a
    registry.GTJA191_REGISTRY["shared_id"] = g
    with pytest.raises(RuntimeError, match="collision"):
        registry.list_all_factors()


def test_resolve_for_aqml_invokes_impl():
    a = _make_const_factor("alpha001", 0.42)
    registry.register_alpha101(a)
    df = pl.DataFrame({"stock_code": ["x", "y", "z"]})
    out = registry.resolve_for_aqml("alpha001", df)
    assert out.to_list() == [0.42, 0.42, 0.42]


def test_resolve_for_aqml_unknown_symbol_raises_keyerror():
    df = pl.DataFrame({"stock_code": ["x"]})
    with pytest.raises(KeyError, match="alpha999"):
        registry.resolve_for_aqml("alpha999", df)


def test_list_all_factors_returns_independent_copy():
    a = _make_const_factor("alpha001", 0.5)
    registry.register_alpha101(a)
    merged = registry.list_all_factors()
    merged["alpha999"] = a  # mutate the returned dict
    assert "alpha999" not in registry.ALPHA101_REGISTRY
    assert "alpha999" not in registry.GTJA191_REGISTRY
