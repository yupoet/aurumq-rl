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
    # entry is auto-wrapped with sanitizer, so stored != a / g (different impl).
    # Compare by id-set + verify metadata via ``dataclasses.replace`` invariants.
    assert set(registry.ALPHA101_REGISTRY) == {"alpha001"}
    assert set(registry.GTJA191_REGISTRY) == {"gtja_001"}
    assert registry.ALPHA101_REGISTRY["alpha001"].id == "alpha001"
    assert registry.ALPHA101_REGISTRY["alpha001"].direction == "normal"
    merged = registry.list_all_factors()
    assert set(merged) == {"alpha001", "gtja_001"}


def test_register_same_entry_twice_idempotent():
    """Re-registering the same entry twice must not raise (matches old contract)."""
    a = _make_const_factor("alpha001", 0.5)
    registry.register_alpha101(a)
    registry.register_alpha101(a)
    assert set(registry.ALPHA101_REGISTRY) == {"alpha001"}


def test_register_different_entry_same_id_raises():
    a1 = _make_const_factor("alpha001", 0.5)
    a2 = _make_const_factor("alpha001", 0.7)
    registry.register_alpha101(a1)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_alpha101(a2)


# ─── sanitize_factor_series tests ────────────────────────────────────────


def test_sanitize_replaces_pos_and_neg_inf_with_null():
    s = pl.Series("x", [1.0, float("inf"), 2.0, -float("inf"), 3.0])
    out = registry.sanitize_factor_series(s)
    assert out.to_list() == [1.0, None, 2.0, None, 3.0]
    assert int(out.is_infinite().sum()) == 0


def test_sanitize_clips_finite_values_to_limit():
    big = registry.FACTOR_CLIP_LIMIT * 5
    s = pl.Series("x", [-big, -1.0, 1.0, big])
    out = registry.sanitize_factor_series(s).to_list()
    assert out == [-registry.FACTOR_CLIP_LIMIT, -1.0, 1.0, registry.FACTOR_CLIP_LIMIT]


def test_sanitize_preserves_normal_values_and_nan():
    s = pl.Series("x", [1.0, float("nan"), 2.5, None])
    out = registry.sanitize_factor_series(s).to_list()
    assert out[0] == 1.0
    assert out[1] != out[1]  # NaN preserved (not converted)
    assert out[2] == 2.5
    assert out[3] is None


def test_register_alpha101_auto_wraps_impl_so_inf_output_is_sanitized():
    """If a factor accidentally returns inf, the registered impl must clean it."""

    def _crazy_impl(df: pl.DataFrame) -> pl.Series:
        # Deliberately return ±inf + overflow to ensure wrapping cleans them.
        return pl.Series("crazy", [1.0, float("inf"), -float("inf"), 1e308])

    entry = registry.FactorEntry(
        id="alpha_crazy",
        impl=_crazy_impl,
        direction="reverse",
        category="test",
        description="crazy-tail simulator",
    )
    registry.register_alpha101(entry)
    df = pl.DataFrame({"stock_code": ["x"] * 4})
    result = registry.resolve_for_aqml("alpha_crazy", df)
    out = result.to_list()
    assert int(result.is_infinite().sum()) == 0, f"inf leaked through registration: {out}"
    assert out[1] is None and out[2] is None
    assert out[3] == registry.FACTOR_CLIP_LIMIT


def test_wrapped_impl_exposes_original_via_dunder_wrapped():
    a = _make_const_factor("alpha001", 0.5)
    registry.register_alpha101(a)
    stored = registry.ALPHA101_REGISTRY["alpha001"]
    assert hasattr(stored.impl, "__wrapped__")
    assert stored.impl.__wrapped__ is a.impl


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
