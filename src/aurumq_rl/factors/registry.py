"""Factor registry for AurumQ alpha101 + gtja191 unified factor library.

Single source of truth for factor metadata + polars callable implementation.
Used by:

* ``aurumq.factors.alpha101`` / ``aurumq.factors.gtja191`` (via symlink) for
  panel computation pipelines.
* ``aurumq.rules.aqml_polars_compiler`` for the ``resolve_for_aqml`` hook —
  when a user strategy expression like ``Rank(close) - alpha001`` references
  a factor symbol, the compiler resolves it via this registry instead of
  treating it as an ordinary panel column.
* ``aurumq_rl.factors._docs`` to extract docstring → markdown documentation.

Design notes
------------
* ``FactorEntry`` is frozen so registries cannot accidentally mutate metadata.
* ``impl`` is the canonical polars implementation. Each factor function takes
  a single ``pl.DataFrame`` (the enriched panel) and returns a ``pl.Series``
  aligned to the panel rows.
* ``legacy_aqml_expr`` is preserved on alpha101 entries that were migrated
  from the legacy ``aqml_strategy`` string-expression library; it is used as
  a numerical cross-check during the migration period and may be removed
  once parity is verified.
* ``quality_flag``: ``0`` = ok, ``1`` = errata-conservative (gtja191
  ambiguous formulas), ``2`` = stub.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

    FactorImpl = Callable[["pl.DataFrame"], "pl.Series"]
else:
    FactorImpl = Callable


__all__ = [
    "FactorEntry",
    "ALPHA101_REGISTRY",
    "GTJA191_REGISTRY",
    "list_all_factors",
    "register_alpha101",
    "register_gtja191",
    "resolve_for_aqml",
]


@dataclass(frozen=True)
class FactorEntry:
    """Metadata + callable for a single factor."""

    id: str
    impl: FactorImpl
    direction: Literal["normal", "reverse"]
    category: str
    description: str
    legacy_aqml_expr: str | None = None
    quality_flag: int = 0
    references: tuple[str, ...] = field(default_factory=tuple)
    formula_doc_path: str = ""


ALPHA101_REGISTRY: dict[str, FactorEntry] = {}
GTJA191_REGISTRY: dict[str, FactorEntry] = {}


def register_alpha101(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the alpha101 registry (idempotent on identical entry)."""
    if entry.id in ALPHA101_REGISTRY and ALPHA101_REGISTRY[entry.id] is not entry:
        raise ValueError(f"alpha101 factor {entry.id!r} already registered with a different entry")
    ALPHA101_REGISTRY[entry.id] = entry
    return entry


def register_gtja191(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the gtja191 registry (idempotent on identical entry)."""
    if entry.id in GTJA191_REGISTRY and GTJA191_REGISTRY[entry.id] is not entry:
        raise ValueError(f"gtja191 factor {entry.id!r} already registered with a different entry")
    GTJA191_REGISTRY[entry.id] = entry
    return entry


def list_all_factors() -> dict[str, FactorEntry]:
    """Return a merged view of alpha101 + gtja191 registries.

    Mutating the returned dict does NOT affect the underlying registries.
    Both factor families share an id-namespace by convention (``alpha`` and
    ``gtja_`` prefixes); collisions raise.
    """
    overlap = ALPHA101_REGISTRY.keys() & GTJA191_REGISTRY.keys()
    if overlap:
        raise RuntimeError(f"Factor id collision across registries: {sorted(overlap)}")
    return {**ALPHA101_REGISTRY, **GTJA191_REGISTRY}


def resolve_for_aqml(name: str, df: pl.DataFrame) -> pl.Series:
    """Hook called by ``aqml_polars_compiler`` to resolve a factor symbol.

    Parameters
    ----------
    name :
        Factor id, e.g. ``"alpha001"`` or ``"gtja_042"``.
    df :
        The enriched panel DataFrame the AQML compiler is currently
        evaluating against.

    Returns
    -------
    pl.Series
        Output of the registered factor implementation.

    Raises
    ------
    KeyError
        If ``name`` is not in any registry. The caller (AQML compiler) should
        fall back to treating the symbol as an ordinary panel column.
    """
    factors = list_all_factors()
    if name not in factors:
        raise KeyError(name)
    entry = factors[name]
    return entry.impl(df)
