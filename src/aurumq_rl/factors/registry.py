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
* ``impl_incremental`` / ``max_window``: OPTIONAL opt-in fields (both default
  ``None``) implementing the incremental-computation protocol — see the
  "Incremental computation protocol" section below. A factor that does not
  set them behaves exactly as before (``impl`` is the only required path).

Incremental computation protocol (issue #10)
---------------------------------------------
Every registered factor keeps its full-history path: ``impl(df) -> pl.Series``
recomputes the factor over the entire panel and remains the source of truth.
Some factors — pure per-stock time-series formulas with a bounded rolling
lookback — can ALSO opt into a second, optional path meant for a daily
refresh: recompute only the new rows using a bounded "tail buffer" per stock
instead of replaying the full history.

A factor opts in by registering two additional fields on ``FactorEntry``:

* ``max_window: int`` — the factor's maximum lookback in rows (the deepest
  rolling window / delay it uses, expressed as "how many prior rows are
  needed before the first row a caller cares about can be computed").
* ``impl_incremental: Callable[[pl.DataFrame, int], pl.Series]`` — given a
  **tail buffer** ``tail_df`` and an integer ``n_new``, returns the factor
  values for exactly the last ``n_new`` rows of every stock group in
  ``tail_df``.

Tail buffer contract
~~~~~~~~~~~~~~~~~~~~~
``tail_df`` must be sorted ``[stock_code, trade_date]`` ascending (same
convention as the full-history panel passed to ``impl``) and must contain,
for every stock present in its last ``n_new`` rows, **at least**
``max_window + n_new`` rows for that stock — i.e. ``max_window`` rows of
prior history immediately followed by the ``n_new`` new rows. A caller
assembles this by taking, per stock, the last ``max_window + n_new`` rows of
that stock's full history (see :func:`compute_incremental` for a ready-made
slicer over a full panel).

Given a buffer that satisfies the contract, ``impl_incremental(tail_df,
n_new)`` MUST return a ``pl.Series`` of length ``n_new * n_stocks`` (in the
same per-stock grouped order as ``tail_df``'s trailing rows) that is
numerically identical, within float tolerance, to slicing
``impl(full_history_df)`` down to those same trailing ``n_new`` rows per
stock. A conforming implementation may reject a buffer shorter than the
minimum with a clear ``ValueError`` naming the offending stock(s); silently
returning wrong numbers is never acceptable. Registered ``impl_incremental``
callables are auto-wrapped with :func:`sanitize_factor_series`, exactly like
``impl``, so both paths apply the same inf/overflow cleanup.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import polars as pl

    FactorImpl = Callable[["pl.DataFrame"], "pl.Series"]
    FactorImplIncremental = Callable[["pl.DataFrame", int], "pl.Series"]
else:
    FactorImpl = Callable
    FactorImplIncremental = Callable


__all__ = [
    "FactorEntry",
    "ALPHA101_REGISTRY",
    "GTJA191_REGISTRY",
    "FACTOR_CLIP_LIMIT",
    "list_all_factors",
    "register_alpha101",
    "register_gtja191",
    "resolve_for_aqml",
    "resolve_incremental",
    "compute_incremental",
    "sanitize_factor_series",
]


# Clip limit for factor outputs. Any finite value outside this range is treated
# as overflow / numerical artifact and clipped. ±1e6 leaves plenty of room for
# legitimate-but-large factor values (e.g. alpha_083 swings to ±30k on real data)
# while catching overflow cases like gtja_017 reaching 1e+302 from `rank ^ delta`.
FACTOR_CLIP_LIMIT: float = 1.0e6


def sanitize_factor_series(series: pl.Series) -> pl.Series:
    """Replace ±inf with null and clip finite values to ±``FACTOR_CLIP_LIMIT``.

    Defends downstream consumers (``np.percentile``, cross-section z-score,
    ML training) against factor-library overflow / divide-by-zero artifacts:
        - ±inf → null (numpy's percentile returns nan + warns on inf input)
        - |x| > 1e6 → clipped (gtja_017's ``rank ^ delta`` blew up to 1e+302)
        - finite, in-range → unchanged

    Returns a new Float64 Series of the same length and name.
    Cheap (one with_columns expression), so safe to apply on every factor.
    """
    import polars as pl

    if series.dtype != pl.Float64:
        series = series.cast(pl.Float64, strict=False)
    name = series.name
    return pl.DataFrame({name: series}).with_columns(
        pl.when(pl.col(name).is_infinite())
        .then(None)
        .otherwise(pl.col(name).clip(-FACTOR_CLIP_LIMIT, FACTOR_CLIP_LIMIT))
        .alias(name)
    )[name]


def _wrap_impl_with_sanitizer(impl: FactorImpl) -> FactorImpl:
    """Wrap a factor ``impl(df) -> pl.Series`` so its output is sanitized."""

    def _sanitized(df):  # type: ignore[no-untyped-def]
        return sanitize_factor_series(impl(df))

    # Preserve introspection-friendly attributes so debugging / docs still work.
    _sanitized.__name__ = getattr(impl, "__name__", "_sanitized_impl")
    _sanitized.__doc__ = getattr(impl, "__doc__", None)
    _sanitized.__wrapped__ = impl  # type: ignore[attr-defined]
    return _sanitized


def _wrap_impl_incremental_with_sanitizer(
    impl_incremental: FactorImplIncremental,
) -> FactorImplIncremental:
    """Wrap ``impl_incremental(tail_df, n_new) -> pl.Series`` with the sanitizer.

    Mirrors :func:`_wrap_impl_with_sanitizer` so the incremental path applies
    the exact same inf/overflow cleanup as the full-history path — required
    for the two paths to stay numerically identical (see module docstring).
    """

    def _sanitized(tail_df, n_new):  # type: ignore[no-untyped-def]
        return sanitize_factor_series(impl_incremental(tail_df, n_new))

    _sanitized.__name__ = getattr(impl_incremental, "__name__", "_sanitized_impl_incremental")
    _sanitized.__doc__ = getattr(impl_incremental, "__doc__", None)
    _sanitized.__wrapped__ = impl_incremental  # type: ignore[attr-defined]
    return _sanitized


@dataclass(frozen=True)
class FactorEntry:
    """Metadata + callable for a single factor.

    ``impl_incremental`` / ``max_window`` are OPTIONAL (default ``None``) —
    see the "Incremental computation protocol" section of the module
    docstring. A factor that leaves them unset is unaffected; ``impl`` and
    the full-history recompute path are always available and unchanged.
    """

    id: str
    impl: FactorImpl
    direction: Literal["normal", "reverse"]
    category: str
    description: str
    legacy_aqml_expr: str | None = None
    quality_flag: int = 0
    references: tuple[str, ...] = field(default_factory=tuple)
    formula_doc_path: str = ""
    impl_incremental: FactorImplIncremental | None = None
    max_window: int | None = None


ALPHA101_REGISTRY: dict[str, FactorEntry] = {}
GTJA191_REGISTRY: dict[str, FactorEntry] = {}


def _validate_incremental_metadata(entry: FactorEntry) -> None:
    """Enforce that ``impl_incremental`` and ``max_window`` are set together.

    ``max_window`` is only required for factors that provide
    ``impl_incremental`` (see module docstring); it stays optional
    otherwise, so this never rejects the ~99% of factors that don't opt in.
    """
    if entry.impl_incremental is not None and entry.max_window is None:
        raise ValueError(
            f"factor {entry.id!r} provides impl_incremental but no max_window "
            "— max_window is required whenever impl_incremental is set"
        )


def register_alpha101(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the alpha101 registry (idempotent on identical entry).

    The entry's ``impl`` is automatically wrapped with :func:`sanitize_factor_series`
    so every registered factor produces inf-free, clipped output regardless of how
    the underlying formula handles divide-by-zero / overflow. If ``impl_incremental``
    is set, it is wrapped the same way (see module docstring for the protocol).
    """
    _validate_incremental_metadata(entry)
    wrapped_incremental = (
        _wrap_impl_incremental_with_sanitizer(entry.impl_incremental)
        if entry.impl_incremental is not None
        else None
    )
    sanitized_entry = dataclasses.replace(
        entry,
        impl=_wrap_impl_with_sanitizer(entry.impl),
        impl_incremental=wrapped_incremental,
    )
    if entry.id in ALPHA101_REGISTRY and ALPHA101_REGISTRY[entry.id] is not sanitized_entry:
        # Allow re-registration only if the underlying (unwrapped) impl is identical.
        existing_impl = getattr(
            ALPHA101_REGISTRY[entry.id].impl, "__wrapped__", ALPHA101_REGISTRY[entry.id].impl
        )
        if existing_impl is not entry.impl:
            raise ValueError(
                f"alpha101 factor {entry.id!r} already registered with a different entry"
            )
    ALPHA101_REGISTRY[entry.id] = sanitized_entry
    return sanitized_entry


def register_gtja191(entry: FactorEntry) -> FactorEntry:
    """Register a factor in the gtja191 registry (idempotent on identical entry).

    See :func:`register_alpha101` for the auto-sanitization contract (applies
    identically to ``impl_incremental`` when set).
    """
    _validate_incremental_metadata(entry)
    wrapped_incremental = (
        _wrap_impl_incremental_with_sanitizer(entry.impl_incremental)
        if entry.impl_incremental is not None
        else None
    )
    sanitized_entry = dataclasses.replace(
        entry,
        impl=_wrap_impl_with_sanitizer(entry.impl),
        impl_incremental=wrapped_incremental,
    )
    if entry.id in GTJA191_REGISTRY and GTJA191_REGISTRY[entry.id] is not sanitized_entry:
        existing_impl = getattr(
            GTJA191_REGISTRY[entry.id].impl, "__wrapped__", GTJA191_REGISTRY[entry.id].impl
        )
        if existing_impl is not entry.impl:
            raise ValueError(
                f"gtja191 factor {entry.id!r} already registered with a different entry"
            )
    GTJA191_REGISTRY[entry.id] = sanitized_entry
    return sanitized_entry


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


def resolve_incremental(name: str, tail_df: pl.DataFrame, n_new: int) -> pl.Series:
    """Incremental counterpart of :func:`resolve_for_aqml`.

    Looks up ``name`` in the merged registry and invokes its
    ``impl_incremental(tail_df, n_new)``. See the module docstring's
    "Incremental computation protocol" section for the tail-buffer contract
    ``tail_df`` must satisfy.

    Parameters
    ----------
    name :
        Factor id, e.g. ``"alpha009"``.
    tail_df :
        Per-stock tail buffer: >= ``max_window + n_new`` rows per stock,
        sorted ``[stock_code, trade_date]`` ascending.
    n_new :
        Number of new (trailing) rows per stock to compute.

    Returns
    -------
    pl.Series
        Factor values for the last ``n_new`` rows of every stock group in
        ``tail_df``.

    Raises
    ------
    KeyError
        If ``name`` is not in any registry.
    ValueError
        If the factor is registered but does not provide
        ``impl_incremental`` (i.e. has not opted into the protocol).
    """
    factors = list_all_factors()
    if name not in factors:
        raise KeyError(name)
    entry = factors[name]
    if entry.impl_incremental is None:
        raise ValueError(
            f"factor {name!r} has no incremental implementation "
            "(impl_incremental is None) — use resolve_for_aqml / impl instead"
        )
    return entry.impl_incremental(tail_df, n_new)


def compute_incremental(entry: FactorEntry, panel: pl.DataFrame, n_new: int) -> pl.Series:
    """Slice the per-stock tail buffer from a full panel and run the incremental path.

    Convenience helper so callers don't have to hand-roll the "last
    ``max_window + n_new`` rows per stock" slice documented in the module's
    "Incremental computation protocol" section. ``panel`` should be the same
    full-history frame that would otherwise be passed to ``entry.impl`` —
    this helper only reads the trailing rows it needs per stock, so it stays
    ``O(n_stocks * (max_window + n_new))`` regardless of ``panel``'s total
    length.

    Parameters
    ----------
    entry :
        A :class:`FactorEntry` with both ``impl_incremental`` and
        ``max_window`` set (i.e. one that opted into the protocol).
    panel :
        Full-history panel, sorted ``[stock_code, trade_date]`` ascending.
        May contain more history than needed — only the tail is read.
    n_new :
        Number of new (trailing) rows per stock to compute.

    Returns
    -------
    pl.Series
        Same as calling ``entry.impl_incremental`` directly on the sliced
        tail buffer.

    Raises
    ------
    ValueError
        If ``entry`` did not register ``impl_incremental`` / ``max_window``,
        or if any stock present in the trailing ``n_new`` rows has fewer
        than ``max_window + n_new`` rows of history available in ``panel``
        (surfaced by the underlying ``impl_incremental`` call).
    """
    if entry.impl_incremental is None or entry.max_window is None:
        raise ValueError(
            f"factor {entry.id!r} does not support incremental computation "
            "(impl_incremental / max_window not set)"
        )
    if n_new < 1:
        raise ValueError(f"n_new must be >= 1, got {n_new}")
    buffer_size = entry.max_window + n_new
    tail_df = (
        panel.sort(["stock_code", "trade_date"])
        .group_by("stock_code", maintain_order=True)
        .tail(buffer_size)
    )
    return entry.impl_incremental(tail_df, n_new)
