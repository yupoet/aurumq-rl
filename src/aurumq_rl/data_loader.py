"""Factor panel loader: Parquet → numpy 3D panel.

Prefix-based factor recognition
-------------------------------
Input Parquet must contain columns prefixed with one of:

* ``alpha_*``   alpha101 quant-volume factors
* ``mf_*``      main-force capital flow
* ``mfp_*``     main-force capital pressure / persistence (separate from mf_*)
* ``hm_*``      hot-money seats
* ``hk_*``      northbound capital
* ``inst_*``    institutional flow (limit-up/down list)
* ``mg_*``      margin trading
* ``cyq_*``     chip distribution
* ``senti_*``   limit-up sentiment
* ``sh_*``      shareholders
* ``fund_*``    fundamentals (PE/PB/ROE/...)
* ``ind_*``     industry relative strength
* ``mkt_*``     market regime
* ``gtja_*``    Guotai Junan Alpha191 (GTJA short-period price-volume alphas)

The loader picks **all** matching columns (sorted alphabetically) up to
``n_factors``. Missing prefixes are silently skipped — RL never errors out
because of missing factor groups; the model just sees those positions as 0.

Universe filter
---------------
Default ``UniverseFilter.MAIN_BOARD_NON_ST`` excludes:

* BSE (.BJ, codes 8/4)
* STAR market (688)
* ChiNext (300/301)
* ST/*ST stocks

Other modes: ``ALL_A`` (no filter), ``HS300``, ``ZZ500``, ``ZZ1000``.

This module has **no PyTorch / gymnasium dependency**, safe to import in any
environment.
"""

from __future__ import annotations

import datetime
import re
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # pragma: no cover — Python 3.10 compat shim
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 backport of :class:`enum.StrEnum`."""

        def __str__(self) -> str:  # type: ignore[override]
            return str(self.value)


import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARQUET_PATH: str = "data/factor_panel.parquet"
NEW_STOCK_PROTECT_DAYS: int = 60

# Per-stock factor column prefixes (consumed by the per-stock encoder).
# Phase 21 NARROWS this to per-stock cross-section factors only — `mkt_*`
# columns are now considered cross-section-constant regime context and are
# explicitly forbidden from reaching the per-stock encoder. They remain
# usable for regime feature computation; see `_compute_regime_features`.
STOCK_FACTOR_PREFIXES: tuple[str, ...] = (
    "alpha_",
    "mf_",
    "mfp_",  # main-force pressure / persistence; distinct from mf_*
    "hm_",
    "hk_",
    "inst_",
    "mg_",
    "cyq_",
    "senti_",
    "sh_",
    "fund_",
    "ind_",
    "gtja_",
)

# Prefixes that must NEVER appear as per-stock encoder input. The schema
# lock at training startup re-asserts this; data_loader silently filters
# them out of `discover_factor_columns`.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "mkt_",
    "index_",
    "regime_",
    "global_",
)

# Backwards-compat alias kept for any external code reading the V1 name.
# Will be removed in a follow-up cleanup; do NOT add new references.
FACTOR_COL_PREFIXES = STOCK_FACTOR_PREFIXES

# Phase 21 v0 regime feature names — the 8 cross-section/index time-series
# columns computed by :func:`_compute_regime_features` and stored in
# :attr:`FactorPanel.regime_array`. Order is load-bearing: the RegimeEncoder
# consumes these by position.
REGIME_FEATURE_NAMES: tuple[str, ...] = (
    "regime_breadth_d",
    "regime_breadth_20d",
    "regime_xs_disp_d",
    "regime_xs_disp_20d",
    "regime_idx_ret_20d",
    "regime_idx_ret_60d",
    "regime_idx_vol_20d",
    "regime_extreme_imbalance_norm",
)

# Required columns in input Parquet
REQUIRED_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "close", "pct_chg", "vol")

# Optional columns (used if present, defaulted otherwise)
OPTIONAL_COLUMNS: tuple[str, ...] = ("is_st", "days_since_ipo", "industry_code", "name")


class UniverseFilter(StrEnum):
    """Stock universe selection mode."""

    ALL_A = "all_a"
    MAIN_BOARD_NON_ST = "main_board_non_st"
    HS300 = "hs300"
    ZZ500 = "zz500"
    ZZ1000 = "zz1000"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class FactorPanel(NamedTuple):
    """Container for a 3D factor panel + auxiliary arrays.

    Phase 21 narrows ``factor_array`` / ``factor_names`` to per-stock factors
    ONLY (mkt_/index_/regime_/global_ excluded by data_loader's allowlist),
    and adds ``regime_array`` / ``regime_names`` for the date-level regime
    context tensor consumed by the new RegimeEncoder.

    Attributes
    ----------
    factor_array:
        shape (n_dates, n_stocks, n_factors), cross-sectionally z-scored.
    return_array:
        shape (n_dates, n_stocks), forward log-returns.
    pct_change_array:
        shape (n_dates, n_stocks), daily pct change as **decimals** (+10% = 0.10).
    is_st_array:
        shape (n_dates, n_stocks), bool.
    is_suspended_array:
        shape (n_dates, n_stocks), bool (volume == 0).
    days_since_ipo_array:
        shape (n_dates, n_stocks).
    dates:
        list[date], length n_dates.
    stock_codes:
        list[str], length n_stocks.
    factor_names:
        list[str], length n_factors.
    regime_array:
        shape (n_dates, 8), float32. The 8 v0 cross-section/index regime
        features per :data:`REGIME_FEATURE_NAMES`. Default empty so legacy
        callers that build FactorPanel by hand continue to work.
    regime_names:
        list[str], length 8. Names of the regime features. Default empty
        list for legacy compatibility.
    """

    factor_array: np.ndarray
    return_array: np.ndarray
    pct_change_array: np.ndarray
    is_st_array: np.ndarray
    is_suspended_array: np.ndarray
    days_since_ipo_array: np.ndarray
    dates: list[datetime.date]
    stock_codes: list[str]
    factor_names: list[str]
    regime_array: np.ndarray = np.zeros((0, 0), dtype=np.float32)
    regime_names: tuple[str, ...] = ()


def align_panel_to_stock_list(
    panel: FactorPanel, target_stock_codes: list[str]
) -> FactorPanel:
    """Realign a FactorPanel to a fixed stock universe (order + count).

    For OOS backtest where the panel's stock universe differs from the one
    used at training time. Returns a new panel with:

      * ``stock_codes`` == ``target_stock_codes`` (same order, same length)
      * factor / return / pct_change arrays sliced + reordered + zero-padded
        for stocks present in target but missing from the panel
      * is_st / is_suspended for missing stocks set to ``True`` (treated
        as un-tradeable so the env doesn't pretend to pick them)
      * days_since_ipo for missing stocks set to 0

    This is the canonical fix for the OOS universe-misalignment issue: the
    env's observation space has a fixed (n_stocks * n_factors,) shape that
    is locked at training time; without alignment, an OOS panel with even a
    single different stock breaks ONNX inference.
    """
    if list(panel.stock_codes) == list(target_stock_codes):
        return panel

    n_dates = panel.factor_array.shape[0]
    n_factors = panel.factor_array.shape[2]
    n_target = len(target_stock_codes)

    # Build idx map: target_idx -> source_idx (or -1 for missing)
    src_idx_by_code = {c: i for i, c in enumerate(panel.stock_codes)}
    idx_map = np.array(
        [src_idx_by_code.get(c, -1) for c in target_stock_codes], dtype=np.int64
    )
    present = idx_map >= 0
    missing_count = int((~present).sum())

    # Helper: gather along axis=1 with -1 → zero/default
    def _gather(arr: np.ndarray, default) -> np.ndarray:
        out_shape = (arr.shape[0], n_target) + arr.shape[2:]
        out = np.full(out_shape, default, dtype=arr.dtype)
        if present.any():
            out[:, present] = arr[:, idx_map[present]]
        return out

    factor_array = _gather(panel.factor_array, 0.0)
    return_array = _gather(panel.return_array, 0.0)
    pct_change_array = _gather(panel.pct_change_array, 0.0)
    is_st_array = _gather(panel.is_st_array, True)         # missing → ST (un-tradeable)
    is_suspended_array = _gather(panel.is_suspended_array, True)  # missing → suspended
    days_since_ipo_array = _gather(panel.days_since_ipo_array, 0)

    return FactorPanel(
        factor_array=factor_array,
        return_array=return_array,
        pct_change_array=pct_change_array,
        is_st_array=is_st_array,
        is_suspended_array=is_suspended_array,
        days_since_ipo_array=days_since_ipo_array,
        dates=list(panel.dates),
        stock_codes=list(target_stock_codes),
        factor_names=list(panel.factor_names),
        regime_array=panel.regime_array.copy(),
        regime_names=tuple(panel.regime_names),
    )


# ---------------------------------------------------------------------------
# Universe filtering
# ---------------------------------------------------------------------------

# Patterns for default "main_board_non_st" filter
_SH_MAIN_PATTERN = re.compile(r"^60[0135]\d{3}\.SH$")
_SZ_MAIN_PATTERN = re.compile(r"^00[0123]\d{3}\.SZ$")
_ST_NAME_PATTERN = re.compile(r"(\*?ST|退)")


def _is_main_board(code: str) -> bool:
    """True if code is SH/SZ main board."""
    return bool(_SH_MAIN_PATTERN.match(code) or _SZ_MAIN_PATTERN.match(code))


def filter_universe(
    df: pl.DataFrame,
    mode: UniverseFilter = UniverseFilter.MAIN_BOARD_NON_ST,
    name_col: str = "name",
) -> pl.DataFrame:
    """Filter the universe of stocks.

    Parameters
    ----------
    df:
        Input dataframe with at least ``ts_code`` column.
    mode:
        Filter mode (see :class:`UniverseFilter`).
    name_col:
        Column holding stock name (used for ST detection). If absent,
        ST filtering is skipped.

    Returns
    -------
    Filtered dataframe.
    """
    if mode == UniverseFilter.ALL_A:
        return df

    if mode == UniverseFilter.MAIN_BOARD_NON_ST:
        # Apply main-board filter
        df = df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))
        # ST exclusion (only if name column exists)
        if name_col in df.columns:
            df = df.filter(~pl.col(name_col).cast(pl.Utf8).str.contains(r"\*?ST|退"))
        return df

    # Index-component modes: prefer explicit boolean column when present,
    # fall back to main-board heuristic otherwise.
    if mode == UniverseFilter.HS300:
        if "is_hs300" in df.columns:
            return df.filter(pl.col("is_hs300") == True)  # noqa: E712
        return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))

    if mode == UniverseFilter.ZZ500:
        if "is_zz500" in df.columns:
            return df.filter(pl.col("is_zz500") == True)  # noqa: E712
        return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))

    # ZZ1000 (and any future enum value) — fall back to main-board heuristic.
    return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_section_zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score normalize along axis=1 (stock dim).

    For each (date, factor), normalize across stocks:
        z = (x - mean) / (std + 1e-8)

    NaN cells in the input (suspended days, pre-IPO, factor warm-up) and
    rows where the entire cross-section is NaN (so mean/std are NaN) are
    replaced with 0.0, the neutral signal per the project convention.
    Without this, env reset() returns observations containing NaN, which
    Box.contains() rejects and SB3's check_env asserts on.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=1, keepdims=True)
        std = np.nanstd(arr, axis=1, keepdims=True)

    with np.errstate(invalid="ignore"):
        z = (arr - mean) / (std + 1e-8)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_feature_group_weights(
    factor_array: np.ndarray,
    factor_names: list[str],
    feature_group_weights: dict[str, float] | None,
) -> np.ndarray:
    """Multiply factor columns by per-prefix scalar weights, in-place.

    Applied AFTER cross-section z-score so the boost survives
    ``VecNormalize`` (which would otherwise re-standardise it away).

    For each ``(prefix, weight)`` entry in ``feature_group_weights``,
    every factor column whose name starts with ``prefix`` is multiplied
    by ``weight`` along axis 2. Columns whose prefix is not in the dict
    keep an implicit weight of 1.0.

    Parameters
    ----------
    factor_array:
        Shape ``(n_dates, n_stocks, n_factors)``, modified in place.
    factor_names:
        Length ``n_factors``, column names ordered to match axis 2.
    feature_group_weights:
        Mapping prefix → scalar weight. ``None`` or empty is a no-op.
        Empty-string prefix ``""`` matches all columns. Prefixes not
        present in ``factor_names`` are silently ignored. Weight ``0.0``
        zeroes the column. Negative weights are allowed (flip signal).

    Returns
    -------
    The same ``factor_array`` reference (mutated in place for clarity).

    Raises
    ------
    TypeError
        If ``feature_group_weights`` is not a dict or maps to non-numeric
        values.
    """
    if not feature_group_weights:
        return factor_array

    if not isinstance(feature_group_weights, dict):
        raise TypeError(
            "feature_group_weights must be a dict[str, float], got "
            f"{type(feature_group_weights).__name__}"
        )

    for prefix, weight in feature_group_weights.items():
        if not isinstance(prefix, str):
            raise TypeError(
                "feature_group_weights keys must be str (factor prefix), got "
                f"{type(prefix).__name__}"
            )
        try:
            w = float(weight)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"feature_group_weights[{prefix!r}] must be a float, got "
                f"{weight!r} ({type(weight).__name__})"
            ) from e

        # Empty prefix matches everything; otherwise prefix-match column names.
        if prefix == "":
            col_idx = list(range(len(factor_names)))
        else:
            col_idx = [i for i, name in enumerate(factor_names) if name.startswith(prefix)]
        if not col_idx:
            # Silently ignore prefixes not present in the panel.
            continue
        factor_array[:, :, col_idx] *= w

    return factor_array


def _safe_log_return(price_now: np.ndarray, price_fwd: np.ndarray) -> np.ndarray:
    """Compute log return with NaN/zero-price safety."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (price_now > 0) & (price_fwd > 0),
            price_fwd / price_now,
            np.nan,
        )
        log_ret = np.where(
            np.isfinite(ratio) & (ratio > 0),
            np.log(ratio),
            0.0,
        )
    return log_ret.astype(np.float32)


def _compute_regime_features(
    pct_change: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Compute the 8 v0 regime features per :data:`REGIME_FEATURE_NAMES`.

    Parameters
    ----------
    pct_change:
        (T, S) decimal pct change (e.g. +10% = 0.10).
    valid_mask:
        (T, S) bool. Cells where the stock is suspended / pre-IPO / ST should
        be False so they don't contribute to cross-section stats. The mask
        used here MUST match the env's ``valid_mask`` so train- and OOS-time
        regime stats stay comparable.

    Returns
    -------
    np.ndarray of shape (T, 8), dtype float32, all finite.
    """
    T, S = pct_change.shape
    pct = pct_change.astype(np.float32, copy=False)
    valid = valid_mask.astype(np.bool_, copy=False)

    breadth_d = np.zeros(T, dtype=np.float32)
    xs_disp_d = np.zeros(T, dtype=np.float32)
    idx_ret_d = np.zeros(T, dtype=np.float32)
    extreme_imb = np.zeros(T, dtype=np.float32)

    for t in range(T):
        v = valid[t]
        n = int(v.sum())
        if n == 0:
            continue
        p = pct[t][v]
        breadth_d[t] = float((p > 0).mean())
        if n > 1:
            xs_disp_d[t] = float(p.std())
        idx_ret_d[t] = float(p.mean())
        up = int((p >= 0.099).sum())
        dn = int((p <= -0.099).sum())
        extreme_imb[t] = float(up - dn) / float(n)

    out = np.zeros((T, 8), dtype=np.float32)

    def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
        r = np.zeros_like(a)
        cs = np.cumsum(a, dtype=np.float64)
        for t in range(len(a)):
            lo = max(0, t - w + 1)
            seg_sum = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
            r[t] = float(seg_sum / float(t - lo + 1))
        return r.astype(np.float32, copy=False)

    out[:, 0] = breadth_d
    out[:, 1] = _rolling_mean(breadth_d, 20)
    out[:, 2] = xs_disp_d
    out[:, 3] = _rolling_mean(xs_disp_d, 20)

    log1p_idx = np.log1p(idx_ret_d.astype(np.float64))
    cs = np.cumsum(log1p_idx)
    for t in range(T):
        lo20 = max(0, t - 19)
        seg20 = cs[t] - (cs[lo20 - 1] if lo20 > 0 else 0.0)
        out[t, 4] = float(np.expm1(seg20))
        lo60 = max(0, t - 59)
        seg60 = cs[t] - (cs[lo60 - 1] if lo60 > 0 else 0.0)
        out[t, 5] = float(np.expm1(seg60))

    for t in range(T):
        lo = max(0, t - 19)
        seg = idx_ret_d[lo:t + 1]
        if len(seg) > 1:
            out[t, 6] = float(seg.std()) * float(np.sqrt(252.0))

    out[:, 7] = extreme_imb

    if not np.isfinite(out).all():
        col_idx = int(np.argmax(~np.isfinite(out).all(axis=0)))
        bad = REGIME_FEATURE_NAMES[col_idx] if col_idx < len(REGIME_FEATURE_NAMES) else f"col_{col_idx}"
        raise ValueError(
            f"_compute_regime_features produced non-finite values in column {bad!r}; "
            "check upstream pct_change for NaN / inf."
        )
    return out


def discover_factor_columns(
    df: pl.DataFrame,
    n_factors: int | None = None,
    prefixes: tuple[str, ...] = STOCK_FACTOR_PREFIXES,
) -> list[str]:
    """Return per-stock factor column names sorted alphabetically.

    Columns whose prefix appears in :data:`FORBIDDEN_PREFIXES` are silently
    dropped — the per-stock encoder must never see them (Phase 21 schema
    lock). Columns whose prefix is not in the allowlist are also dropped.

    Parameters
    ----------
    df:
        Input DataFrame.
    n_factors:
        If given, truncate to first N columns (alphabetical order).
        If None, return all matched columns.
    prefixes:
        Recognized prefixes (allowed columns).

    Returns
    -------
    Sorted list of factor column names.
    """
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(fp) for fp in FORBIDDEN_PREFIXES):
            continue
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    cols.sort()
    if n_factors is not None:
        cols = cols[:n_factors]
    return cols


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class FactorPanelLoader:
    """Load factor panels from a Parquet file.

    Parameters
    ----------
    parquet_path:
        Path to a Parquet file (single-file or hive-partitioned glob).
    """

    def __init__(self, parquet_path: str | Path = DEFAULT_PARQUET_PATH) -> None:
        self.parquet_path = Path(parquet_path)

    def load_panel(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        n_factors: int | None = None,
        forward_period: int = 10,
        universe_filter: UniverseFilter = UniverseFilter.MAIN_BOARD_NON_ST,
        feature_group_weights: dict[str, float] | None = None,
        factor_names: list[str] | None = None,
    ) -> FactorPanel:
        """Load a factor panel from Parquet.

        Parameters
        ----------
        start_date / end_date:
            Inclusive date range.
        n_factors:
            Number of factors to use (None = all matched).
        forward_period:
            Forward-return window in trading days.
        universe_filter:
            Stock universe filtering mode.
        feature_group_weights:
            Optional ``{prefix: weight}`` map applied AFTER the cross-section
            z-score (so the boost survives ``VecNormalize``). Factor columns
            whose name starts with ``prefix`` are multiplied by ``weight``.
            Columns without an explicit weight default to 1.0. ``None`` /
            ``{}`` are no-ops. See :func:`_apply_feature_group_weights`.
        factor_names:
            If given, load EXACTLY these columns in the EXACT order specified,
            bypassing prefix-based discovery and ``n_factors``. Required for
            OOS evaluation of a model trained with a fixed factor order: any
            column missing from the parquet raises. Use this when the panel
            schema has changed between train and eval (e.g., a new factor
            prefix was added) and you must preserve the model's input layout.

        Returns
        -------
        FactorPanel

        Raises
        ------
        FileNotFoundError if the Parquet file is missing.
        ValueError if no factor columns are found.
        TypeError if ``feature_group_weights`` is not a dict-of-str-to-float.
        """
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {self.parquet_path}\n"
                "Generate one with `scripts/generate_synthetic.py` (demo) "
                "or `scripts/export_factor_panel.py` (real data)."
            )

        return self._load_from_parquet(
            start_date=start_date,
            end_date=end_date,
            n_factors=n_factors,
            forward_period=forward_period,
            universe_filter=universe_filter,
            feature_group_weights=feature_group_weights,
            factor_names=factor_names,
        )

    def _load_from_parquet(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        n_factors: int | None,
        forward_period: int,
        universe_filter: UniverseFilter,
        feature_group_weights: dict[str, float] | None = None,
        factor_names: list[str] | None = None,
    ) -> FactorPanel:
        """Internal Parquet → FactorPanel conversion."""
        # Use polars scan for memory efficiency
        df_lazy = pl.scan_parquet(str(self.parquet_path))

        # Date filter
        df = df_lazy.filter(
            (pl.col("trade_date") >= start_date) & (pl.col("trade_date") <= end_date)
        ).collect()

        if df.is_empty():
            raise ValueError(
                f"No data in range {start_date}..{end_date}. "
                f"Parquet covers: {self.get_date_range()}"
            )

        # Validate required columns
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise ValueError(
                f"Required columns missing from Parquet: {missing_required}\n"
                f"Parquet must contain: {REQUIRED_COLUMNS}"
            )

        # Universe filter
        df = filter_universe(df, mode=universe_filter)
        if df.is_empty():
            raise ValueError(
                f"Universe filter {universe_filter} eliminated all rows. "
                "Check ts_code format or universe selection."
            )

        return self._df_to_panel(
            df,
            n_factors=n_factors,
            forward_period=forward_period,
            feature_group_weights=feature_group_weights,
            factor_names=factor_names,
        )

    def _df_to_panel(
        self,
        df: pl.DataFrame,
        n_factors: int | None,
        forward_period: int,
        feature_group_weights: dict[str, float] | None = None,
        factor_names: list[str] | None = None,
    ) -> FactorPanel:
        """Convert polars DataFrame to numpy 3D panel."""
        dates = df["trade_date"].unique().sort().to_list()
        stock_codes = df["ts_code"].unique().sort().to_list()

        n_dates = len(dates)
        n_stocks = len(stock_codes)

        # Discover factor columns. If `factor_names` is given, use that EXACT
        # list/order — required for OOS eval of models with a fixed input
        # layout. Otherwise prefix-discover.
        if factor_names is not None:
            df_cols = set(df.columns)
            missing = [c for c in factor_names if c not in df_cols]
            if missing:
                raise ValueError(
                    f"factor_names contains columns not in panel: {missing[:8]}"
                    f"{'...' if len(missing) > 8 else ''} "
                    f"(panel has {len(df.columns)} columns)"
                )
            factor_cols = list(factor_names)
        else:
            factor_cols = discover_factor_columns(df, n_factors=n_factors)
        if not factor_cols:
            raise ValueError(
                f"No factor columns found. Expected columns prefixed with "
                f"{FACTOR_COL_PREFIXES}. Got columns: {df.columns[:20]}..."
            )

        n_factors_actual = len(factor_cols)

        date_index = {d: i for i, d in enumerate(dates)}
        stock_index = {s: j for j, s in enumerate(stock_codes)}

        factor_array = np.zeros((n_dates, n_stocks, n_factors_actual), dtype=np.float32)
        close_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        pct_change_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        is_st_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
        # Phase 21: default to True (suspended). Only (t, j) cells that have a
        # parquet row are then UPDATED below — pre-IPO and delisted (t, j) stay
        # True. The previous default (False) silently let the env treat zero-padded
        # rows as tradeable, which contaminated cross-section centering and
        # `valid_mask` once n_stocks * (1 - listed_fraction) ≳ 5%.
        is_suspended_array = np.ones((n_dates, n_stocks), dtype=np.bool_)
        days_since_ipo_array = np.full(
            (n_dates, n_stocks), NEW_STOCK_PROTECT_DAYS * 2, dtype=np.float32
        )

        has_is_st = "is_st" in df.columns
        has_days_ipo = "days_since_ipo" in df.columns

        for row in df.iter_rows(named=True):
            t = date_index.get(row["trade_date"])
            j = stock_index.get(row["ts_code"])
            if t is None or j is None:
                continue

            for fi, col in enumerate(factor_cols):
                v = row.get(col)
                if v is not None:
                    factor_array[t, j, fi] = float(v)

            close_v = row.get("close")
            if close_v is not None:
                close_array[t, j] = float(close_v)
            pct_v = row.get("pct_chg")
            if pct_v is not None:
                pct_change_array[t, j] = float(pct_v)
            vol_v = row.get("vol")
            is_suspended_array[t, j] = (vol_v is None) or (vol_v == 0)

            if has_is_st:
                is_st_array[t, j] = bool(row.get("is_st") or False)
            if has_days_ipo:
                days_v = row.get("days_since_ipo")
                if days_v is not None:
                    days_since_ipo_array[t, j] = float(days_v)

        # Forward return
        return_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        for t in range(n_dates - forward_period):
            return_array[t] = _safe_log_return(close_array[t], close_array[t + forward_period])

        # Cross-section z-score
        factor_array = _cross_section_zscore(factor_array)

        # Optional per-prefix scalar weighting AFTER z-score so that a
        # subsequent VecNormalize wrapper cannot re-standardise the boost
        # away. See `_apply_feature_group_weights` for semantics.
        factor_array = _apply_feature_group_weights(
            factor_array, factor_cols, feature_group_weights
        )

        # Phase 21: regime tensor. Use the same valid_mask the env will use
        # so train- and eval-time regime stats are comparable.
        valid_for_regime = (
            (~is_st_array)
            & (~is_suspended_array)
            & (days_since_ipo_array >= NEW_STOCK_PROTECT_DAYS)
        )
        regime_array = _compute_regime_features(pct_change_array, valid_for_regime)

        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_cols,
            regime_array=regime_array,
            regime_names=tuple(REGIME_FEATURE_NAMES),
        )

    def get_date_range(self) -> tuple[datetime.date | None, datetime.date | None]:
        """Return (min_date, max_date) of the Parquet, or (None, None) if empty."""
        if not self.parquet_path.exists():
            return (None, None)
        try:
            df = (
                pl.scan_parquet(str(self.parquet_path))
                .select(
                    [
                        pl.col("trade_date").min().alias("min"),
                        pl.col("trade_date").max().alias("max"),
                    ]
                )
                .collect()
            )
            return (df["min"][0], df["max"][0])
        except Exception:
            return (None, None)

    @staticmethod
    def build_synthetic(
        n_dates: int = 500,
        n_stocks: int = 200,
        n_factors: int = 64,
        forward_period: int = 10,
        seed: int = 42,
        prefix: str = "alpha_",
        feature_group_weights: dict[str, float] | None = None,
    ) -> FactorPanel:
        """Build a synthetic panel for smoke testing — no real data needed.

        Useful for CI, demos, and smoke tests of the training pipeline.
        Stock codes are synthetic (``SYN_001`` etc.) — not real codes.

        ``feature_group_weights`` mirrors :meth:`load_panel` so unit tests
        can exercise the weighting path without writing a Parquet.
        """
        rng = np.random.default_rng(seed)

        factor_array = rng.standard_normal((n_dates, n_stocks, n_factors)).astype(np.float32)

        # Synthetic returns: factor mean + noise
        factor_mean = factor_array.mean(axis=2)
        return_array = (
            factor_mean * 0.01 + rng.standard_normal((n_dates, n_stocks)) * 0.02
        ).astype(np.float32)

        # Random pct change (decimal form)
        pct_change_array = (rng.standard_normal((n_dates, n_stocks)) * 0.02).astype(np.float32)

        # 5% ST, 2% suspended
        is_st_array = (rng.random((n_dates, n_stocks)) < 0.05).astype(np.bool_)
        is_suspended_array = (rng.random((n_dates, n_stocks)) < 0.02).astype(np.bool_)

        # All synthetic stocks are mature
        days_since_ipo_array = rng.integers(
            NEW_STOCK_PROTECT_DAYS * 2,
            NEW_STOCK_PROTECT_DAYS * 20,
            size=(n_dates, n_stocks),
        ).astype(np.float32)

        # Cross-section z-score
        factor_array = _cross_section_zscore(factor_array)

        # Synthetic codes (NOT real stock codes)
        factor_names = [f"{prefix}{i:03d}" for i in range(n_factors)]

        # Optional per-prefix weighting (parity with load_panel).
        factor_array = _apply_feature_group_weights(
            factor_array, factor_names, feature_group_weights
        )

        # Synthetic dates: weekdays starting 2020-01-01
        dates: list[datetime.date] = []
        current = datetime.date(2020, 1, 1)
        while len(dates) < n_dates:
            if current.weekday() < 5:
                dates.append(current)
            current += datetime.timedelta(days=1)

        # Synthetic codes (NOT real stock codes)
        stock_codes = [f"SYN_{i:05d}" for i in range(n_stocks)]

        # Phase 21: regime tensor (same valid_mask logic as _df_to_panel)
        valid_for_regime = (
            (~is_st_array)
            & (~is_suspended_array)
            & (days_since_ipo_array >= NEW_STOCK_PROTECT_DAYS)
        )
        regime_array = _compute_regime_features(pct_change_array, valid_for_regime)

        return FactorPanel(
            factor_array=factor_array,
            return_array=return_array,
            pct_change_array=pct_change_array,
            is_st_array=is_st_array,
            is_suspended_array=is_suspended_array,
            days_since_ipo_array=days_since_ipo_array,
            dates=dates,
            stock_codes=stock_codes,
            factor_names=factor_names,
            regime_array=regime_array,
            regime_names=tuple(REGIME_FEATURE_NAMES),
        )


__all__ = [
    "FactorPanel",
    "FactorPanelLoader",
    "UniverseFilter",
    "STOCK_FACTOR_PREFIXES",
    "FORBIDDEN_PREFIXES",
    "FACTOR_COL_PREFIXES",  # legacy alias
    "REGIME_FEATURE_NAMES",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "discover_factor_columns",
    "filter_universe",
    "align_panel_to_stock_list",
]
