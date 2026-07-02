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
* ``tech_*``    classical TA (MA / KDJ / MACD / Bollinger / ATR / amplitude)
* ``cmf_*``     cumulative main-force flow (60d / 120d, amount / active)
* ``zt_*``      limit-up streaks (count / first-board / max-streak / dt-imbalance)

The loader picks **all** matching columns (sorted alphabetically) up to
``n_factors``. Missing prefixes are silently skipped — RL never errors out
because of missing factor groups; the model just sees those positions as 0.

Universe filter
---------------
Default ``UniverseFilter.MAIN_BOARD_NON_ST`` excludes at load time:

* BSE (.BJ, codes 8/4)
* STAR market (688)
* ChiNext (300/301)

ST/退 exclusion is PER DATE, not load time (C3): rows stay in the panel and
``is_st_array`` (from the ``is_st`` column, or the row-level ``name`` as a
fallback) flags the dates a stock is actually ST. Downstream eligibility
masks — the env trading mask, train_v2's valid mask, the eval scripts —
exclude those (date, stock) cells. Dropping a stock's whole history because
its *current* name contains ST/退 is survivorship bias.

Other modes: ``ALL_A`` (no filter), ``HS300``, ``ZZ500``, ``ZZ1000``.

This module has **no PyTorch / gymnasium dependency**, safe to import in any
environment.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
import warnings
from functools import lru_cache
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

# Factor column prefixes (recognized by data_loader)
FACTOR_COL_PREFIXES: tuple[str, ...] = (
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
    "mkt_",
    "gtja_",
    "tech_",  # Phase 26 — classical TA (MA / KDJ / MACD / Bollinger / ATR)
    "cmf_",  # Phase 26 — cumulative main-force flow ratios
    "zt_",  # Phase 26 — limit-up streak metrics
)

# Required columns in input Parquet
REQUIRED_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "close", "pct_chg", "vol")

# Optional columns (used if present, defaulted otherwise)
OPTIONAL_COLUMNS: tuple[str, ...] = ("is_st", "days_since_ipo", "industry_code", "name")


class UniverseFilter(StrEnum):
    """Stock universe selection mode.

    Source of truth: paris's `2026-05-14-5-universe-lock/` + `2026-05-14-npf-v2-1-main-board/`
    bundles (cached locally at ``data/universes/<name>_membership.parquet``). When a
    membership parquet is absent the filter falls back to the regex-based main-board
    heuristic so synthetic and pre-lock panels still work.

    The five paris-locked universes (2026-05-14):
        * ``MAIN_BOARD`` — A-share main board, 3003 stocks (static).
        * ``CSI300`` — Hu-Shen 300 index, point-in-time (~300 stocks/day).
        * ``CSI500`` — CSI 500 index, point-in-time (~500 stocks/day).
        * ``NPF`` — 新质生产力 default, main-board only, 401 stocks (static, v2.1).
        * ``GROWTH_BOARDS`` — 创业板/科创板/北交所, 2253 stocks (static).

    NPF tiers (paris v2.1 main-board-locked):
        * ``NPF`` — default, 401 stocks (Layer 1A + 1B, main board only).
        * ``NPF_FULL`` — 618 stocks (+ Layer 2 hot cross + Layer 3 concept backfill).
        * ``NPF_CROSS_BOARD`` — 779 stocks, cross-board (exploration only).

    Backward-compat aliases:
        * ``MAIN_BOARD_NON_ST`` → ``MAIN_BOARD`` (legacy name, same set).
        * ``HS300`` → ``CSI300`` (legacy spelling).
        * ``ZZ500`` → ``CSI500`` (legacy spelling).
    """

    ALL_A = "all_a"
    # New paris-locked names
    MAIN_BOARD = "main_board"
    CSI300 = "csi300"
    CSI500 = "csi500"
    NPF = "npf"
    NPF_FULL = "npf_full"
    NPF_CROSS_BOARD = "npf_cross_board"
    GROWTH_BOARDS = "growth_boards"
    # Legacy aliases — kept for backward compat with older training scripts
    MAIN_BOARD_NON_ST = "main_board_non_st"
    HS300 = "hs300"
    ZZ500 = "zz500"
    ZZ1000 = "zz1000"


_LEGACY_ALIAS = {
    UniverseFilter.MAIN_BOARD_NON_ST: UniverseFilter.MAIN_BOARD,
    UniverseFilter.HS300: UniverseFilter.CSI300,
    UniverseFilter.ZZ500: UniverseFilter.CSI500,
}


# Static universes use just stock_code, trade_date is NULL.
#
# LIMITATION (C3, disclosed — not fixed): these are date-less membership
# snapshots locked on ``STATIC_UNIVERSE_LOCK_DATE`` and applied to FULL
# history. Stocks that delisted before the lock date are absent from every
# historical date, which inflates historical labels/backtests (survivorship
# bias). True point-in-time membership (as CSI300/CSI500 already have) needs
# upstream data this repo does not ship; until then :func:`filter_universe`
# emits a UserWarning whenever a static universe is applied to a panel that
# starts meaningfully before the lock date.
_STATIC_UNIVERSES = frozenset(
    {
        UniverseFilter.MAIN_BOARD,
        UniverseFilter.NPF,
        UniverseFilter.NPF_FULL,
        UniverseFilter.NPF_CROSS_BOARD,
        UniverseFilter.GROWTH_BOARDS,
    }
)

# Membership lock date of the static universes (paris `2026-05-14-5-universe-lock`
# / `2026-05-14-npf-v2-1-main-board` bundles).
STATIC_UNIVERSE_LOCK_DATE = datetime.date(2026, 5, 14)

# Panels starting more than this many days before the lock date trigger the
# survivorship disclosure warning ("meaningfully before" the lock).
_SURVIVORSHIP_WARN_GRACE_DAYS = 30

# Point-in-time universes use (stock_code, trade_date) — index membership rebalances quarterly.
_PIT_UNIVERSES = frozenset({UniverseFilter.CSI300, UniverseFilter.CSI500})


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


class FactorPanel(NamedTuple):
    """Container for a 3D factor panel + auxiliary arrays.

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
    close_array:
        shape (n_dates, n_stocks), RAW (unadjusted) close prices with NaN
        for missing cells, or ``None`` when the source has no prices
        (e.g. the synthetic panel). Used by
        :func:`aurumq_rl.price_limits.compute_at_limit_masks` to
        reconstruct rounded limit prices (M7); price-limit rules operate
        on raw exchange prices, so this stays unadjusted.
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
    close_array: np.ndarray | None = None


def align_panel_to_stock_list(panel: FactorPanel, target_stock_codes: list[str]) -> FactorPanel:
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

    panel.factor_array.shape[0]
    panel.factor_array.shape[2]
    n_target = len(target_stock_codes)

    # Build idx map: target_idx -> source_idx (or -1 for missing)
    src_idx_by_code = {c: i for i, c in enumerate(panel.stock_codes)}
    idx_map = np.array([src_idx_by_code.get(c, -1) for c in target_stock_codes], dtype=np.int64)
    present = idx_map >= 0
    int((~present).sum())

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
    is_st_array = _gather(panel.is_st_array, True)  # missing → ST (un-tradeable)
    is_suspended_array = _gather(panel.is_suspended_array, True)  # missing → suspended
    days_since_ipo_array = _gather(panel.days_since_ipo_array, 0)
    # missing → NaN close (price-limit detection falls back to pct epsilon)
    close_array = _gather(panel.close_array, np.nan) if panel.close_array is not None else None

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
        close_array=close_array,
    )


def build_tradeable_mask(panel: FactorPanel) -> np.ndarray:
    """(T, S) bool mask of cells eligible for ENTRY at decision date t.

    ``~suspended & ~ST & (days_since_ipo >= 60) & ~at_limit_up & ~at_limit_down``

    SINGLE SOURCE OF TRUTH (C4/M5): the GPU training valid_mask
    (scripts/train_v2.py) and the backtest eval scripts
    (eval_backtest / _eval_all_checkpoints / _ensemble_eval) must both use
    this function so training and evaluation agree exactly.

    Parity rule: matches the CPU env's ``env._apply_trading_mask`` — BOTH
    limit-up and limit-down closes are untradeable (a limit-up close cannot
    be bought; a limit-down close is treated symmetrically per the CPU env).
    """
    from aurumq_rl.price_limits import compute_at_limit_masks

    at_up, at_down = compute_at_limit_masks(
        pct_chg=panel.pct_change_array,
        stock_codes=list(panel.stock_codes),
        is_st=panel.is_st_array,
        days_since_ipo=panel.days_since_ipo_array,
        close=panel.close_array,
    )
    return (
        (~panel.is_st_array)
        & (~panel.is_suspended_array)
        & (panel.days_since_ipo_array >= NEW_STOCK_PROTECT_DAYS)
        & ~at_up
        & ~at_down
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


def _universe_dir() -> Path:
    """Directory holding paris-shipped universe membership parquets."""
    env = os.environ.get("AURUMQ_UNIVERSE_DIR")
    if env:
        return Path(env)
    # Default: <project_root>/data/universes/
    return Path(__file__).resolve().parents[2] / "data" / "universes"


@lru_cache(maxsize=16)
def _load_static_universe(name: str) -> frozenset[str]:
    """Load a static universe (NPF / MAIN_BOARD / etc.) from local parquet."""
    path = _universe_dir() / f"{name}_membership.parquet"
    if not path.exists():
        return frozenset()
    df = pl.read_parquet(path)
    code_col = "stock_code" if "stock_code" in df.columns else "ts_code"
    return frozenset(df[code_col].to_list())


@lru_cache(maxsize=4)
def _load_pit_universe(name: str) -> pl.DataFrame:
    """Load a point-in-time universe (CSI300 / CSI500) — schema (stock_code, trade_date)."""
    path = _universe_dir() / f"{name}_membership.parquet"
    if not path.exists():
        return pl.DataFrame(schema={"stock_code": pl.Utf8, "trade_date": pl.Date})
    df = pl.read_parquet(path)
    code_col = "stock_code" if "stock_code" in df.columns else "ts_code"
    # Normalize column name
    if code_col != "stock_code":
        df = df.rename({code_col: "stock_code"})
    if df["trade_date"].dtype != pl.Date:
        df = df.with_columns(pl.col("trade_date").cast(pl.Date))
    return df


def _warn_static_membership_survivorship(df: pl.DataFrame, mode: UniverseFilter) -> None:
    """Emit the C3 survivorship disclosure for static membership snapshots.

    Fires when a date-less locked membership set is applied to a panel whose
    date range starts meaningfully before ``STATIC_UNIVERSE_LOCK_DATE``:
    stocks delisted before the lock date are absent from ALL dates, so
    historical windows carry survivorship bias.
    """
    if "trade_date" not in df.columns or df.is_empty():
        return
    start = df["trade_date"].min()
    if isinstance(start, datetime.datetime):
        start = start.date()
    if not isinstance(start, datetime.date):
        return
    grace = datetime.timedelta(days=_SURVIVORSHIP_WARN_GRACE_DAYS)
    if start >= STATIC_UNIVERSE_LOCK_DATE - grace:
        return
    warnings.warn(
        f"Universe '{mode.value}' is a static membership snapshot locked on "
        f"{STATIC_UNIVERSE_LOCK_DATE}; stocks delisted before the lock date are "
        f"absent from ALL dates, so this panel (starts {start}) carries "
        "survivorship bias in historical windows. Point-in-time membership "
        "(as CSI300/CSI500 have) is the real fix.",
        UserWarning,
        stacklevel=3,
    )


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
        Unused; retained for backward compatibility. ST exclusion is no
        longer name-based at load time — it is enforced PER DATE via the
        panel's ``is_st`` column and the downstream eligibility masks (C3:
        dropping a stock's whole history for a *current* ST/退 name is
        survivorship bias).

    Returns
    -------
    Filtered dataframe.
    """
    if mode == UniverseFilter.ALL_A:
        return df

    # Resolve legacy aliases (HS300 → CSI300, ZZ500 → CSI500, MAIN_BOARD_NON_ST → MAIN_BOARD)
    mode = _LEGACY_ALIAS.get(mode, mode)

    # Static universe filter from paris's locked membership parquets.
    # NOTE: board membership (code patterns / locked sets) is applied at load
    # time; ST-ness is NOT — rows stay in the panel and `is_st_array` carries
    # the per-date flag for the env/train/eval eligibility masks.
    if mode in _STATIC_UNIVERSES:
        codes = _load_static_universe(mode.value.upper())
        if codes:
            _warn_static_membership_survivorship(df, mode)
            return df.filter(pl.col("ts_code").is_in(list(codes)))
        # Fall back to regex when membership parquet missing (synthetic / pre-lock
        # data). Code-based board patterns are time-invariant — no snapshot bias.
        return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))

    # Point-in-time universe (CSI300 / CSI500): require trade_date column for proper
    # per-day membership. Prefer explicit `is_csi300` boolean column if present
    # (matches the input-data contract in CLAUDE.md); else inner-join with the PIT
    # membership parquet; else final fallback to is_hs300/is_zz500 column or
    # main-board regex.
    if mode in _PIT_UNIVERSES:
        legacy_col = {UniverseFilter.CSI300: "is_hs300", UniverseFilter.CSI500: "is_zz500"}[mode]
        if legacy_col in df.columns:
            return df.filter(pl.col(legacy_col) == True)  # noqa: E712
        pit = _load_pit_universe(mode.value.upper())
        if len(pit) > 0 and "trade_date" in df.columns:
            return df.join(
                pit.rename({"stock_code": "ts_code"}),
                on=["ts_code", "trade_date"],
                how="inner",
            )
        # Last-resort fallback: main-board heuristic
        return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))

    # ZZ1000 / unknown — fall back to main-board heuristic for safety.
    return df.filter(pl.col("ts_code").map_elements(_is_main_board, return_dtype=pl.Boolean))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cross_section_zscore(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score normalize along axis=1 (stock dim).

    For each (date, factor), normalize across stocks:
        z = (x - mean) / (std + 1e-8)

    NaN and ±inf cells in the input (suspended days, pre-IPO, factor warm-up,
    upstream factor-computation overflow such as gtja_005/017/114, alpha_045)
    and rows where the entire cross-section is NaN (so mean/std are NaN) are
    replaced with 0.0, the neutral signal per the project convention.
    Without this, env reset() returns observations containing NaN, which
    Box.contains() rejects and SB3's check_env asserts on.

    Inf-protection: a single +inf in a (date, factor) cross-section would
    propagate through nanmean -> std -> z and zero out the ENTIRE cross-
    section under nan_to_num, silently dropping ~3000 stocks of signal for
    that day. Replacing inf with nan up-front lets nanmean/nanstd skip it
    like a regular missing value, so only the offending stock's z is set
    to 0 rather than the whole column. Phase 26 audit found ~22 factors
    with inf rates 1e-5..2.5%; gtja_005 alone had 108k inf cells.
    """
    if not np.all(np.isfinite(arr) | np.isnan(arr)):
        arr = np.where(np.isinf(arr), np.nan, arr)

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
    """Compute log return; invalid prices (NaN / <= 0) yield NaN, not 0.

    Returning 0.0 for missing prices used to make absent (date, stock) cells
    look like tradeable zero-return observations, biasing rewards/backtests.
    Downstream consumers mask on ``np.isfinite`` (or nan-guard) instead.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (price_now > 0) & (price_fwd > 0),
            price_fwd / price_now,
            np.nan,
        )
        log_ret = np.where(
            np.isfinite(ratio) & (ratio > 0),
            np.log(ratio),
            np.nan,
        )
    return log_ret.astype(np.float32)


def pivot_adjusted_close(
    df: pl.DataFrame,
    stock_codes: list[str],
    dates: list[datetime.date],
) -> np.ndarray:
    """Pivot close (× ``adj_factor`` when present) to a dense (T, S) array.

    Shared by the wave-label scripts (``train_v2.py``,
    ``_eval_main_wave_v1.py``, ``_eval_main_wave_episode.py``) so their
    label / MA computations run on ADJUSTED close and stay correct across
    corporate actions. When the parquet lacks ``adj_factor`` the raw close
    is used (legacy behavior; the loader-level warning covers the bias).

    Parameters
    ----------
    df:
        Long-format frame with ``trade_date, ts_code, close`` and optionally
        ``adj_factor`` (already date-filtered by the caller).
    stock_codes:
        Column order of the output; codes absent from ``df`` yield zeros.
    dates:
        Row order of the output; dates absent from ``df`` are skipped
        (mirrors the scripts' historical re-alignment behavior).

    Returns
    -------
    float32 array of shape (len(dates present in df), len(stock_codes)),
    missing cells filled with 0.0 (legacy script convention). Cells with a
    present close but a null adj_factor are NaN (invalid — matches the
    loader, which never mixes raw and adjusted prices in one series).
    """
    if "adj_factor" in df.columns:
        # Loader-matching null semantics: a null CLOSE is a missing quote and
        # keeps the legacy 0.0 pivot convention (null propagates through the
        # product into the post-pivot fill_null). A present close with a null
        # ADJ_FACTOR must NOT become a raw-price or 0.0 point inside an
        # otherwise adjusted series (that fabricates returns / MA values) —
        # emit float NaN, which fill_null leaves alone, so downstream treats
        # the cell as invalid.
        df = df.with_columns(
            pl.when(pl.col("adj_factor").is_null())
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("close") * pl.col("adj_factor"))
            .alias("_adj_close")
        )
        field = "_adj_close"
    else:
        field = "close"
    piv = (
        df.select(["trade_date", "ts_code", field])
        .pivot(values=field, index="trade_date", on="ts_code")
        .sort("trade_date")
    )
    existing = {c for c in piv.columns if c != "trade_date"}
    arrs: list[np.ndarray] = []
    for code in stock_codes:
        if code in existing:
            col = piv.get_column(code).fill_null(0.0).to_numpy()
        else:
            col = np.zeros(piv.height, dtype=np.float32)
        arrs.append(col.astype(np.float32, copy=False))
    stacked = np.stack(arrs, axis=1)
    piv_dates = piv.get_column("trade_date").to_list()
    d2r = {d: i for i, d in enumerate(piv_dates)}
    idx = [d2r[d] for d in dates if d in d2r]
    return stacked[idx]


def discover_factor_columns(
    df: pl.DataFrame,
    n_factors: int | None = None,
    prefixes: tuple[str, ...] = FACTOR_COL_PREFIXES,
) -> list[str]:
    """Discover factor columns in a DataFrame by prefix matching.

    Parameters
    ----------
    df:
        Input DataFrame.
    n_factors:
        If given, truncate to first N columns (alphabetical order).
        If None, return all matched columns.
    prefixes:
        Recognized prefixes.

    Returns
    -------
    Sorted list of factor column names.
    """
    matched = sorted([c for c in df.columns if c.startswith(prefixes)])
    if n_factors is not None:
        return matched[:n_factors]
    return matched


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

        # Missing (date, stock) cells must NOT look like tradeable zeros:
        # factor / close default to NaN (excluded from z-score stats, NaN
        # forward return), is_suspended defaults to True and days_since_ipo
        # to 0 (fails the 60-day IPO gate) — mirroring the semantics
        # `align_panel_to_stock_list` uses for whole-stock misses. Cells with
        # actual rows overwrite these defaults below.
        factor_array = np.full((n_dates, n_stocks, n_factors_actual), np.nan, dtype=np.float32)
        close_array = np.full((n_dates, n_stocks), np.nan, dtype=np.float32)
        pct_change_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        is_st_array = np.zeros((n_dates, n_stocks), dtype=np.bool_)
        is_suspended_array = np.ones((n_dates, n_stocks), dtype=np.bool_)
        days_since_ipo_array = np.zeros((n_dates, n_stocks), dtype=np.float32)

        has_is_st = "is_st" in df.columns
        if not has_is_st and "name" in df.columns:
            # Per-date ST fallback (C3): with no `is_st` column, derive the
            # flag per ROW from the name. When the exporter stores historical
            # names this is point-in-time; with current-name snapshots it
            # degrades to the old name-based behavior, but confined to the
            # eligibility mask instead of erasing pre-ST history from the panel.
            df = df.with_columns(
                pl.col("name")
                .cast(pl.Utf8)
                .str.contains(_ST_NAME_PATTERN.pattern)
                .fill_null(False)
                .alias("is_st")
            )
            has_is_st = True
        has_days_ipo = "days_since_ipo" in df.columns
        # `adj_factor` is optional: when present, forward returns are computed
        # on adjusted close (close * adj_factor) so corporate actions
        # (dividends / splits) don't fabricate large fake returns. The raw
        # `close` array is kept UNCHANGED — price-limit logic, pct_chg and
        # amount stay on raw prices.
        has_adj_factor = "adj_factor" in df.columns
        adj_close_array = (
            np.full((n_dates, n_stocks), np.nan, dtype=np.float32) if has_adj_factor else None
        )

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
            if adj_close_array is not None:
                adj_v = row.get("adj_factor")
                if close_v is not None and adj_v is not None:
                    adj_close_array[t, j] = float(close_v) * float(adj_v)
            pct_v = row.get("pct_chg")
            if pct_v is not None:
                pct_change_array[t, j] = float(pct_v)
            vol_v = row.get("vol")
            is_suspended_array[t, j] = (vol_v is None) or (vol_v == 0)

            if has_is_st:
                is_st_array[t, j] = bool(row.get("is_st") or False)
            # Rows that exist but carry no days_since_ipo info are treated as
            # mature (legacy default); only ABSENT cells keep the 0 default.
            days_v = row.get("days_since_ipo") if has_days_ipo else None
            days_since_ipo_array[t, j] = (
                float(days_v) if days_v is not None else NEW_STOCK_PROTECT_DAYS * 2
            )

        # Forward return — on ADJUSTED close when adj_factor is available.
        # The per-stock scaling constant cancels in the ratio, so no
        # rebasing is needed.
        if adj_close_array is not None:
            price_for_returns = adj_close_array
        else:
            warnings.warn(
                "Panel parquet has no 'adj_factor' column: forward returns are "
                "computed on UNADJUSTED close prices and are CORRUPTED around "
                "corporate actions (dividends/splits/rights). Re-export the "
                "panel with adj_factor (scripts/export_factor_panel.py).",
                UserWarning,
                stacklevel=2,
            )
            price_for_returns = close_array
        return_array = np.zeros((n_dates, n_stocks), dtype=np.float32)
        for t in range(n_dates - forward_period):
            return_array[t] = _safe_log_return(
                price_for_returns[t], price_for_returns[t + forward_period]
            )

        # Cross-section z-score
        factor_array = _cross_section_zscore(factor_array)

        # Optional per-prefix scalar weighting AFTER z-score so that a
        # subsequent VecNormalize wrapper cannot re-standardise the boost
        # away. See `_apply_feature_group_weights` for semantics.
        factor_array = _apply_feature_group_weights(
            factor_array, factor_cols, feature_group_weights
        )

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
            # RAW close (NaN for missing cells) — price limits use exchange
            # prices, never adjusted ones.
            close_array=close_array,
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
        )


__all__ = [
    "FactorPanel",
    "FactorPanelLoader",
    "STATIC_UNIVERSE_LOCK_DATE",
    "UniverseFilter",
    "FACTOR_COL_PREFIXES",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "discover_factor_columns",
    "filter_universe",
    "pivot_adjusted_close",
    "align_panel_to_stock_list",
    "build_tradeable_mask",
]
