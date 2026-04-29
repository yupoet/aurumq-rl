"""Alpha101 — momentum category factors.

This module implements the WorldQuant 101 alphas that the original
``aurumq.rules.alpha101_library`` classifies as ``category='momentum'``,
plus the two project-internal customs (``alpha_custom_decaylinear_mom``
and ``alpha_custom_argmax_recent``).

Each function takes a sorted (``stock_code``, ``trade_date``) panel
:class:`pl.DataFrame` and returns a :class:`pl.Series` aligned to its
rows. Functions that mix time-series (``stock_code`` partitioned) and
cross-section (``trade_date`` partitioned) operators must materialise an
intermediate column first — polars cannot reliably nest different
``over(...)`` partitions inside one expression. Helpers from
:mod:`aurumq_rl.factors.alpha101._ops` are reused throughout.

All factors here are flagged ``direction='reverse'`` in
:data:`aurumq.rules.alpha101_library.ALPHA101_FACTORS`; that flag is
preserved on registration.
"""
from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    cs_rank,
    delay,
    delta,
    if_then_else,
    sign_,
    signed_power,
    ts_argmax,
    ts_corr,
    ts_decay_linear,
    ts_max,
    ts_min,
    ts_rank,
    ts_sum,
)

# ---------------------------------------------------------------------------
# alpha007 — Volume-conditional 7-day signed momentum rank
# ---------------------------------------------------------------------------


def alpha007(panel: pl.DataFrame) -> pl.Series:
    """Alpha #007 — Volume-conditional 7-day signed momentum rank.

    WorldQuant Formula (Kakushadze 2015, eq. 7)
    -------------------------------------------
        ((adv20 < volume) ? -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1)

    Legacy AQML Expression
    ----------------------
        If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * Sign(Delta(close, 7)), -1)

    Polars Implementation Notes
    ---------------------------
    1. ``delta(close, 7)`` is a per-stock 7-day price change.
    2. ``ts_rank(abs(...), 60)`` is a per-stock 60-window rank in [0, 1].
    3. The condition selects between the rank-based momentum reversal and
       a constant ``-1``. When volume is below ``adv20``, the alpha collapses
       to a flat ``-1`` (i.e. the day carries no signal apart from the
       constant).

    Required panel columns: ``adv20``, ``volume``, ``close``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d7 = delta(pl.col("close"), 7)
    rank60 = ts_rank(d7.abs(), 60)
    branch = -1.0 * rank60 * sign_(d7)
    expr = if_then_else(pl.col("adv20") < pl.col("volume"), branch, pl.lit(-1.0))
    return panel.select(expr.alias("alpha007").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha008 — Acceleration of (open*returns) sum compared to 10 days ago
# ---------------------------------------------------------------------------


def alpha008(panel: pl.DataFrame) -> pl.Series:
    """Alpha #008 — Acceleration of (open · returns) sum compared to 10 days ago.

    WorldQuant Formula (Kakushadze 2015, eq. 8)
    -------------------------------------------
        -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))

    Legacy AQML Expression
    ----------------------
        -1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))

    Polars Implementation Notes
    ---------------------------
    1. Build the per-stock 5-day rolling sum of ``open`` and ``returns``.
    2. Multiply pointwise -> momentum proxy.
    3. Subtract the 10-day-ago version -> acceleration.
    4. Materialise before ``cs_rank`` (CS partition differs from TS).

    Required panel columns: ``open``, ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    sum_open = ts_sum(pl.col("open"), 5)
    sum_ret = ts_sum(pl.col("returns"), 5)
    product = sum_open * sum_ret
    accel = product - delay(product, 10)
    staged = panel.with_columns(accel.alias("__a008_accel"))
    return staged.select(
        (-1.0 * cs_rank(pl.col("__a008_accel"))).alias("alpha008").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# alpha009 — Trend-confirmed price-change momentum
# ---------------------------------------------------------------------------


def alpha009(panel: pl.DataFrame) -> pl.Series:
    """Alpha #009 — Trend-confirmed price-change momentum.

    WorldQuant Formula (Kakushadze 2015, eq. 9)
    -------------------------------------------
        ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
         ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

    Legacy AQML Expression
    ----------------------
        If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1),
           If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1),
              -1 * Delta(close, 1)))

    Polars Implementation Notes
    ---------------------------
    1. If the past 5-day minimum daily change is positive (consistent up
       trend), pass-through the daily delta.
    2. Else if the past 5-day maximum daily change is negative (consistent
       down trend), still pass-through the daily delta.
    3. Otherwise (mixed regime) flip sign of the daily delta — i.e. mean
       reversion within choppy markets.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d1 = delta(pl.col("close"), 1)
    inner = if_then_else(ts_max(d1, 5) < 0.0, d1, -1.0 * d1)
    expr = if_then_else(ts_min(d1, 5) > 0.0, d1, inner)
    return panel.select(expr.alias("alpha009").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha010 — Cross-sectional rank of trend-confirmed price change
# ---------------------------------------------------------------------------


def alpha010(panel: pl.DataFrame) -> pl.Series:
    """Alpha #010 — Cross-sectional rank of trend-confirmed price change.

    WorldQuant Formula (Kakushadze 2015, eq. 10)
    --------------------------------------------
        rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
              ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))

    Legacy AQML Expression
    ----------------------
        Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1),
               If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1),
                  -1 * Delta(close, 1))))

    Polars Implementation Notes
    ---------------------------
    Same logic as :func:`alpha009` but with a 4-day lookback for the
    trend confirmation, then cross-section ranked. Materialise before
    ranking so that the TS lookback finishes before the CS partition.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d1 = delta(pl.col("close"), 1)
    inner = if_then_else(ts_max(d1, 4) < 0.0, d1, -1.0 * d1)
    inner_branch = if_then_else(ts_min(d1, 4) > 0.0, d1, inner)
    staged = panel.with_columns(inner_branch.alias("__a010_branch"))
    return staged.select(
        cs_rank(pl.col("__a010_branch")).alias("alpha010").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# alpha017 — Momentum exhaustion: rank-mom × second-derivative × volume-surge
# ---------------------------------------------------------------------------


def alpha017(panel: pl.DataFrame) -> pl.Series:
    """Alpha #017 — Momentum exhaustion: rank-mom × second-derivative × volume-surge.

    WorldQuant Formula (Kakushadze 2015, eq. 17)
    --------------------------------------------
        (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
         rank(ts_rank((volume / adv20), 5)))

    Legacy AQML Expression
    ----------------------
        (-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) *
        Rank(Ts_Rank(volume / adv20, 5))

    Polars Implementation Notes
    ---------------------------
    1. Three components, each a CS rank of a TS quantity. We materialise
       the three TS columns first, then cross-section rank, then multiply.
    2. ``delta(delta(close, 1), 1)`` is the discrete second derivative —
       acceleration of price.

    Required panel columns: ``close``, ``volume``, ``adv20``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    tsr_close = ts_rank(pl.col("close"), 10)
    accel = delta(delta(pl.col("close"), 1), 1)
    vol_surge = ts_rank(pl.col("volume") / pl.col("adv20"), 5)
    staged = panel.with_columns(
        tsr_close.alias("__a017_a"),
        accel.alias("__a017_b"),
        vol_surge.alias("__a017_c"),
    )
    expr = (
        (-1.0 * cs_rank(pl.col("__a017_a")))
        * cs_rank(pl.col("__a017_b"))
        * cs_rank(pl.col("__a017_c"))
    )
    return staged.select(expr.alias("alpha017").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha019 — 7d-return sign times annual rank-mom multiplier
# ---------------------------------------------------------------------------


def alpha019(panel: pl.DataFrame) -> pl.Series:
    """Alpha #019 — 7d-return sign times annual rank-mom multiplier.

    WorldQuant Formula (Kakushadze 2015, eq. 19)
    --------------------------------------------
        ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
         (1 + rank((1 + sum(returns, 250)))))

    Legacy AQML Expression
    ----------------------
        (-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) *
        (1 + Rank(1 + Ts_Sum(returns, 250)))

    Polars Implementation Notes
    ---------------------------
    1. ``close - delay(close, 7)`` and ``delta(close, 7)`` are
       algebraically identical; the WorldQuant paper writes both for
       robustness. We follow the formula verbatim — adding the same
       quantity twice doubles the magnitude but keeps the sign.
    2. The annual ``ts_sum(returns, 250)`` rank-multiplier requires a
       250-day window that is longer than the typical synthetic panel
       (60 days), so the output is null for the first ~250 rows of each
       stock. STHSF reference is computed on the same panel and shows
       the same null pattern.

    Required panel columns: ``close``, ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    sign_part = -1.0 * sign_((pl.col("close") - delay(pl.col("close"), 7)) + delta(pl.col("close"), 7))
    annual = ts_sum(pl.col("returns"), 250)
    staged = panel.with_columns(
        sign_part.alias("__a019_sign"),
        annual.alias("__a019_ann"),
    )
    expr = pl.col("__a019_sign") * (1.0 + cs_rank(1.0 + pl.col("__a019_ann")))
    return staged.select(expr.alias("alpha019").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha038 — Rank of 10d close rolling rank times close-over-open rank, negated
# ---------------------------------------------------------------------------


def alpha038(panel: pl.DataFrame) -> pl.Series:
    """Alpha #038 — Rank of 10d close rolling rank times close-over-open rank, negated.

    WorldQuant Formula (Kakushadze 2015, eq. 38)
    --------------------------------------------
        ((-1 * rank(ts_rank(close, 10))) * rank((close / open)))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)

    Polars Implementation Notes
    ---------------------------
    1. Two CS ranks multiplied; each consumes a TS-column input. Materialise
       intermediates so the two ``cs_rank`` partitions don't collide.

    Required panel columns: ``close``, ``open``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    tsr_close = ts_rank(pl.col("close"), 10)
    staged = panel.with_columns(tsr_close.alias("__a038_tsr"))
    expr = -1.0 * cs_rank(pl.col("__a038_tsr")) * cs_rank(pl.col("close") / pl.col("open"))
    return staged.select(expr.alias("alpha038").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha045 — Lagged-MA rank × short-corr × long/short-MA-corr rank, negated
# ---------------------------------------------------------------------------


def alpha045(panel: pl.DataFrame) -> pl.Series:
    """Alpha #045 — Lagged-MA rank × short-corr × long/short-MA-corr rank, negated.

    WorldQuant Formula (Kakushadze 2015, eq. 45)
    --------------------------------------------
        (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
               rank(correlation(sum(close, 5), sum(close, 20), 2))))

    Legacy AQML Expression
    ----------------------
        -1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) *
              Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))

    Polars Implementation Notes
    ---------------------------
    1. ``Ts_Sum(Delay(close, 5), 20) / 20`` -> per-stock lagged 20d MA.
    2. Two ``Ts_Corr`` calls inside, both 2-window — they're noisy by design.
    3. Two CS ranks; materialise the lagged-MA and the long/short-MA
       correlation before ranking.

    Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    lagged_ma = ts_sum(delay(pl.col("close"), 5), 20) / 20.0
    short_corr = ts_corr(pl.col("close"), pl.col("volume"), 2)
    long_short_corr = ts_corr(ts_sum(pl.col("close"), 5), ts_sum(pl.col("close"), 20), 2)
    staged = panel.with_columns(
        lagged_ma.alias("__a045_ma"),
        short_corr.alias("__a045_sc"),
        long_short_corr.alias("__a045_lsc"),
    )
    expr = -1.0 * (
        cs_rank(pl.col("__a045_ma")) * pl.col("__a045_sc")
    ) * cs_rank(pl.col("__a045_lsc"))
    return staged.select(expr.alias("alpha045").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha046 — Trend-shape conditional one-day reversal
# ---------------------------------------------------------------------------


def alpha046(panel: pl.DataFrame) -> pl.Series:
    """Alpha #046 — Trend-shape conditional one-day reversal.

    WorldQuant Formula (Kakushadze 2015, eq. 46)
    --------------------------------------------
        ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
                  ((delay(close, 10) - close) / 10))) ? -1 :
         ((((delay(close, 20) - delay(close, 10)) / 10) -
           ((delay(close, 10) - close) / 10)) < 0) ? 1 :
         (-1 * (close - delay(close, 1))))

    Legacy AQML Expression
    ----------------------
        If((Delay(close, 20) - Delay(close, 10)) / 10 -
           (Delay(close, 10) - close) / 10 > 0.25, -1,
           If((Delay(close, 20) - Delay(close, 10)) / 10 -
              (Delay(close, 10) - close) / 10 < 0, 1,
              -1 * (close - Delay(close, 1))))

    Polars Implementation Notes
    ---------------------------
    1. Compute the ``trend curvature`` once and reuse the materialised
       column twice in the nested ``if``.
    2. Three branches: strong upward curvature -> -1; downward curvature
       -> 1; otherwise reverse the daily change.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    seg_a = (delay(pl.col("close"), 20) - delay(pl.col("close"), 10)) / 10.0
    seg_b = (delay(pl.col("close"), 10) - pl.col("close")) / 10.0
    curvature = seg_a - seg_b
    inner = if_then_else(
        curvature < 0.0,
        pl.lit(1.0),
        -1.0 * (pl.col("close") - delay(pl.col("close"), 1)),
    )
    expr = if_then_else(curvature > 0.25, pl.lit(-1.0), inner)
    return panel.select(expr.alias("alpha046").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha051 — Trend curvature conditional reversal
# ---------------------------------------------------------------------------


def alpha051(panel: pl.DataFrame) -> pl.Series:
    """Alpha #051 — Trend curvature conditional reversal.

    WorldQuant Formula (Kakushadze 2015, eq. 51)
    --------------------------------------------
        (((((delay(close, 20) - delay(close, 10)) / 10) -
           ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 :
         (-1 * (close - delay(close, 1))))

    Legacy AQML Expression
    ----------------------
        If((Delay(close, 20) - Delay(close, 10)) / 10 -
           (Delay(close, 10) - close) / 10 < -0.05, 1,
           -1 * (close - Delay(close, 1)))

    Polars Implementation Notes
    ---------------------------
    1. Same curvature definition as :func:`alpha046` but with a single
       conditional: ``curvature < -0.05`` -> +1, else daily-reversal.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    seg_a = (delay(pl.col("close"), 20) - delay(pl.col("close"), 10)) / 10.0
    seg_b = (delay(pl.col("close"), 10) - pl.col("close")) / 10.0
    curvature = seg_a - seg_b
    expr = if_then_else(
        curvature < -0.05,
        pl.lit(1.0),
        -1.0 * (pl.col("close") - delay(pl.col("close"), 1)),
    )
    return panel.select(expr.alias("alpha051").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha052 — Low-shift × medium-term excess return rank × volume rank
# ---------------------------------------------------------------------------


def alpha052(panel: pl.DataFrame) -> pl.Series:
    """Alpha #052 — Low-shift × medium-term excess return rank × volume rank.

    WorldQuant Formula (Kakushadze 2015, eq. 52)
    --------------------------------------------
        ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
          rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))

    Legacy AQML Expression
    ----------------------
        (-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) *
         Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) *
         Ts_Rank(volume, 5)

    Polars Implementation Notes
    ---------------------------
    1. ``-Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)`` measures the change
       in the rolling-min low over the last 5 days vs 5 days ago.
    2. The medium-term excess return uses 240 - 20 = 220 day window of
       carry, scaled by 1/220.
    3. Materialise the rank-input, then ``cs_rank``, then multiply with
       the TS components.

    Required panel columns: ``low``, ``returns``, ``volume``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    low_min5 = ts_min(pl.col("low"), 5)
    low_term = -1.0 * low_min5 + delay(low_min5, 5)
    excess = (ts_sum(pl.col("returns"), 240) - ts_sum(pl.col("returns"), 20)) / 220.0
    vol_tsr = ts_rank(pl.col("volume"), 5)
    staged = panel.with_columns(
        low_term.alias("__a052_low"),
        excess.alias("__a052_ex"),
        vol_tsr.alias("__a052_vt"),
    )
    expr = pl.col("__a052_low") * cs_rank(pl.col("__a052_ex")) * pl.col("__a052_vt")
    return staged.select(expr.alias("alpha052").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha084 — VWAP-vs-15d-max rank, sign-preserving
# ---------------------------------------------------------------------------


def alpha084(panel: pl.DataFrame) -> pl.Series:
    """Alpha #084 — VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised).

    WorldQuant Formula (Kakushadze 2015, eq. 84)
    --------------------------------------------
        SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
                    delta(close, 4.96796))

    Legacy AQML Expression (linearised exponent)
    --------------------------------------------
        SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)

    Polars Implementation Notes
    ---------------------------
    1. The original WorldQuant formula uses a fractional-day rolling max
       and an exponent equal to a per-row delta; our migrated AQML form
       linearises the exponent to ``1.0`` and rounds the windows to
       integers (15 and 21). We follow the migrated form verbatim — this
       matches both the legacy AQML evaluator and the STHSF reference.
    2. With exponent=1 the operation degenerates to identity (sign · |x|),
       so the alpha simplifies to just the rolling rank itself.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap_minus_max = pl.col("vwap") - ts_max(pl.col("vwap"), 15)
    rk = ts_rank(vwap_minus_max, 21)
    expr = signed_power(rk, 1.0)
    return panel.select(expr.alias("alpha084").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# alpha_custom_decaylinear_mom — Decay-linear-weighted 10d momentum rank
# ---------------------------------------------------------------------------


def alpha_custom_decaylinear_mom(panel: pl.DataFrame) -> pl.Series:
    """Alpha custom — Decay-linear-weighted 10d momentum rank.

    Project-internal custom factor (NOT in WorldQuant 101 paper).

    Legacy AQML Expression
    ----------------------
        Rank(Ts_DecayLinear(returns, 10))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_decay_linear(returns, 10)`` -> per-stock weighted MA of returns
       with linearly decaying weights ``[10, 9, ..., 1] / 55``.
    2. Cross-section rank (``cs_rank``) the materialised column. Two
       partitions -> stage first.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    decay = ts_decay_linear(pl.col("returns"), 10)
    staged = panel.with_columns(decay.alias("__a_custom_dl_mom"))
    return staged.select(
        cs_rank(pl.col("__a_custom_dl_mom")).alias("alpha_custom_decaylinear_mom").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# alpha_custom_argmax_recent — Inverse rank of days-since-20d-max
# ---------------------------------------------------------------------------


def alpha_custom_argmax_recent(panel: pl.DataFrame) -> pl.Series:
    """Alpha custom — Inverse rank of days-since-20d-max (recency-of-peak).

    Project-internal custom factor (NOT in WorldQuant 101 paper).

    Legacy AQML Expression
    ----------------------
        1 - Rank(Ts_ArgMax(close, 20))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_argmax(close, 20)`` -> position 0..19 of the highest close in
       last 20 days; high values ⇒ peak was recent (today the peak).
    2. ``1 - cs_rank(...)`` flips the rank so that recent peaks score high.
       Materialise the argmax column before ranking.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    arg = ts_argmax(pl.col("close"), 20)
    staged = panel.with_columns(arg.alias("__a_custom_arg_recent"))
    return staged.select(
        (1.0 - cs_rank(pl.col("__a_custom_arg_recent")))
        .alias("alpha_custom_argmax_recent")
        .cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# Registry self-population — runs at module import time
# ---------------------------------------------------------------------------


_ENTRIES = [
    FactorEntry(
        id="alpha007",
        impl=alpha007,
        direction="reverse",
        category="momentum",
        description="Volume-conditional 7-day signed momentum rank",
        legacy_aqml_expr=(
            "If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * "
            "Sign(Delta(close, 7)), -1)"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 7",),
        formula_doc_path="docs/factor_library/alpha101/alpha_007.md",
    ),
    FactorEntry(
        id="alpha008",
        impl=alpha008,
        direction="reverse",
        category="momentum",
        description="Acceleration of (open*returns) sum compared to 10 days ago",
        legacy_aqml_expr=(
            "-1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - "
            "Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 8",),
        formula_doc_path="docs/factor_library/alpha101/alpha_008.md",
    ),
    FactorEntry(
        id="alpha009",
        impl=alpha009,
        direction="reverse",
        category="momentum",
        description="Trend-confirmed price-change momentum",
        legacy_aqml_expr=(
            "If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1), "
            "If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1), -1 * Delta(close, 1)))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 9",),
        formula_doc_path="docs/factor_library/alpha101/alpha_009.md",
    ),
    FactorEntry(
        id="alpha010",
        impl=alpha010,
        direction="reverse",
        category="momentum",
        description="Cross-sectional rank of trend-confirmed price change",
        legacy_aqml_expr=(
            "Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1), "
            "If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1), -1 * Delta(close, 1))))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 10",),
        formula_doc_path="docs/factor_library/alpha101/alpha_010.md",
    ),
    FactorEntry(
        id="alpha017",
        impl=alpha017,
        direction="reverse",
        category="momentum",
        description="Momentum exhaustion: rank-mom × second-derivative × volume-surge",
        legacy_aqml_expr=(
            "(-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) * "
            "Rank(Ts_Rank(volume / adv20, 5))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 17",),
        formula_doc_path="docs/factor_library/alpha101/alpha_017.md",
    ),
    FactorEntry(
        id="alpha019",
        impl=alpha019,
        direction="reverse",
        category="momentum",
        description="7d-return sign times annual rank-mom multiplier",
        legacy_aqml_expr=(
            "(-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) * "
            "(1 + Rank(1 + Ts_Sum(returns, 250)))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 19",),
        formula_doc_path="docs/factor_library/alpha101/alpha_019.md",
    ),
    FactorEntry(
        id="alpha038",
        impl=alpha038,
        direction="reverse",
        category="momentum",
        description="Rank of 10d close rolling rank times close-over-open rank, negated",
        legacy_aqml_expr="-1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 38",),
        formula_doc_path="docs/factor_library/alpha101/alpha_038.md",
    ),
    FactorEntry(
        id="alpha045",
        impl=alpha045,
        direction="reverse",
        category="momentum",
        description="Lagged-MA rank × short-corr × long/short-MA-corr rank, negated",
        legacy_aqml_expr=(
            "-1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) * "
            "Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 45",),
        formula_doc_path="docs/factor_library/alpha101/alpha_045.md",
    ),
    FactorEntry(
        id="alpha046",
        impl=alpha046,
        direction="reverse",
        category="momentum",
        description="Trend-shape conditional one-day reversal",
        legacy_aqml_expr=(
            "If((Delay(close, 20) - Delay(close, 10)) / 10 - "
            "(Delay(close, 10) - close) / 10 > 0.25, -1, "
            "If((Delay(close, 20) - Delay(close, 10)) / 10 - "
            "(Delay(close, 10) - close) / 10 < 0, 1, -1 * (close - Delay(close, 1))))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 46",),
        formula_doc_path="docs/factor_library/alpha101/alpha_046.md",
    ),
    FactorEntry(
        id="alpha051",
        impl=alpha051,
        direction="reverse",
        category="momentum",
        description="Trend curvature conditional reversal",
        legacy_aqml_expr=(
            "If((Delay(close, 20) - Delay(close, 10)) / 10 - "
            "(Delay(close, 10) - close) / 10 < -0.05, 1, "
            "-1 * (close - Delay(close, 1)))"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 51",),
        formula_doc_path="docs/factor_library/alpha101/alpha_051.md",
    ),
    FactorEntry(
        id="alpha052",
        impl=alpha052,
        direction="reverse",
        category="momentum",
        description="Low-shift × medium-term excess return rank × volume rank",
        legacy_aqml_expr=(
            "(-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) * "
            "Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) * "
            "Ts_Rank(volume, 5)"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 52",),
        formula_doc_path="docs/factor_library/alpha101/alpha_052.md",
    ),
    FactorEntry(
        id="alpha084",
        impl=alpha084,
        direction="reverse",
        category="momentum",
        description="VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised)",
        legacy_aqml_expr="SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)",
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 84",),
        formula_doc_path="docs/factor_library/alpha101/alpha_084.md",
    ),
    FactorEntry(
        id="alpha_custom_decaylinear_mom",
        impl=alpha_custom_decaylinear_mom,
        direction="reverse",
        category="momentum",
        description="Decay-linear-weighted 10d momentum rank",
        legacy_aqml_expr="Rank(Ts_DecayLinear(returns, 10))",
        references=("AurumQ project-internal custom factor",),
        formula_doc_path="docs/factor_library/alpha101/alpha_custom_decaylinear_mom.md",
    ),
    FactorEntry(
        id="alpha_custom_argmax_recent",
        impl=alpha_custom_argmax_recent,
        direction="reverse",
        category="momentum",
        description="Inverse rank of days-since-20d-max (recency-of-peak)",
        legacy_aqml_expr="1 - Rank(Ts_ArgMax(close, 20))",
        references=("AurumQ project-internal custom factor",),
        formula_doc_path="docs/factor_library/alpha101/alpha_custom_argmax_recent.md",
    ),
]

for _e in _ENTRIES:
    register_alpha101(_e)
