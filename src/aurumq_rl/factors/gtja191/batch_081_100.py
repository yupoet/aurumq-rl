"""GTJA-191 factor library — batch 081 through 100.

Translated from Daic115/alpha191 (formula reference, no code vendored).
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_gtja191

from ._ops import (
    abs_,
    corr,
    covariance,
    decay_linear,
    delay,
    delta,
    mean,
    rank,
    sma,
    std_,
    sum_,
    ts_max,
    ts_min,
    ts_rank,
)

# ---------------------------------------------------------------------------
# gtja_081 — SMA(VOLUME, 21, 2)
# ---------------------------------------------------------------------------


def gtja_081(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #081 — EWMA(21, 2) of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    return panel.select(sma(pl.col("volume"), 21, 2).alias("gtja_081").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_082 — Smoothed inverse-RSV (20, 1)
# ---------------------------------------------------------------------------


def gtja_082(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #082 — SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))*100, 20, 1).

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    hmax = ts_max(pl.col("high"), 6)
    lmin = ts_min(pl.col("low"), 6)
    raw = (hmax - pl.col("close")) / (hmax - lmin) * 100.0
    return panel.select(sma(raw, 20, 1).alias("gtja_082").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_083 — -RANK(COV(RANK(H), RANK(V), 5))
# ---------------------------------------------------------------------------


def gtja_083(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #083 — Negated rank of 5d covariance of rank(H) vs rank(V).

    Guotai Junan Formula
    --------------------
        -1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5))

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("high")).alias("__g083_rh"),
        rank(pl.col("volume")).alias("__g083_rv"),
    )
    staged = staged.with_columns(
        covariance(pl.col("__g083_rh"), pl.col("__g083_rv"), 5).alias("__g083_c")
    )
    return staged.select(
        (-1.0 * rank(pl.col("__g083_c"))).alias("gtja_083").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_084 — 20d signed-volume sum
# ---------------------------------------------------------------------------


def gtja_084(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #084 — 20d signed-volume sum (close-direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    c = pl.col("close")
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    signed_vol = (
        pl.when(cond_up)
        .then(pl.col("volume"))
        .otherwise(pl.when(cond_dn).then(-pl.col("volume")).otherwise(0.0))
    )
    return panel.select(sum_(signed_vol, 20).alias("gtja_084").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_085 — TSRANK(V/MA(V,20), 20) × TSRANK(-DELTA(C, 7), 8)
# ---------------------------------------------------------------------------


def gtja_085(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #085 — Volume-ratio TS-rank × negated 7d-close-delta TS-rank.

    Guotai Junan Formula
    --------------------
        TSRANK(V / MEAN(V, 20), 20) * TSRANK(-1 * DELTA(C, 7), 8)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vol_ratio = pl.col("volume") / mean(pl.col("volume"), 20)
    arm1 = ts_rank(vol_ratio, 20)
    arm2 = ts_rank(-1.0 * delta(pl.col("close"), 7), 8)
    return panel.select((arm1 * arm2).alias("gtja_085").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_086 — Acceleration regime ternary
# ---------------------------------------------------------------------------


def gtja_086(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #086 — 20/10/0 close-acceleration regime ternary.

    Guotai Junan Formula
    --------------------
        part1 = (DELAY(C, 20) - DELAY(C, 10)) / 10
        part2 = (DELAY(C, 10) - C) / 10
        if (0.25 < (part1 - part2)) -1
        elif ((part1 - part2) < 0) 1
        else -1 * (C - DELAY(C, 1))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col("close")
    part1 = (delay(c, 20) - delay(c, 10)) / 10.0
    part2 = (delay(c, 10) - c) / 10.0
    diff = part1 - part2
    base = -1.0 * (c - delay(c, 1))
    expr = pl.when(diff > 0.25).then(-1.0).otherwise(pl.when(diff < 0.0).then(1.0).otherwise(base))
    return panel.select(expr.alias("gtja_086").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_087 — Rank-decay-vwap-delta + TS-rank-decay-asymmetric-spread
# ---------------------------------------------------------------------------


def gtja_087(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #087 — Rank-decay(vwap delta) + TS-rank-decay(asymmetric spread), negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
         TSRANK(DECAYLINEAR(((L*0.9 + L*0.1) - VWAP) / (O - (H+L)/2), 11), 7)) * -1

    Required panel columns: ``vwap``, ``low``, ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col("vwap"), 4), 7)
    spread_num = (pl.col("low") * 0.9 + pl.col("low") * 0.1) - pl.col("vwap")
    spread_den = pl.col("open") - (pl.col("high") + pl.col("low")) / 2.0 + 1e-7
    arm2_inner = decay_linear(spread_num / spread_den, 11)
    arm2 = ts_rank(arm2_inner, 7)
    staged = panel.with_columns(arm1_inner.alias("__g087_a1_inner"))
    staged = staged.with_columns(rank(pl.col("__g087_a1_inner")).alias("__g087_r1"))
    return staged.select(
        ((pl.col("__g087_r1") + arm2) * -1.0).alias("gtja_087").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_088 — 20d % change × 100
# ---------------------------------------------------------------------------


def gtja_088(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #088 — 20-day % change × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    return panel.select(
        ((c - delay(c, 20)) / delay(c, 20) * 100.0).alias("gtja_088").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_089 — MACD-style oscillator
# ---------------------------------------------------------------------------


def gtja_089(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #089 — MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27)).

    Guotai Junan Formula
    --------------------
        2 * (SMA(C, 13, 2) - SMA(C, 27, 2) -
             SMA(SMA(C, 13, 2) - SMA(C, 27, 2), 10, 2))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col("close")
    ma_short = sma(c, 13, 2)
    ma_long = sma(c, 27, 2)
    diff = ma_short - ma_long
    return panel.select(
        (2.0 * (ma_short - ma_long - sma(diff, 10, 2))).alias("gtja_089").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_090 — -RANK(CORR(RANK(VWAP), RANK(V), 5))
# ---------------------------------------------------------------------------


def gtja_090(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #090 — Negated rank of 5d corr(rank-VWAP, rank-V).

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("vwap")).alias("__g090_rw"),
        rank(pl.col("volume")).alias("__g090_rv"),
    )
    staged = staged.with_columns(
        corr(pl.col("__g090_rw"), pl.col("__g090_rv"), 5).alias("__g090_c")
    )
    return staged.select(
        (-1.0 * rank(pl.col("__g090_c"))).alias("gtja_090").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_091 — Two rank arms multiplied negated
# ---------------------------------------------------------------------------


def gtja_091(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #091 — Negated product of two rank arms.

    Guotai Junan Formula
    --------------------
        -1 * RANK(C - TSMAX(C, 5)) * RANK(CORR(MEAN(V, 40), L, 5))

    Required panel columns: ``close``, ``volume``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c = pl.col("close")
    arm1_inner = c - ts_max(c, 5)
    cor = corr(mean(pl.col("volume"), 40), pl.col("low"), 5)
    staged = panel.with_columns(
        arm1_inner.alias("__g091_a1"),
        cor.alias("__g091_c"),
    )
    staged = staged.with_columns(
        rank(pl.col("__g091_a1")).alias("__g091_r1"),
        rank(pl.col("__g091_c")).alias("__g091_r2"),
    )
    return staged.select(
        (pl.col("__g091_r1") * pl.col("__g091_r2") * -1.0).alias("gtja_091").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_092 — MAX of rank-decay-delta and TS-rank-decay-abs-corr
# ---------------------------------------------------------------------------


def gtja_092(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #092 — MAX of rank-decay-delta and TS-rank-decay-abs-corr, negated.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(DELTA(C*0.35 + VWAP*0.65, 2), 3)),
          TSRANK(DECAYLINEAR(|CORR(MEAN(V, 180), C, 13)|, 5), 15)
        ) * -1

    Required panel columns: ``close``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("close") * 0.35 + pl.col("vwap") * 0.65
    arm1_inner = decay_linear(delta(weighted, 2), 3)
    cor = corr(mean(pl.col("volume"), 180), pl.col("close"), 13)
    arm2_inner = decay_linear(abs_(cor), 5)
    arm2 = ts_rank(arm2_inner, 15)
    staged = panel.with_columns(arm1_inner.alias("__g092_a1"))
    staged = staged.with_columns(rank(pl.col("__g092_a1")).alias("__g092_r1"))
    return staged.select(
        (pl.max_horizontal(pl.col("__g092_r1"), arm2) * -1.0).alias("gtja_092").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_093 — DBM-style 20d sum
# ---------------------------------------------------------------------------


def gtja_093(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #093 — 20d sum of conditional max(O-L, O-DELAY(O,1)) when O<DELAY(O,1).

    Guotai Junan Formula
    --------------------
        SUM((O >= DELAY(O, 1) ? 0 : MAX(O - L, O - DELAY(O, 1))), 20)

    Required panel columns: ``open``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    o = pl.col("open")
    o_lag = delay(o, 1)
    inner = pl.max_horizontal(o - pl.col("low"), o - o_lag)
    expr = pl.when(o >= o_lag).then(0.0).otherwise(inner)
    return panel.select(sum_(expr, 20).alias("gtja_093").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_094 — 30d signed-volume sum
# ---------------------------------------------------------------------------


def gtja_094(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #094 — 30d signed-volume sum.

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C, 1) ? V : (C < DELAY(C, 1) ? -V : 0)), 30)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    c = pl.col("close")
    cond_up = c > delay(c, 1)
    cond_dn = c < delay(c, 1)
    signed_vol = (
        pl.when(cond_up)
        .then(pl.col("volume"))
        .otherwise(pl.when(cond_dn).then(-pl.col("volume")).otherwise(0.0))
    )
    return panel.select(sum_(signed_vol, 30).alias("gtja_094").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_095 — STD(amount, 20)
# ---------------------------------------------------------------------------


def gtja_095(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #095 — 20-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col("amount"), 20).alias("gtja_095").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_096 — Double-smoothed stochastic %K
# ---------------------------------------------------------------------------


def gtja_096(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #096 — SMA(SMA(stochastic-%K, 3, 1), 3, 1).

    Guotai Junan Formula
    --------------------
        SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    raw = (
        (pl.col("close") - ts_min(pl.col("low"), 9))
        / (ts_max(pl.col("high"), 9) - ts_min(pl.col("low"), 9))
        * 100.0
    )
    return panel.select(sma(sma(raw, 3, 1), 3, 1).alias("gtja_096").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_097 — STD(volume, 10)
# ---------------------------------------------------------------------------


def gtja_097(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #097 — 10-day std of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col("volume"), 10).alias("gtja_097").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_098 — Long-term trend regime ternary
# ---------------------------------------------------------------------------


def gtja_098(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #098 — Long-trend ternary: 100d MA acceleration regime.

    Guotai Junan Formula
    --------------------
        cond = DELTA(SUM(C, 100)/100, 100) / DELAY(C, 100)
        cond <= 0.05 ? -1*(C - TSMIN(C, 100)) : -1 * DELTA(C, 3)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col("close")
    cond_val = delta(sum_(c, 100) / 100.0, 100) / delay(c, 100)
    branch_low = -1.0 * (c - ts_min(c, 100))
    branch_hi = -1.0 * delta(c, 3)
    expr = pl.when(cond_val <= 0.05).then(branch_low).otherwise(branch_hi)
    return panel.select(expr.alias("gtja_098").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# gtja_099 — -RANK(COV(RANK(C), RANK(V), 5))
# ---------------------------------------------------------------------------


def gtja_099(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #099 — Negated rank of 5d covariance of rank(C) vs rank(V).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        rank(pl.col("close")).alias("__g099_rc"),
        rank(pl.col("volume")).alias("__g099_rv"),
    )
    staged = staged.with_columns(
        covariance(pl.col("__g099_rc"), pl.col("__g099_rv"), 5).alias("__g099_co")
    )
    return staged.select(
        (-1.0 * rank(pl.col("__g099_co"))).alias("gtja_099").cast(pl.Float64)
    ).to_series()


# ---------------------------------------------------------------------------
# gtja_100 — STD(volume, 20)
# ---------------------------------------------------------------------------


def gtja_100(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #100 — 20-day std of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col("volume"), 20).alias("gtja_100").cast(pl.Float64)).to_series()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_DOC_BASE = "docs/factor_library/gtja191"
_REF_BASE = "Guotai Junan 2017, '191 Alphas', via Daic115/alpha191 (formula only)"

_ENTRIES: list[FactorEntry] = [
    FactorEntry(
        id="gtja_081",
        impl=gtja_081,
        direction="normal",
        category="volume_price",
        description="EWMA(21, 2) of volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_081.md",
    ),
    FactorEntry(
        id="gtja_082",
        impl=gtja_082,
        direction="reverse",
        category="mean_reversion",
        description="SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 20, 1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_082.md",
    ),
    FactorEntry(
        id="gtja_083",
        impl=gtja_083,
        direction="reverse",
        category="volume_price",
        description="-RANK(COV(RANK(H), RANK(V), 5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_083.md",
    ),
    FactorEntry(
        id="gtja_084",
        impl=gtja_084,
        direction="normal",
        category="volume_price",
        description="20d signed-volume sum (close-direction)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_084.md",
    ),
    FactorEntry(
        id="gtja_085",
        impl=gtja_085,
        direction="reverse",
        category="momentum",
        description="TSRANK(V/MA20V, 20) × TSRANK(-DELTA(C, 7), 8)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_085.md",
    ),
    FactorEntry(
        id="gtja_086",
        impl=gtja_086,
        direction="reverse",
        category="momentum",
        description="20/10/0 close-acceleration regime ternary",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_086.md",
    ),
    FactorEntry(
        id="gtja_087",
        impl=gtja_087,
        direction="reverse",
        category="volume_price",
        description="-(rank-decay-vwap-delta + TS-rank-decay-asymmetric-spread)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_087.md",
    ),
    FactorEntry(
        id="gtja_088",
        impl=gtja_088,
        direction="normal",
        category="momentum",
        description="20-day % change × 100",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_088.md",
    ),
    FactorEntry(
        id="gtja_089",
        impl=gtja_089,
        direction="normal",
        category="momentum",
        description="MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_089.md",
    ),
    FactorEntry(
        id="gtja_090",
        impl=gtja_090,
        direction="reverse",
        category="volume_price",
        description="-RANK(CORR(RANK(VWAP), RANK(V), 5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_090.md",
    ),
    FactorEntry(
        id="gtja_091",
        impl=gtja_091,
        direction="reverse",
        category="volume_price",
        description="-RANK(C - TSMAX(C, 5)) × RANK(CORR(MEAN(V, 40), L, 5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_091.md",
    ),
    FactorEntry(
        id="gtja_092",
        impl=gtja_092,
        direction="reverse",
        category="volume_price",
        description="-MAX of rank-decay-delta + TS-rank-decay-|corr| arms",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_092.md",
    ),
    FactorEntry(
        id="gtja_093",
        impl=gtja_093,
        direction="normal",
        category="volatility",
        description="20d sum of conditional max(O-L, O-O-1) when O<O-1",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_093.md",
    ),
    FactorEntry(
        id="gtja_094",
        impl=gtja_094,
        direction="normal",
        category="volume_price",
        description="30d signed-volume sum",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_094.md",
    ),
    FactorEntry(
        id="gtja_095",
        impl=gtja_095,
        direction="normal",
        category="volatility",
        description="20-day std of amount",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_095.md",
    ),
    FactorEntry(
        id="gtja_096",
        impl=gtja_096,
        direction="normal",
        category="momentum",
        description="Double-smoothed stochastic %K (3,1)(3,1)",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_096.md",
    ),
    FactorEntry(
        id="gtja_097",
        impl=gtja_097,
        direction="normal",
        category="volatility",
        description="10-day std of volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_097.md",
    ),
    FactorEntry(
        id="gtja_098",
        impl=gtja_098,
        direction="reverse",
        category="momentum",
        description="100d MA-acceleration regime ternary",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_098.md",
    ),
    FactorEntry(
        id="gtja_099",
        impl=gtja_099,
        direction="reverse",
        category="volume_price",
        description="-RANK(COV(RANK(C), RANK(V), 5))",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_099.md",
    ),
    FactorEntry(
        id="gtja_100",
        impl=gtja_100,
        direction="normal",
        category="volatility",
        description="20-day std of volume",
        references=(_REF_BASE,),
        formula_doc_path=f"{_DOC_BASE}/gtja_100.md",
    ),
]

for _e in _ENTRIES:
    register_gtja191(_e)
