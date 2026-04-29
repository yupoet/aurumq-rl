"""Alpha101 — volume_price category factors (31 factors).

Translated from AQML expressions in
``aurumq.rules.alpha101_library.ALPHA101_FACTORS`` into polars-native
implementations. Each factor self-registers into ``ALPHA101_REGISTRY``
at module import time.

Implementation notes
--------------------
Most volume-price alphas mix time-series (per-stock) and cross-sectional
(per-date) operators. Polars cannot reliably mix two ``over`` partitions
(``stock_code`` for TS, ``trade_date`` for CS) inside a single expression
when one is computed from another, so we materialise intermediate
columns into the panel before applying the next layer. The pattern
mirrors :mod:`aurumq_rl.factors.alpha101.volatility`.

A few helper operators required by this module are defined locally as
(legacy comment removed: _local_ ops have all been folded into _ops.py)
sub-agent. Once that lands, the helpers will be deleted and imported
directly.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    TS_PART,
    cs_rank,
    cs_scale,
    delay,
    delta,
    if_then_else,
    log_,
    pmax,
    pmin,
    power,
    sign_,
    ts_argmax,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_std,
    ts_sum,
)

# ---------------------------------------------------------------------------
# Alpha factor implementations
# ---------------------------------------------------------------------------


def alpha002(panel: pl.DataFrame) -> pl.Series:
    """Alpha #002 — Volume change rank vs intraday return rank correlation.

    WorldQuant Formula
    ------------------
        -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)

    Polars Implementation Notes
    ---------------------------
    1. Inner ``Rank`` partitions by ``trade_date`` (CS); outer ``Ts_Corr``
       partitions by ``stock_code`` (TS). Materialise the two per-row
       ranked Series before correlating.

    Required panel columns: ``volume``, ``close``, ``open``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    # Two-stage: first materialise the TS-side intermediate (delta of log
    # volume) per stock, then CS-rank that staged column on a clean panel.
    # Polars cannot compose a CS ``.over(trade_date)`` on top of a TS
    # ``.over(stock_code)`` inside a single expression — every level of
    # ``over`` after the first silently nulls out the result.
    staged_ts = panel.with_columns(
        delta(log_(pl.col("volume")), 2).alias("__a002_dlogv"),
    )
    staged = staged_ts.with_columns(
        cs_rank(pl.col("__a002_dlogv")).alias("__a002_rv"),
        cs_rank((pl.col("close") - pl.col("open")) / pl.col("open")).alias("__a002_rret"),
    )
    return staged.select(
        (-1.0 * ts_corr(pl.col("__a002_rv"), pl.col("__a002_rret"), 6)).alias("alpha002")
    ).to_series()


def alpha003(panel: pl.DataFrame) -> pl.Series:
    """Alpha #003 — Open price rank vs volume rank correlation.

    WorldQuant Formula
    ------------------
        -1 * correlation(rank(open), rank(volume), 10)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(Rank(open), Rank(volume), 10)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks materialised before TS corr.

    Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("open")).alias("__a003_ro"),
        cs_rank(pl.col("volume")).alias("__a003_rv"),
    )
    return staged.select(
        (-1.0 * ts_corr(pl.col("__a003_ro"), pl.col("__a003_rv"), 10)).alias("alpha003")
    ).to_series()


def alpha005(panel: pl.DataFrame) -> pl.Series:
    """Alpha #005 — Open vs 10d VWAP-mean rank, scaled by close-VWAP rank deviation.

    WorldQuant Formula
    ------------------
        rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))

    Legacy AQML Expression
    ----------------------
        Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))

    Polars Implementation Notes
    ---------------------------
    Both inner expressions can be evaluated as expressions, then ranked
    cross-sectionally. We stage the TS-sum and the inner deviations to
    keep the CS rank step pure-CS.

    Required panel columns: ``open``, ``vwap``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner1 = pl.col("open") - ts_sum(pl.col("vwap"), 10) / 10.0
    inner2 = pl.col("close") - pl.col("vwap")
    staged = panel.with_columns(inner1.alias("__a005_inner1"), inner2.alias("__a005_inner2"))
    return staged.select(
        (cs_rank(pl.col("__a005_inner1")) * (-1.0 * cs_rank(pl.col("__a005_inner2")).abs())).alias(
            "alpha005"
        )
    ).to_series()


def alpha006(panel: pl.DataFrame) -> pl.Series:
    """Alpha #006 — 10-day correlation between open and volume.

    WorldQuant Formula
    ------------------
        -1 * correlation(open, volume, 10)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(open, volume, 10)

    Polars Implementation Notes
    ---------------------------
    Pure TS — no CS staging needed.

    Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select(
        (-1.0 * ts_corr(pl.col("open"), pl.col("volume"), 10)).alias("alpha006")
    ).to_series()


def alpha012(panel: pl.DataFrame) -> pl.Series:
    """Alpha #012 — Volume direction times negative price change.

    WorldQuant Formula
    ------------------
        sign(delta(volume, 1)) * (-1 * delta(close, 1))

    Legacy AQML Expression
    ----------------------
        Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))

    Polars Implementation Notes
    ---------------------------
    Pure TS — sign of one-day volume change times negative one-day
    close change. Sign returns 0 on zero change.

    Required panel columns: ``volume``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select(
        (sign_(delta(pl.col("volume"), 1)) * (-1.0 * delta(pl.col("close"), 1))).alias("alpha012")
    ).to_series()


def alpha013(panel: pl.DataFrame) -> pl.Series:
    """Alpha #013 — Negative rank of close-rank vs volume-rank covariance.

    WorldQuant Formula
    ------------------
        -1 * rank(covariance(rank(close), rank(volume), 5))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))

    Polars Implementation Notes
    ---------------------------
    Two CS ranks materialised, then TS covariance, then outer CS rank.
    Two staging passes (CS → TS) followed by a final select.

    Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged1 = panel.with_columns(
        cs_rank(pl.col("close")).alias("__a013_rc"),
        cs_rank(pl.col("volume")).alias("__a013_rv"),
    )
    staged2 = staged1.with_columns(
        ts_cov(pl.col("__a013_rc"), pl.col("__a013_rv"), 5).alias("__a013_cov")
    )
    return staged2.select((-1.0 * cs_rank(pl.col("__a013_cov"))).alias("alpha013")).to_series()


def alpha014(panel: pl.DataFrame) -> pl.Series:
    """Alpha #014 — Returns-acceleration rank scaled by open-volume correlation.

    WorldQuant Formula
    ------------------
        (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

    Legacy AQML Expression
    ----------------------
        (-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)

    Polars Implementation Notes
    ---------------------------
    Inner Delta is TS, then CS rank, then outer multiplied by TS corr.
    We stage the delta to make the CS rank pure.

    Required panel columns: ``returns``, ``open``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        delta(pl.col("returns"), 3).alias("__a014_dret"),
    )
    return staged.select(
        (
            (-1.0 * cs_rank(pl.col("__a014_dret"))) * ts_corr(pl.col("open"), pl.col("volume"), 10)
        ).alias("alpha014")
    ).to_series()


def alpha015(panel: pl.DataFrame) -> pl.Series:
    """Alpha #015 — Negative 3d sum of ranked high-volume rank correlation.

    WorldQuant Formula
    ------------------
        -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks are materialised before the 3-day TS correlation. STHSF
    fills NaN/inf correlation rows with zero before the outer rank; we do
    the same so the early-window rank/sum semantics match the reference.

    Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged1 = panel.with_columns(
        cs_rank(pl.col("high")).alias("__a015_rh"),
        cs_rank(pl.col("volume")).alias("__a015_rv"),
    )
    staged2 = staged1.with_columns(
        ts_corr(pl.col("__a015_rh"), pl.col("__a015_rv"), 3)
        .fill_nan(0.0)
        .fill_null(0.0)
        .alias("__a015_corr")
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a015_corr")).alias("__a015_rcorr"),
    )
    return staged3.select(
        (-1.0 * ts_sum(pl.col("__a015_rcorr"), 3)).alias("alpha015").cast(pl.Float64)
    ).to_series()


def alpha022(panel: pl.DataFrame) -> pl.Series:
    """Alpha #022 — Change in high-volume correlation scaled by 20d stdev rank.

    WorldQuant Formula
    ------------------
        -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

    Legacy AQML Expression
    ----------------------
        -1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))

    Polars Implementation Notes
    ---------------------------
    Inner TS corr → TS delta → multiplied by CS rank of TS std. Stage the
    20-day std before the CS rank.

    Required panel columns: ``high``, ``volume``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    corr5 = ts_corr(pl.col("high"), pl.col("volume"), 5)
    delta_corr = (corr5 - corr5.shift(5)).over(TS_PART)
    staged = panel.with_columns(
        delta_corr.alias("__a022_dcorr"),
        ts_std(pl.col("close"), 20).alias("__a022_std"),
    )
    return staged.select(
        (-1.0 * pl.col("__a022_dcorr") * cs_rank(pl.col("__a022_std"))).alias("alpha022")
    ).to_series()


def alpha025(panel: pl.DataFrame) -> pl.Series:
    """Alpha #025 — Rank of negative-returns × volume-weighted price-tail.

    WorldQuant Formula
    ------------------
        rank(((-1 * returns) * adv20) * vwap * (high - close))

    Legacy AQML Expression
    ----------------------
        Rank((-1 * returns) * adv20 * vwap * (high - close))

    Polars Implementation Notes
    ---------------------------
    Build the multiplicative payload first, then CS rank.

    Required panel columns: ``returns``, ``adv20``, ``vwap``, ``high``, ``close``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner = (
        (-1.0 * pl.col("returns"))
        * pl.col("adv20")
        * pl.col("vwap")
        * (pl.col("high") - pl.col("close"))
    )
    staged = panel.with_columns(inner.alias("__a025_inner"))
    return staged.select(cs_rank(pl.col("__a025_inner")).alias("alpha025")).to_series()


def alpha026(panel: pl.DataFrame) -> pl.Series:
    """Alpha #026 — Max recent volume-rank vs high-rank correlation.

    WorldQuant Formula
    ------------------
        -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)

    Polars Implementation Notes
    ---------------------------
    Pure TS chain. Stage the two TS ranks before the corr.

    Required panel columns: ``volume``, ``high``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        ts_rank(pl.col("volume"), 5).alias("__a026_rv"),
        ts_rank(pl.col("high"), 5).alias("__a026_rh"),
    )
    corr = ts_corr(pl.col("__a026_rv"), pl.col("__a026_rh"), 5)
    return staged.select((-1.0 * ts_max(corr, 3)).alias("alpha026")).to_series()


def alpha028(panel: pl.DataFrame) -> pl.Series:
    """Alpha #028 — Standardised mid-price vs close gap with volume modifier.

    WorldQuant Formula
    ------------------
        scale(correlation(adv20, low, 5) + (high + low) / 2 - close)

    Legacy AQML Expression
    ----------------------
        Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)

    Polars Implementation Notes
    ---------------------------
    TS corr then arithmetic, then CS scale (rescale to ``sum(|x|) == 1`` per day).

    Required panel columns: ``adv20``, ``low``, ``high``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner = (
        ts_corr(pl.col("adv20"), pl.col("low"), 5)
        + (pl.col("high") + pl.col("low")) / 2.0
        - pl.col("close")
    )
    staged = panel.with_columns(inner.alias("__a028_inner"))
    return staged.select(cs_scale(pl.col("__a028_inner")).alias("alpha028")).to_series()


def alpha035(panel: pl.DataFrame) -> pl.Series:
    """Alpha #035 — Volume rank × inverse range rank × inverse returns rank.

    WorldQuant Formula
    ------------------
        ts_rank(volume, 32)
        * (1 - ts_rank(close + high - low, 16))
        * (1 - ts_rank(returns, 32))

    Legacy AQML Expression
    ----------------------
        Ts_Rank(volume, 32)
        * (1 - Ts_Rank(close + high - low, 16))
        * (1 - Ts_Rank(returns, 32))

    Polars Implementation Notes
    ---------------------------
    Pure TS chain. Three rolling ranks combined multiplicatively.

    Required panel columns: ``volume``, ``close``, ``high``, ``low``, ``returns``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select(
        (
            ts_rank(pl.col("volume"), 32)
            * (1.0 - ts_rank(pl.col("close") + pl.col("high") - pl.col("low"), 16))
            * (1.0 - ts_rank(pl.col("returns"), 32))
        ).alias("alpha035")
    ).to_series()


def alpha043(panel: pl.DataFrame) -> pl.Series:
    """Alpha #043 — Volume-surge rank × 7d-decline rank.

    WorldQuant Formula
    ------------------
        ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

    Legacy AQML Expression
    ----------------------
        Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)

    Polars Implementation Notes
    ---------------------------
    Pure TS chain.

    Required panel columns: ``volume``, ``adv20``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select(
        (
            ts_rank(pl.col("volume") / pl.col("adv20"), 20)
            * ts_rank(-1.0 * delta(pl.col("close"), 7), 8)
        ).alias("alpha043")
    ).to_series()


def alpha044(panel: pl.DataFrame) -> pl.Series:
    """Alpha #044 — Negative high-vs-volume-rank correlation.

    WorldQuant Formula
    ------------------
        -1 * correlation(high, rank(volume), 5)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(high, Rank(volume), 5)

    Polars Implementation Notes
    ---------------------------
    Stage the CS rank of volume before the TS corr.

    Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(cs_rank(pl.col("volume")).alias("__a044_rv"))
    return staged.select(
        (-1.0 * ts_corr(pl.col("high"), pl.col("__a044_rv"), 5)).alias("alpha044")
    ).to_series()


def alpha055(panel: pl.DataFrame) -> pl.Series:
    """Alpha #055 — Negative correlation between %K rank and volume rank.

    WorldQuant Formula
    ------------------
        -1 * correlation(
            rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
            rank(volume),
            6,
        )

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(
            Rank((close - Ts_Min(low, 12)) / (Ts_Max(high, 12) - Ts_Min(low, 12))),
            Rank(volume),
            6,
        )

    Polars Implementation Notes
    ---------------------------
    Compute the %K series TS-wise, CS rank it, CS rank volume, then TS corr.

    Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    pct_k = (pl.col("close") - ts_min(pl.col("low"), 12)) / (
        ts_max(pl.col("high"), 12) - ts_min(pl.col("low"), 12)
    )
    staged = panel.with_columns(
        pct_k.alias("__a055_pk"),
    )
    staged2 = staged.with_columns(
        cs_rank(pl.col("__a055_pk")).alias("__a055_rk"),
        cs_rank(pl.col("volume")).alias("__a055_rv"),
    )
    return staged2.select(
        (-1.0 * ts_corr(pl.col("__a055_rk"), pl.col("__a055_rv"), 6)).alias("alpha055")
    ).to_series()


def alpha060(panel: pl.DataFrame) -> pl.Series:
    """Alpha #060 — Williams %R volume rank minus argmax rank, scaled.

    WorldQuant Formula
    ------------------
        -1 * (
            2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
            - scale(rank(ts_argmax(close, 10)))
        )

    Legacy AQML Expression
    ----------------------
        -1 * (
            2 * Scale(Rank(((close - low) - (high - close)) / (high - low) * volume))
            - Scale(Rank(Ts_ArgMax(close, 10)))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS-scaled rank chains, one based on the Williams-%R-style payload,
    the other on TS argmax of close.

    Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    williams = (
        ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close")))
        / (pl.col("high") - pl.col("low"))
        * pl.col("volume")
    )
    staged = panel.with_columns(
        williams.alias("__a060_will"),
        ts_argmax(pl.col("close"), 10).alias("__a060_arg"),
    )
    staged2 = staged.with_columns(
        cs_rank(pl.col("__a060_will")).alias("__a060_rwill"),
        cs_rank(pl.col("__a060_arg")).alias("__a060_rarg"),
    )
    return staged2.select(
        (-1.0 * (2.0 * cs_scale(pl.col("__a060_rwill")) - cs_scale(pl.col("__a060_rarg")))).alias(
            "alpha060"
        )
    ).to_series()


def alpha065(panel: pl.DataFrame) -> pl.Series:
    """Alpha #065 — Volume-weighted price vs adv60 corr rank vs open-min rank.

    WorldQuant Formula
    ------------------
        if rank(correlation(open*0.0078 + vwap*0.9922, sum(adv60, 9), 6))
              < rank(open - ts_min(open, 14))
        then -1 else 1

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(open*0.0078 + vwap*0.9922, Ts_Sum(adv60, 9), 6))
                < Rank(open - Ts_Min(open, 14)), -1, 1)

    Polars Implementation Notes
    ---------------------------
    AQML uses ``Ts_Sum`` (rolling sum); STHSF reference uses ``sma`` (rolling
    mean), so STHSF parity may diverge by a constant scale factor that
    drops out of CS rank — but tie-breaking around boundary cases can
    flip a few signs. We follow AQML.

    Required panel columns: ``open``, ``vwap``, ``adv60``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("open") * 0.0078 + pl.col("vwap") * 0.9922
    sum_adv60 = ts_sum(pl.col("adv60"), 9)
    staged = panel.with_columns(
        ts_corr(weighted, sum_adv60, 6).alias("__a065_corr"),
        (pl.col("open") - ts_min(pl.col("open"), 14)).alias("__a065_omin"),
    )
    staged2 = staged.with_columns(
        cs_rank(pl.col("__a065_corr")).alias("__a065_rcorr"),
        cs_rank(pl.col("__a065_omin")).alias("__a065_romin"),
    )
    return staged2.select(
        if_then_else(pl.col("__a065_rcorr") < pl.col("__a065_romin"), -1.0, 1.0)
        .cast(pl.Float64)
        .alias("alpha065")
    ).to_series()


def alpha068(panel: pl.DataFrame) -> pl.Series:
    """Alpha #068 — Composite price/adv15 rank vs weighted-price delta.

    WorldQuant Formula
    ------------------
        if ts_rank(correlation(rank(high), rank(adv15), 9), 14)
              < rank(delta(close*0.518 + low*0.482, 1))
        then -1 else 1

    Legacy AQML Expression
    ----------------------
        If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14)
                < Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)

    Polars Implementation Notes
    ---------------------------
    Stage CS rank of high and adv15 → TS corr → TS rank → compare with
    CS rank of TS delta of weighted price.

    Required panel columns: ``high``, ``adv15``, ``close``, ``low``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("high")).alias("__a068_rh"),
        cs_rank(pl.col("adv15")).alias("__a068_ra"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a068_rh"), pl.col("__a068_ra"), 9).alias("__a068_corr"),
        delta(pl.col("close") * 0.518 + pl.col("low") * 0.482, 1).alias("__a068_dpx"),
    )
    staged3 = staged2.with_columns(
        ts_rank(pl.col("__a068_corr"), 14).alias("__a068_trc"),
        cs_rank(pl.col("__a068_dpx")).alias("__a068_rdpx"),
    )
    return staged3.select(
        if_then_else(pl.col("__a068_trc") < pl.col("__a068_rdpx"), -1.0, 1.0)
        .cast(pl.Float64)
        .alias("alpha068")
    ).to_series()


def alpha071(panel: pl.DataFrame) -> pl.Series:
    """Alpha #071 — Decayed correlation vs decayed weighted-price-square rank, max.

    WorldQuant Formula
    ------------------
        max(
          ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16),
          ts_rank(decay_linear(rank(low + open - 2*vwap)^2, 16), 4)
        )

    Legacy AQML Expression
    ----------------------
        Max(
          Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16),
          Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4)
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel TS chains, combined element-wise via ``max_horizontal``.

    Required panel columns: ``close``, ``adv180``, ``low``, ``open``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(
        ts_rank(pl.col("close"), 3).alias("__a071_trc"),
        ts_rank(pl.col("adv180"), 12).alias("__a071_tra"),
        cs_rank(pl.col("low") + pl.col("open") - pl.col("vwap") - pl.col("vwap")).alias(
            "__a071_rlovw"
        ),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a071_trc"), pl.col("__a071_tra"), 18).alias("__a071_corr"),
        pl.col("__a071_rlovw").pow(2).alias("__a071_rsq"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a071_corr"), 4).alias("__a071_dl1"),
        ts_decay_linear(pl.col("__a071_rsq"), 16).alias("__a071_dl2"),
    )
    p1 = ts_rank(pl.col("__a071_dl1"), 16)
    p2 = ts_rank(pl.col("__a071_dl2"), 4)
    return staged3.select(pmax(p1, p2).alias("alpha071")).to_series()


def alpha072(panel: pl.DataFrame) -> pl.Series:
    """Alpha #072 — Decayed mid-price vs adv40 corr / decayed VWAP-volume corr.

    WorldQuant Formula
    ------------------
        rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
        / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

    Legacy AQML Expression
    ----------------------
        Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10))
        / Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))

    Polars Implementation Notes
    ---------------------------
    Two TS chains each ending in CS rank; the final result is their ratio.

    Required panel columns: ``high``, ``low``, ``adv40``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    staged = panel.with_columns(
        ts_rank(pl.col("vwap"), 4).alias("__a072_trv"),
        ts_rank(pl.col("volume"), 19).alias("__a072_trvol"),
    )
    staged2 = staged.with_columns(
        ts_corr(mid, pl.col("adv40"), 9).alias("__a072_corr1"),
        ts_corr(pl.col("__a072_trv"), pl.col("__a072_trvol"), 7).alias("__a072_corr2"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a072_corr1"), 10).alias("__a072_dl1"),
        ts_decay_linear(pl.col("__a072_corr2"), 3).alias("__a072_dl2"),
    )
    return staged3.select(
        (cs_rank(pl.col("__a072_dl1")) / cs_rank(pl.col("__a072_dl2"))).alias("alpha072")
    ).to_series()


def alpha073(panel: pl.DataFrame) -> pl.Series:
    """Alpha #073 — Negative max of decayed VWAP delta rank and blend reversal rank.

    WorldQuant Formula
    ------------------
        -1 * max(
            rank(decay_linear(delta(vwap, 5), 3)),
            ts_rank(decay_linear(-delta(open*0.147155 + low*0.852845, 2)
                    / (open*0.147155 + low*0.852845), 3), 17)
        )

    Legacy AQML Expression
    ----------------------
        -1 * Max(
            Rank(Ts_DecayLinear(Delta(vwap, 5), 3)),
            Ts_Rank(Ts_DecayLinear(
                -1 * Delta(open * 0.147155 + low * 0.852845, 2)
                / (open * 0.147155 + low * 0.852845), 3), 17)
        )

    Polars Implementation Notes
    ---------------------------
    The paper's fractional windows round to the standard STHSF integer
    windows: 5, 3, 2, 3 and 17.

    Required panel columns: ``vwap``, ``open``, ``low``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    blend = pl.col("open") * 0.147155 + pl.col("low") * 0.852845
    staged1 = panel.with_columns(
        ts_decay_linear(delta(pl.col("vwap"), 5), 3).alias("__a073_p1_raw"),
        ts_decay_linear(-1.0 * delta(blend, 2) / blend, 3).alias("__a073_p2_raw"),
    )
    staged2 = staged1.with_columns(
        cs_rank(pl.col("__a073_p1_raw")).alias("__a073_p1"),
        ts_rank(pl.col("__a073_p2_raw"), 17).alias("__a073_p2"),
    )
    return staged2.select(
        (-1.0 * pmax(pl.col("__a073_p1"), pl.col("__a073_p2"))).alias("alpha073").cast(pl.Float64)
    ).to_series()


def alpha074(panel: pl.DataFrame) -> pl.Series:
    """Alpha #074 — Close vs adv30-sum corr vs weighted-price-volume corr.

    WorldQuant Formula
    ------------------
        if rank(correlation(close, sum(adv30, 37), 15))
              < rank(correlation(rank(high*0.0261 + vwap*0.9739), rank(volume), 11))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15))
                < Rank(Ts_Corr(Rank(high*0.0261 + vwap*0.9739), Rank(volume), 11)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    AQML uses ``Ts_Sum``; STHSF uses ``sma`` (rolling mean) — STHSF parity
    may diverge as for #065. False branch is 0 (not 1) per AQML.

    Required panel columns: ``close``, ``adv30``, ``high``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("high") * 0.0261 + pl.col("vwap") * 0.9739
    sum_adv30 = ts_sum(pl.col("adv30"), 37)
    staged = panel.with_columns(
        cs_rank(weighted).alias("__a074_rw"),
        cs_rank(pl.col("volume")).alias("__a074_rv"),
        ts_corr(pl.col("close"), sum_adv30, 15).alias("__a074_corr1"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a074_rw"), pl.col("__a074_rv"), 11).alias("__a074_corr2"),
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a074_corr1")).alias("__a074_rcorr1"),
        cs_rank(pl.col("__a074_corr2")).alias("__a074_rcorr2"),
    )
    return staged3.select(
        if_then_else(pl.col("__a074_rcorr1") < pl.col("__a074_rcorr2"), -1.0, 0.0)
        .cast(pl.Float64)
        .alias("alpha074")
    ).to_series()


def alpha077(panel: pl.DataFrame) -> pl.Series:
    """Alpha #077 — Min of two decayed-rank features.

    WorldQuant Formula
    ------------------
        min(
          rank(decay_linear((high+low)/2 + high - vwap - high, 20)),
          rank(decay_linear(correlation((high+low)/2, adv40, 3), 6))
        )

    Legacy AQML Expression
    ----------------------
        Min(
          Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)),
          Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6))
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel TS+CS chains combined by element-wise min.

    Required panel columns: ``high``, ``low``, ``vwap``, ``adv40``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    payload = mid + pl.col("high") - pl.col("vwap") - pl.col("high")
    staged = panel.with_columns(
        ts_decay_linear(payload, 20).alias("__a077_dl1"),
        ts_decay_linear(ts_corr(mid, pl.col("adv40"), 3), 6).alias("__a077_dl2"),
    )
    staged2 = staged.with_columns(
        cs_rank(pl.col("__a077_dl1")).alias("__a077_r1"),
        cs_rank(pl.col("__a077_dl2")).alias("__a077_r2"),
    )
    return staged2.select(
        pmin(pl.col("__a077_r1"), pl.col("__a077_r2")).alias("alpha077")
    ).to_series()


def alpha078(panel: pl.DataFrame) -> pl.Series:
    """Alpha #078 — Power composition of two correlation ranks.

    WorldQuant Formula
    ------------------
        rank(correlation(sum(low*0.352 + vwap*0.648, 20), sum(adv40, 20), 7))
        ^ rank(correlation(rank(vwap), rank(volume), 6))

    Legacy AQML Expression
    ----------------------
        Power(
          Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), Ts_Sum(adv40, 20), 7)),
          Rank(Ts_Corr(Rank(vwap), Rank(volume), 6))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS-rank values produce the (base, exponent) pair for ``Power``.

    Required panel columns: ``low``, ``vwap``, ``adv40``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("low") * 0.352 + pl.col("vwap") * 0.648
    sum_w = ts_sum(weighted, 20)
    sum_adv40 = ts_sum(pl.col("adv40"), 20)
    staged = panel.with_columns(
        cs_rank(pl.col("vwap")).alias("__a078_rv"),
        cs_rank(pl.col("volume")).alias("__a078_rvol"),
        ts_corr(sum_w, sum_adv40, 7).alias("__a078_corr1"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a078_rv"), pl.col("__a078_rvol"), 6).alias("__a078_corr2"),
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a078_corr1")).alias("__a078_base"),
        cs_rank(pl.col("__a078_corr2")).alias("__a078_exp"),
    )
    return staged3.select(
        power(pl.col("__a078_base"), pl.col("__a078_exp")).alias("alpha078")
    ).to_series()


def alpha081(panel: pl.DataFrame) -> pl.Series:
    """Alpha #081 — Log-product of double-rank corr vs vwap-volume corr.

    WorldQuant Formula
    ------------------
        if rank(log(product(rank(rank(correlation(vwap, sum(adv10, 50), 8)^4)), 15)))
              < rank(correlation(rank(vwap), rank(volume), 5))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, Ts_Sum(adv10, 50), 8), 4))), 15)))
                < Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    Heavy nested expression. Stage step-by-step.

    Required panel columns: ``vwap``, ``adv10``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    sum_adv10 = ts_sum(pl.col("adv10"), 50)
    staged = panel.with_columns(
        ts_corr(pl.col("vwap"), sum_adv10, 8).alias("__a081_corr1"),
        cs_rank(pl.col("vwap")).alias("__a081_rv"),
        cs_rank(pl.col("volume")).alias("__a081_rvol"),
    )
    staged2 = staged.with_columns(
        pl.col("__a081_corr1").pow(4).alias("__a081_corr1_p4"),
        ts_corr(pl.col("__a081_rv"), pl.col("__a081_rvol"), 5).alias("__a081_corr2"),
    )
    staged3 = staged2.with_columns(
        cs_rank(cs_rank(pl.col("__a081_corr1_p4"))).alias("__a081_rr"),
    )
    staged4 = staged3.with_columns(
        ts_product(pl.col("__a081_rr"), 15).alias("__a081_prod"),
    )
    staged5 = staged4.with_columns(
        cs_rank(log_(pl.col("__a081_prod"))).alias("__a081_lhs"),
        cs_rank(pl.col("__a081_corr2")).alias("__a081_rhs"),
    )
    return staged5.select(
        if_then_else(pl.col("__a081_lhs") < pl.col("__a081_rhs"), -1.0, 0.0)
        .cast(pl.Float64)
        .alias("alpha081")
    ).to_series()


def alpha083(panel: pl.DataFrame) -> pl.Series:
    """Alpha #083 — Range/MA delay rank times volume rank-squared, scaled.

    WorldQuant Formula
    ------------------
        (rank(delay((high - low) / (sum(close, 5) / 5), 2)) * rank(rank(volume)))
        / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))

    Legacy AQML Expression
    ----------------------
        (Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * Rank(Rank(volume)))
        / (((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))

    Polars Implementation Notes
    ---------------------------
    Stage the range-over-MA series, its delay, and the double-CS-rank of
    volume; final division involves the present-day ratio and price gap.

    Required panel columns: ``high``, ``low``, ``close``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    range_ma = (pl.col("high") - pl.col("low")) / (ts_sum(pl.col("close"), 5) / 5.0)
    staged = panel.with_columns(
        range_ma.alias("__a083_rm"),
    )
    staged2 = staged.with_columns(
        delay(pl.col("__a083_rm"), 2).alias("__a083_drm"),
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a083_drm")).alias("__a083_r1"),
        cs_rank(cs_rank(pl.col("volume"))).alias("__a083_r2"),
    )
    return staged3.select(
        (
            (pl.col("__a083_r1") * pl.col("__a083_r2"))
            / (pl.col("__a083_rm") / (pl.col("vwap") - pl.col("close")))
        ).alias("alpha083")
    ).to_series()


def alpha085(panel: pl.DataFrame) -> pl.Series:
    """Alpha #085 — Power composition of weighted-price/adv30 and rank-rank corrs.

    WorldQuant Formula
    ------------------
        rank(correlation(high*0.876 + close*0.124, adv30, 10))
        ^ rank(correlation(ts_rank((high+low)/2, 4), ts_rank(volume, 10), 7))

    Legacy AQML Expression
    ----------------------
        Power(
          Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)),
          Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS ranks form base/exponent of pow.

    Required panel columns: ``high``, ``close``, ``adv30``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col("high") * 0.876 + pl.col("close") * 0.124
    mid = (pl.col("high") + pl.col("low")) / 2.0
    staged = panel.with_columns(
        ts_corr(weighted, pl.col("adv30"), 10).alias("__a085_corr1"),
        ts_rank(mid, 4).alias("__a085_trmid"),
        ts_rank(pl.col("volume"), 10).alias("__a085_trv"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a085_trmid"), pl.col("__a085_trv"), 7).alias("__a085_corr2"),
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a085_corr1")).alias("__a085_base"),
        cs_rank(pl.col("__a085_corr2")).alias("__a085_exp"),
    )
    return staged3.select(
        power(pl.col("__a085_base"), pl.col("__a085_exp")).alias("alpha085")
    ).to_series()


def alpha088(panel: pl.DataFrame) -> pl.Series:
    """Alpha #088 — Min of decayed rank-spread and decayed correlation rank.

    WorldQuant Formula
    ------------------
        min(
          rank(decay_linear(rank(open) + rank(low) - rank(high) - rank(close), 8)),
          ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
        )

    Legacy AQML Expression
    ----------------------
        Min(
          Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - (Rank(high) + Rank(close)), 8)),
          Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3)
        )

    Polars Implementation Notes
    ---------------------------
    Two parallel chains, combined by element-wise min.

    Required panel columns: ``open``, ``low``, ``high``, ``close``, ``adv60``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    spread = (
        cs_rank(pl.col("open"))
        + cs_rank(pl.col("low"))
        - cs_rank(pl.col("high"))
        - cs_rank(pl.col("close"))
    )
    staged = panel.with_columns(
        spread.alias("__a088_spread"),
        ts_rank(pl.col("close"), 8).alias("__a088_trc"),
        ts_rank(pl.col("adv60"), 21).alias("__a088_tra"),
    )
    staged2 = staged.with_columns(
        ts_decay_linear(pl.col("__a088_spread"), 8).alias("__a088_dl1"),
        ts_corr(pl.col("__a088_trc"), pl.col("__a088_tra"), 8).alias("__a088_corr"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a088_corr"), 7).alias("__a088_dl2"),
        cs_rank(pl.col("__a088_dl1")).alias("__a088_r1"),
    )
    p1 = pl.col("__a088_r1")
    p2 = ts_rank(pl.col("__a088_dl2"), 3)
    return staged3.select(pmin(p1, p2).alias("alpha088")).to_series()


def alpha094(panel: pl.DataFrame) -> pl.Series:
    """Alpha #094 — Negative power of VWAP-trough rank with correlation exponent.

    WorldQuant Formula
    ------------------
        -1 * rank(vwap - ts_min(vwap, 12))
              ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Power(
          Rank(vwap - Ts_Min(vwap, 12)),
          Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)
        )

    Polars Implementation Notes
    ---------------------------
    Stage TS ranks before TS corr; CS rank base; TS rank exponent.

    Required panel columns: ``vwap``, ``adv60``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    base_inner = pl.col("vwap") - ts_min(pl.col("vwap"), 12)
    staged = panel.with_columns(
        ts_rank(pl.col("vwap"), 20).alias("__a094_trv"),
        ts_rank(pl.col("adv60"), 4).alias("__a094_tra"),
        base_inner.alias("__a094_base_inner"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a094_trv"), pl.col("__a094_tra"), 18).alias("__a094_corr"),
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a094_base_inner")).alias("__a094_base"),
        ts_rank(pl.col("__a094_corr"), 3).alias("__a094_exp"),
    )
    return staged3.select(
        (-1.0 * power(pl.col("__a094_base"), pl.col("__a094_exp"))).alias("alpha094")
    ).to_series()


def alpha099(panel: pl.DataFrame) -> pl.Series:
    """Alpha #099 — Mid-price-adv60 corr vs low-volume corr.

    WorldQuant Formula
    ------------------
        if rank(correlation(sum((high+low)/2, 19), sum(adv60, 19), 8))
              < rank(correlation(low, volume, 6))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8))
                < Rank(Ts_Corr(low, volume, 6)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    AQML ``Ts_Sum`` vs STHSF ``sma`` again — STHSF parity may diverge for
    constant-factor reasons. False branch is 0.

    Required panel columns: ``high``, ``low``, ``adv60``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col("high") + pl.col("low")) / 2.0
    sum_mid = ts_sum(mid, 19)
    sum_adv60 = ts_sum(pl.col("adv60"), 19)
    staged = panel.with_columns(
        ts_corr(sum_mid, sum_adv60, 8).alias("__a099_corr1"),
        ts_corr(pl.col("low"), pl.col("volume"), 6).alias("__a099_corr2"),
    )
    staged2 = staged.with_columns(
        cs_rank(pl.col("__a099_corr1")).alias("__a099_r1"),
        cs_rank(pl.col("__a099_corr2")).alias("__a099_r2"),
    )
    return staged2.select(
        if_then_else(pl.col("__a099_r1") < pl.col("__a099_r2"), -1.0, 0.0)
        .cast(pl.Float64)
        .alias("alpha099")
    ).to_series()


# ---------------------------------------------------------------------------
# Self-registration to ALPHA101_REGISTRY
# ---------------------------------------------------------------------------


_ENTRIES = [
    FactorEntry(
        id="alpha002",
        impl=alpha002,
        direction="reverse",
        category="volume_price",
        description="Volume change rank vs intraday return rank correlation",
        legacy_aqml_expr=(
            "-1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)"
        ),
        references=("Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 2",),
        formula_doc_path="docs/factor_library/alpha101/alpha_002.md",
    ),
    FactorEntry(
        id="alpha003",
        impl=alpha003,
        direction="reverse",
        category="volume_price",
        description="Open price rank vs volume rank correlation",
        legacy_aqml_expr="-1 * Ts_Corr(Rank(open), Rank(volume), 10)",
        references=("Kakushadze 2015, eq. 3",),
        formula_doc_path="docs/factor_library/alpha101/alpha_003.md",
    ),
    FactorEntry(
        id="alpha005",
        impl=alpha005,
        direction="reverse",
        category="volume_price",
        description="Open vs 10d VWAP mean rank, scaled by close-VWAP rank deviation",
        legacy_aqml_expr=("Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))"),
        references=("Kakushadze 2015, eq. 5",),
        formula_doc_path="docs/factor_library/alpha101/alpha_005.md",
    ),
    FactorEntry(
        id="alpha006",
        impl=alpha006,
        direction="reverse",
        category="volume_price",
        description="10d correlation between open and volume",
        legacy_aqml_expr="-1 * Ts_Corr(open, volume, 10)",
        references=("Kakushadze 2015, eq. 6",),
        formula_doc_path="docs/factor_library/alpha101/alpha_006.md",
    ),
    FactorEntry(
        id="alpha012",
        impl=alpha012,
        direction="reverse",
        category="volume_price",
        description="Volume direction times negative price change",
        legacy_aqml_expr="Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))",
        references=("Kakushadze 2015, eq. 12",),
        formula_doc_path="docs/factor_library/alpha101/alpha_012.md",
    ),
    FactorEntry(
        id="alpha013",
        impl=alpha013,
        direction="reverse",
        category="volume_price",
        description="Negative rank of close-rank vs volume-rank covariance",
        legacy_aqml_expr="-1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))",
        references=("Kakushadze 2015, eq. 13",),
        formula_doc_path="docs/factor_library/alpha101/alpha_013.md",
    ),
    FactorEntry(
        id="alpha014",
        impl=alpha014,
        direction="reverse",
        category="volume_price",
        description="Returns-acceleration rank scaled by open-volume correlation",
        legacy_aqml_expr=("(-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)"),
        references=("Kakushadze 2015, eq. 14",),
        formula_doc_path="docs/factor_library/alpha101/alpha_014.md",
    ),
    FactorEntry(
        id="alpha015",
        impl=alpha015,
        direction="reverse",
        category="volume_price",
        description="Negative 3d sum of ranked high-volume rank correlation",
        legacy_aqml_expr=("-1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)"),
        references=("Kakushadze 2015, eq. 15",),
        formula_doc_path="docs/factor_library/alpha101/alpha_015.md",
    ),
    FactorEntry(
        id="alpha022",
        impl=alpha022,
        direction="reverse",
        category="volume_price",
        description="Change in high-vol correlation scaled by 20d stdev rank",
        legacy_aqml_expr=("-1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))"),
        references=("Kakushadze 2015, eq. 22",),
        formula_doc_path="docs/factor_library/alpha101/alpha_022.md",
    ),
    FactorEntry(
        id="alpha025",
        impl=alpha025,
        direction="reverse",
        category="volume_price",
        description="Rank of negative-returns × volume-weighted price-tail",
        legacy_aqml_expr="Rank((-1 * returns) * adv20 * vwap * (high - close))",
        references=("Kakushadze 2015, eq. 25",),
        formula_doc_path="docs/factor_library/alpha101/alpha_025.md",
    ),
    FactorEntry(
        id="alpha026",
        impl=alpha026,
        direction="reverse",
        category="volume_price",
        description="Max recent volume-rank vs high-rank correlation",
        legacy_aqml_expr=("-1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)"),
        references=("Kakushadze 2015, eq. 26",),
        formula_doc_path="docs/factor_library/alpha101/alpha_026.md",
    ),
    FactorEntry(
        id="alpha028",
        impl=alpha028,
        direction="reverse",
        category="volume_price",
        description="Standardised mid-price vs close gap with volume modifier",
        legacy_aqml_expr=("Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)"),
        references=("Kakushadze 2015, eq. 28",),
        formula_doc_path="docs/factor_library/alpha101/alpha_028.md",
    ),
    FactorEntry(
        id="alpha035",
        impl=alpha035,
        direction="reverse",
        category="volume_price",
        description="Volume rank × inverse range rank × inverse returns rank",
        legacy_aqml_expr=(
            "Ts_Rank(volume, 32) * (1 - Ts_Rank(close + high - low, 16)) "
            "* (1 - Ts_Rank(returns, 32))"
        ),
        references=("Kakushadze 2015, eq. 35",),
        formula_doc_path="docs/factor_library/alpha101/alpha_035.md",
    ),
    FactorEntry(
        id="alpha043",
        impl=alpha043,
        direction="reverse",
        category="volume_price",
        description="Volume-surge rank × 7d-decline rank",
        legacy_aqml_expr=("Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)"),
        references=("Kakushadze 2015, eq. 43",),
        formula_doc_path="docs/factor_library/alpha101/alpha_043.md",
    ),
    FactorEntry(
        id="alpha044",
        impl=alpha044,
        direction="reverse",
        category="volume_price",
        description="Negative high-vs-volume-rank correlation",
        legacy_aqml_expr="-1 * Ts_Corr(high, Rank(volume), 5)",
        references=("Kakushadze 2015, eq. 44",),
        formula_doc_path="docs/factor_library/alpha101/alpha_044.md",
    ),
    FactorEntry(
        id="alpha055",
        impl=alpha055,
        direction="reverse",
        category="volume_price",
        description="Negative correlation between %K rank and volume rank",
        legacy_aqml_expr=(
            "-1 * Ts_Corr(Rank((close - Ts_Min(low, 12)) / "
            "(Ts_Max(high, 12) - Ts_Min(low, 12))), Rank(volume), 6)"
        ),
        references=("Kakushadze 2015, eq. 55",),
        formula_doc_path="docs/factor_library/alpha101/alpha_055.md",
    ),
    FactorEntry(
        id="alpha060",
        impl=alpha060,
        direction="reverse",
        category="volume_price",
        description="Williams %R volume rank minus argmax rank, scaled",
        legacy_aqml_expr=(
            "-1 * (2 * Scale(Rank(((close - low) - (high - close)) / "
            "(high - low) * volume)) - Scale(Rank(Ts_ArgMax(close, 10))))"
        ),
        references=("Kakushadze 2015, eq. 60",),
        formula_doc_path="docs/factor_library/alpha101/alpha_060.md",
    ),
    FactorEntry(
        id="alpha065",
        impl=alpha065,
        direction="reverse",
        category="volume_price",
        description="Volume-weighted price vs adv60 correlation rank vs open-min rank",
        legacy_aqml_expr=(
            "If(Rank(Ts_Corr(open * 0.0078 + vwap * 0.9922, "
            "Ts_Sum(adv60, 9), 6)) < Rank(open - Ts_Min(open, 14)), -1, 1)"
        ),
        references=("Kakushadze 2015, eq. 65",),
        formula_doc_path="docs/factor_library/alpha101/alpha_065.md",
    ),
    FactorEntry(
        id="alpha068",
        impl=alpha068,
        direction="reverse",
        category="volume_price",
        description="Composite price/adv15 rank vs weighted-price delta",
        legacy_aqml_expr=(
            "If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14) "
            "< Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)"
        ),
        references=("Kakushadze 2015, eq. 68",),
        formula_doc_path="docs/factor_library/alpha101/alpha_068.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="alpha071",
        impl=alpha071,
        direction="reverse",
        category="volume_price",
        description="Decayed correlation vs decayed weighted-price-square rank, max",
        legacy_aqml_expr=(
            "Max(Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), "
            "Ts_Rank(adv180, 12), 18), 4), 16), "
            "Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4))"
        ),
        references=("Kakushadze 2015, eq. 71",),
        formula_doc_path="docs/factor_library/alpha101/alpha_071.md",
    ),
    FactorEntry(
        id="alpha072",
        impl=alpha072,
        direction="reverse",
        category="volume_price",
        description="Decayed mid-price vs adv40 corr / decayed VWAP-volume corr",
        legacy_aqml_expr=(
            "Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10)) / "
            "Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))"
        ),
        references=("Kakushadze 2015, eq. 72",),
        formula_doc_path="docs/factor_library/alpha101/alpha_072.md",
    ),
    FactorEntry(
        id="alpha073",
        impl=alpha073,
        direction="reverse",
        category="volume_price",
        description="Negative max of decayed VWAP delta rank and blend reversal rank",
        legacy_aqml_expr=(
            "-1 * Max(Rank(Ts_DecayLinear(Delta(vwap, 5), 3)), "
            "Ts_Rank(Ts_DecayLinear(-1 * Delta(open * 0.147155 + low * 0.852845, 2) "
            "/ (open * 0.147155 + low * 0.852845), 3), 17))"
        ),
        references=("Kakushadze 2015, eq. 73",),
        formula_doc_path="docs/factor_library/alpha101/alpha_073.md",
    ),
    FactorEntry(
        id="alpha074",
        impl=alpha074,
        direction="reverse",
        category="volume_price",
        description="Close vs adv30-sum corr vs weighted-price-volume corr",
        legacy_aqml_expr=(
            "If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15)) < "
            "Rank(Ts_Corr(Rank(high * 0.0261 + vwap * 0.9739), Rank(volume), 11)), -1, 0)"
        ),
        references=("Kakushadze 2015, eq. 74",),
        formula_doc_path="docs/factor_library/alpha101/alpha_074.md",
    ),
    FactorEntry(
        id="alpha077",
        impl=alpha077,
        direction="reverse",
        category="volume_price",
        description="Min of two decayed-rank features",
        legacy_aqml_expr=(
            "Min(Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)), "
            "Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6)))"
        ),
        references=("Kakushadze 2015, eq. 77",),
        formula_doc_path="docs/factor_library/alpha101/alpha_077.md",
    ),
    FactorEntry(
        id="alpha078",
        impl=alpha078,
        direction="reverse",
        category="volume_price",
        description="Power composition of two correlation ranks",
        legacy_aqml_expr=(
            "Power(Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), "
            "Ts_Sum(adv40, 20), 7)), Rank(Ts_Corr(Rank(vwap), Rank(volume), 6)))"
        ),
        references=("Kakushadze 2015, eq. 78",),
        formula_doc_path="docs/factor_library/alpha101/alpha_078.md",
    ),
    FactorEntry(
        id="alpha081",
        impl=alpha081,
        direction="reverse",
        category="volume_price",
        description="Log-product of double-rank corr vs vwap-volume corr",
        legacy_aqml_expr=(
            "If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, "
            "Ts_Sum(adv10, 50), 8), 4))), 15))) < "
            "Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)"
        ),
        references=("Kakushadze 2015, eq. 81",),
        formula_doc_path="docs/factor_library/alpha101/alpha_081.md",
    ),
    FactorEntry(
        id="alpha083",
        impl=alpha083,
        direction="reverse",
        category="volume_price",
        description="Range/MA delay rank times volume rank-squared, scaled",
        legacy_aqml_expr=(
            "(Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * "
            "Rank(Rank(volume))) / "
            "(((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))"
        ),
        references=("Kakushadze 2015, eq. 83",),
        formula_doc_path="docs/factor_library/alpha101/alpha_083.md",
    ),
    FactorEntry(
        id="alpha085",
        impl=alpha085,
        direction="reverse",
        category="volume_price",
        description="Power composition of weighted-price/adv30 and rank-rank correlations",
        legacy_aqml_expr=(
            "Power(Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)), "
            "Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7)))"
        ),
        references=("Kakushadze 2015, eq. 85",),
        formula_doc_path="docs/factor_library/alpha101/alpha_085.md",
    ),
    FactorEntry(
        id="alpha088",
        impl=alpha088,
        direction="reverse",
        category="volume_price",
        description="Min of decayed rank-spread and decayed correlation rank",
        legacy_aqml_expr=(
            "Min(Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - "
            "(Rank(high) + Rank(close)), 8)), "
            "Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), "
            "Ts_Rank(adv60, 21), 8), 7), 3))"
        ),
        references=("Kakushadze 2015, eq. 88",),
        formula_doc_path="docs/factor_library/alpha101/alpha_088.md",
    ),
    FactorEntry(
        id="alpha094",
        impl=alpha094,
        direction="reverse",
        category="volume_price",
        description="Negative power of VWAP-trough rank with correlation exponent",
        legacy_aqml_expr=(
            "-1 * Power(Rank(vwap - Ts_Min(vwap, 12)), "
            "Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3))"
        ),
        references=("Kakushadze 2015, eq. 94",),
        formula_doc_path="docs/factor_library/alpha101/alpha_094.md",
        quality_flag=1,
    ),
    FactorEntry(
        id="alpha099",
        impl=alpha099,
        direction="reverse",
        category="volume_price",
        description="Mid-price-adv60 corr vs low-volume corr",
        legacy_aqml_expr=(
            "If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8)) "
            "< Rank(Ts_Corr(low, volume, 6)), -1, 0)"
        ),
        references=("Kakushadze 2015, eq. 99",),
        formula_doc_path="docs/factor_library/alpha101/alpha_099.md",
    ),
]


for _e in _ENTRIES:
    register_alpha101(_e)
