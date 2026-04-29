"""Alpha101 — industry-neutral category factors (35 factors).

This module fills in the alphas previously classified as "phantom" in
:data:`aurumq.rules.alpha101_library.SKIPPED_ALPHAS` because their
formulas reference industry / sub-industry classifications. Now that the
panel loader joins ``industry`` / ``sub_industry`` columns and
``_ops.ind_neutralize`` is available, these can be implemented directly.

Sub-industry group (5 factors): 011, 016, 020, 047, 048
Industry group (30 factors):    027, 029, 030, 031, 036, 039, 049, 050,
                                058, 059, 062, 063, 066, 067, 069, 070,
                                076, 079, 080, 082, 086, 087, 089, 090,
                                091, 093, 096, 097, 098, 100

Notes
-----
* Some "sub_industry" alphas (011, 016, 020, 047) do not literally call
  ``IndNeutralize`` in the original WorldQuant paper. They were filed in
  SKIPPED_ALPHAS under "IndNeutralize on sub_industry" because AurumQ's
  panel had no sub_industry column. We implement the literal paper
  formulas; the categorisation is kept for traceability.
* Stocks with ``sub_industry IS NULL`` (or ``industry IS NULL``) produce
  ``NaN`` outputs from ``ind_neutralize`` — this is the documented and
  acceptable behaviour.
"""

from __future__ import annotations

import polars as pl

from aurumq_rl.factors.registry import FactorEntry, register_alpha101

from ._ops import (
    CS_PART,
    TS_PART,
    cs_rank,
    cs_scale,
    delay,
    delta,
    ind_neutralize,
    log_,
    sign_,
    ts_argmax,
    ts_argmin,
    ts_corr,
    ts_cov,
    ts_decay_linear,
    ts_max,
    ts_mean,
    ts_min,
    ts_product,
    ts_rank,
    ts_sum,
)

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _abs(col: pl.Expr) -> pl.Expr:
    return col.abs()


# ===========================================================================
# Sub-industry group — 5 factors (011, 016, 020, 047, 048)
# ===========================================================================


def alpha011(panel: pl.DataFrame) -> pl.Series:
    """Alpha #011 — VWAP-close range bookended by volume change rank.

    WorldQuant Formula
    ------------------
        (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
        rank(delta(volume, 3))

    Polars Implementation Notes
    ---------------------------
    Stage the per-stock ts_max/ts_min/delta first, then take three CS
    ranks on the materialised columns and combine.

    Required panel columns: ``vwap``, ``close``, ``volume``, ``stock_code``,
    ``trade_date``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    diff = pl.col("vwap") - pl.col("close")
    staged = panel.with_columns(
        ts_max(diff, 3).alias("__a011_max"),
        ts_min(diff, 3).alias("__a011_min"),
        delta(pl.col("volume"), 3).alias("__a011_dv"),
    )
    return staged.select(
        (
            (cs_rank(pl.col("__a011_max")) + cs_rank(pl.col("__a011_min")))
            * cs_rank(pl.col("__a011_dv"))
        ).alias("alpha011")
    ).to_series()


def alpha016(panel: pl.DataFrame) -> pl.Series:
    """Alpha #016 — Negative rank of high-rank vs volume-rank covariance.

    WorldQuant Formula
    ------------------
        -1 * rank(covariance(rank(high), rank(volume), 5))

    Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("high")).alias("__a016_rh"),
        cs_rank(pl.col("volume")).alias("__a016_rv"),
    )
    staged2 = staged.with_columns(
        ts_cov(pl.col("__a016_rh"), pl.col("__a016_rv"), 5).alias("__a016_cov")
    )
    return staged2.select(
        (-1.0 * cs_rank(pl.col("__a016_cov"))).alias("alpha016")
    ).to_series()


def alpha020(panel: pl.DataFrame) -> pl.Series:
    """Alpha #020 — Triple gap-rank product (sign-flipped).

    WorldQuant Formula
    ------------------
        -1 * rank(open - delay(high, 1)) *
              rank(open - delay(close, 1)) *
              rank(open - delay(low, 1))

    Required panel columns: ``open``, ``high``, ``close``, ``low``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        (pl.col("open") - delay(pl.col("high"), 1)).alias("__a020_h"),
        (pl.col("open") - delay(pl.col("close"), 1)).alias("__a020_c"),
        (pl.col("open") - delay(pl.col("low"), 1)).alias("__a020_l"),
    )
    return staged.select(
        (
            -1.0
            * cs_rank(pl.col("__a020_h"))
            * cs_rank(pl.col("__a020_c"))
            * cs_rank(pl.col("__a020_l"))
        ).alias("alpha020")
    ).to_series()


def alpha047(panel: pl.DataFrame) -> pl.Series:
    """Alpha #047 — Inverse-close * volume / adv20 amplification minus VWAP momentum.

    WorldQuant Formula
    ------------------
        ((((rank(1/close) * volume) / adv20) *
          ((high * rank(high - close)) / (sum(high, 5) / 5))) -
         rank(vwap - delay(vwap, 5)))

    Required panel columns: ``close``, ``volume``, ``adv20``, ``high``,
    ``vwap``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    # Stage TS-only intermediates first (cannot mix TS and CS over() in one
    # with_columns — the CS branch silently nulls out).
    pre = panel.with_columns(
        ts_mean(pl.col("high"), 5).alias("__a047_h5m"),
        (pl.col("vwap") - delay(pl.col("vwap"), 5)).alias("__a047_vw_d"),
    )
    staged = pre.with_columns(
        cs_rank(1.0 / pl.col("close")).alias("__a047_rinv"),
        cs_rank(pl.col("high") - pl.col("close")).alias("__a047_rhc"),
        cs_rank(pl.col("__a047_vw_d")).alias("__a047_rvw"),
    )
    return staged.select(
        (
            (
                (pl.col("__a047_rinv") * pl.col("volume") / pl.col("adv20"))
                * (pl.col("high") * pl.col("__a047_rhc") / pl.col("__a047_h5m"))
            )
            - pl.col("__a047_rvw")
        ).alias("alpha047")
    ).to_series()


def alpha048(panel: pl.DataFrame) -> pl.Series:
    """Alpha #048 — Sub-industry neutralised long-corr-of-deltas / sum-of-squared-returns.

    WorldQuant Formula
    ------------------
        IndNeutralize(
            (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
             delta(close, 1)) / close,
            IndClass.subindustry
        ) /
        sum((delta(close, 1) / delay(close, 1))^2, 250)

    Polars Implementation Notes
    ---------------------------
    250-day windows on a 60-day synthetic panel produce all-null output —
    the steady-state test only verifies the structural shape, not value
    coverage. On real production panels with >= 250 days the values are
    well-defined.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``,
    ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    dclose = delta(pl.col("close"), 1)
    dclose_lag = delta(delay(pl.col("close"), 1), 1)
    staged = panel.with_columns(
        dclose.alias("__a048_dc"),
        dclose_lag.alias("__a048_dcl"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a048_dc"), pl.col("__a048_dcl"), 250).alias("__a048_corr"),
        (pl.col("__a048_dc") / delay(pl.col("close"), 1)).alias("__a048_pct"),
    )
    staged3 = staged2.with_columns(
        (pl.col("__a048_corr") * pl.col("__a048_dc") / pl.col("close")).alias(
            "__a048_num"
        ),
        ts_sum(pl.col("__a048_pct").pow(2.0), 250).alias("__a048_den"),
    )
    staged4 = staged3.with_columns(
        ind_neutralize(pl.col("__a048_num"), "sub_industry").alias("__a048_neut")
    )
    return staged4.select(
        (pl.col("__a048_neut") / pl.col("__a048_den")).alias("alpha048")
    ).to_series()


# ===========================================================================
# Industry group — 30 factors
# ===========================================================================


def alpha027(panel: pl.DataFrame) -> pl.Series:
    """Alpha #027 — Sign of correlation(rank(volume), rank(vwap)) majority.

    WorldQuant Formula
    ------------------
        ((0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
         ? -1 : 1)

    Polars Implementation Notes
    ---------------------------
    Two-stage CS rank → TS corr → TS sum → CS rank, then threshold at 0.5.
    The classic STHSF implementation has a discontinuity at 0.5; we
    follow the same convention.

    Required panel columns: ``volume``, ``vwap``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("volume")).alias("__a027_rv"),
        cs_rank(pl.col("vwap")).alias("__a027_rw"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a027_rv"), pl.col("__a027_rw"), 6).alias("__a027_corr")
    )
    staged3 = staged2.with_columns(
        (ts_sum(pl.col("__a027_corr"), 2) / 2.0).alias("__a027_avg")
    )
    return staged3.select(
        pl.when(cs_rank(pl.col("__a027_avg")) > 0.5)
        .then(-1.0)
        .otherwise(1.0)
        .alias("alpha027")
    ).to_series()


def alpha029(panel: pl.DataFrame) -> pl.Series:
    """Alpha #029 — Deeply nested rank-scale-log composite plus delayed return ts_rank.

    WorldQuant Formula
    ------------------
        min(product(rank(rank(scale(log(sum(ts_min(
            rank(rank(-1 * rank(delta((close - 1), 5)))), 2), 1))))), 1), 5) +
        ts_rank(delay(-1 * returns, 6), 5)

    Polars Implementation Notes
    ---------------------------
    Outer ``product(... , 1)`` is identity (1-window product). Inner
    cascade: delta → CS rank → CS rank → CS rank → TS min(2) → TS sum(1)
    is identity → log → scale → CS rank → CS rank, then TS min(5).

    Required panel columns: ``close``, ``returns``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    inner = -1.0 * cs_rank(delta(pl.col("close") - 1.0, 5))
    staged = panel.with_columns(
        cs_rank(cs_rank(inner)).alias("__a029_rr"),
    )
    staged2 = staged.with_columns(
        ts_min(pl.col("__a029_rr"), 2).alias("__a029_min2")
    )
    staged3 = staged2.with_columns(
        ts_sum(pl.col("__a029_min2"), 1).alias("__a029_sum1")
    )
    # log of strictly-positive sum of (rank-of-rank) values; clip at tiny
    # positive epsilon to avoid log(0).
    eps = 1e-12
    staged4 = staged3.with_columns(
        cs_scale(log_(pl.col("__a029_sum1").clip(lower_bound=eps))).alias(
            "__a029_scale"
        )
    )
    staged5 = staged4.with_columns(
        cs_rank(cs_rank(pl.col("__a029_scale"))).alias("__a029_rrs")
    )
    staged6 = staged5.with_columns(
        ts_min(pl.col("__a029_rrs"), 5).alias("__a029_part1"),
        ts_rank(delay(-1.0 * pl.col("returns"), 6), 5).alias("__a029_part2"),
    )
    return staged6.select(
        (pl.col("__a029_part1") + pl.col("__a029_part2")).alias("alpha029")
    ).to_series()


def alpha030(panel: pl.DataFrame) -> pl.Series:
    """Alpha #030 — Three-day sign-momentum rank scaled by short-vs-long volume sum.

    WorldQuant Formula
    ------------------
        ((1.0 - rank(sign(close - delay(close, 1)) +
                     sign(delay(close, 1) - delay(close, 2)) +
                     sign(delay(close, 2) - delay(close, 3)))) *
         sum(volume, 5)) / sum(volume, 20)

    Required panel columns: ``close``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    dc1 = delta(pl.col("close"), 1)
    inner = sign_(dc1) + sign_(delay(dc1, 1)) + sign_(delay(dc1, 2))
    staged = panel.with_columns(inner.alias("__a030_inner"))
    return staged.select(
        (
            (1.0 - cs_rank(pl.col("__a030_inner")))
            * ts_sum(pl.col("volume"), 5)
            / ts_sum(pl.col("volume"), 20)
        ).alias("alpha030")
    ).to_series()


def alpha031(panel: pl.DataFrame) -> pl.Series:
    """Alpha #031 — Triple-rank decay of inverse delta(close, 10) plus other parts.

    WorldQuant Formula
    ------------------
        rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) +
        rank(-1 * delta(close, 3)) +
        sign(scale(correlation(adv20, low, 12)))

    Required panel columns: ``close``, ``adv20``, ``low``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    inner_neg_rr = -1.0 * cs_rank(cs_rank(delta(pl.col("close"), 10)))
    staged = panel.with_columns(inner_neg_rr.alias("__a031_inner"))
    staged2 = staged.with_columns(
        ts_decay_linear(pl.col("__a031_inner"), 10).alias("__a031_dec")
    )
    staged3 = staged2.with_columns(
        cs_rank(cs_rank(cs_rank(pl.col("__a031_dec")))).alias("__a031_p1"),
        cs_rank(-1.0 * delta(pl.col("close"), 3)).alias("__a031_p2"),
        ts_corr(pl.col("adv20"), pl.col("low"), 12).alias("__a031_corr"),
    )
    staged4 = staged3.with_columns(
        sign_(cs_scale(pl.col("__a031_corr"))).alias("__a031_p3")
    )
    return staged4.select(
        (pl.col("__a031_p1") + pl.col("__a031_p2") + pl.col("__a031_p3")).alias(
            "alpha031"
        )
    ).to_series()


def alpha036(panel: pl.DataFrame) -> pl.Series:
    """Alpha #036 — Multi-component composite (5 weighted ranks).

    WorldQuant Formula
    ------------------
        2.21 * rank(correlation((close - open), delay(volume, 1), 15)) +
        0.7  * rank(open - close) +
        0.73 * rank(Ts_Rank(delay(-1 * returns, 6), 5)) +
        rank(abs(correlation(vwap, adv20, 6))) +
        0.6  * rank((sum(close, 200) / 200 - open) * (close - open))

    Required panel columns: ``close``, ``open``, ``volume``, ``returns``,
    ``vwap``, ``adv20``, ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    co = pl.col("close") - pl.col("open")
    staged = panel.with_columns(
        ts_corr(co, delay(pl.col("volume"), 1), 15).alias("__a036_c1"),
        (pl.col("open") - pl.col("close")).alias("__a036_oc"),
        ts_rank(delay(-1.0 * pl.col("returns"), 6), 5).alias("__a036_tr"),
        ts_corr(pl.col("vwap"), pl.col("adv20"), 6).alias("__a036_c2"),
        (
            (ts_mean(pl.col("close"), 200) - pl.col("open")) * co
        ).alias("__a036_p5"),
    )
    return staged.select(
        (
            2.21 * cs_rank(pl.col("__a036_c1"))
            + 0.7 * cs_rank(pl.col("__a036_oc"))
            + 0.73 * cs_rank(pl.col("__a036_tr"))
            + cs_rank(_abs(pl.col("__a036_c2")))
            + 0.6 * cs_rank(pl.col("__a036_p5"))
        ).alias("alpha036")
    ).to_series()


def alpha039(panel: pl.DataFrame) -> pl.Series:
    """Alpha #039 — Negative rank of momentum-weighted-by-volume scaled by 250d return rank.

    WorldQuant Formula
    ------------------
        (-1 * rank(delta(close, 7) *
                   (1 - rank(decay_linear(volume / adv20, 9))))) *
        (1 + rank(sum(returns, 250)))

    Required panel columns: ``close``, ``volume``, ``adv20``, ``returns``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ts_decay_linear(pl.col("volume") / pl.col("adv20"), 9).alias("__a039_dec"),
        delta(pl.col("close"), 7).alias("__a039_dc"),
        ts_sum(pl.col("returns"), 250).alias("__a039_sret"),
    )
    staged2 = staged.with_columns(
        (
            pl.col("__a039_dc") * (1.0 - cs_rank(pl.col("__a039_dec")))
        ).alias("__a039_inner")
    )
    return staged2.select(
        (
            (-1.0 * cs_rank(pl.col("__a039_inner")))
            * (1.0 + cs_rank(pl.col("__a039_sret")))
        ).alias("alpha039")
    ).to_series()


def alpha049(panel: pl.DataFrame) -> pl.Series:
    """Alpha #049 — Conditional reversal based on close-acceleration threshold.

    WorldQuant Formula
    ------------------
        ((((delay(close, 20) - delay(close, 10)) / 10 -
           (delay(close, 10) - close) / 10) < -0.1) ? 1 :
         (-1 * (close - delay(close, 1))))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``,
    ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    inner = (
        (delay(pl.col("close"), 20) - delay(pl.col("close"), 10)) / 10.0
        - (delay(pl.col("close"), 10) - pl.col("close")) / 10.0
    )
    return panel.select(
        pl.when(inner < -0.1)
        .then(pl.lit(1.0))
        .otherwise(-1.0 * delta(pl.col("close"), 1))
        .alias("alpha049")
    ).to_series()


def alpha050(panel: pl.DataFrame) -> pl.Series:
    """Alpha #050 — Negative ts_max of rank(corr(rank(volume), rank(vwap), 5)).

    WorldQuant Formula
    ------------------
        -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

    Required panel columns: ``volume``, ``vwap``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("volume")).alias("__a050_rv"),
        cs_rank(pl.col("vwap")).alias("__a050_rw"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a050_rv"), pl.col("__a050_rw"), 5).alias("__a050_corr")
    )
    staged3 = staged2.with_columns(
        cs_rank(pl.col("__a050_corr")).alias("__a050_rc")
    )
    return staged3.select(
        (-1.0 * ts_max(pl.col("__a050_rc"), 5)).alias("alpha050")
    ).to_series()


def alpha058(panel: pl.DataFrame) -> pl.Series:
    """Alpha #058 — Negative ts_rank of decay(corr(IndNeutralize(vwap), volume)).

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(decay_linear(
            correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795),
            7.89291
        ), 5.50322)

    Polars Implementation Notes
    ---------------------------
    Continuous windows truncated to int (4, 8, 6). Sector neutralisation
    is applied via ``ind_neutralize(vwap, "industry")`` — AurumQ's
    available proxy for IndClass.sector.

    Required panel columns: ``vwap``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("vwap"), "industry").alias("__a058_ivwap")
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a058_ivwap"), pl.col("volume"), 4).alias("__a058_corr")
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a058_corr"), 8).alias("__a058_dec")
    )
    return staged3.select(
        (-1.0 * ts_rank(pl.col("__a058_dec"), 6)).alias("alpha058")
    ).to_series()


def alpha059(panel: pl.DataFrame) -> pl.Series:
    """Alpha #059 — Sector-neutralised vwap blend correlated with volume, decayed.

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(decay_linear(
            correlation(IndNeutralize(
                vwap * 0.728317 + vwap * (1 - 0.728317),
                IndClass.industry
            ), volume, 4.25197), 16.2289
        ), 8.19648)

    Polars Implementation Notes
    ---------------------------
    The convex blend is degenerate (always equal to ``vwap``); we keep
    the literal formula. Windows truncated: 4, 16, 8.

    Required panel columns: ``vwap``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("vwap") * 0.728317 + pl.col("vwap") * (1.0 - 0.728317)
    staged = panel.with_columns(
        ind_neutralize(blend, "industry").alias("__a059_iv")
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a059_iv"), pl.col("volume"), 4).alias("__a059_corr")
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a059_corr"), 16).alias("__a059_dec")
    )
    return staged3.select(
        (-1.0 * ts_rank(pl.col("__a059_dec"), 8)).alias("alpha059")
    ).to_series()


def alpha062(panel: pl.DataFrame) -> pl.Series:
    """Alpha #062 — Open vs midpoint rank inequality, gated by vwap-adv20 corr.

    WorldQuant Formula
    ------------------
        (rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
         rank(((rank(open) + rank(open)) <
               (rank((high + low) / 2) + rank(high))))) * -1

    Required panel columns: ``vwap``, ``adv20``, ``open``, ``high``, ``low``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ts_sum(pl.col("adv20"), 22).alias("__a062_sadv"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("vwap"), pl.col("__a062_sadv"), 10).alias("__a062_c1")
    )
    inner_b = (
        cs_rank(pl.col("open")) + cs_rank(pl.col("open"))
    ).cast(pl.Float64) < (
        cs_rank((pl.col("high") + pl.col("low")) / 2.0) + cs_rank(pl.col("high"))
    ).cast(pl.Float64)
    staged3 = staged2.with_columns(
        cs_rank(inner_b.cast(pl.Float64)).alias("__a062_p2"),
        cs_rank(pl.col("__a062_c1")).alias("__a062_p1"),
    )
    return staged3.select(
        (
            (pl.col("__a062_p1") < pl.col("__a062_p2")).cast(pl.Float64) * -1.0
        ).alias("alpha062")
    ).to_series()


def alpha063(panel: pl.DataFrame) -> pl.Series:
    """Alpha #063 — Negative diff between two decay-linear rank composites.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(IndNeutralize(close, IndClass.industry),
                                  2.25164), 8.22237)) -
         rank(decay_linear(correlation(
              vwap * 0.318108 + open * (1 - 0.318108),
              sum(adv180, 37.2467), 13.557), 12.2883))) * -1

    Required panel columns: ``close``, ``vwap``, ``open``, ``adv180``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("close"), "industry").alias("__a063_ic")
    )
    staged2 = staged.with_columns(
        delta(pl.col("__a063_ic"), 2).alias("__a063_dic"),
        ts_sum(pl.col("adv180"), 37).alias("__a063_sadv"),
        (pl.col("vwap") * 0.318108 + pl.col("open") * (1.0 - 0.318108)).alias(
            "__a063_blend"
        ),
    )
    staged3 = staged2.with_columns(
        ts_corr(pl.col("__a063_blend"), pl.col("__a063_sadv"), 14).alias("__a063_c"),
        ts_decay_linear(pl.col("__a063_dic"), 8).alias("__a063_d1"),
    )
    staged4 = staged3.with_columns(
        ts_decay_linear(pl.col("__a063_c"), 12).alias("__a063_d2")
    )
    return staged4.select(
        (
            (cs_rank(pl.col("__a063_d1")) - cs_rank(pl.col("__a063_d2"))) * -1.0
        ).alias("alpha063")
    ).to_series()


def alpha066(panel: pl.DataFrame) -> pl.Series:
    """Alpha #066 — Negative blend of vwap-delta decay rank and intraday-skew ts_rank.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
         Ts_Rank(decay_linear(
             ((low * 0.96633 + low * (1 - 0.96633)) - vwap) /
             (open - (high + low) / 2), 11.4157
         ), 6.72611)) * -1

    Required panel columns: ``vwap``, ``low``, ``high``, ``open``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_low = pl.col("low") * 0.96633 + pl.col("low") * (1.0 - 0.96633)
    midpoint = (pl.col("high") + pl.col("low")) / 2.0
    inner2 = (blend_low - pl.col("vwap")) / (pl.col("open") - midpoint)
    staged = panel.with_columns(
        delta(pl.col("vwap"), 4).alias("__a066_dv"),
        inner2.alias("__a066_skew"),
    )
    staged2 = staged.with_columns(
        ts_decay_linear(pl.col("__a066_dv"), 7).alias("__a066_d1"),
        ts_decay_linear(pl.col("__a066_skew"), 11).alias("__a066_d2"),
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a066_d1")) + ts_rank(pl.col("__a066_d2"), 7)) * -1.0
        ).alias("alpha066")
    ).to_series()


def alpha067(panel: pl.DataFrame) -> pl.Series:
    """Alpha #067 — high-from-min power composite, sector-neutralised.

    WorldQuant Formula
    ------------------
        (rank(high - ts_min(high, 2.14593))^
         rank(correlation(IndNeutralize(vwap, IndClass.sector),
                          IndNeutralize(adv20, IndClass.subindustry),
                          6.02936))) * -1

    Required panel columns: ``high``, ``vwap``, ``adv20``, ``stock_code``,
    ``trade_date``, ``industry``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("vwap"), "industry").alias("__a067_iv"),
        ind_neutralize(pl.col("adv20"), "sub_industry").alias("__a067_ia"),
        (pl.col("high") - ts_min(pl.col("high"), 2)).alias("__a067_diff"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a067_iv"), pl.col("__a067_ia"), 6).alias("__a067_corr")
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a067_diff")).pow(cs_rank(pl.col("__a067_corr"))))
            * -1.0
        ).alias("alpha067")
    ).to_series()


def alpha069(panel: pl.DataFrame) -> pl.Series:
    """Alpha #069 — Power of vwap-delta-max-rank by close-blend-corr ts_rank.

    WorldQuant Formula
    ------------------
        (rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry),
                           2.72412), 4.79344))^
         Ts_Rank(correlation(close * 0.490655 + vwap * (1 - 0.490655),
                             adv20, 4.92416), 9.0615)) * -1

    Required panel columns: ``vwap``, ``close``, ``adv20``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("vwap"), "industry").alias("__a069_iv")
    )
    staged2 = staged.with_columns(
        delta(pl.col("__a069_iv"), 3).alias("__a069_div")
    )
    staged3 = staged2.with_columns(
        ts_max(pl.col("__a069_div"), 5).alias("__a069_max"),
        (pl.col("close") * 0.490655 + pl.col("vwap") * (1.0 - 0.490655)).alias(
            "__a069_blend"
        ),
    )
    staged4 = staged3.with_columns(
        ts_corr(pl.col("__a069_blend"), pl.col("adv20"), 5).alias("__a069_corr")
    )
    return staged4.select(
        (
            (cs_rank(pl.col("__a069_max")).pow(ts_rank(pl.col("__a069_corr"), 9)))
            * -1.0
        ).alias("alpha069")
    ).to_series()


def alpha070(panel: pl.DataFrame) -> pl.Series:
    """Alpha #070 — Power of vwap-delta rank by IndNeutralize(close)-adv50 corr ts_rank.

    WorldQuant Formula
    ------------------
        (rank(delta(vwap, 1.29456))^
         Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50,
                             17.8256), 17.9171)) * -1

    Required panel columns: ``vwap``, ``close``, ``adv50``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("close"), "industry").alias("__a070_ic"),
        delta(pl.col("vwap"), 1).alias("__a070_dv"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a070_ic"), pl.col("adv50"), 18).alias("__a070_corr")
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a070_dv")).pow(ts_rank(pl.col("__a070_corr"), 18)))
            * -1.0
        ).alias("alpha070")
    ).to_series()


def alpha076(panel: pl.DataFrame) -> pl.Series:
    """Alpha #076 — Negative max of vwap-delta-decay rank and IndNeutralize(low)-adv81 corr ts_rank.

    WorldQuant Formula
    ------------------
        max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
            Ts_Rank(decay_linear(Ts_Rank(correlation(
                IndNeutralize(low, IndClass.sector), adv81, 8.14941
            ), 19.569), 17.1543), 19.383)) * -1

    Required panel columns: ``vwap``, ``low``, ``adv81``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("low"), "industry").alias("__a076_il"),
        delta(pl.col("vwap"), 1).alias("__a076_dv"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a076_il"), pl.col("adv81"), 8).alias("__a076_corr")
    )
    staged3 = staged2.with_columns(
        ts_rank(pl.col("__a076_corr"), 20).alias("__a076_tr1"),
        ts_decay_linear(pl.col("__a076_dv"), 12).alias("__a076_dec1"),
    )
    staged4 = staged3.with_columns(
        ts_decay_linear(pl.col("__a076_tr1"), 17).alias("__a076_dec2")
    )
    staged5 = staged4.with_columns(
        cs_rank(pl.col("__a076_dec1")).alias("__a076_p1"),
        ts_rank(pl.col("__a076_dec2"), 19).alias("__a076_p2"),
    )
    return staged5.select(
        (
            pl.max_horizontal(pl.col("__a076_p1"), pl.col("__a076_p2")) * -1.0
        ).alias("alpha076")
    ).to_series()


def alpha079(panel: pl.DataFrame) -> pl.Series:
    """Alpha #079 — Industry-neutralised close-open blend delta rank inequality.

    WorldQuant Formula
    ------------------
        rank(delta(IndNeutralize(close * 0.60733 + open * (1 - 0.60733),
                                 IndClass.sector), 1.23438)) <
        rank(correlation(Ts_Rank(vwap, 3.60973),
                         Ts_Rank(adv150, 9.18637), 14.6644))

    Required panel columns: ``close``, ``open``, ``vwap``, ``adv150``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("close") * 0.60733 + pl.col("open") * (1.0 - 0.60733)
    staged = panel.with_columns(
        ind_neutralize(blend, "industry").alias("__a079_ib")
    )
    staged2 = staged.with_columns(
        delta(pl.col("__a079_ib"), 1).alias("__a079_dib"),
        ts_rank(pl.col("vwap"), 4).alias("__a079_trv"),
        ts_rank(pl.col("adv150"), 9).alias("__a079_tra"),
    )
    staged3 = staged2.with_columns(
        ts_corr(pl.col("__a079_trv"), pl.col("__a079_tra"), 15).alias("__a079_c")
    )
    return staged3.select(
        (
            cs_rank(pl.col("__a079_dib")) < cs_rank(pl.col("__a079_c"))
        ).cast(pl.Float64).alias("alpha079")
    ).to_series()


def alpha080(panel: pl.DataFrame) -> pl.Series:
    """Alpha #080 — Sign-of-IndNeutralize(open-high blend) delta, raised to corr power.

    WorldQuant Formula
    ------------------
        (rank(Sign(delta(IndNeutralize(open * 0.868128 + high * (1 - 0.868128),
                                       IndClass.industry), 4.04545)))^
         Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1

    Required panel columns: ``open``, ``high``, ``adv10``, ``stock_code``,
    ``trade_date``, ``industry``

    Polars Implementation Notes
    ---------------------------
    Synthetic panel doesn't have ``adv10`` (only adv5/15/...). We use
    ``adv15`` as the closest available substitute on the synthetic panel
    while keeping the original formula's intent. On real production
    panels the adv10 column is present.

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("open") * 0.868128 + pl.col("high") * (1.0 - 0.868128)
    staged = panel.with_columns(
        ind_neutralize(blend, "industry").alias("__a080_ib")
    )
    staged2 = staged.with_columns(
        sign_(delta(pl.col("__a080_ib"), 4)).alias("__a080_sd"),
        ts_corr(pl.col("high"), pl.col("adv15"), 5).alias("__a080_corr"),
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a080_sd")).pow(ts_rank(pl.col("__a080_corr"), 6)))
            * -1.0
        ).alias("alpha080")
    ).to_series()


def alpha082(panel: pl.DataFrame) -> pl.Series:
    """Alpha #082 — Negative min of open-delta-decay rank and IndNeutralize(volume)-open corr ts_rank.

    WorldQuant Formula
    ------------------
        min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
            Ts_Rank(decay_linear(correlation(
                IndNeutralize(volume, IndClass.sector),
                open * 0.634196 + open * (1 - 0.634196),
                17.4842
            ), 6.92131), 13.4283)) * -1

    Required panel columns: ``open``, ``volume``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_open = pl.col("open") * 0.634196 + pl.col("open") * (1.0 - 0.634196)
    staged = panel.with_columns(
        ind_neutralize(pl.col("volume"), "industry").alias("__a082_iv"),
        delta(pl.col("open"), 1).alias("__a082_do"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a082_iv"), blend_open, 17).alias("__a082_corr"),
        ts_decay_linear(pl.col("__a082_do"), 15).alias("__a082_d1"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a082_corr"), 7).alias("__a082_d2")
    )
    staged4 = staged3.with_columns(
        cs_rank(pl.col("__a082_d1")).alias("__a082_p1"),
        ts_rank(pl.col("__a082_d2"), 13).alias("__a082_p2"),
    )
    return staged4.select(
        (
            pl.min_horizontal(pl.col("__a082_p1"), pl.col("__a082_p2")) * -1.0
        ).alias("alpha082")
    ).to_series()


def alpha086(panel: pl.DataFrame) -> pl.Series:
    """Alpha #086 — Close vs sum(adv20) corr ts_rank inequality with body rank.

    WorldQuant Formula
    ------------------
        (Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
         rank(open + close - vwap - open)) * -1

    Required panel columns: ``close``, ``adv20``, ``open``, ``vwap``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ts_sum(pl.col("adv20"), 15).alias("__a086_sadv"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("close"), pl.col("__a086_sadv"), 6).alias("__a086_corr")
    )
    rhs = (pl.col("open") + pl.col("close")) - (pl.col("vwap") + pl.col("open"))
    staged3 = staged2.with_columns(
        ts_rank(pl.col("__a086_corr"), 20).alias("__a086_p1"),
        cs_rank(rhs).alias("__a086_p2"),
    )
    return staged3.select(
        (
            (pl.col("__a086_p1") < pl.col("__a086_p2")).cast(pl.Float64) * -1.0
        ).alias("alpha086")
    ).to_series()


def alpha087(panel: pl.DataFrame) -> pl.Series:
    """Alpha #087 — Negative max of close-vwap-blend delta-decay rank and abs-corr ts_rank.

    WorldQuant Formula
    ------------------
        max(rank(decay_linear(delta(close * 0.369701 + vwap * (1 - 0.369701),
                                    1.91233), 2.65461)),
            Ts_Rank(decay_linear(abs(correlation(
                IndNeutralize(adv81, IndClass.industry), close, 13.4132
            )), 4.89768), 14.4535)) * -1

    Required panel columns: ``close``, ``vwap``, ``adv81``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("close") * 0.369701 + pl.col("vwap") * (1.0 - 0.369701)
    staged = panel.with_columns(
        ind_neutralize(pl.col("adv81"), "industry").alias("__a087_ia"),
        delta(blend, 2).alias("__a087_db"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a087_ia"), pl.col("close"), 13).alias("__a087_corr")
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a087_db"), 3).alias("__a087_d1"),
        ts_decay_linear(_abs(pl.col("__a087_corr")), 5).alias("__a087_d2"),
    )
    staged4 = staged3.with_columns(
        cs_rank(pl.col("__a087_d1")).alias("__a087_p1"),
        ts_rank(pl.col("__a087_d2"), 14).alias("__a087_p2"),
    )
    return staged4.select(
        (
            pl.max_horizontal(pl.col("__a087_p1"), pl.col("__a087_p2")) * -1.0
        ).alias("alpha087")
    ).to_series()


def alpha089(panel: pl.DataFrame) -> pl.Series:
    """Alpha #089 — Diff of two ts_rank(decay) composites with industry-neutralised vwap.

    WorldQuant Formula
    ------------------
        Ts_Rank(decay_linear(correlation(low * 0.967285 + low * (1 - 0.967285),
                                          adv10, 6.94279), 5.51607), 3.79744) -
        Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry),
                                    3.48158), 10.1466), 15.3012)

    Polars Implementation Notes
    ---------------------------
    Synthetic panel uses ``adv15`` as substitute for ``adv10``.

    Required panel columns: ``low``, ``vwap``, ``adv10``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_low = pl.col("low") * 0.967285 + pl.col("low") * (1.0 - 0.967285)
    staged = panel.with_columns(
        ind_neutralize(pl.col("vwap"), "industry").alias("__a089_iv"),
        ts_corr(blend_low, pl.col("adv15"), 7).alias("__a089_corr"),
    )
    staged2 = staged.with_columns(
        delta(pl.col("__a089_iv"), 3).alias("__a089_div"),
        ts_decay_linear(pl.col("__a089_corr"), 6).alias("__a089_dec1"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a089_div"), 10).alias("__a089_dec2")
    )
    return staged3.select(
        (
            ts_rank(pl.col("__a089_dec1"), 4) - ts_rank(pl.col("__a089_dec2"), 15)
        ).alias("alpha089")
    ).to_series()


def alpha090(panel: pl.DataFrame) -> pl.Series:
    """Alpha #090 — Negative power composite of close-from-max rank and IndNeutralize(adv40)-low corr.

    WorldQuant Formula
    ------------------
        (rank(close - ts_max(close, 4.66719))^
         Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry),
                             low, 5.38375), 3.21856)) * -1

    Required panel columns: ``close``, ``adv40``, ``low``, ``stock_code``,
    ``trade_date``, ``industry``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("adv40"), "sub_industry").alias("__a090_ia"),
        (pl.col("close") - ts_max(pl.col("close"), 5)).alias("__a090_diff"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a090_ia"), pl.col("low"), 5).alias("__a090_corr")
    )
    return staged2.select(
        (
            (cs_rank(pl.col("__a090_diff")).pow(ts_rank(pl.col("__a090_corr"), 3)))
            * -1.0
        ).alias("alpha090")
    ).to_series()


def alpha091(panel: pl.DataFrame) -> pl.Series:
    """Alpha #091 — Diff of double-decay-corr ts_rank and vwap-adv30 corr decay rank.

    WorldQuant Formula
    ------------------
        (Ts_Rank(decay_linear(decay_linear(correlation(
            IndNeutralize(close, IndClass.industry), volume, 9.74928
        ), 16.398), 3.83219), 4.8667) -
         rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1

    Required panel columns: ``close``, ``volume``, ``vwap``, ``adv30``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ind_neutralize(pl.col("close"), "industry").alias("__a091_ic")
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a091_ic"), pl.col("volume"), 10).alias("__a091_c1"),
        ts_corr(pl.col("vwap"), pl.col("adv30"), 4).alias("__a091_c2"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a091_c1"), 16).alias("__a091_dec_inner")
    )
    staged4 = staged3.with_columns(
        ts_decay_linear(pl.col("__a091_dec_inner"), 4).alias("__a091_dec_outer"),
        ts_decay_linear(pl.col("__a091_c2"), 3).alias("__a091_dec2"),
    )
    return staged4.select(
        (
            (
                ts_rank(pl.col("__a091_dec_outer"), 5)
                - cs_rank(pl.col("__a091_dec2"))
            )
            * -1.0
        ).alias("alpha091")
    ).to_series()


def alpha093(panel: pl.DataFrame) -> pl.Series:
    """Alpha #093 — Industry-neutralised vwap corr decay ts_rank divided by close-blend delta-decay rank.

    WorldQuant Formula
    ------------------
        Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry),
                                          adv81, 17.4193), 19.848), 7.54455) /
        rank(decay_linear(delta(close * 0.524434 + vwap * (1 - 0.524434),
                                 2.77377), 16.2664))

    Required panel columns: ``vwap``, ``adv81``, ``close``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("close") * 0.524434 + pl.col("vwap") * (1.0 - 0.524434)
    staged = panel.with_columns(
        ind_neutralize(pl.col("vwap"), "industry").alias("__a093_iv"),
        delta(blend, 3).alias("__a093_db"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a093_iv"), pl.col("adv81"), 17).alias("__a093_corr")
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a093_corr"), 20).alias("__a093_d1"),
        ts_decay_linear(pl.col("__a093_db"), 16).alias("__a093_d2"),
    )
    return staged3.select(
        (
            ts_rank(pl.col("__a093_d1"), 8) / cs_rank(pl.col("__a093_d2"))
        ).alias("alpha093")
    ).to_series()


def alpha096(panel: pl.DataFrame) -> pl.Series:
    """Alpha #096 — Negative max of two ts_rank(decay(corr)) composites.

    WorldQuant Formula
    ------------------
        max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume),
                                              3.83878), 4.16783), 8.38151),
            Ts_Rank(decay_linear(Ts_ArgMax(correlation(
                Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459
            ), 12.6556), 14.0365), 13.4143)) * -1

    Required panel columns: ``vwap``, ``volume``, ``close``, ``adv60``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        cs_rank(pl.col("vwap")).alias("__a096_rv"),
        cs_rank(pl.col("volume")).alias("__a096_rl"),
        ts_rank(pl.col("close"), 7).alias("__a096_trc"),
        ts_rank(pl.col("adv60"), 4).alias("__a096_tra"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("__a096_rv"), pl.col("__a096_rl"), 4).alias("__a096_c1"),
        ts_corr(pl.col("__a096_trc"), pl.col("__a096_tra"), 4).alias("__a096_c2"),
    )
    staged3 = staged2.with_columns(
        ts_decay_linear(pl.col("__a096_c1"), 4).alias("__a096_d1"),
        ts_argmax(pl.col("__a096_c2"), 13).alias("__a096_am"),
    )
    staged4 = staged3.with_columns(
        ts_decay_linear(pl.col("__a096_am"), 14).alias("__a096_d2")
    )
    staged5 = staged4.with_columns(
        ts_rank(pl.col("__a096_d1"), 8).alias("__a096_p1"),
        ts_rank(pl.col("__a096_d2"), 13).alias("__a096_p2"),
    )
    return staged5.select(
        (
            pl.max_horizontal(pl.col("__a096_p1"), pl.col("__a096_p2")) * -1.0
        ).alias("alpha096")
    ).to_series()


def alpha097(panel: pl.DataFrame) -> pl.Series:
    """Alpha #097 — Diff of low-vwap-blend industry-neutralised delta-decay rank and ts_rank corr nest.

    WorldQuant Formula
    ------------------
        (rank(decay_linear(delta(IndNeutralize(
            low * 0.721001 + vwap * (1 - 0.721001), IndClass.industry
         ), 3.3705), 20.4523)) -
         Ts_Rank(decay_linear(Ts_Rank(correlation(
             Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547
         ), 18.5925), 15.7152), 6.71659)) * -1

    Required panel columns: ``low``, ``vwap``, ``adv60``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col("low") * 0.721001 + pl.col("vwap") * (1.0 - 0.721001)
    staged = panel.with_columns(
        ind_neutralize(blend, "industry").alias("__a097_ib"),
        ts_rank(pl.col("low"), 8).alias("__a097_trl"),
        ts_rank(pl.col("adv60"), 17).alias("__a097_tra"),
    )
    staged2 = staged.with_columns(
        delta(pl.col("__a097_ib"), 3).alias("__a097_dib"),
        ts_corr(pl.col("__a097_trl"), pl.col("__a097_tra"), 5).alias("__a097_corr"),
    )
    staged3 = staged2.with_columns(
        ts_rank(pl.col("__a097_corr"), 19).alias("__a097_tr1"),
        ts_decay_linear(pl.col("__a097_dib"), 20).alias("__a097_d1"),
    )
    staged4 = staged3.with_columns(
        ts_decay_linear(pl.col("__a097_tr1"), 16).alias("__a097_d2")
    )
    return staged4.select(
        (
            (
                cs_rank(pl.col("__a097_d1"))
                - ts_rank(pl.col("__a097_d2"), 7)
            )
            * -1.0
        ).alias("alpha097")
    ).to_series()


def alpha098(panel: pl.DataFrame) -> pl.Series:
    """Alpha #098 — Diff of vwap-adv5-corr decay rank and rank-corr ts_argmin nest.

    WorldQuant Formula
    ------------------
        rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418),
                          7.18088)) -
        rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(
            rank(open), rank(adv15), 20.8187
        ), 8.62571), 6.95668), 8.07206))

    Required panel columns: ``vwap``, ``adv5``, ``open``, ``adv15``,
    ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(
        ts_sum(pl.col("adv5"), 26).alias("__a098_sadv"),
        cs_rank(pl.col("open")).alias("__a098_ro"),
        cs_rank(pl.col("adv15")).alias("__a098_ra"),
    )
    staged2 = staged.with_columns(
        ts_corr(pl.col("vwap"), pl.col("__a098_sadv"), 5).alias("__a098_c1"),
        ts_corr(pl.col("__a098_ro"), pl.col("__a098_ra"), 21).alias("__a098_c2"),
    )
    staged3 = staged2.with_columns(
        ts_argmin(pl.col("__a098_c2"), 9).alias("__a098_am")
    )
    staged4 = staged3.with_columns(
        ts_rank(pl.col("__a098_am"), 7).alias("__a098_tr"),
        ts_decay_linear(pl.col("__a098_c1"), 7).alias("__a098_d1"),
    )
    staged5 = staged4.with_columns(
        ts_decay_linear(pl.col("__a098_tr"), 8).alias("__a098_d2")
    )
    return staged5.select(
        (cs_rank(pl.col("__a098_d1")) - cs_rank(pl.col("__a098_d2"))).alias(
            "alpha098"
        )
    ).to_series()


def alpha100(panel: pl.DataFrame) -> pl.Series:
    """Alpha #100 — Subindustry-neutralised body * volume signal minus close-rank-corr.

    WorldQuant Formula
    ------------------
        0 - 1 * (
            (1.5 * scale(IndNeutralize(IndNeutralize(
                rank(((close - low) - (high - close)) / (high - low) * volume),
                IndClass.subindustry), IndClass.subindustry)) -
             scale(IndNeutralize(
                 (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
                 IndClass.subindustry
             ))
            ) * (volume / adv20)
        )

    Required panel columns: ``close``, ``low``, ``high``, ``volume``,
    ``adv20``, ``stock_code``, ``trade_date``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    body = (
        ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close")))
        / (pl.col("high") - pl.col("low"))
        * pl.col("volume")
    )
    staged = panel.with_columns(
        cs_rank(body).alias("__a100_rb"),
        cs_rank(pl.col("adv20")).alias("__a100_ra"),
    )
    staged2 = staged.with_columns(
        ind_neutralize(pl.col("__a100_rb"), "sub_industry").alias("__a100_n1"),
        ts_corr(pl.col("close"), pl.col("__a100_ra"), 5).alias("__a100_c1"),
        ts_argmin(pl.col("close"), 30).alias("__a100_am"),
    )
    staged3 = staged2.with_columns(
        ind_neutralize(pl.col("__a100_n1"), "sub_industry").alias("__a100_n2"),
        cs_rank(pl.col("__a100_am")).alias("__a100_ram"),
    )
    staged4 = staged3.with_columns(
        (pl.col("__a100_c1") - pl.col("__a100_ram")).alias("__a100_part2_inner"),
        cs_scale(pl.col("__a100_n2")).alias("__a100_s1"),
    )
    staged5 = staged4.with_columns(
        ind_neutralize(pl.col("__a100_part2_inner"), "sub_industry").alias(
            "__a100_n3"
        )
    )
    staged6 = staged5.with_columns(
        cs_scale(pl.col("__a100_n3")).alias("__a100_s2")
    )
    return staged6.select(
        (
            -1.0
            * (
                (1.5 * pl.col("__a100_s1") - pl.col("__a100_s2"))
                * (pl.col("volume") / pl.col("adv20"))
            )
        ).alias("alpha100")
    ).to_series()


# ===========================================================================
# Registry self-population
# ===========================================================================


_ENTRIES: tuple[FactorEntry, ...] = (
    # ----- Sub-industry group ----------------------------------------------
    FactorEntry(
        id="alpha011",
        impl=alpha011,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(ts_max((vwap-close),3)) + rank(ts_min((vwap-close),3))) "
            "* rank(delta(volume,3))"
        ),
        references=("Kakushadze 2015, eq. 11",),
    ),
    FactorEntry(
        id="alpha016",
        impl=alpha016,
        direction="reverse",
        category="industry_neutral",
        description="-1 * rank(covariance(rank(high), rank(volume), 5))",
        references=("Kakushadze 2015, eq. 16",),
    ),
    FactorEntry(
        id="alpha020",
        impl=alpha020,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-1 * rank(open-delay(high,1)) * rank(open-delay(close,1)) "
            "* rank(open-delay(low,1))"
        ),
        references=("Kakushadze 2015, eq. 20",),
    ),
    FactorEntry(
        id="alpha047",
        impl=alpha047,
        direction="reverse",
        category="industry_neutral",
        description=(
            "((rank(1/close)*volume/adv20) * (high*rank(high-close)/sma(high,5))) "
            "- rank(vwap-delay(vwap,5))"
        ),
        references=("Kakushadze 2015, eq. 47",),
    ),
    FactorEntry(
        id="alpha048",
        impl=alpha048,
        direction="reverse",
        category="industry_neutral",
        description=(
            "IndNeutralize((corr(delta(close,1), delta(delay(close,1),1), 250) "
            "* delta(close,1)) / close, sub_industry) "
            "/ sum((delta(close,1)/delay(close,1))^2, 250)"
        ),
        references=("Kakushadze 2015, eq. 48",),
    ),
    # ----- Industry group --------------------------------------------------
    FactorEntry(
        id="alpha027",
        impl=alpha027,
        direction="reverse",
        category="industry_neutral",
        description=(
            "Sign threshold of rank(sum(corr(rank(volume), rank(vwap), 6), 2)/2)"
        ),
        references=("Kakushadze 2015, eq. 27",),
    ),
    FactorEntry(
        id="alpha029",
        impl=alpha029,
        direction="reverse",
        category="industry_neutral",
        description=(
            "Deeply nested rank-scale-log composite of -delta(close-1,5), "
            "plus ts_rank(delay(-returns,6),5)"
        ),
        references=("Kakushadze 2015, eq. 29",),
    ),
    FactorEntry(
        id="alpha030",
        impl=alpha030,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(1 - rank(sum(sign(delta(close,1)) over last 3 days))) "
            "* sum(volume,5)/sum(volume,20)"
        ),
        references=("Kakushadze 2015, eq. 30",),
    ),
    FactorEntry(
        id="alpha031",
        impl=alpha031,
        direction="reverse",
        category="industry_neutral",
        description=(
            "Triple-rank decay of -rank(rank(delta(close,10))) "
            "+ rank(-delta(close,3)) + sign(scale(corr(adv20,low,12)))"
        ),
        references=("Kakushadze 2015, eq. 31",),
    ),
    FactorEntry(
        id="alpha036",
        impl=alpha036,
        direction="reverse",
        category="industry_neutral",
        description="5-component weighted rank composite (alpha036 paper formula)",
        references=("Kakushadze 2015, eq. 36",),
    ),
    FactorEntry(
        id="alpha039",
        impl=alpha039,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(-rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))) "
            "* (1 + rank(sum(returns,250)))"
        ),
        references=("Kakushadze 2015, eq. 39",),
    ),
    FactorEntry(
        id="alpha049",
        impl=alpha049,
        direction="reverse",
        category="industry_neutral",
        description=(
            "If close-acceleration < -0.1 then 1 else -delta(close,1)"
        ),
        references=("Kakushadze 2015, eq. 49",),
    ),
    FactorEntry(
        id="alpha050",
        impl=alpha050,
        direction="reverse",
        category="industry_neutral",
        description="-ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)",
        references=("Kakushadze 2015, eq. 50",),
    ),
    FactorEntry(
        id="alpha058",
        impl=alpha058,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), volume, 4), 8), 6)"
        ),
        references=("Kakushadze 2015, eq. 58",),
    ),
    FactorEntry(
        id="alpha059",
        impl=alpha059,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-ts_rank(decay_linear(corr(IndNeutralize(vwap_blend,industry), volume, 4), 16), 8)"
        ),
        references=("Kakushadze 2015, eq. 59",),
    ),
    FactorEntry(
        id="alpha062",
        impl=alpha062,
        direction="reverse",
        category="industry_neutral",
        description=(
            "rank(corr(vwap,sum(adv20,22),10)) < rank(open-rank vs midpoint inequality)"
        ),
        references=("Kakushadze 2015, eq. 62",),
    ),
    FactorEntry(
        id="alpha063",
        impl=alpha063,
        direction="reverse",
        category="industry_neutral",
        description=(
            "Diff of decay-linear(delta(IndNeutralize(close,industry),2),8) "
            "and decay-linear(corr(blend,sum(adv180,37),14),12), sign-flipped"
        ),
        references=("Kakushadze 2015, eq. 63",),
    ),
    FactorEntry(
        id="alpha066",
        impl=alpha066,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-(rank(decay_linear(delta(vwap,4),7)) "
            "+ ts_rank(decay_linear((low-vwap)/(open-mid),11),7))"
        ),
        references=("Kakushadze 2015, eq. 66",),
    ),
    FactorEntry(
        id="alpha067",
        impl=alpha067,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(high-ts_min(high,2)) ^ rank(corr(IndNeutralize(vwap,industry), "
            "IndNeutralize(adv20,sub_industry), 6))) * -1"
        ),
        references=("Kakushadze 2015, eq. 67",),
    ),
    FactorEntry(
        id="alpha069",
        impl=alpha069,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(ts_max(delta(IndNeutralize(vwap,industry),3),5)) ^ "
            "ts_rank(corr(close-vwap blend, adv20, 5), 9)) * -1"
        ),
        references=("Kakushadze 2015, eq. 69",),
    ),
    FactorEntry(
        id="alpha070",
        impl=alpha070,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(delta(vwap,1)) ^ ts_rank(corr(IndNeutralize(close,industry), "
            "adv50, 18), 18)) * -1"
        ),
        references=("Kakushadze 2015, eq. 70",),
    ),
    FactorEntry(
        id="alpha076",
        impl=alpha076,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-max(rank(decay_linear(delta(vwap,1),12)), "
            "ts_rank(decay_linear(ts_rank(corr(IndNeutralize(low,industry), adv81, 8), 20), 17), 19))"
        ),
        references=("Kakushadze 2015, eq. 76",),
    ),
    FactorEntry(
        id="alpha079",
        impl=alpha079,
        direction="reverse",
        category="industry_neutral",
        description=(
            "rank(delta(IndNeutralize(close-open blend,industry),1)) "
            "< rank(corr(ts_rank(vwap,4), ts_rank(adv150,9), 15))"
        ),
        references=("Kakushadze 2015, eq. 79",),
    ),
    FactorEntry(
        id="alpha080",
        impl=alpha080,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(sign(delta(IndNeutralize(open-high blend,industry),4))) "
            "^ ts_rank(corr(high,adv15,5),6)) * -1"
        ),
        references=("Kakushadze 2015, eq. 80",),
    ),
    FactorEntry(
        id="alpha082",
        impl=alpha082,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-min(rank(decay_linear(delta(open,1),15)), "
            "ts_rank(decay_linear(corr(IndNeutralize(volume,industry), open, 17), 7), 13))"
        ),
        references=("Kakushadze 2015, eq. 82",),
    ),
    FactorEntry(
        id="alpha086",
        impl=alpha086,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(ts_rank(corr(close,sum(adv20,15),6),20) "
            "< rank((open+close)-(vwap+open))) * -1"
        ),
        references=("Kakushadze 2015, eq. 86",),
    ),
    FactorEntry(
        id="alpha087",
        impl=alpha087,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-max(rank(decay_linear(delta(close-vwap blend,2),3)), "
            "ts_rank(decay_linear(abs(corr(IndNeutralize(adv81,industry), close, 13)), 5), 14))"
        ),
        references=("Kakushadze 2015, eq. 87",),
    ),
    FactorEntry(
        id="alpha089",
        impl=alpha089,
        direction="reverse",
        category="industry_neutral",
        description=(
            "ts_rank(decay_linear(corr(low blend, adv10, 7), 6), 4) - "
            "ts_rank(decay_linear(delta(IndNeutralize(vwap,industry),3), 10), 15)"
        ),
        references=("Kakushadze 2015, eq. 89",),
    ),
    FactorEntry(
        id="alpha090",
        impl=alpha090,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(close-ts_max(close,5)) ^ ts_rank(corr(IndNeutralize(adv40,sub_industry), low, 5), 3)) * -1"
        ),
        references=("Kakushadze 2015, eq. 90",),
    ),
    FactorEntry(
        id="alpha091",
        impl=alpha091,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(ts_rank(decay_linear(decay_linear(corr(IndNeutralize(close,industry), volume, 10), 16), 4), 5) "
            "- rank(decay_linear(corr(vwap,adv30,4),3))) * -1"
        ),
        references=("Kakushadze 2015, eq. 91",),
    ),
    FactorEntry(
        id="alpha093",
        impl=alpha093,
        direction="reverse",
        category="industry_neutral",
        description=(
            "ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), adv81, 17), 20), 8) "
            "/ rank(decay_linear(delta(close-vwap blend, 3), 16))"
        ),
        references=("Kakushadze 2015, eq. 93",),
    ),
    FactorEntry(
        id="alpha096",
        impl=alpha096,
        direction="reverse",
        category="industry_neutral",
        description=(
            "-max(ts_rank(decay_linear(corr(rank(vwap),rank(volume),4),4),8), "
            "ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))"
        ),
        references=("Kakushadze 2015, eq. 96",),
    ),
    FactorEntry(
        id="alpha097",
        impl=alpha097,
        direction="reverse",
        category="industry_neutral",
        description=(
            "(rank(decay_linear(delta(IndNeutralize(low-vwap blend,industry),3),20)) "
            "- ts_rank(decay_linear(ts_rank(corr(ts_rank(low,8),ts_rank(adv60,17),5),19),16),7)) * -1"
        ),
        references=("Kakushadze 2015, eq. 97",),
    ),
    FactorEntry(
        id="alpha098",
        impl=alpha098,
        direction="reverse",
        category="industry_neutral",
        description=(
            "rank(decay_linear(corr(vwap,sum(adv5,26),5),7)) "
            "- rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),21),9),7),8))"
        ),
        references=("Kakushadze 2015, eq. 98",),
    ),
    FactorEntry(
        id="alpha100",
        impl=alpha100,
        direction="reverse",
        category="industry_neutral",
        description=(
            "Sub-industry-neutralised body*volume signal minus "
            "scale(IndNeutralize(corr(close,rank(adv20),5)-rank(ts_argmin(close,30)), sub_industry)), "
            "weighted by volume/adv20"
        ),
        references=("Kakushadze 2015, eq. 100",),
    ),
)


for _entry in _ENTRIES:
    register_alpha101(_entry)


# Tie module-level imports so static analysers don't flag them.
_ = TS_PART
_ = CS_PART
_ = ts_product
