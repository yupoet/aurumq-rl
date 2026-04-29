# alpha059 — -ts_rank(decay_linear(corr(IndNeutralize(vwap_blend,industry), volume, 4), 16), 8)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * Ts_Rank(decay_linear(
        correlation(IndNeutralize(
            vwap * 0.728317 + vwap * (1 - 0.728317),
            IndClass.industry
        ), volume, 4.25197), 16.2289
    ), 8.19648)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

The convex blend is degenerate (always equal to ``vwap``); we keep
the literal formula. Windows truncated: 4, 16, 8.

Required panel columns: ``vwap``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

vwap``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 59
