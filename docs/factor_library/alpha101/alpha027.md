# alpha027 — Sign threshold of rank(sum(corr(rank(volume), rank(vwap), 6), 2)/2)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
     ? -1 : 1)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Two-stage CS rank → TS corr → TS sum → CS rank, then threshold at 0.5.
The classic STHSF implementation has a discontinuity at 0.5; we
follow the same convention.

Required panel columns: ``volume``, ``vwap``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

volume``, ``vwap``, ``stock_code``,

## References

- Kakushadze 2015, eq. 27
