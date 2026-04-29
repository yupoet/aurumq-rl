# alpha050 — -ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)

Required panel columns: ``volume``, ``vwap``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

volume``, ``vwap``, ``stock_code``,

## References

- Kakushadze 2015, eq. 50
