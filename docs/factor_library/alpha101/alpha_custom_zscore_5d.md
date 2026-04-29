# alpha_custom_zscore_5d — Negative 5-day rolling z-score of close

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Zscore(close, 5)

```
-1 * Ts_Zscore(close, 5)
```

## Polars Implementation Notes

A simple short-horizon mean-reversion factor: positive values indicate
close is well below its 5-day mean (after sign flip), suggesting a
contrarian buy.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``stock_code``, ``trade_date

## References

_(not specified)_
