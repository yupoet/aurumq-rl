# alpha_custom_argmin_recent — Cross-section rank of 20-day Ts_ArgMin of close

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(Ts_ArgMin(close, 20))

```
Rank(Ts_ArgMin(close, 20))
```

## Polars Implementation Notes

``ts_argmin(close, 20)`` returns the index of the recent 20-day low.
A larger value (= more recent low) ranks higher; the resulting CS rank
flags stocks where the recent bottom is fresh — a setup for mean-revert
bounce trades.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``stock_code``, ``trade_date

## References

_(not specified)_
