# alpha083 — Range/MA delay rank times volume rank-squared, scaled

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(delay((high - low) / (sum(close, 5) / 5), 2)) * rank(rank(volume)))
    / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * Rank(Rank(volume)))
    / (((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))

```
(Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * Rank(Rank(volume))) / (((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))
```

## Polars Implementation Notes

Stage the range-over-MA series, its delay, and the double-CS-rank of
volume; final division involves the present-day ratio and price gap.

Required panel columns: ``high``, ``low``, ``close``, ``volume``, ``vwap``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``low``, ``close``, ``volume``, ``vwap``,

## References

- Kakushadze 2015, eq. 83
