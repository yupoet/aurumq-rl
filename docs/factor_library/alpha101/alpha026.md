# alpha026 — Max recent volume-rank vs high-rank correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)

```
-1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)
```

## Polars Implementation Notes

Pure TS chain. Stage the two TS ranks before the corr.

Required panel columns: ``volume``, ``high``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

volume``, ``high``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 26
