# alpha044 — Negative high-vs-volume-rank correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * correlation(high, rank(volume), 5)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Corr(high, Rank(volume), 5)

```
-1 * Ts_Corr(high, Rank(volume), 5)
```

## Polars Implementation Notes

Stage the CS rank of volume before the TS corr.

Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 44
