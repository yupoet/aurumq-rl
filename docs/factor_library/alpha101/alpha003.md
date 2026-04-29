# alpha003 — Open price rank vs volume rank correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * correlation(rank(open), rank(volume), 10)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Corr(Rank(open), Rank(volume), 10)

```
-1 * Ts_Corr(Rank(open), Rank(volume), 10)
```

## Polars Implementation Notes

Two CS ranks materialised before TS corr.

Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

open``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 3
