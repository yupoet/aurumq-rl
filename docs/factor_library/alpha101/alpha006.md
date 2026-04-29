# alpha006 — 10d correlation between open and volume

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * correlation(open, volume, 10)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Corr(open, volume, 10)

```
-1 * Ts_Corr(open, volume, 10)
```

## Polars Implementation Notes

Pure TS — no CS staging needed.

Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

open``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 6
