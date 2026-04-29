# alpha015 — Negative 3d sum of ranked high-volume rank correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)

```
-1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)
```

## Polars Implementation Notes

Two CS ranks are materialised before the 3-day TS correlation. STHSF
fills NaN/inf correlation rows with zero before the outer rank; we do
the same so the early-window rank/sum semantics match the reference.

Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 15
