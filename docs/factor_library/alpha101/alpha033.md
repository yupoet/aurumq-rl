# alpha033 — Cross-section rank of (open/close - 1)

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank((-1 * ((1 - (open / close))^1)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(-1 * Power(1 - (open / close), 1))

```
Rank(-1 * Power(1 - (open / close), 1))
```

## Polars Implementation Notes

Algebraic simplification used by STHSF: ``-1 * (1 - open/close)``
collapses to ``open/close - 1``. Cross-section pct rank only.

Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

open``, ``close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 33
