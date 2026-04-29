# alpha004 — Negative 9-day Ts_Rank of cross-section rank of low

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * Ts_Rank(rank(low), 9)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Rank(Rank(low), 9)

```
-1 * Ts_Rank(Rank(low), 9)
```

## Polars Implementation Notes

1. ``Rank(low)`` is a per-day cross-section pct rank.
2. ``Ts_Rank(..., 9)`` is the 9-day rolling pct rank (last value within
   window) of that ranked column, per stock.
3. Final ``-1 *`` flips the sign so high values indicate "low has been
   persistently low" — a contrarian / mean-revert signal.

Required panel columns: ``low``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

low``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 4
