# alpha037 — Rank of 200-day correlation between delayed (open-close) and close, plus rank of (open-close)

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(correlation(delay((open - close), 1), close, 200)) +
        rank((open - close))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(Ts_Corr(Delay(open - close, 1), close, 200))
        + Rank(open - close)

```
Rank(Ts_Corr(Delay(open - close, 1), close, 200)) + Rank(open - close)
```

## Polars Implementation Notes

The 200-day window will yield NaN on the synthetic panel (60 days). The
second term still produces values immediately. Both ranks are CS pct.

Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

open``, ``close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 37
