# alpha032 — Scaled 7-day MA divergence + 20x scaled 230-day correlation between vwap and delayed close

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

scale((sum(close, 7) / 7 - close)) +
        20 * scale(correlation(vwap, delay(close, 5), 230))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Scale(Ts_Sum(close, 7) / 7 - close)
        + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))

```
Scale(Ts_Sum(close, 7) / 7 - close) + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))
```

## Polars Implementation Notes

1. Two cross-section scaled terms summed; both ``scale`` calls normalise
   ``sum(|x|) == 1`` per trade_date, matching STHSF.
2. The 230-day correlation is heavy — most synthetic panel rows will be
   NaN. That's fine for parity (reference is also all-NaN here).

Required panel columns: ``close``, ``vwap``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``vwap``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 32
