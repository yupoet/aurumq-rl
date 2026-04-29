# alpha040 — Negative rank(stddev(high,10)) * correlation(high, volume, 10)

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)

```
-1 * Rank(Ts_Std(high, 10)) * Ts_Corr(high, volume, 10)
```

## Polars Implementation Notes

1. CS rank of 10-day std of high prices (volatility regime).
2. 10-day rolling correlation between high and volume (price-volume
   confirmation).
3. Multiply with sign flip: large vol + positive corr ⇒ persistent
   breakout that mean-reverts.

Required panel columns: ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

high``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 40
