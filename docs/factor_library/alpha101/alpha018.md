# alpha018 — Negative CS rank of: 5-day std(|close-open|) + (close-open) + 10-day correlation(close, open)

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank((stddev(abs((close - open)), 5) +
               (close - open)) +
              correlation(close, open, 10))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Std(Abs(close - open), 5) +
              (close - open) +
              Ts_Corr(close, open, 10))

```
-1 * Rank(Ts_Std(Abs(close - open), 5) + (close - open) + Ts_Corr(close, open, 10))
```

## Polars Implementation Notes

1. ``Ts_Std(Abs(close - open), 5)``: 5-day std of body magnitude.
2. Plus today's body ``close - open``.
3. Plus 10-day rolling correlation between close and open (NaN/inf
   replaced with null then handled by CS rank's nulls treatment).
4. CS rank, sign-flipped.

Required panel columns: ``close``, ``open``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

close``, ``open``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 18
