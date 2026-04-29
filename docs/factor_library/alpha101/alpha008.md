# alpha008 — Acceleration of (open*returns) sum compared to 10 days ago

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))

```
-1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))
```

## Polars Implementation Notes

1. Build the per-stock 5-day rolling sum of ``open`` and ``returns``.
2. Multiply pointwise -> momentum proxy.
3. Subtract the 10-day-ago version -> acceleration.
4. Materialise before ``cs_rank`` (CS partition differs from TS).

Required panel columns: ``open``, ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

open``, ``returns``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 8
