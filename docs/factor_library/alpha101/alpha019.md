# alpha019 — 7d-return sign times annual rank-mom multiplier

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
     (1 + rank((1 + sum(returns, 250)))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) *
    (1 + Rank(1 + Ts_Sum(returns, 250)))

```
(-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) * (1 + Rank(1 + Ts_Sum(returns, 250)))
```

## Polars Implementation Notes

1. ``close - delay(close, 7)`` and ``delta(close, 7)`` are
   algebraically identical; the WorldQuant paper writes both for
   robustness. We follow the formula verbatim — adding the same
   quantity twice doubles the magnitude but keeps the sign.
2. The annual ``ts_sum(returns, 250)`` rank-multiplier requires a
   250-day window that is longer than the typical synthetic panel
   (60 days), so the output is null for the first ~250 rows of each
   stock. STHSF reference is computed on the same panel and shows
   the same null pattern.

Required panel columns: ``close``, ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``returns``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 19
