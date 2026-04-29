# alpha023 — High-breakout-conditional negative 2d high change

> **Category**: breakout | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)

```
If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)
```

## Polars Implementation Notes

1. The condition flags any day whose ``high`` exceeds the 20-day
   average — a breakout candidate.
2. On such days the alpha equals the **negative** 2-day change of
   ``high``: rising momentum -> negative signal, falling -> positive.
   On non-breakout days the alpha is zero (no opinion).

Required panel columns: ``high``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``breakout``

## Required Panel Columns

high``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 23
