# alpha046 — Trend-shape conditional one-day reversal

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
              ((delay(close, 10) - close) / 10))) ? -1 :
     ((((delay(close, 20) - delay(close, 10)) / 10) -
       ((delay(close, 10) - close) / 10)) < 0) ? 1 :
     (-1 * (close - delay(close, 1))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If((Delay(close, 20) - Delay(close, 10)) / 10 -
       (Delay(close, 10) - close) / 10 > 0.25, -1,
       If((Delay(close, 20) - Delay(close, 10)) / 10 -
          (Delay(close, 10) - close) / 10 < 0, 1,
          -1 * (close - Delay(close, 1))))

```
If((Delay(close, 20) - Delay(close, 10)) / 10 - (Delay(close, 10) - close) / 10 > 0.25, -1, If((Delay(close, 20) - Delay(close, 10)) / 10 - (Delay(close, 10) - close) / 10 < 0, 1, -1 * (close - Delay(close, 1))))
```

## Polars Implementation Notes

1. Compute the ``trend curvature`` once and reuse the materialised
   column twice in the nested ``if``.
2. Three branches: strong upward curvature -> -1; downward curvature
   -> 1; otherwise reverse the daily change.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 46
