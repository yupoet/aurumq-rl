# alpha051 — Trend curvature conditional reversal

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(((((delay(close, 20) - delay(close, 10)) / 10) -
       ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 :
     (-1 * (close - delay(close, 1))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If((Delay(close, 20) - Delay(close, 10)) / 10 -
       (Delay(close, 10) - close) / 10 < -0.05, 1,
       -1 * (close - Delay(close, 1)))

```
If((Delay(close, 20) - Delay(close, 10)) / 10 - (Delay(close, 10) - close) / 10 < -0.05, 1, -1 * (close - Delay(close, 1)))
```

## Polars Implementation Notes

1. Same curvature definition as :func:`alpha046` but with a single
   conditional: ``curvature < -0.05`` -> +1, else daily-reversal.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 51
