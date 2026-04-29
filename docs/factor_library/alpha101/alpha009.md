# alpha009 — Trend-confirmed price-change momentum

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
     ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1),
       If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1),
          -1 * Delta(close, 1)))

```
If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1), If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1), -1 * Delta(close, 1)))
```

## Polars Implementation Notes

1. If the past 5-day minimum daily change is positive (consistent up
   trend), pass-through the daily delta.
2. Else if the past 5-day maximum daily change is negative (consistent
   down trend), still pass-through the daily delta.
3. Otherwise (mixed regime) flip sign of the daily delta — i.e. mean
   reversion within choppy markets.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 9
