# alpha010 — Cross-sectional rank of trend-confirmed price change

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
          ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1),
           If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1),
              -1 * Delta(close, 1))))

```
Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1), If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1), -1 * Delta(close, 1))))
```

## Polars Implementation Notes

Same logic as :func:`alpha009` but with a 4-day lookback for the
trend confirmation, then cross-section ranked. Materialise before
ranking so that the TS lookback finishes before the CS partition.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 10
