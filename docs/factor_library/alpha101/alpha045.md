# alpha045 — Lagged-MA rank × short-corr × long/short-MA-corr rank, negated

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
           rank(correlation(sum(close, 5), sum(close, 20), 2))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) *
          Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))

```
-1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) * Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))
```

## Polars Implementation Notes

1. ``Ts_Sum(Delay(close, 5), 20) / 20`` -> per-stock lagged 20d MA.
2. Two ``Ts_Corr`` calls inside, both 2-window — they're noisy by design.
3. Two CS ranks; materialise the lagged-MA and the long/short-MA
   correlation before ranking.

Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``volume``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 45
