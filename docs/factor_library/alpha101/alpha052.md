# alpha052 — Low-shift × medium-term excess return rank × volume rank

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
      rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) *
     Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) *
     Ts_Rank(volume, 5)

```
(-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) * Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) * Ts_Rank(volume, 5)
```

## Polars Implementation Notes

1. ``-Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)`` measures the change
   in the rolling-min low over the last 5 days vs 5 days ago.
2. The medium-term excess return uses 240 - 20 = 220 day window of
   carry, scaled by 1/220.
3. Materialise the rank-input, then ``cs_rank``, then multiply with
   the TS components.

Required panel columns: ``low``, ``returns``, ``volume``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

low``, ``returns``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 52
