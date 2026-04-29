# alpha022 — Change in high-vol correlation scaled by 20d stdev rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))

```
-1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))
```

## Polars Implementation Notes

Inner TS corr → TS delta → multiplied by CS rank of TS std. Stage the
20-day std before the CS rank.

Required panel columns: ``high``, ``volume``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``volume``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, eq. 22
