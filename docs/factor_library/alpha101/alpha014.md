# alpha014 — Returns-acceleration rank scaled by open-volume correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)

```
(-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)
```

## Polars Implementation Notes

Inner Delta is TS, then CS rank, then outer multiplied by TS corr.
We stage the delta to make the CS rank pure.

Required panel columns: ``returns``, ``open``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

returns``, ``open``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 14
