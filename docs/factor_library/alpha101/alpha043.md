# alpha043 — Volume-surge rank × 7d-decline rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)

```
Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)
```

## Polars Implementation Notes

Pure TS chain.

Required panel columns: ``volume``, ``adv20``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

volume``, ``adv20``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, eq. 43
