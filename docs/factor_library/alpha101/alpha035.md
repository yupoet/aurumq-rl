# alpha035 — Volume rank × inverse range rank × inverse returns rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

ts_rank(volume, 32)
    * (1 - ts_rank(close + high - low, 16))
    * (1 - ts_rank(returns, 32))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Ts_Rank(volume, 32)
    * (1 - Ts_Rank(close + high - low, 16))
    * (1 - Ts_Rank(returns, 32))

```
Ts_Rank(volume, 32) * (1 - Ts_Rank(close + high - low, 16)) * (1 - Ts_Rank(returns, 32))
```

## Polars Implementation Notes

Pure TS chain. Three rolling ranks combined multiplicatively.

Required panel columns: ``volume``, ``close``, ``high``, ``low``, ``returns``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

volume``, ``close``, ``high``, ``low``, ``returns``,

## References

- Kakushadze 2015, eq. 35
