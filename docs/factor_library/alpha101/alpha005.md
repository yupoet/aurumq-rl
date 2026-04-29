# alpha005 — Open vs 10d VWAP mean rank, scaled by close-VWAP rank deviation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))

```
Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))
```

## Polars Implementation Notes

Both inner expressions can be evaluated as expressions, then ranked
cross-sectionally. We stage the TS-sum and the inner deviations to
keep the CS rank step pure-CS.

Required panel columns: ``open``, ``vwap``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

open``, ``vwap``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, eq. 5
