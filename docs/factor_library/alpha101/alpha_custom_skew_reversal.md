# alpha_custom_skew_reversal — Negative CS rank of 20-day rolling skew of returns

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Skew(returns, 20))

```
-1 * Rank(Ts_Skew(returns, 20))
```

## Polars Implementation Notes

Captures distributional asymmetry: stocks with strongly positive return
skew (rare large up moves) rank higher in raw form, and the sign flip
bets on reversal.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

returns``, ``stock_code``, ``trade_date

## References

_(not specified)_
