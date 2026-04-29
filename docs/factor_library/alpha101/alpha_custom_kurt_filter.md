# alpha_custom_kurt_filter — Negative CS rank of 20-day rolling kurtosis of returns

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Kurt(returns, 20))

```
-1 * Rank(Ts_Kurt(returns, 20))
```

## Polars Implementation Notes

Excess kurtosis flags fat-tailed return regimes. The sign flip filters
out names whose recent distribution is most leptokurtic — a contrarian
risk-off tilt.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

returns``, ``stock_code``, ``trade_date

## References

_(not specified)_
