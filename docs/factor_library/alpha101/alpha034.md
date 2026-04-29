# alpha034 — Rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) +
          (1 - rank(delta(close, 1)))))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) +
         (1 - Rank(Delta(close, 1))))

```
Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) + (1 - Rank(Delta(close, 1))))
```

## Polars Implementation Notes

STHSF rewrites the inner expression as ``2 - rank(ratio) - rank(delta)``
and replaces inf/NaN in the volatility ratio with 1 to avoid
constant-window pollution. We mirror both behaviours.

Required panel columns: ``returns``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

returns``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 34
