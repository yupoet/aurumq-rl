# alpha001 — Rank of squared-clip ts_argmax within past 5 days

> **Category**: volatility | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5

```
Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5
```

## Polars Implementation Notes

1. Conditional input: when returns < 0 use 20-day std of returns
   (volatility regime), otherwise use raw close
2. Square with sign preservation amplifies extreme moves
3. ts_argmax: position (0..4) of max within last 5 rows
4. Cross-section rank centered at 0.5 (subtract 0.5 -> [-0.5, +0.5])

Required panel columns: ``returns``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

## Required Panel Columns

returns``, ``close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, "101 Formulaic Alphas", arXiv:1601.00991, eq. 1
- STHSF/alpha101 (MIT) for pandas reference impl
- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 1
