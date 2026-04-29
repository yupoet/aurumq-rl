# alpha084 — VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised)

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
                delta(close, 4.96796))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)

```
SignedPower(Ts_Rank(vwap - Ts_Max(vwap, 15), 21), 1.0)
```

## Polars Implementation Notes

1. The original WorldQuant formula uses a fractional-day rolling max
   and an exponent equal to a per-row delta; our migrated AQML form
   linearises the exponent to ``1.0`` and rounds the windows to
   integers (15 and 21). We follow the migrated form verbatim — this
   matches both the legacy AQML evaluator and the STHSF reference.
2. With exponent=1 the operation degenerates to identity (sign · |x|),
   so the alpha simplifies to just the rolling rank itself.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

vwap``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 84
