# alpha081 — Log-product of double-rank corr vs vwap-volume corr

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

if rank(log(product(rank(rank(correlation(vwap, sum(adv10, 50), 8)^4)), 15)))
          < rank(correlation(rank(vwap), rank(volume), 5))
    then -1 else 0

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, Ts_Sum(adv10, 50), 8), 4))), 15)))
            < Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)

```
If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, Ts_Sum(adv10, 50), 8), 4))), 15))) < Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)
```

## Polars Implementation Notes

Heavy nested expression. Stage step-by-step.

Required panel columns: ``vwap``, ``adv10``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

vwap``, ``adv10``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 81
