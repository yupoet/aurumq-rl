# alpha074 — Close vs adv30-sum corr vs weighted-price-volume corr

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

if rank(correlation(close, sum(adv30, 37), 15))
          < rank(correlation(rank(high*0.0261 + vwap*0.9739), rank(volume), 11))
    then -1 else 0

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15))
            < Rank(Ts_Corr(Rank(high*0.0261 + vwap*0.9739), Rank(volume), 11)), -1, 0)

```
If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15)) < Rank(Ts_Corr(Rank(high * 0.0261 + vwap * 0.9739), Rank(volume), 11)), -1, 0)
```

## Polars Implementation Notes

AQML uses ``Ts_Sum``; STHSF uses ``sma`` (rolling mean) — STHSF parity
may diverge as for #065. False branch is 0 (not 1) per AQML.

Required panel columns: ``close``, ``adv30``, ``high``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

close``, ``adv30``, ``high``, ``vwap``, ``volume``,

## References

- Kakushadze 2015, eq. 74
