# alpha099 — Mid-price-adv60 corr vs low-volume corr

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

if rank(correlation(sum((high+low)/2, 19), sum(adv60, 19), 8))
          < rank(correlation(low, volume, 6))
    then -1 else 0

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8))
            < Rank(Ts_Corr(low, volume, 6)), -1, 0)

```
If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8)) < Rank(Ts_Corr(low, volume, 6)), -1, 0)
```

## Polars Implementation Notes

AQML ``Ts_Sum`` vs STHSF ``sma`` again — STHSF parity may diverge for
constant-factor reasons. False branch is 0.

Required panel columns: ``high``, ``low``, ``adv60``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``low``, ``adv60``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 99
