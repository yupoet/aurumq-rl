# alpha072 — Decayed mid-price vs adv40 corr / decayed VWAP-volume corr

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(decay_linear(correlation((high+low)/2, adv40, 9), 10))
    / rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10))
    / Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))

```
Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 9), 10)) / Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(vwap, 4), Ts_Rank(volume, 19), 7), 3))
```

## Polars Implementation Notes

Two TS chains each ending in CS rank; the final result is their ratio.

Required panel columns: ``high``, ``low``, ``adv40``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``low``, ``adv40``, ``vwap``, ``volume``,

## References

- Kakushadze 2015, eq. 72
