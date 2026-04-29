# alpha077 — Min of two decayed-rank features

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

min(
      rank(decay_linear((high+low)/2 + high - vwap - high, 20)),
      rank(decay_linear(correlation((high+low)/2, adv40, 3), 6))
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Min(
      Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)),
      Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6))
    )

```
Min(Rank(Ts_DecayLinear(((high + low) / 2 + high) - (vwap + high), 20)), Rank(Ts_DecayLinear(Ts_Corr((high + low) / 2, adv40, 3), 6)))
```

## Polars Implementation Notes

Two parallel TS+CS chains combined by element-wise min.

Required panel columns: ``high``, ``low``, ``vwap``, ``adv40``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``low``, ``vwap``, ``adv40``, ``stock_code``,

## References

- Kakushadze 2015, eq. 77
