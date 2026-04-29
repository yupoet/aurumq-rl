# alpha065 — Volume-weighted price vs adv60 correlation rank vs open-min rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

if rank(correlation(open*0.0078 + vwap*0.9922, sum(adv60, 9), 6))
          < rank(open - ts_min(open, 14))
    then -1 else 1

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Rank(Ts_Corr(open*0.0078 + vwap*0.9922, Ts_Sum(adv60, 9), 6))
            < Rank(open - Ts_Min(open, 14)), -1, 1)

```
If(Rank(Ts_Corr(open * 0.0078 + vwap * 0.9922, Ts_Sum(adv60, 9), 6)) < Rank(open - Ts_Min(open, 14)), -1, 1)
```

## Polars Implementation Notes

AQML uses ``Ts_Sum`` (rolling sum); STHSF reference uses ``sma`` (rolling
mean), so STHSF parity may diverge by a constant scale factor that
drops out of CS rank — but tie-breaking around boundary cases can
flip a few signs. We follow AQML.

Required panel columns: ``open``, ``vwap``, ``adv60``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

open``, ``vwap``, ``adv60``, ``stock_code``,

## References

- Kakushadze 2015, eq. 65
