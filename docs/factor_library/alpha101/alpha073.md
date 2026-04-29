# alpha073 — Negative max of decayed VWAP delta rank and blend reversal rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * max(
        rank(decay_linear(delta(vwap, 5), 3)),
        ts_rank(decay_linear(-delta(open*0.147155 + low*0.852845, 2)
                / (open*0.147155 + low*0.852845), 3), 17)
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Max(
        Rank(Ts_DecayLinear(Delta(vwap, 5), 3)),
        Ts_Rank(Ts_DecayLinear(
            -1 * Delta(open * 0.147155 + low * 0.852845, 2)
            / (open * 0.147155 + low * 0.852845), 3), 17)
    )

```
-1 * Max(Rank(Ts_DecayLinear(Delta(vwap, 5), 3)), Ts_Rank(Ts_DecayLinear(-1 * Delta(open * 0.147155 + low * 0.852845, 2) / (open * 0.147155 + low * 0.852845), 3), 17))
```

## Polars Implementation Notes

The paper's fractional windows round to the standard STHSF integer
windows: 5, 3, 2, 3 and 17.

Required panel columns: ``vwap``, ``open``, ``low``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

vwap``, ``open``, ``low``, ``stock_code``,

## References

- Kakushadze 2015, eq. 73
