# alpha071 — Decayed correlation vs decayed weighted-price-square rank, max

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

max(
      ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 16),
      ts_rank(decay_linear(rank(low + open - 2*vwap)^2, 16), 4)
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Max(
      Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16),
      Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4)
    )

```
Max(Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 3), Ts_Rank(adv180, 12), 18), 4), 16), Ts_Rank(Ts_DecayLinear(Power(Rank(low + open - vwap - vwap), 2), 16), 4))
```

## Polars Implementation Notes

Two parallel TS chains, combined element-wise via ``max_horizontal``.

Required panel columns: ``close``, ``adv180``, ``low``, ``open``, ``vwap``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

close``, ``adv180``, ``low``, ``open``, ``vwap``,

## References

- Kakushadze 2015, eq. 71
