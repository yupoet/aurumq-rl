# alpha088 — Min of decayed rank-spread and decayed correlation rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

min(
      rank(decay_linear(rank(open) + rank(low) - rank(high) - rank(close), 8)),
      ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60, 21), 8), 7), 3)
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Min(
      Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - (Rank(high) + Rank(close)), 8)),
      Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3)
    )

```
Min(Rank(Ts_DecayLinear((Rank(open) + Rank(low)) - (Rank(high) + Rank(close)), 8)), Ts_Rank(Ts_DecayLinear(Ts_Corr(Ts_Rank(close, 8), Ts_Rank(adv60, 21), 8), 7), 3))
```

## Polars Implementation Notes

Two parallel chains, combined by element-wise min.

Required panel columns: ``open``, ``low``, ``high``, ``close``, ``adv60``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

open``, ``low``, ``high``, ``close``, ``adv60``,

## References

- Kakushadze 2015, eq. 88
