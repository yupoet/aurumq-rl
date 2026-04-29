# alpha_custom_decaylinear_mom — Decay-linear-weighted 10d momentum rank

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(Ts_DecayLinear(returns, 10))

```
Rank(Ts_DecayLinear(returns, 10))
```

## Polars Implementation Notes

1. ``ts_decay_linear(returns, 10)`` -> per-stock weighted MA of returns
   with linearly decaying weights ``[10, 9, ..., 1] / 55``.
2. Cross-section rank (``cs_rank``) the materialised column. Two
   partitions -> stage first.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

returns``, ``stock_code``, ``trade_date``.

## References

- AurumQ project-internal custom factor
