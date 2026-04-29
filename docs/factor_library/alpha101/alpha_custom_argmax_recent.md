# alpha_custom_argmax_recent — Inverse rank of days-since-20d-max (recency-of-peak)

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

1 - Rank(Ts_ArgMax(close, 20))

```
1 - Rank(Ts_ArgMax(close, 20))
```

## Polars Implementation Notes

1. ``ts_argmax(close, 20)`` -> position 0..19 of the highest close in
   last 20 days; high values ⇒ peak was recent (today the peak).
2. ``1 - cs_rank(...)`` flips the rank so that recent peaks score high.
   Materialise the argmax column before ranking.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``.

## References

- AurumQ project-internal custom factor
