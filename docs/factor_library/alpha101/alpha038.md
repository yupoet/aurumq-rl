# alpha038 — Rank of 10d close rolling rank times close-over-open rank, negated

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((-1 * rank(ts_rank(close, 10))) * rank((close / open)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)

```
-1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)
```

## Polars Implementation Notes

1. Two CS ranks multiplied; each consumes a TS-column input. Materialise
   intermediates so the two ``cs_rank`` partitions don't collide.

Required panel columns: ``close``, ``open``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``open``, ``stock_code``, ``trade_date``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 38
