# alpha057 — Negative (close - vwap) divided by 2-day decay-linear of CS rank of 30-day argmax of close

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))

```
-1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))
```

## Polars Implementation Notes

1. ``ts_argmax(close, 30)``: position of 30-day max — encodes recency of
   the recent peak.
2. CS rank of that position, then 2-day decay-linear smoothing.
3. Divide ``close - vwap`` (intraday premium) by the smoothed rank.
4. Sign flipped so that "premium plus stale peak" is bearish.

NOTE: Not present in STHSF reference parquet — parity test skipped.

Required panel columns: ``close``, ``vwap``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``vwap``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 57
