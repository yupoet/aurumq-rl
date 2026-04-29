# alpha002 — Volume change rank vs intraday return rank correlation

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)

```
-1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)
```

## Polars Implementation Notes

1. Inner ``Rank`` partitions by ``trade_date`` (CS); outer ``Ts_Corr``
   partitions by ``stock_code`` (TS). Materialise the two per-row
   ranked Series before correlating.

Required panel columns: ``volume``, ``close``, ``open``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

volume``, ``close``, ``open``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 2
