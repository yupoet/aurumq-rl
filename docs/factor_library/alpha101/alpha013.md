# alpha013 — Negative rank of close-rank vs volume-rank covariance

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank(covariance(rank(close), rank(volume), 5))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))

```
-1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))
```

## Polars Implementation Notes

Two CS ranks materialised, then TS covariance, then outer CS rank.
Two staging passes (CS → TS) followed by a final select.

Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

close``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 13
