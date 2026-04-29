# alpha042 — Rank(vwap - close) / Rank(vwap + close)

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank((vwap - close)) / rank((vwap + close))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank(vwap - close) / Rank(vwap + close)

```
Rank(vwap - close) / Rank(vwap + close)
```

## Polars Implementation Notes

Two CS ranks; divide. ``rank(vwap + close)`` should never be zero on real
data but the synthetic panel may produce ties — Polars handles that with
average-rank semantics.

Required panel columns: ``vwap``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

vwap``, ``close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 42
