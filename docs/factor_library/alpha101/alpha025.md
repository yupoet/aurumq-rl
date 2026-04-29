# alpha025 — Rank of negative-returns × volume-weighted price-tail

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(((-1 * returns) * adv20) * vwap * (high - close))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Rank((-1 * returns) * adv20 * vwap * (high - close))

```
Rank((-1 * returns) * adv20 * vwap * (high - close))
```

## Polars Implementation Notes

Build the multiplicative payload first, then CS rank.

Required panel columns: ``returns``, ``adv20``, ``vwap``, ``high``, ``close``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

returns``, ``adv20``, ``vwap``, ``high``, ``close``,

## References

- Kakushadze 2015, eq. 25
