# alpha017 — Momentum exhaustion: rank-mom × second-derivative × volume-surge

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
     rank(ts_rank((volume / adv20), 5)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) *
    Rank(Ts_Rank(volume / adv20, 5))

```
(-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) * Rank(Ts_Rank(volume / adv20, 5))
```

## Polars Implementation Notes

1. Three components, each a CS rank of a TS quantity. We materialise
   the three TS columns first, then cross-section rank, then multiply.
2. ``delta(delta(close, 1), 1)`` is the discrete second derivative —
   acceleration of price.

Required panel columns: ``close``, ``volume``, ``adv20``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

close``, ``volume``, ``adv20``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 17
