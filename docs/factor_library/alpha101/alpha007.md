# alpha007 — Volume-conditional 7-day signed momentum rank

> **Category**: momentum | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((adv20 < volume) ? -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * Sign(Delta(close, 7)), -1)

```
If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * Sign(Delta(close, 7)), -1)
```

## Polars Implementation Notes

1. ``delta(close, 7)`` is a per-stock 7-day price change.
2. ``ts_rank(abs(...), 60)`` is a per-stock 60-window rank in [0, 1].
3. The condition selects between the rank-based momentum reversal and
   a constant ``-1``. When volume is below ``adv20``, the alpha collapses
   to a flat ``-1`` (i.e. the day carries no signal apart from the
   constant).

Required panel columns: ``adv20``, ``volume``, ``close``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

## Required Panel Columns

adv20``, ``volume``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 7
