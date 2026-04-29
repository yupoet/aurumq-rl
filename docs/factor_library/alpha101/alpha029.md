# alpha029 — Deeply nested rank-scale-log composite of -delta(close-1,5), plus ts_rank(delay(-returns,6),5)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

min(product(rank(rank(scale(log(sum(ts_min(
        rank(rank(-1 * rank(delta((close - 1), 5)))), 2), 1))))), 1), 5) +
    ts_rank(delay(-1 * returns, 6), 5)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Outer ``product(... , 1)`` is identity (1-window product). Inner
cascade: delta → CS rank → CS rank → CS rank → TS min(2) → TS sum(1)
is identity → log → scale → CS rank → CS rank, then TS min(5).

Required panel columns: ``close``, ``returns``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

close``, ``returns``, ``stock_code``,

## References

- Kakushadze 2015, eq. 29
