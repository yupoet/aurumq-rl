# alpha056 — -rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns * cap)

> **Category**: cap_weighted | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

0 - 1 * (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) *
             rank(returns * cap))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

The denominator ``sum(sum(returns,2),3)`` is the 3-day rolling sum of
the 2-day rolling sum, i.e. cumulative returns over an effective
4-day window. ``returns * cap`` is the dollar return; CS rank
captures cross-section preference for high-dollar-return stocks.

Stocks with ``cap IS NULL`` produce a NaN factor row (because
``returns * NULL == NULL``, which propagates through CS rank).

Required panel columns: ``returns``, ``cap``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``cap_weighted``

## Required Panel Columns

returns``, ``cap``, ``stock_code``,

## References

- Kakushadze 2015, eq. 56
