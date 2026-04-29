# alpha039 — (-rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))) * (1 + rank(sum(returns,250)))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(-1 * rank(delta(close, 7) *
               (1 - rank(decay_linear(volume / adv20, 9))))) *
    (1 + rank(sum(returns, 250)))

Required panel columns: ``close``, ``volume``, ``adv20``, ``returns``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``volume``, ``adv20``, ``returns``,

## References

- Kakushadze 2015, eq. 39
