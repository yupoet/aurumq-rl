# alpha031 — Triple-rank decay of -rank(rank(delta(close,10))) + rank(-delta(close,3)) + sign(scale(corr(adv20,low,12)))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(rank(rank(decay_linear(-1 * rank(rank(delta(close, 10))), 10)))) +
    rank(-1 * delta(close, 3)) +
    sign(scale(correlation(adv20, low, 12)))

Required panel columns: ``close``, ``adv20``, ``low``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``adv20``, ``low``, ``stock_code``,

## References

- Kakushadze 2015, eq. 31
