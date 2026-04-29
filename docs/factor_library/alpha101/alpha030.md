# alpha030 — (1 - rank(sum(sign(delta(close,1)) over last 3 days))) * sum(volume,5)/sum(volume,20)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((1.0 - rank(sign(close - delay(close, 1)) +
                 sign(delay(close, 1) - delay(close, 2)) +
                 sign(delay(close, 2) - delay(close, 3)))) *
     sum(volume, 5)) / sum(volume, 20)

Required panel columns: ``close``, ``volume``, ``stock_code``,
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

close``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 30
