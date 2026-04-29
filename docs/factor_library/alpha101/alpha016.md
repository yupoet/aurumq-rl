# alpha016 — -1 * rank(covariance(rank(high), rank(volume), 5))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank(covariance(rank(high), rank(volume), 5))

Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

high``, ``volume``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 16
