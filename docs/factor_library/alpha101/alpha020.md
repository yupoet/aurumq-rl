# alpha020 — -1 * rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * rank(open - delay(high, 1)) *
          rank(open - delay(close, 1)) *
          rank(open - delay(low, 1))

Required panel columns: ``open``, ``high``, ``close``, ``low``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

open``, ``high``, ``close``, ``low``,

## References

- Kakushadze 2015, eq. 20
