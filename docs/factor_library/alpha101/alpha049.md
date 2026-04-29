# alpha049 — If close-acceleration < -0.1 then 1 else -delta(close,1)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((((delay(close, 20) - delay(close, 10)) / 10 -
       (delay(close, 10) - close) / 10) < -0.1) ? 1 :
     (-1 * (close - delay(close, 1))))

Required panel columns: ``close``, ``stock_code``, ``trade_date``,
``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``stock_code``, ``trade_date``,

## References

- Kakushadze 2015, eq. 49
