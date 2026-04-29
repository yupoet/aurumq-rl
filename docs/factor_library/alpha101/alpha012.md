# alpha012 — Volume direction times negative price change

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

sign(delta(volume, 1)) * (-1 * delta(close, 1))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))

```
Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))
```

## Polars Implementation Notes

Pure TS — sign of one-day volume change times negative one-day
close change. Sign returns 0 on zero change.

Required panel columns: ``volume``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

volume``, ``close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 12
