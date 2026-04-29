# alpha028 — Standardised mid-price vs close gap with volume modifier

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

scale(correlation(adv20, low, 5) + (high + low) / 2 - close)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)

```
Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)
```

## Polars Implementation Notes

TS corr then arithmetic, then CS scale (rescale to ``sum(|x|) == 1`` per day).

Required panel columns: ``adv20``, ``low``, ``high``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

adv20``, ``low``, ``high``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, eq. 28
