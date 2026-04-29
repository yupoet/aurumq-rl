# alpha061 — rank(vwap-ts_min(vwap,16)) < rank(corr(vwap, adv180, 18))

> **Category**: adv_extended | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(vwap - ts_min(vwap, 16.1219)) <
    rank(correlation(vwap, adv180, 17.9282))

Required panel columns: ``vwap``, ``adv180``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

vwap``, ``adv180``, ``stock_code``,

## References

- Kakushadze 2015, eq. 61
