# alpha075 — rank(corr(vwap,volume,4)) < rank(corr(rank(low), rank(adv50), 12))

> **Category**: adv_extended | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(correlation(vwap, volume, 4.24304)) <
    rank(correlation(rank(low), rank(adv50), 12.4413))

Required panel columns: ``vwap``, ``volume``, ``low``, ``adv50``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

vwap``, ``volume``, ``low``, ``adv50``,

## References

- Kakushadze 2015, eq. 75
