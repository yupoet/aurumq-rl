# alpha064 — (rank(corr(sum(open-low blend,13), sum(adv120,13), 17)) < rank(delta(midpoint-vwap blend, 4))) * -1

> **Category**: adv_extended | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(correlation(sum(open*0.178404 + low*(1-0.178404), 12.7054),
                      sum(adv120, 12.7054), 16.6208)) <
     rank(delta((((high+low)/2)*0.178404 + vwap*(1-0.178404), 3.69741))) * -1

Required panel columns: ``open``, ``low``, ``adv120``, ``high``, ``vwap``,
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

open``, ``low``, ``adv120``, ``high``, ``vwap``,

## References

- Kakushadze 2015, eq. 64
