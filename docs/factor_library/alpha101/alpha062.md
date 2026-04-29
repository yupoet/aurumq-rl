# alpha062 — rank(corr(vwap,sum(adv20,22),10)) < rank(open-rank vs midpoint inequality)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
     rank(((rank(open) + rank(open)) <
           (rank((high + low) / 2) + rank(high))))) * -1

Required panel columns: ``vwap``, ``adv20``, ``open``, ``high``, ``low``,
``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

vwap``, ``adv20``, ``open``, ``high``, ``low``,

## References

- Kakushadze 2015, eq. 62
