# alpha086 — (ts_rank(corr(close,sum(adv20,15),6),20) < rank((open+close)-(vwap+open))) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) <
     rank(open + close - vwap - open)) * -1

Required panel columns: ``close``, ``adv20``, ``open``, ``vwap``,
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

close``, ``adv20``, ``open``, ``vwap``,

## References

- Kakushadze 2015, eq. 86
