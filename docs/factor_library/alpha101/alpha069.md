# alpha069 — (rank(ts_max(delta(IndNeutralize(vwap,industry),3),5)) ^ ts_rank(corr(close-vwap blend, adv20, 5), 9)) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry),
                       2.72412), 4.79344))^
     Ts_Rank(correlation(close * 0.490655 + vwap * (1 - 0.490655),
                         adv20, 4.92416), 9.0615)) * -1

Required panel columns: ``vwap``, ``close``, ``adv20``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

vwap``, ``close``, ``adv20``, ``stock_code``,

## References

- Kakushadze 2015, eq. 69
