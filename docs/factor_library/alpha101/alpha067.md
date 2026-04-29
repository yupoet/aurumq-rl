# alpha067 — (rank(high-ts_min(high,2)) ^ rank(corr(IndNeutralize(vwap,industry), IndNeutralize(adv20,sub_industry), 6))) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(high - ts_min(high, 2.14593))^
     rank(correlation(IndNeutralize(vwap, IndClass.sector),
                      IndNeutralize(adv20, IndClass.subindustry),
                      6.02936))) * -1

Required panel columns: ``high``, ``vwap``, ``adv20``, ``stock_code``,
``trade_date``, ``industry``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

high``, ``vwap``, ``adv20``, ``stock_code``,

## References

- Kakushadze 2015, eq. 67
