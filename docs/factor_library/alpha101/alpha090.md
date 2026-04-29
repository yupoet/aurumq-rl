# alpha090 — (rank(close-ts_max(close,5)) ^ ts_rank(corr(IndNeutralize(adv40,sub_industry), low, 5), 3)) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(close - ts_max(close, 4.66719))^
     Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry),
                         low, 5.38375), 3.21856)) * -1

Required panel columns: ``close``, ``adv40``, ``low``, ``stock_code``,
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

close``, ``adv40``, ``low``, ``stock_code``,

## References

- Kakushadze 2015, eq. 90
