# alpha080 — (rank(sign(delta(IndNeutralize(open-high blend,industry),4))) ^ ts_rank(corr(high,adv15,5),6)) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(Sign(delta(IndNeutralize(open * 0.868128 + high * (1 - 0.868128),
                                   IndClass.industry), 4.04545)))^
     Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1

Required panel columns: ``open``, ``high``, ``adv10``, ``stock_code``,
``trade_date``, ``industry``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Synthetic panel doesn't have ``adv10`` (only adv5/15/...). We use
``adv15`` as the closest available substitute on the synthetic panel
while keeping the original formula's intent. On real production
panels the adv10 column is present.

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

open``, ``high``, ``adv10``, ``stock_code``,

## References

- Kakushadze 2015, eq. 80
