# alpha070 — (rank(delta(vwap,1)) ^ ts_rank(corr(IndNeutralize(close,industry), adv50, 18), 18)) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(delta(vwap, 1.29456))^
     Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50,
                         17.8256), 17.9171)) * -1

Required panel columns: ``vwap``, ``close``, ``adv50``, ``stock_code``,
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

vwap``, ``close``, ``adv50``, ``stock_code``,

## References

- Kakushadze 2015, eq. 70
