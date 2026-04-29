# alpha079 — rank(delta(IndNeutralize(close-open blend,industry),1)) < rank(corr(ts_rank(vwap,4), ts_rank(adv150,9), 15))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(delta(IndNeutralize(close * 0.60733 + open * (1 - 0.60733),
                             IndClass.sector), 1.23438)) <
    rank(correlation(Ts_Rank(vwap, 3.60973),
                     Ts_Rank(adv150, 9.18637), 14.6644))

Required panel columns: ``close``, ``open``, ``vwap``, ``adv150``,
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

close``, ``open``, ``vwap``, ``adv150``,

## References

- Kakushadze 2015, eq. 79
