# alpha098 — rank(decay_linear(corr(vwap,sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(ts_argmin(corr(rank(open),rank(adv15),21),9),7),8))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418),
                      7.18088)) -
    rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(
        rank(open), rank(adv15), 20.8187
    ), 8.62571), 6.95668), 8.07206))

Required panel columns: ``vwap``, ``adv5``, ``open``, ``adv15``,
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

vwap``, ``adv5``, ``open``, ``adv15``,

## References

- Kakushadze 2015, eq. 98
