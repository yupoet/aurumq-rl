# alpha091 — (ts_rank(decay_linear(decay_linear(corr(IndNeutralize(close,industry), volume, 10), 16), 4), 5) - rank(decay_linear(corr(vwap,adv30,4),3))) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(Ts_Rank(decay_linear(decay_linear(correlation(
        IndNeutralize(close, IndClass.industry), volume, 9.74928
    ), 16.398), 3.83219), 4.8667) -
     rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1

Required panel columns: ``close``, ``volume``, ``vwap``, ``adv30``,
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

close``, ``volume``, ``vwap``, ``adv30``,

## References

- Kakushadze 2015, eq. 91
