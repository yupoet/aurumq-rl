# alpha093 — ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), adv81, 17), 20), 8) / rank(decay_linear(delta(close-vwap blend, 3), 16))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry),
                                      adv81, 17.4193), 19.848), 7.54455) /
    rank(decay_linear(delta(close * 0.524434 + vwap * (1 - 0.524434),
                             2.77377), 16.2664))

Required panel columns: ``vwap``, ``adv81``, ``close``, ``stock_code``,
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

vwap``, ``adv81``, ``close``, ``stock_code``,

## References

- Kakushadze 2015, eq. 93
