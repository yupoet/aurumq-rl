# alpha089 — ts_rank(decay_linear(corr(low blend, adv10, 7), 6), 4) - ts_rank(decay_linear(delta(IndNeutralize(vwap,industry),3), 10), 15)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

Ts_Rank(decay_linear(correlation(low * 0.967285 + low * (1 - 0.967285),
                                      adv10, 6.94279), 5.51607), 3.79744) -
    Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry),
                                3.48158), 10.1466), 15.3012)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Synthetic panel uses ``adv15`` as substitute for ``adv10``.

Required panel columns: ``low``, ``vwap``, ``adv10``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

low``, ``vwap``, ``adv10``, ``stock_code``,

## References

- Kakushadze 2015, eq. 89
