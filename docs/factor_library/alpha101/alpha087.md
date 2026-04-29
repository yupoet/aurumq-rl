# alpha087 — -max(rank(decay_linear(delta(close-vwap blend,2),3)), ts_rank(decay_linear(abs(corr(IndNeutralize(adv81,industry), close, 13)), 5), 14))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

max(rank(decay_linear(delta(close * 0.369701 + vwap * (1 - 0.369701),
                                1.91233), 2.65461)),
        Ts_Rank(decay_linear(abs(correlation(
            IndNeutralize(adv81, IndClass.industry), close, 13.4132
        )), 4.89768), 14.4535)) * -1

Required panel columns: ``close``, ``vwap``, ``adv81``, ``stock_code``,
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

close``, ``vwap``, ``adv81``, ``stock_code``,

## References

- Kakushadze 2015, eq. 87
