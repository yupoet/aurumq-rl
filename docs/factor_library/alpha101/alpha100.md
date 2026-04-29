# alpha100 — Sub-industry-neutralised body*volume signal minus scale(IndNeutralize(corr(close,rank(adv20),5)-rank(ts_argmin(close,30)), sub_industry)), weighted by volume/adv20

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

0 - 1 * (
        (1.5 * scale(IndNeutralize(IndNeutralize(
            rank(((close - low) - (high - close)) / (high - low) * volume),
            IndClass.subindustry), IndClass.subindustry)) -
         scale(IndNeutralize(
             (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),
             IndClass.subindustry
         ))
        ) * (volume / adv20)
    )

Required panel columns: ``close``, ``low``, ``high``, ``volume``,
``adv20``, ``stock_code``, ``trade_date``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``low``, ``high``, ``volume``,

## References

- Kakushadze 2015, eq. 100
