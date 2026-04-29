# alpha058 — -ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), volume, 4), 8), 6)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * Ts_Rank(decay_linear(
        correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795),
        7.89291
    ), 5.50322)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Continuous windows truncated to int (4, 8, 6). Sector neutralisation
is applied via ``ind_neutralize(vwap, "industry")`` — AurumQ's
available proxy for IndClass.sector.

Required panel columns: ``vwap``, ``volume``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

vwap``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 58
