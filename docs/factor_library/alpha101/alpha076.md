# alpha076 — -max(rank(decay_linear(delta(vwap,1),12)), ts_rank(decay_linear(ts_rank(corr(IndNeutralize(low,industry), adv81, 8), 20), 17), 19))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
        Ts_Rank(decay_linear(Ts_Rank(correlation(
            IndNeutralize(low, IndClass.sector), adv81, 8.14941
        ), 19.569), 17.1543), 19.383)) * -1

Required panel columns: ``vwap``, ``low``, ``adv81``, ``stock_code``,
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

vwap``, ``low``, ``adv81``, ``stock_code``,

## References

- Kakushadze 2015, eq. 76
