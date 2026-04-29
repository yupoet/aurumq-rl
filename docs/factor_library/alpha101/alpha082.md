# alpha082 — -min(rank(decay_linear(delta(open,1),15)), ts_rank(decay_linear(corr(IndNeutralize(volume,industry), open, 17), 7), 13))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

min(rank(decay_linear(delta(open, 1.46063), 14.8717)),
        Ts_Rank(decay_linear(correlation(
            IndNeutralize(volume, IndClass.sector),
            open * 0.634196 + open * (1 - 0.634196),
            17.4842
        ), 6.92131), 13.4283)) * -1

Required panel columns: ``open``, ``volume``, ``stock_code``,
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

open``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 82
