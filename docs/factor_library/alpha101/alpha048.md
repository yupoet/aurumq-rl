# alpha048 — IndNeutralize((corr(delta(close,1), delta(delay(close,1),1), 250) * delta(close,1)) / close, sub_industry) / sum((delta(close,1)/delay(close,1))^2, 250)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

IndNeutralize(
        (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *
         delta(close, 1)) / close,
        IndClass.subindustry
    ) /
    sum((delta(close, 1) / delay(close, 1))^2, 250)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

250-day windows on a 60-day synthetic panel produce all-null output —
the steady-state test only verifies the structural shape, not value
coverage. On real production panels with >= 250 days the values are
well-defined.

Required panel columns: ``close``, ``stock_code``, ``trade_date``,
``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

close``, ``stock_code``, ``trade_date``,

## References

- Kakushadze 2015, eq. 48
