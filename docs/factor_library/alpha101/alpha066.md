# alpha066 — -(rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear((low-vwap)/(open-mid),11),7))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
     Ts_Rank(decay_linear(
         ((low * 0.96633 + low * (1 - 0.96633)) - vwap) /
         (open - (high + low) / 2), 11.4157
     ), 6.72611)) * -1

Required panel columns: ``vwap``, ``low``, ``high``, ``open``,
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

vwap``, ``low``, ``high``, ``open``,

## References

- Kakushadze 2015, eq. 66
