# alpha011 — (rank(ts_max((vwap-close),3)) + rank(ts_min((vwap-close),3))) * rank(delta(volume,3))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
    rank(delta(volume, 3))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Stage the per-stock ts_max/ts_min/delta first, then take three CS
ranks on the materialised columns and combine.

Required panel columns: ``vwap``, ``close``, ``volume``, ``stock_code``,
``trade_date``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Required Panel Columns

vwap``, ``close``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 11
