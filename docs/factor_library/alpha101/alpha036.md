# alpha036 — 5-component weighted rank composite (alpha036 paper formula)

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

2.21 * rank(correlation((close - open), delay(volume, 1), 15)) +
    0.7  * rank(open - close) +
    0.73 * rank(Ts_Rank(delay(-1 * returns, 6), 5)) +
    rank(abs(correlation(vwap, adv20, 6))) +
    0.6  * rank((sum(close, 200) / 200 - open) * (close - open))

Required panel columns: ``close``, ``open``, ``volume``, ``returns``,
``vwap``, ``adv20``, ``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``open``, ``volume``, ``returns``,

## References

- Kakushadze 2015, eq. 36
