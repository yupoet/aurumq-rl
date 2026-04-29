# alpha021 — Volatility-vs-momentum regime switch using sma(close,8)+/-std(close,8) with volume/adv20 confirmation

> **Category**: adv_extended | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((sum(close, 8) / 8 + stddev(close, 8)) < (sum(close, 2) / 2))
    ? -1 :
    ((sum(close, 2) / 2) < (sum(close, 8) / 8 - stddev(close, 8)))
    ? 1 :
    ((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : -1

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

The STHSF reference simplifies the cascade: cond1 OR cond2 ⇒ -1
(any volatility-narrowing), else +1. We implement the literal paper
cascade for fidelity.

Required panel columns: ``close``, ``volume``, ``adv20``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

## Required Panel Columns

close``, ``volume``, ``adv20``,

## References

- Kakushadze 2015, eq. 21
