# alpha047 — ((rank(1/close)*volume/adv20) * (high*rank(high-close)/sma(high,5))) - rank(vwap-delay(vwap,5))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((((rank(1/close) * volume) / adv20) *
      ((high * rank(high - close)) / (sum(high, 5) / 5))) -
     rank(vwap - delay(vwap, 5)))

Required panel columns: ``close``, ``volume``, ``adv20``, ``high``,
``vwap``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

_(not specified)_

## Required Panel Columns

close``, ``volume``, ``adv20``, ``high``,

## References

- Kakushadze 2015, eq. 47
