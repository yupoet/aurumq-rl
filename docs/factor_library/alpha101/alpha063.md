# alpha063 — Diff of decay-linear(delta(IndNeutralize(close,industry),2),8) and decay-linear(corr(blend,sum(adv180,37),14),12), sign-flipped

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank(decay_linear(delta(IndNeutralize(close, IndClass.industry),
                              2.25164), 8.22237)) -
     rank(decay_linear(correlation(
          vwap * 0.318108 + open * (1 - 0.318108),
          sum(adv180, 37.2467), 13.557), 12.2883))) * -1

Required panel columns: ``close``, ``vwap``, ``open``, ``adv180``,
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

close``, ``vwap``, ``open``, ``adv180``,

## References

- Kakushadze 2015, eq. 63
