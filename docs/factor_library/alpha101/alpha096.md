# alpha096 — -max(ts_rank(decay_linear(corr(rank(vwap),rank(volume),4),4),8), ts_rank(decay_linear(ts_argmax(corr(ts_rank(close,7),ts_rank(adv60,4),4),13),14),13))

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume),
                                          3.83878), 4.16783), 8.38151),
        Ts_Rank(decay_linear(Ts_ArgMax(correlation(
            Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459
        ), 12.6556), 14.0365), 13.4143)) * -1

Required panel columns: ``vwap``, ``volume``, ``close``, ``adv60``,
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

vwap``, ``volume``, ``close``, ``adv60``,

## References

- Kakushadze 2015, eq. 96
