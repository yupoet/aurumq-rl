# alpha097 — (rank(decay_linear(delta(IndNeutralize(low-vwap blend,industry),3),20)) - ts_rank(decay_linear(ts_rank(corr(ts_rank(low,8),ts_rank(adv60,17),5),19),16),7)) * -1

> **Category**: industry_neutral | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

(rank(decay_linear(delta(IndNeutralize(
        low * 0.721001 + vwap * (1 - 0.721001), IndClass.industry
     ), 3.3705), 20.4523)) -
     Ts_Rank(decay_linear(Ts_Rank(correlation(
         Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547
     ), 18.5925), 15.7152), 6.71659)) * -1

Required panel columns: ``low``, ``vwap``, ``adv60``, ``stock_code``,
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

low``, ``vwap``, ``adv60``, ``stock_code``,

## References

- Kakushadze 2015, eq. 97
