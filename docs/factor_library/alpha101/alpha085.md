# alpha085 — Power composition of weighted-price/adv30 and rank-rank correlations

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(correlation(high*0.876 + close*0.124, adv30, 10))
    ^ rank(correlation(ts_rank((high+low)/2, 4), ts_rank(volume, 10), 7))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Power(
      Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)),
      Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7))
    )

```
Power(Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)), Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7)))
```

## Polars Implementation Notes

Two CS ranks form base/exponent of pow.

Required panel columns: ``high``, ``close``, ``adv30``, ``low``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``close``, ``adv30``, ``low``, ``volume``,

## References

- Kakushadze 2015, eq. 85
