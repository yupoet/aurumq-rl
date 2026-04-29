# alpha078 — Power composition of two correlation ranks

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

rank(correlation(sum(low*0.352 + vwap*0.648, 20), sum(adv40, 20), 7))
    ^ rank(correlation(rank(vwap), rank(volume), 6))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Power(
      Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), Ts_Sum(adv40, 20), 7)),
      Rank(Ts_Corr(Rank(vwap), Rank(volume), 6))
    )

```
Power(Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), Ts_Sum(adv40, 20), 7)), Rank(Ts_Corr(Rank(vwap), Rank(volume), 6)))
```

## Polars Implementation Notes

Two CS-rank values produce the (base, exponent) pair for ``Power``.

Required panel columns: ``low``, ``vwap``, ``adv40``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

low``, ``vwap``, ``adv40``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 78
