# alpha068 — Composite price/adv15 rank vs weighted-price delta

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

if ts_rank(correlation(rank(high), rank(adv15), 9), 14)
          < rank(delta(close*0.518 + low*0.482, 1))
    then -1 else 1

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14)
            < Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)

```
If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14) < Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)
```

## Polars Implementation Notes

Stage CS rank of high and adv15 → TS corr → TS rank → compare with
CS rank of TS delta of weighted price.

Required panel columns: ``high``, ``adv15``, ``close``, ``low``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

high``, ``adv15``, ``close``, ``low``, ``stock_code``,

## References

- Kakushadze 2015, eq. 68
