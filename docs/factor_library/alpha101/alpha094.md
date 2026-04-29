# alpha094 — Negative power of VWAP-trough rank with correlation exponent

> **Category**: volume_price | **Direction**: reverse | **Quality**: warn

## Original WorldQuant Formula

-1 * rank(vwap - ts_min(vwap, 12))
          ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Power(
      Rank(vwap - Ts_Min(vwap, 12)),
      Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)
    )

```
-1 * Power(Rank(vwap - Ts_Min(vwap, 12)), Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3))
```

## Polars Implementation Notes

Stage TS ranks before TS corr; CS rank base; TS rank exponent.

Required panel columns: ``vwap``, ``adv60``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

vwap``, ``adv60``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 94
