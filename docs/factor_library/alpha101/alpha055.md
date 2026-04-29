# alpha055 — Negative correlation between %K rank and volume rank

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * correlation(
        rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
        rank(volume),
        6,
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Ts_Corr(
        Rank((close - Ts_Min(low, 12)) / (Ts_Max(high, 12) - Ts_Min(low, 12))),
        Rank(volume),
        6,
    )

```
-1 * Ts_Corr(Rank((close - Ts_Min(low, 12)) / (Ts_Max(high, 12) - Ts_Min(low, 12))), Rank(volume), 6)
```

## Polars Implementation Notes

Compute the %K series TS-wise, CS rank it, CS rank volume, then TS corr.

Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

close``, ``low``, ``high``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 55
