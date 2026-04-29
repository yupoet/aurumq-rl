# alpha060 — Williams %R volume rank minus argmax rank, scaled

> **Category**: volume_price | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * (
        2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
        - scale(rank(ts_argmax(close, 10)))
    )

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * (
        2 * Scale(Rank(((close - low) - (high - close)) / (high - low) * volume))
        - Scale(Rank(Ts_ArgMax(close, 10)))
    )

```
-1 * (2 * Scale(Rank(((close - low) - (high - close)) / (high - low) * volume)) - Scale(Rank(Ts_ArgMax(close, 10))))
```

## Polars Implementation Notes

Two CS-scaled rank chains, one based on the Williams-%R-style payload,
the other on TS argmax of close.

Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

## Required Panel Columns

close``, ``low``, ``high``, ``volume``, ``stock_code``,

## References

- Kakushadze 2015, eq. 60
