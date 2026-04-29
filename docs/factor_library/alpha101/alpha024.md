# alpha024 — If delta(sma(close,100),100)/delay(close,100) <= 0.05 then -(close-ts_min(close,100)) else -delta(close,3)

> **Category**: cap_weighted | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((delta(sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05)
    ? -1 * (close - ts_min(close, 100))
    : -1 * delta(close, 3)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

The 100-day windows make this a long-horizon factor; on the synthetic
60-day panel the output is mostly NaN until day ~100 (which never
arrives on synthetic). Steady-state test is consequently best-effort.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``cap_weighted``

## Required Panel Columns

close``, ``stock_code``, ``trade_date

## References

- Kakushadze 2015, eq. 24
