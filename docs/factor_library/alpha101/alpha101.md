# alpha101 — Intraday body over range — (close - open) / (high - low + 0.001)

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((close - open) / ((high - low) + 0.001))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

(close - open) / ((high - low) + 0.001)

```
(close - open) / ((high - low) + 0.001)
```

## Polars Implementation Notes

The simplest WorldQuant alpha. Despite its name (#101) it is purely
intraday and ships in STHSF's reference. Acts as a candle-strength
contrarian — large positive bodies relative to the day's range are
expected to fade.

Required panel columns: ``close``, ``open``, ``high``, ``low``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``open``, ``high``, ``low

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 101
