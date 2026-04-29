# alpha053 — Negative 9-day delta of (close-low minus high-close)/(close-low)

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

-1 * delta(((close - low) - (high - close)) / (close - low), 9)

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * Delta(((close - low) - (high - close)) / (close - low), 9)

```
-1 * Delta(((close - low) - (high - close)) / (close - low), 9)
```

## Polars Implementation Notes

1. STHSF guards against ``(close - low) == 0`` by replacing with ``1e-4``.
   We mirror that to avoid division-by-zero NaN.
2. The inner ratio sits in [-1, 1] (close at low → -1, close at high → 1).

Required panel columns: ``close``, ``low``, ``high``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

close``, ``low``, ``high``, ``stock_code``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 53
