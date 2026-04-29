# alpha041 — Geometric mean of high and low minus vwap

> **Category**: mean_reversion | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((high * low)^0.5) - vwap

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Power(high * low, 0.5) - vwap

```
Power(high * low, 0.5) - vwap
```

## Polars Implementation Notes

Pure scalar arithmetic — no rolling or cross-section ops. Negative values
indicate vwap above the geometric mid-price.

Required panel columns: ``high``, ``low``, ``vwap``

Direction: ``reverse``
Category: ``mean_reversion``

## Required Panel Columns

high``, ``low``, ``vwap

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 41
