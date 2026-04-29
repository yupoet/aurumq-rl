# alpha054 — Open^5 / Close^5 weighted intraday tail signal

> **Category**: breakout | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

-1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))

```
-1 * ((low - close) * Power(open, 5)) / ((low - high) * Power(close, 5))
```

## Polars Implementation Notes

1. Element-wise only — no rolling, no rank.
2. The denominator ``low - high`` is non-positive on every regular
   trading day (low <= high). It is zero when high == low (a fully
   static day), which yields ``inf``/``nan`` — STHSF reference shows
   the same behaviour, so we don't guard against it.

Required panel columns: ``low``, ``high``, ``open``, ``close``.

Direction: ``reverse``
Category: ``breakout``

## Required Panel Columns

low``, ``high``, ``open``, ``close``.

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 54
