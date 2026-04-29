# gtja_101 — VWAP-volume corr ranked < volume-mean corr ranked

> **Category**: correlation | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

_(not specified)_

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

_pure-callable factor_

## Polars Implementation Notes

Two-stage ``with_columns`` to materialise the per-stock CORRs before
cross-section ranking, since polars cannot mix CS+TS partitions in a
single expression.

Direction: ``reverse`` (binary -1/0 — multiply by -1 in spec).
Category: ``correlation``.

## Required Panel Columns

_(not specified)_

## References

- Guotai Junan 191 short-period factor report, 2017
