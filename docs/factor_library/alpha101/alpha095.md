# alpha095 — Open-trough rank vs medium-term correlation rank-power

> **Category**: breakout | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

(rank((open - ts_min(open, 12.4105))) <
     Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351),
                                sum(adv40, 19.1351), 12.8742))^5), 11.7584))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

If(Rank(open - Ts_Min(open, 12)) <
       Ts_Rank(Power(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19),
                                  Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)

```
If(Rank(open - Ts_Min(open, 12)) < Ts_Rank(Power(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)
```

## Polars Implementation Notes

1. Left side: cross-section rank of ``open - ts_min(open, 12)``
   (how high open is above its 12d trough).
2. Right side: rolling rank of (rank-of-corr ** 5) — a heavy-tailed
   indicator of how unusual the medium-term correlation between
   ``hl2`` sums and ``adv40`` sums has been over the last 12 days.
3. Two cross-section ranks need staging; the final boolean cast
   returns the WorldQuant 0/1 flag.

Required panel columns: ``open``, ``high``, ``low``, ``adv40``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``breakout``

## Required Panel Columns

open``, ``high``, ``low``, ``adv40``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 95
