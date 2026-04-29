# alpha092 — Pattern-flag rank vs low-adv30 rank correlation

> **Category**: technical | **Direction**: reverse | **Quality**: ok

## Original WorldQuant Formula

Min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)),
                             14.7221), 18.7484),
        Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58553),
                             6.94279), 6.80584))

## Intuition (人工补)

_(not specified)_

## Legacy AQML Expression (deprecated)

Min(Ts_Rank(Ts_DecayLinear(If((high + low) / 2 + close < low + open, 1.0, 0.0),
                               15), 19),
        Ts_Rank(Ts_DecayLinear(Ts_Corr(Rank(low), Rank(adv30), 8), 7), 7))

```
Min(Ts_Rank(Ts_DecayLinear(If((high + low) / 2 + close < low + open, 1.0, 0.0), 15), 19), Ts_Rank(Ts_DecayLinear(Ts_Corr(Rank(low), Rank(adv30), 8), 7), 7))
```

## Polars Implementation Notes

1. **Left branch** — a binary pattern flag (``hl2 + close < low + open``)
   smoothed by a 15-day decay-linear MA, then ranked over 19 days.
   The condition triggers when the candle has a high upper-wick close
   (rough proxy for distribution).
2. **Right branch** — rolling 8-day correlation between cross-section
   rank of ``low`` and rank of ``adv30``, smoothed by a 7-day
   decay-linear MA, then 7-day rolling rank.
3. The output is the **element-wise min** of the two ranks. We use
   ``pl.min_horizontal`` for clarity.
4. Two ``cs_rank`` partitions appear (rank of ``low``, rank of
   ``adv30``) — both must be materialised before being fed into the
   8-window ``ts_corr``.

Required panel columns: ``high``, ``low``, ``open``, ``close``,
``adv30``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``technical``

## Required Panel Columns

high``, ``low``, ``open``, ``close``,

## References

- Kakushadze 2015, '101 Formulaic Alphas', arXiv:1601.00991, eq. 92
