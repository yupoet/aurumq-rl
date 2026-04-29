# Alpha101 Factor Library Index (105 factors)

| ID | Category | Direction | Description | Quality |
|---|---|---|---|---|
| alpha001 | volatility | reverse | Rank of squared-clip ts_argmax within past 5 days | ok |
| alpha018 | volatility | reverse | Negative CS rank of: 5-day std(\|close-open\|) + (close-open) + 10-day correl... | ok |
| alpha034 | volatility | reverse | Rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,... | ok |
| alpha040 | volatility | reverse | Negative rank(stddev(high,10)) * correlation(high, volume, 10) | ok |
| alpha_custom_skew_reversal | volatility | reverse | Negative CS rank of 20-day rolling skew of returns | ok |
| alpha_custom_kurt_filter | volatility | reverse | Negative CS rank of 20-day rolling kurtosis of returns | ok |
| alpha002 | volume_price | reverse | Volume change rank vs intraday return rank correlation | ok |
| alpha003 | volume_price | reverse | Open price rank vs volume rank correlation | ok |
| alpha005 | volume_price | reverse | Open vs 10d VWAP mean rank, scaled by close-VWAP rank deviation | ok |
| alpha006 | volume_price | reverse | 10d correlation between open and volume | ok |
| alpha012 | volume_price | reverse | Volume direction times negative price change | ok |
| alpha013 | volume_price | reverse | Negative rank of close-rank vs volume-rank covariance | ok |
| alpha014 | volume_price | reverse | Returns-acceleration rank scaled by open-volume correlation | ok |
| alpha022 | volume_price | reverse | Change in high-vol correlation scaled by 20d stdev rank | ok |
| alpha025 | volume_price | reverse | Rank of negative-returns × volume-weighted price-tail | ok |
| alpha026 | volume_price | reverse | Max recent volume-rank vs high-rank correlation | ok |
| alpha028 | volume_price | reverse | Standardised mid-price vs close gap with volume modifier | ok |
| alpha035 | volume_price | reverse | Volume rank × inverse range rank × inverse returns rank | ok |
| alpha043 | volume_price | reverse | Volume-surge rank × 7d-decline rank | ok |
| alpha044 | volume_price | reverse | Negative high-vs-volume-rank correlation | ok |
| alpha055 | volume_price | reverse | Negative correlation between %K rank and volume rank | ok |
| alpha060 | volume_price | reverse | Williams %R volume rank minus argmax rank, scaled | ok |
| alpha065 | volume_price | reverse | Volume-weighted price vs adv60 correlation rank vs open-min rank | ok |
| alpha068 | volume_price | reverse | Composite price/adv15 rank vs weighted-price delta | ok |
| alpha071 | volume_price | reverse | Decayed correlation vs decayed weighted-price-square rank, max | ok |
| alpha072 | volume_price | reverse | Decayed mid-price vs adv40 corr / decayed VWAP-volume corr | ok |
| alpha074 | volume_price | reverse | Close vs adv30-sum corr vs weighted-price-volume corr | ok |
| alpha077 | volume_price | reverse | Min of two decayed-rank features | ok |
| alpha078 | volume_price | reverse | Power composition of two correlation ranks | ok |
| alpha081 | volume_price | reverse | Log-product of double-rank corr vs vwap-volume corr | ok |
| alpha083 | volume_price | reverse | Range/MA delay rank times volume rank-squared, scaled | ok |
| alpha085 | volume_price | reverse | Power composition of weighted-price/adv30 and rank-rank correlations | ok |
| alpha088 | volume_price | reverse | Min of decayed rank-spread and decayed correlation rank | ok |
| alpha094 | volume_price | reverse | Negative power of VWAP-trough rank with correlation exponent | ok |
| alpha099 | volume_price | reverse | Mid-price-adv60 corr vs low-volume corr | ok |
| alpha007 | momentum | reverse | Volume-conditional 7-day signed momentum rank | ok |
| alpha008 | momentum | reverse | Acceleration of (open*returns) sum compared to 10 days ago | ok |
| alpha009 | momentum | reverse | Trend-confirmed price-change momentum | ok |
| alpha010 | momentum | reverse | Cross-sectional rank of trend-confirmed price change | ok |
| alpha017 | momentum | reverse | Momentum exhaustion: rank-mom × second-derivative × volume-surge | ok |
| alpha019 | momentum | reverse | 7d-return sign times annual rank-mom multiplier | ok |
| alpha038 | momentum | reverse | Rank of 10d close rolling rank times close-over-open rank, negated | ok |
| alpha045 | momentum | reverse | Lagged-MA rank × short-corr × long/short-MA-corr rank, negated | ok |
| alpha046 | momentum | reverse | Trend-shape conditional one-day reversal | ok |
| alpha051 | momentum | reverse | Trend curvature conditional reversal | ok |
| alpha052 | momentum | reverse | Low-shift × medium-term excess return rank × volume rank | ok |
| alpha084 | momentum | reverse | VWAP-vs-15d-max rank, sign-preserving (delta exponent linearised) | ok |
| alpha_custom_decaylinear_mom | momentum | reverse | Decay-linear-weighted 10d momentum rank | ok |
| alpha_custom_argmax_recent | momentum | reverse | Inverse rank of days-since-20d-max (recency-of-peak) | ok |
| alpha023 | breakout | reverse | High-breakout-conditional negative 2d high change | ok |
| alpha054 | breakout | reverse | Open^5 / Close^5 weighted intraday tail signal | ok |
| alpha095 | breakout | reverse | Open-trough rank vs medium-term correlation rank-power | ok |
| alpha092 | technical | reverse | Pattern-flag rank vs low-adv30 rank correlation | ok |
| alpha004 | mean_reversion | reverse | Negative 9-day Ts_Rank of cross-section rank of low | ok |
| alpha032 | mean_reversion | reverse | Scaled 7-day MA divergence + 20x scaled 230-day correlation between vwap and... | ok |
| alpha033 | mean_reversion | reverse | Cross-section rank of (open/close - 1) | ok |
| alpha037 | mean_reversion | reverse | Rank of 200-day correlation between delayed (open-close) and close, plus rank... | ok |
| alpha041 | mean_reversion | reverse | Geometric mean of high and low minus vwap | ok |
| alpha042 | mean_reversion | reverse | Rank(vwap - close) / Rank(vwap + close) | ok |
| alpha053 | mean_reversion | reverse | Negative 9-day delta of (close-low minus high-close)/(close-low) | ok |
| alpha057 | mean_reversion | reverse | Negative (close - vwap) divided by 2-day decay-linear of CS rank of 30-day ar... | ok |
| alpha101 | mean_reversion | reverse | Intraday body over range — (close - open) / (high - low + 0.001) | ok |
| alpha_custom_zscore_5d | mean_reversion | reverse | Negative 5-day rolling z-score of close | ok |
| alpha_custom_argmin_recent | mean_reversion | reverse | Cross-section rank of 20-day Ts_ArgMin of close | ok |
| alpha011 | industry_neutral | reverse | (rank(ts_max((vwap-close),3)) + rank(ts_min((vwap-close),3))) * rank(delta(vo... | ok |
| alpha016 | industry_neutral | reverse | -1 * rank(covariance(rank(high), rank(volume), 5)) | ok |
| alpha020 | industry_neutral | reverse | -1 * rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(l... | ok |
| alpha047 | industry_neutral | reverse | ((rank(1/close)*volume/adv20) * (high*rank(high-close)/sma(high,5))) - rank(v... | ok |
| alpha048 | industry_neutral | reverse | IndNeutralize((corr(delta(close,1), delta(delay(close,1),1), 250) * delta(clo... | ok |
| alpha027 | industry_neutral | reverse | Sign threshold of rank(sum(corr(rank(volume), rank(vwap), 6), 2)/2) | ok |
| alpha029 | industry_neutral | reverse | Deeply nested rank-scale-log composite of -delta(close-1,5), plus ts_rank(del... | ok |
| alpha030 | industry_neutral | reverse | (1 - rank(sum(sign(delta(close,1)) over last 3 days))) * sum(volume,5)/sum(vo... | ok |
| alpha031 | industry_neutral | reverse | Triple-rank decay of -rank(rank(delta(close,10))) + rank(-delta(close,3)) + s... | ok |
| alpha036 | industry_neutral | reverse | 5-component weighted rank composite (alpha036 paper formula) | ok |
| alpha039 | industry_neutral | reverse | (-rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20, 9))))) * (1 + ra... | ok |
| alpha049 | industry_neutral | reverse | If close-acceleration < -0.1 then 1 else -delta(close,1) | ok |
| alpha050 | industry_neutral | reverse | -ts_max(rank(corr(rank(volume), rank(vwap), 5)), 5) | ok |
| alpha058 | industry_neutral | reverse | -ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), volume, 4), 8), 6) | ok |
| alpha059 | industry_neutral | reverse | -ts_rank(decay_linear(corr(IndNeutralize(vwap_blend,industry), volume, 4), 16... | ok |
| alpha062 | industry_neutral | reverse | rank(corr(vwap,sum(adv20,22),10)) < rank(open-rank vs midpoint inequality) | ok |
| alpha063 | industry_neutral | reverse | Diff of decay-linear(delta(IndNeutralize(close,industry),2),8) and decay-line... | ok |
| alpha066 | industry_neutral | reverse | -(rank(decay_linear(delta(vwap,4),7)) + ts_rank(decay_linear((low-vwap)/(open... | ok |
| alpha067 | industry_neutral | reverse | (rank(high-ts_min(high,2)) ^ rank(corr(IndNeutralize(vwap,industry), IndNeutr... | ok |
| alpha069 | industry_neutral | reverse | (rank(ts_max(delta(IndNeutralize(vwap,industry),3),5)) ^ ts_rank(corr(close-v... | ok |
| alpha070 | industry_neutral | reverse | (rank(delta(vwap,1)) ^ ts_rank(corr(IndNeutralize(close,industry), adv50, 18)... | ok |
| alpha076 | industry_neutral | reverse | -max(rank(decay_linear(delta(vwap,1),12)), ts_rank(decay_linear(ts_rank(corr(... | ok |
| alpha079 | industry_neutral | reverse | rank(delta(IndNeutralize(close-open blend,industry),1)) < rank(corr(ts_rank(v... | ok |
| alpha080 | industry_neutral | reverse | (rank(sign(delta(IndNeutralize(open-high blend,industry),4))) ^ ts_rank(corr(... | ok |
| alpha082 | industry_neutral | reverse | -min(rank(decay_linear(delta(open,1),15)), ts_rank(decay_linear(corr(IndNeutr... | ok |
| alpha086 | industry_neutral | reverse | (ts_rank(corr(close,sum(adv20,15),6),20) < rank((open+close)-(vwap+open))) * -1 | ok |
| alpha087 | industry_neutral | reverse | -max(rank(decay_linear(delta(close-vwap blend,2),3)), ts_rank(decay_linear(ab... | ok |
| alpha089 | industry_neutral | reverse | ts_rank(decay_linear(corr(low blend, adv10, 7), 6), 4) - ts_rank(decay_linear... | ok |
| alpha090 | industry_neutral | reverse | (rank(close-ts_max(close,5)) ^ ts_rank(corr(IndNeutralize(adv40,sub_industry)... | ok |
| alpha091 | industry_neutral | reverse | (ts_rank(decay_linear(decay_linear(corr(IndNeutralize(close,industry), volume... | ok |
| alpha093 | industry_neutral | reverse | ts_rank(decay_linear(corr(IndNeutralize(vwap,industry), adv81, 17), 20), 8) /... | ok |
| alpha096 | industry_neutral | reverse | -max(ts_rank(decay_linear(corr(rank(vwap),rank(volume),4),4),8), ts_rank(deca... | ok |
| alpha097 | industry_neutral | reverse | (rank(decay_linear(delta(IndNeutralize(low-vwap blend,industry),3),20)) - ts_... | ok |
| alpha098 | industry_neutral | reverse | rank(decay_linear(corr(vwap,sum(adv5,26),5),7)) - rank(decay_linear(ts_rank(t... | ok |
| alpha100 | industry_neutral | reverse | Sub-industry-neutralised body*volume signal minus scale(IndNeutralize(corr(cl... | ok |
| alpha024 | cap_weighted | reverse | If delta(sma(close,100),100)/delay(close,100) <= 0.05 then -(close-ts_min(clo... | ok |
| alpha056 | cap_weighted | reverse | -rank(sum(returns,10)/sum(sum(returns,2),3)) * rank(returns * cap) | ok |
| alpha021 | adv_extended | reverse | Volatility-vs-momentum regime switch using sma(close,8)+/-std(close,8) with v... | ok |
| alpha061 | adv_extended | reverse | rank(vwap-ts_min(vwap,16)) < rank(corr(vwap, adv180, 18)) | ok |
| alpha064 | adv_extended | reverse | (rank(corr(sum(open-low blend,13), sum(adv120,13), 17)) < rank(delta(midpoint... | ok |
| alpha075 | adv_extended | reverse | rank(corr(vwap,volume,4)) < rank(corr(rank(low), rank(adv50), 12)) | ok |
