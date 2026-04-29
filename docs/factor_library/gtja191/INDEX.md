# GTJA191 Factor Library Index (191 factors)

| ID | Category | Direction | Description | Quality |
|---|---|---|---|---|
| gtja_001 | volume_price | reverse | Volume-change rank vs intraday return rank correlation, 6d, negated | ok |
| gtja_002 | mean_reversion | reverse | One-day delta of normalised intraday mid-range, negated | ok |
| gtja_003 | volume_price | normal | 6d sum of close-vs-extreme conditional flow | ok |
| gtja_004 | mean_reversion | normal | Trend regime + volume gate ternary signal | ok |
| gtja_005 | volume_price | reverse | Negated 3d max of 5d ts-rank corr (volume vs high) | warn |
| gtja_006 | momentum | reverse | Rank of sign of 4d weighted (open*0.85 + high*0.15) delta, negated | ok |
| gtja_007 | volume_price | normal | (Rank max + Rank min) of (vwap-close,3) × Rank(volume delta,3) | ok |
| gtja_008 | momentum | reverse | Negated rank of 4d delta of HL10+VWAP80 weighted price (Daic115 parity) | ok |
| gtja_009 | volume_price | normal | EWMA(7,2) of mid-price acceleration weighted by (H-L)/Volume | ok |
| gtja_010 | volatility | reverse | Rank of MAX5(((ret<0?std20:close))^2) — Daic115 floor-by-5 parity | ok |
| gtja_011 | volume_price | normal | 6d sum of normalised mid-range × volume | ok |
| gtja_012 | mean_reversion | reverse | Rank(O-MA10VWAP) × -Rank(\|C-VWAP\|) | ok |
| gtja_013 | mean_reversion | normal | Geometric mean of (H, L) minus VWAP | ok |
| gtja_014 | momentum | normal | 5d price change (close - delay(close,5)) | ok |
| gtja_015 | momentum | normal | Overnight gap return: open / prior close - 1 | ok |
| gtja_016 | volume_price | reverse | -1 × TSMAX(rank(corr(rank(vol), rank(vwap), 5)), 5) | ok |
| gtja_017 | momentum | reverse | Rank(vwap - max15(vwap)) ^ delta(close, 5) | ok |
| gtja_018 | momentum | normal | 5d close ratio: close / delay(close, 5) | ok |
| gtja_019 | mean_reversion | normal | Asymmetric 6d vwap change ratio | ok |
| gtja_020 | momentum | normal | 6d vwap % change × 100 | ok |
| gtja_021 | momentum | reverse | Rolling 6d slope of MEAN(close,6) vs sequence (regbeta vs row-index) | warn |
| gtja_022 | mean_reversion | reverse | EWMA(12,1) of (close mean-detrend - 3d-lag) over 6d window | ok |
| gtja_023 | volatility | normal | Up-day STD share over 20d (×100), smoothed via SMA(20,1) | ok |
| gtja_024 | momentum | normal | EWMA(5,1) of 5d price change | ok |
| gtja_025 | momentum | reverse | -Rank(close7d-delta) × Rank(EWMA volume ratio) × (1+Rank(150d ret sum)) | ok |
| gtja_026 | mean_reversion | normal | (MEAN(close,12) - close) + 200d corr(vwap, delay(close,5)) | ok |
| gtja_027 | momentum | normal | EWMA(12,1) of 3d+6d % momentum sum (Daic115 SMA-substituted-WMA) | ok |
| gtja_028 | momentum | normal | KDJ-style 9d stochastic with two SMA(3,1) layers | ok |
| gtja_029 | volume_price | normal | 6d % change × log(volume) (Daic115 log-vol substitution) | ok |
| gtja_030 | volatility | normal | STUB — Fama-French residual^2 WMA (Daic115 unfinished) | broken |
| gtja_031 | mean_reversion | reverse | 12d MA-distance ratio × 100 | ok |
| gtja_032 | volume_price | reverse | -SUM(rank(corr(rank-H, rank-Vol, 3)), 3) | ok |
| gtja_033 | volume_price | normal | Low-min change × ret-spread rank × turnover-proxy rank (amount/cap) | warn |
| gtja_034 | mean_reversion | reverse | MEAN(close, 12) / close | ok |
| gtja_035 | volume_price | reverse | Min of two rank(decay) arms (open delta + vol-weighted-price corr) negated | ok |
| gtja_036 | volume_price | normal | Rank of 2d sum of corr(rank-volume, rank-vwap, 6) | ok |
| gtja_037 | momentum | reverse | -Rank of 10d acceleration of (sum(open,5) × sum(ret,5)) | ok |
| gtja_038 | mean_reversion | reverse | Conditional negated 2d high-delta when high>MEAN(high,20) | ok |
| gtja_039 | momentum | reverse | -(rank-decay close-delta - rank-decay long-vol corr) | ok |
| gtja_040 | volume_price | normal | 26d up-volume / down-volume ratio × 100 | ok |
| gtja_041 | momentum | reverse | -Rank(MAX(DELTA(VWAP, 3), 5)) | ok |
| gtja_042 | volume_price | reverse | -RANK(STD(H, 10)) × CORR(H, V, 10) | ok |
| gtja_043 | volume_price | normal | 6d signed-volume sum (vwap-anchored direction) | ok |
| gtja_044 | volume_price | normal | TS-rank decay corr(low, MA(V,10), 7) + TS-rank decay delta(VWAP, 3) | ok |
| gtja_045 | volume_price | reverse | Rank(C0.6+O0.4 delta) × CORR(VWAP, MEAN(V, 150), 15) | ok |
| gtja_046 | mean_reversion | reverse | (MA3 + MA6 + MA12 + MA24) / (4 × close) | ok |
| gtja_047 | mean_reversion | reverse | Smoothed inverse-RSV: SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 9, 1) | ok |
| gtja_048 | momentum | reverse | -Rank(3-day sign sum) × SUM(V,5)/SUM(V,20) | ok |
| gtja_049 | volatility | reverse | Down-day range share over 12d | ok |
| gtja_050 | volatility | reverse | Down-share - Up-share asymmetric range ratio over 12d | warn |
| gtja_051 | volatility | reverse | Down-share asymmetric range ratio over 12d | warn |
| gtja_052 | momentum | normal | 26d typical-price up/down pressure ratio × 100 | ok |
| gtja_053 | momentum | normal | % of up-days over 12d × 100 | ok |
| gtja_054 | volatility | reverse | -Rank(STD(\|C-O\|+(C-O), 10) + CORR(C, O, 10)) | ok |
| gtja_055 | momentum | normal | 20d sum of TR-normalised acceleration × max(\|H-C-1\|,\|L-C-1\|) | warn |
| gtja_056 | volume_price | reverse | Rank-inequality: open-min vs rank-corr^5 (returns 0/1) | ok |
| gtja_057 | momentum | normal | 3-period EWMA of 9-period stochastic %K | ok |
| gtja_058 | momentum | normal | % of up-days over 20d × 100 (vwap-anchored) | ok |
| gtja_059 | volume_price | reverse | 20d sum of close-vs-extreme conditional flow (vwap-anchored) | ok |
| gtja_060 | volume_price | normal | 20d sum of normalised mid-range × volume | ok |
| gtja_061 | volume_price | reverse | MAX of rank(decay-VWAP-delta), rank(decay-rank(corr(L, MA(V,80), 8))) | ok |
| gtja_062 | volume_price | reverse | -CORR(HIGH, RANK(turn proxy=amount/cap), 5) | warn |
| gtja_063 | momentum | normal | 6-day RSI (vwap-anchored) | ok |
| gtja_064 | volume_price | reverse | MAX of two rank(decay-corr) arms (vwap-vol, close-MA-vol) | ok |
| gtja_065 | mean_reversion | reverse | MEAN(close, 6) / close | ok |
| gtja_066 | mean_reversion | reverse | (close - MA6) / MA6 × 100 | ok |
| gtja_067 | momentum | normal | 24-day RSI | ok |
| gtja_068 | volume_price | normal | EWMA(15,2) of mid-price acceleration × (H-L)/V | ok |
| gtja_069 | momentum | normal | DTM/DBM 20-day asymmetric momentum ratio | warn |
| gtja_070 | volatility | normal | 6-day std of amount | ok |
| gtja_071 | mean_reversion | reverse | (close - MA24) / MA24 × 100 | ok |
| gtja_072 | mean_reversion | reverse | SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6)) × 100, 15, 1) | ok |
| gtja_073 | volume_price | reverse | -TS_RANK(decay-decay-corr(C,V)) - RANK(decay-corr(VWAP, MA30(V))) | warn |
| gtja_074 | volume_price | normal | Rank-corr-sum-weighted-prices + Rank-corr-rank(VWAP, V) | ok |
| gtja_075 | momentum | normal | Up-day count ratio vs CS-mean-return-as-benchmark down-days (50d) | ok |
| gtja_076 | volatility | reverse | STD(\|ret\|/V, 20) / MEAN(\|ret\|/V, 20) | ok |
| gtja_077 | volume_price | reverse | MIN of two rank(decay) arms (synthetic-mid-vs-VWAP, mid-MA40V-corr) | ok |
| gtja_078 | mean_reversion | normal | CCI-style typical-price oscillator (12d) | ok |
| gtja_079 | momentum | normal | 12-day RSI | ok |
| gtja_080 | volume_price | normal | (V - DELAY(V, 5)) / DELAY(V, 5) × 100 | ok |
| gtja_081 | volume_price | normal | EWMA(21, 2) of volume | ok |
| gtja_082 | mean_reversion | reverse | SMA((TSMAX(H,6)-C)/(TSMAX(H,6)-TSMIN(L,6))×100, 20, 1) | ok |
| gtja_083 | volume_price | reverse | -RANK(COV(RANK(H), RANK(V), 5)) | ok |
| gtja_084 | volume_price | normal | 20d signed-volume sum (close-direction) | ok |
| gtja_085 | momentum | reverse | TSRANK(V/MA20V, 20) × TSRANK(-DELTA(C, 7), 8) | ok |
| gtja_086 | momentum | reverse | 20/10/0 close-acceleration regime ternary | ok |
| gtja_087 | volume_price | reverse | -(rank-decay-vwap-delta + TS-rank-decay-asymmetric-spread) | ok |
| gtja_088 | momentum | normal | 20-day % change × 100 | ok |
| gtja_089 | momentum | normal | MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27)) | ok |
| gtja_090 | volume_price | reverse | -RANK(CORR(RANK(VWAP), RANK(V), 5)) | ok |
| gtja_091 | volume_price | reverse | -RANK(C - TSMAX(C, 5)) × RANK(CORR(MEAN(V, 40), L, 5)) | ok |
| gtja_092 | volume_price | reverse | -MAX of rank-decay-delta + TS-rank-decay-\|corr\| arms | ok |
| gtja_093 | volatility | normal | 20d sum of conditional max(O-L, O-O-1) when O<O-1 | ok |
| gtja_094 | volume_price | normal | 30d signed-volume sum | ok |
| gtja_095 | volatility | normal | 20-day std of amount | ok |
| gtja_096 | momentum | normal | Double-smoothed stochastic %K (3,1)(3,1) | ok |
| gtja_097 | volatility | normal | 10-day std of volume | ok |
| gtja_098 | momentum | reverse | 100d MA-acceleration regime ternary | ok |
| gtja_099 | volume_price | reverse | -RANK(COV(RANK(C), RANK(V), 5)) | ok |
| gtja_100 | volatility | normal | 20-day std of volume | ok |
| gtja_101 | correlation | reverse | VWAP-volume corr ranked < volume-mean corr ranked | ok |
| gtja_102 | volume | normal | Volume RSI — SMA(max(dV,0))/SMA(\|dV\|)*100 | ok |
| gtja_103 | momentum | normal | (20-LOWDAY(LOW,20))/20*100 — recency-of-low oscillator | ok |
| gtja_104 | correlation | reverse | -1 * delta(corr(high,vol,5),5) * rank(std(close,20)) | ok |
| gtja_105 | correlation | reverse | -1 * corr(rank(open), rank(volume), 10) | ok |
| gtja_106 | momentum | normal | 20-day close momentum | ok |
| gtja_107 | momentum | reverse | -rank(o-prev_h)*rank(o-prev_c)*rank(o-prev_l) | ok |
| gtja_108 | correlation | reverse | (rank(high-min(high,2)) ^ rank(corr(vwap, ma(v,120), 6))) * -1 | ok |
| gtja_109 | volatility | normal | SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2) | ok |
| gtja_110 | momentum | normal | 20d up-move-vs-down-move pressure ratio | ok |
| gtja_111 | volume | normal | Volume * intraday position SMA-differential (11-4) | ok |
| gtja_112 | momentum | normal | Chande momentum oscillator over 12 days | ok |
| gtja_113 | correlation | reverse | -rank(sum(delay(c,5),20)/20)*corr(c,v,20)*rank(corr(sum_c5,sum_c20,20)) | ok |
| gtja_114 | volatility | normal | rank(delay(rng/ma5,2))*rank(rank(v)) / ((rng/ma5)/(vwap-c)) | ok |
| gtja_115 | correlation | normal | rank(corr(0.9H+0.1C, ma(v,30),10)) ^ rank(corr(tsr_HL2_4, tsr_v_10, 7)) | ok |
| gtja_116 | momentum | normal | 20-day rolling OLS slope of CLOSE on time-index | ok |
| gtja_117 | momentum | normal | tsrank(v,32) * (1-tsrank(c+h-l,16)) * (1-tsrank(ret,32)) | ok |
| gtja_118 | volatility | normal | 20-day open-relative range skew (h-o vs o-l) | ok |
| gtja_119 | correlation | normal | decay-linear of corr(vwap, sum_mean_v) minus decay-linear tsrank-min-corr(ran... | ok |
| gtja_120 | volume | normal | rank(vwap-close) / rank(vwap+close) | ok |
| gtja_121 | correlation | reverse | (rank(vwap-min(vwap,12)) ^ tsrank(corr(tsr_vwap_20, tsr_mv60_2, 18), 3)) * -1 | warn |
| gtja_122 | momentum | normal | Triple-SMA log-close 1-day delta ratio | ok |
| gtja_123 | correlation | reverse | (rank(corr_HL2_sum_mv60) < rank(corr_low_vol)) * -1 | ok |
| gtja_124 | volume | normal | (close-vwap) / decay_linear(rank(ts_max(close,30)), 2) | ok |
| gtja_125 | correlation | normal | rank(decay_linear(corr_vwap_mv80,20)) / rank(decay_linear(delta(0.5C+0.5VWAP,... | ok |
| gtja_126 | price | normal | (close + high + low) / 3 — typical price | ok |
| gtja_127 | volatility | normal | sqrt(mean((100*(c-max(c,12))/max(c,12))^2, 12)) | ok |
| gtja_128 | volume | normal | 14-day money flow index (MFI) on typical price | ok |
| gtja_129 | momentum | reverse | 12-day downside move cumsum | ok |
| gtja_130 | correlation | normal | rank(decay_linear(corr(HL2,mv40,9),10)) / rank(decay_linear(corr(rk_vwap,rk_v... | ok |
| gtja_131 | correlation | normal | rank(delta(vwap,1)) ^ tsrank(corr(close, mv50, 18), 18) | warn |
| gtja_132 | volume | normal | 20-day mean amount (turnover proxy) | ok |
| gtja_133 | momentum | normal | (20-highday(high,20))/20*100 - (20-lowday(low,20))/20*100 | ok |
| gtja_134 | volume | normal | (c-prev_c12)/prev_c12 * volume | ok |
| gtja_135 | momentum | normal | SMA(delay(close/delay(close,20),1), 20, 1) | ok |
| gtja_136 | momentum | reverse | -rank(delta(ret,3)) * corr(open, volume, 10) | ok |
| gtja_137 | volatility | normal | Wilders'-style TR-normalised close move | ok |
| gtja_138 | correlation | reverse | (rank(dl(delta(0.7L+0.3VWAP,3),20)) - tsr(dl(tsr_corr_ll_mv60),16),7)) * -1 | ok |
| gtja_139 | correlation | reverse | -corr(open, volume, 10) | ok |
| gtja_140 | correlation | normal | min(rank(dl(rank_OL_HC,8)), tsr(dl(corr(tsr_c8,tsr_mv60_20,8),7),3)) | ok |
| gtja_141 | correlation | reverse | -rank(corr(rank(high), rank(mean(v,15)), 9)) | ok |
| gtja_142 | momentum | reverse | -rank(tsr_c10) * rank(d2_close) * rank(tsr_vrel_5) | ok |
| gtja_143 | momentum | normal | STUB — recursive SELF reference, formula ambiguous | broken |
| gtja_144 | volatility | normal | 20d down-day mean(\|ret\|/log(amount)) | ok |
| gtja_145 | volume | normal | (mean(v,9) - mean(v,26)) / mean(v,12) * 100 | ok |
| gtja_146 | momentum | normal | mean(ret-sma(ret,61,2),20) * (ret-sma) / SMA((ret-(ret-sma))^2,61,2) | ok |
| gtja_147 | momentum | normal | 12-day rolling OLS slope of MEAN(close,12) on time-index | ok |
| gtja_148 | correlation | reverse | (rank(corr(o, sum(mv60,9), 6)) < rank(o-tsmin(o,14))) * -1 | ok |
| gtja_149 | benchmark | normal | 252d beta vs cross-section-mean-return benchmark proxy (CSI300 wiring is Phas... | ok |
| gtja_150 | volume | normal | typical price * log(volume) (Daic115 variant) | ok |
| gtja_151 | momentum | normal | SMA(close - delay(close,20), 20, 1) | warn |
| gtja_152 | momentum | normal | sma(mean(part,12)-mean(part,26),9,1) where part is delayed sma(c/c9,9,1) | ok |
| gtja_153 | momentum | normal | Average of 4 different moving averages (BBI) | ok |
| gtja_154 | correlation | normal | sign((vwap-min(vwap,16)) < corr(vwap, mv180, 18)) | ok |
| gtja_155 | volume | normal | Volume MACD (13,27,10) histogram | ok |
| gtja_156 | momentum | reverse | -max(rank(dl(d_vwap_5,3)), rank(dl(-d_inner_2/inner,3))) | ok |
| gtja_157 | momentum | normal | ts_min(rank(rank(log(sum(ts_min(rank(rank(-rank(d(c-1,5)))),2),1)))),5) + tsr... | ok |
| gtja_158 | volatility | normal | (H - L) / C — high-low spread normalised by close (SMA terms cancel) | ok |
| gtja_159 | momentum | normal | 3-window cumulative range-position oscillator (KDJ-like) | ok |
| gtja_160 | volatility | normal | EWMA of down-day std(close,20) | ok |
| gtja_161 | volatility | normal | 12-day mean true range | ok |
| gtja_162 | momentum | normal | Stochastic-RSI: (RSI-min(RSI,12)) / (max(RSI,12)-min(RSI,12)) | ok |
| gtja_163 | momentum | reverse | rank(-ret * mv20 * vwap * (high-close)) | ok |
| gtja_164 | momentum | normal | SMA((rec - min(rec,12) / (h-l) * 100), 13, 2) — Daic115 parsing | ok |
| gtja_165 | volatility | normal | MAX(SUMAC(close-ma48)) - MIN(SUMAC(close-ma48)) / STD(close,48) — Daic115 SUM... | warn |
| gtja_166 | momentum | normal | 5*sum(centered_ret,20)/(sum(ma_ret_squared,20))^1.5 — Daic115 errata simplifi... | warn |
| gtja_167 | momentum | normal | 12-day cumulative up-move | ok |
| gtja_168 | volume | reverse | -volume / mean(volume,20) | ok |
| gtja_169 | momentum | normal | SMA(mean(delay(SMA(dC,9,1),1),12) - mean(delay(SMA(dC,9,1),1),26), 10, 1) | ok |
| gtja_170 | volume | normal | rank(1/c)*v/mv20 * (h*rank(h-c))/(sum(h,5)/5) - rank(vwap-d_vwap_5) | ok |
| gtja_171 | momentum | reverse | -(low-close)*open^5 / ((close-high)*close^5) | ok |
| gtja_172 | momentum | normal | 6-day mean of DI-difference oscillator (DX) | ok |
| gtja_173 | momentum | normal | 3*sma(c,13,2) - 2*sma^2(c,13,2) + sma^3(log(c),13,2) | ok |
| gtja_174 | volatility | normal | EWMA of up-day std(close,20) | ok |
| gtja_175 | volatility | normal | 6-day mean true range | ok |
| gtja_176 | correlation | normal | corr(rank(stoch_K_12), rank(volume), 6) | ok |
| gtja_177 | momentum | normal | (20-highday(high,20))/20*100 — recency-of-high oscillator | ok |
| gtja_178 | volume | normal | (c-prev_c)/prev_c * volume | ok |
| gtja_179 | correlation | normal | rank(corr(vwap,v,4)) * rank(corr(rank(low), rank(mv50), 12)) | ok |
| gtja_180 | momentum | reverse | Conditional: -tsrank(\|d_c7\|,60)*sign(d_c7) when V > MV20 else -V | ok |
| gtja_181 | benchmark | normal | Skew-adjusted return-vs-benchmark — CSI300 proxied via CS mean (Phase D) | warn |
| gtja_182 | benchmark | normal | 20d frac of days stock & benchmark co-move — CS-mean OHLC proxy (Phase D) | ok |
| gtja_183 | volatility | normal | MAX(SUMAC(c-ma24,24)) - MIN(SUMAC(c-ma24,24)) / STD(c,24) | warn |
| gtja_184 | correlation | normal | rank(corr(delay(o-c,1), close, 200)) + rank(o-c) | ok |
| gtja_185 | momentum | reverse | rank(-(1 - open/close)^2) | ok |
| gtja_186 | momentum | normal | (mean(DX,6) + delay(mean(DX,6),6)) / 2 — smoothed ADX-style | ok |
| gtja_187 | momentum | normal | 20-day cumulative open-up-gap or daily upper-body-range | ok |
| gtja_188 | volatility | normal | ((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100 | ok |
| gtja_189 | volatility | normal | 6-day mean abs deviation from MA6 | ok |
| gtja_190 | momentum | normal | log((cnt_p1>p2-1)*SUMIF((p1-p2)^2,20,p1<p2)/(cnt_p1<p2 * SUMIF(.,20,p1>p2))) | ok |
| gtja_191 | volume | normal | corr(mv20, low, 5) + (h+l)/2 - close | warn |
