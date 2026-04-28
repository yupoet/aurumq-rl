"use client";

import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

export interface BacktestSeriesData {
  dates: string[];
  ic: number[];
  top_k_returns: number[];
  equity_curve: number[];
  random_baseline_sharpes: number[];
}

export function BacktestSeriesPanel({
  data,
  realizedSharpe,
}: {
  data: BacktestSeriesData;
  realizedSharpe: number;
}) {
  const points = data.dates.map((d, i) => ({
    date: d,
    ic: data.ic[i] ?? 0,
    equity: data.equity_curve[i] ?? 1,
  }));

  const histogramBins = 24;
  const sharpes = data.random_baseline_sharpes;
  const minS = Math.min(...sharpes);
  const maxS = Math.max(...sharpes);
  const step = (maxS - minS) / histogramBins || 1;
  const histogram = Array.from({ length: histogramBins }, (_, i) => ({
    bin: minS + (i + 0.5) * step,
    count: 0,
  }));
  for (const s of sharpes) {
    const idx = Math.min(
      histogramBins - 1,
      Math.max(0, Math.floor((s - minS) / step))
    );
    histogram[idx].count += 1;
  }

  return (
    <section className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5">
      <h2 className="text-lg font-semibold mb-4">Backtest deep-dive</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-xs text-zinc-500 mb-1">IC over time</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} hide />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="ic"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={1.2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-xs text-zinc-500 mb-1">
            Equity curve (top-K, equal-weight)
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} hide />
                <YAxis
                  tick={{ fontSize: 10 }}
                  domain={["auto", "auto"]}
                />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <ReferenceLine y={1} stroke="#888" strokeDasharray="3 3" />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.15}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="md:col-span-2">
          <h3 className="text-xs text-zinc-500 mb-1">
            Random-baseline Sharpe distribution (n=
            {data.random_baseline_sharpes.length})
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis
                  dataKey="bin"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12 }}
                  labelFormatter={(v) => `Sharpe ${Number(v).toFixed(2)}`}
                />
                <ReferenceLine
                  x={realizedSharpe}
                  stroke="#ef4444"
                  strokeWidth={2}
                  label={{
                    value: `realized ${realizedSharpe.toFixed(2)}`,
                    fill: "#ef4444",
                    fontSize: 10,
                    position: "top",
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="#8b5cf6"
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
