import Link from "next/link";
import { notFound } from "next/navigation";
import {
  isRunLive,
  metricsJsonlSize,
  readBacktest,
  readBacktestSeries,
  readFactorImportance,
  readGpuJsonl,
  readMetricsJsonl,
  readSummary,
} from "@/lib/runs";
import { LiveCurves } from "@/components/LiveCurves";
import {
  BacktestSummary,
  type BacktestData,
} from "@/components/BacktestSummary";
import {
  BacktestSeriesPanel,
  type BacktestSeriesData,
} from "@/components/BacktestSeriesPanel";
import { GpuMetricsPanel } from "@/components/GpuMetricsPanel";
import { FactorImportancePanel } from "@/components/FactorImportancePanel";
import type { FactorImportance } from "@/lib/runs-shared";

export const dynamic = "force-dynamic";

interface MetricRow {
  timestep: number;
  [key: string]: number | string;
}

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ id: string[] }>;
}) {
  const { id } = await params;
  const decoded = id.map((s) => decodeURIComponent(s)).join("/");

  const [summary, metrics, backtest, series, gpu, fi, live, initialOffset] =
    await Promise.all([
      readSummary(decoded),
      readMetricsJsonl(decoded),
      readBacktest(decoded) as Promise<BacktestData | null>,
      readBacktestSeries(decoded) as Promise<BacktestSeriesData | null>,
      readGpuJsonl(decoded),
      readFactorImportance(decoded) as Promise<FactorImportance | null>,
      isRunLive(decoded),
      metricsJsonlSize(decoded),
    ]);

  if (!summary && metrics.length === 0 && !backtest) {
    notFound();
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-8 space-y-6">
      <header>
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-300">
          ← back
        </Link>
        <h1 className="text-xl font-semibold mt-2 font-mono">{decoded}</h1>
        {summary && (
          <dl className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-2 text-sm">
            <Stat label="算法" value={summary.algorithm} />
            <Stat
              label="步数"
              value={summary.total_timesteps?.toLocaleString()}
            />
            <Stat label="reward" value={summary.reward_type ?? "—"} />
            <Stat label="universe" value={summary.universe_filter ?? "—"} />
            <Stat label="股票" value={summary.n_stocks?.toString() ?? "—"} />
            <Stat label="因子" value={summary.n_factors?.toString() ?? "—"} />
            <Stat label="top-k" value={summary.top_k?.toString() ?? "—"} />
            <Stat label="env" value={summary.env_type ?? "—"} />
          </dl>
        )}
      </header>

      <LiveCurves
        id={decoded}
        initialRows={metrics as unknown as MetricRow[]}
        initialOffset={initialOffset}
        isLive={live}
      />

      {backtest != null && <BacktestSummary data={backtest} />}
      {series != null && backtest != null && (
        <BacktestSeriesPanel data={series} realizedSharpe={backtest.top_k_sharpe} />
      )}
      {gpu.length > 0 && <GpuMetricsPanel data={gpu} />}
      {fi != null && <FactorImportancePanel data={fi} />}
    </main>
  );
}

function Stat({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="font-mono">{value ?? "—"}</dd>
    </div>
  );
}
