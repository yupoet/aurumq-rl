import Link from "next/link";
import { notFound } from "next/navigation";
import {
  readBacktest,
  readBacktestSeries,
  readMetricsJsonl,
  readSummary,
} from "@/lib/runs";
import { listSeriesKeys, pickSeries } from "@/lib/jsonl";
import { MetricChart } from "@/components/MetricChart";
import {
  BacktestSummary,
  type BacktestData,
} from "@/components/BacktestSummary";
import {
  BacktestSeriesPanel,
  type BacktestSeriesData,
} from "@/components/BacktestSeriesPanel";

export const dynamic = "force-dynamic";

const PRIMARY_KEYS = [
  "rollout/ep_rew_mean",
  "train/loss",
  "train/policy_gradient_loss",
  "train/value_loss",
  "train/explained_variance",
  "train/approx_kl",
  "train/clip_fraction",
  "time/fps",
];

export default async function Page({
  params,
}: {
  params: Promise<{ id: string[] }>;
}) {
  const { id } = await params;
  const decoded = id.map((s) => decodeURIComponent(s)).join("/");

  const summary = await readSummary(decoded);
  if (!summary) notFound();
  const metrics = await readMetricsJsonl(decoded);
  const backtest = (await readBacktest(decoded)) as BacktestData | null;
  const series = (await readBacktestSeries(decoded)) as BacktestSeriesData | null;
  const allKeys = listSeriesKeys(metrics);
  const charts = PRIMARY_KEYS.filter((k) => allKeys.includes(k));

  return (
    <main className="mx-auto max-w-6xl px-6 py-8 space-y-6">
      <header>
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-300">
          ← all runs
        </Link>
        <h1 className="font-mono text-xl mt-2">{decoded}</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {summary.algorithm} · {summary.total_timesteps?.toLocaleString()} steps · reward=
          {summary.reward_type ?? "—"} · universe=
          {summary.universe_filter ?? "—"}
        </p>
      </header>

      {backtest && <BacktestSummary data={backtest} />}
      {series && backtest && (
        <BacktestSeriesPanel data={series} realizedSharpe={backtest.top_k_sharpe} />
      )}

      <section>
        <h2 className="text-lg font-semibold mb-3">Training curves</h2>
        {charts.length === 0 ? (
          <p className="text-sm text-zinc-500">
            No metrics with primary keys yet. {metrics.length} rows total.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {charts.map((k) => (
              <MetricChart key={k} title={k} data={pickSeries(metrics, k)} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
