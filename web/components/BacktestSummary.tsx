export interface BacktestData {
  ic: number;
  ic_ir: number;
  top_k_sharpe: number;
  top_k_cumret: number;
  random_baseline: {
    mean_sharpe: number;
    p05_sharpe?: number;
    p50_sharpe: number;
    p95_sharpe: number;
  };
  n_dates: number;
  n_stocks: number;
  top_k: number;
}

export function BacktestSummary({ data }: { data: BacktestData }) {
  const beats = data.top_k_sharpe > data.random_baseline.p95_sharpe;
  return (
    <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5">
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-lg font-semibold">Backtest</h2>
        <span
          className={`text-xs rounded px-2 py-0.5 ${
            beats
              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
              : "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
          }`}
        >
          {beats ? "beats random p95" : "within random band"}
        </span>
      </div>
      <dl className="grid grid-cols-4 gap-x-4 gap-y-3">
        <Stat label="IC" value={data.ic.toFixed(4)} />
        <Stat label="IC IR" value={data.ic_ir.toFixed(3)} />
        <Stat
          label={`Top-${data.top_k} Sharpe`}
          value={data.top_k_sharpe.toFixed(3)}
        />
        <Stat
          label="Top-K cum.ret"
          value={(data.top_k_cumret * 100).toFixed(2) + "%"}
        />
        <Stat
          label="random p50"
          value={data.random_baseline.p50_sharpe.toFixed(3)}
        />
        <Stat
          label="random p95"
          value={data.random_baseline.p95_sharpe.toFixed(3)}
        />
        <Stat label="dates" value={data.n_dates.toString()} />
        <Stat label="stocks" value={data.n_stocks.toString()} />
      </dl>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="text-base font-mono">{value}</dd>
    </div>
  );
}
