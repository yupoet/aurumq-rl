import Link from "next/link";
import type { RunListEntry } from "@/lib/runs";

const fmt = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

export function RunCard({ run }: { run: RunListEntry }) {
  const s = run.summary;
  return (
    <Link
      href={`/runs/${encodeURIComponent(run.id)}`}
      className="block rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 hover:border-zinc-400 dark:hover:border-zinc-600 transition"
    >
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-mono text-sm">{run.id}</h3>
        <time className="text-xs text-zinc-500">
          {fmt.format(run.modifiedAt)}
        </time>
      </div>
      {s && (
        <dl className="mt-2 grid grid-cols-3 gap-x-3 gap-y-1 text-xs text-zinc-600 dark:text-zinc-400">
          <Stat label="算法" value={s.algorithm} />
          <Stat
            label="步数"
            value={s.total_timesteps?.toLocaleString()}
          />
          <Stat label="reward" value={s.reward_type ?? "—"} />
          <Stat label="universe" value={s.universe_filter ?? "—"} />
          <Stat label="股票" value={s.n_stocks?.toString()} />
          <Stat label="因子" value={s.n_factors?.toString()} />
        </dl>
      )}
      <div className="mt-3 flex gap-1.5 text-[10px] uppercase tracking-wider">
        <Badge ok={run.hasMetrics}>metrics</Badge>
        <Badge ok={run.hasOnnx}>onnx</Badge>
        <Badge ok={run.hasBacktest}>backtest</Badge>
      </div>
    </Link>
  );
}

function Stat({ label, value }: { label: string; value?: string }) {
  return (
    <div>
      <dt className="text-zinc-500">{label}</dt>
      <dd>{value ?? "—"}</dd>
    </div>
  );
}

function Badge({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <span
      className={`px-1.5 py-0.5 rounded ${
        ok
          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"
          : "bg-zinc-100 text-zinc-500 dark:bg-zinc-900 dark:text-zinc-600"
      }`}
    >
      {children}
    </span>
  );
}
