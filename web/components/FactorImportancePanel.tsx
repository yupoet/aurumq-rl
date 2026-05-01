"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FactorImportance } from "@/lib/runs-shared";

const COLORS = [
  "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
  "#14b8a6", "#a855f7", "#f97316", "#22c55e", "#0ea5e9", "#eab308",
];

export function FactorImportancePanel({ data }: { data: FactorImportance }) {
  const groupRows = Object.entries(data.importance_per_group)
    .map(([prefix, m]) => ({
      prefix,
      ic_drop_mean: m.ic_drop_mean,
      ic_drop_std: m.ic_drop_std,
      sharpe_drop_mean: m.sharpe_drop_mean,
      n_factors: m.n_factors,
      saliency_mean: m.saliency_mean ?? 0,
    }))
    .sort((a, b) => b.ic_drop_mean - a.ic_drop_mean);

  // Top 12 most-salient individual factors for the second chart
  const topFactors = Object.entries(data.saliency_per_factor)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, sal]) => ({ name, saliency: sal }));

  return (
    <section className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5 space-y-6">
      <header className="flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">Factor importance</h2>
        <span className="text-xs text-zinc-500 font-mono">{data.method}</span>
      </header>

      <div className="min-w-0">
        <h3 className="text-xs text-zinc-500 mb-1">Per-group IC drop (permutation)</h3>
        <ResponsiveContainer width="100%" height={Math.max(180, groupRows.length * 24)}>
          <BarChart data={groupRows} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 32 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="prefix" tick={{ fontSize: 11, fontFamily: "monospace" }} width={64} />
            <Tooltip contentStyle={{ fontSize: 12 }} formatter={(v) => Number(v).toFixed(4)} />
            <Bar dataKey="ic_drop_mean" isAnimationActive={false}>
              {groupRows.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="min-w-0">
        <h3 className="text-xs text-zinc-500 mb-1">Top-12 individual factor saliency (Integrated Gradients)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={topFactors} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fontFamily: "monospace" }} width={84} />
            <Tooltip contentStyle={{ fontSize: 12 }} formatter={(v) => Number(v).toFixed(4)} />
            <Bar dataKey="saliency" fill="#10b981" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <p className="text-xs text-zinc-500">
        IC drop = permutation-importance per-date cross-section shuffle (preserves time, breaks ranking).
        Saliency = Integrated Gradients average |∂score/∂factor| over a stratified panel sample.
      </p>
    </section>
  );
}
