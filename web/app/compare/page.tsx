"use client";

import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";

import { COMPARE_METRIC_KEYS } from "@/lib/jsonl";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

interface RunListEntry {
  id: string;
  hasMetrics: boolean;
  summary: { algorithm?: string; reward_type?: string } | null;
  modifiedAt: number;
}

function CompareInner() {
  const router = useRouter();
  const sp = useSearchParams();
  const initial = (sp.get("ids") ?? "").split(",").filter(Boolean);

  const [runs, setRuns] = useState<RunListEntry[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>(initial);
  const [seriesByRun, setSeriesByRun] = useState<
    Record<string, Record<string, unknown>[]>
  >({});

  useEffect(() => {
    fetch("/api/runs")
      .then((r) => r.json())
      .then(setRuns)
      .catch(() => {});
  }, []);

  useEffect(() => {
    selectedIds.forEach((id) => {
      if (seriesByRun[id]) return;
      fetch(`/api/runs/${id.split("/").map(encodeURIComponent).join("/")}?part=metrics`)
        .then((r) => r.json())
        .then((data) =>
          setSeriesByRun((prev) => ({ ...prev, [id]: data }))
        )
        .catch(() => {});
    });
  }, [selectedIds, seriesByRun]);

  function persistSelection(ids: string[]) {
    setSelectedIds(ids);
    const q = ids.length ? `?ids=${ids.join(",")}` : "";
    router.replace(`/compare${q}`);
  }

  function toggle(id: string) {
    const next = selectedIds.includes(id)
      ? selectedIds.filter((x) => x !== id)
      : [...selectedIds, id];
    persistSelection(next);
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-8 space-y-6">
      <header>
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-300">
          ← back
        </Link>
        <h1 className="text-xl font-semibold mt-2">Compare runs</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Pick 2+ runs to overlay their training curves.
        </p>
      </header>

      <div className="flex flex-wrap gap-2">
        {runs
          .filter((r) => r.hasMetrics)
          .map((r) => (
            <button
              key={r.id}
              onClick={() => toggle(r.id)}
              className={`text-xs font-mono px-2 py-1 rounded border ${
                selectedIds.includes(r.id)
                  ? "bg-indigo-600 text-white border-indigo-700"
                  : "border-zinc-300 dark:border-zinc-700"
              }`}
            >
              {r.id}
            </button>
          ))}
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {COMPARE_METRIC_KEYS.map((key) => {
          const present = selectedIds.some((id) =>
            (seriesByRun[id] ?? []).some(
              (r) => typeof r[key] === "number"
            )
          );
          if (selectedIds.length > 0 && !present) return null;
          return (
            <div
              key={key}
              className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 min-w-0"
            >
              <h3 className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-2">
                {key}
              </h3>
              <ResponsiveContainer width="100%" height={224}>
                <LineChart>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="currentColor"
                    opacity={0.1}
                  />
                  <XAxis
                    dataKey="timestep"
                    tick={{ fontSize: 10 }}
                    type="number"
                  />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {selectedIds.map((id, i) => {
                    const data = (seriesByRun[id] ?? []).filter(
                      (r: Record<string, unknown>) =>
                        typeof r[key] === "number"
                    );
                    return (
                      <Line
                        key={id}
                        data={data}
                        type="monotone"
                        dataKey={key}
                        stroke={COLORS[i % COLORS.length]}
                        dot={false}
                        strokeWidth={1.5}
                        isAnimationActive={false}
                        name={id}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </section>
    </main>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="px-6 py-8">Loading…</div>}>
      <CompareInner />
    </Suspense>
  );
}
