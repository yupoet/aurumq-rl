"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

import { PRIMARY_METRIC_KEYS } from "@/lib/jsonl";

interface MetricRow {
  timestep: number;
  [key: string]: number | string;
}

export function LiveCurves({
  id,
  initialRows,
  initialOffset,
  isLive,
}: {
  id: string;
  initialRows: MetricRow[];
  initialOffset: number;
  isLive: boolean;
}) {
  const [rows, setRows] = useState<MetricRow[]>(initialRows);
  const [streamLive, setStreamLive] = useState(isLive);

  useEffect(() => {
    if (!isLive) return;
    const path = id.split("/").map(encodeURIComponent).join("/");
    const url = `/api/runs/${path}?part=stream&offset=${initialOffset}`;
    const es = new EventSource(url);

    es.onmessage = (ev) => {
      try {
        const row = JSON.parse(ev.data) as MetricRow;
        setRows((prev) => [...prev, row]);
      } catch {
        // ignore malformed
      }
    };
    es.onerror = () => {
      setStreamLive(false);
      es.close();
    };
    return () => es.close();
  }, [id, isLive, initialOffset]);

  const allKeys = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (k !== "timestep" && typeof r[k] === "number") allKeys.add(k);
    }
  }
  const charts = PRIMARY_METRIC_KEYS.filter((k) => allKeys.has(k));

  return (
    <section>
      <div className="flex items-baseline gap-3 mb-3">
        <h2 className="text-lg font-semibold">Training curves</h2>
        {streamLive && (
          <span className="text-xs rounded px-2 py-0.5 bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200 animate-pulse">
            ● LIVE
          </span>
        )}
      </div>
      {charts.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No metrics yet. {rows.length} rows total.
        </p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {charts.map((k) => (
            <div
              key={k}
              className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3 min-w-0"
            >
              <h3 className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-2">
                {k}
              </h3>
              <ResponsiveContainer width="100%" height={192}>
                <LineChart
                  data={rows.filter((r) => typeof r[k] === "number")}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="currentColor"
                    opacity={0.1}
                  />
                  <XAxis dataKey="timestep" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey={k}
                    stroke="#3b82f6"
                    dot={false}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
