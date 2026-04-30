"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GpuSample } from "@/lib/runs-shared";

export function GpuMetricsPanel({ data }: { data: GpuSample[] }) {
  if (data.length === 0) return null;

  const last = data[data.length - 1];
  const memTotal = last.mem_total_mb;
  const peakUtil = data.reduce((m, s) => Math.max(m, s.util_pct), 0);
  const peakMem = data.reduce((m, s) => Math.max(m, s.mem_used_mb), 0);
  const peakPower = data.reduce((m, s) => Math.max(m, s.power_w), 0);
  const powerYMax = Math.max(300, Math.ceil(peakPower / 50) * 50);

  return (
    <section className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5">
      <div className="flex items-baseline justify-between flex-wrap gap-2 mb-4">
        <div>
          <h2 className="text-lg font-semibold">GPU stats</h2>
          <p className="text-xs text-zinc-500 mt-0.5 font-mono">
            {last.device_name ?? "GPU"} · samples: {data.length} · peak util:{" "}
            {peakUtil}% · peak mem: {peakMem.toLocaleString()} MB
          </p>
        </div>
        <dl className="grid grid-cols-3 gap-x-4 gap-y-1 text-right text-sm">
          <Stat label="util" value={`${last.util_pct}%`} />
          <Stat
            label="mem"
            value={`${last.mem_used_mb.toLocaleString()} / ${memTotal.toLocaleString()} MB`}
          />
          <Stat label="power" value={`${last.power_w.toFixed(1)} W`} />
          <Stat label="temp" value={`${last.temp_c}°C`} />
          <Stat label="step" value={last.timestep.toLocaleString()} />
          <Stat label="now" value={shortTime(last.timestamp)} />
        </dl>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <SubChart
          title="Util %"
          stroke="#10b981"
          data={data}
          yKey="util_pct"
          yDomain={[0, 100]}
          unit="%"
        />
        <SubChart
          title="Memory MB"
          stroke="#3b82f6"
          data={data}
          yKey="mem_used_mb"
          yDomain={[0, memTotal || "auto"]}
          unit="MB"
        />
        <SubChart
          title="Power W"
          stroke="#f59e0b"
          data={data}
          yKey="power_w"
          yDomain={[0, powerYMax]}
          unit="W"
        />
      </div>
    </section>
  );
}

function SubChart({
  title,
  stroke,
  data,
  yKey,
  yDomain,
  unit,
}: {
  title: string;
  stroke: string;
  data: GpuSample[];
  yKey: "util_pct" | "mem_used_mb" | "power_w";
  yDomain: [number, number | "auto"];
  unit: string;
}) {
  return (
    <div className="min-w-0">
      <h3 className="text-xs text-zinc-500 mb-1">{title}</h3>
      <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
            <XAxis
              dataKey="timestep"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 9 }}
              tickFormatter={(v: number) =>
                v >= 1000 ? `${Math.round(v / 1000)}k` : `${v}`
              }
            />
            <YAxis
              tick={{ fontSize: 10 }}
              domain={yDomain as [number, number]}
              width={40}
            />
            <Tooltip
              contentStyle={{ fontSize: 12 }}
              labelFormatter={(v) => `step ${Number(v).toLocaleString()}`}
              formatter={(v) => [`${Number(v)} ${unit}`, title]}
            />
            <Line
              type="monotone"
              dataKey={yKey}
              stroke={stroke}
              dot={false}
              strokeWidth={1.4}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="font-mono">{value}</dd>
    </div>
  );
}

function shortTime(iso: string): string {
  // Render only HH:MM:SS in local time; fall back to the raw string if it
  // doesn't parse.
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString();
}
