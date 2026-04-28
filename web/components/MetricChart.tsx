"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export interface SeriesPoint {
  x: number;
  y: number;
}

export function MetricChart({
  title,
  data,
  color = "#3b82f6",
}: {
  title: string;
  data: SeriesPoint[];
  color?: string;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3">
      <h3 className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-2">
        {title}
      </h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="currentColor"
              opacity={0.1}
            />
            <XAxis dataKey="x" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="y"
              stroke={color}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
