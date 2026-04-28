export function pickSeries(
  rows: Record<string, unknown>[],
  key: string
): { x: number; y: number }[] {
  return rows
    .filter((r) => typeof r[key] === "number" && typeof r.timestep === "number")
    .map((r) => ({ x: r.timestep as number, y: r[key] as number }));
}

export function listSeriesKeys(rows: Record<string, unknown>[]): string[] {
  const keys = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (k === "timestep") continue;
      if (typeof r[k] === "number") keys.add(k);
    }
  }
  return Array.from(keys).sort();
}
