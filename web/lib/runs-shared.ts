// Pure types + helpers shared between server and client components.
// Importing from here is safe in "use client" modules — has NO node:fs imports.

export interface GpuSample {
  timestamp: string;
  timestep: number;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
  power_w: number;
  device_name?: string;
}

export interface RunSummary {
  algorithm: string;
  total_timesteps: number;
  n_envs?: number;
  env_type?: string;
  reward_type?: string;
  universe_filter?: string;
  start_date?: string;
  end_date?: string;
  n_factors?: number;
  n_stocks?: number;
  top_k?: number;
  out_dir?: string;
  onnx_path?: string;
}

export interface RunListEntry {
  id: string;
  hasModel: boolean;
  hasOnnx: boolean;
  hasBacktest: boolean;
  hasMetrics: boolean;
  isLive: boolean;
  summary: RunSummary | null;
  modifiedAt: number;
}

export interface RunGroup {
  parent: string;
  children: RunListEntry[];
  modifiedAt: number;
}

export type RunDisplay =
  | { kind: "single"; run: RunListEntry }
  | { kind: "group"; group: RunGroup };

export function groupRuns(runs: RunListEntry[]): RunDisplay[] {
  const buckets = new Map<string, RunListEntry[]>();
  const standalone: RunListEntry[] = [];

  for (const r of runs) {
    const slash = r.id.indexOf("/");
    if (slash < 0) {
      standalone.push(r);
      continue;
    }
    const parent = r.id.slice(0, slash);
    const arr = buckets.get(parent);
    if (arr) arr.push(r);
    else buckets.set(parent, [r]);
  }

  const out: RunDisplay[] = [];
  for (const [parent, children] of buckets) {
    if (children.length === 1) {
      out.push({ kind: "single", run: children[0] });
    } else {
      const modifiedAt = Math.max(...children.map((c) => c.modifiedAt));
      children.sort((a, b) => b.modifiedAt - a.modifiedAt);
      out.push({ kind: "group", group: { parent, children, modifiedAt } });
    }
  }
  for (const r of standalone) {
    out.push({ kind: "single", run: r });
  }
  out.sort((a, b) => {
    const ta = a.kind === "single" ? a.run.modifiedAt : a.group.modifiedAt;
    const tb = b.kind === "single" ? b.run.modifiedAt : b.group.modifiedAt;
    return tb - ta;
  });
  return out;
}
