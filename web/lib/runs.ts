import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";

const RUNS_DIR = path.join(process.cwd(), "..", "runs");

export const RunSummarySchema = z.object({
  algorithm: z.string(),
  total_timesteps: z.number(),
  n_envs: z.number().optional(),
  env_type: z.string().optional(),
  reward_type: z.string().optional(),
  universe_filter: z.string().optional(),
  start_date: z.string().optional(),
  end_date: z.string().optional(),
  n_factors: z.number().optional(),
  n_stocks: z.number().optional(),
  top_k: z.number().optional(),
  out_dir: z.string().optional(),
  onnx_path: z.string().optional(),
});
export type RunSummary = z.infer<typeof RunSummarySchema>;

export interface RunListEntry {
  id: string;
  hasModel: boolean;
  hasOnnx: boolean;
  hasBacktest: boolean;
  hasMetrics: boolean;
  summary: RunSummary | null;
  modifiedAt: number;
}

async function exists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function* walkRunDirs(
  baseDir: string,
  prefix = "",
  depth = 0
): AsyncGenerator<{ id: string; absPath: string; mtimeMs: number }> {
  if (depth > 2) return;
  let entries: string[];
  try {
    entries = await fs.readdir(baseDir);
  } catch {
    return;
  }
  for (const name of entries) {
    const abs = path.join(baseDir, name);
    const stat = await fs.stat(abs).catch(() => null);
    if (!stat || !stat.isDirectory()) continue;

    const id = prefix ? `${prefix}/${name}` : name;
    const hasSummary = await exists(path.join(abs, "training_summary.json"));
    const hasMetrics = await exists(path.join(abs, "training_metrics.jsonl"));
    if (hasSummary || hasMetrics) {
      yield { id, absPath: abs, mtimeMs: stat.mtimeMs };
    } else {
      // recurse one level for grouped runs (e.g. compare_rewards_*/return)
      yield* walkRunDirs(abs, id, depth + 1);
    }
  }
}

export async function listRuns(): Promise<RunListEntry[]> {
  const out: RunListEntry[] = [];
  for await (const { id, absPath, mtimeMs } of walkRunDirs(RUNS_DIR)) {
    const summary = await readSummaryFromPath(absPath);
    const algo = (summary?.algorithm ?? "ppo").toLowerCase();
    const [hasModel, hasOnnx, hasBacktest, hasMetrics] = await Promise.all([
      exists(path.join(absPath, `${algo}_final.zip`)),
      exists(path.join(absPath, "policy.onnx")),
      exists(path.join(absPath, "backtest.json")),
      exists(path.join(absPath, "training_metrics.jsonl")),
    ]);
    out.push({
      id,
      hasModel,
      hasOnnx,
      hasBacktest,
      hasMetrics,
      summary,
      modifiedAt: mtimeMs,
    });
  }
  out.sort((a, b) => b.modifiedAt - a.modifiedAt);
  return out;
}

function runDirFromId(id: string): string {
  return path.join(RUNS_DIR, ...id.split("/"));
}

async function readSummaryFromPath(dir: string): Promise<RunSummary | null> {
  try {
    const data = JSON.parse(
      await fs.readFile(path.join(dir, "training_summary.json"), "utf-8")
    );
    return RunSummarySchema.parse(data);
  } catch {
    return null;
  }
}

export async function readSummary(id: string): Promise<RunSummary | null> {
  return readSummaryFromPath(runDirFromId(id));
}

export async function readBacktest(id: string): Promise<unknown | null> {
  const p = path.join(runDirFromId(id), "backtest.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf-8"));
  } catch {
    return null;
  }
}

export async function readMetricsJsonl(
  id: string
): Promise<Record<string, unknown>[]> {
  const p = path.join(runDirFromId(id), "training_metrics.jsonl");
  let raw: string;
  try {
    raw = await fs.readFile(p, "utf-8");
  } catch {
    return [];
  }
  const out: Record<string, unknown>[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      out.push(JSON.parse(trimmed));
    } catch {
      // skip malformed lines
    }
  }
  return out;
}

export async function readBacktestSeries(id: string): Promise<unknown | null> {
  const p = path.join(RUNS_DIR, ...id.split("/"), "backtest_series.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf-8"));
  } catch {
    return null;
  }
}
