// Canonical TrainingMetrics schema written by sb3_callbacks.py — use these
// for new runs. Legacy raw SB3 keys are listed alongside so older
// training_metrics.jsonl files still render. The chart filter only keeps
// keys actually present in a given run, so the union is safe.
export const PRIMARY_METRIC_KEYS = [
  // canonical
  "episode_reward_mean",
  "policy_loss",
  "value_loss",
  "entropy",
  "explained_variance",
  "learning_rate",
  "fps",
  // legacy raw SB3
  "rollout/ep_rew_mean",
  "train/loss",
  "train/policy_gradient_loss",
  "train/value_loss",
  "train/explained_variance",
  "train/approx_kl",
  "train/clip_fraction",
  "time/fps",
];

export const COMPARE_METRIC_KEYS = [
  // canonical
  "episode_reward_mean",
  "policy_loss",
  "value_loss",
  "explained_variance",
  // legacy
  "rollout/ep_rew_mean",
  "train/loss",
  "train/explained_variance",
  "train/policy_gradient_loss",
];

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
