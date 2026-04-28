# Web Dashboard Iteration — Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land three parallel dashboard improvements — live training watch, backtest deep-dive visualization, and run grouping/filter — so the dashboard is useful for the upcoming real-data training and reward-shaping comparison runs.

**Architecture:** Three independent feature branches dispatched in parallel via subagent-driven-development with git worktrees. Each branch isolates to a clean scope (Agent A: SSE + LIVE badge in `web/`; Agent B: backtest series in `src/aurumq_rl/backtest.py` + new web panel; Agent C: pure-frontend grouping/filter in `web/`). Merge serially at the end into main with explicit conflict resolution for the few overlapping files (`web/lib/runs.ts`, `web/app/runs/[...id]/page.tsx`).

**Tech Stack:** Next.js 16 (App Router) · TypeScript strict · React 19 · Tailwind 4 · recharts · stable-baselines3 / numpy (Python side for G2)

---

## Context

The `web/` Next.js dashboard currently shows runs and per-run training curves + a backtest summary card. Over the next 48 h the project goes from synthetic-only to real factor data extracted on Ubuntu, training runs that take hours, and four-way reward-type bake-offs. The spec at `docs/superpowers/specs/2026-04-28-web-dashboard-iteration.md` identifies three highest-leverage gaps. This plan implements all three in parallel.

---

## File Structure

**Created (new files)**:

| File | Track | Responsibility |
|---|---|---|
| `web/components/LiveCurves.tsx` | A (G1) | Client component, EventSource consumer, appends new metric points to chart state |
| `web/components/BacktestSeriesPanel.tsx` | B (G2) | Client component, renders IC time-series + equity curve + random Sharpe histogram |
| `web/components/RunGroupCard.tsx` | C (G3) | Server component, expandable card grouping sibling runs |
| `web/components/FilterBar.tsx` | C (G3) | Client component, filter chips + search box, URL-querystring backed |
| `tests/test_backtest_series.py` | B (G2) | Unit tests for `BacktestSeries` dataclass |

**Modified**:

| File | Tracks | What changes |
|---|---|---|
| `web/lib/runs.ts` | A, B, C | A: add `isRunLive`, `tailMetricsJsonl`. B: add `readBacktestSeries`. C: add `groupRuns`, type `RunGroup`. All three APPEND to bottom of file. |
| `web/app/api/runs/[...id]/route.ts` | A, B | A: add `?part=stream` SSE branch. B: add `?part=backtest_series` branch. |
| `web/app/runs/[...id]/page.tsx` | A, B | A: pass `live` flag and embed `<LiveCurves>`. B: render `<BacktestSeriesPanel>` below `<BacktestSummary>`. |
| `web/app/page.tsx` | C | Wire `<FilterBar>` and switch to `groupRuns(runs)` rendering. |
| `web/components/RunCard.tsx` | A | Add LIVE badge based on `isLive` field. |
| `src/aurumq_rl/backtest.py` | B | Add `BacktestSeries` dataclass, `_per_date_top_k_returns`, `run_backtest_with_series`. Existing `run_backtest` unchanged (calls new internal helpers). |
| `scripts/eval_backtest.py` | B | Switch to `run_backtest_with_series`, write `backtest_series.json`. |
| `web/lib/runs.ts` (RunListEntry type) | A, C | Add `isLive: boolean`. C uses but doesn't add. |

**Untouched but referenced**:
- `runs/<id>/training_metrics.jsonl` — append-only, JSON-per-line.
- `runs/<id>/backtest.json` — schema unchanged.

---

## Phase 0: Setup

### Task 0.1: Verify clean main + create worktrees

**Files:** none (shell only)

- [ ] **Step 1: Confirm baseline**

```bash
cd D:/dev/aurumq-rl
git checkout main && git pull --ff-only
git status            # working tree clean
.venv/Scripts/python.exe -m pytest -q   # 132 passed baseline
```

- [ ] **Step 2: Create three worktrees**

```bash
git worktree add ../aurumq-rl-A -b feat/web-live main
git worktree add ../aurumq-rl-B -b feat/web-backtest-viz main
git worktree add ../aurumq-rl-C -b feat/web-ux-grouping main
```

Expected: three sibling directories `D:/dev/aurumq-rl-{A,B,C}` each on its own branch.

---

## Phase A: Live training watch (G1)

> **Worktree:** `D:/dev/aurumq-rl-A` (branch `feat/web-live`)
> **All tasks below run in that worktree.**

### Task A.1: lib/runs.ts — isRunLive + tailMetricsJsonl

**Files:**
- Modify: `web/lib/runs.ts` (append to bottom)

- [ ] **Step 1: Append helper functions**

Add these exports at the bottom of `web/lib/runs.ts`:

```ts
export async function isRunLive(id: string, thresholdSec = 10): Promise<boolean> {
  const dir = path.join(RUNS_DIR, ...id.split("/"));
  const target = path.join(dir, "training_metrics.jsonl");
  try {
    const stat = await fs.stat(target);
    return Date.now() - stat.mtimeMs < thresholdSec * 1000;
  } catch {
    return false;
  }
}

export interface TailResult {
  rows: Record<string, unknown>[];
  newOffset: number;
  totalSize: number;
}

export async function tailMetricsJsonl(
  id: string,
  fromOffset: number,
  maxBytes = 64 * 1024
): Promise<TailResult> {
  const dir = path.join(RUNS_DIR, ...id.split("/"));
  const target = path.join(dir, "training_metrics.jsonl");
  let stat;
  try {
    stat = await fs.stat(target);
  } catch {
    return { rows: [], newOffset: fromOffset, totalSize: 0 };
  }
  if (stat.size <= fromOffset) {
    return { rows: [], newOffset: fromOffset, totalSize: stat.size };
  }
  const start = Math.max(fromOffset, stat.size - maxBytes);
  const length = stat.size - start;
  const fh = await fs.open(target, "r");
  try {
    const buf = Buffer.alloc(length);
    await fh.read(buf, 0, length, start);
    const text = buf.toString("utf-8");
    const lastNewline = text.lastIndexOf("\n");
    if (lastNewline < 0) {
      return { rows: [], newOffset: fromOffset, totalSize: stat.size };
    }
    const usable = text.slice(0, lastNewline);
    const consumed = start + Buffer.byteLength(usable, "utf-8") + 1;
    const out: Record<string, unknown>[] = [];
    for (const line of usable.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        out.push(JSON.parse(trimmed));
      } catch {
        // ignore malformed line
      }
    }
    return { rows: out, newOffset: consumed, totalSize: stat.size };
  } finally {
    await fh.close();
  }
}
```

- [ ] **Step 2: Add `isLive` field to `RunListEntry` type**

Edit the `RunListEntry` interface (existing in `web/lib/runs.ts`):

```ts
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
```

- [ ] **Step 3: Populate `isLive` inside `listRuns()`**

In `listRuns()`, where the existing `Promise.all([...])` populates `hasModel/hasOnnx/hasBacktest/hasMetrics`, add `isRunLive(id)`:

```ts
const [hasModel, hasOnnx, hasBacktest, hasMetrics, isLive] = await Promise.all([
  exists(path.join(absPath, `${algo}_final.zip`)),
  exists(path.join(absPath, "policy.onnx")),
  exists(path.join(absPath, "backtest.json")),
  exists(path.join(absPath, "training_metrics.jsonl")),
  isRunLive(id),
]);
out.push({
  id,
  hasModel,
  hasOnnx,
  hasBacktest,
  hasMetrics,
  isLive,
  summary,
  modifiedAt: mtimeMs,
});
```

- [ ] **Step 4: TypeScript check**

```bash
cd web && npx tsc --noEmit
```

Expected: no errors. (May take ~10 s.)

- [ ] **Step 5: Commit**

```bash
git add web/lib/runs.ts
git commit -m "feat(web): add isRunLive + tailMetricsJsonl helpers"
```

### Task A.2: SSE route handler

**Files:**
- Modify: `web/app/api/runs/[...id]/route.ts`

- [ ] **Step 1: Extend route to handle ?part=stream**

Replace the existing `GET` function with:

```ts
import { NextResponse } from "next/server";
import {
  readBacktest,
  readMetricsJsonl,
  readSummary,
  tailMetricsJsonl,
} from "@/lib/runs";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string[] }> }
) {
  const { id } = await params;
  const decoded = id.map((s) => decodeURIComponent(s)).join("/");
  const url = new URL(request.url);
  const part = url.searchParams.get("part");

  if (part === "stream") {
    const initialOffset = Number(url.searchParams.get("offset") ?? 0);
    return streamMetrics(decoded, initialOffset, request.signal);
  }

  if (part === "metrics") {
    return NextResponse.json(await readMetricsJsonl(decoded));
  }
  if (part === "backtest") {
    return NextResponse.json(await readBacktest(decoded));
  }
  if (part === "summary") {
    return NextResponse.json(await readSummary(decoded));
  }

  const [summary, metrics, backtest] = await Promise.all([
    readSummary(decoded),
    readMetricsJsonl(decoded),
    readBacktest(decoded),
  ]);
  return NextResponse.json({ summary, metrics, backtest });
}

function streamMetrics(
  id: string,
  initialOffset: number,
  signal: AbortSignal
): Response {
  const encoder = new TextEncoder();
  let offset = initialOffset;
  let interval: ReturnType<typeof setInterval> | null = null;

  const stream = new ReadableStream({
    start(controller) {
      const sendInit = () => {
        controller.enqueue(encoder.encode(`event: open\ndata: {}\n\n`));
      };
      sendInit();

      const tick = async () => {
        try {
          const { rows, newOffset } = await tailMetricsJsonl(id, offset);
          offset = newOffset;
          for (const row of rows) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(row)}\n\n`)
            );
          }
        } catch (err) {
          controller.enqueue(
            encoder.encode(
              `event: error\ndata: ${JSON.stringify({
                message: String(err),
              })}\n\n`
            )
          );
        }
      };

      interval = setInterval(tick, 2000);
      signal.addEventListener("abort", () => {
        if (interval) clearInterval(interval);
        try {
          controller.close();
        } catch {
          // already closed
        }
      });
    },
    cancel() {
      if (interval) clearInterval(interval);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
    },
  });
}
```

- [ ] **Step 2: Smoke check (manual)**

```bash
cd web && npm run dev &
# Wait for "Ready in ..."
curl -N --noproxy "*" "http://localhost:3000/api/runs/ppo_100k?part=stream&offset=99999999" &
# Should print "event: open\ndata: {}" and then nothing (no new bytes)
# Ctrl-C to stop
```

Expected: SSE response with content-type `text/event-stream`, the `event: open` line, then idle.

- [ ] **Step 3: Commit**

```bash
git add web/app/api/runs/[...id]/route.ts
git commit -m "feat(web): add ?part=stream SSE branch for live metrics"
```

### Task A.3: LiveCurves client component

**Files:**
- Create: `web/components/LiveCurves.tsx`

- [ ] **Step 1: Write the component**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

interface MetricRow {
  timestep: number;
  [key: string]: number | string;
}

const PRIMARY_KEYS = [
  "rollout/ep_rew_mean",
  "train/loss",
  "train/policy_gradient_loss",
  "train/value_loss",
  "train/explained_variance",
  "train/approx_kl",
  "train/clip_fraction",
  "time/fps",
];

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
  const offsetRef = useRef(initialOffset);

  useEffect(() => {
    if (!isLive) return;
    const path = id.split("/").map(encodeURIComponent).join("/");
    const url = `/api/runs/${path}?part=stream&offset=${offsetRef.current}`;
    const es = new EventSource(url);

    es.onmessage = (ev) => {
      try {
        const row = JSON.parse(ev.data) as MetricRow;
        setRows((prev) => [...prev, row]);
        offsetRef.current += ev.data.length + 8; // approximate, harmless
      } catch {
        // ignore malformed
      }
    };
    es.onerror = () => {
      setStreamLive(false);
      es.close();
    };
    return () => es.close();
  }, [id, isLive]);

  const allKeys = new Set<string>();
  for (const r of rows) {
    for (const k of Object.keys(r)) {
      if (k !== "timestep" && typeof r[k] === "number") allKeys.add(k);
    }
  }
  const charts = PRIMARY_KEYS.filter((k) => allKeys.has(k));

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
              className="rounded-lg border border-zinc-200 dark:border-zinc-800 p-3"
            >
              <h3 className="text-xs font-medium text-zinc-600 dark:text-zinc-400 mb-2">
                {k}
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
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
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/LiveCurves.tsx
git commit -m "feat(web): LiveCurves client component with SSE consumer"
```

### Task A.4: Wire LiveCurves into detail page

**Files:**
- Modify: `web/app/runs/[...id]/page.tsx`

- [ ] **Step 1: Replace the "Training curves" section**

Replace the entire `<section>` that currently renders `MetricChart` for each PRIMARY_KEY with:

```tsx
import { LiveCurves } from "@/components/LiveCurves";
import { isRunLive } from "@/lib/runs";

// inside the page component, after reading metrics + summary + backtest:
const live = await isRunLive(decoded);
const initialOffset = metrics.length === 0
  ? 0
  : Buffer.byteLength(
      metrics.map((m) => JSON.stringify(m)).join("\n") + "\n",
      "utf-8"
    );
```

Then in JSX:

```tsx
<LiveCurves
  id={decoded}
  initialRows={metrics as Parameters<typeof LiveCurves>[0]["initialRows"]}
  initialOffset={initialOffset}
  isLive={live}
/>
```

Replace the entire existing training curves `<section>` (the one that maps `PRIMARY_KEYS` over `MetricChart`) with just the `<LiveCurves>` invocation.

- [ ] **Step 2: TypeScript check**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 3: Manual smoke**

```bash
cd web && npm run dev
# In a new terminal:
curl -s --noproxy "*" http://localhost:3000/runs/ppo_100k | grep -oE "(LIVE|Training curves)" | head
```

Expected: page renders, "Training curves" present. Since `ppo_100k` is old, no LIVE badge. To test LIVE: append a fake JSONL line to `runs/ppo_100k/training_metrics.jsonl` and reload — badge should appear and curves update.

- [ ] **Step 4: Commit**

```bash
git add web/app/runs/[...id]/page.tsx
git commit -m "feat(web): use LiveCurves on run detail page"
```

### Task A.5: LIVE badge on RunCard

**Files:**
- Modify: `web/components/RunCard.tsx`

- [ ] **Step 1: Add Badge for live**

In the badges row of `RunCard.tsx`, add as the first badge:

```tsx
{run.isLive && (
  <span className="px-1.5 py-0.5 rounded bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-200 animate-pulse">
    live
  </span>
)}
```

Place it before the existing `metrics` / `onnx` / `backtest` badges in the same flex row.

- [ ] **Step 2: Smoke**

```bash
cd web && npm run dev
curl -s --noproxy "*" http://localhost:3000/api/runs | head -c 500
```

Expected: API responses now include `isLive: false` for all runs (none are running).

- [ ] **Step 3: Commit**

```bash
git add web/components/RunCard.tsx
git commit -m "feat(web): show LIVE badge on RunCard for active runs"
```

### Task A.6: Push branch

- [ ] **Step 1: Push**

```bash
git push -u origin feat/web-live
```

**Branch done.** The Round-1 integration phase will merge this into main.

---

## Phase B: Backtest deep-dive (G2)

> **Worktree:** `D:/dev/aurumq-rl-B` (branch `feat/web-backtest-viz`)
> **All tasks below run in that worktree.**

### Task B.1: BacktestSeries dataclass + tests

**Files:**
- Modify: `src/aurumq_rl/backtest.py` (additions)
- Create: `tests/test_backtest_series.py`

- [ ] **Step 1: Write failing test**

`tests/test_backtest_series.py`:

```python
"""Tests for BacktestSeries dataclass + run_backtest_with_series."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from aurumq_rl.backtest import (
    BacktestResult,
    BacktestSeries,
    run_backtest_with_series,
)


def test_backtest_series_shape_matches_panel():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.02, size=(40, 80))
    preds = rets + rng.normal(0, 0.01, size=rets.shape)
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    result, series = run_backtest_with_series(
        predictions=preds,
        returns=rets,
        dates=dates,
        top_k=10,
        n_random_simulations=20,
    )
    assert isinstance(result, BacktestResult)
    assert isinstance(series, BacktestSeries)
    assert len(series.dates) == 40
    assert len(series.ic) == 40
    assert len(series.top_k_returns) == 40
    assert len(series.equity_curve) == 40
    assert len(series.random_baseline_sharpes) == 20


def test_backtest_series_to_json_roundtrip(tmp_path):
    series = BacktestSeries(
        dates=["2025-01-02", "2025-01-03"],
        ic=[0.01, -0.02],
        top_k_returns=[0.001, 0.002],
        equity_curve=[1.001, 1.003],
        random_baseline_sharpes=[0.1, -0.3, 0.5],
    )
    out = tmp_path / "bs.json"
    series.to_json(out)
    loaded = BacktestSeries.from_json(out)
    assert loaded.dates == ["2025-01-02", "2025-01-03"]
    assert loaded.ic == [0.01, -0.02]
    assert loaded.equity_curve == [1.001, 1.003]


def test_run_backtest_with_series_equity_curve_starts_at_one_or_close():
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.02, size=(10, 50))
    preds = rng.normal(0.0, 0.02, size=(10, 50))
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    _, series = run_backtest_with_series(
        predictions=preds, returns=rets, dates=dates, top_k=10
    )
    # First entry is 1 + first day's top-k return
    assert abs(series.equity_curve[0] - (1.0 + series.top_k_returns[0])) < 1e-9
    # Equity series is monotonic with cumulative product semantics
    for i in range(1, len(series.equity_curve)):
        expected = series.equity_curve[i - 1] * (1 + series.top_k_returns[i])
        assert abs(series.equity_curve[i] - expected) < 1e-9
```

- [ ] **Step 2: Run, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/test_backtest_series.py -v
```

Expected: ImportError on `BacktestSeries` / `run_backtest_with_series`.

### Task B.2: Implement BacktestSeries + run_backtest_with_series

**Files:**
- Modify: `src/aurumq_rl/backtest.py`

- [ ] **Step 1: Add the dataclass and function**

Append these to `src/aurumq_rl/backtest.py` (after existing functions, before `__all__`):

```python
@dataclass
class BacktestSeries:
    """Per-date series produced alongside the BacktestResult."""

    dates: list[str]
    ic: list[float]
    top_k_returns: list[float]
    equity_curve: list[float]
    random_baseline_sharpes: list[float] = field(default_factory=list)

    def to_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path | str) -> "BacktestSeries":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


def _per_date_top_k_returns(
    predictions: np.ndarray, returns: np.ndarray, top_k: int
) -> list[float]:
    out: list[float] = []
    for t in range(predictions.shape[0]):
        p, r = predictions[t], returns[t]
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() < top_k:
            out.append(0.0)
            continue
        idx = np.argsort(-p[mask])[:top_k]
        out.append(float(r[mask][idx].mean()))
    return out


def _random_sharpes(
    returns: np.ndarray, top_k: int, n_simulations: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_simulations):
        preds = rng.normal(size=returns.shape)
        out.append(compute_top_k_sharpe(preds, returns, top_k=top_k))
    return out


def run_backtest_with_series(
    predictions: np.ndarray,
    returns: np.ndarray,
    dates: list,
    top_k: int = 30,
    n_random_simulations: int = 100,
    random_seed: int = 0,
) -> tuple["BacktestResult", "BacktestSeries"]:
    """One-shot evaluation that also returns per-date / per-simulation series."""
    if predictions.shape != returns.shape:
        raise ValueError("shape mismatch")
    if len(dates) != predictions.shape[0]:
        raise ValueError(
            f"dates length {len(dates)} != n_dates {predictions.shape[0]}"
        )

    ic_per_date = _per_date_ics(predictions, returns)
    if len(ic_per_date) < predictions.shape[0]:
        # Pad to align with dates; degenerate days fill with 0.0
        ic_per_date = ic_per_date + [0.0] * (predictions.shape[0] - len(ic_per_date))
    top_k_rets = _per_date_top_k_returns(predictions, returns, top_k)

    equity = []
    cum = 1.0
    for ret in top_k_rets:
        cum *= 1.0 + ret
        equity.append(cum)

    random_sharpes = _random_sharpes(
        returns, top_k=top_k, n_simulations=n_random_simulations, seed=random_seed
    )

    arr = np.asarray(top_k_rets)
    sharpe = (
        float(arr.mean() / arr.std(ddof=1) * np.sqrt(252))
        if arr.size > 1 and arr.std(ddof=1) > 1e-12
        else 0.0
    )
    cumret = float(equity[-1] - 1.0) if equity else 0.0

    arr_ic = np.asarray(ic_per_date)
    ic_mean = float(arr_ic.mean()) if arr_ic.size else 0.0
    ic_ir = (
        float(arr_ic.mean() / arr_ic.std(ddof=1))
        if arr_ic.size > 1 and arr_ic.std(ddof=1) > 1e-12
        else 0.0
    )

    arr_rs = np.asarray(random_sharpes)
    baseline = {
        "mean_sharpe": float(arr_rs.mean()) if arr_rs.size else 0.0,
        "std_sharpe": float(arr_rs.std(ddof=1)) if arr_rs.size > 1 else 0.0,
        "p05_sharpe": float(np.percentile(arr_rs, 5)) if arr_rs.size else 0.0,
        "p50_sharpe": float(np.percentile(arr_rs, 50)) if arr_rs.size else 0.0,
        "p95_sharpe": float(np.percentile(arr_rs, 95)) if arr_rs.size else 0.0,
    }

    result = BacktestResult(
        ic=ic_mean,
        ic_ir=ic_ir,
        top_k_sharpe=sharpe,
        top_k_cumret=cumret,
        random_baseline=baseline,
        n_dates=predictions.shape[0],
        n_stocks=predictions.shape[1],
        top_k=top_k,
    )

    series = BacktestSeries(
        dates=[str(d) for d in dates],
        ic=ic_per_date,
        top_k_returns=top_k_rets,
        equity_curve=equity,
        random_baseline_sharpes=random_sharpes,
    )

    return result, series
```

- [ ] **Step 2: Update `__all__`**

```python
__all__ = [
    "BacktestResult",
    "BacktestSeries",
    "compute_ic",
    "compute_ic_ir",
    "compute_top_k_sharpe",
    "compute_top_k_cumret",
    "random_baseline",
    "run_backtest",
    "run_backtest_with_series",
]
```

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_backtest_series.py tests/test_backtest.py -v
```

Expected: 3 new + 7 existing = 10 passed.

- [ ] **Step 4: Commit**

```bash
git add src/aurumq_rl/backtest.py tests/test_backtest_series.py
git commit -m "feat(backtest): BacktestSeries + run_backtest_with_series"
```

### Task B.3: eval_backtest emits backtest_series.json

**Files:**
- Modify: `scripts/eval_backtest.py`

- [ ] **Step 1: Switch to with_series**

Replace the section that calls `run_backtest(...)` and writes `backtest.json` with:

```python
    from aurumq_rl.backtest import run_backtest_with_series

    result, series = run_backtest_with_series(
        predictions=out,
        returns=panel.return_array,
        dates=panel.dates,
        top_k=args.top_k,
        n_random_simulations=args.n_random_simulations,
        random_seed=args.seed,
    )

    out_path = args.run_dir / "backtest.json"
    series_path = args.run_dir / "backtest_series.json"
    result.to_json(out_path)
    series.to_json(series_path)
    print(f"[backtest] wrote {out_path}")
    print(f"[backtest] wrote {series_path}")
```

- [ ] **Step 2: Smoke**

```bash
.venv/Scripts/python.exe scripts/eval_backtest.py \
    --run-dir runs/ppo_100k \
    --data-path data/synthetic_demo.parquet \
    --val-start 2023-08-01 --val-end 2023-12-01 \
    --universe-filter all_a --top-k 30
```

Expected: writes `runs/ppo_100k/backtest_series.json`. Inspect:

```bash
.venv/Scripts/python.exe -c "
import json
d = json.load(open('runs/ppo_100k/backtest_series.json'))
print('keys:', list(d.keys()))
print('len(dates):', len(d['dates']))
print('first ic:', d['ic'][:3])
print('last equity:', d['equity_curve'][-1])
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_backtest.py
git commit -m "feat(backtest): eval_backtest writes backtest_series.json"
```

### Task B.4: web — readBacktestSeries + API branch

**Files:**
- Modify: `web/lib/runs.ts` (append)
- Modify: `web/app/api/runs/[...id]/route.ts`

- [ ] **Step 1: Add helper to lib/runs.ts**

Append at the bottom of `web/lib/runs.ts`:

```ts
export async function readBacktestSeries(id: string): Promise<unknown | null> {
  const p = path.join(RUNS_DIR, ...id.split("/"), "backtest_series.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf-8"));
  } catch {
    return null;
  }
}
```

- [ ] **Step 2: Add ?part=backtest_series to route**

In `web/app/api/runs/[...id]/route.ts`, in the switch over `part`, add before the final combined-response branch:

```ts
  if (part === "backtest_series") {
    return NextResponse.json(await readBacktestSeries(decoded));
  }
```

Don't forget to import `readBacktestSeries` at the top of the file.

- [ ] **Step 3: Smoke**

```bash
cd web && npm run dev
# Other terminal:
curl -s --noproxy "*" "http://localhost:3000/api/runs/ppo_100k?part=backtest_series" | head -c 500
```

Expected: JSON with `dates`, `ic`, `equity_curve`, etc.

- [ ] **Step 4: Commit**

```bash
git add web/lib/runs.ts web/app/api/runs/[...id]/route.ts
git commit -m "feat(web): readBacktestSeries lib helper + ?part=backtest_series"
```

### Task B.5: BacktestSeriesPanel component

**Files:**
- Create: `web/components/BacktestSeriesPanel.tsx`

- [ ] **Step 1: Write the component**

```tsx
"use client";

import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

export interface BacktestSeriesData {
  dates: string[];
  ic: number[];
  top_k_returns: number[];
  equity_curve: number[];
  random_baseline_sharpes: number[];
}

export function BacktestSeriesPanel({
  data,
  realizedSharpe,
}: {
  data: BacktestSeriesData;
  realizedSharpe: number;
}) {
  const points = data.dates.map((d, i) => ({
    date: d,
    ic: data.ic[i] ?? 0,
    equity: data.equity_curve[i] ?? 1,
  }));

  const histogramBins = 24;
  const sharpes = data.random_baseline_sharpes;
  const minS = Math.min(...sharpes);
  const maxS = Math.max(...sharpes);
  const step = (maxS - minS) / histogramBins || 1;
  const histogram = Array.from({ length: histogramBins }, (_, i) => ({
    bin: minS + (i + 0.5) * step,
    count: 0,
  }));
  for (const s of sharpes) {
    const idx = Math.min(
      histogramBins - 1,
      Math.max(0, Math.floor((s - minS) / step))
    );
    histogram[idx].count += 1;
  }

  return (
    <section className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-5">
      <h2 className="text-lg font-semibold mb-4">Backtest deep-dive</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="text-xs text-zinc-500 mb-1">IC over time</h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={points}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} hide />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="ic"
                  stroke="#10b981"
                  dot={false}
                  strokeWidth={1.2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="text-xs text-zinc-500 mb-1">
            Equity curve (top-K, equal-weight)
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={points}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} hide />
                <YAxis
                  tick={{ fontSize: 10 }}
                  domain={["auto", "auto"]}
                />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <ReferenceLine y={1} stroke="#888" strokeDasharray="3 3" />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.15}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="md:col-span-2">
          <h3 className="text-xs text-zinc-500 mb-1">
            Random-baseline Sharpe distribution (n=
            {data.random_baseline_sharpes.length})
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.1} />
                <XAxis
                  dataKey="bin"
                  tick={{ fontSize: 10 }}
                  tickFormatter={(v: number) => v.toFixed(2)}
                />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ fontSize: 12 }}
                  labelFormatter={(v: number) => `Sharpe ${Number(v).toFixed(2)}`}
                />
                <ReferenceLine
                  x={realizedSharpe}
                  stroke="#ef4444"
                  strokeWidth={2}
                  label={{
                    value: `realized ${realizedSharpe.toFixed(2)}`,
                    fill: "#ef4444",
                    fontSize: 10,
                    position: "top",
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="#8b5cf6"
                  isAnimationActive={false}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/BacktestSeriesPanel.tsx
git commit -m "feat(web): BacktestSeriesPanel — IC + equity + random hist"
```

### Task B.6: Wire BacktestSeriesPanel into detail page

**Files:**
- Modify: `web/app/runs/[...id]/page.tsx`

- [ ] **Step 1: Import + render**

Add the import at the top:

```tsx
import { BacktestSeriesPanel, type BacktestSeriesData } from "@/components/BacktestSeriesPanel";
import { readBacktestSeries } from "@/lib/runs";
```

After the existing `const backtest = ...` line, add:

```tsx
  const series = (await readBacktestSeries(decoded)) as BacktestSeriesData | null;
```

After `<BacktestSummary>` is rendered (around the existing JSX), add:

```tsx
{series && backtest && (
  <BacktestSeriesPanel data={series} realizedSharpe={backtest.top_k_sharpe} />
)}
```

- [ ] **Step 2: Smoke**

```bash
cd web && npm run dev
# Open http://localhost:3000/runs/ppo_100k
```

Expected: Below the BacktestSummary card, a new "Backtest deep-dive" section with three charts.

- [ ] **Step 3: Commit**

```bash
git add web/app/runs/[...id]/page.tsx
git commit -m "feat(web): show BacktestSeriesPanel on detail page"
```

### Task B.7: Push branch

```bash
git push -u origin feat/web-backtest-viz
```

---

## Phase C: Run grouping + filter + search (G3)

> **Worktree:** `D:/dev/aurumq-rl-C` (branch `feat/web-ux-grouping`)
> **All tasks below run in that worktree.**

### Task C.1: groupRuns + RunGroup type

**Files:**
- Modify: `web/lib/runs.ts` (append)

- [ ] **Step 1: Append types and grouping function**

Add at the bottom of `web/lib/runs.ts`:

```ts
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
      // Singleton: render as standalone instead of group of 1
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
```

- [ ] **Step 2: TypeScript check**

```bash
cd web && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add web/lib/runs.ts
git commit -m "feat(web): groupRuns helper + RunGroup type"
```

### Task C.2: RunGroupCard component

**Files:**
- Create: `web/components/RunGroupCard.tsx`

- [ ] **Step 1: Write the component**

```tsx
import Link from "next/link";
import type { RunGroup } from "@/lib/runs";

const fmt = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

export function RunGroupCard({ group }: { group: RunGroup }) {
  return (
    <details className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-4 group/run">
      <summary className="cursor-pointer list-none">
        <div className="flex items-baseline justify-between gap-3">
          <h3 className="font-mono text-sm">
            <span className="text-zinc-400 mr-1">▸</span>
            {group.parent}{" "}
            <span className="text-zinc-500 text-xs">
              ({group.children.length} runs)
            </span>
          </h3>
          <time className="text-xs text-zinc-500">
            {fmt.format(group.modifiedAt)}
          </time>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5 text-[10px]">
          {group.children.map((c) => (
            <span
              key={c.id}
              className="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-900 font-mono"
            >
              {c.summary?.reward_type ?? c.id.split("/").slice(-1)[0]}
            </span>
          ))}
        </div>
      </summary>

      <ul className="mt-3 space-y-2 border-t border-zinc-200 dark:border-zinc-800 pt-3">
        {group.children.map((c) => (
          <li key={c.id}>
            <Link
              href={`/runs/${encodeURIComponent(c.id)}`}
              className="flex items-baseline justify-between gap-2 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900 rounded px-2 py-1"
            >
              <span className="font-mono text-xs">
                {c.id.split("/").slice(1).join("/") || c.id}
              </span>
              <span className="text-xs text-zinc-500">
                {c.summary?.algorithm} · {c.summary?.total_timesteps?.toLocaleString()} ·{" "}
                {c.summary?.reward_type ?? "—"}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </details>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/RunGroupCard.tsx
git commit -m "feat(web): RunGroupCard expandable group display"
```

### Task C.3: FilterBar component

**Files:**
- Create: `web/components/FilterBar.tsx`

- [ ] **Step 1: Write the component**

```tsx
"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export interface FilterValues {
  q: string;
  algorithm: string;
  reward_type: string;
  universe_filter: string;
}

export function FilterBar({
  algorithms,
  rewardTypes,
  universes,
}: {
  algorithms: string[];
  rewardTypes: string[];
  universes: string[];
}) {
  const router = useRouter();
  const sp = useSearchParams();
  const [q, setQ] = useState(sp.get("q") ?? "");
  const [algo, setAlgo] = useState(sp.get("algo") ?? "");
  const [reward, setReward] = useState(sp.get("reward") ?? "");
  const [universe, setUniverse] = useState(sp.get("universe") ?? "");

  useEffect(() => {
    const params = new URLSearchParams();
    if (q) params.set("q", q);
    if (algo) params.set("algo", algo);
    if (reward) params.set("reward", reward);
    if (universe) params.set("universe", universe);
    const qs = params.toString();
    router.replace(`/${qs ? `?${qs}` : ""}`);
  }, [q, algo, reward, universe, router]);

  return (
    <div className="flex flex-wrap items-center gap-2 mb-4">
      <input
        type="text"
        placeholder="Search id / reward…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        className="text-sm rounded border border-zinc-300 dark:border-zinc-700 bg-transparent px-2 py-1 w-60"
      />
      <Chip label="algo" value={algo} options={algorithms} onChange={setAlgo} />
      <Chip label="reward" value={reward} options={rewardTypes} onChange={setReward} />
      <Chip
        label="universe"
        value={universe}
        options={universes}
        onChange={setUniverse}
      />
      {(q || algo || reward || universe) && (
        <button
          onClick={() => {
            setQ("");
            setAlgo("");
            setReward("");
            setUniverse("");
          }}
          className="text-xs text-zinc-500 hover:text-zinc-300 underline"
        >
          clear
        </button>
      )}
    </div>
  );
}

function Chip({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-xs rounded border border-zinc-300 dark:border-zinc-700 bg-transparent px-2 py-1"
    >
      <option value="">{label}: all</option>
      {options.map((o) => (
        <option key={o} value={o}>
          {label}={o}
        </option>
      ))}
    </select>
  );
}

export function applyFilters(
  runs: { id: string; summary: { algorithm?: string; reward_type?: string; universe_filter?: string } | null }[],
  filters: FilterValues
) {
  const q = filters.q.toLowerCase();
  return runs.filter((r) => {
    if (q && !r.id.toLowerCase().includes(q) && !(r.summary?.reward_type ?? "").toLowerCase().includes(q)) {
      return false;
    }
    if (filters.algorithm && r.summary?.algorithm !== filters.algorithm) return false;
    if (filters.reward_type && r.summary?.reward_type !== filters.reward_type) return false;
    if (filters.universe_filter && r.summary?.universe_filter !== filters.universe_filter) return false;
    return true;
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add web/components/FilterBar.tsx
git commit -m "feat(web): FilterBar with chips + search, URL-querystring backed"
```

### Task C.4: Wire grouping + filtering into home page

**Files:**
- Modify: `web/app/page.tsx`

- [ ] **Step 1: Replace home page**

Overwrite `web/app/page.tsx` with:

```tsx
import Link from "next/link";
import { listRuns, groupRuns } from "@/lib/runs";
import { RunCard } from "@/components/RunCard";
import { RunGroupCard } from "@/components/RunGroupCard";
import { FilterBar } from "@/components/FilterBar";
import HomeClient from "./HomeClient";

export const dynamic = "force-dynamic";

export default async function Page() {
  const runs = await listRuns();
  const algorithms = uniq(runs.map((r) => r.summary?.algorithm).filter(Boolean) as string[]);
  const rewardTypes = uniq(runs.map((r) => r.summary?.reward_type).filter(Boolean) as string[]);
  const universes = uniq(runs.map((r) => r.summary?.universe_filter).filter(Boolean) as string[]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-8">
      <header className="flex items-baseline justify-between mb-6">
        <h1 className="text-2xl font-semibold">AurumQ-RL Dashboard</h1>
        <Link
          href="/compare"
          className="text-sm rounded-lg border border-zinc-300 dark:border-zinc-700 px-3 py-1.5 hover:border-zinc-500"
        >
          对比 →
        </Link>
      </header>
      <HomeClient
        runs={runs}
        algorithms={algorithms}
        rewardTypes={rewardTypes}
        universes={universes}
      />
    </main>
  );
}

function uniq<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}
```

Create the client wrapper `web/app/HomeClient.tsx`:

```tsx
"use client";

import { useSearchParams } from "next/navigation";
import { useMemo, Suspense } from "react";
import type { RunListEntry } from "@/lib/runs";
import { groupRuns } from "@/lib/runs";
import { RunCard } from "@/components/RunCard";
import { RunGroupCard } from "@/components/RunGroupCard";
import { FilterBar, applyFilters } from "@/components/FilterBar";

interface Props {
  runs: RunListEntry[];
  algorithms: string[];
  rewardTypes: string[];
  universes: string[];
}

function Inner({ runs, algorithms, rewardTypes, universes }: Props) {
  const sp = useSearchParams();
  const filters = {
    q: sp.get("q") ?? "",
    algorithm: sp.get("algo") ?? "",
    reward_type: sp.get("reward") ?? "",
    universe_filter: sp.get("universe") ?? "",
  };

  const visible = useMemo(() => applyFilters(runs, filters), [runs, filters]);
  const grouped = useMemo(() => groupRuns(visible), [visible]);

  return (
    <>
      <FilterBar
        algorithms={algorithms}
        rewardTypes={rewardTypes}
        universes={universes}
      />
      <p className="text-sm text-zinc-500 mb-4">
        {visible.length} / {runs.length} 次训练 · {grouped.length} entries
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {grouped.map((g) =>
          g.kind === "single" ? (
            <RunCard key={g.run.id} run={g.run} />
          ) : (
            <RunGroupCard key={g.group.parent} group={g.group} />
          )
        )}
      </div>
    </>
  );
}

export default function HomeClient(props: Props) {
  return (
    <Suspense fallback={<div className="text-sm text-zinc-500">Loading…</div>}>
      <Inner {...props} />
    </Suspense>
  );
}
```

- [ ] **Step 2: Smoke**

```bash
cd web && npm run dev
# Browser: http://localhost:3000
```

Expected: filter chips visible at top; runs grouped under `verify_compare` etc. Clicking a group expands.

- [ ] **Step 3: Commit**

```bash
git add web/app/page.tsx web/app/HomeClient.tsx
git commit -m "feat(web): home page with FilterBar + groupRuns"
```

### Task C.5: Push branch

```bash
git push -u origin feat/web-ux-grouping
```

---

## Phase Z: Integration

> **Worktree:** `D:/dev/aurumq-rl` (the original main checkout)

### Task Z.1: Merge feat/web-live

- [ ] **Step 1: Merge**

```bash
cd D:/dev/aurumq-rl
git checkout main
git merge --no-ff feat/web-live -m "Merge feat/web-live: SSE-based live training watch"
```

Expected: clean merge.

- [ ] **Step 2: Test**

```bash
.venv/Scripts/python.exe -m pytest -q
cd web && npx tsc --noEmit
```

### Task Z.2: Merge feat/web-backtest-viz

- [ ] **Step 1: Merge**

```bash
cd D:/dev/aurumq-rl
git merge --no-ff feat/web-backtest-viz -m "Merge feat/web-backtest-viz: backtest deep-dive visualization"
```

If conflict in `web/app/runs/[...id]/page.tsx` (both branches add imports / JSX in similar regions): keep BOTH imports, BOTH the `<LiveCurves>` and `<BacktestSeriesPanel>` invocations, with `<BacktestSeriesPanel>` placed **between** `<BacktestSummary>` and `<LiveCurves>`.

If conflict in `web/lib/runs.ts`: keep BOTH appended sections (A's `isRunLive`+`tailMetricsJsonl` and B's `readBacktestSeries`).

If conflict in `web/app/api/runs/[...id]/route.ts`: keep BOTH the `?part=stream` branch (A) and `?part=backtest_series` branch (B). Both come before the combined-response branch.

- [ ] **Step 2: Test**

```bash
.venv/Scripts/python.exe -m pytest -q
cd web && npx tsc --noEmit
```

### Task Z.3: Merge feat/web-ux-grouping

- [ ] **Step 1: Merge**

```bash
cd D:/dev/aurumq-rl
git merge --no-ff feat/web-ux-grouping -m "Merge feat/web-ux-grouping: home page filtering + grouping"
```

If conflict in `web/lib/runs.ts`: keep all three appended sections (A, B, C).

If conflict in `web/app/page.tsx`: take C's version entirely (C overwrites the home page deliberately).

- [ ] **Step 2: Test**

```bash
.venv/Scripts/python.exe -m pytest -q
cd web && npx tsc --noEmit
```

### Task Z.4: Push main + delete feature branches + worktrees

- [ ] **Step 1: Push and clean up**

```bash
cd D:/dev/aurumq-rl
git push origin main

# Remove worktrees and branches
git worktree remove ../aurumq-rl-A
git worktree remove ../aurumq-rl-B
git worktree remove ../aurumq-rl-C
git branch -D feat/web-live feat/web-backtest-viz feat/web-ux-grouping
git push origin --delete feat/web-live feat/web-backtest-viz feat/web-ux-grouping
```

---

## Verification

After all merges:

1. **Pytest**: `132 + 3 = 135 passed` (3 new from Task B.1).
2. **TypeScript**: `cd web && npx tsc --noEmit` — no errors.
3. **Dev server**: `cd web && npm run dev`. Visit:
   - `http://localhost:3000` — see FilterBar + grouped run list.
   - `http://localhost:3000/runs/ppo_100k` — see BacktestSummary + BacktestSeriesPanel + LiveCurves (no LIVE since old run).
4. **Live test**: append a fake metrics line:
   ```bash
   echo '{"timestep": 99999, "rollout/ep_rew_mean": 0.05, "time/fps": 1234}' >> runs/ppo_100k/training_metrics.jsonl
   touch runs/ppo_100k/training_metrics.jsonl
   ```
   Reload `/runs/ppo_100k` — LIVE badge appears, EventSource streams in the new point within 2 s.
5. **API**:
   ```bash
   curl -s --noproxy "*" http://localhost:3000/api/runs | python -c "import json,sys; d=json.load(sys.stdin); print('isLive present:', all('isLive' in r for r in d))"
   curl -s --noproxy "*" "http://localhost:3000/api/runs/ppo_100k?part=backtest_series" | head -c 200
   ```
   Expected: first prints `True`; second prints JSON with `dates`/`ic`/`equity_curve`.

---

## Out of scope (Round 2)

- Vitest + RTL for `web/lib` and components
- Live training auto-scroll / smoothing / log-scale toggles
- Per-stock attribution in backtest deep-dive
- Diff view between two runs (config side-by-side)
- Markdown notes per run

When real data lands and the user has a few real training runs, Round 2 will pick from these based on what feels rough.
