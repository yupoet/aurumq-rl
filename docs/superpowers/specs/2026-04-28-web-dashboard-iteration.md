# Web Dashboard Iteration — Round 1 Design

**Date:** 2026-04-28
**Author:** Claude (auto-mode brainstorm; user offline on Ubuntu data extraction)
**Status:** Draft, decisions marked `(unilateral)` are open to user revision when they return.

## Context

After landing the initial dashboard (run list + run detail with 8 fixed curves + multi-run compare) and merging to main, the user is on Ubuntu doing real-data extraction. When real Parquets land back on the GPU box, they will:

1. Train a 200k validation run (~10-15 min) and a 1M production run (~5-8h).
2. Run `compare_rewards.py` across {return, sharpe, sortino, mean_variance} on real data (~1h).
3. Backtest each on a held-out window and decide if any reward type produces real alpha.

The current dashboard has gaps that hurt that workflow:

| Gap | Impact when real training arrives |
|---|---|
| No live updates (post-hoc only) | 5-8h training: must `tail -f train.log` instead of using the UI |
| Backtest is summary numbers only | Cannot tell whether good Sharpe is consistent or driven by a few outlier days |
| Flat run list, no grouping/filtering | `compare_rewards/<reward_type>` floods the list; finding runs harder |
| No tests | Refactoring confidence is low |

This spec covers Round 1: live updates, backtest deep-dive, and run grouping/filtering. Tests and any further polish are deferred to Round 2.

## Goals

- **G1 (Live)**: While `train.py` is running, the run-detail page auto-updates training curves without a full page reload, and visually flags the run as "LIVE".
- **G2 (Backtest deep-dive)**: Per-date IC, top-K equity curve, and random-baseline Sharpe distribution are visible on the run-detail page so the user can judge robustness.
- **G3 (UX foundation)**: Run list groups child runs (e.g. `compare_rewards_*/return`) under a single expandable card; filter chips and search reduce noise as the run count grows.

## Non-goals

- WebSocket-based two-way comms (one-way SSE is sufficient — see decision D1).
- File system watch via inotify / FSEvents (cross-platform polling is simpler — see D2).
- Rewriting backtest module signatures (extend, don't break).
- Vitest / RTL tests (Round 2).
- Authentication / multi-user (single local user).

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │  Next.js 16 (App Router, server components)  │
                    └──┬──────────────────┬───────────────────────┘
                       │                  │
                       │ server route     │ client fetch / SSE
                       ▼                  ▼
       ┌─────────────────────────┐  ┌──────────────────────────┐
       │ /api/runs                │  │ /api/runs/<id...>?part=… │
       │ /api/runs/<id...>        │  │   ?part=stream            │  ◀─── new (G1)
       └────────────┬─────────────┘  └────────────┬──────────────┘
                    │                              │
                    ▼                              ▼
       ┌─────────────────────────────────────────────────────────┐
       │ web/lib/runs.ts                                          │
       │   listRuns / readSummary / readMetricsJsonl / …          │
       │   tailMetricsJsonl(id, fromOffset, limit)        ◀── new │
       │   isRunLive(id, threshSec=10)                    ◀── new │
       └────────────┬─────────────────────────────────────────────┘
                    │
                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Filesystem: D:/dev/aurumq-rl/runs/                       │
       │   <id>/                                                  │
       │     training_metrics.jsonl                               │
       │     training_summary.json                                │
       │     backtest.json                                        │
       │     backtest_series.json                          ◀── new (G2) │
       └─────────────────────────────────────────────────────────┘

       Python side (src/aurumq_rl/):
         backtest.py   — extended to expose per-date series  ◀── new (G2)
         eval_backtest.py — write backtest_series.json       ◀── new (G2)
```

## Components & decisions

### G1. Live training watch

**Decision D1 (unilateral): SSE over WebSocket.** Server pushes new metric rows; client never sends. Next.js App Router supports `ReadableStream` natively. Lower complexity than WebSocket and no client-side ws library.

**Decision D2 (unilateral): 2-second polling of file mtime.** No inotify / chokidar — both add cross-platform complexity. JSONL files grow append-only so we can use a byte-offset cursor (open file, seek to offset, read tail, stream new lines, update offset) and avoid re-parsing.

**Decision D3 (unilateral): "LIVE" if mtime < 10 s.** Training writes metrics every 1k env steps × n_envs ≈ once every few seconds at fps=300+. Threshold of 10 s detects active training without flapping when there's a brief lull.

**Files**:
- `web/lib/runs.ts` — add `isRunLive(id, thresholdSec)` and `tailMetricsJsonl(id, fromOffset, maxLines)`.
- `web/app/api/runs/[...id]/route.ts` — extend to handle `?part=stream` returning SSE.
- `web/app/runs/[...id]/page.tsx` — flag a `live` boolean from server, embed a client-side `<LiveCurves>` component.
- `web/components/LiveCurves.tsx` — new client component using `EventSource` to append data to existing recharts plots.
- `web/components/RunCard.tsx` — show LIVE badge based on `isLive`.

**Data flow**:
1. Page server-render: read summary + initial metrics + `live = isRunLive(id)`.
2. Client mount: open `EventSource("/api/runs/<id...>?part=stream&offset=<initial-byte-count>")`.
3. Server route: every 2 s, stat file, if `mtime > last`, read tail, emit `data: {…}\n\n` per new row.
4. Client: append to chart data state; recharts re-renders.
5. Client closes EventSource on page unmount.

**Backpressure**: bounded by SSE event buffer. If client stalls, server's writes to the response stream will fail and the route exits — we accept that as a "client refresh fixes it" failure mode.

### G2. Backtest deep-dive

**Decision D4 (unilateral): persist per-date series to `backtest_series.json` (sibling of `backtest.json`).** Keeps the existing `backtest.json` schema stable; consumers that only need summary unaffected. New file is optional — UI degrades gracefully when absent.

**Schema** (`backtest_series.json`):

```json
{
  "dates": ["2025-01-02", "2025-01-03", ...],
  "ic": [0.012, -0.041, 0.027, ...],
  "top_k_returns": [0.0014, -0.0031, 0.0022, ...],
  "equity_curve": [1.0, 1.0014, 0.9983, 1.0005, ...],
  "random_baseline_sharpes": [-0.34, 0.12, -1.05, ...]
}
```

**Files**:
- `src/aurumq_rl/backtest.py` — add `BacktestSeries` dataclass and a new function `run_backtest_with_series(...)` that returns `(BacktestResult, BacktestSeries)`. Existing `run_backtest` keeps its signature.
- `scripts/eval_backtest.py` — switch to `run_backtest_with_series` and write both `backtest.json` and `backtest_series.json`.
- `tests/test_backtest.py` — add coverage for `BacktestSeries` shape and JSON round-trip.
- `web/lib/runs.ts` — `readBacktestSeries(id)` helper.
- `web/app/api/runs/[...id]/route.ts` — `?part=backtest_series` branch.
- `web/components/BacktestSeriesPanel.tsx` — IC time-series chart, equity curve, baseline histogram. Lazy-imported on the detail page.

### G3. Run grouping + filter + search

**Decision D5 (unilateral): structural grouping by id prefix.** If multiple run ids share a parent path (e.g. `compare_rewards_20260428_100816/return`, `…/sharpe`, `…/sortino`, `…/mean_variance`), they collapse into a single group card on the home page. Click expands to show children. Standalone runs render as today.

**Decision D6 (unilateral): client-side filter & search.** All run metadata fits in memory comfortably for any practical project size. No need for a search backend.

**Files**:
- `web/lib/runs.ts` — `groupRuns(runs)` helper that returns `{ kind: "group", parent, children } | { kind: "single", run }`.
- `web/components/RunGroupCard.tsx` — new card UI for grouped runs (expandable, summary stats per child).
- `web/components/FilterBar.tsx` — chips for algorithm / reward_type / universe; text search; URL-querystring backed (bookmarkable).
- `web/app/page.tsx` — wire FilterBar + groupRuns.

## Implementation order & agent decomposition

Three independent tracks in Round 1, dispatchable as parallel subagents:

| Track | Agent | Scope (files) | Branch |
|---|---|---|---|
| G1 Live | Agent A | web/lib/runs.ts (additions), web/app/api/runs/[...id]/route.ts (additions), web/app/runs/[...id]/page.tsx, web/components/{LiveCurves,RunCard}.tsx | feat/web-live |
| G2 Backtest viz | Agent B | src/aurumq_rl/backtest.py, scripts/eval_backtest.py, tests/test_backtest.py, web/lib/runs.ts (additions), web/components/BacktestSeriesPanel.tsx, web/app/runs/[...id]/page.tsx | feat/web-backtest-viz |
| G3 Grouping/filter | Agent C | web/lib/runs.ts (additions), web/app/page.tsx, web/components/{RunGroupCard,FilterBar}.tsx | feat/web-ux-grouping |

**File overlap risk**: all three touch `web/lib/runs.ts`. Mitigations:
1. Each agent appends to the bottom of `runs.ts` with a clearly-named exported function. No agent rewrites existing functions.
2. Each agent works in its own git worktree (per the **using-git-worktrees** skill) so concurrent edits don't stomp.
3. Round 1 finishes by sequentially merging the 3 branches into main; conflicts in `runs.ts` resolved by keeping all additions.

**File overlap on `app/runs/[...id]/page.tsx` between G1 and G2**: A injects `<LiveCurves>`, B injects `<BacktestSeriesPanel>`. Mitigation: A modifies in place, B rebases on top of A or merges via small conflict resolve (both add at distinct sections of the page).

Order if conflicts force serialization: A (Live) → B (Backtest viz) → C (Grouping). G3 doesn't touch page detail at all so can stay in pure parallel.

## Testing

Per-track:

- **G1**: smoke test by writing a metrics line to a fake run dir while the dev server is up; visually confirm chart appends.
- **G2**: unit tests in `tests/test_backtest.py` for `BacktestSeries`. Visual check on detail page using existing `runs/ppo_100k`.
- **G3**: visual check by manually renaming a few run dirs into a `compare_rewards_X/{...}` shape and verifying grouping.

Round 2 will add Vitest + RTL coverage for `lib/runs.ts` and components.

## Risks & open questions

| Item | Status |
|---|---|
| Browser EventSource not supported in all environments | Accept — single local user, modern Chrome/Edge |
| File system polling 2s is too aggressive on Windows? | Acceptable; synthetic_demo runs ~340 fps → metric writes every few seconds, quiet during pauses |
| What if user starts streaming a run that already finished? | Server detects mtime > 10s, sends 0 events, client EventSource just stays idle. No special handling. |
| ★ "LIVE" badge accuracy if user pauses training and resumes | mtime resets on next write — accepted edge case |
| ★ Backtest series file size | ~1KB per run for 252 trading days × 5 fields × float64. Negligible. |
| ★ G3 grouping heuristic too aggressive | If a user happens to have unrelated nested run dirs, they'd be grouped. Mitigation: only group when *all* children share a recognizable parent (e.g. parent dir name itself doesn't have a `training_summary.json`). Implemented by current `walkRunDirs` in lib/runs.ts. |

★ marks items I'm confident about; non-marked items are accepted constraints.

## Decisions to revisit (when user returns)

- D1 SSE vs WebSocket
- D2 2 s polling cadence
- D3 10 s LIVE threshold
- D4 separate `backtest_series.json` file
- D5 structural grouping heuristic
- D6 client-side filter

If any of these need to flip, rework is local to the affected component and won't ripple.
