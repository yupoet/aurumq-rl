# SL Path 1 — LightGBM β-regression Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a LightGBM regression ensemble on the P3 4070 bundle that beats `paris's P2 v2 baseline` (= the user's prior 5-seed LightGBM, PR_AUC 0.110 H1 / 0.146 H2) on the new primary metric `mean_top50_proximity_excess` for H1+H2 windows.

**Architecture:** Five-stage pipeline. (1) compute proximity-weighted target `y_t` from realized_returns via self-shift; (2) pure-function eval module with metrics defined by spec §3; (3) single-config training script; (4) orchestrator running 12 hyperparam configs × 3 seeds = 36 LightGBM runs; (5) ensemble (top-3 by VAL_EFF primary metric, seed-mean) + isotonic calibration on H1, eval H2. CPU only (LightGBM, polars). Tests pin formula correctness with small synthetic fixtures before any heavy training runs.

**Tech Stack:** Python 3.11 · polars 1.40 · lightgbm 4.6 · scikit-learn 1.8 (isotonic + metrics) · numpy · pytest

---

## Context

Why this plan exists:
1. The team's prior Path (P3 PPO residual) is dropped per `runs/p3_findings/RESULTS.md`. The P3 bundle (344 features × 3.78 M rows × 2023-2026 main-board) is a clean SL substrate.
2. Spec at `docs/superpowers/specs/2026-05-09-sl-ensemble-training-design.md` defines a new continuous target `y_t = (1.0×excess_T+1 + 0.6×excess_T+2 + 0.3×excess_T+3) / 1.9` with `max(0, ·)` clipping. This encodes hit + magnitude + proximity in one continuous regression target.
3. Path 1 establishes the LightGBM baseline before introducing feature engineering (Path 4) and other model classes (Paths 2, 3). It must produce a clean, calibrated artifact that downstream paths build on.
4. Existing `scripts/p3/sl_baseline.py` is the binary-classification predecessor and stays as-is for reference; new Path 1 files live under `scripts/p3/path1_*.py`.

Pre-existing infra used by this plan:
- `data/p3_4070/_bundle_cache.npz` — 11s cache load (built earlier session)
- `aurumq_rl.p3.load_bundle` — date-fix patched, returns `P3Bundle` with realized + market arrays
- 36 logical CPU cores, 68 GB RAM, ~32 GB free
- LightGBM, scikit-learn, polars all installed in `.venv`

---

## File Structure

**Created (source):**
- `scripts/p3/path1_target.py` — compute `target_y.parquet` from `realized_returns.parquet`
- `scripts/p3/path1_eval.py` — pure-function eval metrics module + CLI evaluator
- `scripts/p3/path1_train.py` — train one LightGBM config; outputs results.json + lgb_model.txt + predictions.npz
- `scripts/p3/path1_grid.py` — subprocess orchestrator running 36 configs
- `scripts/p3/path1_ensemble.py` — pick top-3 configs, seed-mean, isotonic-calibrate, write final predictions

**Created (tests):**
- `tests/p3/__init__.py` — empty marker
- `tests/p3/test_path1_target.py` — verify `y_t` formula on small fixture
- `tests/p3/test_path1_eval.py` — verify metrics on synthetic predictions
- `tests/p3/test_path1_ensemble.py` — verify isotonic calibration + seed-mean

**Created (artifacts during execution):**
- `data/p3_4070/target_y.parquet` (~50 MB) — proximity-weighted target, gitignored
- `runs/sl_path1/<config>_seed<s>/{results.json, lgb_model.txt, predictions.npz}` × 36
- `runs/sl_path1/ensemble.json` — final ensemble metrics
- `runs/sl_path1/predictions.parquet` — (trade_date, ts_code, score_raw, score_calibrated)
- `runs/sl_path1/RESULTS.md` — final report

**Untouched but referenced:**
- `data/p3_4070/realized_returns.parquet` — (trade_date, ts_code, pct_chg_t_plus_1)
- `data/p3_4070/market_returns.parquet` — (trade_date, eq_weight_pct_chg_t_plus_1)
- `data/p3_4070/feature_panel_v3_344.parquet` — features
- `data/p3_4070/universe_mask/year=*.parquet` — main-board non-ST filter
- `data/p3_4070/labels/labels_A_t3_year=*.parquet` — sanity cross-check only
- `data/p3_4070/baseline_predictions.parquet` — paris's P2 v2 predictions for comparison

---

## Phase 0: Environment + scaffolding

### Task 0.1: Create `tests/p3/` directory marker

**Files:**
- Create: `tests/p3/__init__.py` (empty)

- [ ] **Step 1: Create directory**

```bash
mkdir -p tests/p3
```

- [ ] **Step 2: Create empty `__init__.py`**

```python
# tests/p3/__init__.py
```

- [ ] **Step 3: Verify pytest discovers the package**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/ --collect-only 2>&1 | tail -5
```

Expected: `0 tests collected` (no test files yet) without any collection error.

- [ ] **Step 4: Commit**

```bash
git add tests/p3/__init__.py
git commit -m "test(p3): scaffold tests/p3/ package for SL Path 1"
```

---

## Phase 1: target_y formula

The hardest correctness risk in this plan: getting T+2 and T+3 alignment right via self-shift on `realized_returns.parquet`. We pin it with a unit test on a hand-constructed fixture before touching the real bundle.

### Task 1.1: Pure function `compute_target_y`

**Files:**
- Create: `tests/p3/test_path1_target.py`
- Create: `scripts/p3/path1_target.py`

- [ ] **Step 1: Write the failing test**

`tests/p3/test_path1_target.py`:

```python
"""Tests for proximity-weighted target_y formula (spec §2)."""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from path1_target import compute_target_y


def _mk_realized(rows):
    """Helper: rows = list of (date_iso, ts_code, pct_chg_t_plus_1) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, r) for d, c, r in rows],
        schema=["trade_date", "ts_code", "pct_chg_t_plus_1"],
        orient="row",
    )


def _mk_market(rows):
    """Helper: rows = list of (date_iso, eq_weight_pct_chg_t_plus_1) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), r) for d, r in rows],
        schema=["trade_date", "eq_weight_pct_chg_t_plus_1"],
        orient="row",
    )


def test_y_zero_when_excess_all_negative():
    """All forward excess returns negative → max(0,·) clips to 0 → y=0."""
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", -0.01),  # T+1 excess = -0.01 - 0.005 = -0.015
        ("2024-01-03", "600001.SH", -0.02),  # T+2 (= 2024-01-03's T+1)
        ("2024-01-04", "600001.SH", -0.03),  # T+3
    ])
    market = _mk_market([
        ("2024-01-02", 0.005),
        ("2024-01-03", 0.005),
        ("2024-01-04", 0.005),
    ])
    out = compute_target_y(realized, market)
    # Anchor row trade_date=2024-01-02 looks at T+1 (own row),
    # T+2 (2024-01-03's row), T+3 (2024-01-04's row).
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.0)


def test_y_proximity_weighted_when_only_t_plus_1_positive():
    """T+1 excess = +0.05, T+2/T+3 excess = 0 (after market subtraction).

    Expected: y = (1.0 * 0.05 + 0.6 * 0 + 0.3 * 0) / 1.9 = 0.05 / 1.9 ≈ 0.02632
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),     # T+1: 0.05 - 0 = +0.05
        ("2024-01-03", "600001.SH", 0.0),      # T+2: 0 - 0 = 0
        ("2024-01-04", "600001.SH", 0.0),      # T+3
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.05 / 1.9, abs=1e-6)


def test_y_full_proximity_pattern():
    """T+1 = +0.04, T+2 = +0.02, T+3 = +0.01 (all excess after market).

    Expected: y = (1.0*0.04 + 0.6*0.02 + 0.3*0.01) / 1.9
              = (0.04 + 0.012 + 0.003) / 1.9
              = 0.055 / 1.9 ≈ 0.02895
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.04),
        ("2024-01-03", "600001.SH", 0.02),
        ("2024-01-04", "600001.SH", 0.01),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.055 / 1.9, abs=1e-6)


def test_y_at_panel_boundary_drops_when_t_plus_3_missing():
    """For trade_dates near the end of the panel, T+3 doesn't exist → drop them.

    With a 3-date fixture (Jan2, Jan3, Jan4) and the convention that
    T+1 == anchor's own row, T+2 == anchor+1's row, T+3 == anchor+2's row:
      - Jan2: T+1=Jan2 ✓  T+2=Jan3 ✓  T+3=Jan4 ✓  → keep
      - Jan3: T+1=Jan3 ✓  T+2=Jan4 ✓  T+3=Jan5 (missing) → drop
      - Jan4: T+2=Jan5 (missing) → drop

    Output should contain ONLY Jan2.
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.01),
        ("2024-01-03", "600001.SH", 0.01),
        ("2024-01-04", "600001.SH", 0.01),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    out_dates = sorted(out["trade_date"].unique().to_list())
    assert out_dates == [dt.date(2024, 1, 2)]


def test_y_max_zero_clipping_per_horizon():
    """T+1 = +0.05, T+2 = -0.03, T+3 = +0.02.

    Each horizon clipped at 0 BEFORE weighting: T+2 contributes 0 not -0.03.
    Expected: y = (1.0*0.05 + 0.6*0 + 0.3*0.02) / 1.9 = (0.05 + 0 + 0.006)/1.9
             ≈ 0.02947
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-03", "600001.SH", -0.03),
        ("2024-01-04", "600001.SH", 0.02),
    ])
    market = _mk_market([
        ("2024-01-02", 0.0),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx((0.05 + 0.006) / 1.9, abs=1e-6)


def test_y_subtracts_market_per_horizon():
    """T+1 = +0.05, market T+1 = +0.02 → excess T+1 = +0.03.

    Per-horizon market subtraction (not constant market across the window).
    Expected: y = (1.0 * 0.03 + 0 + 0) / 1.9 ≈ 0.01579
    """
    realized = _mk_realized([
        ("2024-01-02", "600001.SH", 0.05),
        ("2024-01-03", "600001.SH", 0.0),
        ("2024-01-04", "600001.SH", 0.0),
    ])
    market = _mk_market([
        ("2024-01-02", 0.02),
        ("2024-01-03", 0.0),
        ("2024-01-04", 0.0),
    ])
    out = compute_target_y(realized, market)
    row = out.filter(pl.col("trade_date") == dt.date(2024, 1, 2)).row(0, named=True)
    assert row["y"] == pytest.approx(0.03 / 1.9, abs=1e-6)
```

- [ ] **Step 2: Run test, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_target.py -v 2>&1 | tail -10
```

Expected: ImportError on `from path1_target import compute_target_y`.

- [ ] **Step 3: Implement `compute_target_y`**

`scripts/p3/path1_target.py`:

```python
"""Compute proximity-weighted target_y from realized_returns + market_returns.

Per spec §2:
    y_t = (1.0 * max(0, excess_T+1)
         + 0.6 * max(0, excess_T+2)
         + 0.3 * max(0, excess_T+3)) / 1.9

where excess_T+d = pct_chg_T+d - eq_weight_market_pct_T+d for each (date, stock).

T+1 comes directly from realized_returns. T+2 and T+3 are obtained by
self-shifting the same table on trade_date: at anchor trade_date D,
T+2 == realized_returns[trade_date == D's next trading day]'s pct_chg_t_plus_1.

Rows where T+3 (i.e. anchor date's third forward trading day) is missing
in the panel are DROPPED — no partial-window y is emitted.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl


logger = logging.getLogger(__name__)


W1, W2, W3 = 1.0, 0.6, 0.3
W_SUM = W1 + W2 + W3  # 1.9


def compute_target_y(
    realized: pl.DataFrame,
    market: pl.DataFrame,
) -> pl.DataFrame:
    """Build (trade_date, ts_code, y) frame from realized + market frames.

    Parameters
    ----------
    realized : pl.DataFrame
        Schema: trade_date, ts_code, pct_chg_t_plus_1 (float).
        Each row's ``pct_chg_t_plus_1`` is the t+1 close-to-close return for
        that stock from the next trading day after ``trade_date``.
    market : pl.DataFrame
        Schema: trade_date, eq_weight_pct_chg_t_plus_1 (float).

    Returns
    -------
    pl.DataFrame  schema: trade_date, ts_code, y (float)
        Anchored at the original trade_date. Rows missing T+2 or T+3 in
        the panel are dropped.
    """
    # 1. Build a (date → ordinal index) for self-shift via inner-join.
    # Using a sequential rank on the unique sorted dates lets us shift by
    # +1 / +2 to find the next-trading-day, regardless of weekends/holidays.
    all_dates = (
        realized.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_row_index("date_idx")
    )
    realized = realized.join(all_dates, on="trade_date")
    market = market.join(all_dates, on="trade_date")

    # 2. Per-stock self-join at date_idx + 1 (T+2 anchor) and date_idx + 2 (T+3 anchor).
    realized_t2 = realized.select(
        pl.col("date_idx").alias("anchor_idx"),  # this row provides T+2 for anchor_idx - 1
        pl.col("ts_code"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_2"),
    ).with_columns(pl.col("anchor_idx") - 1)

    realized_t3 = realized.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("ts_code"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_3"),
    ).with_columns(pl.col("anchor_idx") - 2)

    market_t2 = market.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_2"),
    ).with_columns(pl.col("anchor_idx") - 1)

    market_t3 = market.select(
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_3"),
    ).with_columns(pl.col("anchor_idx") - 2)

    # 3. Anchor on realized's original rows (renamed for clarity), join the shifts.
    base = realized.select(
        pl.col("trade_date"),
        pl.col("ts_code"),
        pl.col("date_idx").alias("anchor_idx"),
        pl.col("pct_chg_t_plus_1").alias("pct_t_plus_1"),
    ).join(
        market.select(
            pl.col("date_idx").alias("anchor_idx"),
            pl.col("eq_weight_pct_chg_t_plus_1").alias("market_t_plus_1"),
        ),
        on="anchor_idx",
        how="inner",
    )
    # T+2 / T+3 must be inner-joined so we drop anchors lacking those views.
    base = base.join(realized_t2, on=["anchor_idx", "ts_code"], how="inner")
    base = base.join(realized_t3, on=["anchor_idx", "ts_code"], how="inner")
    base = base.join(market_t2, on="anchor_idx", how="inner")
    base = base.join(market_t3, on="anchor_idx", how="inner")

    # 4. Compute excess per horizon, clip max(0, ·), weight + sum.
    out = base.with_columns(
        (pl.col("pct_t_plus_1") - pl.col("market_t_plus_1")).alias("e1"),
        (pl.col("pct_t_plus_2") - pl.col("market_t_plus_2")).alias("e2"),
        (pl.col("pct_t_plus_3") - pl.col("market_t_plus_3")).alias("e3"),
    ).with_columns(
        pl.max_horizontal(pl.lit(0.0), pl.col("e1")).alias("p1"),
        pl.max_horizontal(pl.lit(0.0), pl.col("e2")).alias("p2"),
        pl.max_horizontal(pl.lit(0.0), pl.col("e3")).alias("p3"),
    ).with_columns(
        ((W1 * pl.col("p1") + W2 * pl.col("p2") + W3 * pl.col("p3")) / W_SUM).alias("y")
    ).select(["trade_date", "ts_code", "y"])

    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: read realized_returns + market_returns from a bundle dir, write target_y.parquet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Defaults to <bundle>/target_y.parquet")
    args = ap.parse_args(argv)

    out_path = args.out or (args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    logger.info("realized: %d rows  market: %d rows", len(realized), len(market))

    out = compute_target_y(realized, market)
    logger.info("output: %d rows  dates=%d  stocks=%d  y_mean=%.6f  y_std=%.6f",
                len(out), out["trade_date"].n_unique(), out["ts_code"].n_unique(),
                out["y"].mean(), out["y"].std())

    out.write_parquet(out_path, compression="zstd", compression_level=10)
    logger.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Make `path1_target` importable from tests**

The test imports `from path1_target import compute_target_y`. Add an `__init__.py` shim or use sys.path injection. Cleanest: extend pytest's `conftest.py` (project root) to add `scripts/p3` to `sys.path`. Confirm what the existing conftest does:

```bash
head -30 tests/conftest.py 2>&1
```

If `tests/conftest.py` already adds `scripts/` to `sys.path`, just import as `from p3.path1_target import compute_target_y` (adjust test). Otherwise:

`tests/conftest.py` already adds project's `src/` and `scripts/` to `sys.path` (verified during plan creation). The test imports `from path1_target` directly only if `scripts/p3` is on the path. Use `from p3.path1_target import compute_target_y` instead.

**Edit the test imports** in `tests/p3/test_path1_target.py`:

Replace:
```python
from path1_target import compute_target_y
```
With:
```python
from p3.path1_target import compute_target_y
```

Then ensure `scripts/p3/__init__.py` exists:

```bash
test -f scripts/p3/__init__.py || touch scripts/p3/__init__.py
ls scripts/p3/__init__.py
```

- [ ] **Step 5: Run test, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_target.py -v 2>&1 | tail -15
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/p3/__init__.py scripts/p3/path1_target.py tests/p3/test_path1_target.py
git commit -m "feat(sl-path1): proximity-weighted target_y formula + tests

y_t = (1.0×max(0,e1) + 0.6×max(0,e2) + 0.3×max(0,e3)) / 1.9 per spec §2.
Self-shift on trade_date for T+2 / T+3, inner-join drops boundary rows.
6 unit tests pin formula correctness on synthetic fixtures."
```

### Task 1.2: Build `target_y.parquet` on the real bundle

**Files:**
- Run: `scripts/p3/path1_target.py`

- [ ] **Step 1: Build target on real bundle**

```bash
.venv/Scripts/python.exe scripts/p3/path1_target.py --bundle data/p3_4070 2>&1 | tail -5
```

Expected output (approximate):
```
[INFO] realized: 2180517 rows  market: 727 rows
[INFO] output: ~2100000 rows  dates=~720  stocks=~3000  y_mean=~0.001  y_std=~0.005
[INFO] wrote data/p3_4070/target_y.parquet (~50 MB)
```

- [ ] **Step 2: Sanity-check distribution**

```bash
.venv/Scripts/python.exe -c "
import polars as pl
df = pl.read_parquet('data/p3_4070/target_y.parquet')
print(f'rows: {len(df):,}')
print(f'y stats: min={df[\"y\"].min():.6f}  mean={df[\"y\"].mean():.6f}  std={df[\"y\"].std():.6f}  max={df[\"y\"].max():.6f}')
print(f'fraction y>0: {(df[\"y\"] > 0).sum() / len(df):.4f}')
print(f'fraction y>0.01: {(df[\"y\"] > 0.01).sum() / len(df):.4f}')
print(f'fraction y>0.05: {(df[\"y\"] > 0.05).sum() / len(df):.4f}')
"
```

Expected: `min=0.0` (clipped), `mean ~0.001-0.005`, `max ~0.05-0.10` (capped by daily price-limit + clipping). `fraction y>0 ~ 30-50%` (most days some stock has positive forward excess). If `y.max()` exceeds 0.20 something is off.

- [ ] **Step 3: Commit nothing (data file is gitignored)**

`data/p3_4070/` is already in `.gitignore`. Just verify:

```bash
git status data/p3_4070/target_y.parquet
```

Expected: file is untracked / ignored, not staged.

---

## Phase 2: eval metrics module

### Task 2.1: Pure metric functions with tests

**Files:**
- Create: `tests/p3/test_path1_eval.py`
- Create: `scripts/p3/path1_eval.py`

- [ ] **Step 1: Write the failing test**

`tests/p3/test_path1_eval.py`:

```python
"""Tests for Path 1 eval metrics module (spec §3)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from p3.path1_eval import (
    compute_ece_10bin,
    compute_mean_top50_proximity_excess,
    compute_spearman,
    compute_top50_hit_rates,
)


def _mk_eval_frame(rows):
    """rows = list of (date_iso, ts_code, score, actual_y) tuples."""
    return pl.DataFrame(
        [(dt.date.fromisoformat(d), c, s, y) for d, c, s, y in rows],
        schema=["trade_date", "ts_code", "score", "actual_y"],
        orient="row",
    )


def test_mean_top50_proximity_excess_perfect_predictor():
    """Score == actual_y → top-50 each day are exactly the highest-y stocks.

    Build 2 dates × 100 stocks. y = uniform(0, 0.1). score = y exactly.
    Expected: mean_top50_proximity_excess = mean of top-50 actual_y per day,
              averaged across the 2 dates.
    """
    rng = np.random.default_rng(0)
    rows = []
    for d_iso in ("2025-08-01", "2025-08-02"):
        for j in range(100):
            y = float(rng.uniform(0, 0.1))
            rows.append((d_iso, f"S{j:03}.SH", y, y))  # score = y
    df = _mk_eval_frame(rows)

    result = compute_mean_top50_proximity_excess(df, top_k=50)
    # Compute expected manually
    expected = float(
        np.mean([
            df.filter(pl.col("trade_date") == dt.date.fromisoformat(d))
              .sort("score", descending=True).head(50)["actual_y"].mean()
            for d in ("2025-08-01", "2025-08-02")
        ])
    )
    assert result == pytest.approx(expected, abs=1e-9)


def test_mean_top50_proximity_excess_random_predictor():
    """Random score → top-50 daily averages should be close to overall mean of y."""
    rng = np.random.default_rng(1)
    rows = []
    for d_iso in ("2025-08-01",):
        ys = rng.uniform(0, 0.1, size=200)
        scores = rng.uniform(0, 1, size=200)
        for j in range(200):
            rows.append((d_iso, f"S{j:03}.SH", float(scores[j]), float(ys[j])))
    df = _mk_eval_frame(rows)

    result = compute_mean_top50_proximity_excess(df, top_k=50)
    overall_mean = df["actual_y"].mean()
    # Within +/- 30% of overall mean (50/200 = 25% sample, expect close to mean)
    assert abs(result - overall_mean) < 0.3 * overall_mean


def test_spearman_perfect_correlation():
    """score = actual_y exactly → spearman = 1.0."""
    rows = [
        ("2025-08-01", "A.SH", 0.01, 0.01),
        ("2025-08-01", "B.SH", 0.02, 0.02),
        ("2025-08-01", "C.SH", 0.03, 0.03),
        ("2025-08-01", "D.SH", 0.04, 0.04),
    ]
    df = _mk_eval_frame(rows)
    rho = compute_spearman(df)
    assert rho == pytest.approx(1.0, abs=1e-9)


def test_spearman_anti_correlation():
    """score = -actual_y → spearman = -1.0."""
    rows = [
        ("2025-08-01", "A.SH", 0.04, 0.01),
        ("2025-08-01", "B.SH", 0.03, 0.02),
        ("2025-08-01", "C.SH", 0.02, 0.03),
        ("2025-08-01", "D.SH", 0.01, 0.04),
    ]
    df = _mk_eval_frame(rows)
    rho = compute_spearman(df)
    assert rho == pytest.approx(-1.0, abs=1e-9)


def test_top50_hit_rates_returns_three_values():
    """Hit-rate function returns (T1_hit, T13_hit, T1_avg_excess) tuple of floats."""
    # Need realized_excess columns: e1 (T+1 excess), e2 (T+2), e3 (T+3)
    # All in the eval frame.
    rng = np.random.default_rng(2)
    rows = []
    for d_iso in ("2025-08-01",):
        for j in range(100):
            score = float(rng.uniform(0, 1))
            e1 = float(rng.uniform(-0.05, 0.05))
            e2 = float(rng.uniform(-0.05, 0.05))
            e3 = float(rng.uniform(-0.05, 0.05))
            rows.append((d_iso, f"S{j:03}.SH", score, e1, e2, e3))
    df = pl.DataFrame(
        [(dt.date.fromisoformat(d), c, s, e1, e2, e3) for d, c, s, e1, e2, e3 in rows],
        schema=["trade_date", "ts_code", "score", "e1", "e2", "e3"],
        orient="row",
    )
    out = compute_top50_hit_rates(df, top_k=50)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"top50_T1_hit_rate", "top50_T13_hit_rate", "top50_T1_avg_excess"}
    assert 0.0 <= out["top50_T1_hit_rate"] <= 1.0
    assert 0.0 <= out["top50_T13_hit_rate"] <= 1.0


def test_ece_perfect_calibration():
    """If predicted = actual, ECE on 10-bin should be ~0."""
    rng = np.random.default_rng(3)
    n = 5000
    actual = rng.uniform(0, 0.1, size=n)
    pred = actual.copy()  # perfect calibration
    ece = compute_ece_10bin(pred, actual)
    assert ece < 1e-9


def test_ece_constant_prediction_far_from_actual_mean_high_error():
    """Constant prediction far from actual.mean() → all rows in one bin, large |pred-actual|.

    Standard ECE bins by PREDICTION quantile. With a constant prediction, all
    rows fall in one bin and bin's |mean(pred) - mean(actual)| is the gap
    between the constant and the actual mean. Use pred = 0.5 vs actual ~ 0.05
    → ECE ≈ 0.45.
    """
    rng = np.random.default_rng(4)
    n = 5000
    actual = rng.uniform(0, 0.1, size=n)  # mean ≈ 0.05
    pred = np.full(n, 0.5)  # constant, far from actual mean
    ece = compute_ece_10bin(pred, actual)
    assert ece > 0.4  # single bin, |0.5 - 0.05| ≈ 0.45
```

- [ ] **Step 2: Run test, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_eval.py -v 2>&1 | tail -10
```

Expected: ImportError on `from p3.path1_eval import ...`.

- [ ] **Step 3: Implement metrics module**

`scripts/p3/path1_eval.py`:

```python
"""Path 1 eval metrics (spec §3).

All metrics here are pure functions over polars DataFrames or numpy arrays.
The CLI at the bottom is a thin wrapper that joins (predictions, target_y,
realized_returns) and prints the metric block.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pure metric functions
# ------------------------------------------------------------------ #

def compute_mean_top50_proximity_excess(df: pl.DataFrame, top_k: int = 50) -> float:
    """Mean across days of the daily mean-actual-y over top-k by score.

    df: (trade_date, ts_code, score, actual_y). NaN actual_y rows excluded.
    """
    df = df.drop_nulls(["score", "actual_y"])
    if len(df) == 0:
        return 0.0
    daily_means = (
        df.sort(["trade_date", "score"], descending=[False, True])
          .group_by("trade_date", maintain_order=True)
          .head(top_k)
          .group_by("trade_date", maintain_order=True)
          .agg(pl.col("actual_y").mean().alias("topk_mean"))
    )
    return float(daily_means["topk_mean"].mean())


def compute_spearman(df: pl.DataFrame) -> float:
    """Spearman rank correlation between score and actual_y across all rows."""
    df = df.drop_nulls(["score", "actual_y"])
    if len(df) < 2:
        return 0.0
    rho, _ = spearmanr(df["score"].to_numpy(), df["actual_y"].to_numpy())
    if not np.isfinite(rho):
        return 0.0
    return float(rho)


def compute_top50_hit_rates(df: pl.DataFrame, top_k: int = 50) -> dict[str, float]:
    """Top-k hit decomposition (spec §3.3).

    df must have columns: trade_date, ts_code, score, e1 (T+1 excess), e2, e3.
    """
    df = df.drop_nulls(["score", "e1", "e2", "e3"])
    if len(df) == 0:
        return {"top50_T1_hit_rate": 0.0, "top50_T13_hit_rate": 0.0, "top50_T1_avg_excess": 0.0}
    topk = (
        df.sort(["trade_date", "score"], descending=[False, True])
          .group_by("trade_date", maintain_order=True)
          .head(top_k)
    )
    return {
        "top50_T1_hit_rate": float((topk["e1"] > 0).mean()),
        "top50_T13_hit_rate": float(((topk["e1"] > 0) | (topk["e2"] > 0) | (topk["e3"] > 0)).mean()),
        "top50_T1_avg_excess": float(topk["e1"].mean()),
    }


def compute_ece_10bin(pred: np.ndarray, actual: np.ndarray) -> float:
    """Expected Calibration Error binned to 10 quantiles of predictions.

    For each prediction-quantile bin, compare bin's mean prediction vs bin's
    mean actual. Weighted by bin size.
    """
    pred = np.asarray(pred, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[mask], actual[mask]
    if len(pred) < 10:
        return 0.0

    quantiles = np.quantile(pred, np.linspace(0, 1, 11))
    quantiles[0] -= 1e-9  # ensure leftmost edge captures the min
    quantiles[-1] += 1e-9
    bin_idx = np.digitize(pred, quantiles[1:-1], right=False)  # 0..9
    n = len(pred)
    ece = 0.0
    for b in range(10):
        in_bin = bin_idx == b
        if not in_bin.any():
            continue
        ece += (in_bin.sum() / n) * abs(pred[in_bin].mean() - actual[in_bin].mean())
    return float(ece)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

H1 = (dt.date(2025, 7, 1), dt.date(2025, 9, 30))
H2 = (dt.date(2025, 10, 1), dt.date(2025, 12, 31))


def evaluate(
    predictions: pl.DataFrame,    # (trade_date, ts_code, score)
    target_y: pl.DataFrame,       # (trade_date, ts_code, y) — built by path1_target
    realized: pl.DataFrame,       # full bundle realized_returns (for e1/e2/e3 lookup)
    market: pl.DataFrame,         # bundle market_returns
    window: tuple[dt.date, dt.date],
    top_k: int = 50,
) -> dict:
    """Compute the §3 metric block on a (predictions, target_y) join over `window`.

    Joins predictions ⨝ target_y on (trade_date, ts_code) inner; restricts
    to ``window``. Computes T+1/T+2/T+3 excess returns inline for hit-rate.
    """
    lo, hi = window
    pred_w = predictions.filter((pl.col("trade_date") >= lo) & (pl.col("trade_date") <= hi))
    y_w = target_y.filter((pl.col("trade_date") >= lo) & (pl.col("trade_date") <= hi))
    df = pred_w.join(y_w, on=["trade_date", "ts_code"], how="inner")
    df = df.rename({"y": "actual_y"})

    # Build e1/e2/e3 by self-shift (mirrors path1_target logic, unweighted).
    all_dates = realized.select("trade_date").unique().sort("trade_date").with_row_index("date_idx")
    realized_w_idx = realized.join(all_dates, on="trade_date")
    market_w_idx = market.join(all_dates, on="trade_date")

    def _shift(rdf: pl.DataFrame, k: int, alias: str) -> pl.DataFrame:
        return rdf.select(
            (pl.col("date_idx") - k).alias("anchor_idx"),
            pl.col("ts_code"),
            pl.col("pct_chg_t_plus_1").alias(alias),
        )

    def _shift_market(mdf: pl.DataFrame, k: int, alias: str) -> pl.DataFrame:
        return mdf.select(
            (pl.col("date_idx") - k).alias("anchor_idx"),
            pl.col("eq_weight_pct_chg_t_plus_1").alias(alias),
        )

    df = df.join(all_dates, on="trade_date").rename({"date_idx": "anchor_idx"})
    df = df.join(_shift(realized_w_idx, 0, "pct_t_plus_1"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift(realized_w_idx, 1, "pct_t_plus_2"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift(realized_w_idx, 2, "pct_t_plus_3"), on=["anchor_idx", "ts_code"], how="left")
    df = df.join(_shift_market(market_w_idx, 0, "market_t_plus_1"), on="anchor_idx", how="left")
    df = df.join(_shift_market(market_w_idx, 1, "market_t_plus_2"), on="anchor_idx", how="left")
    df = df.join(_shift_market(market_w_idx, 2, "market_t_plus_3"), on="anchor_idx", how="left")
    df = df.with_columns(
        (pl.col("pct_t_plus_1") - pl.col("market_t_plus_1")).alias("e1"),
        (pl.col("pct_t_plus_2") - pl.col("market_t_plus_2")).alias("e2"),
        (pl.col("pct_t_plus_3") - pl.col("market_t_plus_3")).alias("e3"),
    )

    # Drop rows missing any horizon (boundary).
    df = df.drop_nulls(["e1", "e2", "e3", "actual_y", "score"])

    primary = compute_mean_top50_proximity_excess(df, top_k=top_k)
    spearman = compute_spearman(df)
    hit = compute_top50_hit_rates(df, top_k=top_k)
    ece = compute_ece_10bin(df["score"].to_numpy(), df["actual_y"].to_numpy())

    return {
        "n_rows": len(df),
        "n_dates": df["trade_date"].n_unique(),
        "primary_mean_top50_proximity_excess": primary,
        "spearman": spearman,
        "ece_10bin": ece,
        **hit,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: print eval metrics for a predictions parquet on H1 and H2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, type=Path,
                    help="Parquet with (trade_date, ts_code, score) columns")
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional JSON output of the metric blocks")
    args = ap.parse_args(argv)

    predictions = pl.read_parquet(args.predictions)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )

    h1 = evaluate(predictions, target_y, realized, market, H1, args.top_k)
    h2 = evaluate(predictions, target_y, realized, market, H2, args.top_k)

    out = {"H1": h1, "H2": h2}
    print(json.dumps(out, indent=2, default=str))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str))
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_eval.py -v 2>&1 | tail -15
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/p3/path1_eval.py tests/p3/test_path1_eval.py
git commit -m "feat(sl-path1): pure-function eval metrics module + 7 tests

Implements spec §3 metrics:
- mean_top50_proximity_excess (primary, decision-aligned)
- spearman(score, actual_y) (ranking quality)
- top50 hit-rate decomposition (T1 / T13 / T1_avg_excess)
- ECE 10-bin (calibration)
- evaluate(...) CLI joins predictions ⨝ target_y on H1/H2 windows."
```

---

## Phase 3: train one config

### Task 3.1: Single-config LightGBM trainer

**Files:**
- Create: `scripts/p3/path1_train.py`

The training step is integration-heavy (loads features, joins target, fits LightGBM, predicts on multiple windows). Skip TDD — verify by running the smallest possible config and inspecting outputs.

- [ ] **Step 1: Implement trainer**

`scripts/p3/path1_train.py`:

```python
"""Path 1 — train ONE LightGBM regression config on proximity-weighted target.

Usage::

    python scripts/p3/path1_train.py \\
        --bundle data/p3_4070 \\
        --out runs/sl_path1/numleaves63_lr05_minleaf100_seed42 \\
        --num-leaves 63 --learning-rate 0.05 --min-data-in-leaf 100 --seed 42
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

# Path 1 modules (sibling)
from p3.path1_eval import H1, H2, evaluate


logger = logging.getLogger(__name__)


TRAIN_EFF = (dt.date(2023, 1, 3),  dt.date(2024, 12, 4))
VAL_EFF   = (dt.date(2025, 1, 1),  dt.date(2025, 6, 4))


def _load_features_universe(bundle: Path) -> tuple[pl.DataFrame, list[str]]:
    df = pl.read_parquet(bundle / "feature_panel_v3_344.parquet")
    feature_cols = [c for c in df.columns if c not in ("ts_code", "trade_date")]
    uni_parts = []
    for year in (2023, 2024, 2025, 2026):
        p = bundle / "universe_mask" / f"year={year}.parquet"
        if p.exists():
            uni_parts.append(pl.read_parquet(p).select(["trade_date", "ts_code", "in_universe"]))
    uni = pl.concat(uni_parts)
    df = df.join(uni, on=["trade_date", "ts_code"], how="left").filter(
        pl.col("in_universe") == True  # noqa: E712
    )
    return df, feature_cols


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Load features (in-universe only) and target_y
    t0 = time.time()
    feat_df, feature_cols = _load_features_universe(args.bundle)
    logger.info("features: %d rows × %d cols (%.1fs)", len(feat_df), len(feature_cols), time.time() - t0)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    logger.info("target_y: %d rows", len(target_y))

    df = feat_df.join(target_y, on=["trade_date", "ts_code"], how="inner")
    logger.info("joined: %d rows", len(df))

    # 2. Split by trade_date
    train_df = df.filter(
        (pl.col("trade_date") >= TRAIN_EFF[0]) & (pl.col("trade_date") <= TRAIN_EFF[1])
    )
    val_df = df.filter(
        (pl.col("trade_date") >= VAL_EFF[0]) & (pl.col("trade_date") <= VAL_EFF[1])
    )
    logger.info("splits: train=%d val=%d", len(train_df), len(val_df))

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df["y"].to_numpy().astype(np.float32)
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df["y"].to_numpy().astype(np.float32)

    # 3. Train LightGBM regression
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_ds)
    params = {
        "objective": "regression_l2",
        "metric": ["l2", "l1"],
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": args.min_data_in_leaf,
        "verbosity": -1,
        "seed": args.seed,
        "n_jobs": -1,
    }
    logger.info("training: %s", {k: v for k, v in params.items() if k != "metric"})
    t1 = time.time()
    model = lgb.train(
        params, train_ds,
        num_boost_round=args.num_iterations,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t1
    logger.info("trained in %.1fs (best_iter=%d)", train_time, model.best_iteration)

    # 4. Predict on full eval frame (TRAIN union VAL union H1 union H2 — all stocks all dates).
    # Use the universe-filtered df from step 1; predict on everything we have features for.
    X_all = df.select(feature_cols).to_numpy()
    score_all = model.predict(X_all, num_iteration=model.best_iteration).astype(np.float32)
    pred_df = df.select(["trade_date", "ts_code"]).with_columns(pl.Series("score", score_all))

    # 5. Save artifacts
    pred_df.write_parquet(args.out / "predictions.parquet", compression="zstd", compression_level=9)
    model.save_model(str(args.out / "lgb_model.txt"))

    # Cache numpy too for downstream rank/ensemble fast path
    np.savez(
        args.out / "predictions.npz",
        score=score_all,
    )

    # 6. Eval on VAL_EFF, H1, H2
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    val_eval = evaluate(pred_df, target_y, realized, market, VAL_EFF)
    h1_eval = evaluate(pred_df, target_y, realized, market, H1)
    h2_eval = evaluate(pred_df, target_y, realized, market, H2)

    summary = {
        "params": params,
        "best_iteration": model.best_iteration,
        "train_time_s": train_time,
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "VAL_EFF": val_eval,
        "H1": h1_eval,
        "H2": h2_eval,
    }
    (args.out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("VAL primary=%.6f  H1 primary=%.6f  H2 primary=%.6f",
                val_eval["primary_mean_top50_proximity_excess"],
                h1_eval["primary_mean_top50_proximity_excess"],
                h2_eval["primary_mean_top50_proximity_excess"])
    logger.info("done. results: %s", args.out / "results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run with default args**

```bash
.venv/Scripts/python.exe scripts/p3/path1_train.py \
    --bundle data/p3_4070 \
    --out runs/sl_path1/numleaves63_lr05_minleaf100_seed42 \
    --num-leaves 63 --learning-rate 0.05 --min-data-in-leaf 100 --seed 42 \
    --num-iterations 200 \
    2>&1 | tail -20
```

Expected:
- ~3-5 min wall time
- Final log line: `VAL primary=<X> H1 primary=<Y> H2 primary=<Z>`
- All three numbers should be > 0 (we're picking top 50 from a pool with positive mean y)
- `runs/sl_path1/numleaves63_lr05_minleaf100_seed42/` contains `results.json`, `predictions.parquet`, `predictions.npz`, `lgb_model.txt`

- [ ] **Step 3: Sanity-check primary metric is positive on at least one window**

```bash
.venv/Scripts/python.exe -c "
import json
d = json.load(open('runs/sl_path1/numleaves63_lr05_minleaf100_seed42/results.json'))
for w in ('VAL_EFF', 'H1', 'H2'):
    e = d[w]
    print(f'{w}: primary={e[\"primary_mean_top50_proximity_excess\"]:.6f} spearman={e[\"spearman\"]:.4f}')
"
```

Expected: primary values in range [0.001, 0.02]. Spearman positive on at least VAL_EFF (model learned non-trivial ranking).

- [ ] **Step 4: Commit (script only, run output is gitignored)**

```bash
git add scripts/p3/path1_train.py
git commit -m "feat(sl-path1): single-config LightGBM regression trainer

regression_l2 on target_y; early-stopping on VAL_EFF; produces
predictions.parquet (full universe, all dates) + results.json (per-window
metric block) + lgb_model.txt for inspection."
```

---

## Phase 4: 36-run grid

### Task 4.1: Grid orchestrator

**Files:**
- Create: `scripts/p3/path1_grid.py`

- [ ] **Step 1: Implement grid runner**

`scripts/p3/path1_grid.py`:

```python
"""Path 1 — run the 12-config × 3-seed grid sequentially.

Each config is a separate subprocess launch of path1_train.py so a single-config
crash doesn't take down the rest of the grid. Total expected wall time: 1-2h
on 36-core CPU.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path


logger = logging.getLogger(__name__)


# 12 hyperparam configs (3 num_leaves × 2 lr × 2 min_data_in_leaf)
CONFIGS = []
for nl in (31, 63, 127):
    for lr in (0.03, 0.05):
        for mdl in (50, 100):
            CONFIGS.append({"num_leaves": nl, "learning_rate": lr, "min_data_in_leaf": mdl})

SEEDS = (42, 43, 44)


def _config_name(c: dict, seed: int) -> str:
    return (
        f"nl{c['num_leaves']}_lr{int(c['learning_rate']*1000):03d}_"
        f"mdl{c['min_data_in_leaf']}_seed{seed}"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--out-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--num-iterations", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0,
                    help="If > 0, only run the first N (config, seed) combos. For testing.")
    args = ap.parse_args(argv)

    py = sys.executable
    train_script = Path(__file__).resolve().parent / "path1_train.py"

    combos = [(c, s) for c in CONFIGS for s in SEEDS]
    if args.limit > 0:
        combos = combos[: args.limit]
    logger.info("grid: %d configs × %d seeds = %d combos", len(CONFIGS), len(SEEDS), len(combos))

    t_start = time.time()
    n_ok = n_skip = n_fail = 0
    for i, (c, s) in enumerate(combos):
        name = _config_name(c, s)
        out = args.out_root / name
        if (out / "results.json").exists():
            logger.info("[%d/%d] skip %s (already done)", i + 1, len(combos), name)
            n_skip += 1
            continue
        logger.info("[%d/%d] BEGIN %s", i + 1, len(combos), name)
        cmd = [
            py, str(train_script),
            "--bundle", str(args.bundle),
            "--out", str(out),
            "--seed", str(s),
            "--num-leaves", str(c["num_leaves"]),
            "--learning-rate", str(c["learning_rate"]),
            "--min-data-in-leaf", str(c["min_data_in_leaf"]),
            "--num-iterations", str(args.num_iterations),
            "--early-stopping-rounds", str(args.early_stopping_rounds),
        ]
        rc = subprocess.run(cmd, cwd=Path.cwd()).returncode
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            logger.error("[%d/%d] FAIL %s (rc=%d)", i + 1, len(combos), name, rc)

    elapsed = time.time() - t_start
    logger.info("grid done in %.0fs: ok=%d skip=%d fail=%d", elapsed, n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Test on 2-combo limit (smoke)**

```bash
.venv/Scripts/python.exe scripts/p3/path1_grid.py \
    --bundle data/p3_4070 \
    --out-root runs/sl_path1 \
    --num-iterations 200 \
    --limit 2 \
    2>&1 | tail -10
```

Expected: 2 combos run, both succeed. ~6-10 min wall time.

- [ ] **Step 3: Commit**

```bash
git add scripts/p3/path1_grid.py
git commit -m "feat(sl-path1): grid orchestrator (12 configs × 3 seeds = 36 runs)"
```

### Task 4.2: Run full grid

This is the long-running step. Defer the actual launch until tests for downstream tasks are ready, then kick off in background.

- [ ] **Step 1: Run full grid in background**

```bash
.venv/Scripts/python.exe scripts/p3/path1_grid.py \
    --bundle data/p3_4070 \
    --out-root runs/sl_path1 \
    --num-iterations 2000 \
    > runs/sl_path1_grid.log 2>&1 &
```

(Or use the harness's run_in_background mechanism. Wall time: 1-2h on 36-core CPU.)

- [ ] **Step 2: Wait for completion**

Periodically (or via Monitor):
```bash
ls runs/sl_path1/*/results.json | wc -l
```

Expected: 36 when done.

- [ ] **Step 3: Sanity check — print all 36 primary metrics on VAL_EFF**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
results = []
for f in sorted(Path('runs/sl_path1').glob('*/results.json')):
    d = json.load(open(f))
    name = f.parent.name
    val = d['VAL_EFF']['primary_mean_top50_proximity_excess']
    h1 = d['H1']['primary_mean_top50_proximity_excess']
    h2 = d['H2']['primary_mean_top50_proximity_excess']
    results.append((name, val, h1, h2))
results.sort(key=lambda x: -x[1])
for n, v, h1, h2 in results[:5]:
    print(f'{n:50s}  VAL={v:.6f}  H1={h1:.6f}  H2={h2:.6f}')
print('... (sorted desc by VAL primary)')
"
```

Expected: top configs cluster around the same VAL primary value within ~5% spread.

---

## Phase 5: ensemble + calibration

### Task 5.1: Isotonic calibration + seed-mean ensemble

**Files:**
- Create: `tests/p3/test_path1_ensemble.py`
- Create: `scripts/p3/path1_ensemble.py`

- [ ] **Step 1: Write the failing test**

`tests/p3/test_path1_ensemble.py`:

```python
"""Tests for Path 1 ensemble + calibration utilities."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from p3.path1_ensemble import (
    calibrate_isotonic,
    pick_top_configs_by_val,
    seed_mean_ensemble,
)


def test_seed_mean_three_identical_predictions():
    """Three identical predictions averaged → unchanged."""
    df_a = pl.DataFrame({
        "trade_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
        "ts_code": ["S1.SH", "S2.SH"],
        "score": [0.5, 0.7],
    })
    out = seed_mean_ensemble([df_a, df_a.clone(), df_a.clone()])
    assert out["score"].to_list() == pytest.approx([0.5, 0.7])


def test_seed_mean_averages_correctly():
    """Mean of [0.0, 0.5, 1.0] → 0.5 per row."""
    base_keys = pl.DataFrame({
        "trade_date": [dt.date(2025, 1, 1), dt.date(2025, 1, 2)],
        "ts_code": ["S1.SH", "S2.SH"],
    })
    df_a = base_keys.with_columns(pl.lit(0.0).alias("score"))
    df_b = base_keys.with_columns(pl.lit(0.5).alias("score"))
    df_c = base_keys.with_columns(pl.lit(1.0).alias("score"))
    out = seed_mean_ensemble([df_a, df_b, df_c])
    assert out["score"].to_list() == pytest.approx([0.5, 0.5])


def test_isotonic_preserves_order():
    """Isotonic-fit then transform → preserves rank ordering of inputs."""
    rng = np.random.default_rng(0)
    # Strictly increasing target relationship: actual = 2 * pred + noise
    pred_h1 = rng.uniform(0, 1, size=1000)
    actual_h1 = 2 * pred_h1 + rng.normal(0, 0.05, size=1000)
    pred_h2 = rng.uniform(0, 1, size=500)

    calibrator = calibrate_isotonic(pred_h1, actual_h1)
    out_h2 = calibrator(pred_h2)

    # Ranking of pred_h2 must equal ranking of out_h2.
    assert np.all(np.argsort(pred_h2) == np.argsort(out_h2))


def test_pick_top_3_configs_by_val_primary():
    """Pick top-3 distinct CONFIGS (collapsing seeds) by VAL primary."""
    # Mock results structure
    runs = {
        # name → val_primary
        "nl31_lr030_mdl50_seed42": 0.001,
        "nl31_lr030_mdl50_seed43": 0.0015,
        "nl31_lr030_mdl50_seed44": 0.001,
        "nl63_lr050_mdl100_seed42": 0.003,  # best config
        "nl63_lr050_mdl100_seed43": 0.0028,
        "nl63_lr050_mdl100_seed44": 0.0032,
        "nl127_lr050_mdl100_seed42": 0.002,
        "nl127_lr050_mdl100_seed43": 0.0025,
        "nl127_lr050_mdl100_seed44": 0.0022,
        "nl63_lr030_mdl50_seed42": 0.0018,
        "nl63_lr030_mdl50_seed43": 0.0019,
        "nl63_lr030_mdl50_seed44": 0.0017,
    }
    top3 = pick_top_configs_by_val(runs, top_k=3)
    config_names = {n.rsplit("_seed", 1)[0] for n in top3}
    assert "nl63_lr050_mdl100" in config_names
    assert "nl127_lr050_mdl100" in config_names
    assert len(config_names) == 3
```

- [ ] **Step 2: Run test, expect failure**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_ensemble.py -v 2>&1 | tail -10
```

Expected: ImportError on `from p3.path1_ensemble`.

- [ ] **Step 3: Implement ensemble module**

`scripts/p3/path1_ensemble.py`:

```python
"""Path 1 — ensemble (top-3 configs × 3 seeds, mean) + isotonic calibration on H1.

Outputs:
- runs/sl_path1/predictions.parquet — (trade_date, ts_code, score_raw, score_calibrated)
- runs/sl_path1/ensemble.json — H1 + H2 metric blocks for both raw and calibrated.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from p3.path1_eval import H1, H2, evaluate


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Pure functions
# ------------------------------------------------------------------ #

def seed_mean_ensemble(prediction_dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """Average score across multiple seed predictions, joined on (trade_date, ts_code)."""
    if not prediction_dfs:
        raise ValueError("no prediction DataFrames provided")
    base = prediction_dfs[0].select(["trade_date", "ts_code", "score"]).rename({"score": "score_0"})
    for i, df in enumerate(prediction_dfs[1:], start=1):
        base = base.join(
            df.select(["trade_date", "ts_code", "score"]).rename({"score": f"score_{i}"}),
            on=["trade_date", "ts_code"],
            how="inner",
        )
    score_cols = [f"score_{i}" for i in range(len(prediction_dfs))]
    base = base.with_columns(pl.mean_horizontal([pl.col(c) for c in score_cols]).alias("score"))
    return base.select(["trade_date", "ts_code", "score"])


def calibrate_isotonic(pred_calibration: np.ndarray, actual_calibration: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Fit isotonic regression on (pred, actual) and return a callable transformer.

    The returned callable applies the fitted isotonic to any new pred array.
    """
    pred_calibration = np.asarray(pred_calibration, dtype=np.float64)
    actual_calibration = np.asarray(actual_calibration, dtype=np.float64)
    mask = np.isfinite(pred_calibration) & np.isfinite(actual_calibration)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_calibration[mask], actual_calibration[mask])
    return lambda x: iso.transform(np.asarray(x, dtype=np.float64))


def pick_top_configs_by_val(runs: dict[str, float], top_k: int = 3) -> list[str]:
    """Select the top_k DISTINCT configs (collapsing seeds) by their seed-mean VAL primary.

    runs keys: full run names like "nl63_lr050_mdl100_seed42".
    Returns ALL run names (across seeds) belonging to the top_k configs.
    """
    # Group by (config base) = name without _seed{N} suffix
    config_to_seedscores: dict[str, list[float]] = {}
    for name, score in runs.items():
        config_base = name.rsplit("_seed", 1)[0]
        config_to_seedscores.setdefault(config_base, []).append(score)
    # Mean per config
    config_means = {k: float(np.mean(v)) for k, v in config_to_seedscores.items()}
    top = sorted(config_means.items(), key=lambda kv: -kv[1])[:top_k]
    top_configs = {kv[0] for kv in top}
    return [n for n in runs if n.rsplit("_seed", 1)[0] in top_configs]


# ------------------------------------------------------------------ #
# Driver
# ------------------------------------------------------------------ #

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/p3_4070", type=Path)
    ap.add_argument("--runs-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--out-root", default=Path("runs/sl_path1"), type=Path)
    ap.add_argument("--top-k-configs", type=int, default=3)
    args = ap.parse_args(argv)

    # 1. Load all run results, get VAL primary per run
    run_val_primary: dict[str, float] = {}
    pred_paths: dict[str, Path] = {}
    for run_dir in sorted(args.runs_root.glob("*_seed*/")):
        results = run_dir / "results.json"
        if not results.exists():
            continue
        d = json.loads(results.read_text())
        run_val_primary[run_dir.name] = d["VAL_EFF"]["primary_mean_top50_proximity_excess"]
        pred_paths[run_dir.name] = run_dir / "predictions.parquet"
    logger.info("found %d completed runs", len(run_val_primary))
    if len(run_val_primary) < args.top_k_configs * 2:
        logger.error("not enough runs (%d) to form top-%d ensemble", len(run_val_primary), args.top_k_configs)
        return 2

    # 2. Pick top-K configs (collapsing seeds first)
    chosen = pick_top_configs_by_val(run_val_primary, top_k=args.top_k_configs)
    logger.info("chose %d run names across %d distinct configs", len(chosen),
                len({n.rsplit('_seed', 1)[0] for n in chosen}))

    # 3. Seed-mean ensemble
    pred_dfs = [pl.read_parquet(pred_paths[n]) for n in chosen]
    ens = seed_mean_ensemble(pred_dfs)
    logger.info("ensemble: %d rows", len(ens))

    # 4. Calibrate on H1 (against actual_y from target_y.parquet)
    target_y = pl.read_parquet(args.bundle / "target_y.parquet")
    realized = pl.read_parquet(args.bundle / "realized_returns.parquet").select(
        ["trade_date", "ts_code", "pct_chg_t_plus_1"]
    )
    market = pl.read_parquet(args.bundle / "market_returns.parquet").select(
        ["trade_date", "eq_weight_pct_chg_t_plus_1"]
    )
    h1_join = ens.filter(
        (pl.col("trade_date") >= H1[0]) & (pl.col("trade_date") <= H1[1])
    ).join(target_y, on=["trade_date", "ts_code"], how="inner")
    calibrator = calibrate_isotonic(
        h1_join["score"].to_numpy(),
        h1_join["y"].to_numpy(),
    )
    score_calibrated = calibrator(ens["score"].to_numpy())
    ens_cal = ens.with_columns(pl.Series("score_calibrated", score_calibrated))
    logger.info("isotonic fit on %d H1 rows; calibrated %d total rows", len(h1_join), len(ens_cal))

    # 5. Eval raw + calibrated on H1, H2
    eval_raw_h1 = evaluate(ens, target_y, realized, market, H1)
    eval_raw_h2 = evaluate(ens, target_y, realized, market, H2)
    eval_cal_h1 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H1,
    )
    eval_cal_h2 = evaluate(
        ens_cal.select(["trade_date", "ts_code", pl.col("score_calibrated").alias("score")]),
        target_y, realized, market, H2,
    )

    # 6. Compare to paris baseline
    bp = pl.read_parquet(args.bundle / "baseline_predictions.parquet")
    bp_pred = bp.select(["trade_date", "ts_code", pl.col("p_t3_baseline").alias("score")])
    bp_h1 = evaluate(bp_pred, target_y, realized, market, H1)
    bp_h2 = evaluate(bp_pred, target_y, realized, market, H2)

    summary = {
        "chosen_runs": chosen,
        "n_calibration_rows_H1": len(h1_join),
        "ensemble_raw_H1": eval_raw_h1,
        "ensemble_raw_H2": eval_raw_h2,
        "ensemble_calibrated_H1": eval_cal_h1,
        "ensemble_calibrated_H2": eval_cal_h2,
        "paris_baseline_H1": bp_h1,
        "paris_baseline_H2": bp_h2,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "ensemble.json").write_text(json.dumps(summary, indent=2, default=str))
    ens_cal.write_parquet(args.out_root / "predictions.parquet", compression="zstd", compression_level=10)

    logger.info("== Path 1 ensemble vs paris baseline ==")
    logger.info("Window | Metric                       | Path1 raw | Path1 cal | Paris    | Δ vs Paris")
    for window, p1r, p1c, par in (
        ("H1", eval_raw_h1, eval_cal_h1, bp_h1),
        ("H2", eval_raw_h2, eval_cal_h2, bp_h2),
    ):
        for k, label in (
            ("primary_mean_top50_proximity_excess", "primary"),
            ("spearman", "spearman"),
            ("top50_T1_hit_rate", "T1_hit"),
            ("ece_10bin", "ECE"),
        ):
            logger.info(
                "%-6s | %-28s | %+.6f | %+.6f | %+.6f | %+.6f",
                window, label, p1r[k], p1c[k], par[k], p1c[k] - par[k],
            )

    logger.info("wrote %s and %s",
                args.out_root / "ensemble.json", args.out_root / "predictions.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/Scripts/python.exe -m pytest tests/p3/test_path1_ensemble.py -v 2>&1 | tail -10
```

Expected: 4 passed.

- [ ] **Step 5: Run ensemble on completed grid**

```bash
.venv/Scripts/python.exe scripts/p3/path1_ensemble.py \
    --bundle data/p3_4070 \
    --runs-root runs/sl_path1 \
    --out-root runs/sl_path1 \
    --top-k-configs 3 \
    2>&1 | tail -25
```

Expected:
- Picks top-3 configs (9 run names)
- Logs the comparison table: Path1 raw / cal / Paris baseline / Δ
- Writes `runs/sl_path1/ensemble.json` and `runs/sl_path1/predictions.parquet`

- [ ] **Step 6: Commit**

```bash
git add scripts/p3/path1_ensemble.py tests/p3/test_path1_ensemble.py
git commit -m "feat(sl-path1): ensemble + isotonic calibration

Top-3 configs by VAL_EFF primary (collapsing seeds), seed-mean across
their 9 runs, isotonic-calibrate on H1, eval raw+calibrated on H1+H2,
compare to paris's baseline_predictions.parquet."
```

---

## Phase 6: report + delivery

### Task 6.1: Write Path 1 RESULTS.md

**Files:**
- Create: `runs/sl_path1/RESULTS.md`

- [ ] **Step 1: Generate RESULTS.md from ensemble.json**

```bash
.venv/Scripts/python.exe -c "
import json
from datetime import date
d = json.load(open('runs/sl_path1/ensemble.json'))

def fmt(b): return f'{b[\"primary_mean_top50_proximity_excess\"]:+.6f}'
def fmt_sp(b): return f'{b[\"spearman\"]:+.4f}'
def fmt_hit(b): return f'{b[\"top50_T1_hit_rate\"]*100:.2f}%'
def fmt_ece(b): return f'{b[\"ece_10bin\"]:.5f}'

paths_md = [
    '# SL Path 1 — LightGBM β-regression Baseline RESULTS',
    '',
    f'**Date**: {date.today().isoformat()}',
    f'**Spec**: docs/superpowers/specs/2026-05-09-sl-ensemble-training-design.md',
    f'**Companion**: runs/p3_findings/RESULTS.md (P3 PPO findings — dropped)',
    '',
    '## Headline',
    '',
    '| Window | Metric | Path1 raw | Path1 calibrated | Paris P2 v2 | Δ (cal − Paris) |',
    '|---|---|---:|---:|---:|---:|',
    f'| H1 | primary_mean_top50_proximity_excess | {fmt(d[\"ensemble_raw_H1\"])} | {fmt(d[\"ensemble_calibrated_H1\"])} | {fmt(d[\"paris_baseline_H1\"])} | {d[\"ensemble_calibrated_H1\"][\"primary_mean_top50_proximity_excess\"] - d[\"paris_baseline_H1\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H1 | spearman | {fmt_sp(d[\"ensemble_raw_H1\"])} | {fmt_sp(d[\"ensemble_calibrated_H1\"])} | {fmt_sp(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | top50_T1_hit_rate | {fmt_hit(d[\"ensemble_raw_H1\"])} | {fmt_hit(d[\"ensemble_calibrated_H1\"])} | {fmt_hit(d[\"paris_baseline_H1\"])} | — |',
    f'| H1 | ECE_10bin | {fmt_ece(d[\"ensemble_raw_H1\"])} | {fmt_ece(d[\"ensemble_calibrated_H1\"])} | {fmt_ece(d[\"paris_baseline_H1\"])} | — |',
    f'| H2 | primary_mean_top50_proximity_excess | {fmt(d[\"ensemble_raw_H2\"])} | {fmt(d[\"ensemble_calibrated_H2\"])} | {fmt(d[\"paris_baseline_H2\"])} | {d[\"ensemble_calibrated_H2\"][\"primary_mean_top50_proximity_excess\"] - d[\"paris_baseline_H2\"][\"primary_mean_top50_proximity_excess\"]:+.6f} |',
    f'| H2 | spearman | {fmt_sp(d[\"ensemble_raw_H2\"])} | {fmt_sp(d[\"ensemble_calibrated_H2\"])} | {fmt_sp(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | top50_T1_hit_rate | {fmt_hit(d[\"ensemble_raw_H2\"])} | {fmt_hit(d[\"ensemble_calibrated_H2\"])} | {fmt_hit(d[\"paris_baseline_H2\"])} | — |',
    f'| H2 | ECE_10bin | {fmt_ece(d[\"ensemble_raw_H2\"])} | {fmt_ece(d[\"ensemble_calibrated_H2\"])} | {fmt_ece(d[\"paris_baseline_H2\"])} | — |',
    '',
    '## Chosen runs (top-3 configs × 3 seeds = 9 runs)',
    '',
]
for n in d['chosen_runs']:
    paths_md.append(f'- {n}')
paths_md.append('')
paths_md.append('## Go/No-Go to Path 4')
paths_md.append('')
paths_md.append('Per spec §4.1: ensemble H1 primary > paris baseline H1 primary.')
delta = d['ensemble_calibrated_H1']['primary_mean_top50_proximity_excess'] - d['paris_baseline_H1']['primary_mean_top50_proximity_excess']
verdict = 'PASS' if delta > 0 else 'FAIL'
paths_md.append(f'- Path 1 calibrated H1 primary − Paris H1 primary = {delta:+.6f}: **{verdict}**')

open('runs/sl_path1/RESULTS.md', 'w').write('\n'.join(paths_md))
print('wrote runs/sl_path1/RESULTS.md')
"
```

Expected: writes a markdown file with the headline table and Go/No-Go verdict.

- [ ] **Step 2: Inspect**

```bash
cat runs/sl_path1/RESULTS.md
```

Expected: clean markdown report. Verify primary metric numbers match `ensemble.json`.

- [ ] **Step 3: Commit (RESULTS.md is in runs/, gitignored — instead push to OSS in next task)**

`runs/` is gitignored. Skip git commit; this artifact ships via OSS.

### Task 6.2: Upload to OSS

**Files:**
- Create: `scripts/oss_upload_sl_path1.py`

- [ ] **Step 1: Implement uploader (mirror existing oss_upload_*.py pattern)**

`scripts/oss_upload_sl_path1.py`:

```python
"""Upload SL Path 1 artifacts to ledashi-oss (Shenzhen)."""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import oss2

ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "oss-cn-shenzhen.aliyuncs.com"
BUCKET_NAME = "ledashi-oss"
PREFIX = "fromsz/handoffs/2026-05-09-sl-path1-results/"


def _read_env() -> dict[str, str]:
    out: dict[str, str] = {}
    raw = (ROOT / ".env").read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gbk", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _md5_pair(p: Path) -> tuple[str, str]:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii"), h.hexdigest().lower()


def _fmt(s: int) -> str:
    if s > 1 << 30:
        return f"{s/(1<<30):6.2f} GiB"
    if s > 1 << 20:
        return f"{s/(1<<20):6.2f} MiB"
    return f"{s/1024:6.2f} KiB"


def main() -> int:
    env = _read_env()
    auth = oss2.Auth(env["OSS_ACCESS_KEY_ID"], env["OSS_ACCESS_KEY_SECRET"])
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME, connect_timeout=30)
    print(f"[oss-upload] endpoint={ENDPOINT} bucket={BUCKET_NAME} prefix={PREFIX}")

    manifest: list[tuple[Path, str]] = []
    sl_root = ROOT / "runs" / "sl_path1"

    # 1. Top-level: RESULTS.md, ensemble.json, predictions.parquet
    for top in ("RESULTS.md", "ensemble.json", "predictions.parquet"):
        p = sl_root / top
        if p.exists():
            manifest.append((p, top))

    # 2. Per-run results.json + lgb_model.txt (skip the heavier predictions.npz/parquet)
    for run_dir in sorted(sl_root.glob("*_seed*/")):
        for fname in ("results.json", "lgb_model.txt"):
            f = run_dir / fname
            if f.exists():
                manifest.append((f, f"runs/{run_dir.name}/{fname}"))

    total = sum(p.stat().st_size for p, _ in manifest)
    print(f"[oss-upload] manifest: {len(manifest)} files, {_fmt(total)}")

    n_put = n_skip = 0
    for local, sub in manifest:
        key = PREFIX + sub
        size = local.stat().st_size
        b64, hex_ = _md5_pair(local)
        try:
            meta = bucket.get_object_meta(key)
            if meta.etag.strip('"').lower() == hex_:
                print(f"  [skip] {key} ({_fmt(size)})")
                n_skip += 1
                continue
        except oss2.exceptions.NoSuchKey:
            pass
        print(f"  [put]  {key} ({_fmt(size)})")
        bucket.put_object_from_file(key, str(local), headers={"Content-MD5": b64})
        n_put += 1

    print(f"\n[oss-upload] DONE. uploaded={n_put} skipped={n_skip}")
    print(f"[oss-upload] browse: https://oss.console.aliyun.com/bucket/oss-cn-shenzhen/{BUCKET_NAME}/object?path={PREFIX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Upload**

```bash
.venv/Scripts/python.exe scripts/oss_upload_sl_path1.py 2>&1 | tail -10
```

Expected: ~40 files (RESULTS.md + ensemble.json + predictions.parquet + 36 runs × 2 files), ~30-50 MB total. All `[put]` on first run.

- [ ] **Step 3: Commit uploader (kept in repo as it's reusable infra, not internal one-shot)**

Per the project's existing `scripts/oss_upload_phase26fg_v3.py` precedent, this is a peer artifact. Commit it.

```bash
git add scripts/oss_upload_sl_path1.py
git commit -m "chore(oss): upload script for SL Path 1 results bundle"
```

---

## Verification

End-to-end checks for the whole plan, in order:

1. **Phase 0 (scaffolding):**
   ```bash
   .venv/Scripts/python.exe -m pytest tests/p3/ --collect-only 2>&1 | tail -3
   ```
   Expected: ≥ 17 tests collected after all phases.

2. **Phase 1 (target_y):**
   ```bash
   .venv/Scripts/python.exe -m pytest tests/p3/test_path1_target.py -v
   ls -lh data/p3_4070/target_y.parquet
   ```
   Expected: 6 passed; target file ~50 MB.

3. **Phase 2 (eval):**
   ```bash
   .venv/Scripts/python.exe -m pytest tests/p3/test_path1_eval.py -v
   ```
   Expected: 7 passed.

4. **Phase 3 (single train):**
   ```bash
   ls runs/sl_path1/nl63_lr050_mdl100_seed42/results.json
   ```
   Expected: file exists.

5. **Phase 4 (grid):**
   ```bash
   ls runs/sl_path1/*/results.json | wc -l
   ```
   Expected: 36 (full grid done).

6. **Phase 5 (ensemble):**
   ```bash
   .venv/Scripts/python.exe -m pytest tests/p3/test_path1_ensemble.py -v
   ls runs/sl_path1/ensemble.json runs/sl_path1/predictions.parquet
   ```
   Expected: 4 passed; both files exist.

7. **Phase 6 (delivery):**
   ```bash
   ls runs/sl_path1/RESULTS.md
   ```
   Expected: file exists with the headline table.

---

## Out of scope (deferred to Path 4 / future plans)

- Cross-sectional rank-z feature engineering (Path 4 — separate plan)
- Outlier audit + clipping at panel level (Path 4)
- Model-class diversity (CatBoost/XGBoost) — Path 2
- Tabular DL — Path 3
- Live inference / ONNX export — production deploy plan
- Sector / industry-balanced top-K — portfolio decision layer
- Backtesting with execution cost — separate sim plan

These are intentionally NOT in this plan. Land Path 1 first, decide on Path 4 based on go/no-go gate.
