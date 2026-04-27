#!/usr/bin/env python3
"""Synthetic factor panel generator for AurumQ-RL demos and CI.

Produces a Parquet file that satisfies the input data contract defined in
``aurumq_rl.data_loader`` — multiple factor prefix groups, realistic-looking
distributions, and all required + optional columns. **Stock codes are
explicitly synthetic** (``SYN_00001``...) and do NOT correspond to real
A-share tickers.

Usage
-----
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n-stocks 200 --n-dates 500 --out data/synthetic_demo.parquet

Output schema
-------------
Required:
    ts_code (str), trade_date (date), close (float), pct_chg (float, decimal),
    vol (float)
Optional:
    is_st (bool), days_since_ipo (int), industry_code (int), name (str)
Factor groups (≥6):
    alpha_*, mf_*, hm_*, hk_*, fund_*, ind_*

The generated file is intended to fit comfortably under 10 MB at the default
size (200 stocks × 500 dates ≈ 100 k rows) so it can be checked into a repo
or copied around quickly.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_STOCKS: int = 200
DEFAULT_N_DATES: int = 500
DEFAULT_SEED: int = 42
DEFAULT_OUT: str = "data/synthetic_demo.parquet"

# Per-prefix factor counts (kept small so the file stays well under 10 MB)
FACTOR_GROUPS: dict[str, int] = {
    "alpha_": 8,   # alpha101-style quant-volume factors
    "mf_": 4,      # main-force capital flow
    "hm_": 3,      # hot-money seats
    "hk_": 2,      # northbound holdings
    "fund_": 3,    # fundamentals
    "ind_": 2,     # industry relative
}

# Stock-name template — fully synthetic
NAME_PREFIXES: tuple[str, ...] = (
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
)

# Industries (1..N_INDUSTRIES) — neutral integer codes
N_INDUSTRIES: int = 10

# ST probability per stock (a stock is either ST or not for the whole panel)
ST_RATE: float = 0.05

# Default mean of days_since_ipo — most stocks are mature
IPO_MIN: int = 200
IPO_MAX: int = 2500


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _generate_dates(n_dates: int, start: datetime.date) -> list[datetime.date]:
    """Weekday-only synthetic trading calendar (Mon-Fri)."""
    out: list[datetime.date] = []
    cur = start
    while len(out) < n_dates:
        if cur.weekday() < 5:
            out.append(cur)
        cur += datetime.timedelta(days=1)
    return out


def _generate_stock_static(
    rng: np.random.Generator,
    n_stocks: int,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Generate per-stock static attributes.

    Returns
    -------
    ts_codes:
        Synthetic codes ``SYN_00001`` ... ``SYN_NNNNN``.
    names:
        Synthetic names like ``AlphaCorp_001``.
    is_st:
        Bool array, True for ~ST_RATE of stocks (constant across dates).
    industry_codes:
        int8 array in [1, N_INDUSTRIES].
    """
    ts_codes = [f"SYN_{i:05d}" for i in range(1, n_stocks + 1)]
    names = [
        f"{NAME_PREFIXES[i % len(NAME_PREFIXES)]}Corp_{i:03d}"
        for i in range(n_stocks)
    ]
    is_st = rng.random(n_stocks) < ST_RATE
    industry_codes = rng.integers(1, N_INDUSTRIES + 1, size=n_stocks).astype(np.int16)
    return ts_codes, names, is_st, industry_codes


def _generate_factors(
    rng: np.random.Generator,
    n_dates: int,
    n_stocks: int,
) -> dict[str, np.ndarray]:
    """Generate factor panels (date × stock) for each prefix group.

    Each factor is a z-scored Gaussian with mild auto-correlation across dates
    (AR(1) coefficient ~0.3) and per-stock loading drawn at the start.
    """
    factors: dict[str, np.ndarray] = {}
    for prefix, count in FACTOR_GROUPS.items():
        for k in range(count):
            col = f"{prefix}{k:03d}"
            # AR(1) over dates — gives factors a smooth, realistic feel
            x = np.zeros((n_dates, n_stocks), dtype=np.float32)
            x[0] = rng.standard_normal(n_stocks)
            phi = 0.3
            for t in range(1, n_dates):
                x[t] = phi * x[t - 1] + np.sqrt(1.0 - phi * phi) * rng.standard_normal(n_stocks)
            # Cross-section z-score per date so columns are roughly N(0,1)
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-8
            factors[col] = ((x - mu) / sd).astype(np.float32)
    return factors


def _generate_prices_and_volume(
    rng: np.random.Generator,
    n_dates: int,
    n_stocks: int,
    factors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate close, pct_chg (decimal form), and vol panels.

    Returns are weakly correlated with the alpha_ factor average so the data
    has a faint but learnable signal.
    """
    # Mild signal: average of first 4 alpha_ columns drives expected return.
    alpha_cols = sorted(c for c in factors if c.startswith("alpha_"))[:4]
    if alpha_cols:
        signal = np.mean([factors[c] for c in alpha_cols], axis=0)
    else:
        signal = np.zeros((n_dates, n_stocks), dtype=np.float32)

    # Daily pct change in decimal form: signal*0.005 + noise*0.018, clipped at ±9.8%
    noise = rng.standard_normal((n_dates, n_stocks)).astype(np.float32) * 0.018
    pct_chg = signal * 0.005 + noise
    pct_chg = np.clip(pct_chg, -0.098, 0.098).astype(np.float32)

    # Close: cumulative product seeded at 10..50
    seed_price = rng.uniform(10.0, 50.0, size=n_stocks).astype(np.float32)
    close = np.zeros((n_dates, n_stocks), dtype=np.float32)
    close[0] = seed_price
    for t in range(1, n_dates):
        close[t] = close[t - 1] * (1.0 + pct_chg[t])
    close = np.clip(close, 1.0, None)

    # Volume — log-normal with rare suspensions (vol == 0)
    vol = np.exp(rng.normal(loc=14.0, scale=0.8, size=(n_dates, n_stocks))).astype(np.float32)
    suspended = rng.random((n_dates, n_stocks)) < 0.005
    vol = np.where(suspended, 0.0, vol).astype(np.float32)

    return close, pct_chg, vol


def _generate_days_since_ipo(
    rng: np.random.Generator,
    n_dates: int,
    n_stocks: int,
) -> np.ndarray:
    """Each stock starts with a random IPO offset and increments by 1 each date."""
    start = rng.integers(IPO_MIN, IPO_MAX, size=n_stocks).astype(np.int32)
    # shape (n_dates, n_stocks)
    out = start[None, :] + np.arange(n_dates, dtype=np.int32)[:, None]
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_synthetic_dataframe(
    n_stocks: int = DEFAULT_N_STOCKS,
    n_dates: int = DEFAULT_N_DATES,
    seed: int = DEFAULT_SEED,
    start_date: datetime.date | None = None,
) -> pl.DataFrame:
    """Build a polars DataFrame matching the AurumQ-RL Parquet contract.

    Parameters
    ----------
    n_stocks:
        Number of synthetic stocks (default 200).
    n_dates:
        Number of trading dates (default 500, weekdays only).
    seed:
        RNG seed for reproducibility.
    start_date:
        First trading date (default 2022-01-03).

    Returns
    -------
    polars.DataFrame in long format (one row per (date, stock)).
    """
    if n_stocks < 1:
        raise ValueError(f"n_stocks={n_stocks} must be >= 1")
    if n_dates < 1:
        raise ValueError(f"n_dates={n_dates} must be >= 1")

    rng = np.random.default_rng(seed)

    if start_date is None:
        start_date = datetime.date(2022, 1, 3)
    dates = _generate_dates(n_dates, start_date)

    ts_codes, names, is_st_static, industry_codes = _generate_stock_static(rng, n_stocks)

    factors = _generate_factors(rng, n_dates, n_stocks)
    close, pct_chg, vol = _generate_prices_and_volume(rng, n_dates, n_stocks, factors)
    days_since_ipo = _generate_days_since_ipo(rng, n_dates, n_stocks)

    # Long-format assembly
    n_rows = n_dates * n_stocks
    date_col = np.repeat(np.array(dates, dtype="datetime64[D]"), n_stocks)
    code_col = np.tile(np.array(ts_codes, dtype=object), n_dates)
    name_col = np.tile(np.array(names, dtype=object), n_dates)
    industry_col = np.tile(industry_codes, n_dates)
    is_st_col = np.tile(is_st_static, n_dates)

    columns: dict[str, np.ndarray] = {
        "ts_code": code_col,
        "trade_date": date_col,
        "close": close.reshape(n_rows),
        "pct_chg": pct_chg.reshape(n_rows),
        "vol": vol.reshape(n_rows),
        "is_st": is_st_col,
        "days_since_ipo": days_since_ipo.reshape(n_rows).astype(np.int32),
        "industry_code": industry_col.astype(np.int16),
        "name": name_col,
    }
    for col_name, arr in factors.items():
        columns[col_name] = arr.reshape(n_rows)

    df = pl.DataFrame(columns)

    # Cast columns to compact types for smaller Parquet output
    df = df.with_columns(
        [
            pl.col("trade_date").cast(pl.Date),
            pl.col("close").cast(pl.Float32),
            pl.col("pct_chg").cast(pl.Float32),
            pl.col("vol").cast(pl.Float32),
            pl.col("is_st").cast(pl.Boolean),
            pl.col("days_since_ipo").cast(pl.Int32),
            pl.col("industry_code").cast(pl.Int16),
        ]
    )
    return df


def write_synthetic_parquet(
    out_path: Path,
    n_stocks: int = DEFAULT_N_STOCKS,
    n_dates: int = DEFAULT_N_DATES,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Generate and write a synthetic Parquet file.

    Returns the resolved output path.
    """
    df = build_synthetic_dataframe(n_stocks=n_stocks, n_dates=n_dates, seed=seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(out_path), compression="zstd")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic factor panel Parquet for AurumQ-RL demos.",
    )
    parser.add_argument("--n-stocks", type=int, default=DEFAULT_N_STOCKS)
    parser.add_argument("--n-dates", type=int, default=DEFAULT_N_DATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output Parquet path (default {DEFAULT_OUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    print(
        f"[generate_synthetic] n_stocks={args.n_stocks}, n_dates={args.n_dates}, "
        f"seed={args.seed}, out={out_path}"
    )
    written = write_synthetic_parquet(
        out_path=out_path,
        n_stocks=args.n_stocks,
        n_dates=args.n_dates,
        seed=args.seed,
    )
    size_mb = written.stat().st_size / (1024 * 1024)
    print(f"[generate_synthetic] wrote {written} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
