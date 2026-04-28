#!/usr/bin/env python3
"""Export a factor panel from a PostgreSQL data warehouse to Parquet.

This is the bridge between **your** market-data warehouse (where factor
columns are pre-computed) and the AurumQ-RL training pipeline.

What it does
------------
1. Connects to a PostgreSQL database (URL via ``--pg-url`` or ``$PG_URL``).
2. Runs a SQL query in chunks (default minimal query, or override with
   ``--sql-file``) to produce a wide-format result.
3. Streams chunks into polars and writes a single Parquet file (zstd).

Output schema (must match :mod:`aurumq_rl.data_loader` contract)
----------------------------------------------------------------
Required columns:

* ``ts_code``          str (e.g. ``600519.SH``)
* ``trade_date``       date
* ``close``            float
* ``pct_chg``          float (decimal form, +10% = 0.10)
* ``vol``              float (0 = suspended)

Plus factor columns matching the recognized prefixes
(``alpha_*``, ``mf_*``, ``hm_*``, ``hk_*``, ``inst_*``, ``mg_*``, ``cyq_*``,
``senti_*``, ``sh_*``, ``fund_*``, ``ind_*``, ``mkt_*``).

Optional columns (used if present): ``is_st``, ``days_since_ipo``,
``industry_code``, ``name``.

Usage
-----
::

    # Default minimal query (no factors — for testing connectivity only)
    python scripts/export_factor_panel.py \\
        --pg-url postgresql://user:pass@host:5432/your_db \\
        --start-date 2023-01-01 \\
        --end-date 2025-06-30 \\
        --out data/panel.parquet

    # Real export via custom SQL (recommended)
    python scripts/export_factor_panel.py \\
        --sql-file docs/example_query.sql \\
        --start-date 2023-01-01 \\
        --end-date 2025-06-30 \\
        --out data/factor_panel.parquet \\
        --universe-filter main_board

Required environment
--------------------
* ``PG_URL`` (fallback if ``--pg-url`` not given), e.g.
  ``postgresql://user:pass@host:5432/db``.

A dotenv file (``.env``) at the project root is auto-loaded if present.

Disclaimer
----------
This script makes no assumption about which factor library produced the data.
Real factor exports require either an existing wide view (such as
``factor_panel_view`` documented in ``docs/SCHEMA.md``) or a custom
``--sql-file``.

This project is for educational and research purposes only. Backtest
results do not represent future performance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Recognized factor column prefixes (mirrors aurumq_rl.data_loader)
FACTOR_COL_PREFIXES: tuple[str, ...] = (
    "alpha_",
    "mf_",
    "hm_",
    "hk_",
    "inst_",
    "mg_",
    "cyq_",
    "senti_",
    "sh_",
    "fund_",
    "ind_",
    "mkt_",
)

REQUIRED_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "close", "pct_chg", "vol")

# Built-in fallback SQL — minimal, deliberately *no* factor columns.
# Use --sql-file to point at docs/example_query.sql or your own template.
# For a SQL template that includes HS300/ZZ500 membership flags and the full
# factor join, see docs/example_query_with_index.sql.
DEFAULT_SQL: str = """
SELECT
    trade_date,
    ts_code,
    close,
    pct_chg,
    vol
FROM daily_quotes
WHERE trade_date >= :start_date
  AND trade_date <= :end_date
ORDER BY trade_date, ts_code
""".strip()

# Universe-filter SQL fragments. These are appended to the user's WHERE
# clause via a wrapping subquery, so they work regardless of source query.
UNIVERSE_FILTERS: dict[str, str] = {
    "all": "TRUE",
    "main_board": ("(ts_code ~ '^60[0135][0-9]{3}\\.SH$'  OR ts_code ~ '^00[0123][0-9]{3}\\.SZ$')"),
    "main_board_non_st": (
        "(ts_code ~ '^60[0135][0-9]{3}\\.SH$' "
        " OR ts_code ~ '^00[0123][0-9]{3}\\.SZ$') "
        "AND COALESCE(is_st, FALSE) = FALSE"
    ),
    "exclude_bse": "ts_code !~ '\\.BJ$'",
}

# Default chunk size for fetching from PG (rows per round-trip)
DEFAULT_CHUNK_SIZE: int = 100_000


# ---------------------------------------------------------------------------
# Optional .env loading (no hard dependency on python-dotenv)
# ---------------------------------------------------------------------------


def _load_dotenv_if_present(env_path: Path) -> None:
    """Naive .env loader. Only sets variables not already in os.environ."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # Best-effort; don't fail the script on .env issues.
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Export a wide factor panel from a PostgreSQL data warehouse to "
            "Parquet, ready for AurumQ-RL training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/export_factor_panel.py \\\n"
            "    --pg-url postgresql://user:pass@host:5432/db \\\n"
            "    --start-date 2023-01-01 --end-date 2025-06-30 \\\n"
            "    --sql-file docs/example_query.sql \\\n"
            "    --out data/factor_panel.parquet\n"
        ),
    )

    parser.add_argument(
        "--pg-url",
        type=str,
        default=None,
        help="PostgreSQL URL. Falls back to $PG_URL env var.",
    )
    parser.add_argument(
        "--start-date",
        "--start",
        dest="start_date",
        type=str,
        required=True,
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        "--end",
        dest="end_date",
        type=str,
        required=True,
        help="Inclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output Parquet path. Parent directories will be created.",
    )
    parser.add_argument(
        "--sql-file",
        type=Path,
        default=None,
        help=(
            "Optional SQL template. Must contain :start_date / :end_date "
            "placeholders. If omitted, a minimal default query is used "
            "(no factor columns)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per server-side fetch (default {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--universe-filter",
        choices=sorted(UNIVERSE_FILTERS.keys()),
        default="main_board_non_st",
        help=(
            "Optional universe filter applied as an outer WHERE clause "
            "(default 'main_board_non_st')."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved SQL and exit without connecting.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# SQL preparation
# ---------------------------------------------------------------------------


def _read_sql_template(sql_file: Path | None) -> str:
    """Read SQL template file, or return the built-in default."""
    if sql_file is None:
        return DEFAULT_SQL
    if not sql_file.exists():
        raise FileNotFoundError(
            f"--sql-file not found: {sql_file}\n"
            "Provide a valid SQL file or omit --sql-file to use the default."
        )
    return sql_file.read_text(encoding="utf-8").strip()


def _wrap_with_universe_filter(sql: str, universe_filter: str) -> str:
    """Wrap a user query in an outer SELECT applying a universe predicate.

    The wrapping is opt-in. ``all`` returns the inner query unchanged.
    """
    if universe_filter == "all":
        return sql

    predicate = UNIVERSE_FILTERS[universe_filter]
    # Strip trailing semicolons so we can wrap safely
    inner = sql.rstrip().rstrip(";")
    return f"SELECT * FROM (\n{inner}\n) AS panel_inner\nWHERE {predicate}"


def _validate_date(date_str: str, label: str) -> str:
    """Reject obviously bad date strings before we round-trip to PG."""
    import datetime as _dt

    try:
        _dt.date.fromisoformat(date_str)
    except ValueError as exc:
        raise SystemExit(f"Invalid {label} (expected YYYY-MM-DD): {date_str!r} ({exc})") from exc
    return date_str


# ---------------------------------------------------------------------------
# DB execution
# ---------------------------------------------------------------------------


def _import_psycopg2() -> Any:
    """Import psycopg2 with a clear error message if absent."""
    try:
        import psycopg2  # type: ignore[import-not-found]
        import psycopg2.extras  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime env check
        raise SystemExit(
            "psycopg2 is required for export_factor_panel.py.\n"
            "Install factor-export extras with:\n"
            "    pip install 'aurumq-rl[factors]'\n"
            f"(original ImportError: {exc})"
        ) from exc
    return psycopg2


def _import_polars() -> Any:
    """Import polars with a clear error message if absent."""
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - runtime env check
        raise SystemExit(
            "polars is required for export_factor_panel.py.\n"
            "Install with: pip install polars pyarrow\n"
            f"(original ImportError: {exc})"
        ) from exc
    return pl


def _execute_paged(
    connection: Any,
    sql: str,
    params: dict[str, Any],
    chunk_size: int,
) -> tuple[list[str], list[list[Any]]]:
    """Execute the query with a server-side cursor and stream rows.

    Returns
    -------
    (columns, rows) where ``rows`` is a flat list of row tuples (lists).
    """
    columns: list[str] = []
    rows: list[list[Any]] = []
    total_fetched = 0
    started_at = time.monotonic()

    # Server-side named cursor → streams large result sets without OOM.
    cursor_name = "aurumq_rl_export_cursor"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = chunk_size
        cursor.execute(sql, params)

        if cursor.description is None:
            raise RuntimeError(
                "Query returned no result description. Make sure the SQL "
                "is a SELECT producing the expected columns."
            )
        columns = [desc[0] for desc in cursor.description]

        # Validate required columns are present
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise RuntimeError(
                f"Query result is missing required columns: {missing}\n"
                f"Got columns: {columns[:30]}...\n"
                f"AurumQ-RL needs at minimum: {REQUIRED_COLUMNS}"
            )

        while True:
            batch = cursor.fetchmany(chunk_size)
            if not batch:
                break
            rows.extend(list(r) for r in batch)
            total_fetched += len(batch)
            elapsed = time.monotonic() - started_at
            rate = total_fetched / elapsed if elapsed > 0 else 0
            print(
                f"  fetched {total_fetched:,} rows ({rate:,.0f} rows/s)",
                flush=True,
            )

    return columns, rows


def _summarize_factor_groups(columns: list[str]) -> dict[str, int]:
    """Count how many factor columns matched each prefix."""
    summary: dict[str, int] = {}
    for prefix in FACTOR_COL_PREFIXES:
        n = sum(1 for c in columns if c.startswith(prefix))
        if n > 0:
            summary[prefix] = n
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns process exit code (0 on success)."""
    args = parse_args(argv)

    # Best-effort .env load from project root (CWD or script parent)
    here = Path(__file__).resolve().parent
    for candidate in (Path.cwd() / ".env", here.parent / ".env"):
        _load_dotenv_if_present(candidate)

    pg_url = args.pg_url or os.environ.get("PG_URL")
    if not pg_url and not args.dry_run:
        raise SystemExit(
            "PostgreSQL URL not provided. Pass --pg-url or set $PG_URL "
            "(or place it in a .env file)."
        )

    start_date = _validate_date(args.start_date, "--start-date")
    end_date = _validate_date(args.end_date, "--end-date")

    # Load + wrap SQL
    sql_template = _read_sql_template(args.sql_file)
    full_sql = _wrap_with_universe_filter(sql_template, args.universe_filter)

    if args.dry_run:
        print("=== Resolved SQL ===")
        print(full_sql)
        print("\n=== Parameters ===")
        print(f"  start_date = {start_date}")
        print(f"  end_date   = {end_date}")
        print(f"  universe   = {args.universe_filter}")
        return 0

    print("Connecting to PostgreSQL...", flush=True)
    psycopg2 = _import_psycopg2()
    pl = _import_polars()

    connection = psycopg2.connect(pg_url)
    try:
        connection.set_session(readonly=True, autocommit=False)
        print(
            f"Executing query (universe={args.universe_filter}, chunk_size={args.chunk_size:,})...",
            flush=True,
        )
        columns, rows = _execute_paged(
            connection,
            full_sql,
            {"start_date": start_date, "end_date": end_date},
            chunk_size=args.chunk_size,
        )
    finally:
        connection.close()

    if not rows:
        raise SystemExit(
            "Query returned 0 rows. Check date range, universe filter, and source tables."
        )

    print(f"\nBuilding polars DataFrame ({len(rows):,} rows x {len(columns)} cols)...")
    df = pl.DataFrame(rows, schema=columns, orient="row")

    # Print factor-group summary so the user can confirm the export shape.
    factor_summary = _summarize_factor_groups(columns)
    print("\nFactor-group summary (matched prefixes):")
    if factor_summary:
        for prefix, count in sorted(factor_summary.items()):
            print(f"  {prefix:<10} {count:>4} columns")
    else:
        print("  (none matched — output Parquet has no factor columns!)")
        print(
            "  Hint: this likely means you ran with the default SQL. "
            "Use --sql-file docs/example_query.sql for a real factor export."
        )

    # Write Parquet
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting Parquet -> {args.out} (compression=zstd)...")
    df.write_parquet(args.out, compression="zstd")

    size_bytes = args.out.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Done. Wrote {len(rows):,} rows, {len(columns)} columns, {size_mb:,.1f} MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
