"""Build the locked GTJA-191 reference parquet from Daic115/alpha191.

This script is run **once** during Phase B (or whenever the Daic115
reference implementation is updated) to generate
``tests/factors/gtja191/data/gtja191_reference.parquet``. The parquet is
committed to git and serves as a numerical baseline for the polars
implementations of the Guotai-Junan 191 alphas that will live in
``aurumq_rl.factors.gtja191`` (Phase B').

Usage
-----
    python scripts/reference_data/build_gtja191_reference.py

License notice
--------------
``Daic115/alpha191`` carries **no LICENSE file** — we treat it as
"all rights reserved" and use it strictly as a **numerical formula
reference**. We do NOT vendor any of its code into our runtime. The
runtime only loads the resulting parquet, which contains numerical
values (not copyrightable). Daic115 is imported here at *build time*
only.

Implementation notes
--------------------

* ``factor_ops.py`` imports ``talib`` (TA-Lib) unconditionally. We're
  not paying the install cost in the AurumQ-RL test environment, so we
  shim ``talib`` with a no-op module before importing Daic115. The
  alpha191 functions that depend on TA-Lib will simply blow up at call
  time and we'll skip them — that's fine, the goal is partial coverage.

* Two alphas are tagged ``unfinished=True`` in Daic115 and will throw on
  call (``alpha191_029``, ``alpha191_142``). These are skipped via the
  decorator attribute check.

* The qlib import is wrapped in try/except in Daic115 itself, so the
  qlib-dependent alphas (a small subset using ``rolling_slope`` /
  ``rolling_resi`` / ``rolling_rsquare``) will fail at call time. We
  catch and skip them.

* Daic115 expects ``data`` to be a ``dict[str, pd.DataFrame]`` where
  each DataFrame is wide (``date_index × stock_columns``). We build that
  from the synthetic panel via the ``to_pandas_for_sthsf`` helper plus
  a small ``amount`` extension (Daic115 also reads ``data["amount"]``).

Output
------

The parquet has one row per ``(stock_code, trade_date)`` and one column
named ``gtja_NNN`` per alpha that built successfully. NaN values reflect
natural rolling-window warm-up.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io as _io
import pathlib
import sys
import types
import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
DAIC_ROOT: pathlib.Path = REPO_ROOT / "refs" / "gtja191" / "Daic115__alpha191"
DEFAULT_OUTPUT: pathlib.Path = (
    REPO_ROOT / "tests" / "factors" / "gtja191" / "data" / "gtja191_reference.parquet"
)


# ---------------------------------------------------------------------------
# Compatibility shims (talib + qlib stubs)
# ---------------------------------------------------------------------------


def _install_compat_shims() -> None:
    """Insert no-op stubs for ``talib`` (and reinforce qlib) into sys.modules.

    ``Daic115__alpha191/lib/ops/factor_ops.py`` does ``import talib as ta``
    at module load. We don't want to install TA-Lib (which requires a C
    library), so we synthesise a minimal stub that exposes the function
    names the file pulls (``LINEARREG``, ``LINEARREG_SLOPE``,
    ``LINEARREG_ANGLE``, ``LINEARREG_INTERCEPT``, ``TSF``). The stubs
    simply raise on call, so any alpha that touches them is skipped.
    """
    if "talib" not in sys.modules:
        talib_stub = types.ModuleType("talib")

        def _missing(*_args, **_kwargs):  # noqa: ANN001
            raise RuntimeError("talib stub: TA-Lib not installed in this env")

        for name in (
            "LINEARREG",
            "LINEARREG_SLOPE",
            "LINEARREG_ANGLE",
            "LINEARREG_INTERCEPT",
            "TSF",
        ):
            setattr(talib_stub, name, _missing)
        sys.modules["talib"] = talib_stub

    # qlib is already wrapped in try/except inside Daic115, but defensive:
    if "qlib" not in sys.modules:
        qlib_stub = types.ModuleType("qlib")
        sys.modules["qlib"] = qlib_stub


# ---------------------------------------------------------------------------
# Daic115 module loader
# ---------------------------------------------------------------------------


def _load_daic_alpha191() -> types.ModuleType:
    """Import ``Daic115__alpha191/alpha191.py`` and return the module.

    Daic115's source uses bare ``from lib.ops.factor_ops import …`` and
    additionally ``from factors.ops.rolling import …`` (the latter a
    leftover path from the author's local layout). We add Daic115's root
    to ``sys.path`` so ``lib`` resolves, then alias ``factors.ops`` →
    ``lib.ops`` so the rolling import works too. Path is restored on exit.
    """
    if not DAIC_ROOT.exists():
        raise FileNotFoundError(f"Daic115 source not found at {DAIC_ROOT}")

    saved_path = list(sys.path)
    sys.path.insert(0, str(DAIC_ROOT))
    try:
        # Step 1 — load lib.ops.rolling directly via spec (it has no internal
        # dependencies on the broken `factors` import path).
        rolling_spec = importlib.util.spec_from_file_location(
            "lib_ops_rolling", DAIC_ROOT / "lib" / "ops" / "rolling.py"
        )
        if rolling_spec is None or rolling_spec.loader is None:
            raise ImportError("failed to spec lib/ops/rolling.py")
        rolling_mod = importlib.util.module_from_spec(rolling_spec)
        rolling_spec.loader.exec_module(rolling_mod)

        # Step 2 — register `factors.ops.rolling` BEFORE lib.ops is imported,
        # since lib.ops.factor_ops calls `from factors.ops.rolling import …`
        # at module-load time.
        factors_pkg = types.ModuleType("factors")
        factors_pkg.__path__ = []  # mark as package
        factors_ops_pkg = types.ModuleType("factors.ops")
        factors_ops_pkg.__path__ = []
        sys.modules["factors"] = factors_pkg
        sys.modules["factors.ops"] = factors_ops_pkg
        sys.modules["factors.ops.rolling"] = rolling_mod

        # Now load alpha191.py
        spec = importlib.util.spec_from_file_location("daic_alpha191", DAIC_ROOT / "alpha191.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to build spec for {DAIC_ROOT / 'alpha191.py'}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["daic_alpha191"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path[:] = saved_path


# ---------------------------------------------------------------------------
# Long → wide conversion for the synthetic panel
# ---------------------------------------------------------------------------


def _build_data_dict(panel: pl.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the long-format polars panel to Daic115's wide-DataFrame dict.

    Each value is a wide :class:`pandas.DataFrame` with ``trade_date`` as
    the index and ``stock_code`` as the columns.

    Daic115 reads these keys: ``open``, ``high``, ``low``, ``close``,
    ``volume``, ``vwap``, ``amount``. (``turn`` and ``liquidity_value``
    are also referenced but only by alpha191_142, which is unfinished.)
    """

    pdf = panel.to_pandas()
    wanted = ("open", "high", "low", "close", "volume", "vwap", "amount")
    out: dict[str, pd.DataFrame] = {}
    for col in wanted:
        wide = pdf.pivot(index="trade_date", columns="stock_code", values=col)
        wide = wide.sort_index()
        out[col] = wide.astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------


def _alpha_function_names(mod: types.ModuleType) -> list[str]:
    """Sorted list of ``alpha191_NNN`` callables in the Daic115 module."""
    names = [
        name
        for name in dir(mod)
        if name.startswith("alpha191_")
        and len(name) == len("alpha191_") + 3
        and name[len("alpha191_") :].isdigit()
        and callable(getattr(mod, name))
    ]
    return sorted(names)


def _flatten_alpha_result(
    result: object,
    expected_dates: int,
    expected_stocks: int,
    columns: list[str],
) -> np.ndarray | None:
    """Flatten Daic115's wide DataFrame output to a long-format float vector.

    Daic115 alphas return either a wide :class:`pandas.DataFrame`
    (date_index × stock_columns) or, occasionally, a Series (one per
    date). We need a single 1-D vector ordered ``stock_code outer,
    trade_date inner`` to match the panel layout.
    """
    import pandas as pd

    if isinstance(result, pd.Series):
        if len(result) != expected_dates:
            return None
        # Broadcast the per-date Series across all stocks.
        wide = pd.DataFrame({c: result for c in columns})
    elif isinstance(result, pd.DataFrame):
        wide = result.reindex(columns=columns)
    else:
        return None

    if wide.shape != (expected_dates, expected_stocks):
        return None

    if wide.dtypes.eq(bool).all():
        wide = wide.astype(np.float64)

    # Stock outer, date inner: transpose then ravel
    return wide.values.T.reshape(-1).astype(np.float64)


def _is_unfinished(fn: object) -> bool:
    """``factor_attr(unfinished=True)`` decorator marks unbuildable alphas."""
    return bool(getattr(fn, "unfinished", False))


def _compute_reference(
    daic_mod: types.ModuleType,
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    """Run Daic115 over the synthetic panel and return the reference.

    Returns
    -------
    reference:
        polars DataFrame with columns ``stock_code``, ``trade_date`` plus
        one ``gtja_NNN`` column per successful alpha.
    succeeded:
        Sorted list of ``gtja_NNN`` column names that produced output.
    failures:
        Mapping ``alpha_name -> short error string`` for diagnostic logging.
    """
    data_dict = _build_data_dict(panel)
    sample_wide = data_dict["close"]
    expected_dates = sample_wide.shape[0]
    expected_stocks = sample_wide.shape[1]
    columns_order = list(sample_wide.columns)

    base = panel.select(["stock_code", "trade_date"]).sort(["stock_code", "trade_date"])

    succeeded: list[str] = []
    failures: dict[str, str] = {}
    output_columns: dict[str, np.ndarray] = {}

    for name in _alpha_function_names(daic_mod):
        fn = getattr(daic_mod, name)
        out_name = f"gtja_{name.split('_')[1]}"

        if _is_unfinished(fn):
            failures[out_name] = "marked unfinished by Daic115"
            continue

        # Pass a fresh deep copy per call. Several Daic115 alpha191_NNN
        # functions mutate the input DataFrames in place (e.g. alpha023 reassigns
        # ``data["close"][...]`` based on a condition), which silently poisons
        # every subsequent alpha that runs against the same dict. Deep-copying
        # the per-OHLCV DataFrames isolates each call. Mirrors the cascade-fix
        # applied to ``build_alpha101_reference.py``.
        fresh_data = {k: v.copy(deep=True) for k, v in data_dict.items()}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = fn(fresh_data)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            failures[out_name] = f"{type(exc).__name__}: {msg[:120]}"
            continue

        flat = _flatten_alpha_result(result, expected_dates, expected_stocks, columns_order)
        if flat is None:
            failures[out_name] = (
                f"unexpected shape/type: {type(result).__name__} "
                f"shape={getattr(result, 'shape', '?')}"
            )
            continue

        output_columns[out_name] = flat
        succeeded.append(out_name)

    reference = base.with_columns(
        [pl.Series(name=name, values=output_columns[name]) for name in succeeded]
    )
    return reference, succeeded, failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    # Make sure tests/ is importable without installing the package.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    _install_compat_shims()
    daic_mod = _load_daic_alpha191()

    # Suppress Daic115's print-on-import noise.
    _stdout_sink = _io.StringIO()

    from tests.factors._synthetic import build_synthetic_panel  # noqa: PLC0415

    panel = build_synthetic_panel(seed=42, n_stocks=10, n_days=60)
    with contextlib.redirect_stdout(_stdout_sink):
        reference, succeeded, failures = _compute_reference(daic_mod, panel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference.write_parquet(args.output)

    # Diagnostics ----------------------------------------------------------
    print(f"[build_gtja191_reference] wrote {args.output}", file=sys.stderr)
    print(
        f"[build_gtja191_reference] shape={reference.shape}, "
        f"alphas={len(succeeded)}, skipped={len(failures)}",
        file=sys.stderr,
    )
    if succeeded:
        print(
            f"[build_gtja191_reference] succeeded: {', '.join(succeeded[:6])}"
            f"{' …' if len(succeeded) > 6 else ''}",
            file=sys.stderr,
        )
    if failures:
        print("[build_gtja191_reference] skipped (with reason):", file=sys.stderr)
        for name, reason in failures.items():
            print(f"    {name}: {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
