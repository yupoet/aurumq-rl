"""Build the locked alpha101 reference parquet from STHSF/alpha101 (MIT).

This script is run **once** during Phase A (or whenever the STHSF reference
implementation is updated) to generate
``tests/factors/alpha101/data/alpha101_reference.parquet``. The parquet
is committed to git and serves as a numerical baseline for the polars
implementations of the WorldQuant 101 alphas that live in
``aurumq_rl.factors.alpha101``.

Usage
-----
    python scripts/reference_data/build_alpha101_reference.py

License notice
--------------
``STHSF/alpha101`` is MIT-licensed. We import its implementation here at
**build time** purely to compute reference values; we do NOT vendor any of
its code into our runtime. The runtime only loads the resulting parquet,
which contains numerical values (no copyrightable code).

Implementation notes
--------------------

* The STHSF code targets pandas 0.x conventions. To make it run under
  pandas 2.x we monkey-patch the removed ``DataFrame.as_matrix`` method
  on import. This patch lives in this build script only — production
  code never touches it.

* STHSF's ``alpha101_single.calculate()`` packages results into
  ``pd.Panel``, which pandas 1.0+ removed. We bypass ``calculate()`` and
  call each ``alpha###()`` method individually.

* Several STHSF methods rely on later-removed pandas APIs (``pd.Panel``
  inside the method body, deprecated ``decay_linear_pn``, etc.) or carry
  pre-existing bugs (shape mismatches, builtin ``sum`` shadowing). We
  catch and skip those — the surviving alphas are still enough to build
  a meaningful reference baseline. Skipped alphas are reported on stderr.

Output
------

The parquet has one row per ``(stock_code, trade_date)`` and one column
per alpha that built successfully. NaN values reflect the natural
warm-up of rolling windows (e.g. alpha004 uses a 9-day ``ts_rank``, so
the first 8 rows of each stock are NaN).
"""

from __future__ import annotations

import argparse
import importlib.util
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
STHSF_SRC: pathlib.Path = REPO_ROOT / "refs" / "alpha101" / "STHSF__alpha101" / "src"
DEFAULT_OUTPUT: pathlib.Path = (
    REPO_ROOT / "tests" / "factors" / "alpha101" / "data" / "alpha101_reference.parquet"
)


# ---------------------------------------------------------------------------
# Compatibility shims for STHSF (pandas 0.x → 2.x)
# ---------------------------------------------------------------------------


def _install_pandas_compat_shims() -> None:
    """Restore APIs removed by pandas ≥ 1.0 that STHSF relies on.

    Currently only ``DataFrame.as_matrix`` (returns ``df.values``). We do
    NOT try to back-port ``pd.Panel`` — the alphas that need it are
    listed in :data:`_KNOWN_PANEL_ALPHAS` and skipped at calculation time.
    """
    import pandas as pd

    if not hasattr(pd.DataFrame, "as_matrix"):

        def _as_matrix(self):  # type: ignore[no-untyped-def]
            return self.values

        pd.DataFrame.as_matrix = _as_matrix  # type: ignore[attr-defined]


def _load_sthsf() -> object:
    """Import STHSF's ``Alpha101`` class via in-memory module synthesis.

    The STHSF source uses ``from alpha_101.factor_util import *``, which
    expects a top-level ``alpha_101`` package on the import path. Rather
    than mutate ``sys.path`` (which leaks state), we synthesise an
    ``alpha_101`` package directly into ``sys.modules`` and point
    ``alpha_101.factor_util`` at the file under ``refs/``.
    """
    factor_util_path = STHSF_SRC / "factor_util.py"
    alpha101_path = STHSF_SRC / "alpha101_single.py"
    if not factor_util_path.exists() or not alpha101_path.exists():
        raise FileNotFoundError(
            f"STHSF reference files missing under {STHSF_SRC}. "
            "Did you fetch the refs/ submodule?"
        )

    # Synthesise alpha_101 package
    parent = types.ModuleType("alpha_101")
    parent.__path__ = []  # marks as package
    sys.modules["alpha_101"] = parent

    # Load alpha_101.factor_util
    spec = importlib.util.spec_from_file_location("alpha_101.factor_util", factor_util_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to build spec for {factor_util_path}")
    factor_util = importlib.util.module_from_spec(spec)
    sys.modules["alpha_101.factor_util"] = factor_util
    spec.loader.exec_module(factor_util)

    # Load the alpha101_single module
    spec2 = importlib.util.spec_from_file_location("alpha101_single", alpha101_path)
    if spec2 is None or spec2.loader is None:
        raise ImportError(f"failed to build spec for {alpha101_path}")
    mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod)

    return mod.Alpha101


# ---------------------------------------------------------------------------
# Reference computation
# ---------------------------------------------------------------------------


def _alpha_method_names(alpha_obj: object) -> list[str]:
    """Sorted list of ``alpha###`` callables on the STHSF instance.

    Excludes the bare ``alpha`` key (alpha101 in the original ``calculate``
    return) and any non-alpha attribute.
    """
    names: list[str] = []
    for attr in dir(alpha_obj):
        if not attr.startswith("alpha"):
            continue
        # alpha###(...) — three digits exactly
        if len(attr) != len("alpha") + 3:
            continue
        if not attr[5:].isdigit():
            continue
        if not callable(getattr(alpha_obj, attr)):
            continue
        names.append(attr)
    return sorted(names)


def _flatten_alpha_result(
    name: str,
    result: pd.DataFrame | pd.Series,
    expected_rows: int,
    n_stocks: int,
) -> np.ndarray | None:
    """Flatten a wide alpha output (date × stock) into a long-format vector.

    The returned vector has length ``n_days * n_stocks`` ordered by
    ``(stock outer, date inner)`` — matching the ordering of the
    synthetic panel (stocks are repeated per date contiguously).

    Returns ``None`` if the shape is not what we expect (defensive: lets
    the caller skip a malformed alpha rather than crash).
    """
    import pandas as pd

    if isinstance(result, pd.Series):
        # Some alphas (e.g. boolean conditions) return a Series — broadcast
        # to wide if its length matches the number of dates.
        if len(result) == expected_rows // n_stocks:
            wide = pd.DataFrame({c: result for c in range(n_stocks)})
        else:  # unrecognised shape
            return None
    elif isinstance(result, pd.DataFrame):
        wide = result
    else:
        return None

    if wide.shape != (expected_rows // n_stocks, n_stocks):
        return None

    # Boolean → float
    if wide.dtypes.eq(bool).all():
        wide = wide.astype(np.float64)

    # Ravel as stock outer, date inner. STHSF wide DataFrames are
    # date_index × stock_columns, so transpose-then-flatten gives us the
    # right ordering.
    return wide.values.T.reshape(-1).astype(np.float64)


def _compute_reference(
    alpha_cls: type,
    panel: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], dict[str, str]]:
    """Run STHSF over the polars synthetic panel and return a reference.

    Cascade-pollution fix (2026-04-28)
    ----------------------------------
    STHSF's ``alpha001`` MUTATES ``self.close`` in place
    (``self.close = signedpower(...)`` rebinds with a transformed view that
    points back into the same numpy buffer that holds ``self.df_data['close']``).
    A handful of other methods do similar in-place rebinds. Calling every
    alpha method on a single shared instance therefore poisons all subsequent
    computations. To produce a faithful reference we instantiate a **fresh**
    :class:`Alpha101` from a **deep copy** of the panel pandas DataFrame for
    each alpha invocation, then discard the instance.

    Returns
    -------
    reference:
        polars DataFrame with columns ``stock_code``, ``trade_date`` plus
        one column per successful alpha.
    succeeded:
        Sorted list of alpha names that produced usable output.
    failures:
        Mapping ``alpha_name -> short error string`` for diagnostic logging.
    """
    # Late import — only this script needs pandas
    from tests.factors._synthetic import to_pandas_for_sthsf  # noqa: PLC0415

    df_data = to_pandas_for_sthsf(panel)

    def _deep_copy_dict(d: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        # ``to_pandas_for_sthsf`` returns a ``dict[str, pd.DataFrame]``; each
        # frame must be deep-copied so STHSF's in-place rebinds (e.g. alpha001
        # ``self.close = signedpower(...)``) cannot poison the underlying
        # numpy buffer shared by the next alpha invocation.
        return {key: frame.copy(deep=True) for key, frame in d.items()}

    n_stocks = panel["stock_code"].n_unique()
    n_days = panel["trade_date"].n_unique()
    expected_rows = n_days * n_stocks

    # Sort the panel the same way we will assemble the reference: stock
    # outer, date inner. This is the order produced by build_synthetic_panel.
    base = panel.select(["stock_code", "trade_date"]).sort(["stock_code", "trade_date"])

    # Discover the alpha-method names once from a probe instance — this is a
    # read-only operation (we don't call any of the methods here).
    probe = alpha_cls(_deep_copy_dict(df_data))
    method_names = _alpha_method_names(probe)
    del probe

    succeeded: list[str] = []
    failures: dict[str, str] = {}
    columns: dict[str, np.ndarray] = {}

    for name in method_names:
        # CRITICAL: deep-copy each frame so any in-place mutation by STHSF
        # (notably alpha001 rebinding self.close) is contained.
        fresh_df = _deep_copy_dict(df_data)
        instance = alpha_cls(fresh_df)
        try:
            with warnings.catch_warnings():
                # STHSF triggers a sea of FutureWarnings under modern pandas;
                # silence them so the build log stays readable.
                warnings.simplefilter("ignore")
                result = getattr(instance, name)()
        except Exception as exc:  # noqa: BLE001 — defensive, see docstring
            failures[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
            del instance, fresh_df
            continue

        flat = _flatten_alpha_result(name, result, expected_rows, n_stocks)
        del instance, fresh_df
        if flat is None:
            failures[name] = f"unexpected shape/type: {type(result).__name__}"
            continue

        columns[name] = flat
        succeeded.append(name)

    reference = base.with_columns(
        [pl.Series(name=name, values=columns[name]) for name in succeeded]
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

    _install_pandas_compat_shims()
    alpha_cls = _load_sthsf()

    # STHSF's ``decay_linear_pn`` prints the input shape on every call. That
    # noise crowds out the real diagnostics, so redirect stdout for the
    # reference build only.
    import contextlib
    import io as _io
    _stdout_sink = _io.StringIO()

    # Late import: we monkeyed sys.path to make this work in scripts mode.
    from tests.factors._synthetic import build_synthetic_panel  # noqa: PLC0415

    panel = build_synthetic_panel(seed=42, n_stocks=10, n_days=60)
    with contextlib.redirect_stdout(_stdout_sink):
        reference, succeeded, failures = _compute_reference(alpha_cls, panel)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference.write_parquet(args.output)

    # Diagnostics ----------------------------------------------------------
    print(f"[build_alpha101_reference] wrote {args.output}", file=sys.stderr)
    print(
        f"[build_alpha101_reference] shape={reference.shape}, "
        f"alphas={len(succeeded)}, skipped={len(failures)}",
        file=sys.stderr,
    )
    if succeeded:
        print(
            f"[build_alpha101_reference] succeeded: {', '.join(succeeded[:6])}"
            f"{' …' if len(succeeded) > 6 else ''}",
            file=sys.stderr,
        )
    if failures:
        print("[build_alpha101_reference] skipped (with reason):", file=sys.stderr)
        for name, reason in failures.items():
            print(f"    {name}: {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
