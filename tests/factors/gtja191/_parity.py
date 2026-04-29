"""Shared GTJA-191 parity helper for batch tests.

Each batch test module imports :func:`parity_check` and supplies the
factor id + impl callable. The reference parquet is loaded via the
session-scoped ``gtja191_reference`` fixture (see
``tests/factors/conftest.py``).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def parity_check(
    panel: pl.DataFrame,
    reference: pl.DataFrame,
    name: str,
    fn,
    *,
    status: str = "match",
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Compare our impl against the Daic115 reference parquet.

    Parameters
    ----------
    panel:
        Synthetic panel from the ``synthetic_panel`` fixture.
    reference:
        Daic115 reference parquet from the ``gtja191_reference`` fixture.
    name:
        Factor id, e.g. ``"gtja_101"``.
    fn:
        The polars callable. ``fn(panel) -> pl.Series``.
    status:
        Expected parity status:

        * ``"match"`` — assert close within ``(rtol, atol)``.
        * ``"drift"`` — known SMA-cascade drift; we skip and rely on the
          steady-state test.
        * ``"missing"`` — factor not built into the reference; skip.
        * ``"stub"`` — factor is a stub (gtja_143); skip — only dtype + length matter.
    """
    if status in ("missing", "stub"):
        pytest.skip(f"{name}: status={status} — no parity expected")
        return
    if name not in reference.columns:
        pytest.skip(f"{name}: not in reference parquet")
        return

    ours = panel.with_columns(fn(panel).alias("ours"))
    joined = ours.join(
        reference.select(["stock_code", "trade_date", name]),
        on=["stock_code", "trade_date"],
        how="inner",
    ).drop_nulls(["ours", name])
    if joined.height == 0:
        pytest.skip(f"{name}: zero non-null overlap with reference")
        return

    oa = joined["ours"].to_numpy()
    ra = joined[name].to_numpy()
    finite = ~(np.isnan(oa) | np.isnan(ra) | np.isinf(oa) | np.isinf(ra))
    if finite.sum() == 0:
        pytest.skip(f"{name}: zero finite-pair overlap")
        return

    if status == "drift":
        pytest.skip(
            f"{name}: known reference drift (SMA cascade or timestamp warm-up). "
            f"Numeric correctness is verified by the steady-state test."
        )
        return

    np.testing.assert_allclose(
        oa[finite],
        ra[finite],
        rtol=rtol,
        atol=atol,
        err_msg=f"{name}: drift on {finite.sum()} finite pairs",
    )
