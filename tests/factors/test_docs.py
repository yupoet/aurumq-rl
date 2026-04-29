"""Tests for ``aurumq_rl.factors._docs`` markdown extraction utility.

These tests use lightweight fake FactorEntry objects so they do NOT depend on
the real registry (which lives in Phase A-1 deliverables).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from aurumq_rl.factors._docs import (
    extract_factor_doc,
    write_factor_docs,
    write_index_md,
)

# ---------------------------------------------------------------------------
# Minimal stand-in for FactorEntry
# ---------------------------------------------------------------------------


@dataclass
class _FakeEntry:
    """Mirror of the real FactorEntry shape, sufficient for _docs.py."""

    id: str
    impl: object
    direction: str
    category: str
    description: str
    legacy_aqml_expr: str | None = None
    quality_flag: int = 0
    references: list[str] = field(default_factory=list)
    formula_doc_path: str = ""


# ---------------------------------------------------------------------------
# Test fixtures: fake factor functions with rich docstrings
# ---------------------------------------------------------------------------


def _full_alpha001(panel):
    """Alpha #001 — Rank of squared-clip ts_argmax within past 5 days.

    WorldQuant Formula (Kakushadze 2015, eq. 1)
    -------------------------------------------
        rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5

    Legacy AQML Expression (deprecated 2026-04-29, kept as cross-check)
    -------------------------------------------------------------------
        Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5

    Polars Implementation Notes
    ---------------------------
    1. Conditional input: when returns < 0, replace ``close`` with rolling
       std of returns over a 20-day window.
    2. SignedPower(x, 2) is computed as ``sign(x) * abs(x).pow(2)``.
    3. Cross-sectional rank uses ``rank(method="average")`` over each
       trade_date partition.

    Required panel columns: ``returns``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``

    References
    ----------
    - Kakushadze 2015, "101 Formulaic Alphas", arXiv:1601.00991, eq. 1
    - STHSF/alpha101 (MIT) for pandas reference impl
    """


def _pure_callable_factor(panel):
    """Custom momentum factor with no legacy AQML form.

    WorldQuant Formula
    ------------------
        ts_rank(close, 60) * volume_ratio_5d

    Polars Implementation Notes
    ---------------------------
    Trivial — uses ``rolling_quantile`` and elementwise multiply.

    Required panel columns: ``close``, ``volume``

    References
    ----------
    - Internal note 2026-04-22
    """


def _bare_factor(panel):
    """A factor with only a one-line docstring."""


# ---------------------------------------------------------------------------
# extract_factor_doc
# ---------------------------------------------------------------------------


def test_extract_factor_doc_with_full_docstring() -> None:
    # Arrange
    entry = _FakeEntry(
        id="alpha001",
        impl=_full_alpha001,
        direction="reverse",
        category="volatility",
        description="Rank of squared-clip ts_argmax within past 5 days",
        legacy_aqml_expr="Rank(Ts_ArgMax(SignedPower(If(returns < 0, Ts_Std(returns, 20), close), 2), 5)) - 0.5",
        quality_flag=0,
        references=["Custom registry-only ref entry"],
    )

    # Act
    md = extract_factor_doc(entry)

    # Assert — header
    assert md.startswith("# alpha001 — Rank of squared-clip ts_argmax")
    # Metadata line
    assert "**Category**: volatility" in md
    assert "**Direction**: reverse" in md
    assert "**Quality**: ok" in md

    # All required H2 sections present
    for section in (
        "## Original WorldQuant Formula",
        "## Intuition (人工补)",
        "## Legacy AQML Expression (deprecated)",
        "## Polars Implementation Notes",
        "## Required Panel Columns",
        "## References",
    ):
        assert section in md, f"missing section: {section}"

    # Formula body extracted
    assert "rank(Ts_ArgMax(SignedPower" in md
    # Polars notes extracted (numbered list preserved)
    assert "1. Conditional input" in md
    assert "Cross-sectional rank" in md
    # Required columns extracted as inline field
    assert "returns" in md and "trade_date" in md
    # References merged from docstring + entry.references
    assert "Kakushadze 2015" in md
    assert "Custom registry-only ref entry" in md
    # Legacy AQML uses entry expr (canonical) — present somewhere
    assert "Rank(Ts_ArgMax(SignedPower(If(returns < 0" in md


def test_extract_factor_doc_missing_legacy_aqml() -> None:
    # Arrange — no legacy expression on entry, no Legacy AQML section in doc
    entry = _FakeEntry(
        id="custom_mom",
        impl=_pure_callable_factor,
        direction="normal",
        category="momentum",
        description="Pure-callable momentum factor",
        legacy_aqml_expr=None,
        quality_flag=0,
    )

    # Act
    md = extract_factor_doc(entry)

    # Assert
    assert "## Legacy AQML Expression (deprecated)" in md
    assert "_pure-callable factor_" in md
    # Sanity: other sections still present
    assert "## Original WorldQuant Formula" in md
    assert "ts_rank(close, 60)" in md


def test_extract_factor_doc_with_minimal_docstring() -> None:
    """A factor with only a summary line should still render with placeholders."""
    # Arrange
    entry = _FakeEntry(
        id="bare",
        impl=_bare_factor,
        direction="normal",
        category="misc",
        description="A factor with only a one-line docstring",
        legacy_aqml_expr=None,
        quality_flag=1,
    )

    # Act
    md = extract_factor_doc(entry)

    # Assert
    assert md.startswith("# bare — A factor with only a one-line docstring")
    assert "**Quality**: warn" in md
    # Missing sections should fall back to placeholder
    assert "_(not specified)_" in md


# ---------------------------------------------------------------------------
# write_index_md
# ---------------------------------------------------------------------------


def _three_entries() -> dict[str, _FakeEntry]:
    return {
        "alpha001": _FakeEntry(
            id="alpha001",
            impl=_full_alpha001,
            direction="reverse",
            category="volatility",
            description="Rank of squared-clip ts_argmax",
            legacy_aqml_expr="Rank(...)",
            quality_flag=0,
        ),
        "alpha002": _FakeEntry(
            id="alpha002",
            impl=_pure_callable_factor,
            direction="normal",
            category="momentum",
            description="Pure-callable mom",
            legacy_aqml_expr=None,
            quality_flag=1,
        ),
        "alpha003": _FakeEntry(
            id="alpha003",
            impl=_bare_factor,
            direction="reverse",
            category="misc",
            description="Bare factor",
            quality_flag=2,
        ),
    }


def test_write_index_md_table_format(tmp_path: pathlib.Path) -> None:
    # Arrange
    entries = _three_entries()
    out = tmp_path / "INDEX.md"

    # Act
    write_index_md(entries, out, title="Test Library Index")

    # Assert
    content = out.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Title with count
    assert lines[0] == "# Test Library Index (3 factors)"
    # Markdown table headers
    assert "| ID | Category | Direction | Description | Quality |" in lines
    assert "|---|---|---|---|---|" in lines
    # Rows for each entry
    assert any(line.startswith("| alpha001 ") and "volatility" in line for line in lines)
    assert any(line.startswith("| alpha002 ") and "momentum" in line for line in lines)
    assert any(line.startswith("| alpha003 ") and "misc" in line for line in lines)
    # Quality flag mapping is human-readable
    assert any("| ok |" in line for line in lines)
    assert any("| warn |" in line for line in lines)
    assert any("| broken |" in line for line in lines)


# ---------------------------------------------------------------------------
# write_factor_docs
# ---------------------------------------------------------------------------


def test_write_factor_docs_creates_per_factor_files(tmp_path: pathlib.Path) -> None:
    # Arrange
    entries = _three_entries()

    # Act
    write_factor_docs(entries, tmp_path)

    # Assert — N factors + INDEX.md = 4 files
    files = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert files == ["INDEX.md", "alpha001.md", "alpha002.md", "alpha003.md"]

    # Each per-factor file is valid markdown with the canonical title
    for fid in ("alpha001", "alpha002", "alpha003"):
        body = (tmp_path / f"{fid}.md").read_text(encoding="utf-8")
        assert body.startswith(f"# {fid} —"), f"{fid}.md missing title"
        assert "## Original WorldQuant Formula" in body
        assert "## Intuition (人工补)" in body

    # INDEX.md still has the table
    index = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "| ID | Category | Direction | Description | Quality |" in index


# ---------------------------------------------------------------------------
# Edge case: long description gets truncated in INDEX rows
# ---------------------------------------------------------------------------


def test_index_md_truncates_long_descriptions(tmp_path: pathlib.Path) -> None:
    # Arrange
    long_desc = "x" * 200
    entries = {
        "alpha999": _FakeEntry(
            id="alpha999",
            impl=_bare_factor,
            direction="normal",
            category="misc",
            description=long_desc,
            quality_flag=0,
        )
    }
    out = tmp_path / "INDEX.md"

    # Act
    write_index_md(entries, out)

    # Assert
    content = out.read_text(encoding="utf-8")
    # Truncated to 80 chars + ellipsis
    assert "x" * 80 not in content  # original 200 chars
    assert "..." in content
