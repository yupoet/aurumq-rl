"""Internal utility: extract factor function docstrings to markdown documentation.

This is a pure utility module — it does NOT import polars/pandas and treats
``FactorEntry.impl`` as an opaque callable. It only inspects the docstring.

Public API:

    extract_factor_doc(entry) -> str
        Render a single FactorEntry as a markdown document string.

    write_index_md(entries, output_path, title=...) -> None
        Write a top-level INDEX.md with a table of all factors.

    write_factor_docs(entries, output_dir) -> None
        Write one ``<id>.md`` per factor plus an INDEX.md into ``output_dir``.

    main() -> None
        CLI entry point — extracts docs for both alpha101 + gtja191 registries
        into ``docs/factor_library/``.

The docstrings are expected to follow numpy style with section headers:

    Section Title
    -------------
    body text...

Recognised sections (case-insensitive, leading words):

    * ``WorldQuant Formula`` (or just ``Formula``)
    * ``Legacy AQML``
    * ``Polars Implementation``
    * ``References``

Plus inline keys (single line):

    * ``Required panel columns: ...``
    * ``Direction: ...``  (kept here purely as cross-check; the canonical
      direction is on ``FactorEntry`` itself)
    * ``Category: ...``   (same — canonical lives on entry)
"""

from __future__ import annotations

import inspect
import pathlib
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aurumq_rl.factors.registry import FactorEntry  # noqa: F401


# ---------------------------------------------------------------------------
# Local Protocol so this module is self-contained for unit tests
# ---------------------------------------------------------------------------


@runtime_checkable
class _FactorEntryLike(Protocol):
    """Subset of ``FactorEntry`` we read from. Allows fakes in tests."""

    id: str
    impl: object  # callable; inspected via ``inspect.getdoc``
    direction: str
    category: str
    description: str
    legacy_aqml_expr: str | None
    quality_flag: int
    references: list[str]
    formula_doc_path: str


# ---------------------------------------------------------------------------
# Docstring section parsing
# ---------------------------------------------------------------------------


# Matches a numpy-style header: a non-empty line followed by a line of
# only ``-`` or ``=`` characters (length >= 3) that underlines it.
_SECTION_RE = re.compile(
    r"""
    ^([^\n]+?)\n            # group 1: header text
    [ \t]*([-=]{3,})[ \t]*  # group 2: underline of - or =
    (?:\n|$)
    """,
    re.MULTILINE | re.VERBOSE,
)

# Inline ``Key: value`` style fields (greedy across continuation indentation
# is overkill — these fields are all single-line by convention).
_INLINE_KEY_RE = re.compile(
    r"^[ \t]*(?P<key>[A-Za-z][A-Za-z _]+?)[ \t]*:[ \t]*(?P<value>.+?)[ \t]*$",
    re.MULTILINE,
)


def _parse_sections(docstring: str) -> dict[str, str]:
    """Split a numpy-style docstring into ``{header_lower: body}`` chunks.

    The body is the text between the underline of one header and the start
    of the next header (or end of docstring), stripped of trailing/leading
    blank lines but preserving inner formatting.

    A synthetic ``__summary__`` key holds the leading paragraph (everything
    before the first header).
    """
    if not docstring:
        return {}

    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(docstring))

    if not matches:
        sections["__summary__"] = docstring.strip()
        return sections

    summary = docstring[: matches[0].start()].strip()
    if summary:
        sections["__summary__"] = summary

    for idx, m in enumerate(matches):
        header = m.group(1).strip().lower()
        body_start = m.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(docstring)
        body = docstring[body_start:body_end].strip("\n").rstrip()
        # Dedent: numpy docstrings often have a leading 4-space indent inside
        # the function. inspect.getdoc already strips the common leading
        # indent, so we only normalise trailing whitespace per line here.
        body = "\n".join(line.rstrip() for line in body.splitlines())
        sections[header] = body.strip("\n")

    return sections


def _find_section(sections: Mapping[str, str], *needles: str) -> str | None:
    """Return body of the first section whose header startswith any needle.

    Matching is case-insensitive (sections dict is already lower-cased).
    """
    for header, body in sections.items():
        for needle in needles:
            if header.startswith(needle.lower()):
                return body
    return None


def _extract_inline_field(docstring: str, key: str) -> str | None:
    """Pull a ``Key: value`` line out of a docstring.

    Matches at any indentation level, returns the trimmed value.
    """
    if not docstring:
        return None
    pattern = re.compile(
        rf"^[ \t]*{re.escape(key)}[ \t]*:[ \t]*(?P<value>.+?)[ \t]*$",
        re.MULTILINE | re.IGNORECASE,
    )
    m = pattern.search(docstring)
    if not m:
        return None
    return m.group("value").strip().strip("`")


def _quality_flag_label(flag: int) -> str:
    """Map an integer quality flag to a human-readable label."""
    return {
        0: "ok",
        1: "warn",
        2: "broken",
    }.get(flag, f"unknown({flag})")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


_PLACEHOLDER = "_(not specified)_"
_PURE_CALLABLE = "_pure-callable factor_"


def _format_block(body: str | None) -> str:
    """Return ``body`` if non-empty, else the standard placeholder."""
    if body is None:
        return _PLACEHOLDER
    body = body.strip()
    return body if body else _PLACEHOLDER


def _format_references(docstring_section: str | None, entry_refs: Iterable[str]) -> str:
    """Merge docstring References block with entry.references list.

    Both can be partially populated; we emit a unified bullet list.
    """
    bullets: list[str] = []
    if docstring_section:
        for raw in docstring_section.splitlines():
            line = raw.strip().lstrip("-*").strip()
            if line:
                bullets.append(line)

    for ref in entry_refs:
        ref = (ref or "").strip()
        if ref and ref not in bullets:
            bullets.append(ref)

    if not bullets:
        return _PLACEHOLDER
    return "\n".join(f"- {b}" for b in bullets)


def extract_factor_doc(entry: FactorEntry) -> str:
    """Render a single factor as a markdown document string.

    Parses the impl's docstring for structured sections and combines them
    with the canonical fields on ``entry`` (category/direction/quality).
    """
    raw_doc = inspect.getdoc(entry.impl) or ""
    sections = _parse_sections(raw_doc)

    summary = sections.get("__summary__", "").strip()
    headline = entry.description.strip()
    if not headline and summary:
        # Use the first line of the docstring summary as a fallback headline.
        headline = summary.splitlines()[0].strip().rstrip(".")
    if not headline:
        headline = entry.id

    formula = _find_section(sections, "worldquant formula", "formula")
    polars_notes = _find_section(sections, "polars implementation")
    references_block = _find_section(sections, "references", "reference")
    legacy_doc = _find_section(sections, "legacy aqml")

    legacy_value = entry.legacy_aqml_expr
    if legacy_value is None:
        legacy_text = _PURE_CALLABLE
    else:
        legacy_value = legacy_value.strip()
        legacy_text = legacy_value if legacy_value else _PURE_CALLABLE

    # If the docstring has its own Legacy AQML block, prefer the richer one
    # (commentary), but always include the canonical expression too if both
    # exist and differ trivially.
    legacy_combined: str
    if legacy_doc and entry.legacy_aqml_expr:
        legacy_combined = f"{legacy_doc}\n\n```\n{entry.legacy_aqml_expr.strip()}\n```"
    elif legacy_doc:
        legacy_combined = legacy_doc
    else:
        legacy_combined = legacy_text

    required_cols = _extract_inline_field(raw_doc, "Required panel columns") or _PLACEHOLDER

    quality_label = _quality_flag_label(entry.quality_flag)

    references_text = _format_references(references_block, entry.references)

    title = f"# {entry.id} — {headline}" if headline else f"# {entry.id}"

    parts = [
        title,
        "",
        f"> **Category**: {entry.category or _PLACEHOLDER} | "
        f"**Direction**: {entry.direction or _PLACEHOLDER} | "
        f"**Quality**: {quality_label}",
        "",
        "## Original WorldQuant Formula",
        "",
        _format_block(formula),
        "",
        "## Intuition (人工补)",
        "",
        _PLACEHOLDER,
        "",
        "## Legacy AQML Expression (deprecated)",
        "",
        _format_block(legacy_combined),
        "",
        "## Polars Implementation Notes",
        "",
        _format_block(polars_notes),
        "",
        "## Required Panel Columns",
        "",
        required_cols,
        "",
        "## References",
        "",
        references_text,
        "",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# INDEX.md + per-factor file writers
# ---------------------------------------------------------------------------


def _index_table_row(entry: FactorEntry) -> str:
    desc = (entry.description or "").replace("|", "\\|")
    if len(desc) > 80:
        desc = desc[:77].rstrip() + "..."
    return (
        f"| {entry.id} "
        f"| {entry.category or '-'} "
        f"| {entry.direction or '-'} "
        f"| {desc or '-'} "
        f"| {_quality_flag_label(entry.quality_flag)} |"
    )


def write_index_md(
    entries: dict[str, FactorEntry],
    output_path: pathlib.Path,
    title: str = "Factor Library Index",
) -> None:
    """Write an INDEX.md with a markdown table of all factors.

    Rows are ordered by the dict iteration order (insertion order in Py3.7+).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# {title} ({len(entries)} factors)",
        "",
        "| ID | Category | Direction | Description | Quality |",
        "|---|---|---|---|---|",
    ]
    for entry in entries.values():
        lines.append(_index_table_row(entry))
    lines.append("")  # trailing newline

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_factor_docs(
    entries: dict[str, FactorEntry],
    output_dir: pathlib.Path,
) -> None:
    """Write one ``<id>.md`` per factor plus an INDEX.md into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries.values():
        doc = extract_factor_doc(entry)
        (output_dir / f"{entry.id}.md").write_text(doc, encoding="utf-8")
    write_index_md(entries, output_dir / "INDEX.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _try_import_registry(name: str) -> dict[str, FactorEntry]:
    """Best-effort import of a registry submodule's ``REGISTRY`` dict.

    Returns ``{}`` if the registry is unavailable (skeleton not implemented).
    """
    try:
        import importlib

        module = importlib.import_module(f"aurumq_rl.factors.{name}")
    except ImportError:
        return {}
    registry = getattr(module, "REGISTRY", None)
    if not isinstance(registry, dict):
        return {}
    return registry  # type: ignore[return-value]


def main() -> None:
    """CLI entry point: extract docs for both alpha101 + gtja191 registries.

    Output goes to ``<repo>/docs/factor_library/{alpha101,gtja191}/``.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    out_root = repo_root / "docs" / "factor_library"

    for name, title in (
        ("alpha101", "Alpha101 Factor Library Index"),
        ("gtja191", "GTJA191 Factor Library Index"),
    ):
        entries = _try_import_registry(name)
        if not entries:
            print(f"[_docs] skipping {name}: registry empty or not yet implemented")
            continue
        out_dir = out_root / name
        write_factor_docs(entries, out_dir)
        # Override INDEX.md title with the family-specific one.
        write_index_md(entries, out_dir / "INDEX.md", title=title)
        print(f"[_docs] wrote {len(entries)} factors + INDEX.md to {out_dir}")


if __name__ == "__main__":
    main()
