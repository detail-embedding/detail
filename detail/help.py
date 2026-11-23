"""In-app help registry (single-sourced, Qt-free).

A `?` in the GUI references a topic key; this maps it to a docs page + anchor.
No prose is authored here — the in-app panel renders the doc markdown (via
:func:`extract_section`) and the tooltip its lead sentence; parameter help comes
from the pydantic model. The anti-drift test (``tests/test_help.py``) asserts every
topic resolves to a real page + anchor, so a doc that drops an anchor fails CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from detail import docs_access
from detail import surfaces


@dataclass(frozen=True)
class HelpTopic:
    """A docs target: a nav ``page`` (no extension) and an optional element ``anchor``."""

    page: str
    anchor: str | None = None


# Concept topics — each points at a term or section in the shipped docs (no prose here).
# The `?` surfaces in the GUI (PR3/PR6) reference these keys.
HELP: dict[str, HelpTopic] = {
    "recording": HelpTopic("concepts", "recording"),
    # Field-level topics point at a terse glossary term; section/concept topics point at
    # the fuller concepts entry.
    "text-prompt": HelpTopic("glossary", "text-prompt"),
    "class-name": HelpTopic("glossary", "class-name"),
    "field-of-view": HelpTopic("concepts", "field-of-view"),
    "roi": HelpTopic("concepts", "roi-poi"),
    "poi": HelpTopic("concepts", "roi-poi"),
    "distance-calibration": HelpTopic("concepts", "distance-calibration"),
    "matcher": HelpTopic("concepts", "behavior-query-eval"),
    "behavior-query-eval": HelpTopic("concepts", "behavior-query-eval"),
    "behavior-query": HelpTopic("concepts", "behavior-query-eval"),
    "embedding-run": HelpTopic("concepts", "embedding-run"),
    "compute-target": HelpTopic("concepts", "compute-target"),
    "model-kinds": HelpTopic("concepts", "model-variant"),
    "location-export": HelpTopic("concepts", "location-export"),
    "tuned-settings": HelpTopic("gui", "tuned-settings"),
    "comparison-source": HelpTopic("gui", "comparison-source"),
    "comparison-measure": HelpTopic("gui", "comparison-measure"),
    "comparison-groups": HelpTopic("gui", "comparison-groups"),
    "comparison-levels": HelpTopic("gui", "comparison-levels"),
    "comparison-design": HelpTopic("gui", "comparison-design"),
    "comparison-unit": HelpTopic("gui", "one-observation"),
    "comparison-windows": HelpTopic("gui", "comparison-windows"),
    "comparison-recordings": HelpTopic("gui", "comparison-recordings"),
    "comparison-measures": HelpTopic("gui", "comparison-measures"),
    "comparison-floor": HelpTopic("gui", "comparison-floor"),
    "comparison-locations": HelpTopic("gui", "comparison-locations"),
    "comparison-snapshot": HelpTopic("gui", "comparison-snapshot"),
    "gpu": HelpTopic("getting-started", "gpu-and-cuda"),
    "hf-token": HelpTopic("getting-started", "hugging-face-access"),
}

# Screen topics — one per navigable sidebar kind (detail/gui/app.py::_navigate), so every page
# can carry a page-top `?`. Pointed at existing doc anchors for now; the doc reorg repoints these
# to dedicated per-screen pages. test_docs_map keeps this in sync with the actual nav kinds.
PAGE_DOCS: dict[str, HelpTopic] = {
    "config_group": HelpTopic("gui", "configuration"),
    "model_manager_group": HelpTopic("gui", "models"),
    "keypoints_manager_group": HelpTopic("gui", "keypoints-manager"),
    "gpu_cuda": HelpTopic("getting-started", "gpu-and-cuda"),
    "sam3_backend": HelpTopic("getting-started", "sam3-backend"),
    "hugging_face": HelpTopic("getting-started", "hugging-face-access"),
    "global_models": HelpTopic("gui", "models"),
    "distill_model": HelpTopic("gui", "distill"),
    "distill_run": HelpTopic("gui", "distill"),
    "datasets": HelpTopic("gui", "distill"),
    "dataset": HelpTopic("gui", "distill"),
    "models": HelpTopic("gui", "distill"),
    "model": HelpTopic("gui", "distill"),
    "experiments_group": HelpTopic("gui", "experiments"),
    "active_processes": HelpTopic("gui", "active-processes"),
    "experiment": HelpTopic("gui", "experiment"),
    "recordings": HelpTopic("gui", "recordings"),
    "recording": HelpTopic("gui", "recording-detail"),
    "metadata": HelpTopic("gui", "recording-information"),
    "runs": HelpTopic("gui", "runs"),
    "run": HelpTopic("gui", "runs"),
    "analysis": HelpTopic("gui", "analysis"),
    "analysis_run": HelpTopic("gui", "analysis"),
    "comparisons": HelpTopic("gui", "comparisons"),
    "comparison": HelpTopic("gui", "comparisons"),
    "tools_group": HelpTopic("gui", "tools"),
    "quick_segment": HelpTopic("gui", "tools"),
    "tool": HelpTopic("gui", "tools"),
    "tool_run": HelpTopic("gui", "tools"),
    # No "documentation" entry: that screen *is* the docs (it has its own "Open full
    # docs" button), so a page-top "Info" ? pointing back at the index is redundant.
}

# Pages generated at build time (no source .md under docs/); a topic may still link to them.
_GENERATED_PAGES = frozenset(
    {
        "reference/parameters",
        "reference/cli",
        "reference/api",
        "reference/manifest",
        "input-images",
        "embedding-models",
        "querying-methods",
        "behavior-query-eval",
    }
)

_PARAMETERS_PAGE = "reference/parameters"


def module_topic(module_id: str) -> HelpTopic:
    """A module's own doc page (``modules/<id>``, generated by scripts/gen_docs_modules.py)."""
    return HelpTopic(f"modules/{module_id}")


def field_topic(module_id: str, field: str) -> HelpTopic:
    """A field's anchored heading on its module page (``modules/<id>#<field>``). The anchor is
    the field's own name (PR 1 emits ``### <label> {#<field>}``), so no slug math is needed.
    """
    return HelpTopic(f"modules/{module_id}", field)


def _heading_slugs(markdown: str) -> set[str]:
    """Slugs of every ATX heading in a markdown string (the anchors mkdocs will emit)."""
    slugs = set()
    for line in markdown.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            slugs.add(_slugify(heading.group(1)))
    return slugs


def page_source(page: str) -> Path | None:
    """The source ``.md`` for a page, or ``None`` if it is generated (no on-disk source)."""
    if page in _GENERATED_PAGES:
        return None
    path = docs_access.docs_source_dir() / "docs" / f"{page}.md"
    return path if path.is_file() else None


def page_text(page: str) -> str | None:
    """A page's source markdown **as the built site sees it**, or ``None`` if it is generated.

    Stripped through ``surfaces.strip_hidden``, because prose for a hidden capability is marked
    in place rather than deleted (§25) — and the two readers below feed the in-app help panel,
    which renders from source. Reading the raw file here would make the panel a second door onto
    a capability the site drops: the exact leak ``detail/surfaces.py`` exists to prevent.
    """
    source = page_source(page)
    if source is None:
        return None
    return surfaces.strip_hidden(
        source.read_text(encoding="utf-8-sig"), where=str(source)
    )


_SLUG_STRIP = re.compile(r"[^\w\s-]")


def _slugify(text: str) -> str:
    """GitHub/pymdownx-style slug: lowercase, drop punctuation, spaces/underscores → ``-``."""
    return re.sub(r"[\s_]+", "-", _SLUG_STRIP.sub("", text.strip().lower()))


def _anchor_marker(anchor: str) -> re.Pattern:
    """Matches an explicit attr_list id — ``{: #anchor }`` or ``{#anchor}``."""
    return re.compile(r"\{:?\s*#" + re.escape(anchor) + r"\b")


def _module_markdown(module_id: str) -> str | None:
    """The generated markdown for a module page, or None if the id isn't a module."""
    from detail.configs.describe import iter_config_models, render_module_markdown

    model = dict(iter_config_models()).get(module_id)
    return render_module_markdown(module_id, model) if model is not None else None


def anchor_exists(page: str, anchor: str | None) -> bool:
    """Whether ``page`` defines ``anchor`` (or is a valid page-top target when ``anchor`` is None)."""
    if page.startswith("modules/"):  # generated per-module page (no source .md)
        markdown = _module_markdown(page[len("modules/") :])
        if markdown is None:
            return False
        if anchor is None:
            return True
        return bool(
            _anchor_marker(anchor).search(markdown)
        ) or anchor in _heading_slugs(markdown)
    if page == _PARAMETERS_PAGE:
        if anchor is None:
            return True
        from detail.configs.describe import render_parameters_markdown  # lazy

        return anchor in _heading_slugs(render_parameters_markdown())
    source = page_source(page)
    if (
        source is None
    ):  # generated page with no source — only page-top links are checkable
        return anchor is None and page in _GENERATED_PAGES
    if anchor is None:
        return True
    text = page_text(page) or ""
    if _anchor_marker(anchor).search(text):
        return True
    for line in text.splitlines():  # an existing heading whose slug matches
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading and _slugify(heading.group(1)) == anchor:
            return True
    return False


def url_for(topic: HelpTopic) -> str:
    """The full-docs URL for a topic (file:// offline, hosted when configured)."""
    return docs_access.help_url(topic.page, topic.anchor)


def body_for(topic: HelpTopic) -> str | None:
    """The panel markdown for a topic, read from the docs (None if unreadable)."""
    section = extract_section(topic.page, topic.anchor)
    return section[1] if section else None


def doc_url(key: str) -> str:
    """The full-docs URL for a HELP topic key."""
    return url_for(HELP[key])


def topic_body(key: str) -> str | None:
    """The panel markdown for a HELP topic key."""
    return body_for(HELP[key])


_INLINE = [
    (re.compile(r"`([^`]+)`"), r"\1"),  # code span
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),  # bold
    (re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)"), r"\1"),  # italic
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # link → text
]


def _plain(text: str) -> str:
    for pattern, repl in _INLINE:
        text = pattern.sub(repl, text)
    return text


def _lead_sentence(body: str) -> str:
    """A one-line tooltip: the definition's first sentence, markdown stripped."""
    text = re.sub(
        r"^\s*[-*]\s+", "", body.strip()
    )  # drop a list marker (needs a space)
    text = re.sub(r"^\*\*[^*]+\*\*\s*[—-]\s*", "", text)  # drop "**Term** — "
    text = _plain(" ".join(text.split()))
    match = re.search(r"\.(\s|$)", text)
    sentence = text[: match.start() + 1] if match else text
    return sentence[:240].strip()


def _field_section(markdown: str, anchor: str) -> tuple[str, str] | None:
    """The ``### <label> {#anchor}`` subsection of a generated module page — the label (bold),
    its metadata line, and its help — for a field `?` popover."""
    lines = markdown.splitlines()
    marker = _anchor_marker(anchor)
    for i, line in enumerate(lines):
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading and marker.search(line):
            level = len(heading.group(1))
            label = re.sub(r"\s*\{#[^}]+\}\s*$", "", heading.group(2)).strip()
            end = i + 1
            while end < len(lines) and not re.match(rf"^#{{1,{level}}}\s", lines[end]):
                end += 1
            content = "\n".join(lines[i + 1 : end]).strip()
            return _lead_sentence(content), f"**{label}**\n\n{content}"
    return None


def extract_section(page: str, anchor: str | None) -> tuple[str, str] | None:
    """``(summary, body_markdown)`` for a topic, read from the shipped docs.

    ``summary`` is the tooltip's lead sentence; ``body_markdown`` is the element the
    in-app panel renders. ``None`` when there is no readable source (a generated page).
    """
    if page.startswith("modules/") and anchor is not None:  # generated per-module page
        markdown = _module_markdown(page[len("modules/") :])
        return _field_section(markdown, anchor) if markdown is not None else None
    text = page_text(page)
    if text is None or anchor is None:
        return None
    lines = text.splitlines()
    marker = _anchor_marker(anchor)
    for i, line in enumerate(lines):
        if marker.search(line) and re.match(r"^#{1,6}\s", line):
            # An anchored *heading* — ``### Field of view { #field-of-view }``. Its body is
            # what follows, so the backwards walk below (right for a list item, whose anchor
            # trails it) would read the section above instead. attr_list only attaches an id
            # to a heading in this inline form; the trailing-line form is inert there, which
            # is why concepts.md's terms carry it.
            return _heading_body(lines, i)
        if marker.search(
            line
        ):  # the anchored element (its own trailing {: #anchor } line)
            start = i
            while start > 0:
                if re.match(
                    r"^\s*[-*]\s", lines[start]
                ):  # a list item's own marker line
                    break
                if (
                    lines[start - 1].strip() == ""
                ):  # a blank above → a paragraph's first line
                    break
                start -= 1
            block = "\n".join(lines[start:i]).strip()
            # present the element as a clean paragraph: drop a leading list marker and
            # de-indent wrapped lines (a lone "•" reads oddly in a popover).
            body = re.sub(r"^\s*[-*]\s+", "", block)
            body = "\n".join(ln.strip() for ln in body.splitlines())
            return _lead_sentence(body), body
    for i, line in enumerate(lines):  # a heading section
        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading and _slugify(heading.group(2)) == anchor:
            return _heading_body(lines, i)
    return None


def _heading_body(lines: list[str], i: int) -> tuple[str, str]:
    """``(summary, body)`` for the heading at ``lines[i]`` — everything to the next
    heading of the same level or higher."""
    level = len(re.match(r"^(#{1,6})\s", lines[i]).group(1))
    end = i + 1
    while end < len(lines) and not re.match(rf"^#{{1,{level}}}\s", lines[end]):
        end += 1
    body = "\n".join(lines[i + 1 : end]).strip()
    return _lead_sentence(body), body
