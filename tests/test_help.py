"""Help registry + anti-drift (PR2): every `?`-topic must resolve to a real doc anchor.

The registry (``detail/help.py``) is a pure mapping — no prose. These tests are the
guarantee the user asked for: a doc that renames or drops an anchor a topic points at
fails here, so the in-app help can never silently drift from the docs.
"""

from detail import help as help_mod
from detail.configs.describe import iter_config_models


def test_every_topic_resolves_to_a_real_anchor():
    missing = [
        f"{key} -> {topic.page}#{topic.anchor}"
        for key, topic in help_mod.HELP.items()
        if not help_mod.anchor_exists(topic.page, topic.anchor)
    ]
    assert not missing, "help topics with no matching doc anchor: " + ", ".join(missing)


def test_every_module_has_a_resolving_page():
    # Each module has its own generated modules/<id> page.
    missing = []
    for module_id, _ in iter_config_models():
        topic = help_mod.module_topic(module_id)
        if not help_mod.anchor_exists(topic.page, topic.anchor):
            missing.append(module_id)
    assert not missing, "modules with no resolving page: " + ", ".join(missing)


def test_module_topic_points_at_the_module_page():
    assert help_mod.module_topic("query") == help_mod.HelpTopic("modules/query")
    assert help_mod.module_topic("downsampling") == help_mod.HelpTopic(
        "modules/downsampling"
    )


def test_a_bogus_anchor_does_not_resolve():
    # The guard must bite — otherwise the anti-drift test above proves nothing.
    assert not help_mod.anchor_exists("concepts", "no-such-anchor")
    assert not help_mod.anchor_exists("reference/parameters", "no-such-module")


def test_concept_topics_extract_prose_from_the_docs():
    # Every concept topic (a doc page, not a generated one) yields a tooltip + panel body,
    # read from the markdown — never authored in Python.
    empty = []
    for key, topic in help_mod.HELP.items():
        if topic.page in help_mod._GENERATED_PAGES:
            continue
        section = help_mod.extract_section(topic.page, topic.anchor)
        if section is None or not section[0].strip() or not section[1].strip():
            empty.append(key)
    assert not empty, "topics with no extractable prose: " + ", ".join(empty)


def test_field_of_view_summary_reads_from_the_concept_term():
    summary, body = help_mod.extract_section("concepts", "field-of-view")
    assert "polygon" in summary.lower()
    assert (
        "**" not in summary and "`" not in summary
    )  # markdown stripped for the tooltip
    assert "bounds detection" in body  # the panel keeps the full markdown


def test_field_topic_reads_the_glossary_paragraph():
    # A field-level topic points at a terse glossary term (paragraph-anchored, not a list).
    summary, body = help_mod.extract_section("glossary", "text-prompt")
    assert "searches for" in body.lower()
    assert "prompt-capable" in body
    assert "Class name" not in body  # the extractor stops at the paragraph boundary
    assert summary.strip() and "**" not in summary  # tooltip line, markdown stripped


def test_heading_section_extracts():
    summary, body = help_mod.extract_section("getting-started", "gpu-and-cuda")
    assert summary.strip()
    assert body.strip()


def test_doc_url_targets_the_anchor():
    url = help_mod.doc_url("field-of-view")
    assert url.endswith("concepts.html#field-of-view")
    assert help_mod.doc_url("gpu").endswith("getting-started.html#gpu-and-cuda")


def test_the_help_panel_never_renders_prose_the_site_hides(tmp_path, monkeypatch):
    """`extract_section` renders the in-app panel from the **source** markdown, so prose that is
    marked-but-stripped would stay reachable there — a second door onto a hidden capability,
    which is the failure `detail/surfaces.py` exists to prevent (§25).

    No help topic currently points at a section containing a marked block, so this guards the
    *mechanism* rather than today's data: the overlap is built here deliberately, because the
    day someone marks a block inside a documented section is the day it would start leaking,
    and nothing else would notice.
    """
    from detail import help as help_mod
    from detail import surfaces

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "probe.md").write_text(
        "## Probe anchor\n\n"  # slug == the anchor; extract_section's heading branch
        "Visible sentence.\n\n"
        "<!--surface:probe_capability-->\n"
        "SECRET teaching a hidden capability.\n"
        "<!--/surface-->\n\n"
        "Trailing sentence.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(help_mod.docs_access, "docs_source_dir", lambda: tmp_path)
    # A capability declared hidden for this test alone, so the check does not depend on what
    # the shipped table happens to hide.
    monkeypatch.setitem(surfaces.HIDDEN, "probe_capability", surfaces.Hidden("test"))

    assert (
        surfaces.is_surfaced("probe_capability") is False
    )  # precondition for the strip
    summary, body = help_mod.extract_section("probe", "probe-anchor")
    assert "Visible sentence." in body  # the section still renders
    assert "SECRET" not in body  # ...without the part the site drops
    assert "SECRET" not in summary
