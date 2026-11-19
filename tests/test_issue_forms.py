"""The issue forms GitHub shows a reporter, and the body a terminal user files instead.

**The class: two descriptions of one report drifting apart.** A reporter fills in
``.github/ISSUE_TEMPLATE/bug_report.yml`` in the browser. An agent or a contributor in a
terminal files ``.github/bug_report_body.md`` with ``gh issue create --body-file``, because
``gh`` cannot fill in an issue form. GitHub writes a submitted form as one ``### <label>``
section per field, so the body's headings must be the form's labels in the same order, or the
two kinds of report stop reading alike and a triager has to work out which one they have.

The form also names a command a reporter must run. A form that asks for the output of a
command the CLI no longer has is a bug report nobody can complete.
"""

import pathlib
import re

import yaml

from detail.cli import main

REPO = pathlib.Path(__file__).resolve().parents[1]
FORMS = REPO / ".github" / "ISSUE_TEMPLATE"
BODY = REPO / ".github" / "bug_report_body.md"


def _load(name: str) -> dict:
    return yaml.safe_load((FORMS / name).read_text(encoding="utf-8-sig"))


def _field_labels(form: dict) -> list[str]:
    return [
        element["attributes"]["label"]
        for element in form["body"]
        if element["type"] != "markdown"
    ]


def _body_headings(text: str) -> list[str]:
    return re.findall(r"^### (.+)$", text, re.MULTILINE)


def test_every_template_is_an_issue_form_github_will_show_and_enforce():
    # A Markdown template enforces nothing, and a form missing `name`, `description` or `body`
    # is left out of the chooser without an error anyone sees.
    templates = sorted(
        path.name for path in FORMS.iterdir() if path.name != "config.yml"
    )
    assert templates == ["bug_report.yml", "docs_problem.yml", "feature_request.yml"]
    for name in templates:
        form = _load(name)
        assert {"name", "description", "body"} <= form.keys(), name
        ids = [
            element["id"] for element in form["body"] if element["type"] != "markdown"
        ]
        assert len(ids) == len(set(ids)), name
        required = [
            element
            for element in form["body"]
            if element.get("validations", {}).get("required")
        ]
        assert required, f"{name} requires nothing"


def test_blank_issues_are_off_and_questions_and_security_go_elsewhere():
    config = _load("config.yml")
    assert config["blank_issues_enabled"] is False
    urls = [link["url"] for link in config["contact_links"]]
    assert any(url.endswith("/discussions") for url in urls)
    assert any(url.endswith("/security/advisories/new") for url in urls)


def test_the_terminal_body_has_the_bug_forms_sections_in_order():
    labels = _field_labels(_load("bug_report.yml"))
    assert _body_headings(BODY.read_text(encoding="utf-8-sig")) == labels


def test_the_drift_check_sees_a_dropped_or_reordered_section():
    labels = _field_labels(_load("bug_report.yml"))
    text = BODY.read_text(encoding="utf-8-sig")
    assert _body_headings(text.replace("### Workaround\n", "")) != labels
    swapped = (
        text.replace("### Terminal output", "### SWAP")
        .replace("### Run-folder logs", "### Terminal output")
        .replace("### SWAP", "### Run-folder logs")
    )
    assert _body_headings(swapped) != labels


def test_the_command_the_bug_form_requires_exists():
    form = _load("bug_report.yml")
    doctor = next(element for element in form["body"] if element.get("id") == "doctor")
    assert doctor["validations"]["required"] is True
    assert "`detail doctor`" in doctor["attributes"]["description"]
    assert "doctor" in main.commands
