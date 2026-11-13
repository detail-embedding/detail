"""Fail-closed guard: only ``gui/open_external.py`` asks the OS to open a URL.

**The class.** *An opener whose failure is a return value, called by code that discards it.*
``QDesktopServices.openUrl`` answers ``False`` when no handler is registered and raises
nothing; Qt's only trace is a ``qt.qpa.services`` line in a terminal the user may not be
watching. Inside WSL that is the normal answer, since a distro has no ``xdg-open`` browser
or file manager. A click on **Download Ollama** there did nothing visible.

**How it hid.** The file-path half was fixed first (#475): ``pages._open_local_path`` checked
the bool and fell back to Windows Explorer. The six web-link sites (the Hugging Face links,
the two docs opens, the help popover's two, and Download Ollama) were judged out of scope,
recorded as *"the destination is not a filesystem path that can independently go missing"*.
True, and beside the point: the *opener* goes missing, and on WSL it always does. Every site
looked right at its own line; a behavioural test passes on any machine with a browser, which
is every machine a test runs on.

**The rule.** Open a URL through ``open_external.open_url`` (a web page) or
``open_external.open_local_path`` (a folder or file). They check the bool, fall back to
``explorer.exe`` inside WSL, and otherwise say what could not be opened. This scan fails on
any ``.openUrl`` reference, and any ``webbrowser`` import (the same silent bool), elsewhere in
``detail/``. Read from the parse tree, so a docstring that names the call is not a hit and a
call wrapped over two lines is.
"""

import ast
import pathlib
import textwrap

import detail

_PACKAGE = pathlib.Path(detail.__file__).parent
#: The one module allowed to call the openers it wraps.
_HELPER = _PACKAGE / "gui" / "open_external.py"


def _openers(source: str) -> list[int]:
    """Line numbers in ``source`` that reach an OS URL opener directly."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == "openUrl":
            hits.append(node.lineno)
        elif isinstance(node, ast.Import) and any(
            a.name.split(".")[0] == "webbrowser" for a in node.names
        ):
            hits.append(node.lineno)
        elif (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "webbrowser"
        ):
            hits.append(node.lineno)
    return hits


def _offenders(root: pathlib.Path) -> list[str]:
    found = []
    for py in sorted(root.rglob("*.py")):
        if py == _HELPER:
            continue
        for line in _openers(py.read_text(encoding="utf-8-sig")):
            found.append(f"{py.relative_to(root).as_posix()}:{line}")
    return found


def test_only_the_helper_opens_a_url():
    offenders = _offenders(_PACKAGE)
    assert not offenders, (
        f"a URL opened outside gui/open_external.py: {offenders}. openUrl answers False "
        "with no browser or file manager (always, inside WSL), and a caller that drops it "
        "makes the click do nothing. Use open_external.open_url / open_local_path."
    )


def test_the_helper_is_still_where_the_scan_exempts():
    """An exemption for a file that moved would exempt nothing and hide the real opener."""
    assert _openers(_HELPER.read_text(encoding="utf-8-sig")), _HELPER


def test_the_scan_would_catch_the_spellings_a_person_writes():
    for source in (
        "QDesktopServices.openUrl(QUrl(url))",
        "button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(URL)))",
        "ok = QtGui.QDesktopServices.openUrl(\n    QUrl(url)\n)",
        "opener = QDesktopServices.openUrl\nopener(QUrl(url))",
        "import webbrowser\nwebbrowser.open(url)",
        "from webbrowser import open as browse",
    ):
        assert _openers(textwrap.dedent(source)), source


def test_the_scan_ignores_prose_and_the_helper_call():
    for source in (
        '"""QDesktopServices.openUrl returns a bool."""',
        "# QDesktopServices.openUrl(QUrl(url))",
        "open_external.open_url(self, url)",
    ):
        assert not _openers(source), source
