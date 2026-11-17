"""Configuration landing page + its setup-readiness panel (§1)."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QLabel,
    QPlainTextEdit,
    QPushButton,
)

from detail.api.readiness import ReadinessSignals  # noqa: E402
from detail.core.dataclasses import CheckResult, CheckStatus  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.storage import locations  # noqa: E402


def _panel(on_navigate):
    return pages._ReadinessPanel(on_navigate)


def _signals(**kw):
    # `sam3` defaults to the AND of its two halves unless a test overrides it.
    kw.setdefault(
        "sam3", kw.get("sam3_backend", False) and kw.get("sam3_weights", False)
    )
    base = dict(
        gpu=False,
        hf_token=False,
        sam3=False,
        sam3_backend=False,
        sam3_weights=False,
        detection_model=False,
        has_experiment=False,
        quick_segment_run=False,
        distilled_model=False,
    )
    base.update(kw)
    return ReadinessSignals(**base)


def _nav_buttons(w) -> dict:
    return {b.objectName(): b for b in w.findChildren(QPushButton) if b.objectName()}


def _label_texts(w) -> list[str]:
    return [x.text() for x in w.findChildren(QLabel)]


def test_configuration_page_embeds_the_panel(qapp):
    page = pages.ConfigurationPage(lambda kind: None)
    assert page.findChild(pages._ReadinessPanel) is not None


def test_panel_computes_off_thread(qapp):
    # Until the pool thread returns (and the event loop delivers), the panel shows the
    # placeholder — construction never blocks on the readiness probe.
    assert "Checking setup" in _panel(lambda kind: None)._status.text()


def test_panel_buttons_navigate_to_their_nodes_nvidia_path(qapp):
    calls = []
    panel = _panel(calls.append)
    panel._render(_signals(gpu=True))
    buttons = _nav_buttons(panel)
    for kind in (
        "gpu_cuda",
        "sam3_backend",
        "hugging_face",
        "global_models",
        "distill_model",
        "documentation",
    ):
        assert kind in buttons, kind
        buttons[kind].click()
        assert calls[-1] == kind  # each row's button targets its own node


def test_panel_hugging_face_active_without_gpu(qapp):
    # HF is needed on every machine (DINOv3 is gated + runs on CPU/MPS), so it stays a
    # normal actionable step even with no NVIDIA GPU.
    panel = _panel(lambda kind: None)
    panel._render(_signals(gpu=False, detection_model=True))
    buttons = _nav_buttons(panel)
    assert buttons["hugging_face"].isEnabled()
    assert buttons["global_models"].isEnabled()


def test_panel_see_it_segment_opens_the_quick_segment_tool(qapp):
    # It was a dead "(coming soon)" row with a disabled button; the first-run wow is only
    # a wow if it goes somewhere, so it now navigates to the tool.
    opened: list[str] = []
    panel = _panel(opened.append)
    panel._render(_signals(gpu=True))
    buttons = _nav_buttons(panel)
    assert "quick_segment" in buttons and buttons["quick_segment"].isEnabled()
    buttons["quick_segment"].click()
    assert opened == ["quick_segment"]


def test_panel_shows_complete_when_fully_set_up(qapp):
    panel = _panel(lambda kind: None)
    panel._render(
        _signals(
            gpu=True,
            hf_token=True,
            sam3_backend=True,
            sam3_weights=True,
            detection_model=True,
        )
    )
    assert any("Setup complete" in t for t in _label_texts(panel))


def test_panel_shows_new_here_when_incomplete(qapp):
    panel = _panel(lambda kind: None)
    panel._render(_signals(gpu=True))
    assert any("New here" in t for t in _label_texts(panel))


def test_panel_shows_error_when_the_probe_fails(qapp):
    # The off-thread probe raising must surface, not leave "Checking setup…" up forever.
    panel = _panel(lambda kind: None)
    panel._on_error(RuntimeError("boom"))
    assert any("Couldn't check setup" in t and "boom" in t for t in _label_texts(panel))


def test_go_to_node_navigates_and_selects(qapp, tmp_path, monkeypatch):
    from detail.gui import pages as gui_pages
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()

    window._go_to_node("hugging_face")  # a Configuration step → its full-width sub-page
    assert isinstance(window._content.currentWidget(), gui_pages.HuggingFacePage)
    assert window._tree.isHidden()  # a drill-in sub-page, no rail

    window._go_to_node("documentation")  # → the Docs mode
    assert isinstance(window._content.currentWidget(), gui_pages.DocumentationPage)


def test_page_labels_are_selectable_after_navigation(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import Qt

    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    window = MainWindow()
    window._go_to_node("hugging_face")
    # Shown, because the policy is applied when a widget appears rather than inside `_show`.
    # It moved there so dialogs get it too — they never pass through the shell, which is why
    # ~100 labels across ~30 dialogs could not be selected (`gui/selectable.py`). A window the
    # user is looking at is always a shown window, so this is the state to assert about.
    window.show()
    labels = window._content.currentWidget().findChildren(QLabel)
    assert labels  # the page has text
    assert all(
        label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        for label in labels
    )


def test_check_readiness_button_recomputes(qapp, monkeypatch):
    # The button is the "no restart" affordance — clicking it re-runs the checks.
    monkeypatch.setattr(
        pages, "_fresh_signals", lambda: _signals()
    )  # avoid a real spawn
    panel = _panel(lambda kind: None)
    panel._render(
        _signals()
    )  # complete the initial (async) compute → button re-enabled
    button = next(
        b for b in panel.findChildren(QPushButton) if b.text() == "Check readiness"
    )
    assert button.isEnabled()
    recompute = []
    monkeypatch.setattr(panel, "_compute", lambda: recompute.append(True))
    button.click()
    assert recompute == [True]


def test_render_notifies_on_computed(qapp, monkeypatch):
    monkeypatch.setattr(pages, "_fresh_signals", lambda: _signals())
    got = []
    panel = pages._ReadinessPanel(lambda kind: None, on_computed=got.append)
    panel._render(_signals(gpu=True))
    assert got and got[-1].gpu is True  # the shell reuses the panel's compute


def test_fresh_signals_maps_the_checks(qapp, monkeypatch):
    results = [
        CheckResult("cuda", CheckStatus.OK, ""),
        CheckResult("hf_token", CheckStatus.WARN, ""),
        CheckResult("sam3_weights", CheckStatus.OK, ""),
        CheckResult("sam3_backend", CheckStatus.OK, ""),
    ]
    monkeypatch.setattr("detail.gui.pages.api_checks.run_checks", lambda **k: results)
    signals = pages._fresh_signals()
    assert signals.gpu is True and signals.sam3 is True and signals.hf_token is False


# -- SAM 3 backend page (the generated install commands) --------------------- #


def _blocks(page):
    return {t.toPlainText(): t for t in page.findChildren(QPlainTextEdit)}


def test_sam3_backend_page_renders_one_consistent_block_per_os(qapp):
    # Both OSes render the same way: exactly one read-only block each, with inline `#`
    # descriptions for every command.
    page = pages.Sam3BackendPage()
    blocks = list(_blocks(page))
    assert len(blocks) == 2  # macOS/Linux + Windows, one block each
    for body in blocks:
        assert "# Go to your detail folder" in body  # concise per-command descriptions
        assert "# Download the SAM 3 source (Meta ships it off PyPI)" in body


def test_sam3_backend_page_current_os_block_has_the_real_repo_path(qapp):
    page = pages.Sam3BackendPage()

    def has_abs_cd(body):
        return any(
            line.startswith("cd /") or line.startswith("cd C:")
            for line in body.splitlines()
        )

    current = next(b for b in _blocks(page) if has_abs_cd(b))
    clone = next(
        line
        for line in current.splitlines()
        if line.startswith("git clone https://github.com/facebookresearch/sam3 ")
    )
    assert (
        "../sam3" not in clone
    )  # an absolute clone target, not the relative docs form


def test_sam3_backend_page_other_os_block_has_triton_and_copies(qapp):
    # The other OS (Windows on a posix CI) shows the relative forms + the Windows Triton
    # line; its Copy button copies the whole annotated block.
    page = pages.Sam3BackendPage()
    body, widget = next(
        (b, t) for b, t in _blocks(page).items() if "triton-windows" in b
    )
    copy = next(
        b for b in widget.parent().findChildren(QPushButton) if b.text() == "Copy"
    )
    copy.click()
    assert QGuiApplication.clipboard().text() == body


def test_sam3_backend_page_notes_sam3_cannot_run_without_cuda(qapp):
    # SAM 3 needs an NVIDIA GPU; the page says so (but still shows the commands). Driven by
    # the cuda check: WARN → shown, OK → hidden.
    page = pages.Sam3BackendPage()
    page._render([CheckResult("cuda", CheckStatus.WARN, "")])
    assert page._cant_run.isHidden() is False  # shown when no CUDA
    assert "NVIDIA GPU" in page._cant_run.text()
    page._render([CheckResult("cuda", CheckStatus.OK, "")])
    assert page._cant_run.isHidden() is True  # a GPU box can run SAM 3 → hide the note


def test_sam3_backend_page_opens_a_terminal(qapp, monkeypatch):
    # The button opens a terminal in the repo folder (never auto-running installs). Force a
    # non-None argv so the button is enabled on any CI platform, and capture the spawn.
    monkeypatch.setattr(
        "detail.gui.pages.api_checks.terminal_launch_argv",
        lambda repo, **k: ["open", "-a", "Terminal", repo],
    )
    spawned = []
    monkeypatch.setattr(
        "detail.gui.pages.subprocess.Popen", lambda argv, *a, **k: spawned.append(argv)
    )
    page = pages.Sam3BackendPage()
    button = next(
        b for b in page.findChildren(QPushButton) if b.text().startswith("Open ")
    )
    assert button.isEnabled()
    button.click()
    assert spawned and spawned[0][:2] == ["open", "-a"]


def test_sam3_backend_page_terminal_button_greyed_when_unsupported(qapp, monkeypatch):
    # Linux (or no editable install): no reliable launch command → the button is greyed,
    # not hidden (grey-out, don't hide).
    monkeypatch.setattr(
        "detail.gui.pages.api_checks.terminal_launch_argv", lambda repo, **k: None
    )
    page = pages.Sam3BackendPage()
    button = next(
        b for b in page.findChildren(QPushButton) if b.text().startswith("Open ")
    )
    assert not button.isEnabled()


def test_sam3_backend_page_terminal_button_names_the_platform_shell(qapp, monkeypatch):
    # The label names the actual app: "Open PowerShell" on Windows, "Open Terminal" on Mac.
    monkeypatch.setattr(
        "detail.gui.pages.api_checks.terminal_name", lambda: "PowerShell"
    )
    page = pages.Sam3BackendPage()
    assert any(b.text() == "Open PowerShell" for b in page.findChildren(QPushButton))


def test_sam3_backend_page_render_flips_status(qapp):
    page = pages.Sam3BackendPage()
    page._render([CheckResult("sam3_backend", CheckStatus.OK, "")])
    assert "importable" in page._status.text().lower()
    assert "not importable" not in page._status.text().lower()
    page._render([CheckResult("sam3_backend", CheckStatus.WARN, "sam3 not importable")])
    assert "not importable" in page._status.text().lower()


def test_sam3_backend_page_recheck_reruns(qapp, monkeypatch):
    page = pages.Sam3BackendPage()
    page._render(
        [CheckResult("sam3_backend", CheckStatus.OK, "")]
    )  # re-enable the button after the async open-scan
    calls = []
    monkeypatch.setattr(page, "_scan", lambda: calls.append(True))
    button = next(
        b for b in page.findChildren(QPushButton) if b.text() == "Check readiness"
    )
    assert button.isEnabled()
    button.click()
    assert calls == [True]


def test_sam3_backend_page_notifies_shell_on_render(qapp):
    got = []
    page = pages.Sam3BackendPage(on_readiness_changed=lambda: got.append(True))
    page._render([CheckResult("sam3_backend", CheckStatus.OK, "")])
    assert got == [True]
