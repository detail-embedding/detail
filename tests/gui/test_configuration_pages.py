"""GPU / CUDA + Hugging Face Configuration pages (§1)."""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtGui import QGuiApplication  # noqa: E402
from PyQt6.QtWidgets import QLabel, QPushButton  # noqa: E402

from detail.gui import cuda_doctor, pages  # noqa: E402
from detail.gui.cuda_doctor import ProbeResult  # noqa: E402
from detail.gui.workers import BackgroundCall  # noqa: E402


class _SyncBackgroundCall(BackgroundCall):
    """Run the callable inline so done/failed fire synchronously in a test."""

    def start(self):  # type: ignore[override]
        self.run()


def _probe(**kw):
    base = dict(
        cuda=False,
        mps=False,
        cuda_devices=[],
        torch_installed=True,
        torch_cuda_build=None,
        driver_version=None,
        driver_cuda=None,
    )
    base.update(kw)
    return ProbeResult(**base)


def test_gpu_page_reports_cuda(qapp):
    page = pages.GpuCudaPage()
    page._render(
        _probe(
            cuda=True,
            cuda_devices=["RTX 4070"],
            torch_cuda_build="12.8",
            driver_version="570.0",
            driver_cuda="12.8",
        )
    )
    assert "CUDA working" in page._status.text()
    assert "RTX 4070" in page._status.text()  # device name surfaced
    assert page._doctor.isEnabled()  # NVIDIA present → doctor panel active


def test_gpu_page_greys_doctor_without_nvidia(qapp):
    page = pages.GpuCudaPage()
    page._render(_probe(mps=True))
    assert page._status.text() == "Apple GPU (MPS) detected"
    assert not page._doctor.isEnabled()  # greyed, not hidden
    page._render(_probe())  # no GPU at all
    assert "No NVIDIA GPU" in page._status.text()
    assert not page._doctor.isEnabled()


def test_gpu_page_copy_button_puts_fix_command_on_clipboard(qapp):
    page = pages.GpuCudaPage()
    result = _probe(driver_version="560.0", driver_cuda="12.6", torch_cuda_build=None)
    page._render(result)
    command = cuda_doctor.diagnose(result).fix_command
    assert command and "cu126" in command
    copy = next(b for b in page.findChildren(QPushButton) if b.text() == "Copy")
    copy.click()
    assert QGuiApplication.clipboard().text() == command


def _doctor_labels(page) -> list[str]:
    return [label.text() for label in page._doctor.findChildren(QLabel)]


def test_gpu_page_shows_why_this_build_and_to_restart_after_the_fix(qapp):
    # The fix command reinstalls torch under an app that already imported it, so the
    # page has to say a restart is needed, and show how it chose the tag (#506).
    page = pages.GpuCudaPage()
    result = _probe(driver_version="595.95", driver_cuda="13.2", torch_cuda_build=None)
    page._render(result)
    labels = _doctor_labels(page)
    assert cuda_doctor.diagnose(result).fix_reason in labels
    assert cuda_doctor.RESTART_AFTER_FIX in labels


def test_gpu_page_says_nothing_about_restarting_when_there_is_nothing_to_install(qapp):
    page = pages.GpuCudaPage()
    page._render(
        _probe(
            cuda=True,
            cuda_devices=["RTX 4070"],
            torch_cuda_build="12.8",
            driver_version="570.0",
            driver_cuda="12.8",
        )
    )
    assert cuda_doctor.RESTART_AFTER_FIX not in _doctor_labels(page)


def test_gpu_page_shows_error_when_the_probe_fails(qapp):
    # The off-thread probe raising must surface on the detected line, not leave
    # "Detecting accelerator…" up forever, and the doctor greys out (nothing to act on).
    page = pages.GpuCudaPage()
    page._on_error(RuntimeError("nvidia-smi exploded"))
    assert "Couldn't detect the accelerator" in page._status.text()
    assert "nvidia-smi exploded" in page._status.text()
    assert not page._doctor.isEnabled()


def test_call_task_emits_failed_on_exception_and_done_on_success(qapp):
    # The off-thread primitive must report a raising callable via `failed` (so a
    # consumer can render an error) and a normal one via `done` — never swallow either.
    got = {}
    signals = pages._CallSignals()
    signals.done.connect(lambda r: got.__setitem__("done", r))
    signals.failed.connect(lambda e: got.__setitem__("failed", e))

    pages._CallTask(
        lambda: 1 // 0, signals
    ).run()  # same-thread emit fires the slot now
    assert isinstance(got.get("failed"), ZeroDivisionError) and "done" not in got

    got.clear()
    pages._CallTask(lambda: 42, signals).run()
    assert got.get("done") == 42 and "failed" not in got


def test_gpu_page_keeps_fix_actionable_when_torch_missing(qapp):
    # No NVIDIA + no torch → not "CUDA-compatible", but the install fix must stay
    # copyable (the panel isn't greyed away).
    page = pages.GpuCudaPage()
    page._render(_probe(torch_installed=False))
    assert page._doctor.isEnabled()
    assert any(b.text() == "Copy" for b in page.findChildren(QPushButton))


def test_gpu_page_diagnostic_labels_are_selectable(qapp):
    from PyQt6.QtCore import Qt

    page = pages.GpuCudaPage()
    page._render(_probe(cuda=True, cuda_devices=["RTX 4070"], torch_cuda_build="12.8"))
    labels = page._doctor.findChildren(QLabel)
    assert labels and all(
        label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        for label in labels
    )


def test_hf_page_shows_current_token_state(qapp, monkeypatch):
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: "hf_x")
    assert "saved" in pages.HuggingFacePage()._status.text()
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    assert "No token" in pages.HuggingFacePage()._status.text()


def test_hf_page_validate_saves_and_reports_account(qapp, monkeypatch):
    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    saved = {}

    def _fake_set(token):
        saved["token"] = token
        return "octocat"

    monkeypatch.setattr("detail.gui.pages.credentials.set_hf_token", _fake_set)
    page = pages.HuggingFacePage()
    page._token.setText("hf_secret")
    page._save_token()
    assert saved["token"] == "hf_secret"  # the entered token was validated + saved
    assert "octocat" in page._status.text()


def test_hf_page_validate_surfaces_error(qapp, monkeypatch):
    monkeypatch.setattr(pages, "BackgroundCall", _SyncBackgroundCall)
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)

    def _bad(token):
        raise ValueError("Hugging Face token is invalid or could not be validated")

    monkeypatch.setattr("detail.gui.pages.credentials.set_hf_token", _bad)
    page = pages.HuggingFacePage()
    page._token.setText("bad-token")
    page._save_token()
    assert "invalid" in page._status.text()  # the error is surfaced, not swallowed


def test_hf_page_remove_token_clears_and_notifies(qapp, monkeypatch):
    token = {"v": "hf_x"}
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: token["v"])
    cleared, changed = [], []
    monkeypatch.setattr(
        "detail.gui.pages.credentials.clear_hf_token",
        lambda: (cleared.append(True), token.update(v=None)),
    )
    page = pages.HuggingFacePage(on_readiness_changed=lambda: changed.append(True))
    assert page._remove_btn.isEnabled()  # a saved token → Remove available
    page._remove_token()
    assert cleared == [True] and changed == [True]  # cleared + shell notified
    assert "No token" in page._status.text()


def test_hf_page_lists_external_links_and_gated_models(qapp, monkeypatch):
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    page = pages.HuggingFacePage()
    labels = [b.text() for b in page.findChildren(QPushButton)]
    assert "Create account ↗" in labels  # ↗ marks an external link
    assert "Create token ↗" in labels
    assert "SAM 3 ↗" in labels  # friendly name + external marker
    assert any("DINOv3" in text and "↗" in text for text in labels)  # DINO now shown


def test_hf_status_is_top_bold_and_larger(qapp, monkeypatch):
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    page = pages.HuggingFacePage()
    assert page._status.font().bold()
    assert page._status.font().pointSize() == QLabel().font().pointSize() + 2


def test_hf_page_validate_rejects_empty_token(qapp, monkeypatch):
    # An empty field warns and never calls set_hf_token.
    monkeypatch.setattr("detail.gui.pages.credentials.get_hf_token", lambda: None)
    called = []
    monkeypatch.setattr(
        "detail.gui.pages.credentials.set_hf_token",
        lambda token: called.append(token),
    )
    warned = []
    monkeypatch.setattr(
        "detail.gui.pages.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    page = pages.HuggingFacePage()
    page._token.setText("   ")
    page._save_token()
    assert warned and not called
