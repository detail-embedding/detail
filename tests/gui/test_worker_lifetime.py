"""A background result is never delivered into a widget that has been destroyed.

The class: **a queued Qt delivery whose receiver is not a QObject, so destruction cannot
sever it.** Qt disconnects a connection when the *receiver QObject* dies, which is why
``worker.done.connect(self._on_done)`` has always been safe for free. A closure has no
receiver — ``connect(lambda paths, g=generation: self._apply(g, paths))``, the ordinary
idiom for carrying a generation — so there is nothing for Qt to sever, and a result posted
before the page was torn down runs against deleted widgets:

    File "detail/gui/pages.py", line 1657, in <lambda>
      lambda paths, g=generation: self._apply_previews(g, paths)
    File "detail/gui/pages.py", line 1669, in _apply_previews
      card.set_preview(path)
    File "detail/gui/pages.py", line 1527, in set_preview
      self._thumb.setPixmap(
    RuntimeError: wrapped C/C++ object of type QLabel has been deleted

Opening **Home** decodes one preview still per experiment off the UI thread. Navigate away
before it lands — ordinary, and a certainty on a workspace slow enough that the stills take
seconds — and the app shows "Something went wrong" for having changed page. The window
survives only because the shell's excepthook reports rather than dies.

**Why the guard that was already there did not hold.** ``_apply_previews`` opened with a
generation check whose own comment claimed it covered "a refresh (or closing the page)". It
covered the refresh. A generation expresses **supersession** — a later request owns the view
now — and destruction is **liveness**; nothing bumps a counter on the way out, because the
shell tears a page down with ``deleteLater`` and no page code runs at all. The two guards
answer different questions and neither subsumes the other, so both are kept: the generation
covers a refresh replacing the cards, and ``receiver=self`` on the worker covers the page
going away. The comment that claimed one of them did both is now two comments that are true.

**Why it is fixed on the base class rather than at the two sites that crashed.** The idiom
is not rare — it was live in ``HomePage._load_previews`` and, identically, in
``analysis_clip_annotation._load_continuity``, and ten modules construct these threads. So
:class:`~detail.gui.workers.GuiWorker` owns it: ``receiver`` is a required keyword, every
emission goes out through ``emit_to_ui``, and the payload takes one hop through the worker
(which lives on the UI thread) where it is dropped if the receiver's C++ side has gone.
``busy.py`` had hand-rolled the same check for its own two entry points; it now names its
target as the receiver instead, so there is one mechanism rather than two.

The guards below are in three layers, because no one of them is enough:

- the mechanism, behaviourally, in both directions — a dead receiver drops, a live one
  delivers (a gate that dropped everything would pass the first test alone);
- the reported bug end to end, through the real page, torn down the way the shell tears it
  down. **The trap here is real and cost the reporter an afternoon:** ``processEvents()``
  does *not* deliver ``DeferredDelete``, so a test that only pumps leaves the page alive,
  the delivery succeeds, and the test passes against the unfixed code. Hence the explicit
  ``sip.isdeleted`` assertions before anything is pumped;
- two source scans, so a site added tomorrow is covered rather than remembered — every
  worker in ``detail/gui/`` descends from the base, and every construction names a receiver.
  The scans are what covered the ``analysis_clip_annotation`` sibling (since removed) and
  cover every future one; the behavioural pair above is what makes them mean something.
"""

import ast
import threading
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6 import sip  # noqa: E402
from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from detail.api.experiment import Experiment  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.gui.workers import BackgroundCall  # noqa: E402
from detail.storage import locations  # noqa: E402
from tests.gui.conftest import drain_busy_calls  # noqa: E402
from tests import waits  # noqa: E402

_GUI = Path(__file__).resolve().parents[2] / "detail" / "gui"

#: The one class allowed to inherit ``QThread`` directly — it is the base every other
#: worker gets the receiver gate from.
_THE_BASE = "GuiWorker"

#: Constructions allowed to pass ``receiver=None``, each with the reason it reports to
#: nothing that can be destroyed. Empty: every worker in the GUI belongs to a widget.
#: ``None`` is legal in tests, which drive ``run()`` directly and own no widget.
_NULL_RECEIVER_OK: dict[str, str] = {}


# --------------------------------------------------------------------------- #
# The mechanism
# --------------------------------------------------------------------------- #
def test_a_result_delivered_after_its_receiver_is_destroyed_is_dropped(qapp):
    """The whole class in six lines: a closure slot, a destroyed receiver, a live result."""
    host = QWidget()
    release = threading.Event()
    ran = threading.Event()
    delivered = []

    def work():
        release.wait(5)
        ran.set()
        return "payload"

    call = BackgroundCall(work, receiver=host)
    # A closure, not a bound method: Qt has no receiver QObject here to disconnect on
    # destruction, which is the entire reason this class exists.
    call.done.connect(lambda result: delivered.append(result))
    call.start()

    sip.delete(host)  # the page swap, synchronously
    release.set()
    waits.finished(call, "the worker")
    drain_busy_calls(qapp)

    assert ran.is_set(), "the work never ran — the test proved nothing about delivery"
    assert delivered == [], "a result was handed to a slot holding a destroyed widget"


def test_a_result_is_still_delivered_while_its_receiver_is_alive(qapp):
    """The other direction. Without this, a gate that dropped every result would pass."""
    host = QWidget()
    delivered = []
    call = BackgroundCall(lambda: "payload", receiver=host)
    call.done.connect(lambda result: delivered.append(result))
    call.start()
    waits.finished(call, "the worker")
    drain_busy_calls(qapp)
    assert delivered == ["payload"]


def test_a_failure_delivered_after_its_receiver_is_destroyed_is_dropped(qapp):
    """``failed`` is a delivery too — and the handler for one is the surface most likely to
    put up a message box, which is worse than a swallowed traceback."""
    host = QWidget()
    release = threading.Event()
    seen = []

    def boom():
        release.wait(5)
        raise ValueError("nope")

    call = BackgroundCall(boom, receiver=host)
    call.failed.connect(lambda exc: seen.append(exc))
    call.start()
    sip.delete(host)
    release.set()
    waits.finished(call, "the worker")
    drain_busy_calls(qapp)
    assert seen == []


# --------------------------------------------------------------------------- #
# The reported bug, through the real page
# --------------------------------------------------------------------------- #
def test_navigating_away_from_home_mid_preview_touches_no_deleted_widget(
    qapp, tmp_path, monkeypatch
):
    """Open Home, leave before the stills arrive. The reported crash, end to end.

    The still is a real PNG so the delivery would reach ``set_preview``'s ``setPixmap``
    branch — the line in the traceback — rather than the ``setText`` one a ``None`` takes.
    """
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    still = tmp_path / "still.png"
    QPixmap(160, 96).save(str(still))
    Experiment.create("alpha", tmp_path)
    Experiment.create("beta", tmp_path)

    release = threading.Event()
    decoded = threading.Event()

    def slow_preview(self):
        release.wait(5)
        decoded.set()
        return still

    monkeypatch.setattr(Experiment, "preview_image", slow_preview)

    escaped: list = []
    monkeypatch.setattr("sys.excepthook", lambda *info: escaped.append(info))

    page = pages.HomePage(on_open=lambda *_: None)
    card = page._cards[0]
    worker = page._preview_worker

    # Exactly what `MainWindow._discard_content` does on every navigation.
    page.deleteLater()
    # `processEvents()` does NOT deliver DeferredDelete. Without this the page is never
    # destroyed, the delivery succeeds, and this test passes against the unfixed code.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert sip.isdeleted(page), "the page was not destroyed — this test proves nothing"
    assert sip.isdeleted(card), "the cards outlived the page they belong to"

    release.set()
    waits.finished(worker, "the worker")
    drain_busy_calls(qapp)

    assert decoded.is_set(), "the preview never finished — nothing was ever delivered"
    assert not escaped, (
        "a preview landed on the destroyed page: "
        f"{escaped[0][0].__name__}: {escaped[0][1]}"
    )


def test_home_still_shows_its_previews_when_nobody_navigates_away(
    qapp, tmp_path, monkeypatch
):
    """The positive control for the page above: the stills do arrive on a page that stays."""
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    still = tmp_path / "still.png"
    QPixmap(160, 96).save(str(still))
    Experiment.create("alpha", tmp_path)
    monkeypatch.setattr(Experiment, "preview_image", lambda self: still)

    page = pages.HomePage(on_open=lambda *_: None)
    waits.finished(page._preview_worker, "the worker")
    drain_busy_calls(qapp)

    thumb = page._cards[0]._thumb
    assert thumb.pixmap() is not None and not thumb.pixmap().isNull()


def test_a_refresh_drops_the_previews_computed_for_the_cards_it_replaced(
    qapp, tmp_path, monkeypatch
):
    """The other half, and the reason the generation check stays.

    Here the page is perfectly alive and the *cards* are the thing that went away, so
    liveness cannot see it — supersession and destruction are different questions. The
    stale result is a one-element list computed for the card that no longer exists; applied
    to the new card set it would paint one experiment's still onto another's row.
    """
    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    still = tmp_path / "still.png"
    QPixmap(160, 96).save(str(still))
    Experiment.create("alpha", tmp_path)

    release = threading.Event()
    monkeypatch.setattr(
        Experiment, "preview_image", lambda self: (release.wait(5), still)[1]
    )
    page = pages.HomePage(on_open=lambda *_: None)
    stale_worker = page._preview_worker

    # A second experiment appears and the page refreshes. The fresh read finds no stills,
    # so anything a card ends up showing came from the read for the *previous* card set.
    Experiment.create("beta", tmp_path)
    monkeypatch.setattr(Experiment, "preview_image", lambda self: None)
    page._refresh()
    waits.finished(page._preview_worker, "the worker")
    drain_busy_calls(qapp)

    release.set()  # only now does the read for the old, single-card page come back
    waits.finished(stale_worker, "the worker")
    drain_busy_calls(qapp)

    assert len(page._cards) == 2
    for card in page._cards:
        assert card._thumb.pixmap().isNull(), (
            "a still computed for the card set a refresh replaced was painted onto the "
            "new one"
        )


# --------------------------------------------------------------------------- #
# The scans: every worker, every construction
# --------------------------------------------------------------------------- #
def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def raw_qthread_subclasses(source: str) -> list[str]:
    """Classes inheriting ``QThread`` directly — i.e. bypassing the receiver gate."""
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
        and "QThread" in _base_names(node)
        and node.name != _THE_BASE
    ]


def worker_class_names(source: str) -> set[str]:
    """Worker classes defined in ``source`` — anything descending from the base."""
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
        and ({"QThread", _THE_BASE} & _base_names(node))
        and node.name != _THE_BASE
    }


def constructions_without_a_receiver(source: str, workers: set[str]) -> list[str]:
    """``<Worker>(...)`` calls that name no receiver, or name ``None`` as one."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in workers:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if "receiver" not in kw:
            bad.append(f"{node.func.id} (line {node.lineno}): no receiver=")
        elif isinstance(kw["receiver"], ast.Constant) and kw["receiver"].value is None:
            bad.append(f"{node.func.id} (line {node.lineno}): receiver=None")
    return bad


def _gui_sources() -> dict[str, str]:
    return {
        str(p.relative_to(_GUI.parent.parent)).replace("\\", "/"): p.read_text(
            encoding="utf-8-sig"
        )
        for p in sorted(_GUI.rglob("*.py"))
    }


def test_every_gui_thread_descends_from_the_one_that_checks_its_receiver():
    """A worker written against ``QThread`` directly inherits none of this, and nothing
    fails when it is written — the crash arrives later, on somebody else's navigation.
    """
    offenders = {
        rel: names
        for rel, source in _gui_sources().items()
        if (names := raw_qthread_subclasses(source))
    }
    assert not offenders, (
        f"these threads bypass {_THE_BASE} and cannot drop a delivery into a destroyed "
        f"widget: {offenders}"
    )


def test_every_worker_construction_names_the_widget_it_reports_to():
    sources = _gui_sources()
    workers = set().union(*(worker_class_names(s) for s in sources.values()))
    assert (
        "BackgroundCall" in workers
    ), "the scan found no workers — it is scanning nothing"

    offenders = {}
    for rel, source in sources.items():
        found = [
            bad
            for bad in constructions_without_a_receiver(source, workers)
            if f"{rel}:{bad.split(' ')[0]}" not in _NULL_RECEIVER_OK
        ]
        if found:
            offenders[rel] = found
    assert not offenders, (
        "a worker with no receiver delivers its result whatever happened to the page that "
        f"started it: {offenders}"
    )


def test_the_scans_would_catch_one():
    """Guard the guards: a scan that matches nothing today passes vacuously forever."""
    assert raw_qthread_subclasses("class Sneaky(QThread):\n    pass\n") == ["Sneaky"]
    assert raw_qthread_subclasses(f"class Ok({_THE_BASE}):\n    pass\n") == []

    workers = worker_class_names(f"class Ok({_THE_BASE}):\n    pass\n")
    assert workers == {"Ok"}
    assert constructions_without_a_receiver("x = Ok(fn)", workers) == [
        "Ok (line 1): no receiver="
    ]
    assert constructions_without_a_receiver("x = Ok(fn, receiver=None)", workers) == [
        "Ok (line 1): receiver=None"
    ]
    assert constructions_without_a_receiver("x = Ok(fn, receiver=self)", workers) == []
