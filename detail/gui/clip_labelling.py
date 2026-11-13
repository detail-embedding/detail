"""The clip window's labelling panel (§22 *Clip checks*): name the cluster, and say of each clip
whether it fits — the two things a person does while watching a cluster's clips.

**The opener owns the run.** The panel is handed a :class:`ClusterLabelling` by the page that
opened the window, and every write goes through it: a label through the clustering page's own
labelling (its queue, its tree, its undo-free order of writes), a check through the run's
``check_member``. The panel reads nothing from disk on the UI thread — the checks so far arrive
through a ``BackgroundCall`` — and after an answer it recomputes the figure from the answers it
holds (``member_checks.summarize``) rather than reading the file again.

**What the figure is.** ``fits / (fits + doesn't)`` over the clips looked at, with its 95% Wilson
interval, and said to be the clips looked at, most typical first: a check on the cluster's core,
not an unbiased precision (the §19 audit draws for that, once the cluster is labelled).

**One row, under the picture.** The window exists to *see* the behaviour, so labelling takes
one line below the strips and the picture keeps the width: the label box and its button, the
three answers, and the figure in brief (:func:`brief`) with the whole sentence (:func:`describe`)
on hover. It was a column beside the player, which took a third of the window's width from
the footage to hold a few buttons.
"""

from __future__ import annotations

from typing import Protocol

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from detail.core.utils import subject_key
from detail.gui.sizing import text_width

from detail.storage import member_checks
from detail.storage.member_checks import DOESNT, FITS, UNCLEAR, ClipKey

#: The three answers: button text, the key that gives it, its verdict.
ANSWERS = (("Fits", "Y", FITS), ("Doesn't", "N", DOESNT), ("Unclear", "U", UNCLEAR))


class ClusterLabelling(Protocol):
    """What the clip window needs of the page that opened it to label one cluster."""

    def title(self) -> str: ...  # "cluster #42"

    def current_label(self) -> str: ...

    def label_choices(self) -> list[str]: ...

    def edit_reason(self) -> str: ...  # "" where it may be changed

    def apply_label(self, name: str) -> None: ...

    def check(self, key: ClipKey, verdict: str) -> None: ...

    def node(self) -> int: ...

    def load_checks(self) -> member_checks.CheckSummary: ...  # called off the UI thread

    # -- the cluster sections (``gui/clip_cluster_info.py``) -------------------- #
    def experiment(self): ...

    def fingerprint(
        self, clip, progress
    ): ...  # off the UI thread; api.clip_fingerprint

    def clip_measures(self, clip, names: list[str]) -> dict: ...  # off the UI thread

    def sibling(self) -> tuple[str, list] | None: ...  # (title, its most typical clips)

    def member_clips(self, n: int) -> list: ...  # this cluster's most typical clips

    # Optional: the clip window offers *Cluster profile…* only when a page provides it.
    def open_profile(self, parent) -> None: ...  # the Cluster profile window


def clip_key(node: int, clip) -> ClipKey:
    """The check key of a ``ClipSpec`` in ``node``."""
    return ClipKey(
        int(node),
        str(clip.recording_id),
        subject_key(clip.track_id),
        int(clip.start_t_ns),
        int(clip.end_t_ns),
    )


class ClipLabelPanel(QWidget):
    """Label the cluster; answer for the clip on screen. ``answered`` fires after an answer is
    given, so the window can move on to the next clip."""

    answered = pyqtSignal(str)

    def __init__(self, labelling: ClusterLabelling, parent=None) -> None:
        super().__init__(parent)
        self._labelling = labelling
        self._clip = None
        self._answers: dict[ClipKey, str] = {}
        self._loaded = False
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(QLabel(f"<b>Label {labelling.title()}</b>"))
        self._name = QComboBox()
        self._name.setObjectName("clip_label_name")
        self._name.setEditable(True)
        self._name.addItems(labelling.label_choices())
        self._name.setCurrentText(labelling.current_label())
        self._name.setMinimumWidth(text_width("a label of some length"))
        self._name.setToolTip(
            "The cluster's label: pick one you have used, or type a new one."
        )
        row.addWidget(self._name)
        self._apply = QPushButton("Label cluster")
        self._apply.setObjectName("clip_label_apply")
        self._apply.setToolTip(
            "Give the cluster the label in the box, as the tree does."
        )
        self._apply.clicked.connect(self._apply_label)
        row.addWidget(self._apply)
        self._label_state = QLabel("")
        self._label_state.setObjectName("clip_label_state")
        row.addWidget(self._label_state)

        row.addSpacing(text_width("MM"))
        row.addWidget(QLabel("<b>Does this clip fit?</b>"))
        self._buttons: dict[str, QPushButton] = {}
        for text, key, verdict in ANSWERS:
            button = QPushButton(f"{text} ({key})")
            button.setObjectName(f"clip_check_{verdict}")
            button.setCheckable(True)
            button.setToolTip(
                f"Record that this clip {_verb(verdict)} the cluster, and go to the next "
                f"clip. Key: {key}."
            )
            button.clicked.connect(lambda _on, v=verdict: self.answer(v))
            row.addWidget(button)
            self._buttons[verdict] = button
        self._figure = QLabel("Reading the checks so far…")
        self._figure.setObjectName("clip_check_figure")
        # The brief figure takes what the row has left and never widens the window for it:
        # the whole sentence is on hover.
        self._figure.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self._figure, 1)
        self._sync_editable()
        self._load()

    # -- inputs -------------------------------------------------------------- #
    def set_clip(self, clip) -> None:
        """The clip on screen, whose answer the buttons show."""
        self._clip = clip
        self._show_answer()

    def answer(self, verdict: str) -> None:
        """Record ``verdict`` for the clip on screen (a button or its key)."""
        if self._clip is None or self._labelling.edit_reason():
            self._show_answer()
            return
        key = clip_key(self._labelling.node(), self._clip)
        self._labelling.check(key, verdict)
        self._answers[key] = verdict
        self._show_answer()
        self._show_figure()
        self.answered.emit(verdict)

    def summary(self) -> member_checks.CheckSummary:
        return member_checks.summarize(dict(self._answers))

    # -- the panel's own state ------------------------------------------------ #
    def _apply_label(self) -> None:
        name = self._name.currentText().strip()
        self._labelling.apply_label(name)
        self._label_state.setText("Labelled." if name else "Label removed.")
        self._label_state.setToolTip(
            f"Labelled “{name}”." if name else "Label removed from the cluster."
        )

    def _sync_editable(self) -> None:
        reason = self._labelling.edit_reason()
        if not reason:
            return
        for control in (self._name, self._apply, *self._buttons.values()):
            control.setEnabled(False)
            control.setToolTip(reason)
        self._label_state.setText("Read only.")
        self._label_state.setToolTip(reason)

    def _load(self) -> None:
        from detail.gui.workers import BackgroundCall

        call = BackgroundCall(self._labelling.load_checks, receiver=self)  # off-thread
        call.done.connect(self._loaded_checks)
        call.failed.connect(
            lambda exc: self._figure.setText(f"Could not read the checks: {exc}")
        )
        call.start()

    def _loaded_checks(self, summary) -> None:
        # Answers given while the read was in flight are newer than the file's.
        self._answers = {**summary.answers, **self._answers}
        self._loaded = True
        self._show_answer()
        self._show_figure()

    def _show_answer(self) -> None:
        current = (
            self._answers.get(clip_key(self._labelling.node(), self._clip))
            if self._clip is not None
            else None
        )
        for verdict, button in self._buttons.items():
            button.setChecked(verdict == current)

    def _show_figure(self) -> None:
        summary = self.summary()
        self._figure.setText(brief(summary))
        self._figure.setToolTip(describe(summary))


def brief(summary: member_checks.CheckSummary) -> str:
    """The figure in a few words, for the labelling row: the counts and the estimate."""
    if summary.checked == 0:
        return "No clips checked yet."
    counts = (
        f"Checked {summary.checked}: {summary.fits} fit, {summary.doesnt} don't, "
        f"{summary.unclear} unclear"
    )
    if summary.estimate is None:
        return counts + "."
    low, high = summary.interval
    return f"{counts} · about {summary.estimate:.0%} fit (95%: {low:.0%}–{high:.0%})."


def describe(summary: member_checks.CheckSummary) -> str:
    """The figure, in words: what was checked, the estimate with its interval, and what it is
    an estimate of."""
    if summary.checked == 0:
        return "No clips checked yet. Answer for a few to see how many fit."
    counts = (
        f"Checked {summary.checked}: {summary.fits} fit, {summary.doesnt} don't, "
        f"{summary.unclear} unclear."
    )
    if summary.estimate is None:
        return counts
    low, high = summary.interval
    return (
        f"{counts} About {summary.estimate:.0%} fit (95% interval {low:.0%}–{high:.0%}), "
        "over the clips you looked at, most typical first: the cluster's core, not every "
        "clip in it."
    )


def _verb(verdict: str) -> str:
    return {FITS: "fits", DOESNT: "does not fit", UNCLEAR: "is unclear for"}[verdict]


class PageLabelling:
    """INTERPRET's door into the clip window for one cluster (a :class:`ClusterLabelling`),
    opened from the page itself or from one of its cluster windows (``origin``).

    A label goes through the page's own ``_label`` — its queue, its undo history, its
    **Finish** lock, its tree — so the window, the page and every cluster window never
    disagree; ``origin`` is the cluster window that opened the clip, so a prompt opens over it
    and INTERPRET's selection stays where it was. A clip check is not a label and changes
    nothing the page shows, so it does not ride the label queue: it has its own, kept in order
    so a changed answer lands after the one it replaces (``storage/member_checks.py``).
    """

    def __init__(self, page, node: int, *, origin=None) -> None:
        self._page = page
        self._node = int(node)
        self._origin = origin
        self._run = page._run
        self._queue: list = []
        self._busy = False
        # Each animal's track, read once by the fingerprint and reused for every clip after.
        self._tracks: dict = {}
        # *Cluster profile…* in the cluster sections only where the page has the window.
        profile = getattr(page, "open_profile", None)
        if callable(profile):
            self.open_profile = lambda parent: profile(self._node, parent)

    def node(self) -> int:
        return self._node

    def title(self) -> str:
        return f"{self._page._terms.unit} #{self._node}"

    def current_label(self) -> str:
        return self._page._names.get(str(self._node), "")

    def label_choices(self) -> list[str]:
        return sorted({n for n in self._page._names.values() if n})

    def edit_reason(self) -> str:
        if not self._page._can_edit:
            return self._page._edit_reason() or "This run cannot be changed here."
        return ""

    def apply_label(self, name: str) -> None:
        self._page._label(self._node, name, origin=self._origin)

    def load_checks(self):
        return self._run.member_checks(
            self._node
        )  # off-thread (the panel's BackgroundCall)

    def experiment(self):
        return self._page._experiment

    def fingerprint(self, clip, progress):
        from detail.api.clip_fingerprint import cluster_fingerprint

        return cluster_fingerprint(  # off-thread (the panel's StageProgressWorker)
            self._page._experiment,
            self._run,
            self._node,
            clip_tuple(clip),
            progress=progress,
            tracks=self._tracks,
        )

    def clip_measures(self, clip, names: list[str]) -> dict:
        from detail.api.clip_fingerprint import clip_measures

        return clip_measures(  # off-thread (the panel's BackgroundCall)
            self._page._experiment,
            self._run,
            clip_tuple(clip),
            names,
            self._tracks,
        )

    def member_clips(self, n: int) -> list:
        return self._clips(self._node, n)

    def sibling(self):
        view = self._page._view
        parents = view.ancestors(self._node)
        if not parents:
            return None
        others = [c for c in view.children(parents[-1]) if int(c) != self._node]
        if not others:
            return None
        sibling = int(others[0])
        name = self._page._names.get(str(sibling), "")
        title = f"{self._page._terms.unit} #{sibling}" + (f" ({name})" if name else "")
        return title, self._clips(sibling, 10)

    def _clips(self, node: int, n: int) -> list:
        """``node``'s ``n`` most typical clips (``NodeView.clips``, in memory)."""
        from detail.gui.behavior_query import ClipSpec

        page = self._page
        view = page._view
        clips = view.clips(int(node))
        out = []
        for i in clips.order("typical")[:n]:
            recording_id = clips.recording_ids[int(clips.recording[i])]
            out.append(
                ClipSpec(
                    recording_id,
                    subject_key(clips.subject[i]),
                    page._state.embed_run_for(recording_id),
                    int(clips.start_ns[i]),
                    int(clips.end_ns[i]),
                    f"#{node} · {view.recording_names.get(recording_id, recording_id)}",
                )
            )
        return out

    def check(self, key: ClipKey, verdict: str) -> None:
        run = self._run
        self._queue.append(
            lambda: run.check_member(
                key.node,
                key.recording_id,
                key.subject,
                key.start_ns,
                key.end_ns,
                verdict,
            )
        )
        self._next()

    def _next(self) -> None:
        from PyQt6.QtWidgets import QMessageBox

        from detail.gui.workers import BackgroundCall

        if self._busy or not self._queue:
            return
        self._busy = True
        call = BackgroundCall(self._queue.pop(0), receiver=self._page)

        def done(_result) -> None:
            self._busy = False
            self._next()

        def failed(exc) -> None:
            self._busy = False
            self._queue.clear()
            QMessageBox.critical(
                self._origin or self._page, "Check not saved", str(exc)
            )

        call.done.connect(done)
        call.failed.connect(failed)
        call.start()


def page_lane(page, node: int):
    """The clip window's cluster strip around a clip, with ``node`` as "this"
    (``NodeView.lane``, read in memory) — ``lane(clip, start_ns, end_ns)``."""
    view = page._view
    names = dict(page._names)

    def lane(clip, start_ns: int, end_ns: int) -> list[dict]:
        return view.lane(
            clip.recording_id,
            subject_key(clip.track_id),
            start_ns,
            end_ns,
            int(node),
            names,
        )

    return lane


def clip_tuple(clip) -> tuple:
    """A ``ClipSpec`` as ``api.clip_fingerprint`` names a clip."""
    return (
        str(clip.recording_id),
        subject_key(clip.track_id),
        int(clip.start_t_ns),
        int(clip.end_t_ns),
    )
