"""Where the camera looks at the animal from — one vocabulary and one resolver (§10).

The view is a fact about the rig, so it is set once: an experiment-wide default in
``config.yaml``, overridden by a recording's own ``recording.yaml`` when that recording was
made on a different rig. Every consumer resolves it through :func:`resolve`, so "which view
does this recording have" has one answer however it is asked.

**Not set is an answer, never a guess.** A recording with no view of its own and no
experiment default resolves to ``view=None``; a consumer that needs a view skips what
depends on it and says why. Guessing "top down" because most rigs are would turn a missing
fact into a wrong measurement, quietly.

Pure: dicts in, a dataclass out. The storage layer reads and writes the two keys; this module
only knows what they mean.
"""

from collections.abc import Iterable
from dataclasses import dataclass

#: The stored values and what a person reads, in the order every picker shows them.
VIEWS: dict[str, str] = {
    "top_down": "Top down",
    "side": "Side view",
    "angled_down": "Angled down view",
    "other": "Other",
}

VIEW_KEY = "camera_view"
NOTE_KEY = "camera_view_note"

SOURCE_RECORDING = "recording"
SOURCE_EXPERIMENT = "experiment"


@dataclass(frozen=True)
class CameraView:
    """A resolved camera view and where it came from.

    ``source`` is ``"recording"`` (the recording's own), ``"experiment"`` (the default it
    inherits) or ``None`` (not set anywhere, and then ``view`` is ``None`` too).
    """

    view: str | None
    note: str | None = None
    source: str | None = None

    @property
    def is_set(self) -> bool:
        return self.view is not None

    @property
    def label(self) -> str:
        """What a person reads: ``"Side view"``, ``"Other: <note>"``, or ``"Not set"``."""
        if self.view is None:
            return "Not set"
        if self.view == "other" and self.note:
            return f"Other: {self.note}"
        return VIEWS[self.view]

    @property
    def described(self) -> str:
        """The label with where it came from: ``"Side view (experiment default)"`` when
        inherited, the bare label when the recording sets its own or nothing is set."""
        if self.source == SOURCE_EXPERIMENT:
            return f"{self.label} (experiment default)"
        return self.label

    def to_record(self) -> dict:
        """The resolved value as a consumer persists it (Tenet 9): the view, its note, and
        where it came from — so a record still says what it ran on after the default moves.
        """
        return {"view": self.view, "note": self.note, "source": self.source}


def validate(view: str | None, note: str | None) -> tuple[str | None, str | None]:
    """Check a view + note pair and return it normalized (note stripped, ``None`` if blank).

    Raises:
        ValueError: If ``view`` is not one of :data:`VIEWS`, if ``other`` has no note, or if
            a note is given with any view other than ``other``.
    """
    note = note.strip() if isinstance(note, str) else note
    note = note or None
    if view is None:
        if note is not None:
            raise ValueError("a camera view note needs a view of 'other'")
        return None, None
    if view not in VIEWS:
        raise ValueError(
            f"unknown camera view {view!r}; expected one of {', '.join(VIEWS)}"
        )
    if view == "other" and note is None:
        raise ValueError("camera view 'other' needs a description (the note)")
    if view != "other" and note is not None:
        raise ValueError(
            f"a camera view note is only kept for 'other', not {view!r}; "
            "drop the note or choose 'other'"
        )
    return view, note


def stated(block: dict) -> tuple[str | None, str | None]:
    """The ``(view, note)`` one file states for itself — a ``recording.yaml`` or
    ``config.yaml`` dict, read without inheritance and validated.

    A malformed stored value raises rather than resolving to "not set": a file that says
    something we cannot read is not a file that says nothing.
    """
    return validate(block.get(VIEW_KEY), block.get(NOTE_KEY))


def resolve(recording: dict, config: dict) -> CameraView:
    """The recording's camera view: its own, else the experiment default, else not set."""
    view, note = stated(recording)
    if view is not None:
        return CameraView(view, note, SOURCE_RECORDING)
    view, note = stated(config)
    if view is not None:
        return CameraView(view, note, SOURCE_EXPERIMENT)
    return CameraView(None)


def all_top_down(views: Iterable[CameraView]) -> bool:
    """Whether there is at least one view and every one is ``top_down``.

    Not set is unknown rather than top down, so one recording with no view makes this False:
    a consumer that drops something for overhead cameras keeps it until the view says so.
    """
    seen = [v.view for v in views]
    return bool(seen) and all(view == "top_down" for view in seen)


def write(block: dict, view: str | None, note: str | None) -> dict:
    """``block`` with the two keys set to ``view`` / ``note``, or both removed for ``None``.

    Returns the same dict, mutated, so a storage writer can read-modify-write in one line.

    Raises:
        ValueError: As :func:`validate`.
    """
    view, note = validate(view, note)
    block.pop(VIEW_KEY, None)
    block.pop(NOTE_KEY, None)
    if view is not None:
        block[VIEW_KEY] = view
        if note is not None:
            block[NOTE_KEY] = note
    return block
