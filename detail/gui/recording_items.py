"""Filling a recording picker, and reading the selection back.

A recording's **id is the tool's and its name is the user's** (§11). A picker must show the
name — the Recordings table shows only names, so a list of `recording_000…recording_NNN` is
unreadable without opening each recording's page — while the *selection* must still be ids,
because that is what every API call takes.

Those two facts pull against each other whenever the id is used as the row's display text
and read back with ``item.text()``: showing a name then silently breaks selection. Keeping
the id in ``UserRole`` separates them, so "read the id back" stops meaning "read the label".
``gui/launch.py`` already did this for the processing-run dialog; these helpers are that
pattern, shared, so the next picker inherits it.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidget, QListWidgetItem

_ID_ROLE = Qt.ItemDataRole.UserRole


def fill_recording_list(listwidget: QListWidget, recordings) -> None:
    """Replace ``listwidget``'s rows with one per recording: name shown, id in ``UserRole``."""
    listwidget.clear()
    for recording in recordings:
        item = QListWidgetItem(recording.name, listwidget)
        item.setData(_ID_ROLE, recording.id)


def select_recording_ids(listwidget: QListWidget, recording_ids) -> None:
    """Select the rows holding ``recording_ids``; clear the selection when it is empty.

    The twin of :func:`selected_recording_ids`, so a picker seeded from a saved record restores
    the set that record was run over. Ids not in the list are ignored -- a recording can be
    deleted after a run.
    """
    wanted = set(recording_ids or ())
    listwidget.clearSelection()
    for row in range(listwidget.count()):
        item = listwidget.item(row)
        if (item.data(_ID_ROLE) or item.text()) in wanted:
            item.setSelected(True)


def selected_recording_ids(listwidget: QListWidget) -> list[str]:
    """The ids of the selected rows, in list order.

    Falls back to the row's text for an item written without an id — a picker mid-migration
    still selects rather than silently returning nothing.
    """
    return [item.data(_ID_ROLE) or item.text() for item in listwidget.selectedItems()]


def fill_sampling_list(listwidget: QListWidget, recordings) -> int:
    """Fill a picker for a frame sample, and return how many of ``recordings`` have video.

    An empty selection there means *every recording with video* (``Experiment.label_dataset``),
    so a recording without one is greyed and says why: it has nothing to sample, and selecting
    it is refused by name. Call inside ``read_cache.pinned()`` — each row reads its recording's
    name and its video parts.
    """
    recordings = list(recordings)
    fill_recording_list(listwidget, recordings)
    with_video = 0
    for row, recording in enumerate(recordings):
        if recording.video_parts:
            with_video += 1
            continue
        item = listwidget.item(row)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        item.setText(f"{recording.name}   (no video)")
    return with_video


def sampling_label(with_video: int) -> str:
    """The row label for :func:`fill_sampling_list`'s picker: what selecting nothing uses."""
    return f"Recordings (none = all {with_video} with video):"
