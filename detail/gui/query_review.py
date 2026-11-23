"""AUDIT on a query run: every clip a behavior's cutoff keeps, one recording at a time (§20).

The query half of AUDIT, on the clustering review's pane (``label_clean.ReviewPane``) and its
marks (``review_marks.ReviewMarks``): where ``label_check`` lists labels and their clusters, this
lists the workspace's behavior's searched recordings. A recording is reviewed whole — every clip
its cutoff keeps, playing, a click removing a wrong one, Finish writing the census
(``QueryRun.finish_review``) — so its precision is a count.

**Blind, structurally.** The clips come from ``QueryRun.review_clips``, which hands out no score
and a seeded random order: there is no score here to hide, and no rank to read one from. A
caption is the recording and its time. The score is in BROWSE.

**No moving.** Each behavior is its own search under its own cutoff, so right-click offers *Put
back* and nothing else.

**A figure is exact or it is not shown.** A behavior's line gives its figure once every recording is
exact at the cutoff; a recording reviewed at a higher cutoff says how far down it was reviewed
(``storage/audit_figures.query_review_line`` / ``query_review_cell``, the CLI's wording too).

Every read runs off the UI thread (``run_busy``), and every write through the run's write queue,
which outlives the page.
"""

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QMessageBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detail.gui.busy import run_busy
from detail.gui.grip_splitter import GripSplitter
from detail.gui.label_check import (
    KEY_ROLE,
    LIST_MIN_PX,
    LIST_PX,
    ROW_ROLE,
    ReviewListDelegate,
    percent,
)
from detail.gui.label_clean import (
    ReviewPane,
    ScopeClips,
    day_of,
    tiles_for,
)
from detail.gui.live_montage import context_span
from detail.gui.review_marks import ReviewMarks, apply_change
from detail.gui.sizing import lines
from detail.storage import audit_figures

#: Where the list's width is kept, apart from the clustering AUDIT's.
LIST_SPLIT_KEY = "query_review_list"
#: The ``QSettings`` group the grid's orientation, layers and context are kept under.
KEYS = "query_review"

#: Why a control that would write is dead, by cause.
FROZEN = (
    "This is a snapshot — the calibration as it was pinned. Go back to the live run to "
    "review its clips."
)
FINISHED = "This run is finished. Reopen it to review its clips."
NO_CUTOFF = (
    "This behavior has no cutoff yet. Set one on CALIBRATE first: the review covers every "
    "clip the cutoff keeps."
)
EMPTY = "Open a recording in the list to review its clips."


class QueryReviewTab(QWidget):
    """The query workspace's AUDIT tab: the behavior's recordings on the left, one recording's
    clips in the review pane on the right."""

    #: A removal or a finish was written: other tabs reading the artifact read it again.
    changed = pyqtSignal()

    def __init__(self, experiment, run) -> None:
        super().__init__()
        self._experiment = experiment
        self._run = run
        self._behavior: str | None = None
        #: The last read, with every mark not yet read back laid over its ``taken``.
        self._data: dict = {}
        #: The recording in the pane — ``{rid, clips, display, edits}`` — or ``None``.
        self._open: dict | None = None
        #: The recording whose finish is being written; its result shows once read back.
        self._finishing: str | None = None
        #: Reviewed recordings reopened and not finished again: their old figure is out of date.
        self._reopened: set[str] = set()
        self._catalog = None
        self._marks = ReviewMarks(
            run, self, on_written=self.changed.emit, on_failed=self._write_failed
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.figure = QLabel("")
        self.figure.setContentsMargins(lines(0.4), lines(0.25), lines(0.4), lines(0.25))
        self.figure.setWordWrap(True)
        self.figure.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self.figure)

        self._split = GripSplitter(Qt.Orientation.Horizontal, key=LIST_SPLIT_KEY)
        self._list = QTreeWidget()
        self._list.setHeaderHidden(True)
        self._list.setColumnCount(1)
        self._list.setMinimumWidth(LIST_MIN_PX)
        self._list.setRootIsDecorated(False)
        self._list.setItemDelegate(ReviewListDelegate(self._list))
        self._list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.itemClicked.connect(self._on_chosen)
        self._list.itemActivated.connect(self._on_chosen)
        self._split.addWidget(self._list)

        from detail.gui.montage_export import ExportSource

        self.pane = ReviewPane(
            export=ExportSource(
                clips=self.export_clips,
                experiment_dir=lambda: self._experiment.dir,
                origin=lambda: None,
                name=self._export_stem,
                empty=lambda: EMPTY,
            ),
            noun="recording",
            keys=KEYS,
        )
        self.pane.take_out_requested.connect(self._take_out)
        self.pane.put_back_requested.connect(self._put_back)
        self.pane.undo_requested.connect(self._undo)
        self.pane.finish_requested.connect(self._finish)
        self.pane.next_requested.connect(self._next)
        self.pane.again_requested.connect(self._again)
        self.pane.set_targets([])
        self.pane.montage.clip_activated.connect(
            lambda i: QTimer.singleShot(0, lambda: self._play_clip(i))
        )
        self.pane.montage.context_changed.connect(
            lambda _s: self._show_tiles(new_source=True)
        )
        self._split.addWidget(self.pane)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.set_foldable(0, "Recordings")
        if not self._split.remember():
            self._split.setSizes([LIST_PX, lines(80)])
        root.addWidget(self._split, 1)
        self.pane.set_editable("Choose a behavior to review.")

    # -- the behavior -------------------------------------------------------------------
    def set_behavior(self, behavior_id: str | None) -> None:
        """Review ``behavior_id`` — the workspace's selector."""
        if behavior_id == self._behavior:
            return
        self._behavior = behavior_id
        self._open = None
        self._finishing = None
        self._reopened.clear()
        self._forget_history()
        self.reload()

    # -- reading (off the UI thread) ------------------------------------------------------
    def reload(self) -> None:
        if self._behavior is None:
            return
        landed = self._marks.landed
        run_busy(
            self,
            self._compute,
            lambda data, n=landed: self._render(data, landed_before=n),
            label="Reading the review…",
            supersedes="query_review",
        )

    def _compute(self) -> dict:
        from detail.api import clip_tiles

        run, behavior_id = self._run, self._behavior
        state = run.state()
        behavior = state.behavior(behavior_id)
        out: dict = {
            "behavior": behavior_id,
            "name": behavior.name,
            "cutoff": behavior.cutoff,
            "editable": bool(run.editable),
            "finished": bool(run.editable) and run.status == "complete",
        }
        if behavior.cutoff is None:
            return out
        rids = list(state.search_recording_ids)
        embeds = {rid: state.embed_run_id for rid in rids}
        tracking = clip_tiles.tracking_runs(  # off-thread (run_busy)
            self._experiment.dir, embeds
        )
        out.update(
            {
                "rows": run.review_by_recording(behavior_id),
                "estimate": run.review_precision(behavior_id),
                "taken": {
                    _key(r): {**r, "from_path": behavior.name, "to_path": None}
                    for r in run.taken_out(behavior_id)
                },
                "embeds": embeds,
                "tracking": tracking,
                "missing": clip_tiles.missing(  # off-thread (run_busy)
                    self._experiment, tracking, rids
                )["proxy"],
            }
        )
        return out

    def _render(self, data: dict, *, landed_before: int | None = None) -> None:
        from detail.api import clip_tiles

        if data.get("behavior") != self._behavior:
            return
        data = {
            **data,
            "taken": self._marks.overlay(
                dict(data.get("taken") or {}), landed_before=landed_before
            ),
        }
        self._data = data
        if data.get("tracking") is not None:
            self._catalog = clip_tiles.TileCatalog(
                self._experiment, data["tracking"], data.get("embeds")
            )
        self._sync_figure()
        self._fill_list()
        self.pane.set_editable(self._reason())
        if self._open is not None:
            self._refresh_open()

    def _reason(self) -> str:
        """Why nothing here can be written now, or ``""``."""
        data = self._data
        if not data.get("editable", True):
            return FROZEN
        if data.get("finished"):
            return FINISHED
        if data.get("cutoff") is None:
            return NO_CUTOFF
        return ""

    def _sync_figure(self) -> None:
        data = self._data
        if data.get("cutoff") is None:
            self.figure.setText(NO_CUTOFF)
            return
        text = audit_figures.query_review_line(
            data["name"], data["cutoff"], data.get("estimate"), data.get("rows") or []
        )
        missing = [self._name(rid) for rid in data.get("missing") or []]
        if missing:
            text += (
                f"\nNo preview yet for {', '.join(missing)}: build the previews "
                "(detail cache previews build) to review their clips."
            )
        self.figure.setText(text)

    # -- the list -----------------------------------------------------------------------
    def _rows(self) -> list[dict]:
        return list(self._data.get("rows") or [])

    def _row(self, rid: str) -> dict:
        return next((r for r in self._rows() if r["recording_id"] == rid), {})

    def _name(self, rid: str) -> str:
        return self._row(rid).get("name") or rid

    def _fill_list(self) -> None:
        self._list.clear()
        for row in self._rows():
            item = QTreeWidgetItem([row["name"]])
            item.setData(0, KEY_ROLE, row["recording_id"])
            item.setData(0, ROW_ROLE, self._look(row))
            item.setToolTip(0, audit_figures.query_review_cell(row))
            self._list.addTopLevelItem(item)
            if self._open is not None and self._open["rid"] == row["recording_id"]:
                self._list.setCurrentItem(item)

    def _look(self, row: dict) -> dict:
        """A recording's list row, drawn as a cluster's is (``ReviewListDelegate``)."""
        rid = row["recording_id"]
        is_open = self._open is not None and self._open["rid"] == rid
        out = (
            len(self._open["edits"])
            if is_open
            else sum(1 for key in self._data.get("taken") or {} if key[0] == rid)
        )
        if rid in self._reopened:
            state, status = (
                "reviewing",
                f"was {percent(row['estimate'])} · {out:,} removed",
            )
        elif (
            row["exact"]
            and row["size"]
            and not (is_open and not self.pane.is_finished())
        ):
            state, status = "reviewed", f"{percent(row['estimate'])} · {out:,} removed"
        elif not row["size"]:
            state, status = "reviewed", "no clips at this cutoff"
        elif is_open:
            state, status = "reviewing", f"reviewing · {out:,} removed"
        else:
            state, status = "not", audit_figures.query_review_cell(row)
        return {
            "kind": "cluster",
            "name": row["name"],
            "state": state,
            "side": f"{int(row['size']):,} clips",
            "line": status,
            "fraction": 1.0 if state == "reviewed" else 0.0,
        }

    def _on_chosen(self, item, _column=0) -> None:
        rid = item.data(0, KEY_ROLE)
        if rid:
            self.open_recording(str(rid))

    # -- one recording ------------------------------------------------------------------
    def open_recording(self, rid: str) -> None:
        """Open ``rid``'s clips in the pane: reviewing, or its result if it is reviewed."""
        if self._data.get("cutoff") is None:
            return
        behavior = self._behavior
        self._forget_history()
        self.pane.show_scope(self._name(rid), self._data.get("name", ""))
        self.pane.set_waiting("reading this recording's clips…")
        run_busy(
            self,
            lambda: self._run.review_clips(behavior, rid),
            lambda clips, r=rid, b=behavior: self._show_open(r, b, clips),
            label="Reading the clips…",
            supersedes="query_review_clips",
        )

    def _show_open(self, rid: str, behavior: str, clips: list[dict]) -> None:
        if behavior != self._behavior:
            return
        rows = [
            {
                "recording_id": c["recording_id"],
                "subject": int(c["subject"]),
                "start_t_ns": int(c["t_ns"]),
                "end_t_ns": int(c["end_t_ns"]),
            }
            for c in clips
        ]
        names = {r["recording_id"]: self._name(r["recording_id"]) for r in self._rows()}
        self._open = {
            "rid": rid,
            "clips": ScopeClips(rows, np.zeros(len(rows)), names),
            "edits": {},
        }
        self.pane.set_waiting("")
        self.pane.set_was("")
        row = self._row(rid)
        if row.get("exact") and row.get("size") and rid not in self._reopened:
            self._show_result(row)
        else:
            self.pane.start_review()
        self._refresh_open(new_source=True)

    def _refresh_open(self, *, new_source: bool = False) -> None:
        opened = self._open
        clips: ScopeClips = opened["clips"]
        taken = self._data.get("taken") or {}
        opened["edits"] = {
            clips.key(i): taken[clips.key(i)]
            for i in range(len(clips.rows))
            if clips.key(i) in taken
        }
        row = self._row(opened["rid"])
        if self._finishing == opened["rid"] and row.get("exact"):
            self._finishing = None
            self._reopened.discard(opened["rid"])
            self.pane.set_was("")
            self._show_result(row)
        self._show_tiles(new_source=new_source)

    def _display(self) -> tuple[list[int], list[tuple[str, int]] | None]:
        """The review order (the run's shuffle) while reviewing; once finished, the removed
        clips first, then the kept ones, each under its heading."""
        opened = self._open
        order = list(range(len(opened["clips"].rows)))
        if not self.pane.is_finished():
            return order, None
        clips: ScopeClips = opened["clips"]
        removed = [i for i in order if clips.key(i) in opened["edits"]]
        kept = [i for i in order if clips.key(i) not in opened["edits"]]
        sections = [(f"Removed ({len(removed):,})", len(removed))] if removed else []
        sections.append(
            (f"Kept as {self._data.get('name', '')} ({len(kept):,})", len(kept))
        )
        return removed + kept, sections

    def _show_tiles(self, *, new_source: bool = False) -> None:
        from detail.api import clip_tiles

        opened = self._open
        if opened is None:
            return
        clips: ScopeClips = opened["clips"]
        order, sections = self._display()
        scroll_to_top = order != opened.get("display")
        opened["display"] = order
        montage = self.pane.montage
        context = montage.context_s()
        montage.set_clips(
            tiles_for(clips, order, opened["edits"], context),
            keep_scroll=not scroll_to_top,
            sections=sections,
        )
        if (scroll_to_top or new_source) and self._catalog is not None:
            tiles = []
            for i in order:
                row = clips.rows[i]
                start, end = context_span(row["start_t_ns"], row["end_t_ns"], context)
                tiles.append(
                    clip_tiles.TileClip(row["recording_id"], row["subject"], start, end)
                )
            montage.set_source(clip_tiles.ClipTiles(self._catalog, tiles))
        for i in range(self._list.topLevelItemCount()):
            item = self._list.topLevelItem(i)
            if item.data(0, KEY_ROLE) == opened["rid"]:
                item.setData(0, ROW_ROLE, self._look(self._row(opened["rid"])))
        self.pane.refresh()

    def _show_result(self, row: dict) -> None:
        estimate = row.get("estimate") or {}
        n, yes = int(estimate.get("n") or 0), int(estimate.get("n_confirmed") or 0)
        name = self._data.get("name", "")
        self.pane.show_result(
            percent(estimate),
            f"{yes:,} of {n:,} are {name} · {n - yes:,} removed",
            self._next_name(),
            removed=n - yes,
            when=day_of(row.get("reviewed_at")),
        )

    def _next_rid(self) -> str | None:
        """The next recording not reviewed at the cutoff, in list order after the open one."""
        rows = [r for r in self._rows() if r["size"]]
        rids = [r["recording_id"] for r in rows]
        at = (
            rids.index(self._open["rid"])
            if self._open and self._open["rid"] in rids
            else -1
        )
        for row in rows[at + 1 :] + rows[: max(at, 0)]:
            if not row["exact"] and row["recording_id"] != (self._open or {}).get(
                "rid"
            ):
                return row["recording_id"]
        return None

    def _next_name(self) -> str | None:
        rid = self._next_rid()
        return None if rid is None else self._name(rid)

    def _next(self) -> None:
        rid = self._next_rid()
        if rid is not None:
            self.open_recording(rid)

    def _again(self) -> None:
        opened = self._open
        if opened is None or self._reason():
            return
        row = self._row(opened["rid"])
        self._reopened.add(opened["rid"])
        self.pane.set_was(f"was {percent(row.get('estimate'))} · reviewing again")
        self.pane.start_review()
        self._show_tiles()

    # -- marks ----------------------------------------------------------------------------
    def _rows_at(self, indices: list) -> list[dict]:
        opened = self._open
        return [opened["clips"].rows[int(opened["display"][int(i)])] for i in indices]

    def _take_out(self, indices: list, _to_label) -> None:
        """Remove the clips at display ``indices``: faded at once, written in order, undoable.
        A query clip is never moved, so a target is ignored (the pane offers none)."""
        if self._open is None or self._reason() or not indices:
            return
        rows = self._rows_at(indices)
        self._remember(rows, f"Removed {len(rows):,} from {self._data.get('name', '')}")
        self._set_marks(rows)

    def _put_back(self, indices: list) -> None:
        if self._open is None or self._reason() or not indices:
            return
        rows = self._rows_at(indices)
        self._remember(rows, f"Put back {len(rows):,} in {self._data.get('name', '')}")
        self._set_marks(rows, back=True)

    def _remember(self, rows: list[dict], text: str) -> None:
        edits = self._open["edits"]
        self._marks.remember(
            text, rows, {_row_key(r): edits.get(_row_key(r)) for r in rows}
        )
        self.pane.set_undo(text)

    def _set_marks(self, rows: list[dict], *, back: bool = False) -> None:
        behavior, name = self._behavior, self._data.get("name", "")
        change = {_row_key(r): None if back else _edit(_row_key(r), name) for r in rows}
        apply_change(self._data.setdefault("taken", {}), change)
        self._refresh_open()
        run = self._run
        refs = [_row_key(r) for r in rows]
        if back:
            self._marks.write(
                lambda: run.put_back_clips(refs, behavior), "Clips not put back", change
            )
        else:
            self._marks.write(
                lambda: run.take_out_clips(refs, behavior), "Clips not removed", change
            )

    def _undo(self) -> None:
        if self._open is None or self._reason() or self._marks.last_text() is None:
            return
        change = self._marks.pop()
        unmarked = [r for r in change["rows"] if change["before"][_row_key(r)] is None]
        marked = [
            r for r in change["rows"] if change["before"][_row_key(r)] is not None
        ]
        if unmarked:
            self._set_marks(unmarked, back=True)
        if marked:
            self._set_marks(marked)
        self.pane.set_undo(self._marks.last_text())

    def _forget_history(self) -> None:
        self._marks.forget()
        self.pane.set_undo(None)

    def _finish(self) -> None:
        """Write the census — after every mark already queued — then read the figure back."""
        opened = self._open
        if opened is None or self._reason() or not self.pane.reached_end():
            return
        rid, behavior, run = opened["rid"], self._behavior, self._run
        self._finishing = rid
        self._forget_history()
        self.pane.show_saving()
        self._marks.write(lambda: run.finish_review(behavior, rid), "Review not saved")
        self._marks.when_idle(self.reload)

    def _write_failed(self, what: str, exc) -> None:
        self._finishing = None
        QMessageBox.critical(self, what, str(exc))
        self._open = None
        self.reload()

    def writes_pending(self) -> bool:
        """Whether a change made on this run is not yet on disk."""
        return self._marks.busy()

    # -- watching and exporting -----------------------------------------------------------
    def _play_clip(self, index: int) -> None:
        """A tile double-clicked (or Space): the clip in the clip window, in the order shown,
        with no score — the review is blind."""
        from detail.gui import clip_popup
        from detail.gui.behavior_query import ClipSpec

        opened = self._open
        if opened is None or not 0 <= index < len(opened.get("display", [])):
            return
        clips: ScopeClips = opened["clips"]
        embeds = self._data.get("embeds", {})
        sequence = [
            ClipSpec(
                clips.rows[i]["recording_id"],
                clips.rows[i]["subject"],
                embeds.get(clips.rows[i]["recording_id"], ""),
                clips.rows[i]["start_t_ns"],
                clips.rows[i]["end_t_ns"],
                f"#{rank} · {clips.caption(i)}",
            )
            for rank, i in enumerate(opened["display"], start=1)
        ]
        clip_popup.open_clip_window(
            self,
            self._experiment,
            sequence[index],
            sequence=sequence,
            index=index,
            context_s=1,
            order_note=(
                "Removed first, then kept" if self.pane.is_finished() else "Shuffled"
            ),
        )

    def export_clips(self) -> list:
        """The open recording's clips in the order shown, for **Export ▾**."""
        from detail.api import clip_montage as montage_api

        opened = self._open
        if opened is None or "display" not in opened:
            return []
        clips: ScopeClips = opened["clips"]
        embeds = self._data.get("embeds", {})
        return [
            montage_api.MontageClip(
                rank=rank,
                recording_id=clips.rows[i]["recording_id"],
                recording_name=clips.names.get(clips.rows[i]["recording_id"], ""),
                subject=clips.rows[i]["subject"],
                embed_run_id=embeds.get(clips.rows[i]["recording_id"], ""),
                start_t_ns=clips.rows[i]["start_t_ns"],
                end_t_ns=clips.rows[i]["end_t_ns"],
            )
            for rank, i in enumerate(opened["display"], start=1)
        ]

    def _export_stem(self) -> str:
        opened = self._open or {}
        return (
            f"{self._data.get('name', 'behavior')}_{self._name(opened.get('rid', ''))}"
        )


def _key(record: dict) -> tuple[str, int, int, int]:
    return (
        str(record["recording_id"]),
        int(record.get("subject") or 0),
        int(record["t_ns"]),
        int(record["end_t_ns"]),
    )


def _row_key(row: dict) -> tuple[str, int, int, int]:
    return (
        str(row["recording_id"]),
        int(row["subject"]),
        int(row["start_t_ns"]),
        int(row["end_t_ns"]),
    )


def _edit(key: tuple, name: str) -> dict:
    """A removal record as the page holds it until the run's own is read back."""
    return {
        "recording_id": key[0],
        "subject": key[1],
        "t_ns": key[2],
        "end_t_ns": key[3],
        "from_path": name,
        "to_path": None,
        "at": "",
    }
