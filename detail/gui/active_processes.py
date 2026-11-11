"""Active processes — everything going on, across every experiment, in one place.

**The question this answers.** Five detail pages each knew whether the one run they were showing
was still going. Nothing knew whether *anything* was, so finding out whether last night's
detection finished meant remembering which experiment it was in and opening it. The records were
always there; nobody was asking across them.

**Processes, not runs.** A processing run is one family of the things in flight here; an export
(#354) is another, and a tool run will be a third. They are shown as **titled sections of one
page** rather than one flat table, because their columns genuinely differ: a run's useful row is
what it is and how far along, an export's is *where it is going* — a question no run has. One
table would force the lowest common denominator on both, and a `kind` column plus a sort is not
the same as saying what the two things are.

One page rather than a rail or a drill-in: this screen exists to be read at a glance, and
sub-navigation would hide half of what it is for behind a click.

**It routes rather than reports.** A row says what, where, how far along the status is, and since
when — enough to recognise a run and rank it — and opening one hands over to that run's own page,
where the bar, the activity log and the per-recording detail already live. Drawing a second live
bar here would be the second event→render mapping DESIGN_PLAN.md:1351 warns about, and it would
be the one nobody tests.

**Failed runs are listed with the active ones**, deliberately. A run that stopped overnight is
exactly as urgent as one still going, and it is the other thing somebody opening this screen is
looking for. It is also the case that is hardest to find otherwise: an active run at least has a
page you remember launching.

**The scan is off the UI thread**, because it walks every experiment's run records — the shape
this package has a recorded class about, where an O(N) sweep of individually-cheap reads is
nothing locally and seconds over a share. Milliseconds on a local disk is exactly the
number that makes it tempting to do inline.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from detail.gui.pages import IndexTable, _heading, _instructions
from detail.gui.tables import fit_table_height, give_slack_to, row_id
from detail.gui.workers import BackgroundCall

#: A run's row. An export's differs — see `_EXPORT_COLUMNS` — which is the argument for sections.
_COLUMNS = ["experiment", "what", "run", "status", "started"]

#: An export's row carries **to** as well: the folder its files went to. That column is the
#: reason a finished export is listed here at all — the files are outside the experiment, so
#: "where are they" is a question no run id answers, and it is the last question of every export.
_EXPORT_COLUMNS = ["experiment", "exporting", "to", "run", "status", "started"]

#: The kind `api.active.RUN_KINDS` gives an export run. Named once so the split is one constant
#: rather than a string repeated at each of the three places that sort rows into a section.
_EXPORT_KIND = "export"

#: An import's row carries no run id, because an import is not a run — it is a recording being
#: filled in. So the two identifying columns are the recording's **name**, which is what the
#: user called it, and its id, which is what the CLI and every run record speak (§11). There is
#: no "what" column: every row in this section is the same verb.
_IMPORT_COLUMNS = ["experiment", "recording", "id", "status", "started"]

#: The kind `api.active.RUN_KINDS` gives a recording import.
_IMPORT_KIND = "import"


class _Section(QWidget):
    """One titled listing — a heading, a caption, and a table that says when it is empty."""

    def __init__(
        self,
        title: str,
        columns: list[str],
        empty_text: str,
        on_open,
        *,
        stretch: int = 0,
        detail: bool = False,
    ) -> None:
        super().__init__()
        self._on_open = on_open
        self._rows: list = []
        #: Which column takes the slack — the one whose content genuinely varies in length. For
        #: runs that is the experiment name; for exports it is the destination path, which is
        #: both the longest thing in the row and the thing being read.
        self._stretch = stretch
        #: Whether this family's rows carry `ActiveRun.detail`. A section says so rather than
        #: the page branching on which section it is building.
        self._detail = detail

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_heading(title))
        self._caption = _instructions("Reading every experiment…")
        layout.addWidget(self._caption)
        self._table = IndexTable(columns, empty_text=empty_text, navigable=True)
        # `view.doubleClicked` + `row_id`, not `index.row()`: every table sorts now, so a row's
        # position stops meaning what it meant the moment someone clicks a header.
        self._table.view.doubleClicked.connect(lambda idx: self._open(row_id(idx)))
        self._columns = columns
        self._size_columns()
        layout.addWidget(self._table)
        # Hidden until the scan lands. `IndexTable` swaps in its notice at zero rows, and zero
        # rows before the read is not a claim about the workspace — it is a claim about the
        # clock. Rendered, the page said "Reading every experiment…" and "(nothing is running)"
        # one line apart: the recorded empty-state class, invisible on a local disk and the
        # first thing you see on a share, where the caption alone is the weaker of the two
        # messages on screen.
        self._table.hide()

    def _size_columns(self) -> None:
        """Give `self._stretch` the slack; every column stays user-resizable.

        `_table` gives the slack to the last data column, which here is a fixed-width timestamp —
        so the widest thing on screen would be the one whose content never varies, while the
        experiment names that actually differ in length get truncated.

        **This used to set every other column to `ResizeToContents` and this one to `Stretch`**,
        which locked every column: `ResizeToContents` refuses a drag, and so does `Stretch`, which
        also made a column to its right grow leftward when dragged. `tables.give_slack_to` keeps
        them all `Interactive`, and `IndexTable.set_rows` re-fits it after every fill.
        """
        give_slack_to(self._table.view, self._stretch)

    def set_runs(self, runs, summary: str) -> None:
        self._rows = list(runs)
        self._table.show()  # the read has happened, so an empty table now means something
        rows = [
            (
                run.experiment_name,
                run.label,
                *((run.detail,) if self._detail else ()),
                run.run_id,
                run.status,
                (run.created_at or "").replace("T", " ")[:16],
            )
            for run in self._rows
        ]
        self._table.set_rows(rows, ids=list(range(len(rows))))
        # And then re-derive the height, because `set_rows` already derived one — against the
        # *natural* column widths `resizeColumnsToContents` had just set, which for a destination
        # path are wider than the view. `fit_table_height` reserves a horizontal scrollbar when
        # the columns overrun, `set_rows` then re-fits the slack column back inside the view, and
        # the bar never appears: 18px of empty grid under the last row of the exports table,
        # visible only in a render. Whoever changes the columns after the fill owns the height.
        fit_table_height(self._table.view)
        # The hint goes with the rows: inviting a double-click on a table with none is the
        # empty-state class this page's own notice exists to avoid.
        self._caption.setText(
            summary + (" — double-click a row to open it" if rows else "")
        )

    def _open(self, index) -> None:
        if self._on_open is None or index is None:
            return
        if 0 <= int(index) < len(self._rows):
            self._on_open(self._rows[int(index)])


class ActiveProcessesPage(QWidget):
    """The listing. ``on_open(run)`` receives the :class:`~detail.api.active.ActiveRun` picked."""

    def __init__(self, on_open=None) -> None:
        super().__init__()
        #: The empty-state guard settles on this rather than asserting mid-flight
        #: (`docs/gui_design.md`): until the scan lands, a table with no rows is not a claim
        #: that there is nothing to show.
        self._reading = True

        layout = QVBoxLayout(self)
        self._runs = _Section(
            "Runs",
            _COLUMNS,
            "(nothing is running — a run you start from any experiment appears here, "
            "and so does one that failed while you were away)",
            on_open,
        )
        layout.addWidget(self._runs)
        self._exports = _Section(
            "Exports",
            _EXPORT_COLUMNS,
            "(no exports yet — one you start with Export… on any screen appears here)",
            on_open,
            # The destination takes the slack: it is the longest cell and the one being read.
            stretch=_EXPORT_COLUMNS.index("to"),
            detail=True,
        )
        layout.addWidget(self._exports)
        self._imports = _Section(
            "Imports",
            _IMPORT_COLUMNS,
            "(no imports yet — video you add to an experiment with Add recordings appears "
            "here, and your recent imports stay listed once they finish)",
            on_open,
            # The recording's name takes the slack: it is what the row is *about*, and the one
            # cell whose length genuinely varies once every row names the same verb.
            stretch=_IMPORT_COLUMNS.index("recording"),
        )
        layout.addWidget(self._imports)
        layout.addStretch()
        self.refresh()

    def refresh(self) -> None:
        """Re-scan, off the UI thread.

        The caption says *reading* until the scan lands rather than showing an empty table: a
        table with no rows is a claim about the workspace, and making it before the data is read
        tells someone with four running jobs that they have none — the recorded empty-state class,
        which is invisible on a local disk and the first thing you see on a share.
        """
        from detail.api import active

        call = BackgroundCall(active.list_active, receiver=self)
        call.done.connect(self._show)
        call.start()

    def _show(self, runs) -> None:
        rows = list(runs)
        self._reading = False
        own_section = {_EXPORT_KIND: self._exports, _IMPORT_KIND: self._imports}
        for kind, section in own_section.items():
            picked = [r for r in rows if r.kind == kind]
            section.set_runs(picked, _summary(picked))
        # Runs take the **remainder**, not a list of kinds. A family added to `RUN_KINDS` and
        # forgotten here then lands in Runs under the generic columns — visible and imprecise,
        # which someone will report — rather than in no section at all, which is a row that
        # exists in the API and appears on no screen and which nothing would fail over.
        others = [r for r in rows if r.kind not in own_section]
        self._runs.set_runs(others, _summary(others))


def _summary(runs) -> str:
    """``N running`` plus whatever else is on screen, and nothing that is not.

    It used to read `not is_failed` as "running", which was true while the listing held only
    active and failed rows and became a wrong number the moment finished exports joined them —
    ten complete exports reported as ten running. The three states are counted from what they
    are, and a bucket with nothing in it is not mentioned: a standing "0 failed" is noise on
    every screen where nothing has gone wrong, which is most of them.
    """
    running = sum(1 for r in runs if r.is_active)
    failed = sum(1 for r in runs if r.is_failed)
    done = len(runs) - running - failed
    parts = [f"{running} running"]
    if failed:
        parts.append(f"{failed} failed")
    if done:
        parts.append(f"{done} finished")
    return " · ".join(parts)
