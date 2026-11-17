"""Tests for the crash-safe append-only log (detail/storage/append_log.py).

Extracted from annotation_runs, whose own tests still cover it through ``labels.csv``. What
is tested here is the primitive's contract on its own terms, because ``audit.csv`` is now a
second consumer and the failure modes are data loss rather than a wrong answer.

**The class: an append-only file that trusts its last byte.** A crash mid-row leaves the file
without a final newline. The next ``open(path, "a")`` then glues a new row onto the half row,
and a reader that *stopped* at the first row of the wrong width hid every decision written
after it — a clip review that read "0 of 8 reviewed" however often Finish was pressed, and a
``migrate`` that then rewrote the file from that read and dropped the rest for good. The
docstrings promised a crash costs "at most the row being written". The old test here wrote its
torn row *with* a newline and never appended after it, which is why it passed. The rule: an
append cuts a torn tail back first, and a read skips a bad row instead of stopping
(``test_a_torn_row_costs_only_itself*``).
"""

import csv
from unittest.mock import patch

from detail.core import atomic
from detail.storage.append_log import AppendLog

COLUMNS = ["id", "value", "note"]


def _log(tmp_path) -> AppendLog:
    return AppendLog(tmp_path / "log.csv", COLUMNS)


def test_header_written_once_at_creation(tmp_path):
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    log.append(["b", 2, "second"])
    lines = log.path.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0] == "id,value,note"
    assert len(lines) == 3  # one header, two rows


def test_missing_file_reads_empty(tmp_path):
    # A run with no decisions yet is a normal state, not an error.
    assert _log(tmp_path).read() == ([], [])
    assert _log(tmp_path).rows() == []


def test_rows_round_trip(tmp_path):
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    assert log.rows() == [{"id": "a", "value": "1", "note": "first"}]


def test_a_torn_trailing_line_is_tolerated(tmp_path):
    """A crash mid-append leaves a partial final row; every earlier decision must survive."""
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    log.append(["b", 2, "second"])
    with open(log.path, "a", newline="", encoding="utf-8") as f:
        f.write("c,3\n")  # torn: two fields where three were being written
    rows = log.rows()
    assert [r["id"] for r in rows] == ["a", "b"]


def test_rows_written_under_an_older_header_keep_their_values(tmp_path):
    """One file can hold two widths — the header is written once, so a log that predates a
    column keeps the narrow header while later appends carry the full one.

    Read positionally against a single header, every field past the missing column shifts by
    one and the old rows silently take the wrong values. The width identifies the shape.
    """
    # A file that already drifted mid-run: narrow header, a narrow row, then a wide one.
    # Read without appending first, so the migration is not what is under test here.
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])  # the pre-`note` header
        writer.writerow(["a", 1])
        writer.writerow(["b", 2, "second"])
    rows = AppendLog(path, COLUMNS).rows()
    assert rows[0] == {"id": "a", "value": "1"}  # not shifted into `note`
    assert rows[1] == {"id": "b", "value": "2", "note": "second"}


def test_appending_migrates_the_header(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])
        writer.writerow(["a", 1])
    AppendLog(path, COLUMNS).append(["b", 2, "second"])
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == "id,value,note"
    # the row that predated the column is kept, with the column blank rather than dropped
    assert AppendLog(path, COLUMNS).rows()[0] == {"id": "a", "value": "1", "note": ""}


def test_a_header_with_an_unknown_extra_column_still_reads_and_appends(tmp_path):
    # A log written by a newer build. Drop the unknown column, never the rows.
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*COLUMNS, "from_the_future"])
        writer.writerow(["a", 1, "first", "?"])
    AppendLog(path, COLUMNS).append(["b", 2, "second"])
    assert [r["id"] for r in AppendLog(path, COLUMNS).rows()] == ["a", "b"]


def test_migrating_a_current_log_does_not_rewrite_it(tmp_path):
    """Guard the guard: migration must be a no-op once the header matches, or every append
    rewrites the whole file and the append-only guarantee — the reason this exists — is gone.
    """
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    before = log.path.stat().st_mtime_ns

    log.migrate()

    assert log.path.stat().st_mtime_ns == before, "rewrote an already-current log"


def test_migration_is_atomic_no_tmp_left(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["id", "value"])
    AppendLog(path, COLUMNS).append(["b", 2, "second"])
    assert not list(tmp_path.glob("*.tmp"))


def test_migration_never_exposes_a_partial_file_to_a_reader(tmp_path):
    """What "atomic" actually has to mean, which the no-leftover-tmp check above does not test.

    A truncating in-place rewrite leaves no `.tmp` either, and passes that check while a reader
    landing mid-rewrite sees a half-written file — or an empty one. So this reads the log from
    inside the rewrite: `os.replace` is what makes that read see the whole old file, and a
    plain `open(path, "w")` is what makes it see nothing.
    """
    path = tmp_path / "log.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "value"])
        writer.writerow(["a", 1])
    log = AppendLog(path, COLUMNS)

    seen = {}
    real_replace = atomic.replace

    def peek(src, dst):
        # the instant before the swap, a concurrent reader must still see the old log whole
        seen["during"] = AppendLog(path, COLUMNS).read()
        return real_replace(src, dst)

    with patch.object(atomic, "replace", peek):
        log.append(["b", 2, "second"])

    header, rows = seen["during"]
    assert header == [
        "id",
        "value",
    ], "a reader mid-migration saw a torn or empty header"
    assert [row["id"] for row in rows] == ["a"]
    assert [row["id"] for row in log.rows()] == ["a", "b"]


def test_a_zero_byte_log_still_gets_its_header(tmp_path):
    """The crash this class advertises surviving, which it did not.

    `open(path, "a")` creates the file before the header reaches disk, so a crash in that
    window leaves a 0-byte file that *exists*. Keying "is this new" on existence then skipped
    the header, wrote a bare data row, and left `read` to eat that row as the header — and the
    next append's `migrate` rewrote the file under the real columns and dropped it for good.

    Unrecomputable primary data (a human's decision), lost silently, in precisely the crash the
    module's docstring says costs at most the row being written.
    """
    path = tmp_path / "log.csv"
    path.touch()  # the interrupted create
    assert path.stat().st_size == 0

    log = AppendLog(path, COLUMNS)
    log.append(["a", 1, "first"])
    log.append(["b", 2, "second"])

    header, rows = log.read()
    assert header == COLUMNS
    assert [row["id"] for row in rows] == ["a", "b"]  # neither decision was eaten


def _tear(log: AppendLog, text: str) -> None:
    """What a crash mid-append leaves: part of a row and no newline."""
    with open(log.path, "a", newline="", encoding="utf-8") as f:
        f.write(text)


def test_a_torn_row_costs_only_itself(tmp_path):
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    _tear(log, "b,2")
    log.append(["c", 3, "third"])
    log.append_many([["d", 4, "fourth"], ["e", 5, "fifth"]])
    assert [r["id"] for r in log.rows()] == ["a", "c", "d", "e"]


def test_a_torn_row_inside_an_open_quote_costs_only_itself(tmp_path):
    """A row cut inside a quoted field: the csv reader would otherwise read the next rows as
    part of that field, and at end of file it parses the torn row as a whole one."""
    log = _log(tmp_path)
    log.append(["a", 1, "line one\nline two"])
    _tear(log, 'b,2,"half a note\nstill in it')
    assert [r["id"] for r in log.rows()] == ["a"]  # the unterminated row is not read
    log.append(["c", 3, "third"])
    rows = log.rows()
    assert [r["id"] for r in rows] == ["a", "c"]
    assert rows[0]["note"] == "line one\nline two"


def test_a_torn_header_is_rewritten_as_a_new_file(tmp_path):
    log = _log(tmp_path)
    _tear(log, "id,va")
    log.append(["a", 1, "first"])
    assert log.path.read_text(encoding="utf-8-sig").splitlines()[0] == "id,value,note"
    assert [r["id"] for r in log.rows()] == ["a"]


def test_rows_after_a_torn_row_survive_a_migration(tmp_path):
    """``migrate`` rewrites the file from ``read``: a read that stopped at the torn row made the
    migration drop every later decision for good."""
    path = tmp_path / "log.csv"
    narrow = AppendLog(path, ["id", "value"])
    narrow.append(["a", 1])
    _tear(narrow, "b,")
    narrow.append(["c", 3])
    AppendLog(path, COLUMNS).append(["d", 4, "fourth"])
    rows = AppendLog(path, COLUMNS).rows()
    assert [r["id"] for r in rows] == ["a", "c", "d"]
    assert path.read_text(encoding="utf-8-sig").splitlines()[0] == "id,value,note"


def test_a_row_glued_onto_a_torn_one_does_not_hide_the_rest(tmp_path):
    """A file torn before appends cut torn tails holds a glued row mid-file: skip it, read on."""
    log = _log(tmp_path)
    log.append(["a", 1, "first"])
    with open(log.path, "a", newline="", encoding="utf-8") as f:
        f.write("b,2c,3,third\r\n")  # what the old append left: four fields
    log.append(["d", 4, "fourth"])
    assert [r["id"] for r in log.rows()] == ["a", "d"]


def test_a_torn_jsonl_line_is_not_glued_onto_the_next(tmp_path):
    """``history.jsonl`` (query and clustering runs) appends through ``append_line``."""
    import json

    from detail.storage.append_log import append_line

    path = tmp_path / "history.jsonl"
    append_line(path, json.dumps({"op": "a"}))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"op": "b"')  # killed mid-line
    append_line(path, json.dumps({"op": "c"}))
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    assert [json.loads(line)["op"] for line in lines] == ["a", "c"]
