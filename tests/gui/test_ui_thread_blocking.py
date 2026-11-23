"""Fail-closed guard: no GUI widget calls a known-slow seam inline on the UI thread.

A slow seam (a video frame decode, a run-data preview read, a synchronous export)
called directly in a widget ``__init__`` or slot freezes the GUI with no feedback,
the recurring "black preview / dead button" bug. Such work must run off the UI
thread: through ``detail/gui/busy.py``'s ``run_busy`` / ``run_progress``, a
``*Worker``, or the scrubber's ``set_decoder``.

This scans ``detail/gui`` for the seams and fails on any call not on a line that also
names an off-thread wrapper, unless the ``(file, seam)`` pair is on ``ALLOWED`` (the
shrinking list of not-yet-migrated surfaces). New inline offenders fail immediately;
migrated ones must be removed from ``ALLOWED`` (a stale entry also fails). ``ALLOWED``
shrinks to empty as the migration lands.

**A second class, added after a freeze this scan structurally could not see.** *A cost
model that assumes local disk, applied to a workspace that can be configured onto a network
share.* The seams above are method **calls**, so a ``name(`` pattern finds them;
``.recordings`` / ``.processing_runs`` / ``.analysis_runs`` are **properties** — no
parentheses, nothing to match — and each is an O(recordings) or O(runs) sweep of reads that
are individually tiny. This file reasoned its way out of covering exactly that shape ("a
small YAML is not a slow seam"), which is true on an SSD and false sixty times over on SMB:
Home's ``_ExperimentCard`` was measured at **seconds of blocking I/O for a single card**
on a workspace held on a share, and it scales with the number of experiments. So the property
seams are scanned too, on the same terms — a line that also names an off-thread wrapper is
fine, everything else needs an ``ALLOWED`` entry.
"""

import re
from pathlib import Path

import detail

_GUI = Path(detail.__file__).parent / "gui"

# Method-call substrings that block the UI thread when run inline.
_SEAMS = (
    "get_frame(",
    "resolve_constructor(",
    "frame_timestamps(",
    "stream_events(",
    # A per-stream row count reads each stream's file to the end (#35: 3 s for one
    # 400,000-row sensor trace), so it is a seam like the read it replaced.
    "stream_row_counts(",
    "detection_preview(",
    "tracking_preview(",
    "input_image_preview(",
    "stream_preview(",
    "track_presence(",
    "frame_view(",
    "measure_run(",
    "track_continuity(",
    "check_examples(",
    ".export_ethogram(",  # dotted: skip the `def _export_ethogram(` method name
    "export_spatial_csv(",
    "export_metadata_csv(",
    "ensure_site_built(",
    # §20 calibration: the calls that do the work, not the ones that read its cache.
    # `load_results` reads a shard per searched unit and forces the extraction;
    # `read_behavior_output` / `.artifact(` grow with the artifact; `source_crop` is a video
    # seek per thumbnail.
    #
    # Deliberately **not** here: `frequency_curve(`, `candidates_at(` and
    # `score_distribution(`. Each is a bisect slice over the candidates `load_results` already
    # extracted, and the calibration view calls them on every drag *because* they are cheap —
    # listing them would mean allow-listing them straight back. Guarding `load_results` is
    # what makes them cheap, so that is the one to guard. `read_query_state(` is out for the
    # same reason a `read_recording` would be: a small YAML is not a slow seam.
    # Sweeps every run family of the open experiment. Added when the quit guard stopped
    # reading `.processing_runs` / `.analysis_runs` itself and started asking `api.active`:
    # the *cost* did not move, only the spelling, and a property seam that becomes a function
    # call in another package drops out of this scan silently. Guard the call instead.
    "active_ids(",
    "load_results(",
    # §3 video previews: what a run is missing is a stat (and, for a part never seen here, a
    # fingerprint read) per part, and a recording's tracking run is a YAML read per recording —
    # O(recordings) on what may be a share. The clustering page asks both off the UI thread.
    "missing_previews(",
    "clip_tiles.missing(",
    "clip_tiles.tracking_runs(",
    # §22 Time in session: each recording's first and last frame time, from its timestamp map
    # (a CSV parse per part, the first time) — O(recordings), read with the page off the UI
    # thread.
    "session_spans(",
    # §12 *Test cut-off…*: loads the teacher's model and labels a dozen frames.
    "test_keypoints(",
    "run_teacher_test(",
    "read_behavior_output(",
    ".artifact(",  # dotted: skip the `def artifact(` definition
    "source_crop(",
    # The masked forms are separate strings: `"source_crop("` does not match
    # `source_crop_masked(`, so the strip's thumbnail and playback decodes were outside the
    # scan the day they were written. `source_crops_masked(` is the span form — one open and
    # one seek for a whole clip, but still seconds of decode.
    "source_crop_masked(",
    "source_crops_masked(",
    # §22 clip window. One frame with every mask, the keypoint run the embedding read (its
    # whole series per part, the first time a part is asked for), the strips' curves over a
    # window of the kinematics sidecar; ranking a node's members on demand loads the fit
    # (megabytes) and ranks every member; an enlarged pane's export decodes and draws every
    # frame it writes.
    "clip_frame(",
    "clip_views(",
    "clip_curves(",
    "clip_has_points(",
    "typical_members(",
    "export_clip_pane(",
    # §22 clip montage. Finding a montage's clips ranks every member of a node; preparing
    # reads every clip's frame times and boxes; the exports decode and draw every frame they
    # write, a clip list reads every clip's frame span.
    "montage_members(",
    "clip_boxes(",
    "export_clip_gifs(",
    "export_clip_list(",
    # §24 comparisons. `rank_by_difference` is a permutation test -- 200 shuffles of every
    # observation -- and it ran three times per page open because three widgets each re-read
    # `stats.csv` and re-ranked it, about a second to open the page on a small cohort.
    # `_read_comparison(` is the one seam that does both, so guarding
    # it covers the read as well. `drift(` shells out to `git`.
    "rank_by_difference(",
    "_read_comparison(",
    # §21's repair: re-derives the whole §19 artifact — a shard read and a frame-view zarr per
    # analyzed unit, slow on a share.
    ".reproject(",  # dotted: skip the `def reproject(` definition
    ".drift(",  # dotted: skip the `def drift(` definition
    # `core/buildinfo` shells out to `git` — several processes for the build, a `fetch` over
    # the network for the update check. Module-qualified to skip the popover's
    # `def _check_for_updates(` slot and any other `history(`.
    # §19 audit reads. Each reads the run's artifact and its audit log — a file that may sit on
    # a network share. The one-by-one audit's draw and judge were here too, until that audit
    # was removed; it wrote an answer inline on every key press until the review that added
    # these.
    "audit_by_cluster(",
    "audit_decisions(",
    "audit_labels(",
    "audit_precision(",
    # A query run's review reads its search the first time: as slow as the shard load.
    "review_by_recording(",
    "review_clips(",
    "buildinfo.build_info(",
    "buildinfo.history(",
    "buildinfo.check_for_updates(",
    "buildinfo.head_sha(",
    "provenance.code_version(",
    # §26 example sets: a sample decodes frames (seconds), an add decodes one, and a marks
    # write rewrites the set's YAML and hashes its images — the labeling screen saves on
    # every click, so all three run through a worker.
    "sample_example_frames(",
    "add_example_frame(",
    "set_example_marks(",
    # §26 keypoints: every pose instance a detect run placed in one part — the whole part's
    # zarr, each column. The run viewer reads a frame at a time (`KeypointDetectRun.at`); a
    # surface wanting a part must read it off the UI thread.
    "keypoint_instances(",
    # §12 *Keypoint models in the catalog*: a restore unzips weights and may add a skeleton
    # to the library; an import copies and hashes them; reading a checkpoint loads it
    # through Ultralytics to learn its task, point count and flip_idx. Seconds, each.
    "import_model_archive(",
    "restore_model_archive(",
    ".import_model(",  # dotted: skip the `def _import_model(` slot
    "read_checkpoint(",
    # The clustering montage's tile pictures: a preview-cache read, crop, shrink and mask draw
    # per tile per tick. `LiveMontage` asks its worker; nothing calls a source inline.
    "source.frame(",
    # §22 BROWSE's cluster lane: a node's stretches through `NodeView.labelled_runs`, a Python
    # walk over every leaf run of the fit. INTERPRET asks for it on every node click, and it ran
    # inline on each one while BROWSE was a hidden tab.
    "labelled_runs(",
)

# Property seams: an O(recordings) / O(runs) sweep spelled as an attribute, so nothing
# above can match it. `\b` keeps `.recordings_dir` and `.recording_ids` out — the latter is
# the *listing*, one enumeration, which is what a caller wanting a count should reach for.
_PROPERTY_SEAMS = {
    ".recordings": re.compile(r"\.recordings\b"),
    ".processing_runs": re.compile(r"\.processing_runs\b"),
    ".analysis_runs": re.compile(r"\.analysis_runs\b"),
}

# A seam call on a line that also names one of these is already off-thread.
_WRAPPERS = (
    "run_busy",
    "run_progress",
    "set_decoder",
    "lambda",
    "# off-thread",
    "BackgroundCall",
    "StageProgressWorker",
    "_CallTask",
    "_ExportWorker",
    "def compute",  # a decoder compute_fn run via set_decoder
)

# (file relative to gui/, seam) still called inline on the UI thread. Every preview,
# editor, export, and stream dialog is migrated, and so is the per-stream row count on the
# recording-detail streams table: "typically small streams" was the reason it stayed inline,
# and a 400,000-row trace froze the page for 3 s on every build (#35). It is a
# `BackgroundCall` filling the Rows column now, so no `stream_events(` allowance is left.
ALLOWED: set[tuple[str, str]] = {
    # --- the property seams, still built inline (see the second class above) ------- #
    # Home's card is migrated (`_card_facts` runs through a BackgroundCall), the experiment
    # overview always was, and the **rail** followed in #316 — `sections.py` now reads a
    # section's children in a `BackgroundCall` when that section opens, so its three entries
    # are gone from here. What remains is the launch/selection surfaces. Each entry is one
    # file's worth of sites; it shrinks as the migration lands, and a surface that stops
    # offending fails as a stale entry.
    # `closeEvent` asks which runs are in flight before deciding whether to prompt. It is the
    # one moment a blocking read is the lesser evil: the alternative is closing the window and
    # *then* discovering there was a distill run to offer to cancel. Kept declared rather than
    # migrated, and kept visible rather than dropped when the read moved into `api.active`.
    ("app.py", "active_ids("),
    ("comparison.py", ".recordings"),
    ("fit.py", ".processing_runs"),
    ("fit.py", ".recordings"),
    ("launch.py", ".recordings"),
    # `pages.py`'s `.analysis_runs` is gone: the analysis index reads its rows in a
    # `BackgroundCall` too. `.recordings` and `.processing_runs` remain for the surfaces that
    # still build inline — the metadata table, the runs index, the bundle-import validate and
    # two detail dialogs.
    ("pages.py", ".processing_runs"),
    ("pages.py", ".recordings"),
    ("start_from.py", ".analysis_runs"),
    ("start_from.py", ".processing_runs"),
    # A query run's AUDIT tab draws and writes inline: its batches are small and it predates
    # the audit seams. Clustering's AUDIT (`label_check.py`) does both off the UI thread.
}


def _seams_in(line: str) -> set[str]:
    """The seams a source line calls inline — empty if it is a comment or already wrapped.

    Factored out of the scan so the scan itself can be shown to catch something: a source
    scan that matches nothing passes vacuously forever, and this file had no such case.
    """
    stripped = line.strip()
    if stripped.startswith("#"):
        return set()
    # A definition is not a call site. `.artifact(` dodged this with a leading dot; a
    # module-level helper named after the work it does cannot, so say it once here.
    if stripped.startswith("def ") or stripped.startswith("async def "):
        return set()
    if any(wrapper in stripped for wrapper in _WRAPPERS):
        return set()
    found = {seam for seam in _SEAMS if seam in stripped}
    found |= {seam for seam, rx in _PROPERTY_SEAMS.items() if rx.search(stripped)}
    return found


def _violations() -> set[tuple[str, str]]:
    found = set()
    for path in _GUI.rglob("*.py"):
        rel = path.relative_to(_GUI).as_posix()
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            for seam in _seams_in(raw):
                found.add((rel, seam))
    return found


def test_no_unwrapped_slow_seam_outside_allowlist():
    new = _violations() - ALLOWED
    assert not new, (
        "slow seam called inline on the UI thread — wrap the call in "
        f"run_busy / run_progress (or add # off-thread if already off-thread): {new}"
    )


def test_the_scan_would_actually_catch_one():
    """Guard the guard: a source scan that matches nothing passes vacuously forever."""
    assert _seams_in("        rgb = preview.source_crop(sid, emb, 0, 12, 150)") == {
        "source_crop("
    }
    # ...and it does not fire on a call that is already off the UI thread, or on prose.
    assert (
        _seams_in("        run_busy(self, lambda: run.load_results(bid), self._show)")
        == set()
    )
    assert _seams_in("    # source_crop( is named in this comment, not called") == set()
    # The two spellings the streams table's Rows column has had: the old inline read, and
    # the count that replaced it, which is just as slow if called inline.
    assert _seams_in(
        '                n_events = len(self._recording.stream_events(s["name"]))'
    ) == {"stream_events("}
    assert _seams_in("        counts = self._recording.stream_row_counts()") == {
        "stream_row_counts("
    }
    assert (
        _seams_in(
            "        call = BackgroundCall(lambda: recording.stream_row_counts(), "
            "receiver=table)"
        )
        == set()
    )
    # The spelling the lane read had while it ran inline on every INTERPRET node click.
    assert _seams_in(
        "            for run in view.labelled_runs({str(int(node)): name}, apply_edits=False):"
    ) == {"labelled_runs("}


def test_the_scan_would_catch_a_property_seam():
    """The half a `name(` pattern is blind to — no parentheses, nothing to match on."""
    assert _seams_in("        n = len(self._experiment.recordings)") == {".recordings"}
    assert _seams_in("        for run in experiment.processing_runs:") == {
        ".processing_runs"
    }
    # ...off-thread on the line, or handed to a worker, and it is not an offender.
    assert _seams_in("    runs = experiment.processing_runs  # off-thread") == set()
    assert (
        _seams_in(
            "        BackgroundCall(lambda: [_card_facts(e) for e in experiments])"
        )
        == set()
    )
    # ...and the listing, which is one enumeration rather than a read per recording, is
    # deliberately not a seam: it is what a caller wanting a *count* should reach for.
    assert _seams_in("        n = len(experiment.recording_ids)") == set()
    assert _seams_in("        root = experiment_store.recordings_dir(d)") == set()


def test_allowlist_has_no_stale_entries():
    stale = ALLOWED - _violations()
    assert (
        not stale
    ), f"ALLOWED entries are no longer inline offenders — remove them: {stale}"
