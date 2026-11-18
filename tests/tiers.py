"""The one table that says what a test costs and what it protects.

A tier is not a label a test author remembers to write; it is *derived* from where the
file lives, so a new test file inherits its directory's tier the day it is added and
nothing has to be updated for it to be selected correctly. Where a file contradicts its
directory — a real subprocess workload dropped into ``tests/storage/``, a source scan
that needs no Qt sitting in ``tests/gui/`` — the exception is written down here, once,
next to every other exception, rather than as a marker buried at the top of one file.

Two axes, deliberately independent:

**Cost** (``unit`` / ``gui`` / ``slow``) — exactly one per test, and it decides *when*
the test runs. ``unit`` is in-process with no Qt and no real workload; ``gui`` needs a
``QApplication``; ``slow`` spawns real processes, drives a real pipeline, encodes video,
or pays for a heavy numeric backend (numba JIT, HDBSCAN, UMAP, torch).

**Value** (``guard``) — orthogonal, and it decides *whether the tier can ever be
skipped*. A guard is a file in ``GUARD_FILES`` below, added by hand: a source scan, a
registry-parity sweep, a structural or anti-drift check, invisible to behavioural tests
because nothing is wrong at the line where the bug is written. Many of the classes in
``docs/recorded_classes.md`` are protected by one, but that table's Guard column names
the test that protects a class whatever its tier. Guards run on every invocation of the
gate, at every stage. They are
cheap by construction — a guard is never in the ``slow`` tier, and
``test_test_tiers.py`` enforces that.

``tests/conftest.py``'s collection hook is the only thing that *applies* this table, as
the ``unit`` / ``gui`` / ``slow`` / ``guard`` markers; ``.github/workflows/ci.yml`` runs
one job per selection. ``tests/test_test_tiers.py`` is the guard on both: it fails when a
test file is not covered, when the four selections do not partition the suite, when an
entry here points at a file that no longer exists, and when a ``unit``-tier file carries
the signature of a heavier tier without a written reason.

The signature scan is a lower bound, not the whole check — a file can be expensive
through an import it never names (a real backend model, a clustering kernel two modules
down). The other half is measurement: the weekly ``full`` job runs ``--durations=25``, and
a file that turns up there belongs in ``FILE_TIER`` as ``slow``. That is how
``test_trained_yolo`` (14.5s, fetches real weights), ``test_pose_clustering_module``
(54.7s) and ``test_finch`` (18.7s) were found sitting in the ``unit`` tier while the scan
was green.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The cost tiers. Exactly one applies to every test.
COST_TIERS = ("unit", "gui", "slow")

#: The value marker, orthogonal to cost.
GUARD = "guard"

# --------------------------------------------------------------------------- #
# Cost: the directory default, then the per-file exceptions.
# --------------------------------------------------------------------------- #

#: Directory (relative to the repo root) -> cost tier. The longest matching prefix wins,
#: so a nested directory can override its parent.
DIRECTORY_TIER = {
    "tests": "unit",
    "tests/executor": "slow",
    "tests/gui": "gui",
}

#: Files whose content contradicts their directory's tier. Each entry is a claim about
#: *why*, in the same spirit as ``test_no_orphan_functions.py``'s allow-list: an entry
#: for a file that has since been rewritten — or deleted — fails rather than sitting
#: here excusing nothing.
FILE_TIER = {
    # --- a source scan that lives beside the widget it guards ------------------------ #
    # It reads `gui/clip_strips.py` as text and builds no widget, so it runs in the tier
    # that runs on every change rather than only when the GUI paths-filter fires.
    "tests/gui/test_clip_strips_scan.py": "unit",
    # Both read `detail/gui` as text (ast) and build no widget.
    "tests/gui/test_dialog_close_scan.py": "unit",
    "tests/gui/test_menu_once_scan.py": "unit",
    # --- real subprocess workloads: a fresh interpreter that imports `detail` ------- #
    "tests/core/test_protocols.py": "slow",
    "tests/core/test_registries.py": "slow",
    "tests/models/test_sam3.py": "slow",
    "tests/gui/test_fatal_slots.py": "slow",
    "tests/checks/test_checks.py": "slow",
    # Real runs of the lock script contending for one lock, one of them killed mid-hold, ~4s:
    # a fresh interpreter each.
    "tests/scripts/test_with_suite_lock.py": "slow",
    # Real pytest runs in a fresh interpreter (plain and -n 2), and the watcher following one: ~15s.
    "tests/scripts/test_suite_progress.py": "slow",
    # Real pytest runs in a fresh interpreter, -n 2, each made to crash, hang or stall: ~25s.
    "tests/scripts/test_suite_watchdog.py": "slow",
    # Two real pytest runs in a fresh interpreter, and a git clone with a worktree: ~5s.
    "tests/scripts/test_flaky_report.py": "slow",
    # Runs the sh launcher against fake PATHs shaped like Windows and macOS: a subprocess each.
    "tests/scripts/test_run_python.py": "slow",
    # --- a real LocalExecutor run: worker subprocess, job.json, poll loop ----------- #
    "tests/api/test_experiment.py": "slow",
    "tests/pipeline/test_tracking.py": "slow",
    # Spawns a worker per test through `Experiment.run_export`, which the signature scan cannot
    # see — it names no heavy symbol of its own. ~5s; the measurement half of the rule, not the
    # scan half.
    "tests/api/test_export_run.py": "slow",
    # --- real video encode/decode through cv2 -------------------------------------- #
    "tests/api/test_clip_render.py": "slow",
    # Encodes a real mp4 and decodes it back to check *which* frames landed — the whole point
    # of a clip export, and not something a stub can answer.
    "tests/api/test_clip_export.py": "slow",
    "tests/api/test_keypoint_export.py": "slow",
    "tests/api/test_imports.py": "slow",
    "tests/api/test_recording.py": "slow",
    "tests/api/test_run_preview.py": "slow",
    "tests/pipeline/test_dataset_export.py": "slow",
    "tests/pipeline/test_detection.py": "slow",
    "tests/pipeline/test_detection_worker.py": "slow",
    "tests/pipeline/test_embedding.py": "slow",
    "tests/pipeline/test_embedding_worker.py": "slow",
    "tests/pipeline/test_labeling.py": "slow",
    "tests/pipeline/test_prompt_test.py": "slow",
    "tests/pipeline/test_render_preview.py": "slow",
    "tests/storage/test_export.py": "slow",
    "tests/storage/test_manifest.py": "slow",
    "tests/storage/test_video.py": "slow",
    "tests/storage/test_video_integrity.py": "slow",
    "tests/tools/test_quick_segment.py": "slow",
    "tests/test_cli.py": "slow",
    # --- heavy numeric backends: numba JIT, HDBSCAN, UMAP, sklearn on real matrices - #
    "tests/analysis/test_extraction_scaling.py": "slow",
    "tests/analysis/test_retrieval.py": "slow",
    "tests/analysis/test_segmentation.py": "slow",
    "tests/analysis/test_tw_finch.py": "slow",
    "tests/analysis/test_finch_abd.py": "slow",
    "tests/pipeline/test_query_eval.py": "slow",
    "tests/pipeline/test_query_module.py": "slow",
    # Three real fits — one per registered cluster method — each paying a UMAP layout it never
    # names, so the signature scan cannot see the cost. Measured at ~12 s. The merge tree's own
    # guard is `tests/analysis/test_cluster_tree.py`, which builds no fits and runs in ~1 s.
    "tests/cluster_methods/test_hierarchy.py": "slow",
    # --- matplotlib figure rendering ------------------------------------------------ #
    "tests/analysis/test_comparison_figures_shape.py": "slow",
    "tests/report/test_comparison_figures.py": "slow",
    "tests/report/test_figures.py": "slow",
    "tests/report/test_behavior_figures.py": "slow",
    # --- a real backend model, fetched and run --------------------------------------- #
    "tests/models/test_trained_yolo.py": "slow",
    "tests/models/test_ultralytics.py": "slow",
    # --- a real mkdocs site build ---------------------------------------------------- #
    "tests/test_docs_access.py": "slow",
    # --- lives under tests/gui but never builds a widget ---------------------------- #
    "tests/gui/test_ui_thread_blocking.py": "unit",
    "tests/gui/test_advanced_rebuild.py": "unit",
    "tests/gui/test_cuda_doctor.py": "unit",
    "tests/gui/test_readiness_guide.py": "unit",
    "tests/gui/test_import_cycles.py": "unit",
    "tests/gui/test_table_width.py": "unit",
    # --- lives under tests/executor but never spawns -------------------------------- #
    # `test_liveness_guard` monkeypatches `subprocess.run` and never lets one start; it is
    # the identity guard's decision table, which costs a millisecond and guards a re-submit
    # stacking onto a live process, so it belongs where it runs at every stage.
    "tests/executor/test_liveness_guard.py": "unit",
    # `test_coordinator_reap` drives `reap_coordinator` over stand-in exited handles from two
    # threads held at a barrier — no process is spawned; the race it pins is deterministic
    # and costs milliseconds, and it guards a GUI abort, so it runs at every stage.
    "tests/executor/test_coordinator_reap.py": "unit",
    # `test_process_liveness` does spawn — see its UNIT_EXEMPTIONS entry for why and what it costs.
    "tests/executor/test_process_liveness.py": "unit",
    "tests/executor/test_base.py": "unit",
    "tests/executor/test_rundir.py": "unit",
    "tests/executor/test_runners.py": "unit",
    "tests/executor/test_handles.py": "unit",
    "tests/executor/test_analysis_modules.py": "unit",
    "tests/executor/test_processing_modules.py": "unit",
    "tests/executor/test_batch_module.py": "unit",
    # Lives under tests/gui because the package it scans is `detail/gui`, but it parses
    # source and builds no widget — so it belongs in the tier that runs on every PR, not
    # the one a GUI-only paths filter skips.
    "tests/gui/test_frame_panes.py": "unit",
    # -- keypoints (§26): a PR's exemptions go here, alphabetical, with the reason --
}

# --------------------------------------------------------------------------- #
# Value: the guards. Many of docs/recorded_classes.md's guards live here, not all.
# --------------------------------------------------------------------------- #

#: Every file whose tests are source scans, registry-parity sweeps, structural checks,
#: or anti-drift checks. These run at every stage of the gate — they are the reason a
#: fast tier is trustworthy at all, and they are cheap enough that skipping them buys
#: nothing.
GUARD_FILES = frozenset(
    {
        # --- source scans over detail/ ------------------------------------------- #
        "tests/storage/test_path_serialization.py",
        # Every write of a stream's alignment goes through a clock gate (#30).
        "tests/storage/test_alignment_doors.py",
        # The clip window's strips are fed `Curve`s and spell no curve's name themselves.
        "tests/gui/test_clip_strips_scan.py",
        # One rule for a read-bound pool's thread count; `os.cpu_count()` nowhere else.
        "tests/core/test_cpu.py",
        # The montage's CLI opens on the API's defaults (the window reads them directly).
        "tests/api/test_montage_surface_parity.py",
        # A numpy channel swap is a strided copy ~45x slower than visualize's cv2 helpers.
        "tests/core/test_channel_swap_scan.py",
        # k-means built outside `_common.kmeans_level` runs uncapped OpenMP over OpenBLAS.
        "tests/cluster_methods/test_blas_safe_openmp.py",
        "tests/storage/test_zarr_seam.py",
        # Pins the preview cache's keys, layout and encode parameters, and the node cache's
        # version: a change there makes every user rebuild, and no fresh-cache test sees it.
        "tests/storage/test_cache_compat.py",
        # Counts parses of the machine's storage.yaml: a lookup under a per-item path that
        # re-parses it is correct at every line and slow by the file's size per item.
        "tests/storage/test_machine_config_reads.py",
        "tests/storage/test_measured_time.py",
        "tests/storage/test_id_ordering.py",
        # Every id family through its real mint and delete, and a scan for an allocator that
        # is not the shared never-reused one: a reused id is wrong only in an old record.
        "tests/storage/test_id_reuse.py",
        "tests/storage/test_atomic_writes.py",
        # Scans for a listing that reads its records bare, and plants the crash shape (a
        # zero-filled record) under every listing a page reads.
        "tests/storage/test_crash_consistency.py",
        # A file a user hands in is parsed through one reader per format, so a malformed
        # one is a message naming the file and line, not the parser's traceback (#33).
        "tests/storage/test_user_file_readers.py",
        # The CLI root shows a UserInputError as one line and lets a bug's traceback
        # through; one row per #33 finding with a CLI step.
        "tests/test_cli_user_errors.py",
        "tests/pipeline/test_atomic_write_seam.py",
        # Counts the syscalls a listing makes rather than scanning for a pattern — same
        # job: the cost model is invisible to every behavioural test.
        "tests/storage/test_dirscan.py",
        "tests/models/test_no_implicit_network.py",
        # A package probe in the app answers from startup; an editable install made while
        # it is open stays invisible until the .pth files are re-read (#506).
        "tests/core/test_installed.py",
        "tests/test_encoding_guard.py",
        # A Qt method restored by hand no longer binds; the next caller in the process fails.
        "tests/test_no_hand_patched_qt.py",
        # Every piece of machine-local state has a session floor, so a thread that outlives
        # its test cannot write the real preview cache, model cache or config.
        "tests/test_machine_state_isolation.py",
        # Every CLI / GUI call to an API method that takes `progress` passes one; the set is
        # read off the signatures, so a method that gains a callback is covered (#916).
        "tests/test_progress_callers.py",
        "tests/gui/test_ui_thread_blocking.py",
        "tests/gui/test_advanced_rebuild.py",
        "tests/gui/test_frame_panes.py",
        "tests/gui/test_nav_dispatch.py",
        "tests/gui/test_table_width.py",
        "tests/test_no_orphan_functions.py",
        # A name imported inside a function fails only when a user reaches that function.
        "tests/test_deferred_imports_resolve.py",
        # The scan over the scans: a source guard with no negative case cannot be told
        # from one whose pattern stopped matching (#373).
        "tests/test_guard_negative_cases.py",
        # A ratchet on real executor runs per module: the defect is "done again", not "slow",
        # and neither test count nor wall clock can see it (#534, R10).
        "tests/test_real_run_budget.py",
        # The suite lock and progress dir resolve to one place per machine on every branch;
        # a split lock serialises nothing and fails nothing (2026-10-23).
        "tests/scripts/test_machine_dir.py",
        # --- CLI output survives the locale it lands in --------------------------- #
        "tests/test_console.py",
        # --- registry / capability parity ----------------------------------------- #
        "tests/configs/test_record_completeness.py",
        "tests/configs/test_identity_completeness.py",
        # Walks the command tree against the config models: a surface that writes a default
        # in by hand is a second copy of a fact, and the copies drift in silence.
        "tests/configs/test_cli_defaults_match_models.py",
        "tests/configs/test_unit_conventions.py",
        "tests/configs/test_describe.py",
        # Walks every config model: a field naming a run, an example set or a recording list
        # is declared a selection, or a shipped baseline can carry it.
        "tests/configs/test_selection_scope.py",
        "tests/api/test_job_reattach.py",
        # Every surfaced analysis that selects an upstream run resolves it per recording by
        # settings, and no GUI picker outside its allow-list intersects ids (#547).
        "tests/api/test_analysis_selection.py",
        "tests/api/test_run_lifecycle_parity.py",
        "tests/api/test_authored_edit_ordering.py",
        "tests/api/test_wrapper_signatures.py",
        "tests/api/test_export_registry.py",
        "tests/pipeline/test_producer_parity.py",
        # Over every consumer of a per-part store's frame_idx: a frame and a time written together
        # agree through the recording's map on part 2 as on part 1 (#23).
        "tests/pipeline/test_part_frames.py",
        # Data-driven over the temporal registry: a per-timeline statistic measured over
        # many recordings' timelines pooled reads the jitter between them, not a frame
        # period, and a gap rule over pooled rows never cuts at a seam.
        "tests/cluster_methods/test_pooled_timelines.py",
        "tests/query_methods/test_registry.py",
        "tests/trackers/test_tracker_progress.py",
        "tests/trackers/test_tracker_resume.py",
        # Over TRACKERS: n_gaps counts frames the run looked at, at any stride and part (#32).
        "tests/trackers/test_tracker_gaps.py",
        "tests/test_registry_isolation.py",
        "tests/test_registry_prepopulation.py",
        "tests/test_surfaces.py",
        # --- the chart design language: every palette measured, in both themes ---- #
        "tests/core/test_chart_theme.py",
        # --- GUI structural guards (build widgets; still cheap) ------------------- #
        "tests/gui/test_active_run_openers.py",
        "tests/gui/test_advanced_form.py",
        "tests/gui/test_export_control.py",
        "tests/gui/test_launch_space.py",
        "tests/gui/test_export_reach.py",
        "tests/gui/test_reachability.py",
        "tests/gui/test_editing.py",
        # A run's label writes outlive the page that queued them; quitting asks (#24).
        "tests/gui/test_run_writes.py",
        "tests/gui/test_form_help_coverage.py",
        "tests/gui/test_frame_source.py",
        "tests/gui/test_failed_run_surface.py",
        "tests/gui/test_model_availability.py",
        "tests/gui/test_recording_names_in_pickers.py",
        "tests/gui/test_smoke_construct.py",
        "tests/gui/test_requirements.py",
        "tests/gui/test_spin_widgets.py",
        "tests/gui/test_module_names.py",
        "tests/gui/test_import_cycles.py",
        "tests/gui/test_dialog_space.py",
        "tests/gui/test_empty_states.py",
        "tests/gui/test_pydantic_form_defaults.py",
        "tests/executor/test_liveness_guard.py",
        "tests/executor/test_coordinator_reap.py",
        "tests/executor/test_process_liveness.py",
        "tests/gui/test_scroll.py",
        "tests/gui/test_dialog_close_scan.py",
        "tests/gui/test_menu_once_scan.py",
        "tests/gui/test_table_height.py",
        "tests/gui/test_pane_floors.py",
        "tests/gui/test_collapse.py",
        "tests/gui/test_window_state.py",
        "tests/gui/test_decode_priority.py",
        "tests/gui/test_strip_reuse.py",
        "tests/gui/test_audit_playback.py",
        "tests/gui/test_worker_lifetime.py",
        # A bounded wait raises naming what never happened, or its answer is used.
        "tests/gui/test_waits_are_checked.py",
        # No test inherits a modifier key or mouse button held by an earlier one.
        "tests/gui/test_input_state.py",
        # --- label leakage and the ground-truth boundary -------------------------- #
        "tests/pipeline/test_label_leakage.py",
        "tests/analysis/test_no_unfiltered_behavior_streams.py",
        "tests/analysis/test_behavior_stream_origin.py",
        # --- documentation / help anti-drift -------------------------------------- #
        "tests/test_docs_nav.py",
        "tests/test_docs_cli_examples.py",
        "tests/test_docs_voice.py",
        "tests/test_user_guide_legends.py",
        "tests/test_docs_generated_pages.py",
        "tests/test_docs_map.py",
        "tests/test_docs_modules_nav.py",
        "tests/test_design_plan_status.py",
        # The always-loaded file itself: a line ceiling, and a router that has to resolve. 122 of
        # the 143 commits that touched CLAUDE.md grew it and two ever shrank it, so the ratchet is
        # the mechanism and this is where it fails.
        "tests/test_rulebook.py",
        "tests/test_recorded_classes.py",
        # Its sibling: that one checks every citation in the table resolves, this one that the
        # table has not been torn in half by a merge. The first passed throughout the four
        # commits the second exists for.
        "tests/test_no_conflict_markers.py",
        # A wide-scoped fixture imported across test modules is rebuilt per importing file.
        "tests/test_fixture_imports.py",
        # One tool pinned in three places, and the only full-suite gate while CI is down.
        "tests/test_pre_commit_config.py",
        "tests/test_help.py",
        "tests/analysis/test_query_methods_docs.py",
        "tests/analysis/test_metrics_docs.py",
        "tests/analysis/test_measures_docs.py",
        "tests/transforms/test_constructor_docs.py",
        "tests/checks/test_commands.py",
        # --- progress / ETA truth -------------------------------------------------- #
        "tests/pipeline/test_progress_truth.py",
        # --- this table's own guard ------------------------------------------------- #
        "tests/test_test_tiers.py",
        # --- and its sibling's: which tests a *change* needs (tests/impact.py) ------- #
        # A guard, not merely a unit test: a selector that stops selecting fails by
        # running less and going green, which no behavioural test can see.
        "tests/test_impact.py",
    }
)

# --------------------------------------------------------------------------- #
# The content scan: does a file's source contradict the tier it was given?
# --------------------------------------------------------------------------- #

#: Call targets whose *presence as a call* means the file pays a heavier cost than the
#: ``unit`` tier promises. Matched on the AST, not the text, so a monkeypatch target
#: written as the string ``"detail.cli.subprocess.run"`` is not a hit.
_HEAVY_CALLS = {
    "subprocess.run": "spawns a process",
    "subprocess.Popen": "spawns a process",
    "subprocess.check_output": "spawns a process",
    "subprocess.check_call": "spawns a process",
    "os.fork": "spawns a process",
    "LocalExecutor": "drives the real executor",
    "ProcessPoolExecutor": "spawns a process pool",
    "multiprocessing.Process": "spawns a process",
    "cv2.VideoWriter": "encodes video",
}

#: Modules whose *import* means the file pays a heavier cost than ``unit`` promises.
_HEAVY_IMPORTS = {
    "PyQt6": "needs a QApplication",
    "torch": "imports torch",
    "sklearn": "imports scikit-learn",
    "numba": "imports numba (JIT compile)",
    "hdbscan": "imports hdbscan",
    "umap": "imports umap",
    "matplotlib": "renders figures",
}

#: A ``unit``-tier file that trips the scan needs an entry here, and the entry is a
#: claim about why the signature does not actually cost anything — not a way to opt out.
#: Each reason carries the **measured** cost of the whole file, because the tier's promise
#: is about time and an unmeasured "it is fine" is how the promise erodes. Checked against
#: the live scan, so an entry that stops being tripped fails.
UNIT_EXEMPTIONS = {
    "tests/api/test_export_reingest_origin.py": "cv2.VideoWriter writes one 30-frame 16x16 "
    "stub clip per streams-export test, only so a behavior stream has frames to resolve "
    "against; milliseconds, and the guard belongs with the unit tier it runs in",
    "tests/api/test_analysis.py": "Popen(['python', '-c', 'pass']) only, to fabricate a "
    "live PID for the reattach check — no workload, ~50 ms",
    "tests/api/test_job_reattach.py": "same: a `python -c pass` stand-in process is the "
    "only way to test that a surviving run reattaches",
    "tests/api/test_run_lifecycle_parity.py": "same PID stand-in, one per run kind",
    "tests/api/test_run_status_evidence.py": "the same `python -c pass` stand-in, for a "
    "worker pid that is certainly dead — no workload",
    "tests/api/test_export_estimates.py": "cv2.VideoWriter writes one 30-frame 64x64 stub "
    "clip so the video estimate times real frames — 6.3 s for the 34-test file on the macOS "
    "box, the rest of which is no-video estimates",
    "tests/core/test_resource_peaks.py": "two `python -c` stand-ins: one holds a 200 MB block "
    "so the sampler must count a child's memory, one proves nothing imports torch — no "
    "workload, ~1 s for the file",
    "tests/executor/test_process_liveness.py": "a `python -c sleep` stand-in per test — "
    "the liveness probe is only wrong against a real process, so nothing smaller tests it; "
    "no workload. 4.7 s for the file on Windows (three PowerShell CIM lookups ~0.6 s each, "
    "two repo-wide AST scans ~1 s each)",
    "tests/api/test_recording_identity.py": "cv2.VideoWriter writes 16x16 stub clips so a "
    "recording has a video part at all — 0.4 s for the file, no decode, no pipeline",
    "tests/models/test_dinov3.py": "torch is imported to build tiny fake tensors for the "
    "stand-in HF model; the module under test is torch-free at import (Tenet 13) and the "
    "file is 1.0 s",
    "tests/pipeline/test_keypoint_detect.py": "cv2.VideoWriter encodes five 40x40 frames so "
    "the stage has a part to walk (Tenet 11: it reads the video); milliseconds, 2.4 s for "
    "the file",
    "tests/analysis/test_cluster_metrics.py": "imports sklearn.metrics scorers only "
    "(AMI/ARI on tiny arrays) — no fit, no JIT",
    "tests/cluster_methods/test_blas_safe_openmp.py": "sklearn is imported to load the "
    "OpenMP/OpenBLAS pair the cap sits between, and fits one 600x8 k-means (0.04 s); the file "
    "is 3 s, 1.6 of it the repo scan and 1.3 the import",
    "tests/core/test_buildinfo.py": "`git init` / `commit` / `clone` / `fetch` in throwaway "
    "repos under tmp_path — the facts under test are git's answers, so a stand-in would test "
    "the stand-in. Local only (the 'origin' is a sibling directory), no workload; 2.1 s for "
    "the file on Windows",
    "tests/test_no_conflict_markers.py": "one `git ls-files` to enumerate tracked paths, then "
    "a regex over ~600 text files — no checkout, no workload. 0.12 s for the scan, 0.44 s for "
    "the file, and it has to be a guard: the thing it catches arrives with a merge, which is "
    "exactly the change that runs the narrowest tier selection",
    "tests/gui/test_import_cycles.py": "the one real spawn in this tier, and deliberate: "
    "it imports each detail.gui module first in a fresh interpreter, which is the only "
    "check with no import model to get wrong. Pooled across threads it is 6 s, and it has "
    "to be a guard rather than slow-tier — a gui-only PR does not run the slow job, and a "
    "gui-only PR is where an import cycle comes from",
}


def relpath(path: pathlib.Path) -> str:
    """Repo-relative posix path — the key every table here is written in."""
    return path.resolve().relative_to(REPO).as_posix()


def cost_tier(rel: str) -> str | None:
    """The cost tier for a repo-relative test path, or ``None`` if uncovered."""
    if rel in FILE_TIER:
        return FILE_TIER[rel]
    parent = pathlib.PurePosixPath(rel).parent
    for candidate in (parent, *parent.parents):
        tier = DIRECTORY_TIER.get(str(candidate))
        if tier is not None:
            return tier
    return None


def is_guard(rel: str) -> bool:
    """Whether a repo-relative test path is a guard."""
    return rel in GUARD_FILES


def heavy_signatures(source: str) -> dict[str, str]:
    """Heavier-tier signatures in a test file's source, as ``{signature: why}``.

    Uses the AST so a name that appears only inside a string — a ``monkeypatch.setattr``
    target such as ``"detail.cli.subprocess.run"`` — is not counted. A syntax error is
    reported as a signature of its own rather than swallowed; an unparseable test file
    is a problem either way.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"<syntax error>": str(exc)}

    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _HEAVY_IMPORTS:
                    found[root] = _HEAVY_IMPORTS[root]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in _HEAVY_IMPORTS:
                found[root] = _HEAVY_IMPORTS[root]
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in _HEAVY_CALLS:
                found[name] = _HEAVY_CALLS[name]
    return found


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}"
    return None


def touches_qt(source: str) -> bool:
    """Whether a test file's source touches Qt at all — its claim on the ``gui`` tier.

    The ``gui`` job is the one that *does not always run*: it is gated on a change under
    ``detail/gui/`` and friends. So a file tiered ``gui`` with no connection to Qt is a
    file skipped on most PRs for no reason, and that is what this answers. The reverse is
    deliberately fine — a ``unit``-tier file may import ``detail.gui.workers`` for a
    progress helper, because the ``unit`` job installs PyQt6 and always runs.

    Three signals, because the dependency is usually indirect. A Qt import or the ``qapp``
    / ``qtbot`` fixture is the obvious half; the load-bearing half is an import of any
    ``detail.gui.*`` module, which pulls in PyQt6 whether or not the test ever spells Qt
    out. Without that, a file that imports its widget *inside* the test body and takes ``qapp``
    reads as Qt-free (the clusterer-parity and benchmark-widget files did, before they were
    removed) — and a check written as a grep for ``"PyQt6"`` would move it into a tier with
    no ``QApplication`` to build it in.

    Imports are read off the AST (a name in a docstring is not an import); the fixtures
    are matched as text, since a fixture reaches a test as a bare parameter name.
    """
    if "qapp" in source or "qtbot" in source:
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name == "PyQt6" or name.startswith("PyQt6."):
                return True
            if name == "detail.gui" or name.startswith("detail.gui."):
                return True
    return False


def testpath_roots() -> list[pathlib.Path]:
    """The roots pytest collects from, read from ``testpaths`` in ``pyproject.toml``.

    Read rather than hardcoded so the table cannot fall behind the config: a root added
    to ``testpaths`` whose directory has no ``DIRECTORY_TIER`` entry is exactly the
    "selected by no tier, and nothing says so" failure, and deriving the roots here is
    what makes the coverage test able to see it.
    """
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8-sig"))
    paths = config["tool"]["pytest"]["ini_options"]["testpaths"]
    return [REPO / path for path in paths]


def test_files() -> list[pathlib.Path]:
    """Every collected test module, from the same roots as ``testpaths``."""
    files: list[pathlib.Path] = []
    for root in testpath_roots():
        if root.is_dir():
            files.extend(
                p for p in root.rglob("test_*.py") if "__pycache__" not in p.parts
            )
    return sorted(files)
