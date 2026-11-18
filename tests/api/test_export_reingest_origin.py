"""Every label file DETAIL writes re-imports with its own origin; exported streams are aligned.

**The class (re-broke recorded class row 54).** *A label file DETAIL writes that its own
importer reads back as ground truth.* The catalogue ``ethogram`` target wrote
``start_frame,end_frame,label`` with no ``origin`` (the other writer of that shape,
``write_stream_csv``, had one), so re-importing a method's output made it ground truth and the
recording became eligible to score the method against itself. ``instances.csv`` had the same
hole and also spanned several recordings, and the ``streams`` export dropped a predicted
stream's origin the same way.

**How it hid.** Row 54's guard, ``tests/analysis/test_no_unfiltered_behavior_streams.py``,
covers *readers* of behavior streams. Nothing covered *writers*. This file does, data-driven
over the export registry, so a new target that writes a file the ``behavior_csv`` importer
accepts is checked the day it is registered.

**The rule.** A file the ``behavior_csv`` importer accepts carries ``origin`` (and is per
recording), so a re-import keeps where its labels came from. Written by
``behavior_output.write_stream_csv`` where the shape is a method's spans.

The second half guards the ``streams`` target's "aligned": with a **non-zero** offset (zero
cannot fail), ``video_t_ns`` is ``apply_alignment`` of each row and ``streams.csv`` reports the
stored offset, which it once read from a key that does not exist and reported as 0.
"""

import csv

import pytest

from detail.api import export_targets  # noqa: F401  (registers the catalogue)
from detail.api import exporters
from detail.api.exporters import ExportScope, ExportSubject
from detail.api.recording import Recording
from detail.importers.behavior import BehaviorCsvImporter
from detail.storage import analysis_runs as analysis_store
from detail.storage import behavior_output as bo
from detail.storage import streams as streams_store


def _rows(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def reimported_origin(path) -> str:
    """The origin ``add_stream`` would give ``path`` imported with no ``--origin``."""
    importer = BehaviorCsvImporter()
    result = importer.import_stream(path, path.parent, {})
    return Recording._resolve_stream_origin(importer, result, None)


def _label_files(result) -> list:
    return [
        result.dest / f
        for f in result.files
        if f.endswith(".csv") and BehaviorCsvImporter().can_import(result.dest / f)
    ]


def test_every_analysis_export_the_label_importer_accepts_reimports_as_predicted(
    authored_cohort, tmp_path
):
    """Over the registry: each runnable target, each file of it the importer would take."""
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    one = sorted(exporters.recordings_of(subject))[0]
    checked = []
    for target in exporters.available_targets(subject):
        # Narrowed to one recording, so a file spanning recordings is checked for its origin
        # rather than stopped at the one-recording rule below.
        result = exporters.write_export(
            subject,
            [target],
            tmp_path / target.id,
            scope=ExportScope(recording_ids=(one,)),
        )
        for path in _label_files(result):
            assert reimported_origin(path) == "predicted", f"{target.id}: {path.name}"
            checked.append(f"{target.id}/{path.name}")
    # Not vacuous: the ethogram and the instance table are both label files today.
    assert {c.split("/")[0] for c in checked} >= {"ethogram", "instances"}, checked


def test_the_instance_table_across_recordings_is_refused_not_merged(
    authored_cohort, tmp_path
):
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    result = exporters.write_export(
        subject, [exporters.target("analysis_run", "instances")], tmp_path
    )
    with pytest.raises(ValueError, match=r"holds rows for 2 recordings \(r1, r2\)"):
        reimported_origin(result.dest / "instances.csv")


def test_the_ethogram_carries_subject_and_origin(authored_cohort, tmp_path):
    subject = ExportSubject("analysis_run", authored_cohort.experiment.dir, "an_01")
    result = exporters.write_export(
        subject, [exporters.target("analysis_run", "ethogram")], tmp_path
    )
    rows = _rows(result.dest / "r1_ethogram.csv")
    assert list(rows[0]) == bo.STREAM_COLUMNS
    assert {r["origin"] for r in rows} == {"predicted"}


def test_the_command_writer_reimports_as_predicted(authored_cohort, tmp_path):
    """``export-query`` / ``export-temporal-clustering`` write through ``write_stream_csv``."""
    run_dir = analysis_store.analysis_dir(authored_cohort.experiment.dir, "an_01")
    out = bo.write_stream_csv(
        tmp_path / "s.csv", bo.read_behavior_output(run_dir), "r1"
    )
    assert reimported_origin(out) == "predicted"


def test_the_scan_would_catch_a_writer_without_origin(tmp_path):
    """Negative case: the ethogram's old header, re-imported, is ground truth."""
    old = tmp_path / "r1_ethogram.csv"
    old.write_text("start_frame,end_frame,label\n1,5,c01\n", encoding="utf-8")
    assert BehaviorCsvImporter().can_import(old)
    assert reimported_origin(old) == "ground_truth"


# -- the streams target ----------------------------------------------------------------- #
def _recording_with_streams(tmp_path):
    import cv2
    import numpy as np

    from detail.api.experiment import Experiment

    proj = Experiment.create("p", tmp_path)
    recording = proj.create_recording("m1")
    vid = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (16, 16))
    for _ in range(30):
        writer.write(np.zeros((16, 16, 3), np.uint8))
    writer.release()
    recording.add_video_part(vid)
    cues = tmp_path / "cues.csv"
    cues.write_text("seconds,label\n1.0,cue\n2.0,cue\n", encoding="utf-8")
    recording.add_stream(
        "cues", raw_path=cues, importer="events_csv", offset_ns=3_480_000_000
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("start_frame,end_frame,label\n1,3,groom\n", encoding="utf-8")
    recording.add_stream(
        "predicted_labels", raw_path=labels, importer="behavior_csv", origin="predicted"
    )
    return proj, recording


def test_exported_streams_carry_video_time_and_the_real_offset(tmp_path):
    proj, recording = _recording_with_streams(tmp_path)
    subject = ExportSubject("recording", proj.dir, recording.id)
    result = exporters.write_export(
        subject, [exporters.target("recording", "streams")], tmp_path / "out"
    )
    manifest = next(s for s in recording.streams if s["name"] == "cues")
    rows = _rows(result.dest / f"{recording.id}_cues.csv")
    assert [(int(r["t_ns"]), int(r["video_t_ns"])) for r in rows] == [
        (t, streams_store.apply_alignment(t, manifest))
        for t in (1_000_000_000, 2_000_000_000)
    ]
    assert int(rows[0]["video_t_ns"]) == 4_480_000_000

    index = {r["stream"]: r for r in _rows(result.dest / "streams.csv")}
    assert index["cues"]["offset_ns"] == "3480000000"
    assert index["cues"]["origin"] == ""  # not labels, so no origin
    assert index["cues"]["clock"] == "stream_seconds"
    assert index["predicted_labels"]["origin"] == "predicted"


def test_an_exported_predicted_stream_reimports_as_predicted(tmp_path):
    proj, recording = _recording_with_streams(tmp_path)
    subject = ExportSubject("recording", proj.dir, recording.id)
    result = exporters.write_export(
        subject, [exporters.target("recording", "streams")], tmp_path / "out"
    )
    path = result.dest / f"{recording.id}_predicted_labels.csv"
    assert BehaviorCsvImporter().can_import(path)
    assert reimported_origin(path) == "predicted"
