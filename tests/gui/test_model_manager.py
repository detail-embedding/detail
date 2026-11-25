"""Model manager page + machine-global model detail + import dialog (§12).

One recorded class is guarded here: **a table rebuilt when a background result lands drops
what the user did to the old one.** The Hugging Face table is built at once with "checking…"
statuses and rebuilt when the off-thread install scan returns, a second or two later; that
rebuild replaced the view, so a row picked or a column sorted in the meantime was lost and
Check access / Install greyed out under the user's hand (#35). The rebuild now keeps the picked
model (by id and variant, not by row, because the rows re-band) and the user's sort, while an
unsorted table still takes the documented order (suggested, installed, the rest).
"""

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QLabel  # noqa: E402

from detail.api import models as api_models  # noqa: E402
from detail.core.registries import TRAINED_LOADERS  # noqa: E402
from detail.gui import pages  # noqa: E402
from detail.storage import locations  # noqa: E402

# Reuse the imported-catalog stub loader (introspect reads no real checkpoint).
from tests.models.test_imported import _StubLoader  # noqa: E402


def _import_stub_model(tmp_path, monkeypatch, name="operant_v2", **meta):
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    w = tmp_path / "best.pt"
    w.write_bytes(b"w")
    return api_models.import_model(w, family="stubfam", name=name, **meta)


def _distilled_model(tmp_path):
    from detail.models import distilled
    from detail.storage import experiment as experiment_store
    from detail.storage import models as models_store

    proj = experiment_store.create_experiment(tmp_path, "rp")
    mdir = models_store.model_dir(proj, "model_001")
    (mdir / "run" / "weights").mkdir(parents=True)
    (mdir / "run" / "weights" / "best.pt").write_bytes(b"w")
    models_store.write_model_yaml(
        mdir,
        {
            "id": "model_001",
            "name": "operant",
            "model_family": "yolo",
            "task": "segment",
            "classes": [{"id": 0, "name": "mouse"}],
            "weights": {"best": "run/weights/best.pt"},
            "metrics": {"box_map50": 0.9},
            "base_model": "yolo11s-seg.pt",
            "trainer_id": "ultralytics_yolo",
            "dataset_id": "ds_1",
            "status": "complete",
        },
    )
    return distilled.register_distilled_model(
        proj, "model_001", source_experiment_name="rp", source_experiment_id="rp"
    )


def test_your_models_table_lists_imported(qapp, tmp_path, monkeypatch):
    mid = _import_stub_model(tmp_path, monkeypatch, made_by="labmate")
    page = pages.ModelManagerPage(None, lambda m: None, lambda k: None, lambda h: None)
    assert page._models and page._models[0].id == mid
    assert page._your.view.model().rowCount() == 1  # one row for the imported model


def test_detail_renders_metadata_and_provenance(qapp, tmp_path):
    gid = _distilled_model(tmp_path)
    detail = pages.GlobalModelDetailPage(gid)
    texts = " ".join(label.text() for label in detail.findChildren(QLabel))
    assert "rp" in texts  # from project(s) — the source experiment
    assert "yolo11s-seg.pt" in texts  # build-history base model
    assert "mouse" in texts  # classes


def test_a_models_preview_opens_full_size(qapp, tmp_path, monkeypatch):
    """The surface this was asked for: whether a distilled model's own segmentation is right
    is not a question a 360px thumbnail answers.

    Through the page, not the widget: the preview is built inside `_build`, so a test that
    constructed an `EnlargeableThumb` itself would prove nothing about this page having one.
    """
    import numpy as np
    from PyQt6.QtGui import QImage

    from detail.gui.full_frame import EnlargeableThumb

    preview = tmp_path / "preview.png"
    QImage(
        np.zeros((40, 60, 3), np.uint8).tobytes(),
        60,
        40,
        180,
        QImage.Format.Format_RGB888,
    ).save(str(preview))
    mid = _import_stub_model(tmp_path, monkeypatch, preview=preview)
    detail = pages.GlobalModelDetailPage(mid)

    thumbs = [t for t in detail.findChildren(EnlargeableThumb) if t.toolTip()]
    assert thumbs, "the model's preview cannot be opened full size"
    image = thumbs[0]._render()  # the recipe the double-click runs, off the UI thread
    assert image.width() == 60 and image.height() == 40, "the render is a scaled copy"


def test_a_model_with_no_preview_advertises_nothing(qapp, tmp_path, monkeypatch):
    """Guard the guard above: the cue is derived from the wiring, so a model with no preview
    image must not offer to enlarge one."""
    from detail.gui.full_frame import EnlargeableThumb

    mid = _import_stub_model(tmp_path, monkeypatch)
    detail = pages.GlobalModelDetailPage(mid)
    assert all(t.toolTip() == "" for t in detail.findChildren(EnlargeableThumb))


def test_detail_not_found_builds(qapp):
    detail = pages.GlobalModelDetailPage("no_such_model")
    texts = " ".join(label.text() for label in detail.findChildren(QLabel))
    assert "not found" in texts  # a missing model renders a note, not a crash


def test_import_dialog_values(qapp):
    dialog = pages.ImportModelDialog()
    dialog._weights = Path("x/best.pt")  # normally set by the file picker
    dialog._name.setText("m")
    dialog._made_by.setText("kev")
    dialog._description.setText("d")
    weights, meta = dialog.values()
    assert weights == Path("x/best.pt")
    assert (
        meta["name"] == "m" and meta["made_by"] == "kev" and meta["description"] == "d"
    )
    assert meta["notes"] is None and meta["preview"] is None
    assert meta["skeleton"] is None  # no skeleton step for a file not read as pose


# --------------------------------------------------------------------------- #
# Keypoint models in the catalog (§12): the Kind column, Restore from zip, and the
# import dialog's skeleton step. Only the two Ultralytics readers are stubbed — the
# catalog and the skeleton library are real, isolated per test.
# --------------------------------------------------------------------------- #
def _stub_checkpoint_readers(monkeypatch, task="pose", shape=(3, [0, 1, 2])):
    """Make the probe loader read ``task`` and the pose reader ``shape`` = (K, flip_idx)."""
    from detail.api._runner_modules import PIPELINE_RUNNERS
    from detail.executor.runners import import_runner_modules
    from detail.models import imported as imported_store
    from detail.models import yolo_pose
    from tests.models.test_imported import _PoseStubLoader

    import_runner_modules(PIPELINE_RUNNERS)  # register first, so the stubs stay put
    loader = _PoseStubLoader if task == "pose" else _StubLoader
    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, loader)
    monkeypatch.setattr(
        yolo_pose.TrainedYoloPose,
        "introspect",
        staticmethod(lambda weights_path: shape),
    )


def _import_pose_model(tmp_path, monkeypatch, name="rat_pose"):
    _stub_checkpoint_readers(monkeypatch)
    w = tmp_path / f"{name}.pt"
    w.write_bytes(b"w")
    return api_models.import_model(w, skeleton="midline_3", name=name)


def _cells(table, column):
    model = table.view.model()
    return [model.index(r, column).data() for r in range(model.rowCount())]


def test_the_kind_column_tells_detectors_from_keypoint_models(
    qapp, tmp_path, monkeypatch
):
    _import_stub_model(tmp_path, monkeypatch, name="operant_v2")
    _import_pose_model(tmp_path, monkeypatch)
    page = pages.ModelManagerPage(None, lambda m: None, lambda k: None, lambda h: None)
    kind = pages.ModelManagerPage._YOUR_COLUMNS.index("Kind")
    names = _cells(page._your, 0)
    kinds = dict(zip(names, _cells(page._your, kind)))
    assert kinds == {"operant_v2": "detector", "rat_pose": "keypoints"}


def test_restore_from_zip_brings_a_keypoint_model_back_and_names_its_skeleton(
    qapp, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QFileDialog, QPushButton

    from detail.storage import skeleton_library
    from tests.gui.conftest import drain_until

    _stub_checkpoint_readers(monkeypatch)
    w = tmp_path / "rat_pose.pt"
    w.write_bytes(b"w")
    line = tmp_path / "line.yaml"
    line.write_text(
        "name: rat line\nskeleton:\n  names: [head, body, tail]\n"
        "  edges: [[head, body]]\n",
        encoding="utf-8",
    )
    api_models.import_model(w, skeleton=str(line), name="rat_pose")
    archive = tmp_path / "rat_pose.zip"
    api_models.export_model("rat_pose", archive)
    api_models.remove_imported_model("rat_pose")
    for skeleton_id in skeleton_library.discover():  # a machine that never saw it
        skeleton_library.remove(skeleton_id)

    page = pages.ModelManagerPage(None, lambda m: None, lambda k: None, lambda h: None)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), "")
    )
    button = next(
        b for b in page.findChildren(QPushButton) if b.text() == "Restore from zip"
    )
    button.click()
    assert drain_until(qapp, lambda: page._restored.text() != "")

    entry = skeleton_library.list_records()[0]
    assert entry["name"] == "rat line"
    assert (
        f"added to the Keypoints manager as rat line ({entry['id']})"
        in page._restored.text()
    )
    assert "rat_pose" in _cells(page._your, 0)  # the table was refreshed


def test_restoring_a_model_whose_skeleton_is_known_says_it_matched(
    qapp, tmp_path, monkeypatch
):
    from tests.gui.conftest import drain_until

    _import_pose_model(tmp_path, monkeypatch)
    archive = tmp_path / "rat_pose.zip"
    api_models.export_model("rat_pose", archive)
    api_models.remove_imported_model("rat_pose")
    page = pages.ModelManagerPage(None, lambda m: None, lambda k: None, lambda h: None)
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(archive), "")
    )
    page._restore_model()
    assert drain_until(qapp, lambda: page._restored.text() != "")
    assert "matches Midline (3 points) (midline_3)" in page._restored.text()


def test_a_failed_restore_is_shown_not_swallowed(qapp, tmp_path, monkeypatch):
    from tests.gui.conftest import drain_until

    bad = tmp_path / "not_a_model.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("a/x.txt", "x")
        zf.writestr("b/y.txt", "y")
    shown = []
    monkeypatch.setattr(
        "detail.gui.busy.QMessageBox.critical", lambda *a: shown.append(a[2])
    )
    page = pages.ModelManagerPage(None, lambda m: None, lambda k: None, lambda h: None)
    monkeypatch.setattr(
        pages.QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad), "")
    )
    page._restore_model()
    assert drain_until(qapp, lambda: bool(shown))
    assert "must hold exactly one model folder, and holds a, b" in shown[0]
    assert page._restored.isHidden()


def test_the_import_dialog_asks_no_family(qapp):
    dialog = pages.ImportModelDialog()
    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert "Family" not in labels
    assert not hasattr(dialog, "_family")


def _read(qapp, dialog, weights):
    from tests.gui.conftest import drain_until

    dialog.set_weights(weights)
    assert drain_until(qapp, lambda: dialog._checkpoint is not None)


def test_a_detection_checkpoint_has_no_skeleton_step(qapp, tmp_path, monkeypatch):
    _stub_checkpoint_readers(monkeypatch, task="segment")
    w = tmp_path / "mouse.pt"
    w.write_bytes(b"w")
    dialog = pages.ImportModelDialog()
    _read(qapp, dialog, w)
    assert dialog._skeleton.isHidden() and dialog._check.isHidden()
    assert "Detector" in dialog._kind.text() and "mouse" in dialog._kind.text()
    assert dialog._ok.isEnabled()
    assert dialog.values()[1]["skeleton"] is None


def test_a_pose_checkpoint_waits_for_a_skeleton_that_fits(qapp, tmp_path, monkeypatch):
    _stub_checkpoint_readers(monkeypatch, shape=(3, [0, 1, 2]))
    w = tmp_path / "rat.pt"
    w.write_bytes(b"w")
    dialog = pages.ImportModelDialog()
    _read(qapp, dialog, w)
    assert not dialog._skeleton.isHidden(), "a pose checkpoint shows the skeleton step"
    assert "Keypoints" in dialog._kind.text() and "3 points" in dialog._kind.text()
    assert not dialog._ok.isEnabled(), "Import waits until a skeleton is chosen"

    dialog._skeleton.setCurrentIndex(dialog._skeleton.findData("midline_2"))
    assert "places 3 points but the skeleton has 2" in dialog._check.text()
    assert not dialog._ok.isEnabled()

    dialog._skeleton.setCurrentIndex(dialog._skeleton.findData("midline_3"))
    assert dialog._check.text().startswith("✓")
    assert "wrong way round" in dialog._check.text()  # the check's limit, said plainly
    assert dialog._ok.isEnabled()
    assert dialog.values()[1]["skeleton"] == "midline_3"


def test_a_skeleton_from_a_file_is_checked_too(qapp, tmp_path, monkeypatch):
    _stub_checkpoint_readers(monkeypatch, shape=(3, None))  # no flip_idx recorded
    w = tmp_path / "rat.pt"
    w.write_bytes(b"w")
    paired = tmp_path / "paired.yaml"
    paired.write_text(
        "names: [left_ear, right_ear, tail]\npairs: [[left_ear, right_ear]]\n",
        encoding="utf-8",
    )
    dialog = pages.ImportModelDialog()
    _read(qapp, dialog, w)
    dialog.set_skeleton_file(paired)
    assert "no flip_idx" in dialog._check.text()
    assert not dialog._ok.isEnabled()
    assert dialog.values()[1]["skeleton"] == str(paired)


def test_an_unreadable_checkpoint_is_said_inline(qapp, tmp_path, monkeypatch):
    from detail.models import imported as imported_store
    from tests.gui.conftest import drain_until

    class _Broken(_StubLoader):
        @staticmethod
        def introspect(weights_path):
            raise ValueError("no class names found")

    _stub_checkpoint_readers(monkeypatch)
    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, _Broken)
    w = tmp_path / "junk.pt"
    w.write_bytes(b"w")
    dialog = pages.ImportModelDialog()
    dialog.set_weights(w)
    assert drain_until(qapp, lambda: "Cannot import" in dialog._kind.text())
    assert "no class names found" in dialog._kind.text()
    assert not dialog._ok.isEnabled()


@pytest.mark.parametrize(
    "name, words",
    [
        ("notes.txt", "notes.txt is not a YOLO .pt checkpoint;"),
        (
            "junk.pt",
            "junk.pt is not a YOLO .pt checkpoint Ultralytics 8.4.215 can read",
        ),
    ],
    ids=["not-a-pt", "unreadable-pt"],
)
def test_a_file_that_is_not_a_checkpoint_is_named_inline(
    name, words, qapp, tmp_path, monkeypatch
):
    """Through the real readers (#33 finding 7): Ultralytics stands in raising what 8.4.215
    raised on a text file renamed ``.pt``, so the line shows the reader's sentence and not
    Ultralytics' own text, nor a missing-module error where it is not installed."""
    import sys
    import types

    from tests.gui.conftest import drain_until

    def _yolo(path):
        raise TypeError(f"ERROR {path} is not a loadable checkpoint")

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _yolo
    fake.__version__ = "8.4.215"
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    w = tmp_path / name
    w.write_bytes(b"notes")
    dialog = pages.ImportModelDialog()
    dialog.set_weights(w)
    assert drain_until(qapp, lambda: "Cannot import" in dialog._kind.text())
    assert dialog._kind.text().startswith(f"Cannot import this file: {words}")
    assert "loadable" not in dialog._kind.text()
    assert not dialog._ok.isEnabled()


def test_a_keypoint_models_page_shows_its_skeleton(qapp, tmp_path, monkeypatch):
    mid = _import_pose_model(tmp_path, monkeypatch)
    detail = pages.GlobalModelDetailPage(mid)
    texts = " ".join(label.text() for label in detail.findChildren(QLabel))
    assert "keypoints" in texts  # the Kind row
    assert "head, midpoint, tail" in texts  # its points
    assert "Midline (3 points) (midline_3)" in texts  # its library entry


def test_a_detectors_page_shows_no_skeleton(qapp, tmp_path, monkeypatch):
    mid = _import_stub_model(tmp_path, monkeypatch)
    detail = pages.GlobalModelDetailPage(mid)
    texts = [label.text() for label in detail.findChildren(QLabel)]
    assert "Skeleton" not in texts and "detector" in texts


def test_shell_opens_detail_from_manager(qapp, tmp_path, monkeypatch):
    # Verify through the real entry point: models mode → open a your-model row → its detail.
    from detail.gui.app import MainWindow

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    _import_stub_model(tmp_path, monkeypatch, made_by="k")
    window = MainWindow()
    window._switch_mode("models")
    page = window._content.currentWidget()
    assert isinstance(page, pages.ModelManagerPage)
    page._open_row(page._your.view.model().index(0, 0))  # the double-click target
    assert isinstance(window._content.currentWidget(), pages.GlobalModelDetailPage)


# --------------------------------------------------------------------------- #
# An experiment's trained models: the Models page and the manager's experiment table
# --------------------------------------------------------------------------- #
def _experiment_with_models(tmp_path, n_models=2, *, distilled=("model_002",)):
    """An experiment with `n_models` trained models, the ones in `distilled` made by a run."""
    from detail.api.experiment import Experiment
    from detail.storage import distill_runs as distill_store
    from detail.storage import models as models_store

    experiment = Experiment.create("exp", tmp_path)
    for i in range(1, n_models + 1):
        model_id = f"model_{i:03d}"
        mdir = models_store.model_dir(experiment.dir, model_id)
        mdir.mkdir(parents=True)
        models_store.write_model_yaml(
            mdir,
            {
                "id": model_id,
                "name": f"m{i}",
                "status": "complete",
                "trainer_id": "ultralytics_yolo",
                "dataset_id": "dataset_001",
            },
        )
    for n, model_id in enumerate(distilled, start=1):
        run_id = f"distill_{n:03d}"
        distill_store.write_run(
            experiment.dir,
            run_id,
            {
                "id": run_id,
                "name": "d",
                "status": "complete",
                "stages": {
                    "label": {"status": "complete", "dataset_id": "dataset_001"},
                    "train": {"status": "complete", "model_id": model_id},
                },
            },
        )
    return experiment


def _column(table, header: str) -> list[str]:
    model = table.view.model()
    col = [
        model.headerData(c, pages.Qt.Orientation.Horizontal)
        for c in range(model.columnCount())
    ].index(header)
    return [model.index(r, col).data() for r in range(model.rowCount())]


def test_models_page_lists_newest_first_and_names_the_distill_that_made_each(
    qapp, tmp_path
):
    experiment = _experiment_with_models(tmp_path, 3, distilled=("model_002",))
    page = pages.ModelsPage(
        experiment, lambda _i: None, on_open_distill_run=lambda _i: None
    )
    models, runs = page.findChildren(pages.IndexTable)
    assert _column(models, "Id") == ["model_003", "model_002", "model_001"]
    assert _column(models, "Source") == ["trained", "distill_001", "trained"]
    assert _column(runs, "Id") == ["distill_001"]


def test_train_from_dataset_is_offered_only_with_a_dataset_to_train_on(qapp, tmp_path):
    from detail.storage import datasets as datasets_store

    experiment = _experiment_with_models(tmp_path, 0, distilled=())
    page = pages.ModelsPage(experiment, lambda _i: None)
    assert not page._train_btn.isEnabled()
    assert "Needs a completed dataset" in page._train_btn.toolTip()

    ddir = datasets_store.dataset_dir(experiment.dir, "dataset_001")
    ddir.mkdir(parents=True)
    datasets_store.write_dataset_yaml(
        ddir, {"id": "dataset_001", "name": "d", "status": "complete"}
    )
    page.refresh()
    assert page._train_btn.isEnabled()


def test_a_distill_table_nobody_can_open_does_not_advertise_opening(qapp, tmp_path):
    experiment = _experiment_with_models(tmp_path, 1, distilled=("model_001",))
    page = pages.ModelsPage(experiment, lambda _i: None)  # no distill opener wired
    runs = page.findChildren(pages.IndexTable)[1]
    # Id, Name, Kind, Status, Label, Train — and no trailing `›` column
    assert runs.view.model().columnCount() == 6


def test_manager_shows_the_active_experiments_models(qapp, tmp_path):
    experiment = _experiment_with_models(tmp_path, 2)
    opened = []
    page = pages.ModelManagerPage(
        experiment, lambda m: None, opened.append, lambda h: None
    )
    assert isinstance(page._experiment_table, pages.IndexTable)
    assert _column(page._experiment_table, "Id") == ["model_002", "model_001"]
    view = page._experiment_table.view
    view.doubleClicked.emit(view.model().index(0, 0))
    assert opened == ["model_002"]
    headings = {label.text() for label in page.findChildren(QLabel)}
    assert "Global models" in headings and "Your models" not in headings
    assert "Models in exp" in headings


def test_manager_with_no_experiment_open_says_where_its_models_are(qapp):
    page = pages.ModelManagerPage(None, lambda m: None, lambda m: None, lambda h: None)
    assert isinstance(page._experiment_table, QLabel)
    assert "open an experiment" in page._experiment_table.text()


def test_the_managers_tables_hold_four_rows_and_scroll(qapp, tmp_path):
    """Three tables share one page; each holds `TABLE_ROWS` and scrolls past it, so a long
    one cannot push the others off screen."""
    experiment = _experiment_with_models(tmp_path, 7, distilled=())
    page = pages.ModelManagerPage(
        experiment, lambda m: None, lambda m: None, lambda h: None
    )
    view = page._experiment_table.view
    row = view.verticalHeader().defaultSectionSize()
    assert view.model().rowCount() == 7
    shown = view.sizeHint().height()
    assert shown < 5 * row + 60, f"{shown}px holds more than {page.TABLE_ROWS} rows"
    assert view.maximumHeight() > shown  # the rest is there, behind the scrollbar
    hf = page._hf._table
    assert hf.sizeHint().height() < 5 * row + 60


def test_the_manager_would_notice_an_uncapped_table(qapp, tmp_path, monkeypatch):
    """Guard the guard: with the cap gone the seven-row table shows all seven."""
    monkeypatch.setattr(pages.ModelManagerPage, "TABLE_ROWS", None)
    with pytest.raises(AssertionError, match="holds more than"):
        test_the_managers_tables_hold_four_rows_and_scroll(qapp, tmp_path)


def test_suggested_models_name_real_catalog_rows():
    """A renamed model or variant would silently drop off the top of the list."""
    rows = {(r.id, r.variant) for r in api_models.list_model_variants()}
    missing = [pair for pair in api_models.suggested_models() if pair not in rows]
    assert (
        not missing
    ), f"suggested_models() names rows the catalog does not have: {missing}"


def test_hugging_face_lists_suggested_then_installed_then_the_rest(qapp):
    page = pages.GlobalModelsPage(builtin_only=True, embedded=True)
    # A late-sorting row, so catalog order alone would not put it next.
    installed = ("dinov3_vit", "vit_small")
    page._installed = {
        (r.id, r.variant): (r.id, r.variant) == installed for r in page._rows
    }
    page._render_table()
    order = [(r.id, r.variant) for r in page._rows]
    n = len(api_models.suggested_models())
    assert order[:n] == list(api_models.suggested_models())
    assert order[n] == installed  # the installed one, next
    # and the table shows that order, not the catalog's
    ids = [page._table.model().index(i, 0).data() for i in range(n + 1)]
    assert ids == [pair[0] for pair in order[: n + 1]]


def _rail_children(window, kind):
    from PyQt6.QtWidgets import QApplication

    from detail.gui.app import _ROLE
    from tests.gui.conftest import drain_until

    node = window._top_node(kind)
    node.setExpanded(True)  # the fill is off the UI thread, and lands when it lands
    drain_until(
        QApplication.instance(), lambda: window._top_node(kind).childCount() > 0
    )
    node = window._top_node(kind)
    return [node.child(i).data(0, _ROLE)[1][1] for i in range(node.childCount())]


def test_a_new_model_reaches_the_rail(qapp, tmp_path, monkeypatch):
    """Models are rail children now, so a launch or a delete has to re-read the rail — the
    old hook refreshed only the page on screen, which was all a drill-in page needed."""
    from detail.gui import app as gui_app
    from detail.storage import models as models_store

    monkeypatch.setattr(locations, "experiments_root", lambda: tmp_path)
    experiment = _experiment_with_models(tmp_path, 1, distilled=())
    window = gui_app.MainWindow()
    window._open_experiment(experiment)
    assert _rail_children(window, "models") == ["model_001"]

    mdir = models_store.model_dir(experiment.dir, "model_002")
    mdir.mkdir(parents=True)
    models_store.write_model_yaml(mdir, {"id": "model_002", "status": "running"})
    window._refresh_training()  # what every model/dataset/distill launch and delete calls
    assert _rail_children(window, "models") == ["model_001", "model_002"]
    window.deleteLater()


def test_the_install_scan_landing_keeps_the_users_selection_and_sort(qapp, monkeypatch):
    """The install scan returns a second or two after the page opens, and its landing rebuilt
    the table: a row picked or a column sorted in that window was dropped, and Check access /
    Install greyed out again under the user's hand (#35). The rebuild still moves installed
    rows up (the documented order), and now keeps what the user chose."""
    from PyQt6.QtCore import Qt

    # Hold the scan back, so the user's clicks land before it, as on a real first open.
    monkeypatch.setattr(pages.GlobalModelsPage, "_scan_installed", lambda self: None)
    page = pages.GlobalModelsPage(builtin_only=True, embedded=True)
    assert len(page._rows) >= 3, "the catalog needs a few rows to sort and pick from"

    page._table.sortByColumn(0, Qt.SortOrder.DescendingOrder)
    page._table.selectRow(len(page._rows) - 1)
    picked = page._selected()
    assert picked is not None
    picked_key = (picked.id, picked.variant)
    page._on_installed_scanned(
        {(r.id, r.variant): (r.id, r.variant) != picked_key for r in page._rows}
    )

    after = page._selected()
    assert after is not None and (after.id, after.variant) == picked_key
    header = page._table.horizontalHeader()
    assert header.sortIndicatorSection() == 0
    assert header.sortIndicatorOrder() == Qt.SortOrder.DescendingOrder
    model = page._table.model()
    ids = [model.index(r, 0).data() for r in range(model.rowCount())]
    assert ids == sorted(ids, reverse=True)
    # ...and the actions follow the selection that survived: this one is not installed.
    assert page._check_btn.isEnabled() and page._install_btn.isEnabled()


def test_an_unsorted_table_takes_the_documented_order_when_the_scan_lands(
    qapp, monkeypatch
):
    """The other half: with no column clicked, keeping "the user's sort" must not freeze the
    pre-scan order. Installed rows still move up behind the suggested ones."""
    monkeypatch.setattr(pages.GlobalModelsPage, "_scan_installed", lambda self: None)
    page = pages.GlobalModelsPage(builtin_only=True, embedded=True)
    installed = ("dinov3_vit", "vit_small")
    page._on_installed_scanned(
        {(r.id, r.variant): (r.id, r.variant) == installed for r in page._rows}
    )
    n = len(api_models.suggested_models())
    assert (page._rows[n].id, page._rows[n].variant) == installed
    assert page._table.horizontalHeader().sortIndicatorSection() == -1
