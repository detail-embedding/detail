"""Experiment-free machine-global model catalog (api/models.py).

The model cache + imported catalog are isolated per test by the autouse
``_isolate_model_cache`` fixture (DETAIL_MODEL_DIR -> tmp), so these never
touch the real machine or the network.
"""

import pytest

from detail.api import models as api_models
from detail.core.protocols import EmbeddingModel
from detail.core.registries import TRAINED_LOADERS
from detail.models import imported as imported_store
from detail.models.hf_access import HFHubModel

# Reuse the imported-catalog stub loader (introspect reads no real checkpoint).
from tests.models.test_imported import _StubLoader  # noqa: E402


@pytest.fixture
def stub_family(monkeypatch):
    monkeypatch.setitem(TRAINED_LOADERS, "stubfam", _StubLoader)
    return "stubfam"


def _by_id(rows):
    return {r.id: r for r in rows}


def test_list_models_built_ins_with_architecture_and_source():
    rows = _by_id(api_models.list_models())
    assert rows["sam3"].role == "detection"
    assert rows["sam3"].architecture == "sam"  # distinguishes it from yolo
    assert rows["sam3"].source == "built-in"
    assert rows["dinov3_convnext_base"].role == "embedding"
    assert rows["dinov3_convnext_base"].architecture == "convnext"
    assert (
        "convnext_base" in rows["dinov3_convnext_base"].variants
    )  # size axis surfaced
    assert rows["sam3"].classes == []  # open-vocab → no fixed classes


def test_gated_model_pages_lists_one_link_per_access_gate():
    pages = dict(api_models.gated_model_pages())
    # SAM 3 gates on its own repo, so its key is the model id.
    assert pages["sam3"] == "https://huggingface.co/facebook/sam3"
    # Meta gates the DINOv3 collection as a family: accepting once covers every backbone
    # and size, so the setup flow asks once instead of sending the user to the same form
    # per model. DINOv3 resolves its repo per-instance (from the size variant), so it
    # must be read off an instance, not the class — the regression that once dropped it.
    assert "dinov3" in pages
    assert pages["dinov3"].startswith("https://huggingface.co/facebook/dinov3-")
    assert not [k for k in pages if k.startswith("dinov3_")]  # no per-backbone links
    assert all(url.startswith("https://huggingface.co/") for url in pages.values())


def test_every_embedding_model_exposes_input_and_embedding_size():
    # The Input images and Embedding models doc pages render tables from this catalog, so a new
    # embedding model must carry input_size and output_dim to appear correctly
    # (scripts/gen_docs_input_images.py, scripts/gen_docs_embedding_models.py).
    rows = _by_id(api_models.list_models())
    embedding = [r for r in rows.values() if r.role == "embedding"]
    assert embedding, "expected at least one built-in embedding model"
    for r in embedding:
        assert isinstance(r.input_size, int) and r.input_size > 0, r.id
        assert isinstance(r.output_dim, int) and r.output_dim > 0, r.id
    assert (
        rows["dinov3_convnext_base"].input_size,
        rows["dinov3_convnext_base"].output_dim,
    ) == (
        224,
        1024,
    )
    # detection models have neither
    assert rows["sam3"].input_size is None
    assert rows["sam3"].output_dim is None


def test_describe_embedding_models_covers_all_with_variant_sizes():
    docs = {m.model_id: m for m in api_models.describe_embedding_models()}
    from detail.core.registries import EMBEDDING_MODELS

    api_models._ensure_registered()
    assert set(docs) == set(EMBEDDING_MODELS)  # every registered model is described
    dino = docs["dinov3_convnext_base"]
    assert dino.default_variant == "convnext_base"
    assert dict(dino.variant_sizes) == {
        "convnext_tiny": 768,
        "convnext_small": 768,
        "convnext_base": 1024,
        "convnext_large": 1536,
    }
    assert "cuda" in dino.backends and dino.gated is True
    assert dino.description  # first docstring line, not empty


def test_list_models_includes_imported(tmp_path, stub_family):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    mid = imported_store.import_model(weights, family=stub_family, name="my_yolo")
    row = _by_id(api_models.list_models())[mid]
    assert row.source == "imported"
    assert row.architecture == "stubfam"  # from the record's model_family
    assert row.classes == ["mouse", "pellet"]  # read from the checkpoint
    assert row.role == "detection"


def test_is_installed_is_no_network():
    # Cache is the isolated tmp dir → nothing installed, and no network call/raise.
    assert api_models.is_installed("dinov3_convnext_base") is False
    assert api_models.is_installed("sam3") is False


def test_is_installed_true_when_the_snapshot_is_complete(fake_hf_snapshot):
    """The GUI's Status column read is_cached() against a cache load() never wrote
    to, so it reported "not installed" forever with the weights on disk. It now asks
    the model, which checks the cache install() actually populates."""
    from detail.models.sam3 import SAM3

    fake_hf_snapshot(SAM3.hf_repo, files=SAM3.hf_files)
    assert api_models.is_installed("sam3") is True


def test_is_installed_false_when_the_snapshot_is_partial(fake_hf_snapshot):
    """A half-finished download must not read as installed — load() would die on the
    missing file. is_cached() cannot see this (it stats the folder, not the files)."""
    from detail.models.sam3 import SAM3

    fake_hf_snapshot(SAM3.hf_repo, files=["config.json"])  # sam3.pt never landed
    assert api_models.is_installed("sam3") is False


def test_is_installed_answers_per_variant(fake_hf_snapshot):
    """Each size is a separate repo and a separate download, so install state is
    per variant. Answering for the default variant regardless of the one asked for
    reported the model installed while a run configured for another size failed
    its pre-flight with "is not downloaded" — with no way to fix it, because
    install went to the default variant too.
    """
    from detail.models.dinov3 import DINOv3ConvNextBase

    default = DINOv3ConvNextBase()  # convnext_base
    fake_hf_snapshot(default.hf_repo, files=DINOv3ConvNextBase.hf_files)
    assert api_models.is_installed("dinov3_convnext_base") is True
    assert api_models.is_installed("dinov3_convnext_base", "convnext_base") is True
    assert api_models.is_installed("dinov3_convnext_base", "convnext_tiny") is False


class _SizedStub(HFHubModel, EmbeddingModel):
    """A varianted HF model with no backend dependency.

    DINOv3's own ``readiness`` reports ``missing_dependency`` before it looks at the
    weights when torch/transformers are absent, which is the CI environment — so the
    readiness half of the variant plumbing is pinned on a stub that has no such gate.
    """

    model_id = "sized_stub"
    architecture = "stub"
    VARIANTS = {"small": "stub/repo-small", "large": "stub/repo-large"}
    DEFAULT_VARIANT = "large"
    gated = False
    hf_files = ["config.json"]

    def __init__(self, variant=None):
        super().__init__(variant)
        self.hf_repo = self.weight

    def load(self, device): ...

    def embed(self, image): ...

    def embed_batch(self, images): ...


def test_model_readiness_answers_per_variant(fake_hf_snapshot, monkeypatch):
    """Readiness must answer for the size asked for. Answering for the default sent
    a run that had picked another size to a green pre-flight, or — as reported —
    reported a model installed on the Models tab that the run then rejected."""
    from detail.core.registries import EMBEDDING_MODELS

    api_models._ensure_registered()
    monkeypatch.setitem(EMBEDDING_MODELS, "sized_stub", _SizedStub)
    fake_hf_snapshot("stub/repo-large", files=["config.json"])  # the default only

    assert api_models.model_readiness("sized_stub").ready is True
    readiness = api_models.model_readiness("sized_stub", "small")
    assert readiness.ready is False and readiness.reason == "not_installed"
    # and the message points at the command that installs *that* variant
    assert "detail model install sized_stub --variant small" in readiness.message


def test_install_model_downloads_the_requested_variant(monkeypatch):
    """`install_model(id)` built the model with no variant, so every install — CLI
    and the GUI's Models tab alike — fetched the default size."""
    installed = []

    class _FakeInstall:
        def install(self, progress_cb=None):
            installed.append(self.hf_repo)

    from detail.models.dinov3 import DINOv3ConvNextBase

    monkeypatch.setattr(DINOv3ConvNextBase, "install", _FakeInstall.install)
    api_models.install_model("dinov3_convnext_base", variant="convnext_tiny")
    assert installed == ["facebook/dinov3-convnext-tiny-pretrain-lvd1689m"]


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        api_models.model_readiness("dinov3_convnext_base", "convnext_huge")


def test_list_model_variants_expands_every_size(fake_hf_snapshot):
    """The install-status surfaces (GUI Models tab, `model list`) list one row per
    installable unit — a model with sizes contributes one row per size, so each
    reports its own status and Install targets exactly that download."""
    rows = api_models.list_model_variants()
    dino = [r for r in rows if r.id == "dinov3_convnext_base"]
    assert [r.variant for r in dino] == [
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "convnext_large",
    ]
    # a model with no size axis stays a single row
    sam3 = [r for r in rows if r.id == "sam3"]
    assert len(sam3) == 1 and sam3[0].variant is None
    # list_models stays one row per model — the docs tables render from it
    assert (
        len([r for r in api_models.list_models() if r.id == "dinov3_convnext_base"])
        == 1
    )


def test_is_installed_true_for_imported(tmp_path, stub_family):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    mid = imported_store.import_model(weights, family=stub_family, name="my_yolo")
    assert api_models.is_installed(mid) is True  # the .pt is on disk by definition


def test_model_readiness_ready_for_imported(tmp_path, stub_family):
    # `model check <imported_id>` must not raise "unknown model_id" — an imported
    # model is on disk by definition, so it is ready (no repo/token).
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"weights")
    mid = imported_store.import_model(weights, family=stub_family, name="my_yolo")
    readiness = api_models.model_readiness(mid)
    assert readiness.ready is True
    assert readiness.reason == "ok"


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model_id"):
        api_models.model_readiness("nope")


# --- catalog metadata + preview + export/import (§12) ----------------------- #
def _weights(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"weights")
    return w


def test_import_with_metadata_surfaces_on_the_ref(tmp_path, stub_family):
    img = tmp_path / "shot.png"
    img.write_bytes(b"png-bytes")
    mid = api_models.import_model(
        _weights(tmp_path),
        family=stub_family,
        name="operant_v2",
        made_by="labmate",
        description="operant chamber",
        notes="room 1",
        preview=img,
    )
    ref = api_models.imported_model(mid)
    assert ref.made_by == "labmate"
    assert ref.description == "operant chamber"
    assert ref.notes == "room 1"
    assert ref.created_at  # stamped on import
    assert ref.preview_path is not None and ref.preview_path.is_file()
    assert (
        api_models.imported_models(role=None)[0].made_by == "labmate"
    )  # also on the list


def test_imported_model_missing_raises():
    with pytest.raises(FileNotFoundError):
        api_models.imported_model("nope")


def test_update_model_metadata_edits_only_passed_fields(tmp_path, stub_family):
    mid = api_models.import_model(
        _weights(tmp_path), family=stub_family, name="m", made_by="a"
    )
    api_models.update_model_metadata(mid, description="new desc")
    ref = api_models.imported_model(mid)
    assert ref.made_by == "a"  # untouched — a None field is left alone
    assert ref.description == "new desc"


def test_set_model_preview(tmp_path, stub_family):
    mid = api_models.import_model(_weights(tmp_path), family=stub_family, name="m")
    assert api_models.imported_model(mid).preview_path is None
    img = tmp_path / "p.jpg"
    img.write_bytes(b"jpg")
    api_models.set_model_preview(mid, img)
    ref = api_models.imported_model(mid)
    assert ref.preview_path is not None and ref.preview_path.name == "preview.jpg"


def test_export_import_round_trip_preserves_metadata(tmp_path, stub_family):
    img = tmp_path / "shot.png"
    img.write_bytes(b"png")
    mid = api_models.import_model(
        _weights(tmp_path),
        family=stub_family,
        name="operant_v2",
        made_by="labmate",
        description="d",
        notes="n",
        preview=img,
    )
    events = []
    dest = tmp_path / "operant_v2.zip"
    api_models.export_model(
        mid, dest, progress=lambda s, f, d: events.append((s, f, d))
    )
    assert dest.is_file()
    assert events and events[0][0] == "exporting" and events[-1][1] == 1.0
    api_models.remove_imported_model(mid)
    assert mid not in imported_store.discover(role=None)

    restored = api_models.import_model_archive(dest)
    assert restored == mid
    ref = api_models.imported_model(restored)
    assert ref.made_by == "labmate" and ref.description == "d" and ref.notes == "n"
    assert ref.preview_path is not None and ref.preview_path.is_file()
    assert (imported_store.model_dir(mid) / "weights/best.pt").is_file()


def test_import_archive_rejects_collision(tmp_path, stub_family):
    mid = api_models.import_model(_weights(tmp_path), family=stub_family, name="dup")
    dest = tmp_path / "dup.zip"
    api_models.export_model(mid, dest)
    with pytest.raises(ValueError, match="already exists"):
        api_models.import_model_archive(dest)  # the model still exists → reject


def test_source_experiments_wraps_legacy_single_source():
    # A model promoted before the schema change stored source_experiment_name/id; readers
    # present it as a one-element list so it still shows a 'from project'.
    legacy = {"source_experiment_name": "rp", "source_experiment_id": "rp_id"}
    assert api_models._source_experiments(legacy) == [{"name": "rp", "id": "rp_id"}]
    new = {"source_experiments": [{"name": "a", "id": "a"}, {"name": "b", "id": "b"}]}
    assert api_models._source_experiments(new) == [
        {"name": "a", "id": "a"},
        {"name": "b", "id": "b"},
    ]
    assert api_models._source_experiments(None) == []


def test_tool_models_lists_sam3_and_your_models(tmp_path, stub_family):
    mid = api_models.import_model(
        _weights(tmp_path), family=stub_family, name="my_yolo"
    )
    by_id = {m.id: m for m in api_models.tool_models()}
    assert by_id["sam3"].prompt_capable is True  # a text-cue built-in
    assert by_id[mid].prompt_capable is False  # a closed-set imported model


# --- one lookup across every model registry (§5 "Model registry") ------------ #
def test_all_models_resolves_an_id_to_its_role_and_class():
    models = api_models.all_models()
    assert models["sam3"][0] == "detection"
    assert models["dinov3_convnext_base"][0] == "embedding"
    assert issubclass(models["dinov3_convnext_base"][1], EmbeddingModel)
    # every row list_models() shows for a built-in comes from this view, with the same role
    built_in = {
        r.id: r.role for r in api_models.list_models() if r.source == "built-in"
    }
    assert built_in == {
        mid: role
        for mid, (role, cls) in models.items()
        if not getattr(cls, "shares_weights_with", None)  # listed on its backbone's row
    }


def test_unknown_model_names_what_is_available():
    # supported_backends falls back (imported / trained ids resolve by family, not here)
    assert api_models.supported_backends("nope") == ["cuda", "mps", "cpu"]
    with pytest.raises(ValueError, match="unknown model_id 'nope'; available: \\["):
        api_models._model("nope")


# --- a shared download is listed once, with what it is read for (§14) -------- #
def test_the_backbone_row_carries_patch_features_and_the_match_has_no_row_of_its_own():
    rows = _by_id(api_models.list_models())
    assert rows["dinov3_vit"].capabilities == ["embedding", "patch features"]
    assert rows["dinov3_convnext_base"].capabilities == ["embedding"]
    assert rows["sam3"].capabilities == ["detection"]
    assert (
        "dinov3_match" not in rows
    )  # shares_weights_with: nothing of its own to install
    assert (
        api_models.all_models()["dinov3_match"][0] == "keypoint"
    )  # ...but it resolves
    assert api_models.supported_backends("dinov3_match") == ["cuda", "mps", "cpu"]
    # install state asked of the match is the backbone's: no second cache location
    assert api_models.is_installed(
        "dinov3_match", "vit_base"
    ) == api_models.is_installed("dinov3_vit", "vit_base")


# --- keypoint models in the catalog (§12 *Keypoint models in the catalog*) --------- #
@pytest.fixture
def pose_pt(tmp_path, monkeypatch):
    """A pose checkpoint (stubbed readers: task pose, RAT's K and flip_idx)."""
    from detail.models import yolo_pose
    from tests.models.test_imported import _PoseStubLoader

    api_models._ensure_registered()  # so the real loader cannot overwrite the stub
    monkeypatch.setitem(TRAINED_LOADERS, imported_store._PROBE_FAMILY, _PoseStubLoader)
    monkeypatch.setattr(
        yolo_pose.TrainedYoloPose,
        "introspect",
        staticmethod(lambda weights_path: (4, [0, 2, 1, 3])),
    )
    w = tmp_path / "rat-pose.pt"
    w.write_bytes(b"pose-weights")
    return w


def _rat_yaml(tmp_path, name="rat overhead"):
    import yaml

    from tests.models.test_imported import RAT

    path = tmp_path / "rat.yaml"
    path.write_text(
        yaml.safe_dump({"name": name, "skeleton": RAT.to_dict()}), encoding="utf-8"
    )
    return path


def test_import_takes_a_skeleton_file_preset_or_library_id(tmp_path, pose_pt):
    from detail.api import skeletons
    from tests.models.test_imported import RAT

    mid = api_models.import_model(pose_pt, skeleton=_rat_yaml(tmp_path), name="a")
    ref = api_models.imported_model(mid)
    assert ref.role == "keypoints" and ref.task == "pose"
    assert ref.skeleton == RAT
    assert ref.skeleton_ref["name"] == "rat overhead"  # the file's stated name
    # the library id it was added as works as a source too, and is reused
    again = api_models.import_model(pose_pt, skeleton=ref.skeleton_ref["id"], name="b")
    assert api_models.imported_model(again).skeleton_ref == ref.skeleton_ref
    assert [e["id"] for e in skeletons.skeletons() if not e["builtin"]] == [
        ref.skeleton_ref["id"]
    ]
    # a preset that does not fit (5 points) is refused on its count
    with pytest.raises(
        ValueError, match="model places 4 points but the skeleton has 5"
    ):
        api_models.import_model(pose_pt, skeleton="quadruped_side_5", name="c")


def test_pose_models_are_keypoint_rows_and_kept_out_of_detection(
    tmp_path, pose_pt, stub_family
):
    pose = api_models.import_model(pose_pt, skeleton=_rat_yaml(tmp_path), name="rp")
    det = api_models.import_model(_weights(tmp_path), family=stub_family, name="det")
    rows = _by_id(api_models.list_models())
    assert (rows[pose].role, rows[pose].capabilities) == ("keypoint", ["keypoint"])
    assert (rows[det].role, rows[det].capabilities) == ("detection", ["detection"])
    tools = {m.id for m in api_models.tool_models()}
    assert det in tools and pose not in tools  # Quick segment cannot run a pose model
    assert api_models.imported_model(det).role == "detection"


def test_restore_round_trip_keeps_the_skeleton_and_reuses_its_entry(tmp_path, pose_pt):
    from detail.storage import skeleton_library

    mid = api_models.import_model(pose_pt, skeleton=_rat_yaml(tmp_path), name="rp")
    before = imported_store.read(mid)
    dest = api_models.export_model(mid, tmp_path / "rp.zip")
    api_models.remove_imported_model(mid)
    assert api_models.import_model_archive(dest) == mid
    after = imported_store.read(mid)
    assert after["skeleton"] == before["skeleton"]
    assert after["skeleton_ref"] == before["skeleton_ref"]  # same digest: reused
    assert len(skeleton_library.discover()) == 1


def test_restore_on_a_machine_with_another_skeleton_of_that_name(tmp_path, pose_pt):
    from detail.core.skeleton import Skeleton
    from detail.storage import skeleton_library
    from tests.models.test_imported import RAT

    mid = api_models.import_model(pose_pt, skeleton=_rat_yaml(tmp_path), name="rp")
    dest = api_models.export_model(mid, tmp_path / "rp.zip")
    # "another machine": the model and its entry are gone; a different skeleton has the name
    api_models.remove_imported_model(mid)
    for sid in skeleton_library.discover():
        skeleton_library.remove(sid)
    theirs = skeleton_library.create(
        "rat overhead", Skeleton(names=("head", "tail"), edges=(("head", "tail"),))
    )
    api_models.import_model_archive(dest)
    rec = imported_store.read(mid)
    assert rec["skeleton"] == RAT.to_dict()  # the model keeps its own copy
    assert rec["skeleton_ref"]["name"] == "rat overhead (2)"
    assert rec["skeleton_ref"]["id"] != theirs["id"]
    assert skeleton_library.read(theirs["id"]) == theirs  # never edited


def test_restore_refuses_a_skeleton_that_does_not_match_its_digest(tmp_path, pose_pt):
    import zipfile

    import yaml

    mid = api_models.import_model(pose_pt, skeleton=_rat_yaml(tmp_path), name="rp")
    rec = imported_store.read(mid)
    rec["skeleton"]["edges"].pop()  # edited after export: still valid, a new digest
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as zf:
        zf.writestr(f"{mid}/model.yaml", yaml.safe_dump(rec, sort_keys=False))
        zf.writestr(f"{mid}/weights/best.pt", b"pose-weights")
    api_models.remove_imported_model(mid)
    with pytest.raises(ValueError, match="does not match its recorded digest"):
        api_models.import_model_archive(tampered)
    assert mid not in imported_store.discover(
        role=None
    )  # nothing half-restored left behind
