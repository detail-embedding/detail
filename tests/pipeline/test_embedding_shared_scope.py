"""Resolution of the data-derived constructor parameters across a selection (§14).

The canvas has a scope: ``per_recording`` (the default) leaves each recording's ``validate`` to
derive its own, and ``shared`` resolves it once across the whole selection in
``EmbeddingModule.resolve_shared_config``. The max-contrast background has none — it is always
resolved there, once, from the subjects pooled over the selection, so a multi-recording run never
shows the encoder a background that tells its recordings apart.
"""

import pathlib

import yaml

import pytest

from detail.pipeline import embedding
from detail.pipeline.embedding import EmbeddingModule

CTOR = "masked_color_centered_contrast"


@pytest.fixture
def experiment_with(tmp_path):
    """Factory: an experiment whose recordings report the given largest-mask sizes."""

    def _make(sizes: dict[str, int], means: dict[str, tuple] | None = None):
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"schema_version": 1, "name": "p"}, sort_keys=False),
            encoding="utf-8",
        )
        for name in sizes:
            rdir = tmp_path / "recordings" / name
            rdir.mkdir(parents=True)
            (rdir / "recording.yaml").write_text(
                yaml.dump({"schema_version": 1, "id": name}, sort_keys=False),
                encoding="utf-8",
            )
        return tmp_path

    return _make


@pytest.fixture
def stub_reads(monkeypatch):
    """Stub the two data-reading primitives; the shared resolution is what's under test,
    not mask decoding (which the per-recording tests already cover)."""

    def _apply(sizes: dict[str, int], means: dict[str, tuple] | None = None):
        monkeypatch.setattr(
            embedding,
            "largest_mask_px",
            lambda rdir, *a, **k: sizes[pathlib.Path(rdir).name],
        )
        monkeypatch.setattr(
            embedding,
            "_mean_mask_color_rgb",
            lambda rdir, *a, **k: (means or {})[pathlib.Path(rdir).name],
        )

    return _apply


def _config(**over):
    base = {
        "constructor": CTOR,
        "tracking_run_id": "track_01",
        "track_ids": [0],
        "model_id": "dinov3_vit",
        "canvas_scope": "per_recording",
    }
    base.update(over)
    return base


def test_a_per_recording_canvas_is_left_to_each_recording(experiment_with, stub_reads):
    sizes = {"a": 700, "b": 1000}
    root = experiment_with(sizes)
    stub_reads(sizes)
    out = EmbeddingModule().resolve_shared_config(
        root, ["a", "b"], _config(constructor="silhouette_centered")
    )
    assert "canvas_px" not in out and "bg_color" not in out


@pytest.mark.parametrize(
    "old_setting",
    [None, "per_recording"],
    ids=["no-setting", "an-old-config-asking-for-per-recording"],
)
def test_the_background_is_one_color_for_the_whole_run(
    experiment_with, monkeypatch, old_setting
):
    """The reported case: two dark subjects resolve blue on their own and a magenta one
    resolves green, so per-recording colors made the third recording incomparable with the
    first two. A run over all three gets one color, and a config saved before the setting
    was removed cannot ask for the old behaviour back."""
    means = {"a": (45, 41, 34), "b": (73, 69, 62), "c": (200, 60, 200)}  # RGB
    assert embedding._max_contrast_rgb(means["a"]) != embedding._max_contrast_rgb(
        means["c"]
    )
    root = experiment_with(dict.fromkeys(means, 700))
    monkeypatch.setattr(
        embedding,
        "_mean_mask_color_rgb",
        lambda rdir, *a, **k: means[pathlib.Path(rdir).name],
    )
    monkeypatch.setattr(
        embedding,
        "largest_mask_px",
        lambda *a, **k: pytest.fail("a per-recording canvas is not pooled"),
    )
    config = _config()
    if old_setting is not None:
        config["bg_color_scope"] = old_setting
    out = EmbeddingModule().resolve_shared_config(root, list(means), config)
    assert out["bg_color"] == embedding._max_contrast_rgb((106, 57, 99))
    assert "canvas_px" not in out


def test_each_recording_is_announced_before_it_is_read(experiment_with, monkeypatch):
    """Sampling a recording's subject colour decodes video, seconds per recording, so the
    launch names the recording it is on. The hook fires before the read: fired after, the
    line would name the recording just finished while the next one is the slow one."""
    events = []
    root = experiment_with({"a": 700, "b": 700})
    monkeypatch.setattr(
        embedding,
        "_mean_mask_color_rgb",
        lambda rdir, *a, **k: events.append(("read", pathlib.Path(rdir).name))
        or (50, 50, 50),
    )
    EmbeddingModule().resolve_shared_config(
        root,
        ["a", "b"],
        _config(),
        on_recording=lambda sid: events.append(("announce", sid)),
    )
    assert events == [
        ("announce", "a"),
        ("read", "a"),
        ("announce", "b"),
        ("read", "b"),
    ]


def test_shared_canvas_takes_the_max_across_the_selection(experiment_with, stub_reads):
    """The max is the only value that still contains the largest subject's mask."""
    sizes = {"a": 700, "b": 1038, "c": 941}
    root = experiment_with(sizes)
    stub_reads(sizes)
    out = EmbeddingModule().resolve_shared_config(
        root,
        ["a", "b", "c"],
        _config(constructor="silhouette_centered", canvas_scope="shared"),
    )
    assert out["canvas_px"] == 1038


def test_shared_background_pools_the_subjects(experiment_with, stub_reads):
    """One color for the selection, derived from the *pooled* subject color — not from
    whichever recording happened to be first."""
    sizes = {"a": 700, "b": 900}
    root = experiment_with(sizes)
    stub_reads(sizes, {"a": (45, 41, 34), "b": (73, 69, 62)})  # RGB
    out = EmbeddingModule().resolve_shared_config(root, ["a", "b"], _config())
    assert out["bg_color"] == embedding._max_contrast_rgb((59, 55, 48))
    assert out["bg_color"] == [0, 0, 255]  # these dark warm-grays -> blue, in RGB
    assert "canvas_px" not in out


def test_pinned_renderer_spec_resolves_nothing(experiment_with, stub_reads):
    """The raw-API path already fixes every value; there is nothing to share."""
    sizes = {"a": 700, "b": 1000}
    root = experiment_with(sizes)
    stub_reads(sizes)
    out = EmbeddingModule().resolve_shared_config(
        root,
        ["a", "b"],
        _config(canvas_scope="shared", renderer_spec=[{"type": "crop_to_bbox"}]),
    )
    assert "canvas_px" not in out


def test_constructor_without_a_canvas_resolves_nothing(experiment_with, stub_reads):
    """`full_frame` needs neither a canvas nor a background, so `shared` is a no-op."""
    sizes = {"a": 700, "b": 1000}
    root = experiment_with(sizes)
    stub_reads(sizes)
    out = EmbeddingModule().resolve_shared_config(
        root, ["a", "b"], _config(constructor="full_frame", canvas_scope="shared")
    )
    assert "canvas_px" not in out


def test_resolve_renderer_spec_honours_an_explicit_canvas(tmp_path, monkeypatch):
    """The override is what carries a shared value into each recording's own validate."""
    monkeypatch.setattr(
        embedding, "largest_mask_px", lambda *a, **k: pytest.fail("should not be read")
    )
    spec = embedding.resolve_renderer_spec(
        tmp_path, "track_01", [0], "silhouette_centered", canvas_px=1038
    )
    assert {"type": "center_in_square", "size": 1038} in spec


def test_resolve_renderer_spec_honours_an_explicit_bg_color(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding, "largest_mask_px", lambda *a, **k: 500)
    monkeypatch.setattr(
        embedding,
        "contrast_bg_color",
        lambda *a, **k: pytest.fail("should not be sampled"),
    )
    spec = embedding.resolve_renderer_spec(
        tmp_path, "track_01", [0], CTOR, canvas_px=500, bg_color=[0, 255, 0]
    )
    assert {"type": "apply_mask", "background": [0, 255, 0]} in spec


def test_scopes_reach_the_job_and_the_recorded_run():
    """The canvas scope must be recorded, or a later comparability check cannot tell a
    deliberate per-recording canvas from a mismatch. The background is recorded as shared
    on every new run — whatever an old config says — so a benchmark can tell it from a run
    made before the background was always shared."""
    from detail.core.dataclasses import ComputeTarget, EmbeddingJob

    module = EmbeddingModule()
    config = _config(
        canvas_scope="shared",
        bg_color_scope="per_recording",
        renderer_spec=[{"type": "crop_to_bbox"}],
        compute=ComputeTarget("cpu", None, None, None),
    )
    job = module.build_job("rec", "emb_01", config)
    assert (job.canvas_scope, job.bg_color_scope) == ("shared", "shared")
    assert EmbeddingJob.from_dict(job.to_dict()).canvas_scope == "shared"
    record = module.extra_record_fields(config)
    assert (record["canvas_scope"], record["bg_color_scope"]) == ("shared", "shared")


def test_job_from_dict_defaults_scopes_for_older_jobs():
    from detail.core.dataclasses import ComputeTarget, EmbeddingJob

    d = EmbeddingJob(
        job_id="j", recording_id="r", embedding_run_id="emb_01",
        tracking_run_id="track_01", track_ids=[0], model_id="m",
        renderer_spec=[], batch_size=1, num_workers=0,
        compute=ComputeTarget("cpu", None, None, None), created_at="t",
    ).to_dict()  # fmt: skip
    del d["canvas_scope"], d["bg_color_scope"]
    assert EmbeddingJob.from_dict(d).canvas_scope == "per_recording"
    assert EmbeddingJob.from_dict(d).bg_color_scope == "per_recording"


def test_cli_exposes_the_canvas_scope_and_no_background_scope():
    """Surface parity: the canvas scope only means anything across recordings, so it belongs
    on `process embedding` (the multi-recording path), which the GUI/API also take. The
    background has no choice to expose."""
    from detail.cli import process_embedding

    names = {p.name for p in process_embedding.params}
    assert "canvas_scope" in names
    assert "bg_color_scope" not in names
    opt = next(p for p in process_embedding.params if p.name == "canvas_scope")
    assert opt.default == "per_recording"
    assert set(opt.type.choices) == {"per_recording", "shared"}


def test_no_surface_offers_a_background_scope():
    """The config model is what the GUI form, the CLI help and the parameter reference read,
    so a field here is a setting a user is shown."""
    from detail.configs.embedding import EmbeddingConfig

    assert "bg_color_scope" not in EmbeddingConfig.model_fields
