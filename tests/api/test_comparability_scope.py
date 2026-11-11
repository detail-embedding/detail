"""Comparability across recordings when a constructor derives values from the data (§14).

The centered constructors size their canvas from the masks they embed, and the max-contrast
one picks its background from the animal's colour. The canvas is resolved at a *scope* the run
chooses: a ``per_recording`` canvas is a deliberate normalization and may differ between
recordings; a ``shared`` one was resolved once across the selection and must be identical. The
background has no scope to choose — a multi-recording run always picks one colour for the whole
selection — so it must match exactly, whatever an older run recorded as its scope.

**The class: a step that carries a derived value the check does not know is derived.** The check
masks per-recording values by ``(step type, parameter)``, a table kept apart from the builders
that write those values into the spec. Egocentric alignment's ``align_rotate`` copied the
canvas ``fill`` into a step of its own and the table never heard of it, so a per-recording
background that differed was masked on two steps and compared on the third — the runs were
refused on a difference the scope said was allowed, and nothing about the error pointed at the
rotate step. A test that hand-writes the spec cannot see this: it asserts a copy of what the
builder wrote last time. So the guard below builds every spec through the real producers
(``resolve_renderer_spec`` then ``with_alignment``), for every registered constructor and every
alignment it accepts, and varies one derived input at a time: a new step that carries the
canvas size without being registered makes a per-recording canvas incomparable and fails here,
and one that re-masks the background makes a differing background comparable and fails here.
"""

import numpy as np
import pytest

from detail.api import run_preview
from detail.storage import experiment as experiment_store
from detail.transforms.constructors import ALIGNABLE
from detail.transforms.constructors import CONSTRUCTORS

CTOR = "masked_color_centered_contrast"
BLUE = [0, 0, 255]  # RGB, the order the renderer paints in


def _recording(experiment_dir, name):
    path = experiment_store.recording_path(experiment_dir, name)
    (path / "embeddings").mkdir(parents=True, exist_ok=True)
    return path


def _traj(rng, n, d=8):
    return (
        rng.normal(size=(n, d)).astype(np.float32),
        np.arange(n, dtype=np.int32),
        (np.arange(n, dtype=np.int64) * 1_000_000),
    )


def _spec(canvas, bg=BLUE, resize=384):
    return [
        {"type": "crop_to_bbox"},
        {"type": "apply_mask", "background": list(bg)},
        {"type": "center_in_square", "size": canvas, "fill": list(bg)},
        {"type": "resize", "size": resize},
    ]


@pytest.fixture
def two_runs(experiment_dir, make_embedding_run):
    """Factory: the same recipe in two recordings, each with its own overrides."""

    def _make(**overrides):
        second = overrides.pop("second", {})
        paths = []
        for i, name in enumerate(("s1", "s2")):
            path = _recording(experiment_dir, name)
            kwargs = dict(
                constructor=CTOR,
                input_size=384,
                renderer_spec=_spec(700 + i * 300),
                **overrides,
            )
            if i == 1:
                kwargs.update(second)
            make_embedding_run(path, *_traj(np.random.default_rng(i), 12), **kwargs)
            paths.append(path)
        return [(p, "emb_01") for p in paths]

    return _make


def test_per_recording_canvas_may_differ(two_runs):
    """Same recipe, a different canvas per recording: still comparable.

    The case that blocked a real eval.
    """
    common = run_preview.validate_comparability(two_runs())
    assert common["model_id"] == "dinov3_vit"


def test_shared_canvas_scope_requires_identical_canvas(two_runs):
    pairs = two_runs(canvas_scope="shared")
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(pairs)


def test_shared_canvas_scope_accepts_identical_canvas(two_runs):
    pairs = two_runs(
        canvas_scope="shared",
        second={"renderer_spec": _spec(700), "canvas_scope": "shared"},
    )
    assert run_preview.validate_comparability(pairs)["model_id"]


def test_scopes_themselves_must_agree(two_runs):
    """One run normalized per recording and one shared are not the same representation."""
    pairs = two_runs(second={"canvas_scope": "shared", "renderer_spec": _spec(700)})
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(pairs)


def test_a_differing_background_is_not_comparable(two_runs):
    """Blue frames against green ones: the colour is part of what the encoder was shown."""
    pairs = two_runs(second={"renderer_spec": _spec(1000, bg=[0, 255, 0])})
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(pairs)


@pytest.mark.parametrize("old_scope", ["per_recording", None])
def test_an_older_runs_background_scope_does_not_decide(old_scope):
    """A run recorded before the background was always shared still says ``per_recording``
    (or nothing). Its colour is what matters: the same colour is the same image."""
    from detail.api.run_preview import _comparability_key

    new = _built(_meta_shape(CTOR, "none"), 700, BLUE)
    old = _built(_meta_shape(CTOR, "none"), 1000, BLUE)
    if old_scope is None:
        del old["bg_color_scope"]
    else:
        old["bg_color_scope"] = old_scope
    assert _comparability_key(new) == _comparability_key(old)


def test_aligned_runs_with_differing_backgrounds_are_refused():
    """The reported case, as recorded on disk: three aligned max-contrast runs, the third
    recording's background resolved green where the other two resolved blue."""
    from detail.api.run_preview import _comparability_key

    blue, green = [255, 0, 0], [0, 255, 0]
    first = _built(_meta_shape("masked_color_centered_contrast", "rotate"), 729, blue)
    same = _built(_meta_shape("masked_color_centered_contrast", "rotate"), 705, blue)
    third = _built(_meta_shape("masked_color_centered_contrast", "rotate"), 701, green)
    for meta in (first, same, third):
        meta["bg_color_scope"] = "per_recording"
    assert _comparability_key(first) == _comparability_key(same)
    assert _comparability_key(first) != _comparability_key(third)


# --- the guard: every spec built by the real producers -------------------------------- #
def _cases():
    return [
        pytest.param(key, align, id=f"{key}-{align}")
        for key in CONSTRUCTORS
        for align in ("none", "rotate", "mirror")
        if align == "none" or key in ALIGNABLE
    ]


def _meta_shape(constructor: str, align: str, canvas_scope="per_recording") -> dict:
    return {
        "status": "complete",
        "constructor": constructor,
        "model_id": "m",
        "model_variant": "v",
        "readout": None,
        "input_size": 224,
        "canvas_scope": canvas_scope,
        "bg_color_scope": "shared",
        "align": align,
    }


def _built(meta: dict, canvas_px: int, bg: list[int]) -> dict:
    """``meta`` with the renderer_spec an embedding run would record for these inputs."""
    from detail.pipeline.embedding import resolve_renderer_spec
    from detail.transforms.constructors import with_alignment

    spec = resolve_renderer_spec(
        None,
        "track_01",
        [0],
        meta["constructor"],
        resize=224,
        canvas_px=canvas_px,
        bg_color=bg,
    )
    return {**meta, "renderer_spec": with_alignment(spec, meta["align"])}


@pytest.mark.parametrize(("constructor", "align"), _cases())
def test_a_per_recording_canvas_is_allowed_to_differ(constructor, align):
    from detail.api.run_preview import _comparability_key

    meta = _meta_shape(constructor, align)
    assert _comparability_key(_built(meta, 700, BLUE)) == _comparability_key(
        _built(meta, 1000, BLUE)
    )


@pytest.mark.parametrize(("constructor", "align"), _cases())
def test_a_shared_canvas_must_match(constructor, align):
    from detail.api.run_preview import _comparability_key

    meta = _meta_shape(constructor, align, canvas_scope="shared")
    differs = _comparability_key(_built(meta, 700, BLUE)) != _comparability_key(
        _built(meta, 1000, BLUE)
    )
    assert differs == CONSTRUCTORS[constructor].needs_canvas


@pytest.mark.parametrize("recorded_scope", ["shared", "per_recording"])
@pytest.mark.parametrize(("constructor", "align"), _cases())
def test_a_differing_background_never_compares_equal(
    constructor, align, recorded_scope
):
    """Whatever scope the run recorded: runs on disk from before the background was always
    shared say ``per_recording``, and that is exactly where a relaxed check would hide.
    """
    from detail.api.run_preview import _comparability_key

    meta = {**_meta_shape(constructor, align), "bg_color_scope": recorded_scope}
    differs = _comparability_key(_built(meta, 700, BLUE)) != _comparability_key(
        _built(meta, 700, [0, 255, 0])
    )
    assert differs == CONSTRUCTORS[constructor].needs_bg_color


@pytest.mark.parametrize(
    "second",
    [
        pytest.param({"constructor": "masked_color_centered"}, id="constructor"),
        pytest.param({"model_id": "other_model"}, id="model"),
        pytest.param({"model_variant": "large"}, id="variant"),
        pytest.param({"readout": "cls_patch"}, id="readout"),
        pytest.param({"input_size": 224}, id="input_size"),
        pytest.param({"renderer_spec": _spec(1000, resize=224)}, id="resize"),
        pytest.param({"renderer_spec": [{"type": "crop_to_bbox"}]}, id="step_sequence"),
    ],
)
def test_real_mismatches_still_blocked(two_runs, second):
    """Relaxing the derived values must not relax the recipe identity itself."""
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(two_runs(second=second))


def test_missing_scope_fields_read_as_per_recording(experiment_dir, make_embedding_run):
    """A run written before the scopes existed resolved everything per recording."""
    import yaml

    pairs = []
    for i, name in enumerate(("s1", "s2")):
        path = _recording(experiment_dir, name)
        run_dir = make_embedding_run(
            path,
            *_traj(np.random.default_rng(i), 12),
            constructor=CTOR,
            input_size=384,
            renderer_spec=_spec(700 + i * 300),
        )
        meta = yaml.safe_load(
            (run_dir / "embedding.yaml").read_text(encoding="utf-8-sig")
        )
        del meta["canvas_scope"], meta["bg_color_scope"]
        (run_dir / "embedding.yaml").write_text(
            yaml.dump(meta, sort_keys=False), encoding="utf-8"
        )
        pairs.append((path, "emb_01"))
    assert run_preview.validate_comparability(pairs)["model_id"]


def test_pinned_spec_runs_keep_strict_equality(experiment_dir, make_embedding_run):
    """No named constructor (the raw-API path) = no recipe identity, so the whole
    spec must match — a differing canvas there is not a known normalization."""
    pairs = []
    for i, name in enumerate(("s1", "s2")):
        path = _recording(experiment_dir, name)
        make_embedding_run(
            path,
            *_traj(np.random.default_rng(i), 12),
            constructor=None,
            renderer_spec=_spec(700 + i * 300),
        )
        pairs.append((path, "emb_01"))
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(pairs)


def test_pinned_spec_never_matches_a_constructor_run(
    experiment_dir, make_embedding_run
):
    pairs = []
    for i, name in enumerate(("s1", "s2")):
        path = _recording(experiment_dir, name)
        make_embedding_run(
            path,
            *_traj(np.random.default_rng(i), 12),
            constructor=None if i else CTOR,
            input_size=384,
            renderer_spec=_spec(700),
        )
        pairs.append((path, "emb_01"))
    with pytest.raises(ValueError, match="not comparable"):
        run_preview.validate_comparability(pairs)


# --- temporal stacking is part of the recipe identity --------------------------------- #
def _meta(**over):
    base = {
        "status": "complete",
        "constructor": "full_frame",
        "model_id": "m",
        "model_variant": "v",
        "readout": None,
        "input_size": 384,
        "renderer_spec": [{"type": "resize", "size": 384}],
        "canvas_scope": "per_recording",
        "bg_color_scope": "per_recording",
    }
    base.update(over)
    return base


def test_runs_differing_only_in_temporal_stack_are_not_comparable():
    """It changes what the encoder was shown — three frames' greyscale in the colour channels
    versus one frame — so a cosine comparison across the two is meaningless."""
    from detail.api.run_preview import _comparability_key

    plain = _comparability_key(_meta(temporal_stack_s=0.0))
    stacked = _comparability_key(_meta(temporal_stack_s=0.2))
    assert plain != stacked


def test_a_run_recorded_before_stacking_matches_an_explicit_zero():
    """Old runs have no key and used the single-frame input; they must not all become
    incomparable with new runs for the sake of a missing field."""
    from detail.api.run_preview import _comparability_key

    legacy = _meta()
    legacy.pop("temporal_stack_s", None)
    assert _comparability_key(legacy) == _comparability_key(_meta(temporal_stack_s=0.0))


def test_the_same_stack_stays_comparable():
    from detail.api.run_preview import _comparability_key

    assert _comparability_key(_meta(temporal_stack_s=0.2)) == _comparability_key(
        _meta(temporal_stack_s=0.2)
    )
