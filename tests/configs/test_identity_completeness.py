"""A knob that changes an artifact must be part of that artifact's identity.

The second face of the same recurring bug. ``temporal_stack_s`` changed what the encoder was shown
— one frame versus three frames' greyscale in the colour channels — but was not in the recipe
identity, so ``emb_01`` and ``emb_03`` were judged *comparable*. A clusterer pooling bouts across
the two would have mixed incompatible spaces and reported nothing wrong. Separately,
``cluster_rule`` had to be added to the persisted clusterer params by hand; the next one would have
been forgotten the same way. (The clusterer half of this file went with the clusterers,
``docs/removed_capabilities.md``.)

Neither is visible to a behavioural test: every individual run is correct, and only *comparing*
two of them exposes the gap — which nothing does until someone pools them.

The allow-lists are claims, not exemptions: each entry asserts the field cannot change the
artifact. They are themselves checked against the live field set, so a stale name cannot silently
excuse a new field (the idiom from the removed ``tests/test_cli_cluster_parity.py``).
"""

import pytest

from detail.configs.embedding import EmbeddingConfig

# EmbeddingConfig fields that do not change the embedding *values*.
_NOT_PART_OF_THE_EMBEDDING = {
    "tracking_run_id": "selects which frames are embedded, recorded separately on the run",
    "all_tracks": "selects which tracks, not how any of them is embedded",
    "batch_size": "throughput only; the same images either way",
    "canvas_scope": "recorded as a scope beside renderer_spec and compared there",
    "resize": "feeds the constructor, so it reaches the comparison through renderer_spec",
    "keypoint_run_id": "an id; the keypoint track run reaches the recipe as its identity "
    "(keypoint_track_identity), never as the per-recording counter",
    "heading": "a definition's name; the recipe carries the resolved definition "
    "(heading_definition), so a renamed or edited definition cannot merge two runs",
}


def test_every_embedding_knob_is_part_of_the_recipe_identity():
    """Otherwise two embeddings computed differently compare as interchangeable."""
    from detail.api.run_preview import _RECIPE_KEYS

    missing = [
        name
        for name in EmbeddingConfig.model_fields
        if name not in _RECIPE_KEYS and name not in _NOT_PART_OF_THE_EMBEDDING
    ]
    assert not missing, (
        "these change the embedding but are not in run_preview._RECIPE_KEYS, so two runs "
        f"differing only in them would be judged comparable: {sorted(missing)}"
    )


@pytest.mark.parametrize(
    "allow_list,model",
    [
        (_NOT_PART_OF_THE_EMBEDDING, EmbeddingConfig),
    ],
)
def test_the_allow_lists_do_not_excuse_fields_that_no_longer_exist(allow_list, model):
    stale = sorted(set(allow_list) - set(model.model_fields))
    assert not stale, f"allow-list entries for fields that no longer exist: {stale}"


def test_the_recipe_check_would_catch_a_missing_key(monkeypatch):
    """Guard the guard: drop a live recipe key and the check must fail.

    It used to drop ``temporal_stack_s``, which was the field the guard was written for. That
    knob has since been retired from ``EmbeddingConfig`` (measured worse), so removing it from
    ``_RECIPE_KEYS`` would no longer make this check fail — the test would have passed while
    testing nothing. Pointed at a live field instead.
    """
    import detail.api.run_preview as rp

    monkeypatch.setattr(
        rp, "_RECIPE_KEYS", tuple(k for k in rp._RECIPE_KEYS if k != "model_variant")
    )
    with pytest.raises(AssertionError, match="model_variant"):
        test_every_embedding_knob_is_part_of_the_recipe_identity()


# A recipe key that is no longer a config field, and why it must stay. Retiring a knob does not
# retire the runs made with it: they are on disk recording a value, they were computed
# differently, and dropping the key would make them compare as interchangeable with runs that
# were not — a false *merge*, which is the one direction a comparability check must never fail in.
_RETIRED_RECIPE_KEYS = {
    "temporal_stack_s": "temporal stacking was removed after measuring worse (14/96 configs "
    "improved, mean -0.094 AMI); the emb_03 runs on disk still record 0.2 and must stay "
    "incomparable with single-frame runs",
}


# Not every recipe key is a user knob. A derived run's basis was *computed* by the derive
# module (since removed), not configured — it has no EmbeddingConfig field and never will, and
# it is recipe identity because two runs can agree on every configured field and still hold
# different transforms.
_COMPUTED_RECIPE_KEYS = {
    "derived_basis_digest": "content hash of the fitted transform, written by derive_run; two "
    "derived runs can carry an identical provenance tag over different bases",
    "heading_definition": "the resolved {front, back} sets copied from the named definition "
    "at validate time (§14); the name is the knob, the sets are the identity",
    "keypoint_track_identity": "the keypoint track run's spec identity (method, skeleton "
    "digest, example-set digest, settings), resolved at validate time; never its id",
    "channel_order": "the pixel order the model was handed, fixed by the code rather than "
    "configured; runs before the channel-order fix embedded BGR and record nothing (read as "
    "bgr), so they must stay incomparable with RGB runs",
}


def test_every_recipe_key_is_either_a_live_field_or_a_declared_retirement():
    """The other direction of the identity check, which nothing covered.

    ``test_every_embedding_knob_is_part_of_the_recipe_identity`` walks config -> recipe. Nothing
    walked recipe -> config, so a key could linger with no field behind it and no one would say
    whether that was deliberate. Retiring ``temporal_stack_s`` made that case real.
    """
    from detail.api.run_preview import _RECIPE_KEYS

    fields = set(EmbeddingConfig.model_fields)
    unexplained = [
        k
        for k in _RECIPE_KEYS
        if k not in fields
        and k not in _RETIRED_RECIPE_KEYS
        and k not in _COMPUTED_RECIPE_KEYS
    ]
    assert not unexplained, (
        "these recipe keys have no EmbeddingConfig field behind them. If a knob was retired, "
        f"say so in _RETIRED_RECIPE_KEYS; if it was renamed, update the key: {unexplained}"
    )


def test_a_retirement_entry_names_a_key_that_is_still_in_the_recipe():
    """A stale retirement claim is the same bug one level up."""
    from detail.api.run_preview import _RECIPE_KEYS

    stale = sorted(set(_RETIRED_RECIPE_KEYS) - set(_RECIPE_KEYS))
    assert not stale, (
        "listed as a retired recipe key but no longer in _RECIPE_KEYS — dropping the key is "
        f"what makes old runs falsely comparable, so this needs a look: {stale}"
    )


# KeypointDetectConfig fields that do not change the placed points (§26 "Identity").
_NOT_PART_OF_THE_DETECT = {
    "detection_run_id": "selects which detections get points; the lineage parent, by ref",
    "example_set_id": "a selection; what the run learned from is the example_set_digest",
    "batch_size": "throughput only; the same crops either way",
}


def test_every_keypoint_detect_knob_is_identity_or_declared_not_to_be():
    """Two detect runs that differ in a setting that moves a point must never be judged the
    same spec (#626 merged runs across a renamed skeleton and a changed filter default).
    """
    from detail.configs.keypoint_detect import KeypointDetectConfig
    from detail.pipeline.keypoint_detect import KeypointDetectModule

    identity = set(KeypointDetectModule.identity_fields)
    missing = [
        name
        for name in KeypointDetectConfig.model_fields
        if name not in identity and name not in _NOT_PART_OF_THE_DETECT
    ]
    assert not missing, (
        "these change the placed points but are not in KeypointDetectModule.identity_fields, "
        f"so two runs differing only in them would be judged the same: {sorted(missing)}"
    )
    stale = sorted(
        set(_NOT_PART_OF_THE_DETECT) - set(KeypointDetectConfig.model_fields)
    )
    assert not stale, f"allow-list names fields that no longer exist: {stale}"
    # and every identity field is written to the run record by extra_record_fields
    config = {
        "detection_run_id": "det_01",
        "model_id": "dinov3_match",
        "skeleton_digest": "x",
        "n_candidates": 5,
    }
    recorded = set(KeypointDetectModule().extra_record_fields(config))
    assert identity <= recorded, sorted(identity - recorded)


_NOT_PART_OF_THE_TRACK = {
    "detect_run_id": "selects which detect run is settled; the lineage parent, by ref",
    "tracking_run_id": "an id; the tracking run reaches identity as its spec "
    "(tracking_identity), never as the per-recording counter",
    "match_min_inside": "reaches identity inside match (min_inside)",
    "match_mask_grow": "reaches identity inside match (mask_grow)",
}


def test_every_keypoint_track_knob_is_identity_or_declared_not_to_be():
    """A track run's identity is the tracking run it assigned against (as its spec), the
    match, its method and every field of that method's config (§26): a changed filter or
    match default must produce a new identity, never a false merge."""
    from detail.configs.keypoint_track import METHOD_CONFIGS, KeypointTrackConfig
    from detail.pipeline.keypoint_track import KeypointTrackModule

    identity = set(KeypointTrackModule.identity_fields)
    assert identity == {"tracking_identity", "match", "method", "config"}
    stale = sorted(set(_NOT_PART_OF_THE_TRACK) - set(KeypointTrackConfig.model_fields))
    assert not stale, f"allow-list names fields that no longer exist: {stale}"
    missing = [
        name
        for name in KeypointTrackConfig.model_fields
        if name not in identity and name not in _NOT_PART_OF_THE_TRACK
    ]
    assert not missing, sorted(missing)
    # the method config is recorded whole, so every method field reaches identity through it
    config = {"detect_run_id": "kd_01", "method": "temporal_decode"}
    config["config"] = METHOD_CONFIGS["temporal_decode"]().model_dump()
    config["match"] = {"min_inside": 0.7, "mask_grow": 0.1}
    recorded = KeypointTrackModule().extra_record_fields(config)
    assert recorded["match"] == {"min_inside": 0.7, "mask_grow": 0.1}
    assert identity <= set(recorded)
    assert (
        recorded["config"] == config["config"]
        and recorded["method"] == "temporal_decode"
    )
    assert set(METHOD_CONFIGS["temporal_decode"].model_fields) <= set(
        recorded["config"]
    )


@pytest.mark.parametrize("constructor", [None, "masked_color_centered_contrast"])
def test_the_channel_order_is_part_of_the_recipe_identity(constructor):
    """Not a config knob — the code fixed it — but it changed every image the model saw.

    Runs before the channel-order fix embedded BGR frames and record nothing; a run after it
    records ``channel_order: rgb``. If the key were not compared, an old and a new run of the
    same recipe would pool into one space whose two halves disagree about what red is. Both
    branches of the comparability key: the named-recipe one and the pinned-spec one.
    """
    from detail.api.run_preview import _RECIPE_KEYS, _comparability_key
    from detail.pipeline.embedding import CHANNEL_ORDER

    assert "channel_order" in _RECIPE_KEYS
    assert CHANNEL_ORDER == "rgb"
    old = {
        "constructor": constructor,
        "model_id": "dinov3_vit",
        "model_variant": "vit_large",
        "renderer_spec": [{"type": "crop_to_bbox"}],
    }
    new = {**old, "channel_order": CHANNEL_ORDER}
    explicit_old = {**old, "channel_order": "bgr"}
    assert _comparability_key(old) != _comparability_key(new)
    assert _comparability_key(old) == _comparability_key(explicit_old)


def test_a_new_run_records_its_channel_order_where_comparisons_read_it():
    """``validate_comparability`` reads ``embedding.yaml``, so the manifest entry a new run
    writes has to carry the key, or old and new runs pool. (A benchmark's held-constant check
    read it too, until benchmarks were removed.)"""
    from detail.pipeline.embedding import CHANNEL_ORDER, EmbeddingModule

    entry = EmbeddingModule().extra_record_fields(
        {"tracking_run_id": "track_01", "model_id": "dinov3_vit", "track_ids": [0]}
    )
    assert entry["channel_order"] == CHANNEL_ORDER
