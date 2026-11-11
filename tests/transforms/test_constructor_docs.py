"""The Input images page derives from code — guard that derivation."""

from detail.transforms.constructor_docs import describe_constructors
from detail.transforms.constructors import CONSTRUCTORS, DEFAULT_CONSTRUCTOR


def test_describes_every_registered_constructor():
    docs = describe_constructors()
    assert [d.key for d in docs] == list(CONSTRUCTORS)


def test_label_and_description_come_from_code():
    by_key = {d.key: d for d in describe_constructors()}
    for key, recipe in CONSTRUCTORS.items():
        assert by_key[key].label == recipe.label
        # description is the build function's docstring — every recipe must carry one,
        # since that docstring is the single source for the docs page.
        assert by_key[key].description, f"{key} needs a docstring on its build function"


def test_steps_and_default_are_derived():
    by_key = {d.key: d for d in describe_constructors()}
    assert by_key[DEFAULT_CONSTRUCTOR].is_default
    assert sum(d.is_default for d in by_key.values()) == 1
    # every recipe produces the model input size: a resize step, after which only
    # size-preserving background fills (noise / radiating bands) may run.
    size_preserving = {"fill_background_noise", "fill_background_bands"}
    for d in by_key.values():
        assert "resize" in d.steps
        last_resize = max(i for i, s in enumerate(d.steps) if s == "resize")
        assert all(s in size_preserving for s in d.steps[last_resize + 1 :])


def test_uses_mask_reflects_the_recipe():
    by_key = {d.key: d for d in describe_constructors()}
    assert by_key["masked_color_centered"].uses_mask
    assert by_key["silhouette_centered"].uses_mask
    assert not by_key["full_frame"].uses_mask
