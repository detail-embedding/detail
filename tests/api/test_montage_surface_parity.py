"""Guard: the clip montage's script surfaces open on one montage.

**The class.** A capability reached through more than one surface with a default written into
each — the number of clips, the order, which layers are drawn — so a change to one leaves the
other opening on a different montage with nothing failing (``CLAUDE.md``, *Surface parity*).
The CLI reads its option defaults from ``api.clip_montage``'s constants; this file is what fails
if a default is ever spelled out again there. The GUI's clip grids export what they show
(**Show ▾**'s look, the grid's clips), so they hold no montage default to drift.

``drift`` is factored out so the guard can be shown to catch a drifted default rather than
passing because nothing was compared.
"""

from detail.api import clip_montage as cm
from detail.cli import analysis_cluster_montage


def cli_defaults() -> dict:
    """Each ``cluster-montage`` option's default, by parameter name."""
    return {p.name: p.default for p in analysis_cluster_montage.params}


def api_defaults() -> dict:
    """What the API opens on, under the CLI's parameter names."""
    return {
        # The count follows --overlay (cm.DEFAULT_CLIPS, or cm.OVERLAY_CLIPS for an overlay),
        # so the option is "unset" and resolved from those constants: pinned below.
        "limit": None,
        "sort": cm.DEFAULT_SORT,
        "raw": cm.DEFAULT_LAYERS.raw,
        "mask": cm.DEFAULT_LAYERS.mask,
        "keypoints": cm.DEFAULT_LAYERS.keypoints,
        # The trail follows --mask, so the CLI's default is "unset" rather than a boolean.
        "mask_trail": None,
        "keypoint_trail": cm.DEFAULT_LAYERS.keypoint_trail,
        "aligned": cm.DEFAULT_LAYERS.aligned,
        "context_s": 0.0,
        "blind": False,
    }


def drift(cli: dict, api: dict) -> dict:
    """``{name: (cli, api)}`` for every API default the CLI does not open on."""
    return {
        name: (cli.get(name), want)
        for name, want in api.items()
        if cli.get(name) != want
    }


def test_the_cli_opens_on_the_apis_montage():
    from detail import cli

    assert drift(cli_defaults(), api_defaults()) == {}
    assert (cli._MONTAGE_CLIPS, cli._OVERLAY_CLIPS) == (
        cm.DEFAULT_CLIPS,
        cm.OVERLAY_CLIPS,
    ), "an unset --clips resolves to the API's counts"
    assert cm.DEFAULT_CLIPS == 25 and cm.DEFAULT_SORT == "typical"
    assert cm.DEFAULT_LAYERS == cm.MontageLayers(mask=True, mask_trail=True)


def test_the_sort_choices_are_the_apis():
    (sort,) = [p for p in analysis_cluster_montage.params if p.name == "sort"]
    assert list(sort.type.choices) == list(cm.SORTS)


def test_the_guard_would_catch_a_drifted_default():
    drifted = {**cli_defaults(), "limit": 50, "raw": True}
    assert drift(drifted, api_defaults()) == {"limit": (50, None), "raw": (True, False)}
    assert drift({}, {"limit": 25}) == {
        "limit": (None, 25)
    }, "a missing option is drift too"
