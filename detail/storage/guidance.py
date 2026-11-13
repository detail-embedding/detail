"""Machine-level choices about the setup checklist (§1): ``~/.config/detail/guidance.yaml``.

One choice so far: the person opted out of keypoints (§26, #717). Every experiment is set up
with a keypoint model by default, because alignment and pose in the embedding — and through
them every pose-aware analysis — need one; opting out is allowed, and is a decision rather
than a dismissed nag, so it lives here and not in the GUI's ``guidance/dismissed`` setting
(which is cleared whenever its step reads as done).

Located through ``locations.CONFIG_DIR`` at call time, so the one override that relocates this
machine's config (``DETAIL_CONFIG_DIR``, and the test isolation that patches it) covers it too.
"""

from pathlib import Path

import yaml

from detail.core import atomic
from detail.core import yaml_load
from detail.storage import locations

SCHEMA_VERSION = 1
GUIDANCE_FILENAME = "guidance.yaml"


def guidance_path() -> Path:
    return locations.CONFIG_DIR / GUIDANCE_FILENAME


def _read() -> dict:
    path = guidance_path()
    if not path.is_file():
        return {}
    return yaml_load.safe_load(atomic.read_text(path)) or {}


def keypoints_skipped() -> bool:
    """Whether this machine's user chose to go without a keypoint model."""
    return bool(_read().get("keypoints_skipped", False))


def keypoints_opted_out() -> bool:
    """:func:`keypoints_skipped` for a surface explaining something: never raises, and a
    damaged ``guidance.yaml`` reads as not opted out, as the checklist does."""
    try:
        return keypoints_skipped()
    except (OSError, yaml.YAMLError):
        return False


#: Where a keypoint model comes from, on both surfaces a reader might be on.
_SET_UP = (
    "set keypoints up (Models ▸ Set up keypoints…, or `detail keypoints setup start`)"
)


def missing_keypoints_hint(stages: str = "keypoint detect and keypoint track") -> str:
    """What to do about a missing keypoint run, worded for the end of a sentence — the one
    text every surface that explains a keypoint option's absence uses (§26, #717).
    ``stages`` names what the reader would run: both stages by default, ``keypoint detect``
    where only a detect run is wanted (a distill teacher).

    When the person opted out of keypoints, it says so: telling them to run keypoints without
    saying they chose not to reads as a broken feature. Never raises — a damaged
    ``guidance.yaml`` reads as not opted out, as the checklist does.
    """
    if keypoints_opted_out():
        return (
            "keypoints were skipped at setup, so there is no keypoint model; take that back "
            f"(Configuration ▸ Get started ▸ Undo skip), then {_SET_UP}"
        )
    return f"run {stages} on it first, or {_SET_UP}"


def set_keypoints_skipped(skipped: bool) -> None:
    """Record (or withdraw) the choice to go without a keypoint model. Atomic: the readiness
    check reads this file from a worker thread while the GUI writes it."""
    data = {"schema_version": SCHEMA_VERSION}
    data.update({k: v for k, v in _read().items() if k != "schema_version"})
    data["keypoints_skipped"] = bool(skipped)
    path = guidance_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    atomic.replace(tmp, path)
