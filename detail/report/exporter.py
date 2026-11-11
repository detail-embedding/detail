"""What a re-export into an existing ``exports/`` tree may delete: only what it wrote last time.

An export writes ``stats.csv`` + ``tables/`` + ``figures/`` + ``summary.md`` + ``manifest.json``
under its output directory; the comparison report (§24) is the one writer today. This held a
per-module exporter registry for benchmarks (§18) too, removed with them
(``docs/removed_capabilities.md``).
"""

import json
from pathlib import Path


def previous_artifacts(manifest_path: Path) -> set[str]:
    """The relative paths the last export into this directory recorded writing.

    The manifest is the only record of authorship there is, so it is the only safe basis for
    deleting: a file that is not in it was put there by somebody else.
    """
    if not manifest_path.is_file():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        # A truncated manifest cannot say what was ours, so nothing is.
        return set()
    artifacts = manifest.get("artifacts")
    return set(artifacts) if isinstance(artifacts, dict) else set()


def drop_superseded(out_dir: Path, previous: set[str], current: set[str]) -> None:
    """Remove what the *last* export wrote and this one did not, after this one has written.

    A figure is named after the measurement it shows, and that name changes when the shape of the
    output changes — collapsing the window out of a figure's identity turned
    `n_instances__rear__whole.png` into `n_instances__rear.png`. Left alone, the old file sits
    beside the new one, the detail page globs the directory, and a user is shown a figure drawn by
    code that no longer exists. A real comparison had twenty of them.

    Two things this deliberately is *not*. It is not a sweep by suffix: that deleted every `.csv`,
    `.png` and `.md` in the directory whoever wrote them, so exporting into a folder someone had
    chosen — a manuscript's figure folder, or the comparison's own directory, whose figures the
    open page was displaying — destroyed their files. And it does not run *before* the render:
    a failed export would then leave an empty directory where a complete older one had been.
    """
    for relative in sorted(previous - current):
        path = out_dir / relative
        # A manifest carries relative paths, but never trust one to stay inside the tree.
        if not path.is_file():
            continue
        if out_dir.resolve() not in path.resolve().parents:
            continue
        path.unlink()
