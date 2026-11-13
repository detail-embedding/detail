"""The keypoint lookups know the built-in methods in a process that has imported none (§26).

**The class: a lookup by id that depends on an import something else happened to make.** Built-in
keypoint methods register themselves when their modules are imported. The GUI form imports them
and every ``Experiment`` launcher imports them first; ``Experiment.test_keypoints`` did not. So
the same call that worked in the app failed from the command line and from a script —
``detail keypoints test --model dinov3_match`` answered that ``dinov3_match`` was not a method
that learns from an example set, ``available: []``. It was found once on the old keypoints stage
and written again on the §26 rebuild, because the fix had been made at the call site. It is now
made in the lookups (``models.resolve.ensure_keypoint_methods``), which also covers a bare
export asking a run's method for its score cut-off.

**Why this is a fresh interpreter.** The suite's own process has long since imported every method,
and the registry fixtures add their own, so an in-process test passes with the fix removed. The
registry is a fact about a process, so the process has to be a new one.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import detail

ROOT = Path(detail.__file__).resolve().parents[1]

SNIPPET = textwrap.dedent("""
    import sys

    from detail.api.experiment import Experiment
    from detail.models.resolve import keypoint_method_class

    experiment = Experiment.create("e", sys.argv[1])
    experiment.create_recording("recording_001")
    try:
        experiment.test_keypoints(
            example_set="ex_01",
            tracking={"recording_001": "track_01"},
            model_id="no_such_method",
        )
    except ValueError as error:
        print(error)
    print(keypoint_method_class(None, "dinov3_match").__name__)
    """)


def test_a_bare_process_finds_the_built_in_methods(tmp_path):
    done = subprocess.run(
        [sys.executable, "-c", SNIPPET, str(tmp_path)],
        capture_output=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr[-800:]
    refusal, method_class = done.stdout.strip().splitlines()[-2:]
    assert "'no_such_method' is not one" in refusal
    # the built-in, which only an import of its module makes available
    assert "dinov3_match" in refusal
    assert method_class == "DINOv3Match"
