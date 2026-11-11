"""The frame-stride contract (§4): how a downstream stage composes with its input.

Every stage counts *source* frames, so a stage can only thin further by a whole multiple
of what it received. These are the rules that keep a run from asking for frames that were
never processed — which would not error anywhere downstream, it would just quietly analyse
a different set of frames than the one requested.
"""

import pytest

from detail.api import run_spec
from detail.core.utils import resolve_downstream_every_n


def test_omitting_the_stride_inherits_the_upstream_cadence():
    assert resolve_downstream_every_n(None, 4) == 4
    assert resolve_downstream_every_n(None, 1) == 1


def test_a_multiple_of_the_upstream_stride_is_allowed():
    assert resolve_downstream_every_n(4, 2) == 4
    assert resolve_downstream_every_n(2, 2) == 2  # same cadence: keep everything
    assert resolve_downstream_every_n(3, 1) == 3  # every frame upstream: anything goes


def test_a_non_multiple_is_refused_and_names_what_is_reachable():
    # 3 against an upstream of 2 asks for frames 0, 3, 6 — but only 0, 2, 4, … exist.
    with pytest.raises(ValueError, match="not a multiple") as exc:
        resolve_downstream_every_n(3, 2)
    assert "2, 4, 6, 8" in str(exc.value)  # the achievable values, not a rounded guess


def test_a_finer_stride_than_the_input_is_refused():
    """Frames the upstream skipped cannot be recovered by asking for more of them."""
    with pytest.raises(ValueError, match="finer than"):
        resolve_downstream_every_n(2, 4)


def test_the_spec_label_names_a_thinned_run_and_stays_quiet_otherwise():
    """The cadence has to be visible where an upstream run is picked — it decides what the
    downstream stage may ask for."""
    assert run_spec.every_n_label(4) == " · every 4th frame"
    assert run_spec.every_n_label(2) == " · every 2nd frame"
    assert run_spec.every_n_label(3) == " · every 3rd frame"
    assert run_spec.every_n_label(21) == " · every 21st frame"
    assert run_spec.every_n_label(1) == ""  # every frame: nothing to say
    assert run_spec.every_n_label(None) == ""


def test_a_record_written_before_frame_stride_existed_reads_as_every_frame():
    """``every_n_frames`` joined detection's identity fields after runs existed on disk.

    Without a declared default those records resolve to a per-recording "unresolved"
    sentinel that never matches another recording — so no batch could ever again reuse
    them by kind. The absence has one honest reading: the run processed every frame.
    """
    legacy = {"id": "det_01", "model_id": "sam3", "conf_threshold": 0.5}
    current = dict(legacy, every_n_frames=1)
    assert run_spec.freeze_spec(
        run_spec.spec_of({}, "detection", legacy)
    ) == run_spec.freeze_spec(run_spec.spec_of({}, "detection", current))
