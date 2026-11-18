"""A failed label write's dialog says whether the label is on disk (#24).

An authored edit writes the state and then rebuilds the results. When the rebuild dies the
label **is** saved and the results are out of date, but INTERPRET titled every failure "Label
not saved", which sent people to label the cluster again. The title is read off the error the
edit raises: :class:`~detail.storage.analysis_runs.RebuildFailedError` means saved.
"""

from detail.gui.node_workspace import write_failure_title
from detail.storage.analysis_runs import RebuildFailedError


def test_a_failed_rebuild_reads_as_saved():
    exc = RebuildFailedError("an_01", OSError("disk full"))
    assert write_failure_title(exc) == "Label saved, results not rebuilt"
    assert "the change was saved" in str(exc)
    assert "detail analysis rebuild an_01" in str(exc)


def test_a_refused_label_reads_as_not_saved():
    assert (
        write_failure_title(ValueError("no label called 'REAR'")) == "Label not saved"
    )
