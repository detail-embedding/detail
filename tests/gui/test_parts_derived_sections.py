"""A section derived from the video parts, rebuilt by only some of the actions that change them.

The recording page builds several sections from ``recording.video_parts``. The parts table is
one; the scene geometry section is another — its preview draws part 0's first frame, and its
*Edit scene geometry…* button is greyed until a part exists. *Add video part*, *Locate video* and
*Move up / down* each rebuilt the parts table and nothing else, so after adding the first part the
page showed a table listing it above a preview still saying "No video imported", beside a greyed
editor button. After a reorder the preview kept drawing the old part 0.

It hid because each handler is correct at the line where it is written — it rebuilds the thing
it obviously changed — and because every construction test builds the page *after* its parts
exist, where both sections read the same state and agree.

The rule: an action that changes the parts goes through ``RecordingDetailPage._parts_changed``,
the one entry point that rebuilds every section derived from them. A handler calling
``_rebuild_parts`` directly is the bug coming back, so this scans the page's source for one,
fail-closed, and a handler written tomorrow is covered the day it is written.
``test_pages.py::test_recording_detail_preview_follows_an_added_video_part`` is the behavioural
half, through the page's own *Add video part* path.
"""

import ast
import inspect
import textwrap

#: ``__init__`` builds every section once, in order, before ``_parts_changed``'s other half exists.
_ALLOWED = ("__init__", "_parts_changed")


def _direct_parts_rebuilders(cls) -> list[str]:
    """Methods of ``cls`` that call ``_rebuild_parts`` other than the allowed two."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name in _ALLOWED:
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_rebuild_parts"
            ):
                offenders.append(node.name)
    return offenders


def test_every_video_part_change_goes_through_parts_changed():
    from detail.gui.pages import RecordingDetailPage

    assert _direct_parts_rebuilders(RecordingDetailPage) == []


def test_the_scan_would_catch_a_handler_rebuilding_only_the_table():
    """Guard the guard: the shape the three handlers had before the fix must be caught."""

    class _Counterexample:
        def _parts_changed(self):
            self._rebuild_parts()

        def _add_part(self):
            self._rebuild_parts()

    assert _direct_parts_rebuilders(_Counterexample) == ["_add_part"]
