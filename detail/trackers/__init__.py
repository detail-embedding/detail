"""Trackers (§13) — the algorithms that turn per-frame detections into tracks.

Each concrete ``Tracker`` lives in its own module and self-registers on import (Tenet 1);
``pipeline.tracking`` is what imports them, so ``TRACKERS`` is populated before any read.
There is no ``import_trackers`` helper — importing the stage is what a caller already does.

**``anchor_tracker`` is the default (``configs/tracking.DEFAULT_TRACKER``)**, because every
recording this tool has been pointed at so far holds one animal, and the single-subject
trackers give that animal one track id for the whole recording — which is what every later
stage reads. Neither tracker follows more than one subject; the multi-subject tracker that
linked by frame-to-frame overlap was removed (``docs/removed_capabilities.md``).

The default used to be declared nowhere, which meant it was declared everywhere: the CLI
option said ``iou_tracker`` and the GUI combo opened on whatever sorted first. Name it in one
place, and the surfaces have nothing left to disagree about.
"""
