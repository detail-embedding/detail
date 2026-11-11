"""The check every analysis module's live run is held to: its messages name its declared phases.

``AnalysisModule.phases`` is a claim about the runner's own messages (§16, issue #393), read by
the run page through ``core.progress.message_phase``. Each module test hands the messages a real
run just emitted to :func:`assert_messages_name_their_phases`, so a declaration and the runner it
describes cannot drift apart without that module's test failing.

The class this guards: a message the page reads as the wrong phase, or as none. The first
version matched a phase's words anywhere in a message, and query-eval's querying messages carry
``loading <subject>`` in their note — every run showed "loading" again once per subject, and
eight hand-rolled per-module checks each asked only whether a phase's words appeared *somewhere*,
which that message passes. Asking "which phase does the page draw for this message" is the
question that fails on it.
"""

from detail.core.progress import message_phase


def assert_messages_name_their_phases(messages, phases: tuple[str, ...]) -> None:
    """Once the first phase has begun, every message maps to a declared phase (a ``done`` head
    closes a pass), and every phase declared is one the run actually reached.

    A message *before* the first phase ("starting", "preparing 3 recordings") may name none —
    every row queued is the truth then. After it, an unplaced message flashes every row back to
    queued mid-run, which reads as the run starting over.
    """
    messages = [m for m in messages if m]
    assert messages, "the runner emitted no progress messages at all"
    mapped = [(m, message_phase(m, phases)) for m in messages]
    first = next((k for k, (_, i) in enumerate(mapped) if i is not None), len(mapped))
    unnamed = [m for m, i in mapped[first:] if i is None]
    assert not unnamed, (
        f"mid-run messages the run page cannot place in {phases} — it would draw every phase "
        f"queued while they show: {unnamed}"
    )
    reached = {i for _, i in mapped}
    missing = [p for i, p in enumerate(phases) if i not in reached]
    assert not missing, f"declared but never emitted: {missing}; messages: {messages}"
