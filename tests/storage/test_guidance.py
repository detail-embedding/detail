"""The keypoints opt-out, and the one hint every keypoint-absence message uses (§26, #717).

**The class.** A message telling the reader to run keypoints, written beside the shared hint
rather than through it. Once a machine can opt out of keypoints, "run Keypoints first" sends a
reader who opted out to the one step they chose to skip, with nothing saying why the option in
front of them is missing — it reads as a broken feature. Six such messages existed when the
opt-out shipped: the embedding form's reason line and launch refusal, the API/CLI refusal in
``pipeline/embedding.py`` (both for alignment/pose and for ``pose='draw'``), ``run_spec``'s
keypoints-spec refusal, the input-image preview, and the keypoint distill dialog. All of them now
end with ``storage.guidance.missing_keypoints_hint()``, which names the opt-out and how to take
it back. The keypoint export reasons are kept short (the GUI prints them after the target's name),
so they append "keypoints were skipped at setup" only when it is true.

**The guard** scans ``detail/`` for a hand-written "run keypoints first" (with or without "on
<run>"): three of the six were phrased that way. The other three had their own wording and are
pinned by skipped-wording tests beside their existing ones (``test_embedding_alignment_form``,
``test_embedding_alignment``, ``test_exporters``, ``test_fit``) — a gap for the scan, recorded in
``docs/recorded_classes.md``. "Run keypoints again" (a run exists and has to be redone) is not an
absence message and is not matched.
"""

from detail.storage import guidance


def test_not_opted_out_until_the_person_says_so():
    assert not guidance.keypoints_skipped()
    assert not guidance.keypoints_opted_out()


def test_the_choice_round_trips_with_schema_version_first():
    guidance.set_keypoints_skipped(True)
    assert guidance.keypoints_skipped()
    first = guidance.guidance_path().read_text(encoding="utf-8-sig").splitlines()[0]
    assert first == "schema_version: 1"
    guidance.set_keypoints_skipped(False)
    assert not guidance.keypoints_skipped()


def test_the_hint_says_how_to_get_a_keypoint_model():
    hint = guidance.missing_keypoints_hint()
    assert hint.startswith("run keypoint detect and keypoint track on it first")
    assert "Set up keypoints" in hint and "detail keypoints setup start" in hint
    assert "skipped" not in hint


def test_the_hint_names_the_opt_out_and_how_to_take_it_back():
    guidance.set_keypoints_skipped(True)
    hint = guidance.missing_keypoints_hint()
    assert hint.startswith("keypoints were skipped at setup")
    assert "Undo skip" in hint and "Set up keypoints" in hint


def test_a_damaged_file_reads_as_not_opted_out_without_raising():
    guidance.guidance_path().parent.mkdir(parents=True, exist_ok=True)
    guidance.guidance_path().write_text(
        "keypoints_skipped: [unclosed", encoding="utf-8"
    )
    assert not guidance.keypoints_opted_out()
    assert guidance.missing_keypoints_hint().startswith(
        "run keypoint detect and keypoint track on it first"
    )


# --- the guard: every keypoint-absence message goes through the hint (module docstring) - #
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import detail  # noqa: E402

# "run keypoints first" (the stage before §26) and "run keypoint detect [and keypoint
# track] [on it] first" (its two stages after): one sentence shape, three spellings.
_ABSENCE = re.compile(
    r"\b[Rr]un\s+[Kk]eypoint(?:s|\s+detect|\s+track)\b"
    r"(?:\s+and\s+keypoint\s+track)?\s+(?:on\s+\S+\s+)?first\b"
)


def absence_messages(source: str) -> list[int]:
    """Lines of ``source`` that tell the reader to run keypoints first, in their own words."""
    return [i for i, line in enumerate(source.splitlines(), 1) if _ABSENCE.search(line)]


def test_every_keypoint_absence_message_uses_the_hint():
    package = Path(detail.__file__).parent
    offenders = [
        f"{path.relative_to(package.parent).as_posix()}:{line}"
        for path in sorted(package.rglob("*.py"))
        if path.name != "guidance.py"
        for line in absence_messages(path.read_text(encoding="utf-8-sig"))
    ]
    assert not offenders, (
        "say why keypoints are missing through storage.guidance.missing_keypoints_hint(), "
        f"which names an opt-out: {offenders}"
    )


def test_the_scan_would_catch_a_hand_written_absence_message():
    source = 'msg = "No keypoints run here: run Keypoints first."\nok = "run keypoints again"\n'
    assert absence_messages(source) == [1]
    assert absence_messages('x = "run keypoints on trk_01 first"') == [1]
    assert absence_messages('x = "run keypoint detect and keypoint track first"') == [1]
    assert absence_messages('x = "run keypoint detect on it first"') == [1]
