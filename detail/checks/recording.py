"""Per-recording data-integrity checks (§1, §2) — the referenced-video presence/fingerprint.

The first ``target_kind == "recording"`` check: it runs once per recording (the runner
enumerates them and passes a ``RecordingTarget``) and reuses the exact ``validate_source_videos``
the pre-run gate enforces, so the experiment view and a launched run agree on what's wrong. In
process (a file stat + head/tail hash), advisory — it reports; the gate blocks.
"""

from detail.core.dataclasses import CheckResult, CheckStatus, RecordingTarget
from detail.core.protocols import Check
from detail.core.registries import register_check
from detail.help import HelpTopic
from detail.storage import video_integrity


@register_check("video_integrity")
class VideoIntegrityCheck(Check):
    check_id = "video_integrity"
    label = "Video files"
    group = "Recordings"
    target_kind = "recording"
    isolation = "inproc"
    fix = HelpTopic("gui", "recordings")

    def run(self, target: RecordingTarget | None = None) -> CheckResult:
        if target is None:  # a machine-context call — nothing to check
            return CheckResult(
                self.check_id, CheckStatus.SKIPPED, "no recording target"
            )
        issues = video_integrity.validate_source_videos(
            target.recording_dir, target.recording_data
        )
        if not issues:
            return CheckResult(
                self.check_id,
                CheckStatus.OK,
                "present and unchanged",
                target.recording_id,
            )
        detail = "; ".join(f"part {i.part_id}: {i.kind}" for i in issues)
        return CheckResult(self.check_id, CheckStatus.FAIL, detail, target.recording_id)
