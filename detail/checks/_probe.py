"""Fresh-interpreter probe (§1) for the import-sensitive checks.

Spawned as ``python -m detail.checks._probe`` by ``api/checks.run_checks``: because it is a
brand-new interpreter, a mid-session ``pip install`` is visible here even though the parent
app has an older module loaded. It runs every ``isolation="subprocess"`` check and prints
their results as one JSON line — the *sole* stdout write, so the parent can parse it past
any backend import chatter. Each check is guarded, so one failure can't sink the probe.
"""

import json
import sys


def main() -> int:
    from detail.checks import import_checks
    from detail.core.dataclasses import CheckResult, CheckStatus
    from detail.core.registries import CHECKS

    import_checks()
    results: dict[str, dict] = {}
    for check_id in sorted(CHECKS):
        cls = CHECKS[check_id]
        if cls.isolation != "subprocess":
            continue
        try:
            result = cls().run()
        except Exception as exc:  # isolation boundary — never sink the whole probe
            result = CheckResult(
                check_id, CheckStatus.UNKNOWN, f"{type(exc).__name__}: {exc}"
            )
        results[check_id] = result.to_dict()
    print(
        json.dumps(results)
    )  # sole stdout write — the parent parses the last JSON line
    return 0


if __name__ == "__main__":
    sys.exit(main())
