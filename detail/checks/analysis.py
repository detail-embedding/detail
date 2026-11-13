"""Optional analysis setup check (§1): Cluster profile's SHAP.

It is not needed to run anything, so it never FAILs: a missing SHAP is a WARN that names what
you get without it, and ``detail doctor`` exits 0 over it. It backs the setup checklist's
optional *Cluster profile* step (``gui/readiness_guide.py``).

``shap`` is import-sensitive, so it runs in the throwaway-subprocess probe like the other
package checks (``checks/environment.py``): an install made while the app is open shows on the
next re-check.
"""

from detail.core.dataclasses import CheckResult, CheckStatus
from detail.core.protocols import Check
from detail.core.registries import register_check
from detail.help import HelpTopic

#: Where the check points for its fix: Configuration ▸ Cluster profile.
FIX = HelpTopic("gui", "cluster-profile-setup")


@register_check("shap")
class ShapInstalled(Check):
    check_id = "shap"
    label = "SHAP (Cluster profile)"
    group = "Analysis (optional)"
    isolation = "subprocess"
    fix = FIX

    def run(self, target=None) -> CheckResult:
        from detail.api import extras

        version = extras.package_version("shap")
        if version is not None:
            return CheckResult(
                self.check_id,
                CheckStatus.OK,
                f"shap {version}: Cluster profile explains with SHAP",
            )
        return CheckResult(
            self.check_id,
            CheckStatus.WARN,
            "not installed: Cluster profile falls back to permutation importance "
            "(detail install-extra analysis)",
        )
