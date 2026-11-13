"""ComputeTargetRow's "remote" backend (§6) — opt-in per embedder (allow_remote),
never on by default, since this is a widget seven other forms already share.
"""

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QFormLayout  # noqa: E402

from detail.configs.remote import RemoteTargetConfig  # noqa: E402
from detail.core import utils as core_utils  # noqa: E402
from detail.gui.compute_target import ComputeTargetRow  # noqa: E402
from detail.storage import remote_targets  # noqa: E402


@pytest.fixture
def isolated_targets(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )


def _register(name="tillicum"):
    remote_targets.register_remote_target(
        RemoteTargetConfig(name=name, host=name, remote_workdir="/scratch/x")
    )


def _items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def test_default_never_offers_remote(qapp, isolated_targets):
    _register()
    row = ComputeTargetRow()  # allow_remote defaults False
    row.set_backends(["cpu", "cuda", "mps"])
    assert "remote" not in _items(row.backend)
    assert row.remote_target is None


def test_allow_remote_with_no_targets_does_not_offer_remote(qapp, isolated_targets):
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cpu", "cuda", "mps"])
    assert "remote" not in _items(row.backend)


def test_allow_remote_with_a_target_offers_remote_when_cuda_supported(
    qapp, isolated_targets
):
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cpu", "cuda", "mps"])
    assert "remote" in _items(row.backend)
    assert _items(row.remote_target) == ["tillicum"]


def test_allow_remote_requires_cuda_in_supported_backends(qapp, isolated_targets):
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cpu"])  # a cpu-only model
    assert "remote" not in _items(row.backend)


def test_remote_offered_even_with_no_local_accelerator(
    qapp, isolated_targets, monkeypatch
):
    # A GPU-less laptop with a cuda-only model: `usable` is empty, but a registered
    # cluster still lets the run happen — the whole point of this feature.
    monkeypatch.setattr(core_utils, "cuda_available", lambda: False)
    monkeypatch.setattr(core_utils, "mps_available", lambda: False)
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cuda"])
    assert _items(row.backend) == ["remote"]
    assert row.backend.isEnabled()
    assert row.current_backend() == "remote"


def test_compute_target_returns_remote_target_name(qapp, isolated_targets):
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cpu", "cuda", "mps"])
    row.backend.setCurrentText("remote")
    compute = row.compute_target()
    assert compute.backend == "remote"
    assert compute.remote_target == "tillicum"
    assert compute.device_indices is None


def test_device_and_remote_rows_grey_correctly(qapp, isolated_targets):
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True)
    row.set_backends(["cpu", "cuda", "mps"])

    row.backend.setCurrentText("cuda")
    assert row.device.isEnabled()
    assert not row.remote_target.isEnabled()

    row.backend.setCurrentText("remote")
    assert not row.device.isEnabled()
    assert row.remote_target.isEnabled()


def test_add_to_adds_a_third_row_only_when_remote_is_allowed(qapp, isolated_targets):
    _register("tillicum")
    plain = ComputeTargetRow()
    form = QFormLayout()
    plain.add_to(form)
    assert form.rowCount() == 2

    remote_capable = ComputeTargetRow(allow_remote=True)
    form2 = QFormLayout()
    remote_capable.add_to(form2)
    assert form2.rowCount() == 3


def test_a_cpu_only_row_offers_a_cluster_without_a_model_that_supports_cuda(
    qapp, isolated_targets
):
    """A stage that loads no model (keypoint track) runs on any node's CPU: the cuda gate a
    model's row keeps (above) must not hide the cluster from it."""
    _register("tillicum")
    row = ComputeTargetRow(allow_remote=True, cpu_only=True)
    assert _items(row.backend) == ["cpu", "remote"]
    assert row.current_backend() == "cpu"
    row.backend.setCurrentText("remote")
    compute = row.compute_target()
    assert (compute.backend, compute.remote_target) == ("remote", "tillicum")


def test_a_cpu_only_row_has_no_device_row_and_no_cluster_unless_allowed(
    qapp, isolated_targets
):
    _register("tillicum")
    local = ComputeTargetRow(cpu_only=True)
    assert _items(local.backend) == ["cpu"]
    form = QFormLayout()
    local.add_to(form)
    assert form.rowCount() == 1  # backend only: no state of it picks a CUDA device
