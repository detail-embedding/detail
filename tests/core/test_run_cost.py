"""run_cost's estimate_cost_usd + compute_label — pure functions, no fixtures needed."""

from detail.core import run_cost


def test_estimate_cost_usd_is_none_without_a_configured_rate():
    assert run_cost.estimate_cost_usd(2, 3600.0, None) is None


def test_estimate_cost_usd_multiplies_gpus_hours_and_rate():
    # 2 GPUs, 1 hour, $0.50/GPU-hour -> $1.00
    assert run_cost.estimate_cost_usd(2, 3600.0, 0.50) == 1.0


def test_estimate_cost_usd_fractional_hours():
    # 1 GPU, 30 minutes, $2.00/GPU-hour -> $1.00
    assert run_cost.estimate_cost_usd(1, 1800.0, 2.0) == 1.0


def test_compute_label_none_is_local():
    assert run_cost.compute_label(None) == "local"


def test_compute_label_empty_dict_is_local():
    assert run_cost.compute_label({}) == "local"


def test_compute_label_local_backend_names_the_backend():
    assert run_cost.compute_label({"backend": "cuda"}) == "local (cuda)"
    assert run_cost.compute_label({"backend": "cpu"}) == "local (cpu)"
    assert run_cost.compute_label({"backend": "mps"}) == "local (mps)"


def test_compute_label_remote_names_the_target():
    assert (
        run_cost.compute_label({"backend": "remote", "remote_target": "tillicum"})
        == "remote: tillicum"
    )


def test_compute_label_remote_with_no_target_name_falls_back():
    assert (
        run_cost.compute_label({"backend": "remote", "remote_target": None})
        == "remote: ?"
    )
