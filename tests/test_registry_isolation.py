"""The autouse registry-isolation fixture (conftest) keeps a fake registered mid-test
from leaking into the next — the cause of order-dependent failures like a fake model
sorting ahead of a real one in ``sorted(MODELS)``. The two tests run in definition order
within the module: the first registers a fake in its body, the second asserts it is gone.
"""

from detail.core.protocols import DetectionModel
from detail.core.registries import MODELS, register_model

_LEAK_ID = "registry_isolation_probe"


class _Probe(DetectionModel):
    model_id = _LEAK_ID
    supported_cue_types = []
    supported_backends = ["cpu"]

    def load(self, device):
        pass


def test_registering_a_fake_in_body_is_visible_within_the_test():
    assert _LEAK_ID not in MODELS  # clean slate — no earlier test leaked it in
    register_model(_LEAK_ID)(_Probe)
    assert _LEAK_ID in MODELS


def test_the_fake_does_not_leak_into_the_next_test():
    assert _LEAK_ID not in MODELS  # the fixture restored MODELS after the previous test
