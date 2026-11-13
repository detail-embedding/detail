"""``ClusterLookalikesConfig``: the job's settings and the profile settings it carries by name.

The class guarded: **a profile setting a pair never hears.** Each pair is a cluster profile, so a
field ``ClusterProfileConfig`` gains must either reach every pair (``PROFILE_FIELDS``, a field
of this config too) or be one a pair sets itself (``SET_PER_PAIR``). A field in neither would be
silently left at its default for every pair, and nothing would fail.
"""

from detail.configs.cluster_lookalikes import (
    PROFILE_FIELDS,
    SET_PER_PAIR,
    ClusterLookalikesConfig,
)
from detail.configs.cluster_profile import ClusterProfileConfig


def test_every_profile_field_is_carried_or_set_per_pair():
    profile = set(ClusterProfileConfig.model_fields)
    assert set(PROFILE_FIELDS) | set(SET_PER_PAIR) == profile
    assert not set(PROFILE_FIELDS) & set(SET_PER_PAIR)


def test_every_carried_field_is_a_setting_of_the_job():
    assert set(PROFILE_FIELDS) <= set(ClusterLookalikesConfig.model_fields)


def test_a_pair_profile_built_from_the_job_validates():
    from detail.pipeline.cluster_lookalikes import pair_config

    job = ClusterLookalikesConfig(source_run_id="an_01")
    config = ClusterProfileConfig.model_validate(pair_config(job, 3, 7))
    assert config.compare == "node" and config.against == 7
    assert config.leave_out_place
    assert config.n_per_side == job.n_per_side
