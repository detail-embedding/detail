"""GUI-side module UIs.

Importing this package registers the first-party module UIs (so the launch
dialog can enumerate them). Backend modules stay Qt-free; their bespoke UIs live
here, paired by ``module_id`` (docs/gui_design.md §3.2).
"""

from detail.gui.modules import (  # noqa: F401
    analysis_cluster_lookalikes as _analysis_cluster_lookalikes,
)
from detail.gui.modules import (  # noqa: F401
    analysis_location as _analysis_location,
)
from detail.gui.modules import (  # noqa: F401
    analysis_temporal_clustering as _analysis_temporal_clustering,
)
from detail.gui.modules import (  # noqa: F401
    analysis_profile_describe as _analysis_profile_describe,
)
from detail.gui.modules import analysis_query as _analysis_query  # noqa: F401
from detail.gui.modules import (  # noqa: F401
    analysis_query_eval as _analysis_query_eval,
)
from detail.gui.modules import detection as _detection  # noqa: F401
from detail.gui.modules import embedding as _embedding  # noqa: F401

from detail.gui.modules import keypoint_detect as _keypoint_detect  # noqa: F401

from detail.gui.modules import keypoint_track as _keypoint_track  # noqa: F401
from detail.gui.modules import track_merge as _track_merge  # noqa: F401
from detail.gui.modules import tracking as _tracking  # noqa: F401
