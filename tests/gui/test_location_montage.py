"""The location montage window (§16 "Preview and montage"), on a real tracked video: it builds
off the UI thread, plays, follows the player's speed, and exports a GIF and the last tracing
frame."""

import pytest

from tests.api.test_location_montage import walked  # noqa: F401 — the fixture
from tests.gui.conftest import drain_busy_calls, drain_until


@pytest.fixture
def window(qapp, walked):  # noqa: F811
    from detail.gui.location_montage import LocationMontageDialog

    experiment, plots = walked
    dialog = LocationMontageDialog(experiment, plots, plots.recording_ids, None)
    dialog.resize(900, 700)
    dialog.show()
    assert drain_until(qapp, lambda: dialog._built is not None)
    yield dialog
    dialog.close()
    drain_busy_calls(qapp)


def test_it_builds_one_tile_per_recording_and_shows_a_picture(qapp, window):
    assert len(window._built.montage.clips) == 1
    assert drain_until(qapp, lambda: window._view.pixmap_size() is not None)
    assert window._export_btn.isEnabled() and window._last_btn.isEnabled()


def test_the_player_s_speed_is_the_export_speed(qapp, window):
    window._scrubber.set_speed(2.0)
    assert window.view().speed == 2.0
    # Backward plays at half speed; an export is always forward.
    window._scrubber.set_speed(-0.5)
    assert window.view().speed == 0.5


def test_a_window_change_rebuilds_and_a_tick_only_redraws(qapp, window):
    first = window._built
    window._widgets["mask_trail"].setChecked(False)
    assert window._built is first  # drawing only
    window._widgets["full_frame"].setChecked(True)
    assert drain_until(
        qapp, lambda: window._built is not None and window._built is not first
    )
    assert window.view().full_frame


def test_it_exports_a_gif_and_the_last_tracing_frame(qapp, window, tmp_path):
    window.export_to(str(tmp_path / "m.gif"))
    assert drain_until(qapp, lambda: (tmp_path / "m.gif").is_file())
    window.export_to(str(tmp_path / "last.png"), last_frame=True)
    assert drain_until(qapp, lambda: (tmp_path / "last.png").is_file())
    assert drain_until(qapp, lambda: "Wrote" in window._status.text())


def test_no_layer_ticked_says_so_instead_of_drawing(qapp, window):
    window._widgets["mask_trail"].setChecked(False)
    window._widgets["mask"].setChecked(False)
    assert "turn on at least one" in window._view.text()
