---
name: gui-surface-change
description: Build, test and hand off a change under detail/gui/ — the structural rules, the construction smoke, rendering and reading the picture, sweeping the other embedders, and the launch command to end with. Use when touching a GUI page, dialog, shared widget, plot, table, form or run-preview surface.
---

# A GUI surface change

**64% of merged PRs in this repo touch `detail/gui/`.** Most of what goes wrong is not a crash.

## 1. The rules are a registry, not prose

`detail/gui/requirements.py` holds each structural rule as a predicate with a counterexample the
check must reject, swept over every page, dialog and widget the app can build by
`tests/gui/test_requirements.py`, and printed to `docs/reference/gui-requirements.md` at docs-build
time.

- **Read the generated list** for what is enforced; `docs/gui_design.md` §3.1 for the reasoning
  behind each rule — the measurement that produced it and the failure it was written against.
- **Add a new rule as a predicate there first**, with its counterexample. `NOT_YET_SWEPT` is the
  honest backlog, under a ratchet that is lowered as rules migrate and never raised.
- **`requirements.py` wins when copies disagree**, because it is the one that is executed. A 468-line
  prose copy of these rules lived in `CLAUDE.md` and was deleted for being unable to fail — and it
  was already missing six rules the registry enforces.

Two rules worth naming here because they are about *this* procedure rather than a widget:

**Every GUI thread is a `workers.GuiWorker` and names its `receiver=`** — the widget the result is
for — so a delivery landing after a page swap is dropped instead of touching a deleted widget. A
generation counter does not cover this: it expresses supersession, and nothing bumps it on the way
out, because `deleteLater` runs no page code at all.

**No O(recordings) disk I/O on the UI thread.** Read recording metadata through the cache and wrap a loop that wants several fields in `read_cache.pinned()`; read immutable run outputs through `read_cache.cached_read`; off-thread a one-shot first-time cost through `BackgroundCall`. The recordings table once made several round trips per recording, seconds of frozen window on a network share.

## 2. Construct it in a test — necessary, not sufficient

Every page, dialog, run-preview widget and module config/result UI the app can open **must be
instantiated by a test** (with fakes where real data is heavy). This is the net for a class that
import and lint miss:

- `signal.connect(self._method)` where `_method` was renamed raises at **construction** time, not at
  import — only building the widget catches it.
- A subclass **hook override** (`_build_overview`, `_make_preview`, `_overlay`,
  `_set_overview_cursor`) only runs when the *leaf* class is built. Testing the base is not enough.
- A module's `result_widget(experiment, record)` is built **lazily**, so it needs a fake **complete**
  run record to exercise.

The corpus: `tests/gui/test_smoke_construct.py` (pages + every `GUI_MODULES` config form,
data-driven), `tests/gui/test_run_preview.py`, and `tests/gui/test_run_detail.py` (the end-to-end
net — `RunDetailPage` → each module's `result_widget` against a `completed_runs_experiment` fixture
with real on-disk runs). **When you add a page, dialog, run-preview or module UI, add it to the
relevant list**, and extend the fixture for a new module's result view.

When you rename or remove a method on a shared GUI base, run `pytest tests/gui -q` — the missed call
sites surface there.

## 3. Verify through the real entry point

Constructing a widget proves it *builds*, not that the user **reaches it the way you think**. Before
claiming a GUI feature works, trace the path a user takes — Analysis → *New analysis…* → the module's
config form → its preview dialog — and exercise *that*, not a widget you instantiated in isolation.

A feature that renders in a private harness but sits behind an unrelated surface, or behind a second
duplicate surface, is not done. **When a user says they do not see something you built, look at
*their* screen — ask for a screenshot — before re-asserting from your own render.**

## 4. Render it and look at the image

Asserting on widget state cannot see a black background on a package of white plots, an axis cropped
to its data, tick labels overlapping into a smear, or 325px of dead grid beside two rows. **All four
shipped past a green suite in one PR** and were found by the user opening the app.

```
widget.grab().save(path)
```

with the widget **shown and resized** — an unshown widget is never laid out — and then read the
picture.

**A widget that was never shown is not in the state you are asserting about.** An unlaid-out
`QTableView` reports the default `35.0` axis width whatever `setWidth` was called with; a pyqtgraph
`ViewBox` sits at its 0–1 default whatever the data is; `isVisible()` is False for every child of a
widget nobody showed. Three guards in one PR passed against their own reverted fix on exactly this.
If the property under test is geometry, visibility or layout: `resize()` + `show()` +
`processEvents()` first — and if a **resize** is what triggers the bug, resize again after the draw.

Geometry is measured **after** the layout has run, and a scrolled row has two layouts: inside a
`widgetResizable` `QScrollArea`, `_body.adjustSize()` then `layout.activate()`. Both are ~0.3ms, and
skipping them left a divider 40px shy of centre and queued 300 thumbnail decodes in the wrong order
because every cell reported `width() == 0`.

## 5. Sweep the other embedders — this is where the bugs are

A shared widget (`SpatialMap`, `FrameScrubber`, `ConfigForm`, `ClusterClipStrip`, `ImageView`) is
embedded by surfaces edited at different times, so a capability added while working on one reaches
only that one. Nine PRs state the ratio outright:

| what was added | reached |
|---|---|
| the decode scrim's suppression predicate | **1 of 8** embedders |
| *Start from a previous run* | **2 of 9** analysis modules, no processing module |
| the dialog scroll rule via a helper | **8 of 44** dialogs |
| the video-required guard on scrub surfaces | **2 of 12** |
| the sortable-headers row-identity fix | 1 of its 2 sites, shipping with the other live |

So: **`grep -rn "WidgetName()" detail/gui/` before you start**, and wire every embedder in the same
change. Then the two mechanisms, because a cue and a fact fail differently:

- **An input that decides whether the output is *correct* has no default.** Pass it as a required
  keyword. A default spelling "as before" is where a missing fact hides — the second surface keeps
  rendering, plausibly and wrongly, and no test fails.
- **A cue is derived from the wiring, never set beside it.** A pointing hand, a `›`, a "click to
  open" hint is a *claim*; compute it from whether the signal has a receiver
  (`self.receivers(self.some_signal) > 0`) so a surface cannot advertise what it has not connected.

A genuine *choice* (a tint, a prefix, one row's `advanced=True`) keeps its default and is listed in
the widget's test allowlist **with the reason it is a choice**. Guarded by
`tests/gui/test_spatial_map.py`, which discovers embedders rather than listing them.

**Ask what the widget does at the cohort size a real experiment has**, not the fixture's.
`OVERLAY_ABOVE = 6` pooled every recording into one square above six, and an ethogram strip at
`setFixedHeight(rows * 34)` was 3,400px at a hundred recordings. Both were correct at the size they
were written and tested at.

## 6. End with the launch command

**Measured: 6 of 159 GUI PRs did this.** It is a standing instruction and the least-followed step in
the repo.

Finish a GUI change with:

- the exact command to launch the app on this branch
- which mode and page to click through to
- what to look at, and what "right" looks like

A GUI change handed over without that is a change the person who asked for it cannot check.
