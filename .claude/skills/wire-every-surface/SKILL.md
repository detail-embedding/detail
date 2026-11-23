---
name: wire-every-surface
description: Enumerate every surface a capability is reached through before writing it — Python API, CLI, GUI, docs, run record, launch dialog — and make any skipped surface a written omission. Use when adding or changing a capability, a config dimension, a registry member, a module, or a shared widget's parameter.
---

# A capability reaches every surface, or says why not

**This is the most-repeated failure in the repo's history.** Nine merged PRs state the ratio in their
own words:

| what shipped | reached |
|---|---|
| the decode scrim's suppression predicate | **1 of 8** embedders — seven surfaces played video under a permanent "Decoding frame…" |
| the dialog scroll rule, via a helper each dialog had to call | **8 of 44** dialogs |
| the video-required guard on scrub surfaces | **2 of 12** — "ten of the twelve threw that away, three ways" |
| the row-identity fix under sortable headers | 1 of its 2 sites, **shipping with the other live**, twelve lines away |
| `mono_font` | **four reimplementations**, "all four were faithful, which is the trap" |
| the zoomable image view | "third instance of that shape in two days" |
| *Start from a previous run* | **2 of 9** analysis modules, and no processing module: almost never used |

Nothing fails in any of these. The surface being edited works; every other one renders a plausible
wrong picture, or silently uses a default, and the suite stays green — because a test written next to
a function calls that function.

## 1. Enumerate before you write

**A user-facing capability — a stage, or a config dimension on one (a constructor, a hyperparameter,
a renderer option) — must be exposed in every surface a user reaches it through.** List them first:

- **Python API** (`detail/api/`) — Tenet 2's boundary; the CLI and GUI are surfaces over it
- **CLI** (`detail/cli.py`) — which *claims* to mirror the API, so keep that true
- **GUI** — the module's `ModuleUI.config_widget()`, and any picker that enumerates the thing
- **The run record** — `record["params"]`, derived from the job, never hand-listed
- **Docs** — `docs/cli.md` / `docs/gui.md` / `docs/concepts.md`, and the generated module page
- **Every embedder**, for a shared widget: `grep -rn "WidgetName()" detail/gui/`
- **Every registry consumer**, for a registry member: the picker, the CLI id, the bundle manifest

Then ask `surfaces.is_surfaced(name)` before counting a surface as real. A page reachable only from a
hidden capability is unreachable, and a parity sweep that counts one is counting a surface no user
has.

## 2. Add them in the same change

Not a follow-up. A capability with no way in is the recurring shape: `ConfigWidget.load_config` was
made to work across every form and shipped with **nothing calling it** — the only entry point was a
button gated on `not mutable AND surfaced`.

## 3. A skipped surface is a written omission

If a surface is intentionally skipped — an interactive, iterate-until-it-looks-right tool that only
makes sense in the GUI — record it in `docs/cli.md`'s **"Not in the CLI"** ledger with the reason.
**The undocumented gap is the bug, not the omission itself.**

The model to copy is the PR that gave every surface one default tracker: a verdict column, one row per
swept site, each marked fixed or deliberately-not **with its reason**. That is also step 3.5 of
`fix-the-class`.

## 4. The parameter question, asked once

When adding a parameter to something shared, decide which of three it is:

- **A fact about the subject** that decides whether the output is *correct* → a **required keyword**,
  no default. A default spelling "as before" is where a missing fact hides.
- **A cue** — a pointing hand, a `›`, a "click to open" hint → **derived from the wiring**
  (`receivers(self, signal) > 0`, from `detail.gui.qt`), never set in parallel with the connection, or the claim and the
  connection live in different files and one is wrong.
- **A genuine choice** → keeps its default, and is **allowlisted with the reason it is a choice**.

## 5. A module's config is a pydantic model

A processing or analysis module's config is a `pydantic.BaseModel` in `detail/configs/<module>.py` —
the single source for each parameter's type, bounds, default, help and label. The GUI form
(`gui/_pydantic_form.py`) and CLI `--help` read that metadata. **Hand-writing a parameter's range,
help or default in a widget or a `@click.option` is the bug.** A new module must ship a config model.

And the surface that *sets* a config sets **all** of it, enumerated from `model_fields` rather than
hand-listed. A hand-written `config()` is complete on the day it is written; afterwards the run
silently uses the default for whatever was dropped, and the record it writes is well-formed.

`record["params"]` is derived from the job (`runners.job_params`), carrying config fields **by name** —
`temporal_clustering.build_job` is the pattern. A hand-listed dict is complete until a field is added and
nothing fails when it stops being: `cluster_eval`, a module since removed, dropped `stream_name`
from every run it wrote, including runs that passed one explicitly.

## 6. Guard it over the registry, not over the sites

A guard naming the sites somebody thought of covers the sites somebody thought of. Drive it from the
registry so a member added tomorrow is covered the day it is added:
`tests/configs/test_record_completeness.py`, `tests/api/test_job_reattach.py`,
`tests/gui/test_model_availability.py`, `tests/gui/test_start_from.py`,
`tests/gui/test_spatial_map.py` (which discovers embedders), `tests/gui/test_workspace_parity.py`
(data-driven over both workspaces — a guard naming one page cannot see this class at all).

Ask whether a host gets the behaviour **without opting in**. A guard over a mechanism that takes an
opt-in passes by testing the opt-in.

See `guard-that-can-fail` for making that guard bite, and `docs/recorded_classes.md` for the rows
this class has already earned.
