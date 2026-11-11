# DETAIL — Documentation Design

`docs/docs_design.md` is the authoritative specification for the **documentation system** —
how user-facing docs are structured, written, generated, surfaced across the app's surfaces,
and kept from drifting from the code. It stands alongside `DESIGN_PLAN.md` (schemas and
pipeline behavior) and `docs/gui_design.md` (GUI design) as an agent-facing source of truth:
dense and rationale-heavy, written for the Claude agents and human contributors who *develop*
the docs, not for end users. Read the section that governs a change before making it.

## Sections

1. Purpose, scope & relationship to other docs
2. Documentation tenets
3. Audiences & surfaces
4. Information architecture — the target site, Diátaxis within sections, per-type page templates
5. Authoring conventions & voice
6. Visual media discipline
7. Single source → every surface (API / CLI / GUI)
8. The docs site: tooling & build
9. Anti-drift, testing & maintenance
10. Current state & roadmap

---

## 1. Purpose, scope & relationship to other docs

**What this doc is.** The source of truth for *how DETAIL is documented*. Where
`DESIGN_PLAN.md` says what the system does and `gui_design.md` says how the GUI is built, this
doc says how all of that is explained to the people who install, use, script, and contribute
to the tool — the information architecture, the writing conventions, the code-to-docs
generation mechanisms, and the rules that keep docs true as the tool changes. It is itself
internal (agent-facing), keeps the dense spec voice, and is **not** published to the docs
site.

**In scope** — this doc governs:

- The **user-facing docs**: `README.md` and the docs **site** built from `docs/` (concepts,
  guides, reference, and the destinations §4 prescribes).
- The **code-integrated docs**: API docstrings, CLI `--help` / `describe`, and GUI help — all
  generated from one source per §7.
- The **public contributor docs** (the site's "Contributing" destination): how an outside
  contributor sets up a dev environment, runs the checks, and submits a change.
- The **tooling and conventions** for all of the above (§4–§9).

**Out of scope** — this doc does *not* govern the *content* of the internal specs themselves
(`DESIGN_PLAN.md`, `gui_design.md`, `backend_overview.md`). Those are
agent-facing, keep their own voice, and are not user docs. This doc may reference them, and §7
defines a contract they help satisfy, but their prose is theirs.

**Relationship to the other specs (and precedence).**

- **`DESIGN_PLAN.md`** — the authority on *what the system does* (schemas, dtypes, layouts,
  behavior). Docs *describe* what it specifies. On any conflict about system behavior,
  `DESIGN_PLAN` wins and the doc is corrected to match; this doc governs only *how* that
  behavior is documented.
- **`docs/gui_design.md`** — the authority on GUI *design* (screens, flows, widgets). The
  in-GUI help *widget* — where the description panel sits, how it is laid out — is specified
  there; the docs *contract* it renders (every config field carries `ParamDoc`; essential
  guidance is never tooltip-only — §7) is specified here. The two cross-reference; neither
  restates the other.
- **`docs/backend_overview.md`** — internal architecture narrative; a *source* the docs draw
  from, not a thing this doc governs.

**Relationship to `CLAUDE.md`.** `CLAUDE.md`'s "User-Facing Documentation" section today
carries the operational doc rules (the terse, command-first voice; surface parity; "a change
ships its docs in the same change"). Those rules live **here** in full — this doc is the single
home, so they cannot diverge between two files. `CLAUDE.md` keeps only a short pointer to this
doc plus the must-do operational checklist an agent needs every recording (update the right doc
when a change touches the CLI / GUI / API / install / pipeline; keep surface parity). When the
`CLAUDE.md` checklist and the detail here disagree, **this doc wins** and `CLAUDE.md` is trimmed
to match. (`CLAUDE.md` has been trimmed to that pointer + checklist — done.)

**The user-facing vs internal inventory.** The current split, for reference:

- *User-facing* (governed here): `README.md`, `docs/concepts.md`, `docs/cli.md`,
  `docs/gui.md`, and the forthcoming site pages.
- *Internal* (agent-facing, not user docs): `DESIGN_PLAN.md`, `docs/gui_design.md`,
  `docs/backend_overview.md`, and this file.

---

## 2. Documentation tenets

The non-negotiable invariants of the documentation system. A violation is a **bug**, not a
style preference — this is the line between §2 and §5: these tenets are invariants, the §5
authoring conventions are house style you bend case by case. Each tenet is stated once here
and elaborated by a later section.

1. **Reference is generated from code, never hand-maintained.** The API reference comes from
   docstrings, the CLI reference from the `click` app, the parameter reference from each
   module's `ParamDoc` (§7). If a fact can be generated from the source, it is — a hand-written
   reference table drifts the moment the code changes.

2. **One source per parameter.** A config field's label, help, units, valid range, and default
   are defined exactly once — in the module's config metadata — and rendered to every surface.
   No surface (GUI, CLI, reference page) restates them; each reads the one source (§7).

3. **Surface parity for docs.** A user-facing capability is documented on every surface a user
   reaches it through — Python API, CLI, and GUI — the docs mirror of the code surface-parity
   rule. An intentional omission from a surface is recorded with its reason, not left as a
   silent gap.

4. **A change ships its docs.** Any change that alters what a user sees or types — a CLI
   option, a GUI screen, an API signature, an install step, a parameter — updates the relevant
   doc or metadata in the *same* change. Fixing the doc is part of the change, not a follow-up.

5. **One Diátaxis type per page; no empty buckets.** Every page is exactly one of tutorial /
   how-to / reference / explanation (§4), never a blend — mixing the four is the root of most
   bad docs. And no structural placeholder is created with nothing in it; a section appears
   when it has real content.

6. **Docs build in CI.** `mkdocs build --strict` (a broken link or reference fails the build) runs
   in CI alongside `black --check` and `pytest`. Executable snippets and the per-parameter
   documentation test are planned additions to that gate (§9). A doc that doesn't build is a
   failing build.

7. **Ground examples in what exists today.** Every command, screen, and code sample is one that
   works in the current tool. Not-yet-built features are marked *planned*, never documented as
   working — a doc describing a capability the code lacks is a bug.

---

## 3. Audiences & surfaces

The docs serve three reader personas across five surfaces. Every piece of content is written
for a named persona reaching it on a named surface — content that serves no persona, or that a
persona can't reach, is the failure this section guards against.

**Personas.**

- **Operator** — installs DETAIL and runs the pipeline through the GUI or CLI; does not
  script against the API. Reaches for: installation, getting-started, the concepts / mental
  model, the GUI and CLI guides, troubleshooting.
- **Scripting user** — drives the Python API to batch, integrate, or extend a run
  programmatically. Reaches for: the API reference, runnable examples / recipes, concepts.
- **Contributor** — adds a module, tracker, model, or importer, or fixes a bug. Reaches for the
  public **Contributing** guide (dev environment, the format + test gate, the PR workflow),
  which then points *into* the internal specs for deep architecture. Boundary: the Contributing
  *guide* is public (a site destination, §4); the internal specs (`DESIGN_PLAN.md`,
  `gui_design.md`, `backend_overview.md`) stay internal and are linked, not published.

Claude agents are a fourth reader, but they are served by the internal specs + `CLAUDE.md`, not
the user site; making the *site* AI-consumable (an `llms.txt` / clean-markdown surface) is a
roadmap item (§10), not adopted now.

**Surfaces** — where doc content appears:

- **The docs site** — the primary home; serves all three personas through the §4 information
  architecture. Built from `docs/` (§8).
- **`README.md`** — the repo front door: install + a short quick-start that points into the
  site, not a substitute for it.
- **CLI** — `--help` on every command and a `describe <module>` command; the in-terminal
  reference, generated from the `click` app and `ParamDoc` (§7).
- **GUI** — a per-module **Help** link that opens the module's reference page (§7 contract); no
  inline help panel for now.
- **Docstrings** — the single API-reference source: they feed both the site's generated API
  reference (mkdocstrings) and IDE hovers, so one docstring serves two surfaces.

**Docs surface parity (Tenet 3, made concrete).** A capability reachable through the API, the
CLI, *and* the GUI is documented on each of the three. For parameters this is automatic: the
one `ParamDoc` source (§7) renders to the CLI `--help`, the GUI Help target (the reference
page), and the generated reference together — so parity is a property of the mechanism, not a
thing a writer maintains by hand. A capability intentionally absent from a surface is recorded
with its reason (e.g. `docs/cli.md`'s "Not in the CLI").

**Persona × surface** (primary use):

| Surface           | Operator | Scripting user | Contributor |
| ----------------- | :------: | :------------: | :---------: |
| Docs site         |    ●     |       ●        |      ●      |
| `README.md`       |    ●     |       ●        |      ●      |
| CLI `--help`      |    ●     |       ○        |      ○      |
| GUI Help link     |    ●     |                |             |
| Docstrings / IDE  |          |       ●        |      ●      |

(● primary · ○ occasional)

---

## 4. Information architecture

This section prescribes the **target site navigation** — the concrete top-level destinations
the docs build toward — and the discipline for organizing content within them. Diátaxis
(tutorial / how-to / reference / explanation) governs *within* a section; it is **not** the
top-level menu. Almost no well-run docs site exposes the four Diátaxis words as tabs — they use
a workflow spine with the four types applied inside. We do the same.

**Target navigation.** The full intended top level. Each destination is marked `now` (has
content today) or `planned` (named here, created when its trigger fires):

```
Home / Overview             landing page                                          now
Getting started             Installation, first steps (guided demo is planned)    now
Workflows       (tutorial)  step-by-step task guides: create a experiment, add recordings  now   [workflows/]
Concepts        (explain)   mental model, pipeline, vocabulary                    now   [concepts.md]
  └ Glossary    (reference) term → definition lookup                             now   [glossary.md]
Guides          (how-to)    drive each surface: GUI guide, CLI guide              now   [gui.md, cli.md]
Evaluation      (explain)   how retrieval is scored, one page per eval module     now   [behavior-query-eval.md]
Reference       (reference) Python API + CLI commands (parameters: planned)        now
Examples / Recipes          runnable, task-shaped snippets                        planned  ← first recipe
Troubleshooting / FAQ       problem → answer lookup                               planned  ← first recurring issue
Changelog / Release notes   what changed per version                             planned  ← first tagged release
Migration & version policy  schema_version / layout migrations + support policy   planned  ← first breaking schema change
Contributing                dev setup, format+test gate, PR workflow              planned  ← opening to outside contributors
About                       license, citation, acknowledgements                  planned
```

**No empty buckets (Tenet 5).** The target above names the whole nav so we build toward a known
shape; the live `mkdocs.yml` contains only `now` destinations. A `planned` destination is added
the moment its trigger fires (the arrow above) — never as an empty placeholder.

**Diátaxis mapping, and the one deliberate blend.** Concepts = explanation; Reference =
reference; Examples = how-to (runnable). Getting started is a guided first path
(tutorial-shaped) even before the full demo lands. **Guides blends tutorial and how-to on
purpose** — the near-universal real-site convention (FastAPI/Typer's "Learn", MkDocs'
"User Guide"): a surface guide both walks a new user through the GUI/CLI *and* serves as task
lookup. The blend is allowed at the *section* level; each individual *page* under it still obeys
Tenet 5 (one type per page). The **Glossary** is reference-type but sits under Concepts, where a
user meets the vocabulary while learning; it is user-facing (term → definition), distinct from
the internal writer word-list (§5).

**Reference structure.** One **Reference** section with sub-pages, not a separate site:

- **Python API** — generated by mkdocstrings from docstrings (§7).
- **CLI commands** — generated by mkdocs-click from `detail.cli:main` (§7).
- **Parameters** *(live)* — `reference/parameters.md`, generated from each module's pydantic config
  model (§7): the page the GUI Help link targets and the reference the CLI and GUI share. The IA
  guarantee — *a single generated parameter reference* — now holds; the GUI tooltips and CLI
  `--help` read the same field metadata.

**Evaluation structure.** One **Evaluation** section, **one page per eval module** (the
behavior-query eval today). Each page is generated from that module's metrics single-source (§7):
its `EvalMetrics`-style dataclass and a `describe_*()` feed the metric table, the config defaults
feed the matching rule, and the figure renders from a committed fixture through the real matcher and
scoring. A new eval module adds a sibling page here, generated the same way — never a rewrite of an
existing one. Keep each page scoped to its module (name it for the module, e.g. "Behavior query"),
so the section grows by addition.

**Per-type page templates ("anatomy of a page").** Every page declares its type and follows that
type's skeleton:

- **Tutorial** — one titled goal; prerequisites; a single ordered path where every step yields a
  visible result; ends at a working outcome; defers the *why* to linked Concepts. No branching,
  no options menus.
- **How-to** — a goal-shaped title ("Re-run tracking with a tighter threshold"); the minimal
  ordered steps to that goal; may branch on real forks; assumes competence; does not teach.
- **Reference** — austere and complete; led by the code, not the reader; generated where possible
  (Tenet 1); consistent structure across entries (signature / parameters / returns for the API;
  options for the CLI; label / help / units / range / default for parameters); no narrative.
- **Concept (explanation)** — open with a one or two sentence definition, then the minimal mental
  model and how the pieces relate. Prefer showing (a table, an example, generated output) over
  prose; put the deep *why*, trade-offs, and how-to-choose in linked how-tos or Reference, not here.
  A page that is a catalog of registered options is generated from its registry (§7), not
  hand-written. Readable away from the tool.

---

## 5. Authoring conventions & voice

House style — bendable case by case, unlike the §2 tenets. **Baseline:** for any mechanic not
specified here, follow the [Google developer documentation style guide](https://developers.google.com/style).
This section records only our voice, our deviations, and our word-list — not a full grammar
manual.

**Voice & tone.** Simple, concise, clear. Minimize word count; maximize clarity; no jargon —
prefer the ordinary word (see word-list). Address the reader as "you"; imperative for steps.
Register matches the page type: tutorials plain, how-tos imperative, reference austere, concepts
explanatory. **Personality or playful wording is added by human edit only — never an agent
default.**

**Terseness. Show, don't tell.** Open a page with a one or two sentence definition, then show the
content: a table, an example, or generated output. Add only the mental model a reader cannot get
from the thing itself. Cut any section whose job is to explain *why it matters* or *how to choose*;
if a reader needs that, link a how-to or the Reference. A section that exists only to explain is
what makes a page too wordy, so delete it. The Input images page is the model: two sentences, then a
generated table.

**Overview lead.** An overview or headline names what the user provides and what is supported (the
input, the options), not only what the tool does, in short declarative sentences. Example: "...based
on user-provided ground-truth labels. Point and state annotations are supported." Prefer a few short
sentences over one long clause-stacked sentence.

**Don't hand-write what code owns.** When a page documents a set the code defines (a registry of
options, a config's parameters), do not restate details that drift:

- **No count of a generated set.** Never write "the five methods" or "the other four"; the set
  changes and the prose goes stale. Write "the methods", "the others", or let the generated table
  carry the count.
- **No hard-coded defaults, bounds, or types.** A config model owns these (§7). Pull the default
  into the generated page from `describe_config`, or link the Parameters reference. A literal `0.5`
  typed in prose is a latent bug.
- **State the scope of a shared control.** When an option applies to only some entries on the page,
  say which ("the windowed matchers only", "`soft_knn` only"), not only what it does. A reader asks otherwise.

**Grammar & mechanics** (deviations/emphases over the baseline):

- Sentence-case headings.
- Present tense. Active voice and second person for steps and instructions; an overview or concept
  describes the system in the third person, where passive is fine when the actor is the system or
  unimportant ("Point and span annotations are supported").
- Procedures are numbered steps, one action per step.
- No em dashes (`—`). Use a comma, a colon, parentheses, or two sentences instead.
- Admonitions: `note` = aside, `tip` = optional shortcut, `warning` = data loss or a
  hard-to-reverse action. Don't stack them; don't use them for normal prose.

**Terminology / word-list.** The single source for *writing* choices — distinct from the
user-facing Glossary (§4), which defines domain terms for readers. Add a row when a term is
settled; `tests/test_docs_voice.py` parses this table and fails on a banned term in a
user-facing page, so a new row arms the check. §9 names the three rows it does not enforce
yet, and why.

| Use | Not | Why |
| --- | --- | --- |
| this is easy to get wrong | footgun | plain language |
| the cause | smoking gun | plain language |
| the problem / the issue | "smell" | plain language |
| video part | (calling it a recording) | a recording has many parts |
| recording · run · job | (used interchangeably) | distinct concepts — keep separate |
| product name via `site_name` (§8) | scattering the literal name in prose | provisional placeholder — a rename must be one change |
| substantial / detailed / significant | meaty | plain, precise wording |
| built-in / native / fully supported | first-class | jargon |
| comma, colon, parentheses, or two sentences | `—` (em dash) | reads as agent-written; restructure the sentence |
| evaluation | eval | spell out the abbreviation in prose |
| state annotation / point annotation | "span annotation" (a doc-only synonym) | use the code's established term (`state` / `point` mode), glossed in plain English; a parallel doc term diverges from the codebase and invites a rename |
| the embedding run / the model + input configuration | recipe | internal jargon; name the concrete thing |

**Links & cross-references.**

- **Name it, link it.** When you point the reader to a file — the `README`, a spec — link it, and
  make the **link text the file name in `code` formatting**: `` [`README`](url) ``, not
  `[the experiment README](url)`. Only the file name is the link; surrounding words stay prose. A
  named file with no link is an error. Files outside the site link via their **absolute GitHub
  URL** (the site can't resolve `../` paths). Navigation to a site *page* instead uses the page
  title as the link text (`[Concepts](concepts.md)`), not the file name. (Literal paths the reader
  types, like a config file, are not pointers and need no link.)
- Relative links within the site; no bare URLs in prose.
- Link *to* the reference or concept instead of restating it (one source, not two).
- Descriptive link text naming the destination; never "click here" or "this".
- The *why* goes to Concepts, the *how* to a how-to, the *facts* to Reference.

**Code samples & command formatting.**

- The command is the instruction: a short lead-in line, then its own fenced block. Don't narrate
  what an obvious command does.
- Give a check, not a lecture — for a conditional step, hand the reader a command that reveals
  whether they need it (`python -c "…"`), not an explanation of the mechanism.
- Cut the why — no rationale paragraphs or reassurance; a load-bearing *why* goes in a one-line
  `#` comment inside the block, or is dropped.
- One action per block.
- Shell commands on a **single line** — no backslash line-continuations (easy to copy).
- Real text, never a screenshot of text or output.
- Examples run, and are tested where feasible (§9).
- One capability across surfaces: content tabs (GUI / CLI / API), not three prose blocks.

**Accessibility & inclusive language.**

- Images: intent-focused alt text (≤ ~155 chars, no "image of"); decorative images `alt=""`
  (§6 covers media in full).
- Descriptive link text (above) is also an accessibility rule.
- Video: captions plus a transcript.
- Bias-free language per the baseline.

---

## 6. Visual media discipline

Text-first. An image appears only when prose can't carry the point; never an image of text, code,
or terminal output — use real text (images of text aren't searchable, accessible, or
translatable).

**GUI images are auto-rendered, not hand-captured.** Each documented screen is rendered to a PNG
by `QWidget.grab()` from the offscreen GUI test harness, with fixed sample-data fixtures. The
committed PNG is both the published image and the "golden" reference. Live or interactive
embedding of the Qt GUI in the browser is out of scope (native Qt; WASM / streamed recordings are
impractical with the scientific dependencies).

**Show where you are, then what opens.** A tutorial step that navigates the GUI carries **two**
auto-rendered images: first a **full-window** capture of the screen the step names, with the control
the user clicks marked by a drawn **highlight box**; then the dialog or panel that opens, cropped
tight. Never name a screen or a button in prose without showing it — a step that says "click **Add
recordings…**" shows the screen with that button boxed. The highlight is drawn by the harness from the
widget's geometry (`_save_window` boxes a button by its label, `_save_tree_window` a sidebar tree
node), never hand-painted onto the image, in a high-contrast accent (vermillion) kept distinct from
the in-UI status colours (§ GUI style). Both images come from `scripts/gen_docs_screens.py`: the
full annotated capture plus `_save` for the tight dialog. Where the click opens a native OS dialog (a
file picker) there is nothing to capture, so the full-window shot stands alone.

**Fixtures, not the user's machine.** Every screen renders from a throwaway experiment built in a temp
dir from the committed sample-data fixtures, never the reader's real workspace — point
`locations.experiments_root` at the temp dir so the sidebar lists only that experiment. A committed
screenshot must never bake in a local path or username: set display-only fields (a file picker's
path, a queued video) to a neutral value (e.g. `/data/…`). Check this on any new screen.

**Change trigger → agent review-list.** CI re-renders each screen and compares it to the golden
(tolerant diff; goldens generated in CI / Linux for determinism — §9). On any change it flags the
screen and emits a review-list keyed to the docs page(s) that embed it, so an agent (or human)
checks whether the surrounding prose is now stale. The test mechanism is specified in §9.
**Status:** the first documented screens (the Workflows tutorials, §4) landed the **renderer** —
`scripts/gen_docs_screens.py` grabs each screen headless from sample-data fixtures and writes a
**committed** PNG the docs embed — but chose the simpler committed-PNG path (Option 1) over the
golden-compare trigger, which is deferred (§9, §10). A GUI change is caught by rerunning the
generator, not yet by a failing test. For an undocumented screen, still prefer text.

**Stills.** Crop tight; strip irrelevant UI (a simplified shot changes less often). Annotate with
numbered callouts keyed to the body text — never sentences baked into the image; annotation marks
meet 3:1 contrast.

**Motion.** A GIF only for a brief silent loop; a captioned video with a transcript for a narrated
procedure. Auto-playing motion over five seconds needs a pause control.

**Alt text & location.** Intent-focused alt text per §5; decorative images `alt=""`. Media lives
under `docs/assets/`, named for what it shows.

---

## 7. Single source → every surface

A parameter's documentation is written once and reaches every surface (Tenets 1–3).

**`ParamDoc` — the per-parameter contract.** Every config option carries a `ParamDoc`: a declared
label, help, units, range, and default. **The mechanism is a pydantic v2 config model per module**
in `detail/configs/` (see the `CLAUDE.md` Hard Rule and `DESIGN_PLAN.md` §6.5): one `Field()`
carries the type, bounds, default, and — via `json_schema_extra` — label/unit, feeding validation,
the Qt form, the `click` `--help`, and this reference from a single declaration. `detection` is
the live reference; the rollout across the remaining modules is tracked as planned work. The
`ParamDoc` shape every surface reads is:

- `label` — short display name.
- `help` — one or two plain sentences (§5 voice).
- `unit` — e.g. seconds, frames, or none.
- `choices` / `range` — allowed values, or `(min, max)`.
- `default` — the default value.

Plus a module-level `description` string.

**`describe_config(module)`** reads a module's config-model fields (the same `model_fields` /
JSON-schema that `gui/_pydantic_form.py::field_meta` reads) and returns its `ParamDoc`s.
Every surface consumes this one function:

- **CLI** — `--help` option text reads the model (`Model.model_fields[…].description`); help is
  never hand-typed into `@click.option`. A `describe <module>` command is a possible future add.
- **Reference** — the generated Parameters page (§4): a table per module (label / help / unit /
  range / default), the reference the CLI and GUI share.
- **GUI** — tooltips + widget ranges read the model's field metadata (`field_meta`) for migrated
  modules; a per-module Help link to the Parameters page is planned.
- **API** — method / class docstrings stay the source for API docs (mkdocstrings); `ParamDoc` is
  the source for *parameter* docs. The reference page combines both.

**Registered catalogs generate their own page.** The same single-source discipline applies to any
registry of options a user picks from, not only per-parameter docs. When a page lists registered
things (input-image constructors, embedding models, querying methods, transforms, eval metrics),
generate it from the code: a `describe_*()` in the package returns each entry's metadata, and a
`mkdocs-gen-files` script renders the page. What keeps it from drifting:

- **All the page's per-entry text lives in `describe_*()`**, both the one-line summary and any
  longer "how it works" description. Hand-write only the page headline and short connective
  sentences; a per-entry paragraph typed as markdown is the thing that goes stale.
- **Values a config owns come from `describe_config`**, never retyped. A default, unit, or range
  shown next to an option is read from the field metadata (above), so changing the config changes
  the page.
- **Example images and figures render at build from a committed fixture through the real code
  path** (the real renderer, the real matcher), so they move with the code. A few honest
  imperfections in real output are fine, and preferred over a hand-idealized picture.
- **A drift test pins the page's entry set to the implementation** (e.g. the config `Literal` == the
  implemented registry == the described set), and `describe_*()` raises on an entry it cannot
  describe, so a new option cannot ship undocumented.

Live templates: `detail/transforms/constructor_docs.py` + `scripts/gen_docs_input_images.py`
(Input images) and `detail/analysis/query_methods_docs.py` +
`scripts/gen_docs_querying_methods.py` (Querying methods, including a build-time figure and
config-sourced defaults). Extend the pattern to eval metrics.

**Contract (surface reach).** Each surface exposes a discoverable path to the one parameter
reference: the GUI via the Help link, the CLI via `--help` / `describe`, the API via the docstring.
No essential parameter guidance is hover-only. Inline GUI help (an on-focus line or panel) is a
deferred enhancement (§10); its widget design, when built, is `gui_design.md`'s.

**Drift guard.** A data-driven test over `PROCESSING_MODULES ∪ ANALYSIS_MODULES` asserts every
config field carries a non-empty `label` and `help`, and a `range` for numeric fields. A new
module with an undocumented parameter fails CI (§9).

**Status.** Design only — the *mechanism* decision and the implementation are
pending. This section is the contract the chosen mechanism must meet.

---

## 8. The docs site: tooling & build

**Stack.** MkDocs + Material for MkDocs. Plugins: `mkdocstrings` (Python handler, Google docstring
style — its default; API reference from docstrings) and built-in `search`. (`mike` for versioned
deploys is planned — §10 — and lives in the `[docs]` extra but is not yet wired into `mkdocs.yml`.)

Markdown extensions: `admonition` + `pymdownx.details` (§5 admonitions), `pymdownx.superfences` +
`pymdownx.tabbed` (the GUI/CLI/API content tabs, §5), `toc` with `permalink` and a GitHub-compatible
`slugify`, and `mkdocs-click` (CLI reference from `detail.cli:main`).

**Files.**

- `mkdocs.yml` at the repo root — theme, `nav` (§4), plugins, extensions, `exclude_docs` (keeps the
  internal specs unpublished — §1), and strict link/anchor `validation` (§9).
- A `docs` optional-dependency group in `pyproject.toml` (`mkdocs`, `mkdocs-material`,
  `mkdocstrings[python]`, `mkdocs-click`, `mike`); install with `pip install -e .[docs]`.
- `site/` is build output — gitignored.

**Commands.**

- `mkdocs serve` — live local preview at `localhost:8000`; the current preview path.
- `mkdocs build --strict` — a broken link or reference fails the build; run in CI (§9, Tenet 6).

**Versioning.** `mike` deploys each version into `gh-pages` with `latest` / `stable` aliases; the
support policy (which versions we keep) is §4's Migration & version policy, once a release exists.

**Product name.** Single-sourced as `site_name` in `mkdocs.yml`; prose minimizes the literal name
(§5 word-list) so a rename is one change, not a find-replace.

**Theme risk.** Material entered maintenance mode (Nov 2025 — security fixes, no new features); the
successor **Zensical** (same team, MIT) is a near-drop-in but pre-1.0 with only preliminary
mkdocstrings support. Adopt Material now; revisit Zensical when it stabilizes.

**Hosting — deferred (§10).** The repo is private on the free plan, which blocks free GitHub Pages
and Read the Docs Community. Preview is **local `mkdocs serve`** for now; no host is provisioned.
When we publish, the recorded options are: a public output repo (CI pushes `site/` to a public
Pages repo — free), GitHub Pro (Pages from the private repo, ~$4/mo), making the repo public, or a
gated preview via Cloudflare Pages + Access (free, email allow-list). The site builds locally and
in CI without any of these.

---

## 9. Anti-drift, testing & maintenance

Docs stay true two ways: automated checks that fail the build, and a light human cadence.

**Automated gate** (in CI beside `black --check` and `pytest`). One check is live today; the rest
are planned as the mechanisms they depend on land:

- **`mkdocs build --strict`** *(live)* — a broken internal link or reference fails the build
  (Tenet 6).
- **Nav / `exclude_docs` parity** *(live)* — `tests/test_docs_nav.py` asserts every markdown file
  on disk under `docs/` is either in `mkdocs.yml`'s `nav` or matched by `exclude_docs` (§1). A
  file in neither state is **published anyway** — mkdocs ships whatever it is not told to exclude,
  and reports the nav omission only at INFO — so an internal, agent-facing note lands on the user
  site silently, and the strict build goes red much later, on the first link it grows to a path
  outside `docs/`. There is no third state: user page in the nav, or internal doc in
  `exclude_docs`.
- **Generated reference** — the API (mkdocstrings) and CLI (mkdocs-click) reference generate today
  (part of the strict build); the **parameter reference** (`reference/parameters.md`, §7) generates
  from the config models via `describe_config` (`scripts/gen_docs_parameters.py`, the `gen-files`
  plugin). Generated docs can't drift from the code (Tenet 1).
- **Per-parameter doc test** *(live — §7)* — `tests/configs/test_describe.py`, data-driven over
  `PROCESSING_MODULES ∪ ANALYSIS_MODULES`, asserts every config field carries `label` + `help`; a
  new module with an undocumented parameter fails.
- **Executed snippets** *(planned)* — `>>>` examples in docstrings and Markdown run under `pytest`
  (`--doctest-modules` / `--doctest-glob`, or Sybil for fenced blocks); a stale example fails
  (Tenet 7).
- **Voice check** *(live)* — `tests/test_docs_voice.py`. The §5 word-list's banned column is
  parsed out of the table itself and grepped in the user-facing docs (the nav, plus the
  `*-intro.md` files whose prose is inlined into generated pages); a banned term fails. Three
  rows are not enforced yet and say why in `_NOT_ENFORCED`: `eval`, which is the module's own
  name on every surface; the literal product name, which waits on the rename the row asks for;
  and the em dash, which has a ratchet instead. **Em dashes only decrease**: `_EM_DASH_CEILING`
  is the whole inventory, a page absent from it is checked at zero, and an entry left above the
  real count fails as an allowlist that excuses nothing.
- **GUI render trigger** *(renderer live; golden-compare planned — §6)*: the renderer is
  `scripts/gen_docs_screens.py` (committed PNGs from sample-data fixtures, landed with the Workflows
  tutorials); the registry + tolerant golden-compare + review-list below are the remaining planned half.
    - A **registry** maps each documented screen → its render recipe (built with fixed sample-data
      fixtures) and the docs page(s) that embed it.
    - CI re-renders each screen via `QWidget.grab()` and compares to its committed golden with a
      **tolerant** pixel diff (`pytest-regressions`' `image_regression`, or a small Pillow diff).
      Goldens are generated in **CI / Linux** so they are deterministic and match the published
      images.
    - On any change it flags the screen and emits a **review-list** (`{screen_id, docs_pages,
      old.png, new.png}`) — the worklist for an agent (or human) to check whether the surrounding
      prose is now stale, and to regenerate the golden.

**Human maintenance.**

- Docs are reviewed in the PR like code; **a change ships its docs** (Tenet 4) is checked at
  review — a change touching the CLI / GUI / API / install / pipeline that leaves the docs
  untouched is sent back.
- Settled doc conventions and reversals are recorded in the **Decision log** (end of this doc) so
  they are not re-litigated.

---

## 10. Current state & roadmap

**Current state.** The **site scaffold is built** (roadmap 1): `mkdocs.yml` (Material, the `now`
nav, mkdocstrings + mkdocs-click), the Home / Getting-started / Concepts + Glossary pages, the
generated API + CLI reference, `mkdocs build --strict` in CI, and `mkdocs serve` preview. The
`CLAUDE.md` rule migration is done (roadmap 2). No host yet — preview is local. The next
substantive piece is single-source parameter docs (roadmap 3; mechanism adopted — pydantic config
models, §7; the `describe_config` generator + Parameters page are built, with 5 of 7 modules
migrated).

**Roadmap** — roughly in order:

1. **Site scaffold** *(done)* — MkDocs + Material over `docs/`, the `now` nav (§4), mkdocstrings API
   + mkdocs-click CLI reference, the seeded Glossary, `mkdocs build --strict` in CI, `mkdocs serve`
   preview.
2. **Migrate the doc rules out of `CLAUDE.md`** *(done)* — `CLAUDE.md` is now a pointer + the
   operational checklist; this doc is the full home (§1).
3. **Single-source parameter docs** *(in progress)* — pydantic config models in
   `detail/configs/` (§7): each `Field()`'s metadata + `describe_config` + per-surface renderers
   + the drift test. Built: the generator + the generated `reference/parameters.md` + the doc test,
   with every registered module migrated. Remaining: the GUI Help link. Fixes the opaque-parameter pain.
4. **Getting-started tutorial + sample dataset** — the guided first path (§4).
5. **GUI image pipeline** *(renderer done; trigger planned)* — `scripts/gen_docs_screens.py` grabs
   each screen to a committed PNG (landed with the Workflows tutorials, the first documented screens,
   as Option 1); the tolerant golden-compare trigger + review-list remain (§6, §9).
6. **Planned destinations** — Examples, Troubleshooting/FAQ, Changelog, Migration & version policy,
   Contributing, About — each created when its trigger fires (§4).
7. **Hosting** — decided when publishing (§8); **`mike` versioning** with the first release.
8. **Executed-snippet checks** (§9). The word check landed as `tests/test_docs_voice.py`; what
   is left is running fenced examples, and the three word-list rows that check does not enforce.
9. **AI-consumability** — an `llms.txt` / clean-markdown surface (§3), evaluated later.
10. **Inline GUI help** (on-focus line or panel) beyond the Help link, if the opaque-parameter pain
    persists (§7).

---

## Decision log

Settled documentation decisions (most recent first). Reverse or amend here; don't re-litigate in
prose.

| Decision | Rationale |
| --- | --- |
| Workflows section: step-by-step task tutorials (create a experiment, add recordings) with auto-rendered GUI screenshots + a committed sample dataset | operators need a guided path for common tasks; tutorial-type, distinct from the how-to Guides (§4) |
| GUI screenshots are auto-rendered committed PNGs (Option 1: a `gen_docs_screens.py` generator you rerun) — not hand screenshots, and not the golden-compare pixel-diff (deferred) | a generator keeps images honest without hand-capture and keeps the docs build Qt-free; the golden-compare's cross-platform (Mac↔Linux) determinism cost isn't worth it yet — cheap to add later (§6, §9) |
| Preview via local `mkdocs serve` only; production hosting deferred | repo is private on the free plan; no host is needed to build or preview (§8) |
| Glossary (user-facing, term → definition) seeded now under Concepts | direct fix for opaque vocabulary; distinct from the internal writer word-list (§4) |
| Product name is a placeholder — single-source via `site_name`, minimize literal use | the name will change; a rename must be one edit (§5, §8) |
| Mechanics baseline = the Google developer documentation style guide | avoid authoring a full grammar manual; record only deviations (§5) |
| No playful/personality wording as an agent default — human edit only; language simple, minimal, jargon-free | keep agent-written docs plain and clear (§5) |
| GUI parameter help = a per-module link to the reference page; no inline panel for now | keep the form clean; the reference is the one home; inline help deferred (§7) |
| In-GUI help is a §7 surface contract, not a standalone tenet | it is now "link to the reference"; a surface detail, not an invariant (§2, §7) |
| Site generator = MkDocs + Material for MkDocs | Markdown-first, matches the existing docs + Google-docstring autodoc; closest peer (SLEAP) uses it (§8) |
| IA = a workflow-spine nav with Diátaxis applied *within* sections, not as top-level tabs | matches every well-run peer site; literal Diátaxis tabs are rare (§4) |
| Contributor docs are in scope for the public site | contributor is a first-class audience; the Contributing guide points into the internal specs (§1, §3) |
| This doc is the full home for the doc rules; `CLAUDE.md` keeps a pointer + checklist | one home so the rules can't diverge across two files (§1) |
| Single-source parameter-docs mechanism is **pydantic v2 config models** (`detail/configs/`) | the 2026 SOTA survey favored pydantic (napari's pattern) over a plain dataclass or a `params()` list; `downsampling` is the live reference, rollout tracked as planned work (§7) |
