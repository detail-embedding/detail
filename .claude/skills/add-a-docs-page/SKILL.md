---
name: add-a-docs-page
description: Add or regenerate a documentation page so the build, the nav, the guards and the in-app help all agree. Use when adding a docs page, changing what a page describes, or when a CLI command, GUI screen or public API a doc covers has changed.
---

# Adding a documentation page

The full specification is `docs/docs_design.md` — §4 information architecture, §5 authoring
conventions and voice, §7 single source → every surface, §8 the build, §9 anti-drift. **Read §5
before writing user-facing prose.** This is the sequence, which exists nowhere else as one list.

## The decision that comes first

**Every file under `docs/` is either a user page in `mkdocs.yml`'s `nav`, or an internal
agent-facing doc in `exclude_docs`. There is no third state.** mkdocs publishes whatever it is not
told to exclude, so an internal note ships to the user site and its first cross-tree link fails
`mkdocs build --strict`. `tests/test_docs_nav.py` fails naming the file, and its message points at
`docs/docs_design.md` §1.

## 1. Write it, or generate it

- **Hand-written user page** → `docs/<name>.md`, in the voice of §5: terse, command-first, plain.
  The command is the instruction; give a check, not a lecture; cut the why; no added personality.
- **Generated page** → a `scripts/gen_docs_*.py` listed under the `gen-files` plugin in
  `mkdocs.yml`. Generated pages are **not on disk**, so `test_docs_nav.py` does not see them and
  they need no exclusion. `tests/test_docs_generated_pages.py` is their guard.
- **Internal agent-facing doc** → declare it on line 3 the way `docs/how_we_work.md` does:
  *"**Agent-facing. Not a user page** — it is in `mkdocs.yml`'s `exclude_docs`."*

## 2. Declare it in `mkdocs.yml`

Either a `nav` entry (user page) or a **bare filename** line in the `exclude_docs` block — the
entries are matched against paths relative to `docs_dir` *or* against the basename, which is why
bare filenames work. A module page also needs its row in the `Modules:` nav group, guarded by
`tests/test_docs_modules_nav.py`.

## 3. Mark prose that teaches a hidden capability

If the page describes something `detail/surfaces.py` §25 hides, wrap that prose in
`<!--surface:NAME-->` … `<!--/surface-->`. Marking is for prose that teaches a hidden capability as
though it were available — **not** for prose documenting the absence, like `docs/cli.md`'s
"Not in the CLI" ledger, which must survive the strip.

Three readers must agree about a marker: the site build (`scripts/docs_surface_hook.py`), the guard
(`tests/test_surfaces.py`), and the in-app help panel (`detail/help.py`). The marker grammar and the
strip live in `detail.surfaces`, beside the table that decides what is hidden.

`tests/test_surfaces.py` enumerates **every** `docs/**/*.md`, excluded ones included, so a marker
must name a real `HIDDEN` key and be balanced — an unclosed marker is an error, not a silent
truncation.

## 4. Register a `?` topic if a surface should link to it

`detail/help.py` maps a help topic to a page plus an anchor, and the in-app panel renders that
page's markdown. `tests/test_docs_map.py` asserts every mapped anchor is a real id **in the built
site**, and fails on a new field, screen or module with no doc, or a `?` pointing nowhere.

Note this cannot point at an excluded page: an excluded page produces no HTML, so the anchor check
fails with "page missing".

## 5. Build it the way CI does

```
python -m mkdocs build --strict
```

`.github/workflows/docs.yml` runs exactly that on every push and every PR, and it is a separate
workflow from the test suite — easy to forget, and able to fail a PR on its own. `--strict` promotes
mkdocs' broken-link and anchor warnings to errors; `mkdocs.yml` sets `validation.links.anchors: warn`
so that strictness is where they bite.

**Run it from the worktree** (`python -m mkdocs`, not the console script, which resolves through the
editable install to the primary checkout).

**No test verifies cross-document links inside `docs/`**, and the strict build only covers
*published* pages — so an internal doc's dangling link is caught by nothing. Check those by hand.

## 6. Keep it true afterwards

A change that alters what a user sees or types updates its doc **in the same change** — see
`ship-a-change` §5 for the change → doc table. A capability that exists in one surface and not
another is `wire-every-surface`, and a deliberate omission goes in `docs/cli.md`'s "Not in the CLI"
with its reason.

§9 is the anti-drift inventory: `test_docs_nav.py`, `test_docs_modules_nav.py`,
`test_docs_generated_pages.py`, `test_docs_map.py`, `test_help.py`. If you add a generator or a
surface, add its guard there too.
