"""One set of groups, one order — whoever asks and whatever they hold it as.

**The class.** Two orderings of the same set, derived by different keys, used as if they were
interchangeable. A comparison's groups are tuples of column values (`("veh", "am")`), and they get
sorted twice on their way to the screen: `api/comparison.py` joins each tuple with `|` and sorts
the *strings* into `record["groups"]`, while every reader that starts from the observations sorts
the *tuples*. `sorted` on either is stable and obviously correct on its own, and the two agree on
almost every input — which is what let it ship.

They disagree exactly when one group's first component is a prefix of another's, because `|`
(0x7C) sorts after every letter while the end of a string sorts before them:

    tuples : ("a", "z")  <  ("ab", "c")      -- "a" < "ab"
    strings: "ab|c"      <  "a|z"            -- "b" < "|"

The Overview table took its column *headers* from the joined-and-sorted list and its *cells* from
the tuple-sorted contrast, so on such a cohort every mean, and the direction of every change, was
printed under the other group's name. Nothing failed: both columns hold real numbers, the change
column holds a real percentage, and the only thing wrong is which group each belongs to. It is
reachable from any two-column `Compare across` — `drug` × `site` with a `veh`/`vehicle` pair is
enough.

**The rule.** Groups are ordered as *tuples*, once, and every surface takes that order — the
record's list included. Joining is a rendering step that happens after the ordering is fixed,
never before it.
"""

import pytest

from detail.analysis import compare, comparison_report

# `drug` and `site` chosen so the joined strings and the tuples sort the other way round; two
# recordings per group so a contrast has something to average.
PREFIX_COLLISION = {
    "r1": ({"animal": "A", "drug": "a", "site": "z"}, 2),
    "r2": ({"animal": "B", "drug": "a", "site": "z"}, 4),
    "r3": ({"animal": "C", "drug": "ab", "site": "c"}, 20),
    "r4": ({"animal": "D", "drug": "ab", "site": "c"}, 30),
}

CONFIG = {
    "source_run_id": "an_01",
    "group_by": ["drug", "site"],
    "aggregate_to": [],
    "measures": ["n_instances"],
}


def test_the_two_sort_keys_really_do_disagree():
    """Guard the guard. Everything below is vacuous if this fixture stops being a collision —
    a `sorted` that agrees with itself proves nothing."""
    tuples = [("a", "z"), ("ab", "c")]

    assert sorted(tuples) == tuples
    assert sorted("|".join(t) for t in tuples) == ["ab|c", "a|z"]


def test_the_record_lists_groups_in_the_order_the_observations_sort(tmp_path):
    """The writer's half. `record["groups"]` is the list three surfaces read; it has to be the
    tuple order, or every one of them is off by a permutation."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, PREFIX_COLLISION)
    exp.run_comparison(CONFIG)
    comparison = exp.comparisons[-1]

    obs = _read_observations(comparison)
    expected = ["|".join(g) for g in sorted({o.group for o in obs})]

    assert comparison.record()["groups"] == expected


def test_the_summary_columns_are_the_groups_they_are_computed_from(tmp_path):
    """The exported half, read off the file rather than off the call: `summary.md`'s header row
    and its numbers have to name the same groups."""
    from tests.api.test_comparison import _cohort

    exp = _cohort(tmp_path, PREFIX_COLLISION)
    exp.run_comparison(CONFIG)
    comparison = exp.comparisons[-1]
    out = tmp_path / "export"
    comparison.export(out, formats=("png",))

    header = _measure_row(out / "summary.md", "| measure |")
    values = _measure_row(out / "summary.md", "| Bouts")
    obs = _read_observations(comparison)
    ranking = compare.rank_by_difference(obs, tuple(sorted({o.group for o in obs})))
    first, second = ranking.contrasts[0].a, ranking.contrasts[0].b

    assert header[1] == "|".join(first.group) and header[2] == "|".join(second.group)
    # The means land under their own names: A's mean is 3, B's is 25.
    assert float(values[1]) == pytest.approx(first.mean)
    assert float(values[2]) == pytest.approx(second.mean)


def _read_observations(comparison):
    import csv

    stats = comparison.dir / comparison_report.STATS_FILENAME
    with open(stats, encoding="utf-8-sig") as handle:
        return comparison_report._observations(list(csv.DictReader(handle)))


def _measure_row(path, prefix: str) -> list[str]:
    """Split a Markdown row on its *unescaped* delimiters.

    A group label is its columns joined with `|`, which is also Markdown's separator — so this
    reads correctly only if `summary.md` escapes it, which is the second thing under test here.
    """
    import re

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(prefix):
            body = line.strip().strip("|")
            return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", body)]
    raise AssertionError(f"no row starting {prefix!r} in {path}")
