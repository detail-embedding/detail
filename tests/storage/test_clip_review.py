"""``clip_review.csv`` — the clip review's append-only log (§19/§22)."""

from detail.storage import clip_review as cr
from detail.storage.clip_edits import ClipRef

CLIP = ClipRef("recording_001", 0, 10, 40)
OTHER = ClipRef("recording_001", 0, 50, 90)


def test_a_census_is_one_write_of_every_answer_read_back_in_order(tmp_path):
    written = cr.append_answers(
        tmp_path, [(CLIP, True), (OTHER, False)], node=7, label_path="climb"
    )
    header = cr.review_path(tmp_path).read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.split(",") == cr.REVIEW_COLUMNS
    rows = cr.read_review(tmp_path)
    assert rows == written
    assert [(r.clip, r.node, r.label_path, r.answer) for r in rows] == [
        (CLIP, 7, "climb", "yes"),
        (OTHER, 7, "climb", "no"),
    ]
    assert len({r.drawn_at for r in rows}) == 1, "one census, one time"


def test_an_empty_census_writes_nothing(tmp_path):
    assert cr.append_answers(tmp_path, [], node=7, label_path="climb") == []
    assert not cr.review_path(tmp_path).is_file()


def test_a_later_answer_supersedes_an_earlier_one(tmp_path):
    cr.append_answers(tmp_path, [(CLIP, False)], node=7, label_path="climb")
    cr.append_answers(tmp_path, [(CLIP, True)], node=7, label_path="climb")
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "climb").yes is True


def test_an_answer_is_kept_per_label_path(tmp_path):
    """Judged as climb and later as rear: each label keeps its own answer, and a label the clip
    was never judged as has none."""
    cr.append_answers(tmp_path, [(CLIP, True)], node=7, label_path="climb")
    cr.append_answers(tmp_path, [(CLIP, False)], node=7, label_path="rear")
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "climb").yes is True
    assert cr.answer_for(latest, CLIP.key, "rear").yes is False
    assert cr.answer_for(latest, CLIP.key, "groom") is None


def test_a_row_with_no_label_path_answers_for_any_label_until_a_later_one_does(
    tmp_path,
):
    cr.append_answers(tmp_path, [(CLIP, False)], node=7, label_path="")
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "climb").yes is False
    cr.append_answers(tmp_path, [(CLIP, True)], node=7, label_path="climb")
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "climb").yes is True


def test_an_earlier_draws_unanswered_row_is_read_and_judges_nothing(tmp_path):
    """The 25-clip draw logged a drawn clip with an empty answer; such files stay readable."""
    cr.review_path(tmp_path).write_text(
        ",".join(cr.REVIEW_COLUMNS)
        + "\nrecording_001,0,10,40,7,climb,2026-11-01T00:00:00Z,\n",
        encoding="utf-8",
    )
    (row,) = cr.read_review(tmp_path)
    assert row.yes is None and not row.judged
    assert cr.latest_by_path([row]) == {}


def test_a_torn_final_row_is_dropped_and_the_rest_kept(tmp_path):
    cr.append_answers(tmp_path, [(CLIP, True)], node=7, label_path="climb")
    with open(cr.review_path(tmp_path), "a", encoding="utf-8") as f:
        f.write("recording_001,0,5")
    assert [r.answer for r in cr.read_review(tmp_path)] == ["yes"]


def test_a_query_census_writes_no_cluster_and_reads_back_as_none(tmp_path):
    """A query run's review has no clusters (§20): ``node=None`` is an empty cell, read as
    ``NO_NODE``, and a withdrawal of such a row keeps it empty."""
    cr.append_answers(tmp_path, [(CLIP, False)], node=None, label_path="q01")
    text = cr.review_path(tmp_path).read_text(encoding="utf-8-sig").splitlines()
    assert text[1].split(",")[4] == "", "no cluster is an empty cell, not -1"
    (row,) = cr.read_review(tmp_path)
    assert row.node == cr.NO_NODE
    cr.withdraw(tmp_path, [CLIP.key])
    assert (
        cr.review_path(tmp_path)
        .read_text(encoding="utf-8-sig")
        .splitlines()[2]
        .split(",")[4]
        == ""
    )


def test_withdrawing_under_one_label_path_leaves_the_others(tmp_path):
    """One span can be a candidate of two query behaviors: putting it back in one withdraws
    that behavior's answer only. Without ``label_path`` every path's answer goes, as a
    clustering put-back needs."""
    cr.append_answers(tmp_path, [(CLIP, False)], node=None, label_path="q01")
    cr.append_answers(tmp_path, [(CLIP, False)], node=None, label_path="q02")
    cr.withdraw(tmp_path, [CLIP.key], label_path="q01")
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "q01") is None
    assert cr.answer_for(latest, CLIP.key, "q02").yes is False

    cr.withdraw(tmp_path, [CLIP.key])
    latest = cr.latest_by_path(cr.read_review(tmp_path))
    assert cr.answer_for(latest, CLIP.key, "q02") is None
