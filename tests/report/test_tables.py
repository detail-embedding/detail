"""Table writers: tidy CSV + the pivoted results grid (LaTeX booktabs / Markdown / CSV)."""

import csv

from detail.report import tables


def _rows():
    # (config x metric) tidy rows; context_crop wins AP, silhouette wins precision.
    return [
        {"config": "silhouette", "metric": "AP", "value": 0.42, "std": 0.03},
        {"config": "silhouette", "metric": "Precision", "value": 0.60, "std": 0.02},
        {"config": "context_crop", "metric": "AP", "value": 0.55, "std": 0.04},
        {"config": "context_crop", "metric": "Precision", "value": 0.41, "std": 0.05},
    ]


def test_write_tidy_csv_roundtrips(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "stats.csv"
    tables.write_tidy_csv(path, rows, ["a", "b"])
    with open(path, encoding="utf-8-sig") as handle:
        back = list(csv.DictReader(handle))
    assert [r["a"] for r in back] == ["1", "2"]
    assert [r["b"] for r in back] == ["x", "y"]
    assert not list(tmp_path.glob("*.tmp"))  # atomic: temp gone


def test_write_tidy_csv_raises_on_missing_column(tmp_path):
    import pytest

    with pytest.raises(KeyError):
        tables.write_tidy_csv(tmp_path / "s.csv", [{"a": 1}], ["a", "b"])


def test_write_tidy_csv_refuses_a_field_its_columns_do_not_name(tmp_path):
    """The column list lives apart from the code that builds the rows. ``ranking.csv``'s
    rows gained ``pair_a``/``pair_b`` (the levels a profile's effect is between) and its
    column list did not, so the file shipped without them and nothing said so."""
    import pytest

    path = tmp_path / "ranking.csv"
    with pytest.raises(ValueError, match=r"\['pair_a'\] are not among its columns"):
        tables.write_tidy_csv(path, [{"a": 1, "pair_a": "veh"}], ["a"])
    assert not path.exists() and not list(tmp_path.glob("*.tmp"))


def test_pivot_marks_best_per_column():
    grid = tables.pivot_to_grid(
        _rows(),
        row_key="config",
        col_key="metric",
        value_key="value",
        std_key="std",
        row_order=["silhouette", "context_crop"],
        col_order=["AP", "Precision"],
        higher_is_better=True,
    )
    assert grid.row_labels == ["silhouette", "context_crop"]
    assert grid.columns == ["AP", "Precision"]
    best = {
        (grid.row_labels[r], grid.columns[c])
        for r, row in enumerate(grid.cells)
        for c, cell in enumerate(row)
        if cell.is_best
    }
    assert best == {("context_crop", "AP"), ("silhouette", "Precision")}


def test_latex_is_booktabs_and_bolds_best():
    grid = tables.pivot_to_grid(
        _rows(),
        row_key="config",
        col_key="metric",
        value_key="value",
        std_key="std",
        row_order=["silhouette", "context_crop"],
        col_order=["AP", "Precision"],
        caption="Retrieval by constructor.",
        label="tab:retr",
        row_header="Constructor",
    )
    tex = tables.to_latex(grid)
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "\\caption{Retrieval by constructor.}" in tex
    assert "\\label{tab:retr}" in tex
    assert "\\textbf{0.55 $\\pm$ 0.04}" in tex  # best AP bolded, with std
    assert "0.42 $\\pm$ 0.03" in tex and "\\textbf{0.42" not in tex  # non-best not bold


def test_markdown_bolds_best_with_plusminus():
    grid = tables.pivot_to_grid(
        _rows(),
        row_key="config",
        col_key="metric",
        value_key="value",
        std_key="std",
        row_order=["silhouette", "context_crop"],
        col_order=["AP", "Precision"],
        row_header="Constructor",
    )
    md = tables.to_markdown(grid)
    assert "| Constructor | AP | Precision |" in md
    assert "**0.55 ± 0.04**" in md


def test_escapes_latex_specials():
    grid = tables.pivot_to_grid(
        [{"config": "masked_color", "metric": "AP", "value": 0.5}],
        row_key="config",
        col_key="metric",
        value_key="value",
        row_header="c",
    )
    assert "masked\\_color" in tables.to_latex(grid)


def test_write_grid_csv(tmp_path):
    grid = tables.pivot_to_grid(
        _rows(),
        row_key="config",
        col_key="metric",
        value_key="value",
        std_key="std",
        row_order=["silhouette", "context_crop"],
        col_order=["AP", "Precision"],
        row_header="Constructor",
    )
    path = tmp_path / "grid.csv"
    tables.write_grid_csv(path, grid)
    with open(path, encoding="utf-8-sig") as handle:
        back = list(csv.reader(handle))
    assert back[0] == ["Constructor", "AP", "Precision"]
    assert back[1] == ["silhouette", "0.42 ± 0.03", "0.60 ± 0.02"]
