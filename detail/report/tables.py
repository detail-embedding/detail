"""Module-agnostic table writers: tidy CSV (the data) + a pivoted results grid (the views).

No pandas dependency — the tidy CSV is written with the stdlib ``csv`` module (a Quarto/pandas
user loads it themselves), and the results grid renders to LaTeX booktabs, Markdown, and CSV by
hand. ``pivot_to_grid`` turns tidy rows into a grid and marks the best value in each column.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from detail.core import atomic


def write_tidy_csv(path: Path | str, rows: list[dict], columns: list[str]) -> None:
    """Write long-form rows to ``path`` with ``columns`` as the header, atomically.

    Every row must carry every column key. This is the source-of-truth stats file; keep it long
    (one row per observation) so a notebook pivots it however it likes.
    """
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})
    atomic.replace(tmp, path)


@dataclass(frozen=True)
class Cell:
    """One results-grid cell: a numeric value, an optional std, and whether it wins its column."""

    value: float | int | None
    std: float | None = None
    is_best: bool = False
    decimals: int = 2

    def _num(self) -> str:
        if self.value is None:
            return ""
        if isinstance(self.value, float):
            return f"{self.value:.{self.decimals}f}"
        return str(self.value)

    def std_text(self, pm: str) -> str:
        """Formatted value with the ``± std`` suffix rendered using the given plus-minus token."""
        if self.value is None:
            return ""
        if self.std is None:
            return self._num()
        return f"{self._num()} {pm} {self.std:.{self.decimals}f}"


@dataclass(frozen=True)
class GridTable:
    """A pivoted results table: rows (e.g. configs) x columns (e.g. metrics), plus a caption."""

    caption: str
    label: str  # LaTeX \label id (also the file stem)
    row_header: str  # header cell above the row labels
    columns: list[str]
    row_labels: list[str]
    cells: list[list[Cell]]  # [row][col]


def pivot_to_grid(
    rows: list[dict],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    std_key: str | None = None,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    higher_is_better: bool = True,
    bold_best: bool = True,
    decimals: int = 2,
    row_header: str = "",
    caption: str = "",
    label: str = "",
) -> GridTable:
    """Pivot tidy ``rows`` into a grid keyed by ``row_key`` x ``col_key`` over ``value_key``.

    ``row_order`` / ``col_order`` fix the axis order (else first-seen). The best value in each
    column is flagged when ``bold_best`` (max if ``higher_is_better`` else min).
    """
    row_labels = row_order or _unique(r[row_key] for r in rows)
    col_labels = col_order or _unique(r[col_key] for r in rows)
    lookup = {(r[row_key], r[col_key]): r for r in rows}

    best_by_col: dict[str, float] = {}
    if bold_best:
        for col in col_labels:
            vals = [
                lookup[(row, col)][value_key]
                for row in row_labels
                if (row, col) in lookup and lookup[(row, col)][value_key] is not None
            ]
            if vals:
                best_by_col[col] = max(vals) if higher_is_better else min(vals)

    grid: list[list[Cell]] = []
    for row in row_labels:
        cells: list[Cell] = []
        for col in col_labels:
            rec = lookup.get((row, col))
            value = None if rec is None else rec[value_key]
            std = None if (rec is None or std_key is None) else rec.get(std_key)
            is_best = (
                bold_best
                and value is not None
                and col in best_by_col
                and value == best_by_col[col]
            )
            cells.append(Cell(value=value, std=std, is_best=is_best, decimals=decimals))
        grid.append(cells)
    return GridTable(
        caption=caption,
        label=label,
        row_header=row_header,
        columns=list(col_labels),
        row_labels=list(row_labels),
        cells=grid,
    )


def _unique(values) -> list:
    seen: dict = {}
    for v in values:
        seen.setdefault(v, None)
    return list(seen)


def to_latex(grid: GridTable) -> str:
    """A booktabs ``table`` float; best-in-column values are ``\\textbf``. Needs the ``booktabs``
    package. Underscores in labels are escaped."""
    align = "l" + "c" * len(grid.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{_tex(grid.caption)}}}",
        f"\\label{{{grid.label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join([_tex(grid.row_header)] + [_tex(c) for c in grid.columns]) + " \\\\",
        "\\midrule",
    ]
    for label, cells in zip(grid.row_labels, grid.cells):
        rendered = []
        for cell in cells:
            text = cell.std_text("$\\pm$")
            if cell.is_best and text:
                text = f"\\textbf{{{text}}}"
            rendered.append(text)
        lines.append(" & ".join([_tex(label)] + rendered) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def to_markdown(grid: GridTable) -> str:
    """A GitHub-flavored Markdown table; best-in-column values are ``**bold**``."""
    header = (
        "| "
        + " | ".join([md_cell(grid.row_header)] + [md_cell(c) for c in grid.columns])
        + " |"
    )
    sep = "| " + " | ".join(["---"] * (len(grid.columns) + 1)) + " |"
    lines = [header, sep]
    for label, cells in zip(grid.row_labels, grid.cells):
        rendered = []
        for cell in cells:
            text = cell.std_text("±")
            if cell.is_best and text:
                text = f"**{text}**"
            rendered.append(text)
        lines.append("| " + " | ".join([md_cell(label)] + rendered) + " |")
    return "\n".join(lines) + "\n"


def write_grid_csv(path: Path | str, grid: GridTable) -> None:
    """Write the grid as a plain CSV (no bolding; ``value±std`` text), atomically."""
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([grid.row_header] + grid.columns)
        for label, cells in zip(grid.row_labels, grid.cells):
            writer.writerow([label] + [cell.std_text("±") for cell in cells])
    atomic.replace(tmp, path)


_TEX_ESCAPE = {"_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#"}


def _tex(text: str) -> str:
    return "".join(_TEX_ESCAPE.get(ch, ch) for ch in str(text))


def md_cell(text: str) -> str:
    """Escape the cell delimiter, so a label containing one stays in its own cell.

    A comparison joins a multi-column group with `|` (`veh|am`), which is also Markdown's column
    separator: unescaped, a four-column table renders as six and every value sits under the wrong
    heading. LaTeX already had this via :func:`_tex`; Markdown did not.
    """
    return str(text).replace("|", "\\|")
