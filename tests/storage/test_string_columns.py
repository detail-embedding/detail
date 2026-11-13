"""A text column read back from zarr is ``StringDType``; an unsized fixed-width cast refuses it.

**The class.** Variable-length text columns are written ``dtype="str"`` (§9 "Zarr write idiom
(v3)") and read back as numpy's ``StringDType``. Casting that to a fixed-width string with no
size — ``.astype(str)``, ``.astype("U")``, ``.astype(np.str_)``, ``np.asarray(col, dtype=str)``
— raises ``TypeError: cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>``
(numpy 2.4). Nothing is wrong at the line where it is written: the cast reads as a harmless
normalisation, and the test beside it builds the column by hand as ``np.array([...])``, which is
already ``<U`` and casts fine.

**How it hid.** ``keypoint_dataset._part_classes`` cast a detection part's ``class_name`` that
way. Its only test filled the part cache with a hand-built ``<U`` array, so the read never ran
against a real store, and building a keypoint training set from a two-class detector crashed at
the first frame a per-class teacher asked about.

**The rule.** Read a text column as it comes back; ``str(col[i])`` and ``==`` comparisons work on
``StringDType`` already. Cast only with an explicit size if a fixed width is ever needed. The
scan below reads the parse tree (the ``test_path_serialization.py`` idiom) and fails on any
unsized string cast in ``detail/``; the premise test pins why it exists.
"""

import ast
import pathlib
import textwrap

import numpy as np
import pytest

import detail
from detail.pipeline import keypoint_dataset
from detail.storage.zarr_io import DetectionPartReader, DetectionPartWriter

_UNSIZED_STRING_NAMES = {"str", "str_", "unicode_"}
_UNSIZED_STRING_LITERALS = {"str", "U", "<U", ">U", "=U", "unicode"}
_NUMPY_CONSTRUCTORS = {"asarray", "array", "asanyarray", "fromiter", "full", "empty"}


def _is_unsized_string_dtype(node: ast.AST) -> bool:
    """``str``, ``np.str_``, or a string literal naming an unsized unicode dtype."""
    if isinstance(node, ast.Name):
        return node.id in _UNSIZED_STRING_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _UNSIZED_STRING_NAMES
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in _UNSIZED_STRING_LITERALS
    return False


def _is_python_text(node: ast.AST) -> bool:
    """An input that is visibly Python strings, never a column: a literal, a list or tuple
    display, a comprehension, or a ``str(...)`` call. A constructor sizes ``<U`` from those
    itself, and ``storage/query_runs.py`` needs that fixed width for ``np.savez``."""
    return isinstance(
        node, (ast.Constant, ast.List, ast.Tuple, ast.ListComp, ast.GeneratorExp)
    ) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
    )


def _offenders(root: pathlib.Path) -> list[str]:
    """Every unsized fixed-width string cast under ``root`` whose input may be a column."""
    found = []
    for py in sorted(root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr == "astype":
                dtype = node.args[0] if node.args else None
                dtype = dtype or next(
                    (k.value for k in node.keywords if k.arg == "dtype"), None
                )
                if dtype is not None and _is_unsized_string_dtype(dtype):
                    found.append(f"{py.name}:{node.lineno}")
            elif (
                node.func.attr in _NUMPY_CONSTRUCTORS
                and not (node.args and _is_python_text(node.args[0]))
                and any(
                    k.arg == "dtype" and _is_unsized_string_dtype(k.value)
                    for k in node.keywords
                )
            ):
                found.append(f"{py.name}:{node.lineno}")
    return found


def test_no_unsized_string_cast_in_the_package():
    offenders = _offenders(pathlib.Path(detail.__file__).parent)
    assert not offenders, (
        "an unsized fixed-width string cast raises on a zarr text column (StringDType); "
        "read the column as it comes back, or give the cast an explicit size:\n"
        + "\n".join(offenders)
    )


def test_the_scan_would_catch_every_spelling_of_the_bug(tmp_path):
    spellings = {
        "astype_str": "np.asarray(col[:]).astype(str)",
        "astype_np_str": "col[:].astype(np.str_)",
        "astype_literal": 'col[:].astype("U")',
        "astype_keyword": "col[:].astype(dtype=str)",
        "asarray_dtype": "np.asarray(col[:], dtype=str)",
        "array_dtype_literal": 'np.array(col[:], dtype="str")',
        "wrapped": "np.asarray(\n        col[:]\n    ).astype(\n        str\n    )",
    }
    for name, expr in spellings.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "offender.py").write_text(
            f"def read(np, col):\n    return {expr}\n", encoding="utf-8"
        )
        assert _offenders(directory), f"the scan misses the {name!r} spelling"


def test_the_scan_passes_what_is_actually_correct(tmp_path):
    """The read as it comes back, a sized cast, zarr's own ``dtype="str"`` at creation, and a
    constructor handed Python strings (``storage/query_runs.py``'s ``np.savez`` columns).
    """
    (tmp_path / "fine.py").write_text(
        textwrap.dedent("""
            def read(np, col, root, sample):
                names = np.asarray(col[:])
                labels = [str(c) for c in col[:]]
                fixed = col[:].astype("U32")
                ints = col[:].astype(np.int64)
                root.create_array("class_name", shape=(0,), dtype="str")
                ids = np.array([str(r) for r in sample.ids], dtype="str")
                scope = np.array(str(sample.scope), dtype="str")
                return names, labels, fixed, ints, ids, scope
            """),
        encoding="utf-8",
    )
    assert _offenders(tmp_path) == []


@pytest.fixture
def detection_part(tmp_path):
    path = tmp_path / "part_001.zarr"
    with DetectionPartWriter(path) as writer:
        for row, name in enumerate(["Resident", "Intruder"]):
            writer.add_row(0, 0, row, name, 0.9, (0.1, 0.1, 0.2, 0.2), "")
    return path


def test_a_text_column_reads_back_as_string_dtype_which_an_unsized_cast_refuses(
    detection_part,
):
    """The premise the scan rests on. If zarr starts handing back ``<U``, this fails and the
    scan can be retired with it."""
    column = np.asarray(DetectionPartReader(detection_part).column("class_name")[:])
    assert isinstance(column.dtype, np.dtypes.StringDType)
    with pytest.raises(TypeError, match="StringDType"):
        column.astype(str)


def test_a_per_class_teacher_reads_a_real_part_s_classes(detection_part):
    """``_part_classes`` against a store the real writer made, not a hand-built cache."""
    source = {"detection_dir": detection_part.parent}
    names = keypoint_dataset._part_classes(source, "part_001")
    assert [str(n) for n in names] == ["Resident", "Intruder"]
    assert names[1] == "Intruder"
