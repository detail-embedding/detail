"""scikit-learn's OpenMP estimators run with no more threads than OpenBLAS can take at once.

**The class.** An OpenMP-parallel estimator whose thread count follows the host's cores, where
every thread calls OpenBLAS: k-means splits its rows into 256-row chunks and runs GEMM on each
chunk from its own thread. OpenBLAS is built for a fixed number of concurrent callers
(``MAX_THREADS``; 24 in the Windows numpy/scipy wheels). Past it, it prints "precompiled
NUM_THREADS exceeded, adding auxiliary array" and, on a many-core Windows machine, corrupted
its heap (0xc0000374): the pose-clustering worker died mid-fit with no traceback, only "worker
exited without completing".

**How it hid.** Only a sample with more chunks than OpenBLAS has slots reaches it, on a machine
with more cores than that. Small samples and hosts whose OpenBLAS allows more callers never
reach it; a many-recording sample on a many-core machine does. Nothing is wrong at the line
that constructs the
estimator, and CI's runners have too few cores to cross the limit.

**The rule.** A cluster method runs k-means through ``cluster_methods._common.kmeans_level``, which
holds OpenMP under ``blas_safe_openmp``. The scan below fails on any of these estimators built
anywhere else in ``detail/``.
"""

import ast
import pathlib

import numpy as np
import pytest

import detail
from detail.cluster_methods import _common

#: scikit-learn estimators whose fit runs an OpenMP team that calls BLAS from each thread.
_OPENMP_BLAS_ESTIMATORS = frozenset({"KMeans", "MiniBatchKMeans", "BisectingKMeans"})
_SEAM = pathlib.Path(_common.__file__).name


def _offenders(root: pathlib.Path) -> list[str]:
    """Every construction of an OpenMP+BLAS estimator under ``root``, outside the seam."""
    found = []
    for py in sorted(root.rglob("*.py")):
        if py.name == _SEAM and py.parent.name == "cluster_methods":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name in _OPENMP_BLAS_ESTIMATORS:
                found.append(f"{py.relative_to(root).as_posix()}:{node.lineno}")
    return found


def test_no_openmp_blas_estimator_outside_the_capped_seam():
    offenders = _offenders(pathlib.Path(detail.__file__).parent)
    assert not offenders, (
        "run k-means through cluster_methods._common.kmeans_level — built bare, it calls "
        "OpenBLAS from one OpenMP thread per core and crashed a 72-thread Windows worker:\n"
        + "\n".join(offenders)
    )


def test_the_scan_catches_a_bare_estimator(tmp_path):
    """Guard the guard: each spelling the seam exists to stop is flagged, and its absence is not."""
    spellings = {
        "name": "from sklearn.cluster import KMeans\nKMeans(n_clusters=3).fit(x)\n",
        "attribute": "import sklearn.cluster\nsklearn.cluster.MiniBatchKMeans(3).fit(x)\n",
        "wrapped": "from sklearn.cluster import BisectingKMeans\nBisectingKMeans(\n    3\n)\n",
    }
    for name, body in spellings.items():
        directory = tmp_path / name
        directory.mkdir()
        (directory / "offender.py").write_text(body, encoding="utf-8")
        assert _offenders(directory), f"the scan misses the {name!r} spelling"
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "fine.py").write_text(
        '"""KMeans( in a docstring is not a call."""\nkmeans_level(x, 3, 0)\n',
        encoding="utf-8",
    )
    assert not _offenders(clean)


def _loaded(api: str) -> list[dict]:
    from threadpoolctl import threadpool_info

    return [info for info in threadpool_info() if info["internal_api"] == api]


@pytest.fixture
def openblas_and_openmp():
    """The two libraries the cap is between, loaded; skip where either is absent."""
    pytest.importorskip("sklearn.cluster")
    if not _loaded("openblas") or not _loaded("openmp"):
        pytest.skip("the cap only applies where OpenBLAS and OpenMP are both loaded")
    if max(info["num_threads"] for info in _loaded("openmp")) <= 2:
        pytest.skip("needs more than two OpenMP threads to show a cap of two")


def test_openmp_is_held_to_openblas_threads(openblas_and_openmp):
    """A two-thread OpenBLAS stands in for a build with fewer slots than the host has cores."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2, user_api="blas"):
        with _common.blas_safe_openmp():
            inside = [info["num_threads"] for info in _loaded("openmp")]
    assert max(inside) <= 2
    assert max(info["num_threads"] for info in _loaded("openmp")) > 2, "the cap leaked"


def test_kmeans_level_fits_under_the_cap(openblas_and_openmp, monkeypatch):
    """The fit itself runs capped — not just a context manager that exists beside it."""
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    seen = []
    original_fit = KMeans.fit

    def recording_fit(self, *args, **kwargs):
        seen.append(max(info["num_threads"] for info in _loaded("openmp")))
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(KMeans, "fit", recording_fit)
    sample = np.random.default_rng(0).standard_normal((600, 8))
    with threadpool_limits(limits=2, user_api="blas"):
        (labels,) = _common.kmeans_level(sample, 5, seed=0)
    assert seen and max(seen) <= 2
    assert labels.shape == (600,) and labels.dtype == np.int64
    assert sorted(np.unique(labels)) == [0, 1, 2, 3, 4]
