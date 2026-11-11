"""Whether a package is installed, as a freshly started interpreter would see it.

An editable install (``pip install -e ../sam3``) registers through a ``.pth`` file in
site-packages, and Python reads ``.pth`` files only at interpreter startup. So a probe run
inside the app (``find_spec`` alone, or after ``importlib.invalidate_caches()``) keeps
answering from the startup view. A backend installed while DETAIL is open reads as missing
until a restart, while every run worker, being a fresh interpreter, sees it. A regular
(non-editable) install needs none of this: the path finder notices a new directory in
site-packages by itself.

``is_importable`` is the one probe ``detail/`` uses. It re-reads the ``.pth`` files on a
miss, the way startup does, so the app and its workers agree on what is installed. A
reinstalled *compiled* package (a CPU torch swapped for a CUDA one) is beyond any re-read:
the running process keeps the extension it already loaded, and only a restart changes it.
"""

import importlib
import importlib.util
import site


def _site_dirs() -> list[str]:
    """The directories startup reads ``.pth`` files from, in startup's order."""
    dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    return dirs


def refresh_site_packages() -> None:
    """Re-read every site-packages ``.pth`` file, as interpreter startup does.

    Safe to repeat: ``site.addsitedir`` skips a path entry already on ``sys.path``, and the
    ``import`` lines it re-runs (setuptools' editable finders, ``_distutils_hack``) check
    before they install a hook.
    """
    importlib.invalidate_caches()
    for path in _site_dirs():
        site.addsitedir(path)


def is_importable(module: str) -> bool:
    """Whether ``module`` would import, without importing it (Tenet 13).

    Re-reads site-packages only on a miss, so a package the app already sees costs a
    single ``find_spec``.
    """
    if importlib.util.find_spec(module) is not None:
        return True
    refresh_site_packages()
    return importlib.util.find_spec(module) is not None
