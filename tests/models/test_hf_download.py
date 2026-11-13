"""DETAIL calls huggingface_hub only in ways the installed version accepts.

**The class: a library option written for one version of a dependency, run against another.**
``huggingface_hub`` is unpinned in ``pyproject.toml``, and ``transformers<5`` holds it below
1.0. ``hf_access.install`` passed ``tqdm_class=`` to ``snapshot_download``, which 0.x accepts and
hands only to its file-count bar. The class forwarded byte bars only, so a SAM 3 or DINOv3
install reported nothing until it ended. Measured on 0.36.2: a real two-file download gave zero
progress events. A sibling build, passing ``tqdm_class=`` to ``hf_hub_download``, failed outright
with ``TypeError: hf_hub_download() got an unexpected keyword argument 'tqdm_class'``.

**How it hid.** Every test replaced the download with ``lambda *a, **k``, which takes any
keyword at all, so the ignored keyword never reached a real signature.

**The rule.** One download call, ``hf_access.download``. It measures progress by
reading the bytes in the repo's cache folder while ``snapshot_download`` runs, which every
version writes the same way, instead of asking the library for a hook. The scan below checks
every call DETAIL makes into ``huggingface_hub`` against the installed version's real signature,
and refuses a ``**kwargs`` it cannot check. The stand-ins here bind their arguments to the real
function's signature, so a keyword the real one refuses fails the test as well.

**Where the fix stops.** ``transformers``' ``from_pretrained`` calls are another library's API
and are not scanned. The other tests' ``lambda *a, **k`` stand-ins stay: the scan checks the
call sites themselves, whatever a test puts in their place.
"""

import ast
import inspect
import os
import time
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.file_download import repo_folder_name

import detail
from detail.models import hf_access

PACKAGE = Path(detail.__file__).parent

#: Names a module binds to the ``huggingface_hub`` package itself.
_MODULE_NAMES = {"hf", "huggingface_hub"}


def _hub_target(func, api_names: set[str], imported: dict[str, str]):
    """The dotted ``huggingface_hub`` name a call's function resolves to, or ``None``."""
    if isinstance(func, ast.Name) and func.id in imported:
        return imported[func.id]
    if not isinstance(func, ast.Attribute):
        return None
    owner = func.value
    if isinstance(owner, ast.Name) and owner.id in _MODULE_NAMES:
        return func.attr
    if isinstance(owner, ast.Name) and owner.id in api_names:
        return f"HfApi.{func.attr}"
    if (
        isinstance(owner, ast.Call)
        and isinstance(owner.func, ast.Attribute)
        and owner.func.attr == "HfApi"
        and isinstance(owner.func.value, ast.Name)
        and owner.func.value.id in _MODULE_NAMES
    ):
        return f"HfApi.{func.attr}"
    return None


def _resolve(dotted: str):
    obj = huggingface_hub
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def bad_hub_calls(source: str) -> list[str]:
    """Calls into huggingface_hub the installed version would refuse, as ``line: why``.

    A name it lacks, a keyword its signature lacks, or a ``**mapping`` that cannot be checked.
    """
    tree = ast.parse(source)
    api_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "HfApi"
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    imported = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "huggingface_hub"
        for alias in node.names
    }
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _hub_target(node.func, api_names, imported)
        if dotted is None:
            continue
        target = _resolve(dotted)
        if target is None:
            bad.append(f"{node.lineno}: huggingface_hub has no {dotted}")
            continue
        if not callable(target):
            continue
        params = inspect.signature(target).parameters
        open_ended = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for keyword in node.keywords:
            if keyword.arg is None:
                bad.append(f"{node.lineno}: **mapping into {dotted} cannot be checked")
            elif not open_ended and keyword.arg not in params:
                bad.append(f"{node.lineno}: {dotted} takes no {keyword.arg}=")
    return bad


def test_every_huggingface_hub_call_fits_the_installed_signatures():
    found = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        hits = bad_hub_calls(path.read_text(encoding="utf-8-sig"))
        if hits:
            found[path.relative_to(PACKAGE.parent).as_posix()] = hits
    assert (
        found == {}
    ), f"huggingface_hub {huggingface_hub.__version__} would refuse these calls: {found}"


def test_the_scan_would_catch_the_calls_that_broke():
    assert bad_hub_calls(
        "hf = _require_hf()\nhf.hf_hub_download(r, n, cache_dir=c, tqdm_class=T)\n"
    ) == ["2: hf_hub_download takes no tqdm_class="]
    assert bad_hub_calls("hf.snapshot_download(r, **kwargs)\n") == [
        "1: **mapping into snapshot_download cannot be checked"
    ]
    assert bad_hub_calls("api = hf.HfApi()\napi.get_paths_info(r, p, sizes=True)\n")
    assert bad_hub_calls("hf.HfApi().whoami(token=t, colour=1)\n")
    assert bad_hub_calls(
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(r, progress=p)\n"
    )
    assert bad_hub_calls("hf.download_everything(r)\n")


def test_the_scan_passes_calls_the_installed_version_accepts():
    assert not bad_hub_calls("hf.snapshot_download(r, cache_dir=c, allow_patterns=f)\n")
    assert not bad_hub_calls("other.hf_hub_download(r, tqdm_class=T)\n")


# --- the download and its progress ----------------------------------------------------- #


def _real_signature_stand_in(real, behaviour):
    """A stand-in that refuses what ``real`` would refuse, then does ``behaviour``."""
    signature = inspect.signature(real)

    def stand_in(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        return behaviour(bound.arguments)

    return stand_in


class _Info:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size


@pytest.fixture
def hub(monkeypatch):
    """A Hub whose two files arrive in the cache 64 KiB at a time, as a real download does."""
    sizes = {"config.json": 1_000, "weights.bin": 600_000}
    monkeypatch.setattr(hf_access, "_PROGRESS_POLL_S", 0.005)

    def write(arguments):
        blobs = Path(arguments["cache_dir"]) / repo_folder_name(
            repo_id=arguments["repo_id"], repo_type="model"
        )
        blobs = blobs / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        for name in arguments["allow_patterns"]:
            part = blobs / f"{name}.incomplete"
            with part.open("wb") as out:
                left = sizes[name]
                while left:
                    step = min(left, 65_536)
                    out.write(b"\0" * step)
                    out.flush()
                    left -= step
                    time.sleep(0.01)
            os.replace(part, blobs / name)
        return str(blobs)

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        _real_signature_stand_in(huggingface_hub.snapshot_download, write),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        _real_signature_stand_in(
            huggingface_hub.try_to_load_from_cache, lambda a: None
        ),
    )

    class _Api:
        get_paths_info = staticmethod(
            _real_signature_stand_in(
                huggingface_hub.HfApi().get_paths_info,
                lambda a: [_Info(p, sizes[p]) for p in a["paths"]],
            )
        )

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    return sizes


def test_the_cache_folder_is_where_huggingface_hub_keeps_the_repo():
    assert hf_access._repo_folder_name(
        "Qwen/Qwen3-VL-32B-Instruct"
    ) == repo_folder_name(repo_id="Qwen/Qwen3-VL-32B-Instruct", repo_type="model")


def test_a_download_reports_its_bytes_while_it_runs(hub):
    ticks = []

    hf_access.download("org/model", list(hub), progress_cb=ticks.append)

    total = sum(hub.values())
    assert {t.bytes_total for t in ticks} == {total}
    assert [t.bytes_done for t in ticks] == sorted(t.bytes_done for t in ticks)
    assert any(0 < t.bytes_done < total for t in ticks), "the bar only moved at the end"
    assert ticks[-1].bytes_done == total


def test_a_callback_error_reaches_the_caller(hub):
    def broken(_tick):
        raise ValueError("the caller's bug")

    with pytest.raises(ValueError, match="the caller's bug"):
        hf_access.download("org/model", list(hub), progress_cb=broken)
