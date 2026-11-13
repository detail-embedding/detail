"""Subprocess-based local executor (Section 6).

One worker subprocess per job; multi-GPU part-parallelism lives inside the
detection runner, not here. Communication is filesystem-only: ``submit`` writes
the run directory's job.json and spawns the worker; ``stream_progress`` polls it.
"""

import base64
import binascii
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from detail.core.dataclasses import Job, JobStatus, ProgressEvent, job_type_tag
from detail.executor import rundir
from detail.executor.base import Executor
from detail.executor.runners import (
    JOB_RUNNERS,
    RUNNER_MODULES_ENV,
    import_runner_modules,
)

_TERMINAL = {
    JobStatus.COMPLETE,
    JobStatus.FAILED,
    JobStatus.PARTIAL,
    JobStatus.CANCELLED,
}

#: How long :meth:`LocalExecutor._reap` waits for a worker that has already written a
#: terminal status to actually exit. Generous for an interpreter shutdown, short enough
#: that a wedged worker cannot hold the caller open on a run the record calls finished.
_REAP_TIMEOUT_S = 10.0


#: PROCESS_QUERY_LIMITED_INFORMATION — enough access to read exit status, nothing more.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
#: The sentinel GetExitCodeProcess returns for a process that has not yet terminated.
_STILL_ACTIVE = 259


def _alive(pid: int) -> bool:
    """True if ``pid`` is a running process, queried without side effects.

    On Windows this is deliberately **not** ``os.kill(pid, 0)``, because there it is
    not a query at all: ``signal.CTRL_C_EVENT == 0``, so the call is
    ``GenerateConsoleCtrlEvent(CTRL_C_EVENT, pid)`` — a Ctrl-C sent to process group
    ``pid``. It succeeds whether or not the process has exited, so every worker read
    as alive forever (a reattached run never noticed its worker die, a cancel always
    waited out its hard timeout); and aimed at a pid that is not a live group leader
    it is broadcast to every process on the console, the test runner included.
    ``GetExitCodeProcess`` is the real question — what ``Popen.poll()`` asks.
    ``tests/executor/test_process_liveness.py`` carries the measurements.
    """
    if os.name == "nt":
        import _winapi

        try:
            handle = _winapi.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        except PermissionError:
            return True  # exists, just not queryable by us
        except OSError:
            return False  # no such process
        try:
            return _winapi.GetExitCodeProcess(handle) == _STILL_ACTIVE
        finally:
            _winapi.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return (
            True  # exists, just not signalable by us (shouldn't happen for our worker)
        )
    return True


def _kill_tree(pid: int, *, force: bool) -> None:
    """Kill the worker's whole process tree (worker + its multiprocessing
    children). Windows uses taskkill /T; POSIX signals the recording group.
    ``force`` escalates to an unconditional kill (taskkill /F, SIGKILL) — note
    ``signal.SIGKILL`` does not exist on Windows, so it is referenced only on the
    POSIX branch."""
    if os.name == "nt":
        flag = ["/F"] if force else []
        subprocess.run(["taskkill", *flag, "/T", "/PID", str(pid)], capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass  # gone, or (recycled pid) not ours to signal — don't crash the caller


#: How long to wait for ``ps`` before treating it as unanswerable. Generous: it is a cheap
#: command, and the cost of being wrong is refusing a legitimate retry, not corrupting anything.
_PS_TIMEOUT_S = 10


def _is_our_worker(pid: int, run_dir: Path) -> bool:
    """Best-effort identity guard against PID reuse: the worker is launched as
    ``… detail.executor.worker <run_dir>``, so its unique run-dir path must
    appear in the process command line. A recycled pid won't match → we don't kill
    it (and so never force-kill an unrelated process tree). If the command line
    can't be read, fall back to the active-status guard."""
    if os.name == "nt":
        # Windows has no ``ps``; read the command line via CIM. PowerShell 5.1
        # (powershell.exe) ships with every Windows install. Base64 of the UTF-8 bytes,
        # because PowerShell writes a pipe in the console codepage — neither UTF-8 nor
        # knowable from here — so a run dir under `C:\Users\José` never matched (a live
        # worker read as "not ours") and a `”` in it left `stdout` None.
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$c = [string](Get-CimInstance Win32_Process -Filter 'ProcessId={pid}')"
            ".CommandLine; [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($c))",
        ]
    else:
        # -ww: unlimited width. Without it, ps truncates the command column at
        # the (tty-less) default ~80 cols, dropping the run-dir path at the end
        # of a long command line so the guard never matches (Linux CI).
        cmd = ["ps", "-ww", "-p", str(pid), "-o", "command="]
    # Every way of *failing to get an answer* has to give the same answer, and it has to be
    # the conservative one. This used to treat "ps could not be launched" as "assume it is ours"
    # and "ps ran and failed" as "not ours" — the same situation, opposite conclusions, and the
    # second is the dangerous direction: callers ask this to decide whether a worker is still
    # working (`live_process_pid`), so a False there says a live process is finished and lets a
    # re-submit stack a second one on the same weights and the same device. That is #280's
    # incident, reachable from a transient `ps` failure on a loaded box.
    #
    # The timeout is the same rule for the same reason: unbounded, a wedged `ps` hangs whoever
    # asked, and the callers are reconcile and re-submit paths.
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PS_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True  # can't check → defer to the active-status guard
    if out.returncode != 0:
        # `ps -p` also exits non-zero for "no such process", which sounds like it would now
        # call a dead pid alive. It cannot: both callers are `_alive(pid) and
        # _is_our_worker(...)`, so a pid that is already gone never reaches here. What is
        # left is the gap between those two calls, where "assume still ours" is the
        # conservative answer and the next poll corrects it.
        return True  # ps ran and refused to answer — "can't check", not "not ours"
    command_line = out.stdout
    if os.name == "nt":
        try:
            command_line = base64.b64decode(command_line.strip(), validate=True).decode(
                "utf-8", errors="replace"
            )
        except binascii.Error:
            return True  # an unreadable answer is "can't check", not "not ours"
    return str(run_dir) in command_line


def worker_alive(run_dir: Path | str) -> bool:
    """True if the run dir's recorded worker pid is a live process that is still
    ours (PID-reuse guarded). Lets reconcile tell a detached survivor (the worker
    outlived the GUI) from a dead orphan, without a live handle."""
    run_dir = Path(run_dir)
    pid = rundir.read_pid(run_dir)
    return pid is not None and _alive(pid) and _is_our_worker(pid, run_dir)


def new_process_group_kwargs() -> dict:
    """``subprocess.Popen`` kwargs that isolate the child in its own process group.

    On Windows this is ``CREATE_NEW_PROCESS_GROUP`` so that force-killing the worker
    (taskkill) cannot deliver a console Ctrl-C/Ctrl-Break to *this* process — a
    child spawned without it shares the parent's console group and a kill takes the
    parent down too. ``start_new_session`` is a POSIX-only no-op on Windows. On
    POSIX it starts a new recording so cancel can signal the whole group. Tests that
    spawn stand-in workers must use this too, or killing them interrupts the runner.
    """
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


_SPAWNED_COORDINATORS: list[subprocess.Popen] = []
#: Held for every change to the list above. It is reached from every thread that submits a run
#: (the UI's, a batch's) and from every thread that streams one's progress: a GUI worker per
#: open run page, so two of them can reap the same run at once.
_SPAWNED_LOCK = threading.Lock()


def spawn_coordinator(
    run_dir: Path | str,
    runner_modules: tuple[str, ...],
    entrypoint: str = "detail.executor.coordinator",
) -> int:
    """Spawn a detached **coordinator** for a durable run and return its pid. Same
    own-process-group detach recipe as a worker (``new_process_group_kwargs``), so it
    survives the launching app exiting; it orchestrates the run's workers and publishes
    ``<run_dir>/progress.json``. ``entrypoint`` picks which coordinator module runs —
    the per-recording processing coordinator (default) or the stage-sequencing distill
    coordinator (``detail.executor.distill_coordinator``); both take the run dir as argv
    and carry it in their command line so the liveness helpers below can PID-reuse-guard.
    Fire-and-forget by design — the handle is retained module-side only so a still
    running coordinator isn't GC-warned; exited ones are reaped/dropped."""
    run_dir = Path(run_dir)
    env = {**os.environ, RUNNER_MODULES_ENV: ",".join(runner_modules)}
    with rundir.open_log(run_dir, rundir.COORDINATOR_LOG_FILENAME) as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", entrypoint, str(run_dir)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            **new_process_group_kwargs(),
        )
    with _SPAWNED_LOCK:
        _SPAWNED_COORDINATORS[:] = [
            p for p in _SPAWNED_COORDINATORS if p.poll() is None
        ]
        _SPAWNED_COORDINATORS.append(proc)
    return proc.pid


def reap_coordinator(pid: int | None) -> None:
    """Wait on a coordinator this process spawned, once its run is terminal, and drop it.

    A coordinator writes the run's terminal status and *then* exits, so the moment a reader
    sees ``complete`` the process is usually still on its way out. Its handle was only pruned
    at the *next* spawn, so the last run of a session left a running coordinator behind — a
    ``ResourceWarning: subprocess … is still running`` at interpreter exit, five per parallel
    slow-tier run (#543). This is :meth:`LocalExecutor._reap` for coordinators: bounded for
    the same reason, since a coordinator wedged after its own last write must not hold the
    reader open, and a no-op for a pid this process did not spawn (a reattached run).

    **One reaper owns a handle.** It is taken out of the list under the lock *before* the wait,
    so a second thread reaping the same run (two open pages stream one run's progress) finds
    nothing to do. The list used to be read, waited on, and then ``remove``d by whoever got
    there: the second ``remove`` raised ``ValueError`` out of a progress worker's ``run`` and
    ended its stream. A coordinator still alive after the bounded wait is put back.
    """
    if pid is None:
        return
    with _SPAWNED_LOCK:
        mine = [p for p in _SPAWNED_COORDINATORS if p.pid == pid]
        for proc in mine:
            _SPAWNED_COORDINATORS.remove(proc)
    for proc in mine:
        try:
            proc.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # Still running: keep the handle. A warning at exit beats hanging the caller.
            with _SPAWNED_LOCK:
                _SPAWNED_COORDINATORS.append(proc)


def coordinator_alive(run_dir: Path | str, pid: int | None) -> bool:
    """True if the run's recorded coordinator pid is a live process still running
    our coordinator (PID-reuse guarded via the run-dir path in its command line) —
    the **authoritative** liveness check for reconcile. It shells out to ``ps`` /
    PowerShell, so hot poll loops should trigger on the cheap :func:`pid_alive` and
    only fall back to this to confirm a *dead*-looking coordinator."""
    return pid is not None and _alive(pid) and _is_our_worker(pid, Path(run_dir))


def live_process_pid(run_dir: Path | str) -> int | None:
    """The pid of a worker still *working* in this run dir, or ``None``.

    **A re-submit path must ask the process, not just the record.** A worker force-killed
    mid-flight never writes a terminal status, so the *owning* record (``model.yaml``) can
    say ``failed`` while that worker is alive and still holding the GPU. Re-submitting then
    stacks a second process on the same weights and the same device behind the first —
    #280's resume hung with the killed run's worker still resident, because `taskkill /T`
    on the *trainer* kills the subtree below it and leaves the worker above.

    The **envelope's own status**, not the bare pid, is what separates "still working" from
    "on its way out": a process that has recorded its outcome is finished even if its
    interpreter has not exited yet. Checking liveness alone races every normal shutdown —
    a run's pid is routinely still alive for a moment after it reports done — which turns a
    legitimate retry into a spurious refusal under load.

    Only meaningful for a **worker** run dir, the one shape that keeps its own job.json.
    A coordinator-backed run has none: there, an active record plus ``_reconcile_run``
    (which consults ``coordinator_alive``) already refuses a re-submit over a live
    coordinator, and that is the asymmetry this closes — ``resume_training`` is the
    re-submit path with no reconcile behind it.
    """
    run_dir = Path(run_dir)
    if not (run_dir / rundir.JOB_FILENAME).is_file():
        return None
    if rundir.read_status(run_dir) in _TERMINAL:
        return None  # it reported its outcome; a lingering interpreter is not work
    return rundir.read_pid(run_dir) if worker_alive(run_dir) else None


def pid_alive(pid: int | None) -> bool:
    """Cheap ``_alive`` liveness — no PID-reuse guard, no subprocess. For
    the reattach poll's hot path (a live coordinator writes a terminal status on
    normal completion, so the reuse guard only matters once it looks gone)."""
    return pid is not None and _alive(pid)


def kill_coordinator(run_dir: Path | str, pid: int | None) -> None:
    """Forcefully kill a run's coordinator process group (so it can't launch the next
    recording after a cancel). Force so ``taskkill`` gets ``/F`` on Windows — a graceful
    close a detached console process ignores; the coordinator has no cleanup that must
    run (``cancel_run`` writes the terminal status). No-op if gone or not ours."""
    if pid is None or not _is_our_worker(pid, Path(run_dir)):
        return
    if _alive(pid):
        _kill_tree(pid, force=True)


def request_stop(run_dir: Path | str) -> None:
    """Ask a run to stop at its next safe point, and **never kill it**.

    The difference from :func:`cancel_run_dir` is the whole point, and it is not a tuning
    parameter. Cancel is graceful-then-force: it gives a loop-based runner three seconds and then
    SIGTERMs the process group, because a blocking backend ignores the flag and a user pressing
    Cancel wants it to stop.

    **That kill is what makes "stop and keep" impossible.** Training only sees the flag at an
    epoch boundary (`trainers/ultralytics_yolo.py` sets `trainer.stop`), and an epoch on real data
    is minutes, not three seconds — so the worker is killed mid-epoch and
    `pipeline/training.py`'s post-loop block never runs. That block is what writes
    ``weights.best``, ``metrics``, ``model_family`` and ``classes`` into the record. Ultralytics
    has usually written ``best.pt`` to disk by then, and **nothing points at it**, so the model is
    not usable in the app. The weights survive; the model does not.

    So this sets the flag and returns. The run stops when it reaches a point where stopping is
    safe, runs its own completion path, and records what it produced. A caller that wants it gone
    *now* — and accepts losing the record — still has `cancel_run_dir`.

    A trainer that never reaches a safe point never stops, which is the honest cost: Cancel is
    the escape hatch, and it is the one that says it will stop the worker.

    **Its own marker, not Cancel's.** Both stop the runner, and the worker has to know which was
    asked: a stop is a finish the user chose, so the run ends ``complete``; only a cancel ends it
    ``partial``. It shared Cancel's flag until #486 — every stopped model then read as
    interrupted, and was offered Resume on a checkpoint Ultralytics had already finalized.
    """
    rundir.request_stop(Path(run_dir))


def cancel_run_dir(run_dir: Path | str, grace_s: float = 3.0) -> None:
    """Cancel a run by its directory alone — no live handle needed, so it works
    from a fresh process after the GUI restarted (the orphan case).

    Graceful-then-force: request the cooperative cancel flag and give a
    loop-based runner ``grace_s`` to stop and write its own terminal status;
    if it's still alive (a blocking training call ignores the flag), kill the
    worker's process group (SIGTERM, then SIGKILL). Writes ``cancelled`` only if
    the worker didn't already land a terminal status (preserves a clean partial).

    **A remote run's job is cancelled on the cluster from here.** The process here is then
    the shepherd (``remote_shepherd``), which delivers a cancel only when its next poll reads
    the flag — every few seconds, after an ssh round trip — so the grace above killed it first,
    and the Slurm job ran on to its walltime holding its GPUs, with nothing left that knew
    about it. So the ``scancel`` goes from here before the shepherd is stopped. When it could
    not be delivered (the cluster is unreachable), the shepherd is left running instead: it is
    the one process that asks again at every poll, and the run ends when it gets through.
    """
    from detail.executor import remote  # remote imports this module

    run_dir = Path(run_dir)
    rundir.request_cancel(run_dir)
    pid = rundir.read_pid(run_dir)
    # Don't signal a recycled pid that is no longer our worker (PID reuse guard).
    killable = pid is not None and _is_our_worker(pid, run_dir)
    if remote.cancel_scheduler_job(run_dir) is False and killable and _alive(pid):
        return
    deadline = time.monotonic() + max(grace_s, 0.0)
    while time.monotonic() < deadline:
        if not killable or not _alive(pid) or rundir.read_status(run_dir) in _TERMINAL:
            break
        time.sleep(0.1)
    if killable and _alive(pid):
        _kill_tree(pid, force=False)
        hard = time.monotonic() + 2.0
        while time.monotonic() < hard and _alive(pid):
            time.sleep(0.1)
        if _alive(pid):
            _kill_tree(pid, force=True)
    if rundir.read_status(run_dir) not in _TERMINAL:
        rundir.set_status(run_dir, JobStatus.CANCELLED, error="cancelled by user")


class LocalExecutor(Executor):
    """Runs jobs as local subprocesses (zero Qt dependency)."""

    def __init__(
        self,
        experiment_root: Path | str,
        runner_modules: tuple[str, ...] = (),
        poll_interval: float = 0.25,
    ) -> None:
        self._experiment_root = Path(experiment_root)
        self._runner_modules = tuple(runner_modules)
        self._poll_interval = poll_interval
        self._runs: dict[str, Path] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        import_runner_modules(self._runner_modules)

    def submit(self, job: Job) -> str:
        tag = job_type_tag(job)
        spec = JOB_RUNNERS.get(tag)
        if spec is None:
            raise ValueError(f"no runner registered for job type {tag!r}")
        run_dir = Path(spec.run_dir_for(self._experiment_root, job))
        run_dir.mkdir(parents=True, exist_ok=True)
        rundir.init_run_dir(run_dir, job)
        env = {**os.environ, RUNNER_MODULES_ENV: ",".join(self._runner_modules)}
        # own process group so cancel can reap the worker + its multiprocessing children;
        # stdout+stderr to run_dir/worker.log so the backend's own output survives the
        # launching console (rundir.open_log)
        with rundir.open_log(run_dir) as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "detail.executor.worker", str(run_dir)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                **new_process_group_kwargs(),
            )
        rundir.set_pid(run_dir, proc.pid)  # so a later/restarted cancel can find it
        self._runs[job.job_id] = run_dir
        self._procs[job.job_id] = proc
        return job.job_id

    def status(self, job_id: str) -> JobStatus:
        status = rundir.read_status(self._runs[job_id])
        if status in _TERMINAL:
            self._reap(job_id)
        return status

    def error(self, job_id: str) -> str | None:
        return rundir.read_envelope(self._runs[job_id]).get("error")

    def run_dir(self, job_id: str) -> Path:
        return self._runs[job_id]

    def cancel(self, job_id: str) -> None:
        cancel_run_dir(self._runs[job_id])

    def stream_progress(self, job_id: str) -> Iterator[ProgressEvent]:
        run_dir = self._runs[job_id]
        proc = self._procs.get(job_id)
        last = None
        while True:
            # A poll that cannot read the envelope learns nothing this tick and re-reads on the
            # next. `read_envelope` already retries the Windows sharing violation; when that
            # bounded retry is exhausted the honest answer is "ask again in a moment", not to
            # raise into the caller's isolation boundary, where it is recorded as the *work*
            # failing (#289).
            #
            # `status = None` rather than a `continue`: it must never *invent* a status, and it
            # must not skip the liveness checks below either — an envelope that is permanently
            # unreadable would otherwise poll a dead worker forever, trading one bad failure for
            # a hang.
            try:
                status = rundir.read_status(run_dir)
                progress = rundir.read_progress(run_dir)
            except (PermissionError, OSError, json.JSONDecodeError):
                status, progress = None, None
            if progress is not None and progress != last:
                yield progress
                last = progress
            if status in _TERMINAL:
                self._reap(job_id)
                return
            if proc is not None and proc.poll() is not None:
                # the worker exited without ever writing a terminal status
                if rundir.read_status(run_dir) not in _TERMINAL:
                    rundir.set_status(
                        run_dir,
                        JobStatus.FAILED,
                        error="worker exited without completing",
                    )
                return
            # A run this executor did not spawn — a reattach (§15), or any handle built in
            # a process that is not the launcher — has no Popen to poll, so the branch above
            # can never fire and a terminal status becomes this loop's *only* exit. A worker
            # killed before it wrote one then leaves the caller polling forever. The recorded
            # pid is the liveness signal in that case; `pid_alive` is the cheap trigger the
            # identity guard asks for, and `worker_alive` pays for `ps` only once it looks
            # dead, so a recycled pid is never mistaken for our live worker.
            #
            # Guarded on `pid is not None`: a coordinator-backed durable run records no
            # worker pid at all (only `submit` calls `set_pid`), so an absent pid means
            # "nothing to conclude from", not "dead".
            pid = rundir.read_pid(run_dir)
            if proc is None and pid is not None and not pid_alive(pid):
                if not worker_alive(run_dir):
                    if rundir.read_status(run_dir) not in _TERMINAL:
                        rundir.set_status(
                            run_dir,
                            JobStatus.FAILED,
                            error="worker exited without completing",
                        )
                    return
            time.sleep(self._poll_interval)

    def _reap(self, job_id: str) -> None:
        """Wait on the worker process so it is not left unreaped (avoids a
        ResourceWarning when the Popen is garbage-collected). Idempotent.

        Bounded, because reaping only happens once the status is already terminal: the
        worker has reported its outcome and is on its way out, so a wait that never returns
        is a worker wedged *after* its own last write — and blocking here would hold the
        caller open on a run that every reader of the record calls finished.
        """
        proc = self._procs.get(job_id)
        if proc is None:
            return
        try:
            proc.wait(timeout=_REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_tree(proc.pid, force=True)
            try:
                proc.wait(timeout=_REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass  # unreapable; a ResourceWarning beats hanging the caller
