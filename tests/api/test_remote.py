"""api/remote.py — the experiment-free remote-target catalog + connectivity check."""

import subprocess

import pytest
import yaml

from detail.api import remote as api_remote
from detail.configs.remote import RemoteTargetConfig
from detail.executor import ssh_transport
from detail.storage import credentials, remote_targets


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    cfg = tmp_path / "config" / "detail"
    monkeypatch.setattr(remote_targets, "CONFIG_DIR", cfg)
    monkeypatch.setattr(
        remote_targets, "REMOTE_TARGETS_PATH", cfg / "remote_targets.yaml"
    )


def _config(name="tillicum", **overrides):
    fields = dict(
        name=name,
        host="tillicum",
        remote_workdir="/gscratch/scrubbed/uwnetid/detail_runs",
    )
    fields.update(overrides)
    return RemoteTargetConfig(**fields)


def test_add_list_remove_round_trip():
    api_remote.add_remote_target(_config())
    assert [c.name for c in api_remote.list_remote_targets()] == ["tillicum"]
    assert api_remote.get_remote_target("tillicum").host == "tillicum"
    api_remote.remove_remote_target("tillicum")
    assert api_remote.list_remote_targets() == []


def test_list_sorted_by_name():
    api_remote.add_remote_target(_config(name="zeta", host="z"))
    api_remote.add_remote_target(_config(name="alpha", host="a"))
    assert [c.name for c in api_remote.list_remote_targets()] == ["alpha", "zeta"]


def test_remove_missing_raises_keyerror():
    with pytest.raises(KeyError):
        api_remote.remove_remote_target("nope")


def test_test_remote_target_missing_raises_keyerror():
    with pytest.raises(KeyError):
        api_remote.test_remote_target("nope")


class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_test_remote_target_ok(monkeypatch):
    api_remote.add_remote_target(_config())

    def fake_run(cmd, **kwargs):
        assert cmd[0] == "ssh"
        assert (
            cmd[-2] == "tillicum"
        )  # host passed positionally before the remote command
        return _FakeResult(returncode=0, stdout="DETAIL_SBATCH_OK\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = api_remote.test_remote_target("tillicum")
    assert status.ok
    assert "tillicum" in status.detail


def test_test_remote_target_sbatch_missing(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeResult(returncode=0, stdout="DETAIL_SBATCH_MISSING\n"),
    )
    status = api_remote.test_remote_target("tillicum")
    assert not status.ok
    assert "sbatch" in status.detail


def test_test_remote_target_unreachable(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _FakeResult(returncode=255, stderr="Connection refused"),
    )
    status = api_remote.test_remote_target("tillicum")
    assert not status.ok
    assert "Connection refused" in status.detail


def test_test_remote_target_timeout(monkeypatch):
    api_remote.add_remote_target(_config())

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = api_remote.test_remote_target("tillicum", timeout_s=1)
    assert not status.ok
    assert "timed out" in status.detail


def test_test_remote_target_no_local_ssh(monkeypatch):
    api_remote.add_remote_target(_config())

    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("ssh")

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = api_remote.test_remote_target("tillicum")
    assert not status.ok
    assert "ssh" in status.detail


# --------------------------------------------------------------------------- #
# push_credentials — forwarding this machine's configured credentials to a
# remote target (Hugging Face today; CREDENTIAL_SPECS is where a future
# provider, e.g. a cloud one, joins with no change to push_credentials itself)
# --------------------------------------------------------------------------- #
class _FakeRemoteHome:
    """Stands in for ssh_transport.run_remote across the three commands
    push_credentials issues in order: resolve $HOME, read the existing
    credentials.yaml (if any), write the merged one, chmod it. Records what was
    written so a test can assert on the merged content."""

    def __init__(self, home="/home/uwnetid", existing_yaml: str | None = None):
        self.home = home
        self.existing_yaml = existing_yaml
        self.written: dict[str, str] = {}
        self.calls: list[str] = []

    def __call__(self, host, command, *, stdin_text=None, timeout_s=None, check=True):
        self.calls.append(command)
        if command == "echo $HOME":
            return _Result(0, self.home + "\n", "")
        if command.startswith("cat "):
            if self.existing_yaml is None:
                return _Result(1, "", "No such file or directory")
            return _Result(0, self.existing_yaml, "")
        if "cat >" in command:
            path = command.split("cat > ", 1)[1].strip().strip("'")
            self.written[path] = stdin_text
            return _Result(0, "", "")
        if command.startswith("chmod "):
            return _Result(0, "", "")
        raise AssertionError(f"unexpected remote command: {command!r}")


class _Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_run_remote(monkeypatch, fake):
    def run_remote(
        host,
        command,
        *,
        stdin_text=None,
        timeout_s=None,
        check=True,
        refused_retries=0,
    ):
        result = fake(host, command, stdin_text=stdin_text, timeout_s=timeout_s)
        if check and result.returncode != 0:
            raise ssh_transport.RemoteCommandError(
                ["ssh", host, command], result.returncode, result.stdout, result.stderr
            )
        return result

    monkeypatch.setattr(ssh_transport, "run_remote", run_remote)


def test_push_credentials_requires_a_registered_target():
    with pytest.raises(KeyError):
        api_remote.push_credentials("nope")


def test_push_credentials_requires_a_local_credential(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: None)
    with pytest.raises(ValueError, match="no credentials are configured"):
        api_remote.push_credentials("tillicum")


def test_push_credentials_writes_to_a_fresh_remote_home(monkeypatch):
    """No credentials.yaml on the cluster yet (a fresh account) — the read fails,
    and that must not be treated as an error: push writes one from scratch."""
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")
    fake = _FakeRemoteHome(existing_yaml=None)
    _install_fake_run_remote(monkeypatch, fake)

    status = api_remote.push_credentials("tillicum")

    assert status.ok
    assert "tillicum" in status.detail
    assert "Hugging Face access" in status.detail
    written = fake.written["/home/uwnetid/.config/detail/credentials.yaml"]
    assert yaml.safe_load(written) == {"huggingface": "hf_test_token"}


def test_push_credentials_merges_rather_than_overwrites(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")
    fake = _FakeRemoteHome(existing_yaml=yaml.safe_dump({"some_other_key": "keep-me"}))
    _install_fake_run_remote(monkeypatch, fake)

    status = api_remote.push_credentials("tillicum")

    assert status.ok
    written = fake.written["/home/uwnetid/.config/detail/credentials.yaml"]
    assert yaml.safe_load(written) == {
        "some_other_key": "keep-me",
        "huggingface": "hf_test_token",
    }


def test_push_credentials_sets_restrictive_permissions(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")
    fake = _FakeRemoteHome()
    _install_fake_run_remote(monkeypatch, fake)

    api_remote.push_credentials("tillicum")

    assert any(c.startswith("chmod 600 ") for c in fake.calls)


def test_push_credentials_reports_ssh_failure(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")

    def run_remote(
        host,
        command,
        *,
        stdin_text=None,
        timeout_s=None,
        check=True,
        refused_retries=0,
    ):
        raise ssh_transport.RemoteCommandError(
            ["ssh", host, command], 255, "", "Connection refused"
        )

    monkeypatch.setattr(ssh_transport, "run_remote", run_remote)
    status = api_remote.push_credentials("tillicum")
    assert not status.ok
    assert "Connection refused" in status.detail


def test_push_credentials_reports_timeout(monkeypatch):
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")

    def run_remote(
        host,
        command,
        *,
        stdin_text=None,
        timeout_s=None,
        check=True,
        refused_retries=0,
    ):
        raise subprocess.TimeoutExpired(["ssh", host, command], timeout_s or 0)

    monkeypatch.setattr(ssh_transport, "run_remote", run_remote)
    status = api_remote.push_credentials("tillicum", timeout_s=1)
    assert not status.ok
    assert "timed out" in status.detail


def test_push_credentials_forwards_every_registered_spec(monkeypatch):
    """The generality claim, proven rather than asserted: a second credential type
    registered in CREDENTIAL_SPECS (standing in for a future cloud provider) is
    pushed alongside Hugging Face with no change to push_credentials itself."""
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        api_remote,
        "CREDENTIAL_SPECS",
        (
            *api_remote.CREDENTIAL_SPECS,
            api_remote.CredentialSpec(
                "some_cloud_provider", "Some Cloud provider access", lambda: "sk-fake"
            ),
        ),
    )
    fake = _FakeRemoteHome(existing_yaml=None)
    _install_fake_run_remote(monkeypatch, fake)

    status = api_remote.push_credentials("tillicum")

    assert status.ok
    assert "Hugging Face access" in status.detail
    assert "Some Cloud provider access" in status.detail
    written = fake.written["/home/uwnetid/.config/detail/credentials.yaml"]
    assert yaml.safe_load(written) == {
        "huggingface": "hf_test_token",
        "some_cloud_provider": "sk-fake",
    }


def test_push_credentials_skips_an_unconfigured_spec(monkeypatch):
    """A registered credential type with nothing set locally is left out of the
    push entirely, not pushed as an empty/None value."""
    api_remote.add_remote_target(_config())
    monkeypatch.setattr(credentials, "get_hf_token", lambda: "hf_test_token")
    monkeypatch.setattr(
        api_remote,
        "CREDENTIAL_SPECS",
        (
            *api_remote.CREDENTIAL_SPECS,
            api_remote.CredentialSpec(
                "some_cloud_provider", "Some Cloud provider access", lambda: None
            ),
        ),
    )
    fake = _FakeRemoteHome(existing_yaml=None)
    _install_fake_run_remote(monkeypatch, fake)

    status = api_remote.push_credentials("tillicum")

    assert "Some Cloud provider access" not in status.detail
    written = fake.written["/home/uwnetid/.config/detail/credentials.yaml"]
    assert "some_cloud_provider" not in yaml.safe_load(written)
