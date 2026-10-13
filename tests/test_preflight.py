"""Exercise failure reporting and manifest policy without requiring a GPU."""

import json
import subprocess
from pathlib import Path

import pytest

from gpu_my_ocr import cli, preflight


@pytest.mark.parametrize("returncode,status", [(0, "passed"), (9, "failed")])
def test_command_preserves_evidence(monkeypatch, returncode, status):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], returncode, "out", "err"),
    )
    check = preflight.run_check(["nvidia-smi"])
    assert check["status"] == status
    assert check["returncode"] == returncode
    assert check["stdout"] == "out"
    assert check["stderr"] == "err"


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), subprocess.TimeoutExpired("x", 1)])
def test_command_failure_is_explicit(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert preflight.run_check(["missing"])["status"] == "failed"


def test_cuda_skip_cannot_claim_phase_readiness(monkeypatch):
    monkeypatch.setattr(preflight, "run_check", lambda *a, **kw: {"status": "passed"})
    manifest = preflight.collect_manifest()
    assert manifest["status"] == "passed"
    assert manifest["checks"]["torch"]["status"] == "skipped"
    assert not manifest["phase_0_ready"]


def test_failed_cuda_blocks_readiness(monkeypatch):
    monkeypatch.setattr(
        preflight,
        "run_check",
        lambda command, *a: {"status": "failed" if command[0] == "/missing" else "passed"},
    )
    manifest = preflight.collect_manifest(Path("/missing"))
    assert manifest["status"] == "failed"
    assert not manifest["phase_0_ready"]


def test_engine_interpreter_keeps_venv_symlink(monkeypatch, tmp_path):
    interpreter = tmp_path / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to("/usr/bin/python3")
    commands = []

    def capture(command, *args):
        commands.append(command)
        return {"status": "passed"}

    monkeypatch.setattr(preflight, "run_check", capture)
    manifest = preflight.collect_manifest(interpreter)
    assert commands[-1][0] == str(interpreter)
    assert manifest["phase_0_ready"]


def test_manifest_rejects_overwrite(tmp_path):
    output = tmp_path / "local" / "manifest.json"
    preflight.write_manifest({"status": "failed"}, output)
    with pytest.raises(FileExistsError):
        preflight.write_manifest({"status": "passed"}, output)
    assert json.loads(output.read_text()) == {"status": "failed"}


def test_cli_saves_failure_before_nonzero_exit(monkeypatch, tmp_path):
    output = tmp_path / "manifest.json"
    monkeypatch.setattr("sys.argv", ["gpu-my-ocr", "doctor", "--output", str(output)])
    monkeypatch.setattr(cli, "collect_manifest", lambda _: {"status": "failed"})
    with pytest.raises(SystemExit, match="1"):
        cli.main()
    assert json.loads(output.read_text()) == {"status": "failed"}
