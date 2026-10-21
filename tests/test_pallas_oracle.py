"""Timeouts must terminate isolated workers and the descendants holding their pipes."""

import json
import os
import sys
from pathlib import Path

import pytest

from tests.pallas_oracle import _run_worker, run_worker


def run_script(script, **options):
    return _run_worker([sys.executable, "-u", "-c", script], {"ok": True}, **options)


def report_worker_inputs(*args, **kwargs):
    print("READY", flush=True)
    print(
        "RESULT "
        + json.dumps(
            dict(
                args=args,
                kwargs=kwargs,
                platform=os.environ["JAX_PLATFORMS"],
                flags=os.environ["XLA_FLAGS"],
                cwd=os.getcwd(),
            )
        ),
        flush=True,
    )


def test_named_worker_preserves_arguments_and_isolates_cpu_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("JAX_PLATFORMS", "tpu")
    monkeypatch.setenv("XLA_FLAGS", "parent flags")
    monkeypatch.chdir(tmp_path)
    expected = dict(
        args=[None, True, 128, "quoted ' value", {"nested": [1, False]}],
        kwargs={"mode": "eager"},
        platform="cpu",
        flags="--xla_force_host_platform_device_count=3",
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert (
        run_worker(
            "tests.test_pallas_oracle",
            *expected["args"],
            entry="report_worker_inputs",
            kwargs=expected["kwargs"],
            expected=expected,
            devices=3,
        )
        == expected
    )
    assert os.environ["JAX_PLATFORMS"] == "tpu"
    assert os.environ["XLA_FLAGS"] == "parent flags"
    assert Path.cwd() == tmp_path


def test_worker_requires_a_successful_exit_and_matching_result():
    script = 'print("READY", flush=True); print(\'RESULT {"ok": true}\', flush=True)'
    assert run_script(script) == {"ok": True}
    with pytest.raises(AssertionError, match="exit 7"):
        run_script(script + "; raise SystemExit(7)")
    with pytest.raises(AssertionError, match="oracle failed"):
        run_script(script.replace("true", "false"))
    with pytest.raises(AssertionError, match="unexpected worker result"):
        run_script("print('RESULT {\"ok\": true}', flush=True)")


@pytest.mark.parametrize("ready,phase", [(False, "startup/compilation"), (True, "execution")])
def test_timeout_reports_phase_and_reaps_worker(tmp_path, ready, phase):
    pid_file = tmp_path / "worker.pid"
    script = (
        "import os, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        + ('print("READY", flush=True)\n' if ready else "")
        + "time.sleep(120)\n"
    )
    with pytest.raises(TimeoutError, match=phase):
        run_script(script, startup_timeout=1, execution_timeout=0.1)
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)


@pytest.mark.skipif(sys.platform != "linux", reason="checks Linux descendant process state")
def test_timeout_kills_descendants_even_after_the_worker_exits(tmp_path):
    child_pid = tmp_path / "child.pid"
    child = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(child_pid)!r}, 'w').write(str(os.getpid()))\n"
        'print("READY", flush=True)\n'
        "time.sleep(120)\n"
    )
    script = f"import subprocess, sys; subprocess.Popen([sys.executable, '-u', '-c', {child!r}])"
    with pytest.raises(TimeoutError, match="execution"):
        run_script(script, startup_timeout=5, execution_timeout=0.1)
    # An orphan can briefly remain a zombie until init reaps it; it must not run.
    status = Path(f"/proc/{int(child_pid.read_text())}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"
