"""Bounded, phase-aware subprocess runner for official TPU interpretation."""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def _terminate(process, reader):
    def send(*, force=False):
        try:
            if os.name == "posix":
                # Descendants may still hold stdout after the worker exits.
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif process.poll() is None:
                process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass

    send()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        send(force=True)
        process.wait(timeout=2)
    reader.join(timeout=2)
    if reader.is_alive():
        send(force=True)
        reader.join(timeout=2)


def run_oracle(case, mode, *, execution_timeout=30, devices=2):
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
    }
    return _run_worker(
        [sys.executable, "-u", "-m", "tests.pallas_communication_worker", case, mode],
        {"case": case, "mode": mode, "status": "ok", "races": False},
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        execution_timeout=execution_timeout,
    )


def _run_worker(command, expected, *, cwd=None, env=None, startup_timeout=90, execution_timeout=30):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name == "posix",
    )
    messages = queue.Queue()

    def drain():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    started = time.monotonic()
    deadline, phase = started + startup_timeout, "startup/compilation"
    output, result = [], None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Pallas oracle {expected}: {phase} timeout\n{''.join(output[-30:])}"
                )
            try:
                line = messages.get(timeout=min(remaining, 1))
            except queue.Empty:
                continue
            if line is None:
                break
            output.append(line)
            if line.strip() == "READY":
                assert phase == "startup/compilation", "worker announced READY twice"
                phase, deadline = "execution", time.monotonic() + execution_timeout
            elif line.startswith("RESULT "):
                assert phase == "execution" and result is None, "unexpected worker result"
                result = json.loads(line.removeprefix("RESULT "))
        code = process.wait(timeout=2)
        if code != 0 or result != expected:
            raise AssertionError(
                f"Pallas oracle failed during {phase} (exit {code}):\n{''.join(output[-40:])}"
            )
        return result
    finally:
        _terminate(process, reader)
        process.stdout.close()
