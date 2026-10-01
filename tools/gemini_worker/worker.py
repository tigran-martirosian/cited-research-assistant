"""Antigravity CLI (agy) transport for the Gemini worker: one headless, one-shot turn per call.

Adapted from Spotify's portal-ai-plugins (Apache-2.0, see NOTICE.md). It only runs the CLI and
never reads or reuses a credential. Python reads the files and puts their contents in the task;
only the final response is returned."""
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gemini-3.8-flash-medium"
# Prefixed to every task: the content is inline, so the worker needs no tools at all.
NO_TOOLS = ("Everything you need is included in this message. Do not use any tools, open files "
            "or run commands; answer from the content below.\n\n")


class WorkerError(Exception):
    """A failed delegation. kind is "auth" (sign-in missing, expired or rejected), "network"
    (unreachable, timed out, rate-limited, quota) or "provider" (anything else, including a
    missing CLI); only "auth" means the user must sign in to Antigravity again."""

    def __init__(self, message, kind=None):
        super().__init__(message)
        self.kind = kind or classify(message)


# Antigravity reports failures as text on stderr and, for API failures, an "AGY_ERROR: {...}"
# line with a canonical status such as UNAUTHENTICATED or UNAVAILABLE.
AUTH_TEXT = re.compile(r"\b(?:unauthenticated|not (?:signed|logged) in|(?:please |must |need to )?"
                       r"(?:sign|log) ?in (?:again|required|to continue)|sign-?in (?:is )?required|"
                       r"/login|no active session|credentials? (?:expired|revoked|invalid|missing)|"
                       r"token (?:has )?(?:expired|been revoked)|invalid_grant|401\b)", re.I)
NETWORK_TEXT = re.compile(r"\b(?:unavailable|deadline_exceeded|resource_exhausted|timed? ?out|"
                          r"network|connection (?:error|refused|reset)|econn\w+|getaddrinfo|dns|"
                          r"rate.?limit\w*|429|5\d\d|quota|overloaded)\b", re.I)


def classify(text):
    """"auth", "network" or "provider" for a worker failure message."""
    text = str(text or "")
    if AUTH_TEXT.search(text):
        return "auth"
    if NETWORK_TEXT.search(text):
        return "network"
    return "provider"


def agy_command():
    """The worker executable as an argv prefix. GEMINI_WORKER_AGY_CMD overrides it (a JSON list or a
    shell-style string), which the tests use to substitute a fake CLI."""
    override = os.environ.get("GEMINI_WORKER_AGY_CMD", "").strip()
    if override:
        return json.loads(override) if override.startswith("[") else shlex.split(override)
    found = shutil.which("agy")
    if not found:
        raise WorkerError("Antigravity CLI (agy) not found on PATH. Install it, run `agy` once "
                          "and choose Sign in with Google.", "provider")
    return [found]


def model():
    return os.environ.get("GEMINI_WORKER_MODEL", "").strip() or DEFAULT_MODEL


def timeout():
    value = os.environ.get("GEMINI_WORKER_TIMEOUT_SECONDS", "")
    return int(value) if value.isdigit() else 180


def usage_log():
    return Path(os.environ.get("GEMINI_WORKER_USAGE_LOG") or ROOT / ".cache" / "gemini_worker" / "usage.jsonl")


def parse_events(lines):
    """JSON objects from stream-json lines; anything else (banners, blank lines) is skipped."""
    for line in lines:
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def denials(events):
    """Actions the worker tried and was refused. It is told to use no tools, so any refused
    action means it did not work from the supplied content."""
    found = []
    for e in events:
        step = e.get("step_update") or {}
        info = step.get("tool_info") or {}
        err = info.get("error")
        text = json.dumps(err).lower() if err else ""
        if err and any(w in text for w in ("denied", "deny", "permission", "not allowed")):
            found.append(step.get("tool_name") or info.get("name") or "tool")
        result = e.get("result") or {}
        for key in ("denied", "denials", "permission_denials", "denied_actions"):
            if result.get(key):
                found.append(key)
    return found


def record(entry):
    """Append one telemetry line. Failures to log never break a delegation."""
    try:
        path = usage_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def converse(argv, message, cwd):
    """Send one user event and read events until the final result (or EOF / timeout), then close
    stdin and wait for exit. Threads read the pipes because Windows pipes cannot be polled."""
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd)
    lines, stderr = queue.Queue(), []
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()
    threading.Thread(target=lambda: stderr.extend(proc.stderr), daemon=True).start()
    events, deadline = [], time.monotonic() + timeout()
    try:
        try:
            proc.stdin.write(json.dumps({"event": "user", "message": {"content": message}}) + "\n")
            proc.stdin.flush()
        except OSError:
            pass  # it exited early; its exit code and stderr say why
        while True:
            try:
                line = lines.get(timeout=max(deadline - time.monotonic(), 0.01))
            except queue.Empty:
                raise WorkerError(f"Antigravity did not answer within {timeout()}s "
                                  "(raise GEMINI_WORKER_TIMEOUT_SECONDS or send fewer files).", "network")
            if line is None:
                break
            events += list(parse_events([line]))
            if events and events[-1].get("event") == "result":
                break
        try:
            proc.stdin.close()
        except OSError:
            pass
        code = proc.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired, WorkerError):
        proc.kill()
        proc.wait()
        raise
    finally:
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except OSError:
                pass
    return code, events, "".join(stderr)


def run(mode, task, meta):
    """Run one delegation and return the worker's response text. `task` is the full message
    (instructions and content); `meta` is size metadata for the log (no contents)."""
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "ts": round(time.time(), 3),
             "mode": mode, "model": model(), **meta,
             "input_bytes": len(task.encode("utf-8")), "ok": False}
    started = time.monotonic()
    try:
        argv = agy_command() + ["--input-format", "stream-json", "--output-format", "stream-json",
                                "--model", model()]
        with tempfile.TemporaryDirectory(prefix="gemini-worker-") as scratch:
            try:
                code, events, stderr = converse(argv, NO_TOOLS + task, scratch)
            except OSError as e:
                raise WorkerError(f"could not start the Antigravity CLI: {e}", "provider")
        init = next((e.get("init") or {} for e in events if e.get("event") == "init"), {})
        result = next((e.get("result") or {} for e in reversed(events)
                       if e.get("event") == "result"), None)
        tools = sum(1 for e in events if (e.get("step_update") or {}).get("tool_name"))
        entry.update(worker_models=[init["model"]] if init.get("model") else [],
                     worker_tool_steps=tools)
        if result is not None:
            usage = {k: v for k, v in (result.get("usage") or {}).items()
                     if isinstance(v, (int, float))}
            entry.update(worker_status=result.get("status"), worker_tokens=usage,
                         worker_duration_s=result.get("duration_seconds"),
                         worker_turns=result.get("num_turns"))
        detail = (stderr or "").strip()[-400:]
        if code != 0:
            raise WorkerError(f"Antigravity CLI exited with code {code}. {detail}")
        if result is None:
            raise WorkerError(f"Antigravity CLI ended without a result event. {detail}")
        if result.get("status") != "SUCCESS":
            raise WorkerError(f"Antigravity run status {result.get('status')}: "
                              f"{result.get('error') or detail}")
        denied = denials(events)
        if denied:
            raise WorkerError(f"Antigravity attempted denied actions ({', '.join(denied)}); "
                              "its answer was not based on the supplied content alone.")
        response = result.get("response")
        if not isinstance(response, str) or not response.strip():
            raise WorkerError("Antigravity returned an empty response.")
        entry.update(ok=True, response_bytes=len(response.encode("utf-8")))
        return response
    except WorkerError as e:
        entry.update(error=str(e)[:300], error_kind=e.kind)
        raise
    finally:
        entry["latency_ms"] = round((time.monotonic() - started) * 1000)
        record(entry)


def read_usage():
    try:
        lines = usage_log().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries
