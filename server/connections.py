"""Provider connections for the web app: Claude, NotebookLM and Gemini.

Every provider has the same states:
  unknown          not checked yet
  checking         a status check is running
  connected        signed in and usable
  disconnected     not signed in, or disconnected in this app
  connecting       a sign-in is in progress
  reauth_required  the saved sign-in expired or was rejected, by a status check or by a real call
                   that failed with an authentication error: sign in again
  error            the provider could not be checked or failed for another reason (network,
                   missing CLI, service error); never used for an authentication problem

Status checks and sign-ins reuse the local CLIs the product already runs, so credentials stay
where those CLIs keep them; this module never reads them.
  NotebookLM  status `notebooklm auth check --test --json` (the pipeline's own preflight);
              sign-in `notebooklm login --browser chrome`
  Claude      status `claude auth status --json`; sign-in `claude auth login --claudeai`
  Gemini      through the Antigravity CLI (`agy`), the same executable as the stream-json worker
              in tools/gemini_worker/worker.py. Status `agy -p /quota --output-format json`, a read-only
              slash command that print mode answers without an agent turn or quota; it fails
              fast when nobody is signed in. Sign-in runs `agy` itself, which signs in with
              Google when no session exists (it opens the browser). It gets its own console
              window on Windows, or a pseudo-terminal elsewhere, and is closed as soon as the
              status check sees the sign-in.

Disconnect is local to this app: the provider is no longer used or required-checked, and the
sign-in saved on the computer is left alone (it may serve other tools). Connect re-enables it, and
signs in only when the saved sign-in is not valid. Reconnect always runs the sign-in again.

Claude and NotebookLM are required for research; Gemini is optional.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import settings
from research import CLAUDE, ENV, NOTEBOOKLM, classify_failure, notebooklm_auth_status

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = settings.DATA_DIR / "connections.json"  # disconnected-in-app flags; no credentials
CHECK_TIMEOUT = 90  # seconds for one status check
LOGIN_TIMEOUT = 330  # seconds for a whole sign-in; notebooklm's own browser timeout is 300
LOGIN_POLL = 3  # seconds between status checks while a console sign-in runs
# The in-app sign-in command (CRA_NOTEBOOKLM_APP_LOGIN_CMD); the default assumes the default
# `uv run ... notebooklm` command and adds the browser extra.
NOTEBOOKLM_LOGIN = settings.env_command("CRA_NOTEBOOKLM_APP_LOGIN_CMD", [
    *NOTEBOOKLM[:-3], "--with", "notebooklm-py[browser]", "notebooklm",
    "login", "--browser", "chrome", "--browser-timeout", "300"])
URL = re.compile(r"https://\S+")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b[=>()][0-9A-Za-z]?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
OUTPUT_LINES = 40  # sign-in output kept for display
STATES = ("unknown", "checking", "connected", "disconnected", "connecting", "reauth_required",
          "error")


def load_worker():
    """The Gemini worker's Antigravity transport (tools/gemini_worker/worker.py), loaded by path."""
    spec = importlib.util.spec_from_file_location("gemini_worker", ROOT / "tools" / "gemini_worker" / "worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def kill_tree(proc):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
    except OSError:
        pass


def run_status(cmd, env=None):
    """Run a status command; return (returncode, stdout, stderr), or (None, "", message)."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, timeout=CHECK_TIMEOUT,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return None, "", f"The status check timed out after {CHECK_TIMEOUT}s."
    except OSError as e:
        return None, "", f"not installed or not runnable: {e}"
    return done.returncode, done.stdout or "", done.stderr or ""


# ---- providers --------------------------------------------------------------------------------
# probe() returns (status, account, detail): status is "connected", "signed_out" (no saved
# sign-in), "auth_failed" (a saved sign-in that is expired or rejected) or "error".

class Claude:
    key, name, required = "claude", "Claude", True
    about = "Plans, selects evidence and reasons with your Claude Code subscription."
    console = False

    def login_command(self):
        return [CLAUDE, "auth", "login", "--claudeai"], ENV

    def probe(self):
        code, out, err = run_status([CLAUDE, "auth", "status", "--json"], ENV)
        if code is None:
            return "error", None, f"Claude Code status check failed: {err}"
        try:
            data = json.loads(out)
        except ValueError:
            data = {}
        data = data if isinstance(data, dict) else {}
        if code == 0 and data.get("loggedIn") is True:
            return "connected", data.get("email"), None
        if code == 0 and data.get("loggedIn") is False:
            return "signed_out", None, "Claude Code is not signed in."
        kind = classify_failure(err or out)
        if kind == "auth":
            return "auth_failed", None, "The Claude Code sign-in is no longer valid."
        return "error", None, f"Claude Code status check failed: {(err or out).strip()[-200:]}"


class NotebookLM:
    key, name, required = "notebooklm", "NotebookLM", True
    about = "Searches the source library notebook with your Google account."
    console = False

    def login_command(self):
        return NOTEBOOKLM_LOGIN, None

    def probe(self):
        code, out, err = run_status(NOTEBOOKLM + ["auth", "check", "--test", "--json"])
        if code is None:
            return "error", None, f"NotebookLM status check failed: {err}"
        return notebooklm_auth_status(code, out, err)


class Gemini:
    key, name, required = "gemini", "Gemini", False
    about = (f"Gemini ({worker.DEFAULT_MODEL}) through the Antigravity CLI, signed in "
             "with your Google account.")
    console = True  # agy signs in from its own interactive terminal UI

    def login_command(self):
        return worker.agy_command(), None

    def probe(self):
        try:
            cmd = worker.agy_command()
        except worker.WorkerError:
            return "error", None, ("The Antigravity CLI (agy) is not installed. Install it from "
                                   "https://antigravity.google/cli, then connect Gemini.")
        code, out, err = run_status(cmd + ["-p", "/quota", "--output-format", "json"])
        if code is None:
            return "error", None, f"Antigravity status check failed: {err}"
        if code == 0:
            found = EMAIL.search(out)
            return "connected", found.group(0) if found else None, None
        text = ANSI.sub("", f"{err}\n{out}").strip()
        kind = worker.classify(text)
        if kind == "auth":
            signed_out = re.search(r"not (?:signed|logged) in|no active session|sign-?in (?:is )?required",
                                   text, re.I)
            return ("signed_out" if signed_out else "auth_failed"), None, "Antigravity is not signed in."
        return "error", None, f"Antigravity status check failed: {text[-200:] or f'exit {code}'}"


PROVIDERS = [NotebookLM(), Claude(), Gemini()]


# ---- persisted flags --------------------------------------------------------------------------

_state_lock = threading.Lock()


def load_flags():
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_flag(key, **values):
    with _state_lock:
        data = load_flags()
        data.setdefault(key, {}).update(values)
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        part = STATE_FILE.with_name(f"{STATE_FILE.name}.{os.getpid()}.part")
        part.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(part, STATE_FILE)


# ---- connections ------------------------------------------------------------------------------

class Connection:
    def __init__(self, provider):
        self.provider = provider
        self.key = provider.key
        self.lock = threading.Lock()
        flags = load_flags().get(self.key, {})
        self.disabled = bool(flags.get("disabled"))
        self.ever_connected = bool(flags.get("ever_connected"))
        self.state = "disconnected" if self.disabled else "unknown"
        self.account = None
        self.detail = DISABLED if self.disabled else None
        self.checked_at = None
        self.login = None  # the running sign-in process
        self.pty = None  # its pseudo-terminal, for a console sign-in outside Windows
        self.login_url = None
        self.output = []
        self.checking = None  # thread of the running check

    def view(self):
        with self.lock:
            p = self.provider
            return {"key": self.key, "name": p.name, "about": p.about, "required": p.required,
                    "state": self.state, "account": self.account, "detail": self.detail,
                    "checked_at": self.checked_at, "disabled": self.disabled,
                    "console_login": p.console and sys.platform == "win32",
                    "login_url": self.login_url if self.state == "connecting" else None,
                    "output": self.output[-8:] if self.state == "connecting" else []}

    def set(self, **fields):
        with self.lock:
            for name, value in fields.items():
                setattr(self, name, value)

    def apply(self, status, account, detail):
        """Turn a probe result into the connection state."""
        if status == "connected":
            state, detail = "connected", None
            if not self.ever_connected:
                self.ever_connected = True
                save_flag(self.key, ever_connected=True)
        elif status == "signed_out":
            state = "reauth_required" if self.ever_connected else "disconnected"
        elif status == "auth_failed":
            state = "reauth_required"
        else:
            state = "error"
        self.set(state=state, account=account, detail=detail, checked_at=time.time())

    # --- status checks ---

    def check(self, wait=True):
        """Check the sign-in now (one check at a time); with wait=False, in the background."""
        with self.lock:
            if self.state == "connecting" or self.disabled:
                return
            running = self.checking
            if running is None:
                self.state = "checking"
                running = self.checking = threading.Thread(target=self._check, daemon=True)
                running.start()
        if wait:
            running.join()

    def _check(self):
        try:
            result = self.provider.probe()
        except Exception as e:  # keep the app usable; the card shows what went wrong
            result = ("error", None, f"Status check failed: {e!r}")
        self.apply(*result)
        self.set(checking=None)

    def report_failure(self, kind, detail):
        """A real call to this provider failed. Only an authentication failure changes the
        state (to reauth_required); network and service failures leave it as it is."""
        if kind != "auth" or self.disabled:
            return
        with self.lock:
            if self.state == "connecting":
                return
            self.state, self.detail, self.checked_at = "reauth_required", detail, time.time()

    # --- sign-in ---

    def connect(self, force=True):
        """Sign in. force=False (Connect on a provider disconnected in this app) first re-enables
        it and signs in only when the saved sign-in is not valid."""
        with self.lock:
            if self.state == "connecting":
                return
            was_disabled = self.disabled
            self.disabled = False
            self.state, self.detail, self.login_url, self.output = "connecting", None, None, []
        if was_disabled:
            save_flag(self.key, disabled=False)
        threading.Thread(target=self._connect, args=(force and not was_disabled,), daemon=True).start()

    def _connect(self, force):
        if not force:
            result = self.provider.probe()
            if result[0] == "connected":
                self.apply(*result)
                return
        (self._login_console if self.provider.console else self._login)()

    def _login(self):
        """A sign-in CLI that exits when it is done (Claude, NotebookLM)."""
        cmd, env = self.provider.login_command()
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", env=env)
        except OSError as e:
            self.set(state="error", detail=f"Could not start the {self.provider.name} sign-in: {e}")
            return
        self.set(login=proc)
        timer = threading.Timer(LOGIN_TIMEOUT, kill_tree, [proc])
        timer.start()
        try:
            for line in proc.stdout:
                self._output(line)
            proc.wait()
        finally:
            timer.cancel()
        self._finish(proc)

    def _login_console(self):
        """A sign-in that happens inside an interactive terminal program (Antigravity's `agy`):
        it runs in its own console window (Windows) or a pseudo-terminal, and the status check is
        polled until it reports the sign-in; then the program is closed."""
        try:
            cmd, env = self.provider.login_command()
            if sys.platform == "win32":
                proc = subprocess.Popen(cmd, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                import pty
                master, slave = pty.openpty()
                proc = subprocess.Popen(cmd, env=env, stdin=slave, stdout=slave, stderr=slave,
                                        start_new_session=True, close_fds=True)
                os.close(slave)
                self.set(pty=master)
                threading.Thread(target=self._read_pty, args=(master,), daemon=True).start()
        except (OSError, worker.WorkerError) as e:
            self.set(state="error", detail=f"Could not start the {self.provider.name} sign-in: {e}")
            return
        self.set(login=proc)
        deadline = time.monotonic() + LOGIN_TIMEOUT
        result = None
        while time.monotonic() < deadline and self.login is proc:
            time.sleep(LOGIN_POLL)
            if self.login is not proc:
                break
            result = self.provider.probe()
            if result[0] == "connected" or proc.poll() is not None:
                break
        kill_tree(proc)
        with self.lock:
            master, self.pty = self.pty, None
        if master is not None:
            try:
                os.close(master)
            except OSError:
                pass
        self._finish(proc, result)

    def _read_pty(self, master):
        buf = ""
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                return
            if not chunk:
                return
            buf += ANSI.sub("", chunk.decode("utf-8", errors="replace"))
            *lines, buf = re.split(r"[\r\n]+", buf)
            for line in lines:
                self._output(line)

    def _output(self, line):
        line = line.rstrip()
        if not line.strip():
            return
        with self.lock:
            self.output = (self.output + [line])[-OUTPUT_LINES:]
            url = URL.search(line)
            if url and not self.login_url:
                self.login_url = url.group(0)

    def _finish(self, proc, result=None):
        with self.lock:
            cancelled = self.login is None
            self.login = None
            self.state = "checking"
        status, account, detail = result if result and result[0] == "connected" else self.provider.probe()
        if status != "connected" and (status != "error" or not detail):
            # An error keeps the provider's own reason (for example a missing CLI).
            tail = [line for line in self.output if not URL.search(line)][-3:]
            detail = ("Sign-in cancelled." if cancelled else
                      "Sign-in did not complete." + (f" {' '.join(tail)}" if tail else ""))
        self.apply(status, account, detail)

    def send_code(self, code):
        """Pass an authorization code to a sign-in that asks for one (when the browser cannot
        return to the CLI on its own)."""
        with self.lock:
            proc, master = self.login, self.pty
        if proc is None:
            return False
        try:
            if master is not None:
                os.write(master, (code.strip() + "\r").encode("utf-8"))
            elif proc.stdin is not None:
                proc.stdin.write(code.strip() + "\n")
                proc.stdin.flush()
            else:
                return False
        except OSError:
            return False
        return True

    def cancel(self):
        with self.lock:
            proc, self.login = self.login, None
        if proc:
            kill_tree(proc)

    def disconnect(self):
        """Stop using this provider in the app; the computer's saved sign-in is kept."""
        self.cancel()
        save_flag(self.key, disabled=True)
        self.set(disabled=True, state="disconnected", detail=DISABLED, account=None,
                 checked_at=time.time())


DISABLED = ("Disconnected in Cited Research Assistant. The sign-in saved on this computer is unchanged; "
            "connect again at any time.")

CONNECTIONS = {p.key: Connection(p) for p in PROVIDERS}


def gemini_runtime_failures():
    """An Antigravity call (the Gemini worker's telemetry) that failed with an authentication
    error after the last Gemini status check marks Gemini reauth_required."""
    c = CONNECTIONS["gemini"]
    try:
        entries = worker.read_usage()[-20:]
    except Exception:  # telemetry is best effort
        return
    for e in reversed(entries):
        if not isinstance(e.get("ts"), (int, float)) or e["ts"] <= (c.checked_at or 0):
            return
        if not e.get("ok") and e.get("error_kind") == "auth":
            c.report_failure("auth", "An Antigravity call failed to authenticate. Sign in again.")
            return


def all_views():
    gemini_runtime_failures()
    return [c.view() for c in CONNECTIONS.values()]


def report_failure(provider, kind, detail):
    """Route a runtime failure of a provider call to its connection (see report_failure)."""
    if provider in CONNECTIONS:
        CONNECTIONS[provider].report_failure(kind, detail)


def run_gemini(mode, task, meta):
    """Run one Gemini task through the Gemini worker's Antigravity stream-json worker (the product's
    Gemini runtime entry point). An authentication failure marks Gemini reauth_required."""
    c = CONNECTIONS["gemini"]
    if c.disabled:
        raise worker.WorkerError("Gemini is disconnected in Cited Research Assistant.", "provider")
    try:
        return worker.run(mode, task, meta)
    except worker.WorkerError as e:
        c.report_failure(e.kind, "A Gemini call failed to authenticate. Sign in again.")
        raise


def missing():
    """Names of required connections that are not usable; unchecked ones are checked first."""
    names = []
    for c in CONNECTIONS.values():
        if not c.provider.required:
            continue
        if c.state in ("unknown", "checking"):
            c.check()
        if c.state != "connected":
            names.append(c.provider.name)
    return names
