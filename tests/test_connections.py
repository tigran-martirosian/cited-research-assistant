"""Tests for provider connections (server/connections.py): one state model for Claude,
NotebookLM and Gemini, sign-in and reauthentication, in-app disconnect, and runtime failure
classification. Offline: status commands are scripted, and Antigravity is tests/fake_agy.py.

Run from the repository root:  python -m unittest tests.test_connections -v
"""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research  # noqa: E402
from server import connections as conn  # noqa: E402

FAKE_AGY = REPO / "tests" / "fake_agy.py"
try:
    import fastapi  # noqa: F401
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False


class Isolated(unittest.TestCase):
    """Each test gets its own flags file, Antigravity sign-in file and telemetry log."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.signed_in = self.tmp / "agy-signed-in"
        patches = [mock.patch.object(conn, "STATE_FILE", self.tmp / "connections.json"),
                   mock.patch.object(conn, "LOGIN_POLL", 0.2),
                   mock.patch.dict(os.environ, {
                       "GEMINI_WORKER_AGY_CMD": json.dumps([sys.executable, str(FAKE_AGY)]),
                       "FAKE_AGY_SIGNED_IN": str(self.signed_in),
                       "GEMINI_WORKER_USAGE_LOG": str(self.tmp / "usage.jsonl")})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("FAKE_AGY_STATUS", None)
        os.environ.pop("FAKE_AGY_MODE", None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def connection(self, key):
        provider = next(p for p in conn.PROVIDERS if p.key == key)
        return conn.Connection(provider)

    def wait(self, c, done=lambda c: c.state not in ("connecting", "checking"), seconds=20):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not done(c):
            time.sleep(0.05)
        return c


class StateModel(Isolated):
    def scripted(self, key, *results):
        """A connection whose status command returns these (code, stdout, stderr) in turn."""
        c = self.connection(key)
        patcher = mock.patch.object(conn, "run_status", side_effect=list(results))
        patcher.start()
        self.addCleanup(patcher.stop)
        return c

    def test_claude_signed_in_then_expired_needs_reauthentication(self):
        c = self.scripted("claude", (0, json.dumps({"loggedIn": True, "email": "a@b.c"}), ""),
                          (0, json.dumps({"loggedIn": False}), ""))
        c.check()
        self.assertEqual((c.state, c.account), ("connected", "a@b.c"))
        c.check()
        self.assertEqual(c.state, "reauth_required", "signed in before, so this is an expiry")
        self.assertTrue(json.loads(conn.STATE_FILE.read_text())["claude"]["ever_connected"])

    def test_never_signed_in_is_disconnected_not_reauth(self):
        c = self.scripted("claude", (0, json.dumps({"loggedIn": False}), ""))
        c.check()
        self.assertEqual(c.state, "disconnected")

    def test_network_failure_is_a_provider_error_not_an_auth_failure(self):
        c = self.scripted("claude", (1, "", "API Error: Connection error. getaddrinfo ENOTFOUND"))
        c.ever_connected = True
        c.check()
        self.assertEqual(c.state, "error")
        c = self.scripted("claude", (None, "", "The status check timed out after 90s."))
        c.check()
        self.assertEqual(c.state, "error")

    def test_notebooklm_statuses(self):
        ok = json.dumps({"status": "ok", "account": {"email": "n@b.c"}, "checks": {}, "details": {}})
        missing = json.dumps({"status": "error", "checks": {"storage_exists": False}, "details": {}})
        expired = json.dumps({"status": "error", "checks": {"storage_exists": True, "token_fetch": False},
                              "details": {"error": "Token fetch failed: Authentication expired. "
                                                   "Run 'notebooklm login' to re-authenticate."}})
        offline = json.dumps({"status": "error", "checks": {"storage_exists": True, "token_fetch": False},
                              "details": {"error": "Token fetch failed: [Errno -3] Temporary failure "
                                                   "in name resolution"}})
        self.assertEqual(research.notebooklm_auth_status(0, ok)[:2], ("connected", "n@b.c"))
        self.assertEqual(research.notebooklm_auth_status(1, missing)[0], "signed_out")
        self.assertEqual(research.notebooklm_auth_status(1, expired)[0], "auth_failed")
        self.assertEqual(research.notebooklm_auth_status(1, offline)[0], "error")
        c = self.scripted("notebooklm", (1, expired, ""))
        c.check()
        self.assertEqual(c.state, "reauth_required")

    def test_runtime_failures_change_state_only_for_authentication(self):
        c = self.scripted("claude", (0, json.dumps({"loggedIn": True}), ""))
        c.check()
        c.report_failure("network", "API Error: overloaded")
        self.assertEqual(c.state, "connected")
        c.report_failure("provider", "model error")
        self.assertEqual(c.state, "connected")
        c.report_failure("auth", "Claude needs you to sign in again.")
        self.assertEqual((c.state, c.detail), ("reauth_required", "Claude needs you to sign in again."))

    def test_failure_classification(self):
        cases = {"OAuth token has expired. Please obtain a new token": "auth",
                 "Invalid API key · Please run /login": "auth",
                 "Authentication error: Authentication expired. AUTH_ERROR": "auth",
                 "401 Unauthorized": "auth",
                 "API Error: Connection error.": "network",
                 "Network error: ConnectError NETWORK_ERROR": "network",
                 "NotebookLM search 1/2 timed out after 180s": "network",
                 "overloaded_error (529)": "network",
                 "rate limit exceeded (429)": "network",
                 "model not found": "provider"}
        for text, kind in cases.items():
            self.assertEqual(research.classify_failure(text), kind, text)
        worker = conn.worker
        self.assertEqual(worker.classify("Error: not signed in"), "auth")
        self.assertEqual(worker.classify('AGY_ERROR: {"status": "UNAUTHENTICATED"}'), "auth")
        self.assertEqual(worker.classify('AGY_ERROR: {"status": "UNAVAILABLE"}'), "network")
        self.assertEqual(worker.classify("Antigravity run status ERROR: quota exhausted"), "network")

    def test_disconnect_is_local_and_persisted(self):
        c = self.scripted("claude", (0, json.dumps({"loggedIn": True}), ""))
        c.check()
        c.disconnect()
        self.assertEqual((c.state, c.disabled), ("disconnected", True))
        self.assertIn("unchanged", c.detail)
        c.check()  # no status command runs while disconnected in the app
        self.assertEqual(c.state, "disconnected")
        again = self.connection("claude")
        self.assertEqual((again.state, again.disabled), ("disconnected", True), "survives a restart")

    def test_connect_after_disconnect_reuses_a_valid_sign_in(self):
        c = self.scripted("claude", (0, json.dumps({"loggedIn": True, "email": "a@b.c"}), ""))
        c.disconnect()
        with mock.patch.object(c, "_login", side_effect=AssertionError("no sign-in expected")):
            c.connect()
            self.wait(c)
        self.assertEqual((c.state, c.disabled), ("connected", False))
        self.assertFalse(json.loads(conn.STATE_FILE.read_text())["claude"]["disabled"])

    def test_only_required_providers_block_research(self):
        views = {}
        with mock.patch.dict(conn.CONNECTIONS, {p.key: self.connection(p.key) for p in conn.PROVIDERS}):
            for key, state in (("claude", "connected"), ("notebooklm", "connected"), ("gemini", "disconnected")):
                conn.CONNECTIONS[key].state = state
            self.assertEqual(conn.missing(), [])
            conn.CONNECTIONS["notebooklm"].state = "reauth_required"
            self.assertEqual(conn.missing(), ["NotebookLM"])
            views = {v["key"]: v for v in conn.all_views()}
        self.assertEqual(views["gemini"]["required"], False)
        self.assertEqual(views["claude"]["required"], True)
        self.assertEqual(set(views), {"claude", "notebooklm", "gemini"})


class GeminiConnection(Isolated):
    def test_status_uses_the_quota_command_and_reports_sign_in(self):
        c = self.connection("gemini")
        c.check()
        self.assertEqual(c.state, "disconnected")
        self.signed_in.write_text("g@b.c")
        c.check()
        self.assertEqual((c.state, c.account), ("connected", "g@b.c"))
        self.signed_in.unlink()
        c.check()
        self.assertEqual(c.state, "reauth_required")

    def test_unreachable_service_is_an_error(self):
        os.environ["FAKE_AGY_STATUS"] = "network"
        c = self.connection("gemini")
        c.check()
        self.assertEqual(c.state, "error")

    def test_missing_cli_is_an_error_with_install_hint(self):
        with mock.patch.dict(os.environ, {"GEMINI_WORKER_AGY_CMD": ""}), \
                mock.patch.object(conn.worker.shutil, "which", return_value=None):
            c = self.connection("gemini")
            c.check()
        self.assertEqual(c.state, "error")
        self.assertIn("not installed", c.detail)

    @unittest.skipIf(sys.platform == "win32", "the pseudo-terminal sign-in path is POSIX-only")
    def test_console_sign_in_captures_the_url_accepts_a_code_and_closes_agy(self):
        c = self.connection("gemini")
        c.connect()
        self.wait(c, lambda c: c.login_url is not None)
        view = c.view()
        self.assertEqual(view["state"], "connecting")
        self.assertTrue(view["login_url"].startswith("https://accounts.example.test/"))
        proc = c.login
        self.assertTrue(c.send_code("4/abc"))
        self.wait(c)
        self.assertEqual((c.state, c.account), ("connected", "person@example.test"))
        self.assertIsNotNone(proc.poll(), "the interactive agy is closed after sign-in")

    @unittest.skipIf(sys.platform == "win32", "the pseudo-terminal sign-in path is POSIX-only")
    def test_cancelled_sign_in(self):
        c = self.connection("gemini")
        c.connect()
        self.wait(c, lambda c: c.login is not None)
        c.cancel()
        self.wait(c)
        self.assertEqual(c.state, "disconnected")
        self.assertEqual(c.detail, "Sign-in cancelled.")

    def test_runtime_auth_failure_through_the_worker_marks_reauth(self):
        self.signed_in.write_text("g@b.c")
        c = self.connection("gemini")
        c.check()
        with mock.patch.dict(conn.CONNECTIONS, {"gemini": c}):
            os.environ["FAKE_AGY_MODE"] = "status"  # a run that fails with "quota exhausted"
            with self.assertRaises(conn.worker.WorkerError):
                conn.run_gemini("test", "hello", {})
            self.assertEqual(c.state, "connected", "quota is not a sign-in problem")
            os.environ["FAKE_AGY_MODE"] = "crash"  # "Error: not signed in"
            with self.assertRaises(conn.worker.WorkerError) as err:
                conn.run_gemini("test", "hello", {})
            self.assertEqual(err.exception.kind, "auth")
            self.assertEqual(c.state, "reauth_required")
        entries = conn.worker.read_usage()
        self.assertEqual([e.get("error_kind") for e in entries], ["network", "auth"])

    def test_auth_failure_in_worker_telemetry_marks_reauth(self):
        self.signed_in.write_text("g@b.c")
        c = self.connection("gemini")
        c.check()
        with mock.patch.dict(conn.CONNECTIONS, {"gemini": c}):
            conn.worker.record({"ts": time.time() - 3600, "ok": False, "error_kind": "auth"})
            conn.all_views()
            self.assertEqual(c.state, "connected", "a failure older than the last check is stale")
            conn.worker.record({"ts": time.time() + 1, "ok": False, "error_kind": "network"})
            conn.all_views()
            self.assertEqual(c.state, "connected")
            conn.worker.record({"ts": time.time() + 2, "ok": False, "error_kind": "auth"})
            conn.all_views()
            self.assertEqual(c.state, "reauth_required")

    def test_disconnected_gemini_refuses_runtime_calls(self):
        c = self.connection("gemini")
        c.disconnect()
        with mock.patch.dict(conn.CONNECTIONS, {"gemini": c}):
            with self.assertRaises(conn.worker.WorkerError):
                conn.run_gemini("test", "hello", {})


@unittest.skipUnless(HAVE_FASTAPI, "fastapi is not installed")
class AppFailureRouting(Isolated):
    """A research failure updates the provider's connection only for authentication failures."""

    @classmethod
    def setUpClass(cls):
        # Importing the app initializes the history database: keep it out of the real data/.
        from server import store
        cls.db_dir = tempfile.mkdtemp()
        cls.db_patch = mock.patch.object(store, "DB_PATH", Path(cls.db_dir) / "history.db")
        cls.db_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.db_patch.stop()
        shutil.rmtree(cls.db_dir, ignore_errors=True)

    def run_job(self, error):
        from server import app as web
        cs = {p.key: self.connection(p.key) for p in conn.PROVIDERS}
        for c in cs.values():
            c.state = "connected"
        with mock.patch.dict(conn.CONNECTIONS, cs), \
                mock.patch.object(web, "research", side_effect=error), \
                mock.patch.object(web.store, "update"):
            job = web.Job("id1", "q")
            with mock.patch.object(cs["notebooklm"], "check") as check:
                job.work()
        return job, cs, check

    def test_auth_failure_marks_the_provider_and_points_to_connections(self):
        from server import app as web
        job, cs, _ = self.run_job(research.ResearchError("claude said: OAuth token has expired",
                                                         provider="claude", kind="auth"))
        self.assertEqual(cs["claude"].state, "reauth_required")
        self.assertEqual(cs["notebooklm"].state, "connected")
        self.assertIn("Claude needs you to sign in again", cs["claude"].detail)
        self.assertEqual(job.events[-1]["status"], "error")
        self.assertTrue(web.SIGN_IN_AGAIN)

    def test_network_failure_does_not_ask_for_sign_in(self):
        _, cs, check = self.run_job(research.ResearchError("NotebookLM search 1/2 timed out",
                                                           provider="notebooklm", kind="network"))
        self.assertEqual(cs["notebooklm"].state, "connected")
        check.assert_called_once_with(wait=False)

    def test_disconnect_endpoint(self):
        from fastapi.testclient import TestClient
        from server import app as web
        cs = {p.key: self.connection(p.key) for p in conn.PROVIDERS}
        with mock.patch.dict(conn.CONNECTIONS, cs):
            client = TestClient(web.app)
            view = client.post("/api/connections/gemini/disconnect").json()
            self.assertEqual((view["state"], view["disabled"]), ("disconnected", True))
            self.assertEqual(client.post("/api/connections/nope/disconnect").status_code, 404)


if __name__ == "__main__":
    unittest.main()
