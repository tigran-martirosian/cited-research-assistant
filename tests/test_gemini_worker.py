"""Tests for the Antigravity transport in tools/gemini_worker/worker.py. Offline: agy is replaced
by tests/fake_agy.py; no network, no account, no quota.

Run from the repository root:  python -m unittest tests.test_gemini_worker -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
FAKE = REPO / "tests" / "fake_agy.py"
sys.path.insert(0, str(REPO / "tools" / "gemini_worker"))

import worker  # noqa: E402


class RunWithFakeAgy(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.record = self.tmp / "record.json"
        self.log = self.tmp / "usage.jsonl"
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_worker(self, mode="echo", task="Where is x3?", **env):
        env = {"GEMINI_WORKER_AGY_CMD": json.dumps([sys.executable, str(FAKE)]),
               "GEMINI_WORKER_USAGE_LOG": str(self.log), "FAKE_AGY_MODE": mode,
               "FAKE_AGY_RECORD": str(self.record), **env}
        with mock.patch.dict(os.environ, env):
            return worker.run("test", task, {"source_files": 1})

    def test_returns_only_the_response(self):
        self.assertEqual(self.run_worker(), "- `Thing` at x.ts:12: does a thing")
        call = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual(call["argv"], ["--input-format", "stream-json", "--output-format",
                                        "stream-json", "--model", "gemini-3.8-flash-medium"])
        content = call["event"]["message"]["content"]
        self.assertTrue(content.startswith(worker.NO_TOOLS))
        self.assertTrue(content.endswith("Where is x3?"))
        self.assertNotEqual(Path(call["cwd"]).resolve(), REPO)  # runs in an empty temp folder

    def test_model_is_configurable(self):
        self.run_worker(GEMINI_WORKER_MODEL="gemini-x-pro")
        call = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertEqual(call["argv"][-2:], ["--model", "gemini-x-pro"])

    def test_telemetry_has_sizes_not_contents(self):
        self.run_worker(task="SECRET_SOURCE_MARKER")
        raw = self.log.read_text(encoding="utf-8")
        self.assertNotIn("SECRET_SOURCE_MARKER", raw)
        [entry] = [json.loads(line) for line in raw.splitlines()]
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["worker_status"], "SUCCESS")
        self.assertEqual(entry["worker_tokens"]["total_tokens"], 1245)
        self.assertGreater(entry["response_bytes"], 0)

    def test_worker_failures_are_raised_and_logged(self):
        for mode, text, kind in (("crash", "exited with code 3. Error: not signed in", "auth"),
                                 ("exit", "exited with code 2", "provider"),
                                 ("noresult", "without a result event", "provider"),
                                 ("status", "status ERROR: quota exhausted", "network"),
                                 ("empty", "empty response", "provider"),
                                 ("denied", "denied actions (run_command)", "provider")):
            with self.subTest(mode=mode):
                with self.assertRaises(worker.WorkerError) as ctx:
                    self.run_worker(mode)
                self.assertIn(text, str(ctx.exception))
                self.assertEqual(ctx.exception.kind, kind)
                last = json.loads(self.log.read_text(encoding="utf-8").splitlines()[-1])
                self.assertFalse(last["ok"])
                self.assertEqual(last["error_kind"], kind)

    def test_timeout(self):
        with self.assertRaises(worker.WorkerError) as ctx:
            self.run_worker("hang", GEMINI_WORKER_TIMEOUT_SECONDS="2")
        self.assertIn("did not answer within 2s", str(ctx.exception))
        self.assertEqual(ctx.exception.kind, "network")

    def test_missing_cli(self):
        with mock.patch.dict(os.environ, {"GEMINI_WORKER_AGY_CMD": ""}), \
                mock.patch.object(worker.shutil, "which", return_value=None):
            with self.assertRaises(worker.WorkerError):
                worker.agy_command()


class Classify(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(worker.classify("UNAUTHENTICATED: please sign in again"), "auth")
        self.assertEqual(worker.classify('AGY_ERROR: {"status": "UNAVAILABLE", "code": 503}'), "network")
        self.assertEqual(worker.classify("something odd"), "provider")


class StreamParsing(unittest.TestCase):
    def test_parse_events_skips_noise(self):
        lines = ["Antigravity CLI starting", "", '{"event": "init"}', "{bad json", "[1]",
                 '{"event": "result"}']
        self.assertEqual([e["event"] for e in worker.parse_events(lines)], ["init", "result"])


if __name__ == "__main__":
    unittest.main()
