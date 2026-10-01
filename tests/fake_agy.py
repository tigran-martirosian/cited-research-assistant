"""Stand-in for the Antigravity CLI (agy) in tests: no network, no account, no quota.

Speaks the headless stream-json protocol: reads ONE JSONL user event from stdin, writes init /
step_update / result events to stdout, then waits for stdin to close before exiting (like a real
stream-json session). Records argv, the parsed user event and cwd to FAKE_AGY_RECORD (JSON) and
answers according to FAKE_AGY_MODE.

Two more shapes, for the product's Connections tests:
- `agy -p /quota --output-format json` is the sign-in status check. It succeeds when the file
  named by FAKE_AGY_SIGNED_IN exists (its content is the account), else fails like a headless
  run with nobody signed in; FAKE_AGY_STATUS=network fails like an unreachable service.
- `agy` with no arguments is the interactive sign-in: it prints an authorization URL, reads a
  code from its terminal, writes the FAKE_AGY_SIGNED_IN file, and stays open like the real chat.
"""
import json
import os
import sys
import time

signed_in = os.environ.get("FAKE_AGY_SIGNED_IN")
if sys.argv[1:3] == ["-p", "/quota"]:
    if os.environ.get("FAKE_AGY_STATUS") == "network":
        print('AGY_ERROR: {"status": "UNAVAILABLE", "code": 503}', file=sys.stderr)
        sys.exit(3)
    if signed_in and os.path.exists(signed_in):
        with open(signed_in, encoding="utf-8") as f:
            print(json.dumps({"account": f.read().strip(), "quota": [{"model": "gemini", "remaining": 0.9}]}))
        sys.exit(0)
    print("error: not signed in. Run `agy` in a terminal to sign in with Google.", file=sys.stderr)
    sys.exit(1)
if not sys.argv[1:]:
    print("Sign in with Google: https://accounts.example.test/o/oauth2/auth?client=agy", flush=True)
    print("Paste the authorization code:", flush=True)
    code = sys.stdin.readline().strip()
    if signed_in and code:
        with open(signed_in, "w", encoding="utf-8") as f:
            f.write("person@example.test")
    while True:  # the chat UI stays open until it is closed
        time.sleep(1)

mode = os.environ.get("FAKE_AGY_MODE", "echo")
line = sys.stdin.readline()
record = os.environ.get("FAKE_AGY_RECORD")
if record:
    with open(record, "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv[1:], "event": json.loads(line), "cwd": os.getcwd()}, f)


def emit(obj):
    print(json.dumps(obj), flush=True)


def finish(code=0):
    sys.stdin.read()  # a stream-json session ends when its stdin closes
    sys.exit(code)


model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "?"
usage = {"input_tokens": 1200, "output_tokens": 40, "thinking_tokens": 5, "cache_read_tokens": 0,
         "total_tokens": 1245}


def result(response, status="SUCCESS", **extra):
    emit({"event": "result", "result": {"conversation_id": "conv-1", "status": status,
          "response": response, "duration_seconds": 1.5, "num_turns": 1, "usage": usage, **extra}})


if mode == "crash":
    print("Error: not signed in", file=sys.stderr)
    sys.exit(3)
print("Antigravity CLI starting")  # non-JSON noise is ignored
emit({"event": "init", "conversation_id": "conv-1",
      "init": {"cwd": os.getcwd(), "tools": ["view_file", "run_command"], "model": model}})
answer = os.environ.get("FAKE_AGY_RESPONSE", "- `Thing` at x.ts:12: does a thing")
if mode == "fenced":
    answer = "```python\n" + answer + "\n```\n"
emit({"event": "step_update", "step_update": {"conversation_id": "conv-1", "step_index": 1,
      "state": "ACTIVE", "step_type": "agent_response", "text_delta": answer[:5]}})
if mode == "noresult":  # the session ends mid-turn without a result event
    sys.exit(0)
if mode == "hang":  # the turn never finishes
    finish(0)
if mode == "status":
    result("", status="ERROR", error="quota exhausted")
    finish(0)
if mode == "empty":
    result("   ")
    finish(0)
if mode == "denied":
    emit({"event": "step_update", "step_update": {"conversation_id": "conv-1", "step_index": 2,
          "state": "DONE", "step_type": "tool", "tool_name": "run_command",
          "tool_info": {"name": "run_command", "error": {"type": "PermissionDenied",
                        "message": "Action denied: approval required"}}}})
result(answer)
finish(2 if mode == "exit" else 0)
