"""Local web app: FastAPI backend for the research pipeline plus the built React frontend.

Each question runs research.research() on its own thread. Progress events are persisted to the
history database and streamed to the browser with Server-Sent Events.
"""
import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research import (APP_VERSION, CODE_COMMIT, EXACT_REUSE_SERVE, ResearchCancelled,
                      ResearchError, code_changed_since_start, exact_lookup, research)
from server import connections, store, titles

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
TERMINAL_EVENT = "end"
# Terminal-only progress (wait timers, the CLI's line texts) is not kept.
SKIPPED_EVENTS = {"waiting", "waiting_end", "answer"}
# Streamed answer text: sent to live listeners, never stored with the question's events.
STREAM_EVENTS = {"answer_delta"}
CLI_FIELDS = ("cli", "timed", "summary")
SIGN_IN_AGAIN = "{name} needs you to sign in again. Reconnect it in Connections, then ask again."

logger = logging.getLogger("cra")
app = FastAPI(title="Cited Research Assistant")
store.init()
titles.backfill()


class Job:
    """A question being researched right now."""

    def __init__(self, research_id, question, fresh=False, follow=None):
        self.id = research_id
        self.question = question
        self.fresh = fresh
        self.follow = follow  # the run folder of the answer this question follows up
        self.events = []
        self.cancel = threading.Event()
        self.finished = False

    def on_event(self, event):
        if event["type"] in SKIPPED_EVENTS:
            return
        for field in CLI_FIELDS:
            event.pop(field, None)
        self.events.append(event)
        if event["type"] in STREAM_EVENTS:
            return  # live only: the stored result carries the whole answer
        store.update(self.id, events=[e for e in self.events if e["type"] not in STREAM_EVENTS])

    def work(self):
        fields = {}
        try:
            result = research(self.question, on_event=self.on_event, cancel=self.cancel,
                              fresh=self.fresh, follow=self.follow)
            fields = {"status": "done", "answer": result["answer"], "result": result,
                      "turn": result.get("turn")}
        except ResearchCancelled:
            fields = {"status": "cancelled"}
        except ResearchError as e:
            fields = {"status": "error", "error": str(e)}
            connection = connections.CONNECTIONS.get(e.provider)
            if connection and e.kind == "auth":
                # A sign-in failure: point to Connections, not to a terminal command.
                fields["error"] = SIGN_IN_AGAIN.format(name=connection.provider.name)
                connection.report_failure("auth", fields["error"])
            elif connection:  # network or service trouble: recheck, without assuming sign-in
                connection.check(wait=False)
        except Exception as e:  # keep the server alive and the history row honest
            logger.exception("research failed")
            fields = {"status": "error", "error": f"Unexpected error: {e!r}"}
        finally:
            fields.setdefault("status", "error")
            store.update(self.id, finished_at=time.time(), **fields)
            self.events.append({"type": TERMINAL_EVENT, "status": fields["status"], "at": time.time()})
            self.finished = True
            JOBS.pop(self.id, None)


JOBS: dict[str, Job] = {}


class Ask(BaseModel):
    question: str
    fresh: bool = False  # bypass exact reuse and the retrieval cache (CRA_FRESH=1 does too)
    # The answered question this one follows up (same conversation thread).
    previous_id: str | None = None


class Code(BaseModel):
    code: str


def found(research_id):
    item = store.get(research_id)
    if item is None:
        raise HTTPException(404, "No such question")
    return item


@app.get("/api/version")
def version():
    """The app version and the git commit of the code this backend started with, and the code
    and prompt files edited on disk since it started."""
    return {"app": APP_VERSION, "commit": CODE_COMMIT,
            "changed": code_changed_since_start()}


@app.get("/api/history")
def history():
    return store.history()


@app.post("/api/research")
def ask(body: Ask):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "The question is empty")
    follow = thread = None
    if body.previous_id:
        previous = found(body.previous_id)
        follow = (previous.get("result") or {}).get("run_dir")
        if previous["status"] != "done" or not follow:
            raise HTTPException(409, "The answer to follow up has not finished.")
        thread = previous["thread_id"]
    # An identical earlier result is served without any provider, so its connections need no
    # check (research() repeats the lookup and researches normally if the entry is gone by then).
    # A follow-up is never served that way. Serving is off, so every question checks.
    if (follow or not EXACT_REUSE_SERVE
            or not exact_lookup(question, body.fresh)["hit"]):
        missing = connections.missing()
        if missing:
            raise HTTPException(409, f"Connect {' and '.join(missing)} in Connections before asking.")
    job = Job(uuid.uuid4().hex[:12], question, body.fresh, follow)
    item = store.create(job.id, question, thread, body.previous_id if follow else None)
    JOBS[job.id] = job
    threading.Thread(target=job.work, name=f"research-{job.id}", daemon=True).start()
    if not follow:  # a new conversation: name it for the sidebar
        titles.start(job.id, question)
    return item


@app.get("/api/research/{research_id}")
def get(research_id: str):
    return found(research_id)


@app.post("/api/research/{research_id}/cancel")
def cancel(research_id: str):
    job = JOBS.get(research_id)
    if job:
        job.cancel.set()
    return {"ok": True}


@app.delete("/api/research/{research_id}")
def delete(research_id: str):
    job = JOBS.get(research_id)
    if job:
        job.cancel.set()
    store.delete(research_id)
    return {"ok": True}


def connection(key):
    if key not in connections.CONNECTIONS:
        raise HTTPException(404, "No such connection")
    return connections.CONNECTIONS[key]


@app.get("/api/connections")
def connection_status():
    """Every connection's state; ones never checked are checked in the background."""
    for c in connections.CONNECTIONS.values():
        if c.state == "unknown":
            c.check(wait=False)
    return connections.all_views()


@app.post("/api/connections/{key}/check")
def connection_check(key: str):
    c = connection(key)
    c.check()
    return c.view()


@app.post("/api/connections/{key}/connect")
def connection_connect(key: str):
    """Sign in (Reconnect). A provider disconnected in this app is re-enabled first and signs in
    only if its saved sign-in is not valid."""
    c = connection(key)
    c.connect()
    return c.view()


@app.post("/api/connections/{key}/disconnect")
def connection_disconnect(key: str):
    """Stop using the provider in this app; the sign-in saved on the computer is kept."""
    c = connection(key)
    c.disconnect()
    return c.view()


@app.post("/api/connections/{key}/code")
def connection_code(key: str, body: Code):
    c = connection(key)
    if not body.code.strip() or not c.send_code(body.code):
        raise HTTPException(409, "No login is waiting for a code")
    return c.view()


@app.post("/api/connections/{key}/cancel")
def connection_cancel(key: str):
    c = connection(key)
    c.cancel()
    return c.view()


def sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/research/{research_id}/events")
async def events(research_id: str, request: Request):
    """Every progress event so far, then live ones, ending with an "end" event."""
    job = JOBS.get(research_id)
    if job is None:  # finished (or from an earlier session): replay what was recorded
        item = found(research_id)

        async def replay():
            for event in item["events"] or []:
                yield sse(event)
            yield sse({"type": TERMINAL_EVENT, "status": item["status"]})

        return StreamingResponse(replay(), media_type="text/event-stream")

    async def stream():
        sent, last = 0, time.monotonic()
        while True:
            pending = job.events[sent:]
            for event in pending:
                yield sse(event)
            sent += len(pending)
            if job.finished and sent >= len(job.events):
                return
            if await request.is_disconnected():
                return
            if time.monotonic() - last > 15:
                yield ": keep-alive\n\n"
                last = time.monotonic()
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/")
    def missing_frontend():
        return {"error": "Frontend not built. Run: cd web && npm install && npm run build"}
