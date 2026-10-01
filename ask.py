"""Ask one question against the source library from the terminal.

The pipeline itself lives in research.py (shared with the web app in server/).

Usage: python ask.py [--fresh] [--follow <run dir>] "your question"
       python ask.py --preflight
--fresh (or CRA_FRESH=1) ignores exact reuse and the retrieval cache for this run.
--follow <run dir> asks a follow-up to that run's answer (logs/<timestamp>-<slug>): the question
is rewritten to stand alone from that exchange before research, and "Researching: ..." shows it.
--preflight prints the NotebookLM, Claude CLI and Antigravity (Gemini worker) status without any model call.
Everything for a question is saved in logs/<timestamp>-<slug>/.
"""
import shutil
import sys
import time

from research import ResearchCancelled, ResearchError, preflight, research


class Console:
    """Terminal progress: same-line stage timers, stage summary lines and the NotebookLM wait
    timer."""

    def __init__(self):
        self.inline = False  # a "- stage..." line is waiting for its time
        self.inline_stage = None  # the stage that line belongs to
        self.ticker = False  # a wait-timer line is showing
        self.streamed = ""  # answer text printed as it streamed
        self.mid_answer = False  # the streamed answer's last line is still open

    def __call__(self, ev):
        kind = ev["type"]
        if kind == "waiting":
            print(f"\r  {ev['label']} {ev['seconds']}s", end="", flush=True)
            self.ticker = True
            return
        if self.ticker:
            print("\r" + " " * 60 + "\r", end="", flush=True)
            self.ticker = False
        if self.mid_answer and kind != "answer_delta":
            print(flush=True)
            self.mid_answer = False
        if kind == "stage_start" and ev.get("cli"):
            print(f"- {ev['cli']}...", end="", flush=True)
            self.inline, self.inline_stage = True, ev.get("stage")
        elif kind == "plan" and ev.get("standalone"):
            if self.inline:
                print(flush=True)
                self.inline = False
            print(f"Researching: {ev['standalone']}", flush=True)
        elif kind == "stage_end":
            # Only the stage that opened the line closes it (the preflight, the Checking and
            # Writing steps end while another stage's line is open).
            if self.inline and ev.get("stage") == self.inline_stage:
                print(" " + ev["timed"], flush=True)
                self.inline = False
            if ev.get("summary"):
                print(f"- {ev['summary']}", flush=True)
        elif kind == "answer_delta":
            if self.inline:
                print(flush=True)
                self.inline = False
            if ev["reset"]:  # a new answer starts (a retried or final reasoner call)
                print("\n" if not self.streamed else "\n\n[answer restarted]\n", flush=True)
                self.streamed = ""
            print(ev["text"], end="", flush=True)
            self.streamed += ev["text"]
            self.mid_answer = True
        elif kind == "answer":
            if self.streamed.strip() != ev["answer"].strip():  # else: printed as it streamed
                print("\n" + ev["answer"], flush=True)

    def interrupt(self):
        if self.inline or self.ticker or self.mid_answer:
            print(flush=True)


def timing_summary(stages, total, first_answer=None):
    lines = ["", "Timing:"]
    lines += [f"  {s['stage']:<16} {s['seconds']:6.1f}s"
              + (f"  {s['chars']:,} chars" if s["chars"] is not None else "") for s in stages]
    lines.append(f"  {'Total':<16} {total:6.1f}s")
    if first_answer is not None:
        lines.append(f"  {'First answer text':<16} {first_answer:6.1f}s")
    print("\n".join(lines), file=sys.stderr)


def print_preflight():
    """The research preflight (forced, not cached) plus whether the Antigravity CLI is installed.
    Exit status 1 when research could not start."""
    start = time.monotonic()
    result = preflight(force=True)
    secs = time.monotonic() - start
    nlm, cli = result["notebooklm"], result["claude"]
    print(f"NotebookLM: {nlm['state']}" + (f" ({nlm['account']})" if nlm["account"] else "")
          + (f" - {nlm['detail']}" if nlm["detail"] else "") + f"  [{secs:.1f}s, both checks]")
    print(f"Claude CLI: {'ok ' + cli['version'] if cli['ok'] else 'not callable - ' + str(cli['detail'])}")
    print("Gemini worker: agy " + ("found at " + shutil.which("agy") if shutil.which("agy") else "not on PATH"))
    if not result["ok"]:
        print(f"\n{result['message']}")
    return 0 if result["ok"] else 1


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--preflight" in sys.argv[1:]:
        sys.exit(print_preflight())
    args = sys.argv[1:]
    follow = None
    if "--follow" in args:
        i = args.index("--follow")
        if i + 1 >= len(args):
            sys.exit("--follow needs the run folder of the answer to follow up")
        follow = args[i + 1]
        args = args[:i] + args[i + 2:]
    fresh = "--fresh" in args
    question = " ".join(a for a in args if a != "--fresh").strip()
    if not question:
        sys.exit('usage: python ask.py [--fresh] [--follow <run dir>] "your question"')
    console = Console()
    try:
        result = research(question, on_event=console, fresh=fresh, follow=follow)
    except ResearchCancelled as e:
        console.interrupt()
        print("\nResearch cancelled." + (f"\nLogs: {e.run_dir}" if e.run_dir else ""),
              file=sys.stderr)
        sys.exit(130)
    except ResearchError as e:
        console.interrupt()
        if e.run_dir is None:  # failed before the run started (e.g. NotebookLM login)
            sys.exit(str(e))
        print(f"\nFailed: {e}\nLogs: {e.run_dir}", file=sys.stderr)
        sys.exit(1)
    details = result["details"]
    reuse = details.get("exact_reuse") or {}
    if reuse.get("hit"):
        print(f"\nReused the result of an identical earlier question (run {reuse['source_run']}, "
              f"{reuse['age_hours']:.1f} h old); no research ran. Use --fresh to research again.",
              file=sys.stderr)
    else:
        timing_summary(details["stage_seconds"], details["total_seconds"],
                       details.get("first_answer_seconds"))
    print(f"\nLogs: {result['run_dir']}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C outside research() (e.g. while printing the summary)
        print("\nResearch cancelled.", file=sys.stderr)
        sys.exit(130)
