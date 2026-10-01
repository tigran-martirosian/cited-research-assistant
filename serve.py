"""Start the local research app: python serve.py [--port 8765] [--no-browser]

The default port comes from CRA_PORT (8765 when unset)."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn

import settings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=int(settings.env("CRA_PORT", "8765")))
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def listener_pid(port):
    """The PID listening on 127.0.0.1:port, or None."""
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
                return int(parts[4])
        return None
    out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def is_this_app(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2) as r:
            return "commit" in json.load(r)
    except (OSError, ValueError):
        return False


# An earlier copy still holding the port would keep serving its old code to the browser, so stop
# it first. Anything else on the port is left alone.
if port_busy(args.port):
    pid = listener_pid(args.port)
    if not is_this_app(args.port) or pid is None:
        sys.exit(f"Port {args.port} is in use by another program (PID {pid}); "
                 f"stop it or pass --port.")
    print(f"Stopping the earlier Cited Research Assistant server (PID {pid})")
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not port_busy(args.port):
            break
        time.sleep(0.1)
    else:
        sys.exit(f"The earlier server (PID {pid}) did not stop; end it in Task Manager.")

url = f"http://127.0.0.1:{args.port}"
if not args.no_browser:
    threading.Timer(1.5, webbrowser.open, [url]).start()
print(f"Cited Research Assistant: {url}")
uvicorn.run("server.app:app", host="127.0.0.1", port=args.port, log_level="warning")
