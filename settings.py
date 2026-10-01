"""Runtime settings: every path, model name and external command comes from an environment
variable with a default. A `.env` file at the repo root is read once at import (values already
set in the environment win). See .env.example for the full list.
"""
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_dotenv(path=ROOT / ".env"):
    """Read KEY=VALUE lines into os.environ (existing variables are kept; no interpolation)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


def env(name, default=""):
    """The variable's stripped value, or `default` when unset or empty."""
    return os.environ.get(name, "").strip() or default


def env_path(name, default):
    """A path setting; a relative value is taken from the repo root."""
    value = env(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def env_command(name, default):
    """A command line as an argument list (the default is already a list)."""
    value = env(name)
    if not value:
        return default
    # Windows paths keep their backslashes; surrounding quotes are dropped either way.
    return [part.strip('"') for part in shlex.split(value, posix=os.name != "nt")]


DATA_DIR = env_path("CRA_DATA_DIR", ROOT / "data")  # memory and history databases, reuse index
LOGS_DIR = env_path("CRA_LOGS_DIR", ROOT / "logs")  # one folder per question run
CACHE_DIR = env_path("CRA_CACHE_DIR", ROOT / ".cache")  # source fulltext, titles, vocabulary
# The names community text uses for the corpus author ("<name> said ..."), as a regex
# alternation; used only to label attributed statements in secondary material.
AUTHOR_NAMES = env("CRA_AUTHOR_NAMES", "the author|author|the instructor|instructor")
