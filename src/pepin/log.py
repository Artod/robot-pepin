"""Logging setup shared by every entry point."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FORMAT = "%(asctime)s.%(msecs)03d %(levelname).1s %(name)s: %(message)s"
DATEFMT = "%H:%M:%S"


def git_sha() -> str:
    """Short SHA of the current commit, or ``"unknown"`` if git cannot answer."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def setup_logging(
    name: str, level: int = logging.INFO, log_dir: str | Path = "logs", *, console: bool = True
) -> Path:
    """Log to ``<log_dir>/<timestamp>_<name>.log`` (and the console unless ``console=False``).

    The first line written records the command line, the git revision and the
    log file itself, so a run can always be traced back to the code that made it.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.log"
    logging.basicConfig(
        level=level,
        format=FORMAT,
        datefmt=DATEFMT,
        handlers=[logging.FileHandler(logfile)] + ([logging.StreamHandler()] if console else []),
        force=True,
    )
    logging.getLogger(name).info(
        "run start: argv=%s git=%s log=%s", " ".join(sys.argv), git_sha(), logfile
    )
    return logfile
