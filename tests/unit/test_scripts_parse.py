"""Every entry-point script must at least build its argument parser.

Scripts are not type-checked (argparse namespaces defeat mypy anyway), so a
misnamed option only shows up at launch time on the robot. ``--help`` exits
before touching hardware and catches import errors and parser mistakes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = ["drive.py", "navigate.py", "build_map.py", "base_smoke.py"]  # health_check takes no args


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_exits_cleanly(script: str) -> None:
    path = Path(__file__).resolve().parents[2] / "scripts" / script
    result = subprocess.run(
        [sys.executable, str(path), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr[-500:]
