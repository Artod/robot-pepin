import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pepin.log import setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Drop the file handler afterwards so later tests never write into a stale tmp dir."""
    yield
    logging.basicConfig(force=True)


def test_creates_exactly_one_timestamped_log_file(tmp_path: Path) -> None:
    logfile = setup_logging("probe", log_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [logfile]
    assert re.fullmatch(r"\d{8}_\d{6}_probe\.log", logfile.name)


def test_creates_the_log_directory(tmp_path: Path) -> None:
    logfile = setup_logging("probe", log_dir=tmp_path / "nested" / "logs")
    assert logfile.parent.is_dir()


def test_header_line_records_the_command_line(tmp_path: Path) -> None:
    logfile = setup_logging("probe", log_dir=tmp_path)
    header = logfile.read_text().splitlines()[0]
    assert " ".join(sys.argv) in header
    assert str(logfile) in header


def test_subsequent_records_reach_the_file(tmp_path: Path) -> None:
    logfile = setup_logging("probe", log_dir=tmp_path)
    logging.getLogger("test").info("hello %d", 42)
    assert "hello 42" in logfile.read_text()
