"""Shared pytest configuration.

Test tiers:
- ``tests/unit``      pure logic, milliseconds, run by the pre-commit hook;
- ``tests/hardware``  need the live robot (``pepin.local`` reachable); marked
  ``hardware`` and skipped unless ``--hardware`` is passed.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hardware", action="store_true", help="run tests that need the robot")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--hardware"):
        return
    skip = pytest.mark.skip(reason="needs the robot; pass --hardware")
    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip)
