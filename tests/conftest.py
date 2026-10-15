"""Shared pytest configuration.

Test tiers:
- ``tests/unit``      pure logic, milliseconds (``slow`` marks the rest), run by the pre-push
  hook with pytest-xdist and the coverage floor;
- ``tests/hardware``  need the live robot (``pepin.local`` reachable); marked
  ``hardware`` and skipped unless ``--hardware`` is passed;
- ``tests/weights``   load real model weights (gigabytes, the GPU); marked ``weights`` and
  skipped unless ``--weights`` is passed.
"""

import os

import pytest

# The tests are the public tree's: a private overlay (pepin.overlay, companion/) present in the
# checkout never changes what a config file says to them. A test of the overlay sets its own.
os.environ["PEPIN_OVERLAY"] = "."

GATES = {"hardware": "needs the robot", "weights": "loads real model weights"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hardware", action="store_true", help="run tests that need the robot")
    parser.addoption("--weights", action="store_true", help="run tests that load real models")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for gate, why in GATES.items():
        if config.getoption(f"--{gate}"):
            continue
        skip = pytest.mark.skip(reason=f"{why}; pass --{gate}")
        for item in items:
            if gate in item.keywords:
                item.add_marker(skip)
