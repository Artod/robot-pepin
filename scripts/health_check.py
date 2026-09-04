#!/usr/bin/env python
"""Pepin launch-readiness check: poll every subsystem, report GO / NO GO.

Read-only: pings, register reads, one camera frame each. No motion, no
torque, no writes to any device. The probes live in ``pepin.health`` and are
shared with the menu-bar app; this is their command-line face.

Usage:
    uv run python scripts/health_check.py            # full check (~20 s)
    uv run python scripts/health_check.py --quick    # a few seconds, no camera frames
"""

import argparse
import logging
import sys

from pepin.health import Probe, run_health
from pepin.log import setup_logging
from pepin.transport import board_address

logger = logging.getLogger(__name__)


def show(probe: Probe) -> None:
    mark = "\033[32m GO \033[0m" if probe.ok else "\033[31mFAIL\033[0m"
    print(f"  [{mark}] {probe.system:<16} {probe.detail}", flush=True)
    logger.info("[%s] %-16s %s", "GO" if probe.ok else "FAIL", probe.system, probe.detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll every robot subsystem.")
    parser.add_argument("--quick", action="store_true", help="skip camera frames and ToF chip IDs")
    args = parser.parse_args()
    setup_logging("health_check", console=False)
    print("PEPIN LAUNCH READINESS CHECK")
    print("=" * 60)
    try:
        host = board_address()
    except ConnectionError as exc:
        print(f"  [\033[31mFAIL\033[0m] board            {exc}")
        sys.exit(1)
    report = run_health(host, full=not args.quick, on_probe=show)
    print("=" * 60)
    if report.all_go:
        print(f"  ALL SYSTEMS GO ({report.duration_s:.1f}s)")
        logger.info("ALL SYSTEMS GO (%.1fs)", report.duration_s)
    else:
        print(f"  NO GO — down: {', '.join(report.failed)}")
        logger.info("NO GO — down: %s", ", ".join(report.failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
