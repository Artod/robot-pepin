#!/usr/bin/env python
"""Live robot dashboard in rerun: what the robot sees, measures and feels, in one window.

Panels: the lidar scan around the robot (optionally over a saved map), the
overview camera, ToF ranges and lidar rate over time, board vitals, and a
text log. Read-only — it never commands the servo bus. It can run next to
drive.py or navigate.py: when a drive already holds the lidar bridge the
dashboard leaves it alone (ser2net would otherwise hand the lidar to the
newcomer and blind the driver) and shows camera, ToF and vitals only.
Battery voltage and heading have no sensor yet and are shown as such.

Usage:
    uv run python scripts/dashboard.py [--map data/maps/<map>.npz] [--camera-fps 5]
"""

import argparse
import logging
import queue
import threading
import time
import urllib.request
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from pepin.health import BoardVitals, HealthReport, busy_bridge_ports, probe_board
from pepin.lidar import LidarClient, LidarMount
from pepin.log import setup_logging
from pepin.mapping import OccupancyGrid
from pepin.telemetry import LatencyTracker
from pepin.tof import TOF_PORT, TofClient
from pepin.transport import LIDAR_PORT, board_address

logger = logging.getLogger(__name__)
CAMERA_URL = "http://{host}:8080/snapshot"
VITALS_EVERY_S = 10.0


def camera_thread(
    host: str, fps: float, frames: "queue.Queue[bytes]", stop: threading.Event
) -> None:
    """Snapshots from the board's MJPEG server at a modest rate."""
    url = CAMERA_URL.format(host=host)
    period = 1.0 / fps
    while not stop.is_set():
        started = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                frames.put(response.read())
        except OSError as exc:
            logger.warning("camera snapshot failed: %s", exc)
            stop.wait(2.0)
        stop.wait(max(0.0, period - (time.monotonic() - started)))


def blueprint() -> rrb.Blueprint:
    """Window layout: scan+map on the left, camera and plots on the right, log below."""
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Lidar / map", origin="world"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Overview camera", origin="camera"),
                    rrb.TimeSeriesView(name="ToF ranges, m", origin="tof"),
                    rrb.TimeSeriesView(name="Rates and vitals", origin="rates"),
                ),
                column_shares=[3, 2],
            ),
            rrb.TextLogView(name="Log", origin="log"),
            row_shares=[4, 1],
        ),
        collapse_panels=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live rerun dashboard for the robot.")
    parser.add_argument("--map", type=Path, help="saved occupancy grid to draw under the scan")
    parser.add_argument("--camera-fps", type=float, default=5.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument(
        "--headless",
        type=float,
        metavar="SECONDS",
        help="no viewer: record that many seconds into data/videos/dashboard_<time>.rrd and exit",
    )
    args = parser.parse_args()
    setup_logging("dashboard")
    host = board_address()
    mount = LidarMount.from_json("config/lidar.json")

    rr.init("pepin-dashboard", spawn=args.headless is None, default_blueprint=blueprint())
    if args.headless is not None:
        out = Path("data/videos") / f"dashboard_{time.strftime('%Y%m%d_%H%M%S')}.rrd"
        out.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(out))
        logger.info("headless: recording %.0f s to %s", args.headless, out)
    rr.log("log", rr.TextLog(f"dashboard connected to {host}", level=rr.TextLogLevel.INFO))
    if args.map:
        grid = OccupancyGrid.load(args.map)
        rr.log(
            "world/map",
            rr.Points2D(grid.occupied_xy(), colors=[90, 90, 90], radii=0.025),
            static=True,
        )
    rr.log(
        "world/robot",
        rr.Arrows2D(origins=[[0.0, 0.0]], vectors=[[0.3, 0.0]], colors=[60, 120, 255], radii=0.02),
        static=True,
    )

    frames: queue.Queue[bytes] = queue.Queue()
    stop = threading.Event()
    lidar: LidarClient | None = None
    if LIDAR_PORT in busy_bridge_ports(host):
        logger.warning("lidar bridge in use by a drive script; the scan panel stays empty")
        rr.log("log", rr.TextLog("lidar in use by a drive — not connecting", level="WARN"))
    else:
        # Passive: if a drive starts and takes the bridge, stay off it (do not fight back).
        lidar = LidarClient(host, mount, reconnect=False).start()
    if not args.no_camera:
        threading.Thread(
            target=camera_thread, args=(host, args.camera_fps, frames, stop), daemon=True
        ).start()
    tof = TofClient(host, TOF_PORT).start()

    t0 = time.monotonic()
    scan_rate = LatencyTracker("scan period")
    last_scan_t = None
    next_vitals = t0
    try:
        while args.headless is None or time.monotonic() - t0 < args.headless:
            now = time.monotonic()
            rr.set_time("t", duration=now - t0)
            for scan in lidar.drain() if lidar is not None else []:
                # Revolution period from the scan stamps, not from how fast this
                # loop noticed them: a stalled loop must not read as a fast lidar.
                if last_scan_t is not None:
                    scan_rate.add(scan.stamp - last_scan_t)
                last_scan_t = scan.stamp
                pts = scan.points_xy(mount)
                rr.log("world/scan", rr.Points2D(pts, colors=[255, 200, 0], radii=0.012))
                rr.log("rates/lidar_rev_per_s", rr.Scalars(scan.speed_rps))
                rr.log("rates/scan_points", rr.Scalars(len(pts)))
            while not frames.empty():
                # JPEG passes through untouched: no decode on the laptop, small recordings.
                rr.log(
                    "camera/overview",
                    rr.EncodedImage(contents=frames.get_nowait(), media_type="image/jpeg"),
                )
            r = tof.ranges()
            for name in ("front", "left", "right"):
                value = getattr(r, name)
                if value is not None and r.age_s < 1.0:
                    rr.log(f"tof/{name}", rr.Scalars(value))
            if now >= next_vitals:
                next_vitals = now + VITALS_EVERY_S
                report = HealthReport()
                board = probe_board(host, report)
                v: BoardVitals = report.vitals
                if v.cpu_temp_c is not None:
                    rr.log("rates/cpu_temp_c", rr.Scalars(v.cpu_temp_c))
                if v.mem_free_mb is not None:
                    rr.log("rates/mem_free_mb", rr.Scalars(v.mem_free_mb))
                rr.log(
                    "log",
                    rr.TextLog(
                        f"board {'ok' if board.ok else 'DOWN'}: {board.detail} | "
                        f"tof age {r.age_s:.1f}s | "
                        f"scan period median {scan_rate.summary().median_ms:.0f} ms | "
                        "battery: no sensor | heading: no IMU",
                        level=rr.TextLogLevel.INFO if board.ok else rr.TextLogLevel.ERROR,
                    ),
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        tof.close()


if __name__ == "__main__":
    main()
