"""ros/tools/install_rmw_zenoh_fix.sh, the image builds' step that puts the lost-wake-up
librmw_zenoh_cpp.so over the apt one: only over 0.2.10, never a 0.2.10 image without it, the apt
one kept beside it; and the patch it ships is upstream's #1040 hunk (no unlocked trigger reset)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ros/tools/install_rmw_zenoh_fix.sh"


def _run(
    tmp_path: Path, version: str, with_so: bool
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """The script with a fake dpkg-query answering ``version`` and a fake /opt/ros/jazzy/lib."""
    bin_dir, lib, fix = tmp_path / "bin", tmp_path / "lib", tmp_path / "fix"
    for d in (bin_dir, lib, fix):
        d.mkdir()
    (bin_dir / "dpkg-query").write_text(f"#!/bin/sh\nprintf '%s' '{version}'\n")
    (bin_dir / "dpkg-query").chmod(0o755)
    (lib / "librmw_zenoh_cpp.so").write_bytes(b"apt")
    if with_so:
        (fix / "librmw_zenoh_cpp.so").write_bytes(b"fixed")
        (fix / "BUILD.txt").write_text("built\n")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", ROS_LIB=str(lib))
    env["FIX_INFO"] = str(tmp_path / "info")
    done = subprocess.run(["sh", str(SCRIPT), str(fix)], env=env, capture_output=True, text=True)
    return done, lib


def test_over_0_2_10_the_fixed_so_replaces_the_apt_one_which_is_kept(tmp_path: Path) -> None:
    done, lib = _run(tmp_path, "0.2.10-1noble.20260902.013751", with_so=True)
    assert done.returncode == 0, done.stderr
    assert (lib / "librmw_zenoh_cpp.so").read_bytes() == b"fixed"
    assert (lib / "librmw_zenoh_cpp.so.apt").read_bytes() == b"apt"
    assert (tmp_path / "info/BUILD.txt").read_text() == "built\n"


def test_a_0_2_10_image_without_the_fixed_so_is_refused(tmp_path: Path) -> None:
    done, lib = _run(tmp_path, "0.2.10-1noble.20260902.013751", with_so=False)
    assert done.returncode == 1 and "build_rmw_zenoh_fix.sh" in done.stderr
    assert (lib / "librmw_zenoh_cpp.so").read_bytes() == b"apt"


def test_from_0_2_11_on_the_apt_so_stays(tmp_path: Path) -> None:
    done, lib = _run(tmp_path, "0.2.11-1noble.20261001.000000", with_so=True)
    assert done.returncode == 0 and "upstream" in done.stdout
    assert (lib / "librmw_zenoh_cpp.so").read_bytes() == b"apt"
    assert not (lib / "librmw_zenoh_cpp.so.apt").exists()


def test_the_patch_is_upstreams_hunk_and_every_image_installs_it() -> None:
    patch = (REPO / "ros/patches/rmw_zenoh-lost-wakeup.patch").read_text()
    assert "-  wait_set_data->triggered = false;" in patch
    assert "must never write" in patch and "#1036" in patch and "0.2.11" in patch
    for dockerfile in ("Dockerfile", "Dockerfile.laptop", "Dockerfile.xfeat", "Dockerfile.vio"):
        text = (REPO / "ros" / dockerfile).read_text()
        assert "COPY build/rmw_zenoh_fix/ tools/install_rmw_zenoh_fix.sh" in text, dockerfile
        assert "sh /tmp/rmw_zenoh_fix/install_rmw_zenoh_fix.sh /tmp/rmw_zenoh_fix" in text, (
            dockerfile
        )
