"""The contracts around the place descriptors and the model services that live in launch files and
shell scripts: the census taken by the launch on every start, the rehearsal threshold that follows
the descriptors, the patch marker's name, the patched core's features, the XFeat pin, and the
model jobs that load on demand. Read from the sources, as the robot's launch and scripts are."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import source_facts as sf

from pepin.global_descriptor import CENSUS_ENV, KEEPS_DESCRIPTORS_MARKER

REPO = Path(__file__).resolve().parents[2]
VSLAM_LAUNCH = "ros/pepin_bringup/launch/vslam.launch.py"


def _table(name: str) -> dict[str, object]:
    return dict(ast.literal_eval(sf.assignments(sf.tree(VSLAM_LAUNCH))[name]))


# ---- the launch -------------------------------------------------------------------------------
def test_the_rehearsal_moves_to_the_descriptor_s_scale_only_with_descriptors_on_board() -> None:
    """Measured (scratch/models/rehearsal_calibration.py): of 121 consecutive daylight camera
    pairs the words at 0.30 called 11 the same place; the descriptor scores them 0.57-0.98 and
    reaches that share at 0.925. A null pair (0.5) must never merge."""
    assert _table("RTABMAP")["Mem/RehearsalSimilarity"] == "0.30", "the words' threshold"
    overlay = _table("DESCRIPTOR_REHEARSAL")
    assert overlay == {"Mem/RehearsalSimilarity": "0.92"}
    assert float(str(overlay["Mem/RehearsalSimilarity"])) > 0.5, "a null pair never merges"
    source = (REPO / VSLAM_LAUNCH).read_text()
    assert "table.update(DESCRIPTOR_REHEARSAL)" in source
    assert "place_descriptors(packing)" in source, "decided as sensor_pack decides it"


def test_the_census_is_taken_by_the_launch_on_every_start_of_the_file_rtabmap_is_given() -> None:
    """A census baked into the container at creation outlived a restart after the database was
    swapped (the backfill's backup restored): descriptor mode on nodes without descriptors, and
    RTAB-Map aborts at the first likelihood."""
    module = sf.tree(VSLAM_LAUNCH)
    frames = [
        call
        for call in sf.calls_to(module, "ExecuteProcess")
        if "pepin_bringup.rtabmap_frame" in ast.unparse(call)
    ]
    assert len(frames) == 1
    assert ast.unparse(sf.keywords(frames[0])["additional_env"]) == "census"
    census_calls = sf.calls_to(module, "census_env")
    assert [ast.unparse(c.args[0]) for c in census_calls] == ["database"]
    assert CENSUS_ENV == "PEPIN_PLACE_CENSUS"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "PEPIN_PLACE_CENSUS" not in laptop, "no census baked in at container creation"


def test_laptop_sh_passes_the_start_flags_through_and_brings_the_models_up_and_down() -> None:
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "for var in PEPIN_GLOBAL_DESCRIPTOR PEPIN_REGISTRATION_BACKEND; do" in laptop
    assert '${FLAG_ENV[@]+"${FLAG_ENV[@]}"}' in laptop
    assert '"$HERE/models.sh" start localization' in laptop
    assert '"$HERE/models.sh" stop localization' in laptop


# ---- the patched core -------------------------------------------------------------------------
def test_the_marker_the_nodes_ask_for_is_the_one_the_image_build_leaves() -> None:
    """patch_rtabmap.sh touches /opt/rtabmap_patches/<the patch's file name without rtabmap->;
    sensor_pack, rtabmap_frame and the launch ask for KEEPS_DESCRIPTORS_MARKER."""
    script = (REPO / "ros/xfeat/patch_rtabmap.sh").read_text()
    assert 'name="$(basename "$patch" .patch)"' in script
    assert 'touch "/opt/rtabmap_patches/${name#rtabmap-}"' in script
    patches = sorted(p.name for p in (REPO / "ros/patches").glob("rtabmap-*.patch"))
    markers = {
        f"/opt/rtabmap_patches/{n.removesuffix('.patch').removeprefix('rtabmap-')}" for n in patches
    }
    assert KEEPS_DESCRIPTORS_MARKER in markers, (patches, KEEPS_DESCRIPTORS_MARKER)


def _defines_identical(tmp_path: Path, installed: list[str], built: list[str]) -> tuple[int, str]:
    """ros/xfeat/patch_rtabmap.sh's defines_identical on two Version.h excerpts."""
    (tmp_path / "installed").write_text("".join(f"#define RTABMAP_{d}\n" for d in installed))
    (tmp_path / "built").write_text("".join(f"#define RTABMAP_{d}\n" for d in built))
    script = REPO / "ros" / "xfeat" / "patch_rtabmap.sh"
    run = subprocess.run(
        [
            "bash",
            "-c",
            f"eval \"$(sed -n '/^defines_identical() {{/,/^}}/p' '{script}')\";"
            f" defines_identical '{tmp_path / 'installed'}' '{tmp_path / 'built'}'",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return run.returncode, run.stderr


def test_the_patched_core_is_swapped_in_only_with_the_installed_core_s_features(
    tmp_path: Path,
) -> None:
    """The wrappers were built against the installed configuration: a patched library that lost
    GTSAM (Optimizer/Strategy's compile-time default) or gained a feature must not replace it."""
    kept = ["TORO", "G2O", "GTSAM", "POINTMATCHER", "OCTOMAP", "PYTHON"]
    assert _defines_identical(tmp_path, kept, list(reversed(kept))) == (0, "")
    status, why = _defines_identical(tmp_path, kept, [d for d in kept if d != "GTSAM"])
    assert status == 1 and "< #define RTABMAP_GTSAM" in why
    status, why = _defines_identical(tmp_path, kept, [*kept, "CUDASIFT"])
    assert status == 1 and "> #define RTABMAP_CUDASIFT" in why
    script = (REPO / "ros/xfeat/patch_rtabmap.sh").read_text()
    check = script.index('defines_identical "$PREFIX/include/rtabmap-0.22/rtabmap/core/Version.h"')
    assert check < script.index('cp "$BUILT"'), "checked before the swap"


# ---- the model jobs ---------------------------------------------------------------------------
def _models_sh(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO / "ros/models.sh"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, **env},
    )


def test_the_model_jobs_load_on_demand_and_never_at_login(tmp_path: Path) -> None:
    """RunAtLoad with a plist in ~/Library/LaunchAgents kept ~2 GB of models resident from every
    login. The plists live elsewhere; a job runs from its start to its stop."""
    run = _models_sh("plist", "depth", PEPIN_LAUNCHD_DIR=str(tmp_path))
    assert run.returncode == 0, run.stderr
    plist = run.stdout
    assert "<key>ThrottleInterval</key><integer>30</integer>" in plist
    assert "<key>ProcessType</key><string>Standard</string>" in plist
    script = (REPO / "ros/models.sh").read_text()
    assert "Library/LaunchAgents" not in script.split("JOBS_DIR=", 1)[1].split("\n", 1)[0]
    assert 'JOBS_DIR="${PEPIN_LAUNCHD_DIR:-$HOME/Library/Application Support/pepin/launchd}"' in (
        script
    )
    assert _models_sh("installed", "depth", PEPIN_LAUNCHD_DIR=str(tmp_path)).returncode == 1
    (tmp_path / "com.pepin.models.depth.plist").write_text(plist)
    assert _models_sh("installed", "depth", PEPIN_LAUNCHD_DIR=str(tmp_path)).returncode == 0


def test_an_xfeat_checkout_at_another_commit_than_the_image_pins_is_not_taken(
    tmp_path: Path,
) -> None:
    """auto registration mixes the service's XFeat with the image's local fallback: the host
    must run the commit ros/Dockerfile.xfeat pins."""
    checkout = tmp_path / "xfeat"
    (checkout / "weights").mkdir(parents=True)
    for name in ("xfeat.pt", "xfeat-lighterglue.pt"):
        (checkout / "weights" / name).write_bytes(b"w")
    (checkout / "COMMIT").write_text("deadbeef\n")
    run = _models_sh("plist", "localization", PEPIN_XFEAT_DIR=str(checkout))
    assert f"{checkout} is XFeat deadbeef, the image pins" in run.stderr
    assert f"<string>{checkout}</string>" not in run.stdout
    dockerfile = (REPO / "ros/Dockerfile.xfeat").read_text()
    sha = next(
        line.split("=", 1)[1]
        for line in dockerfile.splitlines()
        if line.startswith("ARG XFEAT_SHA=")
    )
    (checkout / "COMMIT").write_text(f"{sha}\n")
    run = _models_sh("plist", "localization", PEPIN_XFEAT_DIR=str(checkout))
    assert run.returncode == 0 and f"<string>{checkout}</string>" in run.stdout, run.stderr
    (checkout / "COMMIT").write_text("deadbeef\n")
    run = _models_sh(
        "plist", "localization", PEPIN_XFEAT_DIR=str(checkout), PEPIN_XFEAT_UNPINNED="1"
    )
    assert run.returncode == 0 and "taken (PEPIN_XFEAT_UNPINNED=1)" in run.stderr


def test_depth_host_sh_hands_an_installed_host_to_models_sh() -> None:
    script = (REPO / "ros/depth_host.sh").read_text()
    assert 'launchd_owned() { "$HERE/models.sh" installed depth; }' in script
    assert 'exec "$HERE/models.sh" start depth' in script
    assert 'exec "$HERE/models.sh" stop depth' in script
