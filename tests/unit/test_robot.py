"""The robot's static configuration (pepin.robot.RobotConfig)."""

import json

from pepin.robot import CONFIG_DIR, RobotConfig


def test_config_loads_from_the_repo_and_feeds_can_be_switched_off() -> None:
    cfg = RobotConfig.load(CONFIG_DIR)
    assert cfg.enabled("lidar") and cfg.enabled("tof")
    assert cfg.port("base", 0) == 3336 and cfg.port("unknown", 7) == 7
    assert set(cfg.tof_mounts) == {"front", "left", "right"}
    quiet = cfg.without("tof")
    assert not quiet.enabled("tof") and quiet.enabled("lidar")
    assert cfg.enabled("tof")  # the original is untouched


def test_config_rejects_misspelt_feeds_and_defaults_missing_ones_to_on(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import shutil

    import pytest

    for name in ("base.json", "lidar.json", "tof.json"):
        shutil.copy(CONFIG_DIR / name, tmp_path / name)
    (tmp_path / "robot.json").write_text(json.dumps({"feeds": {"lidarr": {"enabled": False}}}))
    with pytest.raises(ValueError, match="lidarr"):
        RobotConfig.load(tmp_path)
    (tmp_path / "robot.json").write_text(json.dumps({"feeds": {"tof": {"enabled": False}}}))
    cfg = RobotConfig.load(tmp_path)
    assert cfg.enabled("lidar") and not cfg.enabled("tof") and cfg.enabled("camera")
