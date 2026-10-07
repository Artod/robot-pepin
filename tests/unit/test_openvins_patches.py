"""ros/Dockerfile.vio's OpenVINS patches: each one copied and applied, in order, by default."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = (REPO / "ros" / "Dockerfile.vio").read_text()


def test_patches_are_copied_and_applied_in_order() -> None:
    order = ["openvins-executor.patch", "openvins-reset.patch", "openvins-imu-queue.patch"]
    applied = [DOCKERFILE.index(f"git apply /opt/openvins/patches/{name}") for name in order]
    assert applied == sorted(applied)
    for name in order:
        assert (REPO / "ros" / "patches" / name).is_file()
        assert f"COPY patches/{name} /opt/openvins/patches/{name}" in DOCKERFILE
    assert "ARG IMU_QUEUE=1" in DOCKERFILE and 'echo "imu queue patch: $IMU_QUEUE"' in DOCKERFILE


def test_imu_queue_patch_replaces_the_5_deep_queue() -> None:
    patch = (REPO / "ros" / "patches" / "openvins-imu-queue.patch").read_text()
    assert "(topic_imu, rclcpp::SensorDataQoS(),\n" in patch
    assert 'get_parameter_or<int>("imu_queue_depth", imu_queue_depth, 200)' in patch
    assert "rclcpp::SensorDataQoS().keep_last(imu_queue_depth)" in patch
