"""ament_python packaging for pepin_bringup: two console scripts, one resource marker."""

from glob import glob

from setuptools import find_packages, setup

package_name = "pepin_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Artem Belousov",
    maintainer_email="gartod@gmail.com",
    description="ROS 2 bridges from Pepin's board servers (base, ToF) to ROS topics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "base_bridge = pepin_bringup.base_bridge:main",
            "tof_bridge = pepin_bringup.tof_bridge:main",
            "relocalizer = pepin_bringup.relocalizer:main",
            "goal_server = pepin_bringup.goal_server:main",
            "run_recorder = pepin_bringup.run_recorder:main",
            "slam_frame = pepin_bringup.slam_frame:main",
        ],
    },
)
