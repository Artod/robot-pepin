# syntax=docker/dockerfile:1
# ROS 2 Jazzy + Nav2 for the Orange Pi Zero 3 (arm64, Armbian Debian trixie host -> Ubuntu 24.04 inside).
# Build on the board:  docker build -t pepin-ros ros/     Run: ros/run.sh
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-nav2-bringup \
        ros-jazzy-slam-toolbox \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-tf2-tools \
        ros-jazzy-rqt-graph \
        python3-colcon-common-extensions \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Scan filtering: the lidar sees the cart's own rear posts; a box filter over the hull removes
# them once for every consumer (costmaps clear the footprint anyway, the collision monitor does not).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-laser-filters ros-jazzy-teleop-twist-keyboard \
    && rm -rf /var/lib/apt/lists/*

# LD19 lidar driver (maintained fork; the vendor's is unmaintained), built from source.
WORKDIR /ws
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/Myzhar/ldrobot-lidar-ros2.git src/ldrobot-lidar-ros2 \
    && . /opt/ros/jazzy/setup.sh \
    && apt-get update && apt-get install -y --no-install-recommends libudev-dev \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths src --ignore-src -y --rosdistro jazzy \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    # the driver builds the vendor SDK as libldlidar.so but never installs it: put it beside the component
    && cp build/ldlidar_component/libldlidar.so install/ldlidar_component/lib/ \
    && rm -rf /var/lib/apt/lists/* build log

# Our bridge package (base server <-> /cmd_vel, /odom, TF; ToF -> Range).
COPY pepin_bringup /ws/src/pepin_bringup
RUN . /opt/ros/jazzy/setup.sh && colcon build --packages-select pepin_bringup \
    && rm -rf build log

# Operator tools run inside the container (goal client, snapshots, latency probe).
COPY tools /tools
# Our pure-numpy localisation library (the whole-map search the relocalizer uses); ros/build.sh
# copies src/pepin here before building.
COPY pepin_src /ws/pepin_src
ENV PYTHONPATH=/ws/pepin_src

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV ROS_DOMAIN_ID=7
COPY entrypoint.sh /pepin_entrypoint.sh
RUN chmod +x /pepin_entrypoint.sh
ENTRYPOINT ["/pepin_entrypoint.sh"]
CMD ["bash"]
