# syntax=docker/dockerfile:1
# ROS 2 Jazzy + Nav2 for the Orange Pi Zero 3 (arm64, Armbian Debian trixie host -> Ubuntu 24.04 inside).
# Build on the board:  docker build -t pepin-ros ros/     Run: ros/run.sh
# Pinned by digest: a moved tag rebuilds every layer (an hour on this board) without warning.
# Refresh on purpose: ssh root@10.0.0.187 docker image inspect ros:jazzy-ros-base --format '{{index .RepoDigests 0}}'
FROM ros:jazzy-ros-base@sha256:2589a8fba5257307857890173c069852c2abf913a0be7970f172478baecb09e4

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

# ROS 2 Jazzy's binaries link every node against LTTng-UST through tracetools, and lttng-ust
# 2.13 takes 128 MB of resident memory per process the moment it is loaded, tracing or not.
# Nine ROS processes on a 1.5 GB board put 1.1 GB of untouched tracing buffers into swap and
# the board froze under a spin (2026-09-10). tracetools rebuilt without LTTng replaces the
# binary one, in the overlay and in /opt/ros/jazzy/lib both; nothing here traces. (Tracepoints
# excluded, not "disabled": disabled makes the package header-only and the binaries need the
# library with its ros_trace_* symbols, empty.)
RUN git clone --depth 1 -b jazzy https://github.com/ros2/ros2_tracing.git src/ros2_tracing \
    && . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select tracetools \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DTRACETOOLS_TRACEPOINTS_EXCLUDED=ON \
            -DTRACETOOLS_STATUS_CHECKING_TOOL=OFF \
    && cp -f install/tracetools/lib/libtracetools.so /opt/ros/jazzy/lib/libtracetools.so \
    && rm -rf build log src/ros2_tracing

# Our bridge packages: the Python one (base server <-> /cmd_vel, /odom, TF; ToF -> Range) and
# the C++ base bridge, same protocol at a tenth of the RAM (robot.launch.py picks one).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        nlohmann-json3-dev \
        ros-jazzy-robot-localization \
    && rm -rf /var/lib/apt/lists/*
COPY pepin_bringup /ws/src/pepin_bringup
COPY pepin_base_cpp /ws/src/pepin_base_cpp
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --packages-select pepin_bringup pepin_base_cpp \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
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
