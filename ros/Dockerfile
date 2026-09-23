# syntax=docker/dockerfile:1
# ROS 2 Jazzy + Nav2 for the Orange Pi Zero 3 (arm64, Armbian Debian trixie host -> Ubuntu 24.04 inside).
# Build on the board:  docker build -t pepin-ros ros/     Run: ros/run.sh
# Pinned by digest: a moved tag rebuilds every layer (an hour on this board) without warning.
# Refresh on purpose: ssh root@10.0.0.187 docker image inspect ros:jazzy-ros-base --format '{{index .RepoDigests 0}}'
FROM ros:jazzy-ros-base@sha256:2589a8fba5257307857890173c069852c2abf913a0be7970f172478baecb09e4

ENV DEBIAN_FRONTEND=noninteractive
# ros2bag + the MCAP storage plugin are here on purpose, named rather than assumed: they are how a
# drive is recorded under PEPIN_RECORDER=bag (pepin_bringup.bag_recorder starts `ros2 bag record`,
# and the board then copies serialised bytes instead of deserialising every scan), and how the
# laptop image — which is built FROM this one — reads that bag back into a tape
# (ros/tools/bag_to_tape.py needs rosbag2_py). ros_base carries rosbag2 itself; the CLI verb and
# the storage plugin are separate packages and an image without them fails at the first goal.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-navigation2 \
        ros-jazzy-nav2-bringup \
        ros-jazzy-slam-toolbox \
        ros-jazzy-foxglove-bridge \
        ros-jazzy-rmw-cyclonedds-cpp \
        ros-jazzy-rmw-zenoh-cpp \
        ros-jazzy-tf2-tools \
        ros-jazzy-rqt-graph \
        python3-colcon-common-extensions \
        ros-jazzy-ros2bag \
        ros-jazzy-rosbag2-storage-mcap \
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

# Laser odometry (RF2O): the cart's planar motion from consecutive scans, scan-to-scan and with
# no map, for the EKF. There is NO apt package — `ros-jazzy-rf2o-laser-odometry` does not exist
# in the Jazzy arm64 index (checked 2026-09-21 against packages.ros.org's noble/arm64 Packages
# list: 8704 packages, no rf2o and no laser_scan_matcher either), so the MAPIRlab ROS 2 port is
# built from source at a pinned commit. Pinned and not `-b ros2`: a moved branch would silently
# rebuild this board for an hour with code nobody read. rosdep is not used on it — its
# package.xml still lists ROS 1's `cmake_modules`, which has no Jazzy key — so its three real
# system dependencies are installed by name; CMakeLists.txt asks for nothing else.
#
# The patch is three upstream defects, each of which would have cost a drive (ros/patches):
#   - twist.linear.x is the LASER's own x step per second, which is the cart's forward speed
#     only for a laser mounted axis-aligned with base_link; ours is yawed -87.5 deg and rolled
#     180. Behind `base_frame_twist`, the node differences the robot pose it already publishes.
#   - twist.covariance is left empty, and robot_localization turns a zero variance into 1e-9 and
#     then believes this source over the wheels AND the gyro. The diagonal is three parameters.
#   - two RCLCPP_INFO per scan is 20 lines a second in the one log every restart check reads.
# All three default to upstream's behaviour; ros/pepin_bringup/launch/robot.launch.py turns them on.
# ONE compiler at a time BY DEFAULT: a -j4 build of this package (Eigen-heavy translation units)
# put the 1.5 GB board into swap and off the network on 2026-09-22; the build stops the stack
# first anyway. RF2O_JOBS is that limit, and it is a build argument because the limit belongs to
# the machine and not to the package: a cross-build on the laptop (ros/build-image.sh) passes the
# core count and compiles this layer in a fraction of the time. Leave the default alone — it is
# what an on-board `docker build` (ros/build.sh) must get without being told.
ARG RF2O_COMMIT=b38c68e46387b98845ecbfeb6660292f967a00d3
ARG RF2O_JOBS=1
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-eigen3-cmake-module libeigen3-dev libboost-dev patch \
    && rm -rf /var/lib/apt/lists/*
COPY patches/rf2o-base-twist.patch /tmp/rf2o-base-twist.patch
RUN git clone https://github.com/MAPIRlab/rf2o_laser_odometry.git src/rf2o_laser_odometry \
    && git -C src/rf2o_laser_odometry checkout --detach "$RF2O_COMMIT" \
    && patch -p1 -d src/rf2o_laser_odometry < /tmp/rf2o-base-twist.patch \
    && . /opt/ros/jazzy/setup.sh \
    && MAKEFLAGS=-j${RF2O_JOBS} colcon build --packages-select rf2o_laser_odometry \
        --executor sequential --parallel-workers 1 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

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
# What `docker stop` sends. SIGINT is the signal ros2 launch answers by shutting its nodes down;
# the default SIGTERM it answers by cancelling itself, and the nodes are SIGKILLed without ever
# closing what they were writing (RTAB-Map's database: ros/maps/rtabmap.db, malformed after the
# kills of 2026-09-13). The docker run lines pass --stop-signal SIGINT too; this makes a
# container started without that flag gentle anyway.
STOPSIGNAL SIGINT
ENTRYPOINT ["/pepin_entrypoint.sh"]
CMD ["bash"]
