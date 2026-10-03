// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// ROS 2 node: the board's base server as /odom, odom->base_link, and a /cmd_vel sink.
// Ported from the Python bridge (git history before 2026-10-02), same parameters, same wire
// protocol: on the board's four A53 cores an rclpy process costs ~190 MB and a tenth of a
// core, this one ~25 MB and ~1%. Its live switches are described in pepin_bringup/base_bridge.py.
//
// The board owns the wheels in real time behind a 0.5 s deadman: it stops them the
// moment commands stop arriving. Nav2 publishes a twist only when it feels like it and
// expects the last one to persist, so this node does two things with one command —
// forward it the instant it arrives (latency), and re-send it at `resend_hz` (the board
// stays fed while the plan is steady). When /cmd_vel goes quiet for `cmd_timeout_s` we
// send one stop and shut up; the deadman is the real safety net, this is the polite
// version that does not rely on it.
//
// Nothing here can kill the node: the socket lives in JsonLineLink on its own thread and
// reconnects forever. State lines are published straight from that reader thread (rclcpp
// publishers are thread-safe): no queue, no drain timer, no CPU spent polling.
//
// /odom is DATED BY ITS ENCODER READ, not by its arrival here: the line's `t` carried onto the
// ROS clock exactly as the neck's (line_time()), so /odom, odom -> base_link and /neck/state of
// one line share one stamp. `odom_stamp` "arrival" is the old stamp, live (2026-10-02: arrival
// ran p50 6.4, p99 8.9, max 26.4 ms behind the read at rest).
//
// /odom's TWIST is measured, not commanded: the state line's v and w are the twist the base
// server was ASKED for, and publishing those puts Nav2's own output where the EKF reads a
// sensor. See odom_twist() below and `odom_twist_source`.
//
// The same process optionally reads an MPU6050 on the board's I2C bus and publishes
// imu/data_raw (a third thread, same reasoning: a blocking bus must not sit in the
// executor). Wheel odometry over-reports a turn in place by 10-25% on carpet; the fix is
// a gyro yaw rate fused with the wheels in an EKF, configured outside this node. The IMU
// is optional in the strong sense: nothing about it can stop the wheels from working.
//
// /imu/data_raw is DATED BY THE CHIP'S SAMPLE: the moment the 14-byte burst came back, less the
// chip's low-pass group delay (`imu_filter_delay_s`, 4.8 ms at DLPF_CFG 3: mpu6050.hpp has the
// table), and the chip refreshes those registers at 1 kHz (`imu_output_rate_hz`), so the sample
// read is at most 1 ms old where at the 100 Hz it ran until 2026-10-02 it was up to 10 ms.
//
// The gyro's ZERO is re-measured for as long as the node lives, from the rest the WHEELS witness:
// the chip's bias moves with temperature, and a zero taken once at boot turned RTAB-Map's map
// +27 deg in 40 min under a parked cart. See gyro_bias.hpp, witness_rest() and read_imu().
//
// And while the cart CERTAINLY stands still — the wheels' rest past the same settle window, no
// fresh command, the gyro quiet — the node tells the EKF so: a zero twist on /zupt, the filter's
// odom2, which otherwise hears nothing at rest but its sources' own drift (rf2o +1.5 deg/min on a
// parked cart, 2026-09-24). See zupt.hpp and publish_zupt().
//
// THE NECK rides the same line. The base server reads the two neck servos in the same sync_read
// as the wheels and puts their ticks in the state line under the same stamp, so this node turns
// every line that carries them into /neck/state (sensor_msgs/JointState: neck_pan positive left,
// head_tilt the pitch below level) and base_link -> camera_link (neck.hpp, the twin of
// pepin.neck), both stamped with the moment of THAT encoder read — the line's `t`, the board's
// monotonic clock, carried onto the ROS clock — at up to `neck_publish_hz` (every 50 Hz line by
// default). The geometry is parameters only (robot.launch.py reads config/neck.json and
// config/camera.json): a retuned mount is a restart, never a rebuild. With no `neck_camera_frame`
// no transform is published; a line without the ticks (a silent neck) publishes nothing, and
// a stale edge is never republished. This replaced pepin_bringup.neck_state (a Python process
// polling the base server at 2-20 Hz, git history before 2026-10-02).

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "pepin_base_cpp/gyro_bias.hpp"
#include "pepin_base_cpp/link.hpp"
#include "pepin_base_cpp/mpu6050.hpp"
#include "pepin_base_cpp/neck.hpp"
#include "pepin_base_cpp/protocol.hpp"
#include "pepin_base_cpp/twist_from_pose.hpp"
#include "pepin_base_cpp/zupt.hpp"

namespace pepin
{

constexpr double kStatusHz = 2.0;  // how often link up/down transitions are logged

// What the EKF is told to believe about the MPU6050. Both are honest over-estimates: the
// gyro's own noise is far below 0.02 rad/s, but the chip rides an unisolated chassis.
constexpr double kGyroStdDev = 0.02;   // rad/s
constexpr double kAccelStdDev = 0.5;   // m/s^2
constexpr double kRadToDeg = 57.29577951308232;

// TwistFromPose's own max_gap_s, named here because a second reader needs the same number: a
// state stream with a gap this long is not a measurement, so the estimator re-primes AND the rest
// the wheels were witnessing is over (witness_rest, still_witness). One number, two users.
constexpr double kStateGapMaxS = 1.0;

// The wheels' word crosses to the IMU thread through plain atomic doubles, so the 50 Hz loop
// never waits on the reader thread's lock. On anything ROS 2 runs on these are single
// instructions; the assert is here so a port that changes that fails to build instead of
// silently taking a mutex inside std::atomic.
static_assert(std::atomic<double>::is_always_lock_free, "the IMU loop must not lock");

/// Bridges the base server to ROS: /odom and odom->base_link out, /cmd_vel down to wheels, and
/// the neck's encoders as /neck/state and base_link -> camera_link.
class BaseBridge : public rclcpp::Node
{
public:
  /// Declare parameters, open the link to the base server, and start publishing.
  explicit BaseBridge(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("base_bridge", options)
  {
    const auto host = declare_parameter<std::string>("host", "127.0.0.1");
    const auto port = static_cast<int>(declare_parameter<int>("port", 3336));
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    cmd_timeout_s_ = declare_parameter<double>("cmd_timeout_s", 0.5);
    const double resend_hz = declare_parameter<double>("resend_hz", 5.0);  // deadman is 0.5 s
    // Hard ceiling for whatever arrives on /cmd_vel — teleop's q key ran the cart at 0.6 m/s
    // and slam_toolbox lost the map; the planner's limits live in nav2_params.yaml.
    max_linear_ = declare_parameter<double>("max_linear_m_s", 0.25);
    max_angular_ = declare_parameter<double>("max_angular_rad_s", 0.6);
    // The IMU shares /dev/i2c-2 with the ToF sensors; 0x68 is the MPU6050's own address.
    // False when an EKF (robot_localization) owns odom -> base_link; /odom is still published.
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    // MUTING A SENSOR LIVE. Both default true and both are read per message, so `ros2 param
    // set` (ros/sensor.sh mute imu | mute odom) silences a sensor without restarting anything:
    // the chip is still read, the base server is still asked, only the message stops. What a
    // consumer sees is what a dead sensor looks like — silence — which is the point: the EKF's
    // sensor_timeout, Nav2's TF lookups and the tracker's fallbacks are exercised in place
    // (ros/README.md, "Muting a sensor live"). `odom_publish` takes the odom -> base_link
    // transform with it when `publish_tf` is on: wheel odometry that keeps broadcasting a
    // transform while /odom is silent is a state no sensor failure produces.
    declare_parameter<bool>("imu_publish", true);
    declare_parameter<bool>("odom_publish", true);
    const bool imu_enable = declare_parameter<bool>("imu_enable", false);
    imu_device_ = declare_parameter<std::string>("imu_device", "/dev/i2c-2");
    imu_address_ = static_cast<int>(declare_parameter<int>("imu_address", 0x68));
    imu_rate_hz_ = declare_parameter<double>("imu_rate_hz", 50.0);
    // The chip's own output rate and its filter's delay (config/imu.json's timing block, passed
    // by robot.launch.py): how fresh the sample we read is, and how far before the read the
    // motion it describes happened. A delay of 0 stamps the read itself.
    imu_output_rate_hz_ = declare_parameter<double>("imu_output_rate_hz", 1000.0);
    imu_filter_delay_s_ = declare_parameter<double>("imu_filter_delay_s", 0.0);
    // Published in the robot's own frame: the mounting rotation is applied here (see
    // to_base_axes), not left to a static transform the filter may or may not apply.
    imu_frame_ = declare_parameter<std::string>("imu_frame", "base_link");
    const auto up = declare_parameter<std::string>("imu_up_axis", "y");
    imu_up_axis_ = up.empty() ? 'z' : static_cast<char>(std::tolower(up[0]));
    imu_bias_s_ = declare_parameter<double>("imu_bias_s", 2.0);
    // KEEPING THE GYRO'S ZERO HONEST (CLAUDE.md rule 19; gyro_bias.hpp has the measurements).
    // On, the bias is re-measured from every block of rest the wheels witness, for as long as the
    // node lives. Off is what this node shipped with: one block over the first `imu_bias_s` after
    // start, subtracted forever — and the chip's bias moves with temperature, so hours later the
    // EKF's yaw crept +0.19, -0.54 and +0.67 deg/min in one night on a cart whose wheels read
    // 0.00, and RTAB-Map's map turned +27 deg in 40 min under it (2026-09-19). Read per sample,
    // so `ros2 param set /base_bridge imu_bias_tracking false` compares the two without a restart
    // (it takes effect at the next block: switching mid-block abandons the block in progress).
    imu_bias_tracking_ = declare_parameter<bool>("imu_bias_tracking", true);
    // WHAT /odom's TWIST MEANS. The base server's state line reports v and w as the twist it
    // was COMMANDED to apply -- snapshot() copies self.twist, which is whatever /cmd_vel last
    // asked for (src/pepin/base_server.py:466) -- while its x/y/theta are integrated from the
    // wheel travel and ARE a measurement. Publishing the command as the twist puts a
    // controller's own output where a filter reads a sensor: robot_localization fuses odom0's
    // vx today (ros/params/ekf.yaml), so until 2026-09-15 the EKF was told the cart is doing
    // exactly what it was asked to do. "measured" (the default) differences two consecutive
    // wheel poses instead (TwistFromPose, the twin of pepin.odometry.TwistFromPose);
    // "commanded" is the old behaviour, one parameter away and no restart:
    // ``ros2 param set /base_bridge odom_twist_source commanded``.
    const auto twist_source = declare_parameter<std::string>("odom_twist_source", "measured");
    twist_measured_ = twist_source != "commanded";
    // WHEN /odom HAPPENED. "encoder" (the default) dates /odom and odom -> base_link with the
    // state line's encoder read, the neck's stamp (line_time); "arrival" is what this node did
    // before 2026-10-02: now() when the line reached the reader thread, p50 6.4 ms (p99 8.9, max
    // 26.4) after the read on a parked cart (scratch/gaze/odom_vs_neck_stamp.py). Read per line,
    // so ``ros2 param set /base_bridge odom_stamp arrival`` switches back live; any other word is
    // "encoder". A line older than `neck_stamp_max_age_s` is dated on arrival either way.
    const auto odom_stamp = declare_parameter<std::string>("odom_stamp", "encoder");
    odom_stamp_encoder_ = odom_stamp != "arrival";
    // THE ZERO-VELOCITY UPDATE (CLAUDE.md rule 19; zupt.hpp has the measurements). On, /zupt
    // carries a twist of exactly zero -- ekf.yaml's odom2 fuses its vx, vy and vyaw -- for as long
    // as the cart CERTAINLY stands still: the wheels have witnessed rest for `zupt_settle_s`, no
    // non-zero /cmd_vel is younger than `zupt_cmd_hold_s`, and the bias-corrected gyro has stayed
    // under `zupt_gyro_quiet_rad_s`; nothing at all otherwise. Off is what this node did before
    // 2026-09-24: nothing on /zupt, and the parked EKF's heading followed its sources' drift at
    // ~5 deg/hour.
    // EVERY ONE OF ITS SETTINGS IS LIVE -- heading drift has many causes and each is tuned on the
    // robot, never in C++: `ros2 param set /base_bridge <name> <value>` is checked against the
    // setting's range (zupt.hpp's kZuptRanges; a refusal is logged and handed back to the caller,
    // the value in force stays), and an accepted value is in force at the next tick -- the rate
    // re-times the timer at once. A set-parameters callback stores each value in an atomic that
    // the timer and the 50 Hz IMU loop read (check_zupt_settings, apply_zupt_settings): no
    // parameter lookup per tick. The status line prints every one of them.
    zupt_publish_ = declare_parameter<bool>("zupt_publish", true);
    gyro_quiet_rad_s_ = declare_zupt_number(
      "zupt_gyro_quiet_rad_s", kGyroQuietRadS,
      "a bias-corrected yaw rate at or above this is a turn and stops the update like a wheel's "
      "move, rad/s (0.005: 7.9 sigma of the parked chip's noise, 2.1x its largest parked "
      "deviation; a slow hand turn is 31x over it)");
    zupt_hz_ = declare_zupt_number(
      "zupt_rate_hz", kZuptHz,
      "how often the update is published while the cart is at rest, Hz (10: the slip watch's; at "
      "10 it is phase-locked to rf2o's scans, zupt.hpp's kZuptHz)");
    zupt_var_linear_ = declare_zupt_number(
      "zupt_var_linear", kRestZuptVariance,
      "the variance the update claims on vx and vy, (m/s)^2 (1e-6: 1 mm/s)");
    zupt_var_yaw_ = declare_zupt_number(
      "zupt_var_yaw", kRestZuptVariance,
      "the variance the update claims on vyaw, (rad/s)^2 (1e-6: 1 mrad/s, 400x under the "
      "gyro's 4e-4)");
    // The two windows default to the numbers they were borrowed from, which stay exactly what they
    // were for their own users: imu_bias_s is still the bias tracker's settle window and block,
    // cmd_timeout_s still the age at which a command is dropped and one stop sent.
    zupt_settle_s_ = declare_zupt_number(
      "zupt_settle_s", zupt_clamped(*zupt_range("zupt_settle_s"), imu_bias_s_),
      "seconds of witnessed rest before the update, and the hold after a gyro turn (default: "
      "imu_bias_s, the window the gyro's bias tracker waits; that one is not moved by this)");
    zupt_cmd_hold_s_ = declare_zupt_number(
      "zupt_cmd_hold_s", zupt_clamped(*zupt_range("zupt_cmd_hold_s"), cmd_timeout_s_),
      "seconds a non-zero /cmd_vel holds the update off (default: cmd_timeout_s, the age up to "
      "which the bridge keeps re-sending a command)");

    declare_neck();

    pose_covariance_ = odometry_pose_covariance();
    twist_covariance_ = odometry_twist_covariance();

    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    // Five deep, RELIABLE: what the EKF subscribes with (odom2_queue_size), and what the slip
    // watch's publisher on the same topic uses.
    zupt_publisher_ = create_publisher<nav_msgs::msg::Odometry>("zupt", 5);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    twist_subscription_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      [this](const geometry_msgs::msg::Twist & message) {
        accept_command(message.linear.x, message.angular.z);
      });
    twist_stamped_subscription_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      "cmd_vel_stamped", 10,
      [this](const geometry_msgs::msg::TwistStamped & message) {
        // The stamp is not needed: the board acts on receipt.
        accept_command(message.twist.linear.x, message.twist.angular.z);
      });

    link_ = std::make_unique<JsonLineLink>(
      host, port,
      [this](const nlohmann::json & message) {on_state_line(message);},
      "base server");
    link_->start();
    status_timer_ = create_wall_timer(period(1.0 / kStatusHz), [this] {log_link_status();});
    resend_timer_ = create_wall_timer(period(1.0 / resend_hz), [this] {resend_command();});
    start_zupt_timer();
    // Registered last, after every parameter is declared: a declaration must not run them.
    zupt_check_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        return check_zupt_settings(parameters);
      });
    zupt_apply_ = add_post_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & parameters) {
        apply_zupt_settings(parameters);
      });

    if (imu_enable) {
      start_imu();
    }
    RCLCPP_INFO(
      get_logger(), "base_bridge: odom twist %s, publish_tf=%s, %s", twist_source.c_str(),
      publish_tf_ ? "on" : "off", switch_state().c_str());
  }

  /// Stop the wheels and join the threads before anything they publish through goes away.
  /// (Loaded as a component there is no main() to call shutdown(); the destructor is the exit.)
  ~BaseBridge() override {shutdown();}

  /// Stop the wheels, close the link and put the IMU down, on the way out.
  void shutdown()
  {
    if (shut_down_.exchange(true)) {
      return;
    }
    link_->send(encode_stop());
    link_->stop();
    stop_imu();
  }

private:
  /// Seconds as the wall-timer period.
  static std::chrono::nanoseconds period(double seconds)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(seconds));
  }

  /// Monotonic seconds: the one clock the IMU thread, the reader thread and the bias tracker
  /// share. Not the ROS clock on purpose — this measures durations on the board, and a clock
  /// that can be stepped or replayed has no business deciding how long a cart has stood still.
  static double monotonic_s()
  {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  /// Reader thread: publish a state line at once; anything else (a pong) is ignored.
  void on_state_line(const nlohmann::json & message)
  {
    const auto state = parse_state(message);
    if (state.has_value()) {
      const LineTime when = line_time(*state);  // once: the head and the wheels share a stamp
      publish_neck(*state, when);  // first: odom_publish mutes the wheels, not the head
      publish_state(*state, when);
    }
  }

  /// The neck's parameters: config/neck.json's geometry (pepin.neck.bridge_parameters), the
  /// frames, the joint names and the rate cap, all from the launch; a model that cannot be right
  /// (a sign that is not +-1, not two joint names) publishes nothing and says so once.
  void declare_neck()
  {
    neck_model_.reference_pan_ticks =
      static_cast<int>(declare_parameter<int>("neck_reference_pan_ticks", -1));
    neck_model_.reference_tilt_ticks =
      static_cast<int>(declare_parameter<int>("neck_reference_tilt_ticks", -1));
    neck_model_.pan_sign = static_cast<int>(declare_parameter<int>("neck_pan_sign", 1));
    neck_model_.tilt_sign = static_cast<int>(declare_parameter<int>("neck_tilt_sign", 1));
    neck_model_.mount_x_m = declare_parameter<double>("neck_mount_x_m", 0.0);
    neck_model_.mount_y_m = declare_parameter<double>("neck_mount_y_m", 0.0);
    neck_model_.mount_z_m = declare_parameter<double>("neck_mount_z_m", 0.0);
    neck_model_.mount_pitch_rad =
      declare_parameter<double>("neck_mount_pitch_deg", 0.0) / kRadToDeg;
    neck_model_.tilt_from_pan_x_m = declare_parameter<double>("neck_tilt_from_pan_x_m", 0.0);
    neck_model_.tilt_from_pan_z_m = declare_parameter<double>("neck_tilt_from_pan_z_m", 0.0);
    neck_model_.camera_from_tilt_x_m = declare_parameter<double>("neck_camera_from_tilt_x_m", 0.0);
    neck_model_.camera_from_tilt_z_m = declare_parameter<double>("neck_camera_from_tilt_z_m", 0.0);
    // The parent is the body the neck stands on; an empty child frame publishes no transform
    // (a laptop broadcasting a static camera edge, ros/laptop.sh vslam --fixed-head, owns it).
    neck_parent_frame_ = declare_parameter<std::string>("neck_parent_frame", base_frame_);
    neck_camera_frame_ = declare_parameter<std::string>("neck_camera_frame", "");
    neck_joint_names_ = declare_parameter<std::vector<std::string>>(
      "neck_joint_names", std::vector<std::string>{"neck_pan", "head_tilt"});
    const double hz = declare_parameter<double>("neck_publish_hz", 50.0);
    neck_grid_ = NeckGrid(hz);
    // How old a line's encoder read may be when it arrives and still be dated by it: on the board
    // the read is milliseconds old; a bridge on another machine has another monotonic clock, and
    // its lines are stamped on arrival instead (counted in the report line). /odom's limit too.
    neck_max_age_s_ = declare_parameter<double>("neck_stamp_max_age_s", 0.5);
    neck_publisher_ = create_publisher<sensor_msgs::msg::JointState>("neck/state", 10);
    const bool signs = std::abs(neck_model_.pan_sign) == 1 && std::abs(neck_model_.tilt_sign) == 1;
    neck_valid_ = signs && neck_joint_names_.size() == 2;
    if (!neck_valid_) {
      RCLCPP_ERROR(
        get_logger(), "neck: signs %d/%d and %zu joint names: /neck/state stays silent",
        neck_model_.pan_sign, neck_model_.tilt_sign, neck_joint_names_.size());
      return;
    }
    const std::string edge = neck_camera_frame_.empty() ?
      std::string(" (no camera transform: neck_camera_frame is empty)") :
      " and " + neck_parent_frame_ + " -> " + neck_camera_frame_;
    RCLCPP_INFO(
      get_logger(),
      "neck: reference pan %d tilt %d ticks%s, signs %+d %+d, mount %.3f m %.1f deg down; "
      "/neck/state%s at <= %g Hz",
      neck_model_.reference_pan_ticks, neck_model_.reference_tilt_ticks,
      neck_model_.known() ? "" : " UNREAD (every pose is the static mount)", neck_model_.pan_sign,
      neck_model_.tilt_sign, neck_model_.mount_z_m, neck_model_.mount_pitch_rad * kRadToDeg,
      edge.c_str(), hz);
  }

  /// One state line's two moments on the ROS clock: when it reached this node, and when its
  /// encoders were read (`read` is `arrival` and `dated` false when the line's age is not one a
  /// line on this machine can have).
  struct LineTime
  {
    rclcpp::Time arrival;
    rclcpp::Time read;
    bool dated;
  };

  /// The line's `t` (the board's monotonic clock, as monotonic_s() here) carried onto the ROS
  /// clock by its age, beside the arrival it was carried from.
  LineTime line_time(const BaseState & state) const
  {
    const rclcpp::Time arrival = now();
    const double age = monotonic_s() - state.stamp_s;
    if (age >= 0.0 && age <= neck_max_age_s_) {
      return {arrival, arrival - rclcpp::Duration::from_seconds(age), true};
    }
    return {arrival, arrival, false};
  }

  /// Reader thread: a line that carries the neck's ticks as /neck/state and, with a camera frame,
  /// base_link -> camera_link, both at that read's stamp, under the rate cap.
  void publish_neck(const BaseState & state, const LineTime & when)
  {
    ++neck_lines_;
    if (!state.neck_read || !neck_valid_) {
      return;
    }
    ++neck_heard_;
    neck_pan_ticks_ = state.pan_ticks;
    neck_tilt_ticks_ = state.tilt_ticks;
    if (!neck_grid_.due(state.stamp_s)) {
      return;
    }
    if (!when.dated) {
      ++neck_arrival_stamped_;
    }
    const rclcpp::Time stamp = when.read;
    const NeckAngles angles = joint_angles(neck_model_, state.pan_ticks, state.tilt_ticks);
    sensor_msgs::msg::JointState joints;
    joints.header.stamp = stamp;
    joints.name = neck_joint_names_;
    joints.position = {angles.pan_rad, angles.pitch_rad};
    neck_publisher_->publish(joints);
    ++neck_sent_;
    if (neck_camera_frame_.empty()) {
      return;
    }
    const NeckPose pose = camera_pose(neck_model_, angles);
    const auto q = quaternion_from_rpy(pose.roll, pose.pitch, pose.yaw);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = neck_parent_frame_;
    transform.child_frame_id = neck_camera_frame_;
    transform.transform.translation.x = pose.x;
    transform.transform.translation.y = pose.y;
    transform.transform.translation.z = pose.z;
    transform.transform.rotation.x = q[0];
    transform.transform.rotation.y = q[1];
    transform.transform.rotation.z = q[2];
    transform.transform.rotation.w = q[3];
    tf_->sendTransform(transform);
  }

  /// The neck as a report line prints it: how many lines carried its ticks, how many went out,
  /// where it points, and how many reads were dated on arrival.
  ///
  /// ``neck: 3000 of 3000 lines carried the ticks, 3000 published; pan 2029 (+0.0 deg) tilt 2311
  /// (23.8 deg down); 0 dated on arrival`` once a minute.
  std::string neck_state() const
  {
    char line[256];
    if (neck_heard_.load() == 0) {
      std::snprintf(
        line, sizeof(line), "neck: none of %ld lines carried the ticks, nothing published",
        neck_lines_.load());
      return line;
    }
    const NeckAngles angles = joint_angles(neck_model_, neck_pan_ticks_, neck_tilt_ticks_);
    std::snprintf(
      line, sizeof(line),
      "neck: %ld of %ld lines carried the ticks, %ld published; pan %d (%+.1f deg) tilt %d "
      "(%.1f deg down); %ld dated on arrival",
      neck_heard_.load(), neck_lines_.load(), neck_sent_.load(), neck_pan_ticks_.load(),
      angles.pan_rad * kRadToDeg, neck_tilt_ticks_.load(), angles.pitch_rad * kRadToDeg,
      neck_arrival_stamped_.load());
    return line;
  }

  /// One state line as a nav_msgs/Odometry on /odom and an odom->base_link transform, both
  /// dated by `odom_stamp`: the line's encoder read (default) or its arrival here.
  void publish_state(const BaseState & state, const LineTime & when)
  {
    const bool publish = get_parameter("odom_publish").as_bool();
    odom_publish_ = publish;
    if (!publish) {
      forget_wheel_twist();  // the gap this mute makes is not a measurement
      return;
    }
    const bool encoder = get_parameter("odom_stamp").as_string() != "arrival";
    odom_stamp_encoder_ = encoder;
    ++odom_lines_;
    if (encoder && !when.dated) {
      ++odom_arrival_stamped_;
    }
    const rclcpp::Time stamp = encoder ? when.read : when.arrival;
    const double qz = std::sin(state.theta / 2.0);
    const double qw = std::cos(state.theta / 2.0);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = state.x;
    odom.pose.pose.position.y = state.y;
    odom.pose.pose.orientation.z = qz;
    odom.pose.pose.orientation.w = qw;
    odom.pose.covariance = pose_covariance_;
    const auto twist = odom_twist(state);
    odom.twist.twist.linear.x = twist.linear;  // the body frame: x forward, yaw CCW
    odom.twist.twist.angular.z = twist.angular;
    odom.twist.covariance = twist_covariance_;
    odom_publisher_->publish(odom);
    if (!publish_tf_) {
      return;
    }

    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = odom_frame_;
    transform.child_frame_id = base_frame_;
    transform.transform.translation.x = state.x;
    transform.transform.translation.y = state.y;
    transform.transform.rotation.z = qz;
    transform.transform.rotation.w = qw;
    tf_->sendTransform(transform);
  }

  /// The twist /odom carries: measured off two wheel poses, or the commanded one.
  ///
  /// The parameter is read per sample so ``ros2 param set`` switches a live filter's input
  /// without a restart; the source is named in every link-up line. Called from the reader
  /// thread only, so the estimator needs no lock; the flag the log line reads is atomic.
  ///
  /// COVARIANCE. Unchanged, and on purpose: the encoder is not this measurement's error.
  /// One tick is pi * 0.125 m / 4096 = 9.6e-5 m of wheel travel (config/base.json), a pose
  /// difference carries the quantisation of two reads (sigma = 9.6e-5 * sqrt(2/12) = 3.9e-5 m
  /// per wheel), and over the 0.02 s between state lines (base_server at 50 Hz since 2026-10-01;
  /// 0.06 s before) that is 1.4e-3 m/s of forward noise and 5.5e-3 rad/s of yaw noise --
  /// variances of 1.9e-6 (m/s)^2 and 3.0e-5 (rad/s)^2, two to three orders below the 0.001 and
  /// 0.01 the message already carries (protocol.hpp:203). Those numbers are slip and
  /// wheel-diameter error, measured against the gyro over 51 tapes (ros/params/ekf.yaml), and
  /// they are what the filter needs to hear. Quantisation would matter only past ~500 Hz.
  BodyTwist odom_twist(const BaseState & state)
  {
    const auto source = get_parameter("odom_twist_source").as_string();
    const bool measured = source != "commanded";
    twist_measured_ = measured;
    // Differenced on every line whatever /odom ends up carrying: the measured twist is also the
    // wheels' word on whether the cart is standing still, which the gyro's bias needs in both
    // modes (witness_rest below), and a switch back to "measured" then has a real twist at once
    // instead of one re-priming zero. `commanded` changes what is published, not what is measured.
    const auto wheels = twist_from_pose_.update(state.x, state.y, state.theta, state.stamp_s);
    witness_rest(state, wheels);
    return measured ? wheels : BodyTwist{state.v, state.w};
  }

  /// Reader thread: publish the wheels' word on rest for the gyro's zero (RestWitness has the
  /// rules and the reasons). Two atomic stores, so the 50 Hz IMU loop never waits on this thread.
  void witness_rest(const BaseState & state, const BodyTwist & wheels)
  {
    (void)wheels;  // the twist is what /odom carries; rest is judged on the wheels' own travel
    const bool twist_is_zero = tick_dither_.still(state.d_left_m, state.d_right_m);
    still_since_.store(
      rest_witness_.judge(state.stamp_s, monotonic_s(), state.moving, twist_is_zero));
    witness_at_.store(rest_witness_.at());
  }

  /// Re-prime the wheels' twist, and with it the rest it was witnessing.
  void forget_wheel_twist()
  {
    twist_from_pose_.reset();
    rest_witness_.forget();
    still_since_.store(0.0);
  }

  /// Any thread (the IMU loop, the zero-velocity timer): the time since which the WHEELS have
  /// witnessed rest, or 0 when they have not.
  double still_witness(double now) const
  {
    return rest_witnessed(still_since_.load(), witness_at_.load(), now, kStateGapMaxS);
  }

  /// Forward a twist at once (clamped to the ceiling); remember it until it goes stale.
  void accept_command(double v, double w)
  {
    v = std::max(-max_linear_, std::min(max_linear_, v));
    w = std::max(-max_angular_, std::min(max_angular_, w));
    if (v != 0.0 || w != 0.0) {
      command_moving_at_.store(monotonic_s());  // the zero-velocity update stops on the intent
    }
    {
      const std::lock_guard<std::mutex> guard(mutex_);
      command_ = std::make_pair(v, w);
      command_at_ = std::chrono::steady_clock::now();
      stop_sent_ = false;
    }
    link_->send(encode_twist(v, w));
  }

  /// Feed the board's deadman while /cmd_vel is steady; send one stop when it goes quiet.
  void resend_command()
  {
    std::optional<std::pair<double, double>> command;
    double age_s = 0.0;
    bool stop_sent = false;
    {
      const std::lock_guard<std::mutex> guard(mutex_);
      command = command_;
      age_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - command_at_).count();
      stop_sent = stop_sent_;
    }
    if (!command.has_value()) {
      return;
    }
    if (age_s < cmd_timeout_s_) {
      link_->send(encode_twist(command->first, command->second));
    } else if (!stop_sent) {
      {
        const std::lock_guard<std::mutex> guard(mutex_);
        stop_sent_ = true;
      }
      link_->send(encode_stop());
      RCLCPP_INFO(get_logger(), "no cmd_vel for %.1f s: wheels stopped", cmd_timeout_s_);
    }
  }

  /// The live switches as a report line prints them: ``imu_publish=on odom_publish=off``.
  std::string switch_state() const
  {
    return std::string("imu_publish=") + (imu_publish_ ? "on" : "off") + " odom_publish=" +
           (odom_publish_ ? "on" : "off") + " imu_bias_tracking=" +
           (imu_bias_tracking_ ? "on" : "off") + " zupt_publish=" +
           (zupt_publish_ ? "on" : "off") + " odom_stamp=" +
           (odom_stamp_encoder_ ? "encoder" : "arrival");
  }

  /// How /odom is dated, as the minute line prints it: ``odom stamp encoder: 3000 lines, 0 dated
  /// on arrival``.
  std::string odom_stamp_state() const
  {
    char line[128];
    std::snprintf(
      line, sizeof(line), "odom stamp %s: %ld lines, %ld dated on arrival",
      odom_stamp_encoder_ ? "encoder" : "arrival", odom_lines_.load(),
      odom_arrival_stamped_.load());
    return line;
  }

  /// The zero-velocity update as a report line prints it: publishing or why not, the count, and
  /// every live setting in force.
  ///
  /// ``zupt publishing for 312 s, 3121 sent [rate 10 Hz, var 1e-06 xy 1e-06 yaw, settle 2 s, cmd
  /// hold 0.5 s, gyro quiet 0.005 rad/s]`` on a parked cart; ``zupt silent (settling after the last
  /// motion), ...`` just after a leg; ``zupt off, ...`` under ``zupt_publish false``.
  std::string zupt_state() const
  {
    char line[320];
    const std::string settings = zupt_settings();
    if (!zupt_publish_) {
      std::snprintf(
        line, sizeof(line), "zupt off, %ld sent [%s]", zupt_sent_.load(), settings.c_str());
    } else if (zupt_publishing_) {
      std::snprintf(
        line, sizeof(line), "zupt publishing for %.0f s, %ld sent [%s]",
        monotonic_s() - zupt_since_.load(), zupt_sent_.load(), settings.c_str());
    } else {
      std::snprintf(
        line, sizeof(line), "zupt silent (%s), %ld sent [%s]",
        describe(static_cast<ZuptVerdict>(zupt_verdict_.load())), zupt_sent_.load(),
        settings.c_str());
    }
    return line;
  }

  /// The live zero-velocity settings in force, as the status line prints them.
  std::string zupt_settings() const
  {
    char line[192];
    std::snprintf(
      line, sizeof(line),
      "rate %g Hz, var %g xy %g yaw, settle %g s, cmd hold %g s, gyro quiet %g rad/s",
      zupt_hz_.load(), zupt_var_linear_.load(), zupt_var_yaw_.load(), zupt_settle_s_.load(),
      zupt_cmd_hold_s_.load(), gyro_quiet_rad_s_.load());
    return line;
  }

  /// Executor: one tick of the zero-velocity update -- a zero twist on /zupt while ZuptGate says
  /// the cart is at rest and `zupt_publish` is on, and nothing at all otherwise.
  ///
  /// Every witness is read fresh each tick from the thread that owns it (the wheels' reader, the
  /// IMU loop, the /cmd_vel callback), and so is every live setting, so the first sample of motion
  /// any witness sees, and any `ros2 param set`, is in force at the next tick. The stamp is the
  /// ROS clock's now, like /odom's.
  void publish_zupt()
  {
    const bool enabled = zupt_publish_.load();
    const double now_s = monotonic_s();
    const ZuptGate gate(zupt_settle_s_.load(), zupt_cmd_hold_s_.load(), kStateGapMaxS);
    const ZuptVerdict verdict = gate.judge(now_s, rest_evidence(now_s));
    zupt_verdict_ = static_cast<int>(verdict);
    const bool publish = enabled && verdict == ZuptVerdict::kAtRest;
    if (publish != zupt_publishing_) {
      note_zupt_change(publish, verdict, now_s);
    }
    if (!publish) {
      return;
    }
    nav_msgs::msg::Odometry message;
    message.header.stamp = now();
    message.header.frame_id = odom_frame_;
    message.child_frame_id = base_frame_;
    message.pose.covariance = zupt_pose_covariance_;  // no pose is claimed: odom2 fuses none
    // The twist itself is zero as built; only its claim is live.
    message.twist.covariance =
      rest_zupt_twist_covariance(zupt_var_linear_.load(), zupt_var_yaw_.load());
    zupt_publisher_->publish(message);
    ++zupt_sent_;
  }

  /// Every witness's last word for ZuptGate: the wheels' rest, the last non-zero command, the
  /// gyro's last sample and last turn. `gyro_at` is read before `gyro_turn_at`, the reverse of
  /// the order witness_gyro() writes them, so a sample seen here is never seen without its turn.
  RestEvidence rest_evidence(double now) const
  {
    RestEvidence evidence;
    evidence.still_since = still_witness(now);
    evidence.command_at = command_moving_at_.load();
    evidence.gyro_at = gyro_at_.load();
    evidence.gyro_turn_at = gyro_turn_at_.load();
    return evidence;
  }

  /// Say it once whenever the update starts or stops, with the reason it stopped.
  void note_zupt_change(bool publishing, ZuptVerdict verdict, double now)
  {
    if (publishing) {
      zupt_since_ = now;
      zupt_publishing_ = true;
      RCLCPP_INFO(
        get_logger(), "zupt: the cart is at rest, /zupt publishing [%s]", zupt_settings().c_str());
      return;
    }
    zupt_publishing_ = false;
    RCLCPP_INFO(
      get_logger(), "zupt: stopped after %.1f s at rest (%s)", now - zupt_since_.load(),
      zupt_publish_ ? describe(verdict) : "zupt_publish off");
  }

  /// (Re)start the zero-velocity timer at the rate in force, cancelling the one it replaces.
  void start_zupt_timer()
  {
    if (zupt_timer_) {
      zupt_timer_->cancel();
    }
    zupt_timer_ = create_wall_timer(period(1.0 / zupt_hz_.load()), [this] {publish_zupt();});
  }

  /// A parameter's value as a number: a double as it is, an integer widened, a string that is
  /// wholly one finite number parsed, anything else nothing.
  ///
  /// The last two are how people type: `ros2 param set ... 50` arrives as an integer, and
  /// `ros2 param set ... 1e-4` as a STRING -- the CLI reads its value as YAML 1.1, where a float
  /// needs a dot (1.0e-4). Refusing either would make the obvious command fail on the robot.
  static std::optional<double> number_of(const rclcpp::Parameter & parameter)
  {
    switch (parameter.get_type()) {
      case rclcpp::ParameterType::PARAMETER_DOUBLE:
        return parameter.as_double();
      case rclcpp::ParameterType::PARAMETER_INTEGER:
        return static_cast<double>(parameter.as_int());
      case rclcpp::ParameterType::PARAMETER_STRING: {
        const std::string text = parameter.as_string();
        char * end = nullptr;
        const double value = std::strtod(text.c_str(), &end);
        const bool whole = !text.empty() && end == text.c_str() + text.size();
        return whole && std::isfinite(value) ? std::optional<double>(value) : std::nullopt;
      }
      default:
        return std::nullopt;
    }
  }

  /// Declare one live zero-velocity number and return the value in force: the launch's when it is
  /// a number inside its range (zupt.hpp's kZuptRanges), else `fallback`, with a warning -- a typo
  /// in a launch file must not keep the base from starting. Typed dynamically, so an integer is
  /// taken as the number it is instead of refused as the wrong type.
  double declare_zupt_number(const std::string & name, double fallback, const char * what)
  {
    const ZuptRange & range = *zupt_range(name);
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.description = what;
    char limits[96];
    std::snprintf(
      limits, sizeof(limits), "live; %g..%g, a value outside is refused", range.low, range.high);
    descriptor.additional_constraints = limits;
    descriptor.dynamic_typing = true;
    const auto value = number_of(
      rclcpp::Parameter(name, declare_parameter(name, rclcpp::ParameterValue(fallback), descriptor)));
    if (value.has_value() && zupt_in_range(range, *value)) {
      return *value;
    }
    RCLCPP_WARN(
      get_logger(), "zupt: %s from the launch refused, outside [%g, %g]; %g in force",
      name.c_str(), range.low, range.high, fallback);
    set_parameter(rclcpp::Parameter(name, fallback));
    return fallback;
  }

  /// The atomic that holds the live zero-velocity number `name`, or nullptr for any other name.
  std::atomic<double> * zupt_setting(const std::string & name)
  {
    if (name == "zupt_rate_hz") {return &zupt_hz_;}
    if (name == "zupt_var_linear") {return &zupt_var_linear_;}
    if (name == "zupt_var_yaw") {return &zupt_var_yaw_;}
    if (name == "zupt_settle_s") {return &zupt_settle_s_;}
    if (name == "zupt_cmd_hold_s") {return &zupt_cmd_hold_s_;}
    if (name == "zupt_gyro_quiet_rad_s") {return &gyro_quiet_rad_s_;}
    return nullptr;
  }

  /// Parameter service, before a set: refuse a zero-velocity number that is not a number or lies
  /// outside its range, with the reason logged and handed back to `ros2 param set` -- the whole
  /// set is refused and the values in force stay. Every other parameter of the node passes.
  rcl_interfaces::msg::SetParametersResult check_zupt_settings(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    for (const auto & parameter : parameters) {
      const ZuptRange * range = zupt_range(parameter.get_name());
      if (range == nullptr) {
        continue;
      }
      const auto value = number_of(parameter);
      if (value.has_value() && zupt_in_range(*range, *value)) {
        continue;
      }
      char given[64];
      if (value.has_value()) {
        std::snprintf(given, sizeof(given), "%g", *value);  // 1e-12, not rclcpp's "0.000000"
      } else {
        std::snprintf(given, sizeof(given), "%s", parameter.value_to_string().c_str());
      }
      char reason[192];
      std::snprintf(
        reason, sizeof(reason), "%s %s refused: a number in [%g, %g] is required",
        range->name, given, range->low, range->high);
      RCLCPP_WARN(
        get_logger(), "zupt: %s; %g stays in force", reason,
        zupt_setting(parameter.get_name())->load());
      result.successful = false;
      result.reason = reason;
      return result;
    }
    return result;
  }

  /// Parameter service, after a set was accepted: put each zero-velocity setting in force for the
  /// next tick, say so once per change, and re-time the timer when the rate moved.
  void apply_zupt_settings(const std::vector<rclcpp::Parameter> & parameters)
  {
    for (const auto & parameter : parameters) {
      const std::string & name = parameter.get_name();
      if (name == "zupt_publish") {
        const bool on = parameter.as_bool();
        if (zupt_publish_.exchange(on) != on) {
          RCLCPP_INFO(get_logger(), "zupt: zupt_publish %s", on ? "on" : "off");
        }
        continue;
      }
      std::atomic<double> * slot = zupt_setting(name);
      const auto value = number_of(parameter);
      if (slot == nullptr || !value.has_value()) {
        continue;
      }
      const double before = slot->exchange(*value);
      if (before == *value) {
        continue;
      }
      RCLCPP_INFO(
        get_logger(), "zupt: %s %g -> %g, in force at the next tick", name.c_str(), before, *value);
      if (slot == &zupt_hz_) {
        start_zupt_timer();
      }
    }
  }

  /// The gyro's zero as a report line prints it: the bias, the rest blocks behind it, its age.
  ///
  /// ``gyro bias +0.001 -0.028 +0.074 deg/s, 12 rest blocks of 100 samples, last 34 s ago`` — the
  /// line the static check reads to see that blocks keep arriving under a parked cart. Written
  /// from the IMU thread's atomics, so the status timer may print it without waiting for a sample.
  std::string gyro_bias_state() const
  {
    char line[192];
    const long blocks = bias_blocks_.load();
    if (blocks == 0) {
      std::snprintf(line, sizeof(line), "gyro bias: no rest block yet, nothing published");
      return line;
    }
    std::snprintf(
      line, sizeof(line),
      "gyro bias %+.3f %+.3f %+.3f deg/s, %ld rest block%s of %ld samples, last %.0f s ago",
      bias_x_.load() * kRadToDeg, bias_y_.load() * kRadToDeg, bias_z_.load() * kRadToDeg,
      blocks, blocks == 1 ? "" : "s", bias_block_samples_.load(),
      monotonic_s() - bias_at_.load());
    return line;
  }

  /// Say it once whenever the link comes up or goes down, with the switches that decide
  /// whether anything is published at all (CLAUDE.md rule 19) — and once a minute, whatever the
  /// link does, where the gyro's zero stands.
  ///
  /// The periodic line lives on this timer rather than on the IMU thread so its "last block N s
  /// ago" is a real age: on a parked cart it is what shows that rest blocks keep arriving, and on
  /// a cart that never stands still it is what shows they do not.
  void log_link_status()
  {
    if (imu_publisher_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 60000, "%s; %s; %s; %s", gyro_bias_state().c_str(),
        zupt_state().c_str(), neck_state().c_str(), odom_stamp_state().c_str());
    } else {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 60000, "%s; %s", neck_state().c_str(),
        odom_stamp_state().c_str());
    }
    const auto change = link_->take_status_change();
    if (!change.has_value()) {
      return;
    }
    if (change->first) {
      RCLCPP_INFO(
        get_logger(), "%s; odom twist: %s, %s; %s; %s; %s", change->second.c_str(),
        twist_measured_ ? "measured" : "commanded", switch_state().c_str(),
        gyro_bias_state().c_str(), zupt_state().c_str(), neck_state().c_str());
    } else {
      RCLCPP_WARN(get_logger(), "%s", change->second.c_str());
    }
  }

  /// Open the IMU and start sampling it; a missing chip is a warning, not a failure.
  void start_imu()
  {
    std::string error;
    if (!imu_.open_device(imu_device_, imu_address_, imu_output_rate_hz_, error)) {
      RCLCPP_ERROR(get_logger(), "no IMU (%s): the bridge runs on wheel odometry", error.c_str());
      return;
    }
    const unsigned divider = imu_.output_divider();
    RCLCPP_INFO(
      get_logger(),
      "IMU on %s read at %d Hz, chip output %g Hz (SMPLRT_DIV %u, DLPF_CFG %u), stamped %.1f ms "
      "before the read (imu_filter_delay_s), WHO_AM_I 0x%02x, %c axis up, published in %s",
      imu_device_.c_str(), static_cast<int>(imu_rate_hz_), 1000.0 / (1.0 + divider), divider,
      static_cast<unsigned>(kDlpfConfig), imu_filter_delay_s_ * 1000.0,
      static_cast<unsigned>(imu_.who_am_i()), imu_up_axis_, imu_frame_.c_str());
    imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 10);
    gyro_bias_ = GyroBiasTracker(imu_bias_s_, imu_rate_hz_);
    bias_block_samples_ = gyro_bias_.block_samples();
    imu_running_ = true;
    imu_thread_ = std::thread([this] {read_imu();});
  }

  /// Ask the IMU thread to finish and join it; safe to call twice.
  void stop_imu()
  {
    imu_running_ = false;
    if (imu_thread_.joinable()) {
      imu_thread_.join();
    }
    imu_.close_device();
  }

  /// IMU thread: keep the gyro's zero from the wheels' rest, and publish once there is one.
  ///
  /// Nothing is published before a bias exists on purpose — a raw yaw rate offset fed to the EKF
  /// exactly while it initialises is worse than no measurement at all. Under `imu_bias_tracking`
  /// that is no longer a stopwatch but the wheels' word: the cart may be rolling when this node
  /// starts (the old code could not know, and took the roll as its zero), so the first block waits
  /// for witnessed rest however long that takes, and says so every 10 s while it waits.
  void read_imu()
  {
    const auto tick = period(1.0 / imu_rate_hz_);
    auto next = std::chrono::steady_clock::now();
    const double boot_s = monotonic_s();
    // `imu_bias_tracking` off is the contract this node shipped with: one block, taken on trust
    // the moment the node starts, never replaced. Dating the last motion one block before the
    // start declares the settle window already over, which is exactly what it used to do.
    const double assumed_motion_at = boot_s - imu_bias_s_;
    if (imu_bias_s_ > 0.0) {
      RCLCPP_INFO(
        get_logger(), "gyro bias: %ld samples (%.1f s) per rest block, %.1f s of settling first",
        gyro_bias_.block_samples(), imu_bias_s_, imu_bias_s_);
    }
    while (imu_running_) {
      next += tick;
      const auto sample_at = std::chrono::steady_clock::now();
      if (next < sample_at) {
        next = sample_at + tick;  // a stall must not turn into a burst of catch-up reads
      }
      std::this_thread::sleep_until(next);
      if (!imu_running_) {
        break;
      }
      std::string error;
      const auto sample = imu_.read_sample(error);
      const rclcpp::Time read_at = now();  // the burst is back: the sample is <= 1 output period old
      if (!sample.has_value()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "IMU read failed: %s", error.c_str());
        continue;
      }
      const double t = monotonic_s();
      const bool tracking = get_parameter("imu_bias_tracking").as_bool();
      imu_bias_tracking_ = tracking;
      bool took_block = false;
      if (tracking) {
        took_block = gyro_bias_.update(
          t, sample->gyro_x, sample->gyro_y, sample->gyro_z, still_witness(t));
      } else if (!gyro_bias_.ready()) {
        took_block = gyro_bias_.update(
          t, sample->gyro_x, sample->gyro_y, sample->gyro_z, assumed_motion_at);
      }
      if (took_block) {
        const GyroBias bias = gyro_bias_.bias();
        bias_x_ = bias.x;
        bias_y_ = bias.y;
        bias_z_ = bias.z;
        bias_blocks_ = gyro_bias_.blocks();
        bias_at_ = t;
        if (gyro_bias_.blocks() == 1) {
          // The one block worth a line of its own: the IMU starts publishing on it. Every block
          // after it is a parked cart's routine, reported once a minute by log_link_status.
          RCLCPP_INFO(get_logger(), "%s; imu/data_raw is live", gyro_bias_state().c_str());
        }
      }
      if (!gyro_bias_.ready()) {
        if (t - boot_s > 2.0 * imu_bias_s_) {  // settle plus block: the earliest one can close
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 10000,
            "gyro bias: the wheels have not witnessed %.1f s of rest (%.1f s after the last "
            "motion) in %.0f s; imu/data_raw stays silent. Is the cart moving, is /odom muted, "
            "is the base link up? `imu_bias_tracking false` takes the old boot-only bias.",
            imu_bias_s_, imu_bias_s_, t - boot_s);
        }
        continue;
      }
      const GyroBias bias = gyro_bias_.bias();
      witness_gyro(t, base_yaw_rate(*sample, bias));
      publish_imu(*sample, bias, read_at);
    }
  }

  /// The bias-corrected yaw rate in base_link, rad/s counter-clockwise: what imu0 index 11 reads.
  double base_yaw_rate(const ImuSample & sample, const GyroBias & bias) const
  {
    double gyro[3];
    to_base_axes(imu_up_axis_, sample.gyro_x - bias.x, sample.gyro_y - bias.y,
      sample.gyro_z - bias.z, gyro);
    return gyro[2];
  }

  /// IMU thread: the gyro's word for the zero-velocity update -- when the last bias-corrected
  /// sample arrived, and when the last one was a turn (zupt.hpp's gyro_turning). The chip is the
  /// witness, not the message: `imu_publish` off mutes /imu/data_raw and leaves this untouched.
  void witness_gyro(double t, double yaw_rate)
  {
    if (gyro_turning(yaw_rate, gyro_quiet_rad_s_.load())) {
      gyro_turn_at_.store(t);  // first: a reader that sees this sample's time sees its turn
    }
    gyro_at_.store(t);
  }

  /// One conversion as sensor_msgs/Imu, gyro bias removed, no orientation claimed.
  /// One reading turned from the chip's axes into the robot's, per ``imu_up_axis``.
  ///
  /// The board is bolted with its Y axis pointing up, so the yaw rate the filter needs sits on the
  /// chip's Y. Publishing raw in an ``imu_link`` frame and leaving the rotation to a static
  /// transform did not work: robot_localization kept the rate on Y, and two_d_mode then zeroed it,
  /// so the filter never turned at all (measured 2026-09-08: gyro 27.9 deg/s, filter 0.0). The
  /// mounting is the driver's business — here the reading comes out in base_link axes and nothing
  /// downstream has to know how the chip is screwed on.
  static void to_base_axes(char up, double x, double y, double z, double out[3])
  {
    switch (up) {
      case 'x':  // chip X up: base (x, y, z) <- (-z, y, x)
        out[0] = -z; out[1] = y; out[2] = x;
        break;
      case 'y':  // chip Y up: base (x, y, z) <- (x, -z, y)
        out[0] = x; out[1] = -z; out[2] = y;
        break;
      default:  // chip Z up already
        out[0] = x; out[1] = y; out[2] = z;
        break;
    }
  }

  /// One sample as sensor_msgs/Imu, unless ``imu_publish`` is off — then nothing goes out. Dated
  /// by its read less the chip's filter delay: when the motion it describes happened.
  void publish_imu(const ImuSample & sample, const GyroBias & bias, const rclcpp::Time & read_at)
  {
    const bool publish = get_parameter("imu_publish").as_bool();
    imu_publish_ = publish;
    if (!publish) {
      return;
    }
    sensor_msgs::msg::Imu message;
    message.header.stamp = read_at - rclcpp::Duration::from_seconds(imu_filter_delay_s_);
    message.header.frame_id = imu_frame_;
    message.orientation_covariance[0] = -1.0;  // the ROS way to say "no orientation here"
    double gyro[3];
    double accel[3];
    to_base_axes(imu_up_axis_, sample.gyro_x - bias.x, sample.gyro_y - bias.y,
      sample.gyro_z - bias.z, gyro);
    to_base_axes(imu_up_axis_, sample.accel_x, sample.accel_y, sample.accel_z, accel);
    message.angular_velocity.x = gyro[0];
    message.angular_velocity.y = gyro[1];
    message.angular_velocity.z = gyro[2];
    message.linear_acceleration.x = accel[0];
    message.linear_acceleration.y = accel[1];
    message.linear_acceleration.z = accel[2];
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::size_t diagonal = axis * 4;  // 0, 4, 8 of a row-major 3x3
      message.angular_velocity_covariance[diagonal] = kGyroStdDev * kGyroStdDev;
      message.linear_acceleration_covariance[diagonal] = kAccelStdDev * kAccelStdDev;
    }
    imu_publisher_->publish(message);
  }

  std::string odom_frame_;
  std::string base_frame_;
  double cmd_timeout_s_ = 0.5;
  double max_linear_ = 0.15;
  bool publish_tf_ = true;
  std::atomic<bool> shut_down_{false};
  double max_angular_ = 0.6;
  std::string imu_device_;
  std::string imu_frame_;
  char imu_up_axis_ = 'z';  // which chip axis points up: the mounting, in one letter
  int imu_address_ = 0x68;
  double imu_rate_hz_ = 50.0;
  double imu_output_rate_hz_ = 1000.0;  // the chip's register refresh (SMPLRT_DIV), not our read
  double imu_filter_delay_s_ = 0.0;     // the DLPF's group delay, subtracted from every stamp
  double imu_bias_s_ = 2.0;
  std::atomic<bool> twist_measured_{true};  // read by the status timer, written by the reader
  std::atomic<bool> imu_publish_{true};   // what the report line says; written by the IMU thread
  std::atomic<bool> odom_publish_{true};  // ... and this one by the reader thread
  std::atomic<bool> imu_bias_tracking_{true};  // ... and this one by the IMU thread too
  std::atomic<bool> odom_stamp_encoder_{true};  // ... and this one by the reader thread
  std::atomic<long> odom_lines_{0};             // lines published on /odom
  std::atomic<long> odom_arrival_stamped_{0};   // ... of them dated on arrival under "encoder"
  TwistFromPose twist_from_pose_{kStateGapMaxS};  // touched from the reader thread only
  RestWitness rest_witness_{kStateGapMaxS};       // ... and so is this one
  // One encoder tick of wheel travel: pi * wheel_diameter_m / ticks_per_rev of config/base.json
  // (0.125 m, 4096) = 9.587e-5 m — the unit the state lines' dl/dr come in, seen live.
  TickDither tick_dither_{3.14159265358979323846 * 0.125 / 4096.0};  // reader thread only
  std::array<double, 36> pose_covariance_{};
  std::array<double, 36> twist_covariance_{};

  // THE WHEELS' WORD ON REST, from the reader thread to the IMU thread (witness_rest,
  // still_witness). Two doubles instead of a lock: the 50 Hz loop must never wait on a TCP
  // reader, and a torn read is impossible for a lock-free atomic double (see the static_assert).
  std::atomic<double> still_since_{0.0};  // monotonic start of the rest spell; 0 = not at rest
  std::atomic<double> witness_at_{0.0};   // when a state line last judged it; staleness is silence
  // The gyro's zero, and the same numbers again for whoever prints the report line. Owned by the
  // IMU thread exactly as twist_from_pose_ is owned by the reader thread.
  GyroBiasTracker gyro_bias_;
  std::atomic<double> bias_x_{0.0};
  std::atomic<double> bias_y_{0.0};
  std::atomic<double> bias_z_{0.0};
  std::atomic<long> bias_blocks_{0};
  std::atomic<long> bias_block_samples_{0};
  std::atomic<double> bias_at_{0.0};

  // THE ZERO-VELOCITY UPDATE'S WITNESSES (publish_zupt, zupt.hpp), each written by the thread that
  // owns it and read by the zupt timer, 0 = never: the last non-zero command (the /cmd_vel
  // callbacks), the gyro's last sample and its last turn (the IMU thread), and the quiet
  // threshold, which the timer copies from its parameter for the IMU thread to judge by.
  std::atomic<double> command_moving_at_{0.0};
  std::atomic<double> gyro_at_{0.0};
  std::atomic<double> gyro_turn_at_{0.0};
  std::atomic<double> gyro_quiet_rad_s_{kGyroQuietRadS};
  // ...the live settings, written by apply_zupt_settings and read by the timer each tick...
  std::atomic<double> zupt_hz_{kZuptHz};
  std::atomic<double> zupt_var_linear_{kRestZuptVariance};
  std::atomic<double> zupt_var_yaw_{kRestZuptVariance};
  std::atomic<double> zupt_settle_s_{0.0};
  std::atomic<double> zupt_cmd_hold_s_{0.0};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr zupt_check_;
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr zupt_apply_;
  // ...and what the update did, for the report line and the start/stop lines.
  std::atomic<bool> zupt_publish_{true};
  std::atomic<bool> zupt_publishing_{false};
  std::atomic<int> zupt_verdict_{static_cast<int>(ZuptVerdict::kNoRest)};
  std::atomic<long> zupt_sent_{0};
  std::atomic<double> zupt_since_{0.0};
  const std::array<double, 36> zupt_pose_covariance_ = diagonal(
    {kUnclaimedVariance, kUnclaimedVariance, kUnclaimedVariance, kUnclaimedVariance,
      kUnclaimedVariance, kUnclaimedVariance});

  // THE NECK (declare_neck, publish_neck): the model and the frames are fixed at start; the grid
  // is the reader thread's alone; the counters and the last ticks are what the report line reads.
  NeckModel neck_model_;
  std::string neck_parent_frame_;
  std::string neck_camera_frame_;
  std::vector<std::string> neck_joint_names_;
  NeckGrid neck_grid_{50.0};
  double neck_max_age_s_ = 0.5;
  bool neck_valid_ = false;
  std::atomic<long> neck_lines_{0};
  std::atomic<long> neck_heard_{0};
  std::atomic<long> neck_sent_{0};
  std::atomic<long> neck_arrival_stamped_{0};
  std::atomic<int> neck_pan_ticks_{0};
  std::atomic<int> neck_tilt_ticks_{0};
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr neck_publisher_;

  std::mutex mutex_;  // guards the command the resend timer repeats
  std::optional<std::pair<double, double>> command_;
  std::chrono::steady_clock::time_point command_at_{};
  bool stop_sent_ = true;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr zupt_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_stamped_subscription_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr resend_timer_;
  rclcpp::TimerBase::SharedPtr zupt_timer_;

  Mpu6050 imu_;  // opened only when imu_enable; ~BaseBridge joins the thread before these die
  std::atomic<bool> imu_running_{false};
  std::thread imu_thread_;
  // Last member on purpose: its destructor joins the reader thread before the publishers die.
  std::unique_ptr<JsonLineLink> link_;
};

}  // namespace pepin

/// Entry point: spin the bridge, and stop the wheels whatever happens on the way out.
// Loadable into any component container (the sensors one, next to the lidar driver: a
// separate rclcpp process costs ~140 MB on the board, a component a few tens); the same
// registration also generates the stand-alone `base_bridge` executable.
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pepin::BaseBridge)
