// Copyright 2026 Artem Belousov. Licensed under the Apache License, Version 2.0.
//
// ROS 2 node: the board's base server as /odom, odom->base_link, and a /cmd_vel sink.
// A C++ port of pepin_bringup/base_bridge.py, same parameters, same wire protocol:
// on the board's four A53 cores an rclpy process costs ~190 MB and a tenth of a core,
// this one ~25 MB and ~1%. The Python node stays; robot.launch.py picks one.
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
// The gyro's ZERO is re-measured for as long as the node lives, from the rest the WHEELS witness:
// the chip's bias moves with temperature, and a zero taken once at boot turned RTAB-Map's map
// +27 deg in 40 min under a parked cart. See gyro_bias.hpp, witness_rest() and read_imu().

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "pepin_base_cpp/gyro_bias.hpp"
#include "pepin_base_cpp/link.hpp"
#include "pepin_base_cpp/mpu6050.hpp"
#include "pepin_base_cpp/protocol.hpp"
#include "pepin_base_cpp/twist_from_pose.hpp"

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

/// Bridges the base server to ROS: /odom and odom->base_link out, /cmd_vel down to wheels.
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

    pose_covariance_ = odometry_pose_covariance();
    twist_covariance_ = odometry_twist_covariance();

    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
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
      publish_state(*state);
    }
  }

  /// One state line as a nav_msgs/Odometry on /odom and an odom->base_link transform.
  void publish_state(const BaseState & state)
  {
    const bool publish = get_parameter("odom_publish").as_bool();
    odom_publish_ = publish;
    if (!publish) {
      forget_wheel_twist();  // the gap this mute makes is not a measurement
      return;
    }
    const auto stamp = now();
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
  /// per wheel), and over the 0.05 s between state lines (base_server --publish-hz 20) that is
  /// 5.5e-4 m/s of forward noise and 2.2e-3 rad/s of yaw noise -- variances of 3.1e-7 (m/s)^2
  /// and 4.8e-6 (rad/s)^2, three to four orders below the 0.001 and 0.01 the message already
  /// carries (protocol.hpp:203). Those numbers are slip and wheel-diameter error, measured
  /// against the gyro over 51 tapes (ros/params/ekf.yaml), and they are what the filter needs
  /// to hear. Quantisation would only matter if the state rate rose far past 20 Hz.
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

  /// IMU thread: the time since which the WHEELS have witnessed rest, or 0 when they have not.
  double still_witness(double now) const
  {
    return rest_witnessed(still_since_.load(), witness_at_.load(), now, kStateGapMaxS);
  }

  /// Forward a twist at once (clamped to the ceiling); remember it until it goes stale.
  void accept_command(double v, double w)
  {
    v = std::max(-max_linear_, std::min(max_linear_, v));
    w = std::max(-max_angular_, std::min(max_angular_, w));
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
           (imu_bias_tracking_ ? "on" : "off");
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
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 60000, "%s", gyro_bias_state().c_str());
    }
    const auto change = link_->take_status_change();
    if (!change.has_value()) {
      return;
    }
    if (change->first) {
      RCLCPP_INFO(
        get_logger(), "%s; odom twist: %s, %s; %s", change->second.c_str(),
        twist_measured_ ? "measured" : "commanded", switch_state().c_str(),
        gyro_bias_state().c_str());
    } else {
      RCLCPP_WARN(get_logger(), "%s", change->second.c_str());
    }
  }

  /// Open the IMU and start sampling it; a missing chip is a warning, not a failure.
  void start_imu()
  {
    std::string error;
    if (!imu_.open_device(imu_device_, imu_address_, imu_rate_hz_, error)) {
      RCLCPP_ERROR(get_logger(), "no IMU (%s): the bridge runs on wheel odometry", error.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(), "IMU on %s at %d Hz, WHO_AM_I 0x%02x, %c axis up, published in %s",
      imu_device_.c_str(), static_cast<int>(imu_rate_hz_),
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
      publish_imu(*sample, gyro_bias_.bias());
    }
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

  /// One sample as sensor_msgs/Imu, unless ``imu_publish`` is off — then nothing goes out.
  void publish_imu(const ImuSample & sample, const GyroBias & bias)
  {
    const bool publish = get_parameter("imu_publish").as_bool();
    imu_publish_ = publish;
    if (!publish) {
      return;
    }
    sensor_msgs::msg::Imu message;
    message.header.stamp = now();
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
  double imu_bias_s_ = 2.0;
  std::atomic<bool> twist_measured_{true};  // read by the status timer, written by the reader
  std::atomic<bool> imu_publish_{true};   // what the report line says; written by the IMU thread
  std::atomic<bool> odom_publish_{true};  // ... and this one by the reader thread
  std::atomic<bool> imu_bias_tracking_{true};  // ... and this one by the IMU thread too
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

  std::mutex mutex_;  // guards the command the resend timer repeats
  std::optional<std::pair<double, double>> command_;
  std::chrono::steady_clock::time_point command_at_{};
  bool stop_sent_ = true;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_stamped_subscription_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  rclcpp::TimerBase::SharedPtr resend_timer_;

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
