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
// The same process optionally reads an MPU6050 on the board's I2C bus and publishes
// imu/data_raw (a third thread, same reasoning: a blocking bus must not sit in the
// executor). Wheel odometry over-reports a turn in place by 10-25% on carpet; the fix is
// a gyro yaw rate fused with the wheels in an EKF, configured outside this node. The IMU
// is optional in the strong sense: nothing about it can stop the wheels from working.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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

#include "pepin_base_cpp/link.hpp"
#include "pepin_base_cpp/mpu6050.hpp"
#include "pepin_base_cpp/protocol.hpp"

namespace pepin
{

constexpr double kStatusHz = 2.0;  // how often link up/down transitions are logged

// What the EKF is told to believe about the MPU6050. Both are honest over-estimates: the
// gyro's own noise is far below 0.02 rad/s, but the chip rides an unisolated chassis.
constexpr double kGyroStdDev = 0.02;   // rad/s
constexpr double kAccelStdDev = 0.5;   // m/s^2
constexpr double kRadToDeg = 57.29577951308232;

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
    const bool imu_enable = declare_parameter<bool>("imu_enable", false);
    imu_device_ = declare_parameter<std::string>("imu_device", "/dev/i2c-2");
    imu_address_ = static_cast<int>(declare_parameter<int>("imu_address", 0x68));
    imu_rate_hz_ = declare_parameter<double>("imu_rate_hz", 50.0);
    imu_frame_ = declare_parameter<std::string>("imu_frame", "imu_link");
    imu_bias_s_ = declare_parameter<double>("imu_bias_s", 2.0);

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
    odom.twist.twist.linear.x = state.v;  // the body frame: x forward, yaw counter-clockwise
    odom.twist.twist.angular.z = state.w;
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

  /// Say it once whenever the link comes up or goes down.
  void log_link_status()
  {
    const auto change = link_->take_status_change();
    if (!change.has_value()) {
      return;
    }
    if (change->first) {
      RCLCPP_INFO(get_logger(), "%s", change->second.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "%s", change->second.c_str());
    }
  }

  /// Open the IMU and start sampling it; a missing chip is a warning, not a failure.
  void start_imu()
  {
    std::string error;
    if (!imu_.open_device(imu_device_, imu_address_, imu_rate_hz_, error)) {
      RCLCPP_WARN(get_logger(), "no IMU (%s): the bridge runs on wheel odometry", error.c_str());
      return;
    }
    RCLCPP_INFO(
      get_logger(), "IMU on %s at %d Hz, WHO_AM_I 0x%02x", imu_device_.c_str(),
      static_cast<int>(imu_rate_hz_), static_cast<unsigned>(imu_.who_am_i()));
    imu_publisher_ = create_publisher<sensor_msgs::msg::Imu>("imu/data_raw", 10);
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

  /// IMU thread: estimate the gyro bias while the cart stands still, then publish forever.
  ///
  /// Nothing is published during the bias window on purpose — a raw yaw rate offset fed to
  /// the EKF exactly while it initialises is worse than no measurement at all.
  void read_imu()
  {
    const auto tick = period(1.0 / imu_rate_hz_);
    auto next = std::chrono::steady_clock::now();
    const auto bias_until = next + period(imu_bias_s_);
    double bias_x = 0.0;
    double bias_y = 0.0;
    double bias_z = 0.0;
    long samples = 0;
    bool calibrating = imu_bias_s_ > 0.0;
    if (calibrating) {
      RCLCPP_INFO(get_logger(), "gyro bias: hold still for %.1f s", imu_bias_s_);
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
      if (calibrating) {
        bias_x += sample->gyro_x;
        bias_y += sample->gyro_y;
        bias_z += sample->gyro_z;
        ++samples;
        if (std::chrono::steady_clock::now() < bias_until) {
          continue;
        }
        calibrating = false;
        if (samples > 0) {
          bias_x /= static_cast<double>(samples);
          bias_y /= static_cast<double>(samples);
          bias_z /= static_cast<double>(samples);
        }
        RCLCPP_INFO(
          get_logger(), "gyro bias %.3f %.3f %.3f deg/s over %ld samples",
          bias_x * kRadToDeg, bias_y * kRadToDeg, bias_z * kRadToDeg, samples);
        continue;
      }
      publish_imu(*sample, bias_x, bias_y, bias_z);
    }
  }

  /// One conversion as sensor_msgs/Imu, gyro bias removed, no orientation claimed.
  void publish_imu(const ImuSample & sample, double bias_x, double bias_y, double bias_z)
  {
    sensor_msgs::msg::Imu message;
    message.header.stamp = now();
    message.header.frame_id = imu_frame_;
    message.orientation_covariance[0] = -1.0;  // the ROS way to say "no orientation here"
    message.angular_velocity.x = sample.gyro_x - bias_x;
    message.angular_velocity.y = sample.gyro_y - bias_y;
    message.angular_velocity.z = sample.gyro_z - bias_z;
    message.linear_acceleration.x = sample.accel_x;
    message.linear_acceleration.y = sample.accel_y;
    message.linear_acceleration.z = sample.accel_z;
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
  int imu_address_ = 0x68;
  double imu_rate_hz_ = 50.0;
  double imu_bias_s_ = 2.0;
  std::array<double, 36> pose_covariance_{};
  std::array<double, 36> twist_covariance_{};

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
