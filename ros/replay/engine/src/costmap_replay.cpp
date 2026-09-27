// costmap_replay: Nav2's own two costmaps stepped through a prepared drive, in bag time, as fast
// as the CPU goes, and the same output for the same input every time.
//
// WHY NOT `ros2 bag play --clock --rate N` INTO THE STOCK NODES. Costmap2DROS::mapUpdateLoop
// paces updateMap() with an rclcpp::WallRate (libnav2_costmap_2d_core.so imports
// rclcpp::WallRate::WallRate(double) and nothing else wall-paced), so at N times real time the
// costmap updates N times less often per second of the drive: every update folds N times more
// scans into one, clears N times fewer rays, and the answer depends on the rate and on thread
// scheduling. A diff between two parameter sets would carry that jitter as signal.
//
// WHAT THIS DOES INSTEAD. The costmaps are the stock nav2_costmap_2d::Costmap2DROS nodes of the
// board's own image, configured from the same parameter files (update_frequency 0.0, so no loop
// thread runs at all), and this program is their clock and their loop:
//   * bag time is their ROS time (use_sim_time, the override set by hand, no /clock);
//   * /tf and /tf_static go straight into each costmap's tf2 buffer;
//   * every observation goes to the ObstacleLayer source that subscribes to its topic, through
//     the layer's own public callback (laserScanCallback, or laserScanValidInfCallback where the
//     source says inf_is_valid) into that source's own ObservationBuffer — exactly where the
//     layer's MessageFilter would have delivered it: as soon as the global frame resolves at the
//     message's stamp, dropped once transform_tolerance of bag time has passed without it;
//   * a StaticLayer gets its map through its own incomingMap;
//   * /replay/clear ("local" | "global") is ClearEntireCostmap: Costmap2DROS::resetLayers();
//   * updateMap() runs at the live update_frequency of each costmap, in bag time.
// After every update the costmap's cost array is appended to the output (and, for the costmap
// named by --audit, the newest scan returns of the --lidar and --camera topics placed in its
// frame at their own stamps), for ros/replay/score.py to read.
//
// Usage (ros/replay.sh builds and runs it):
//   costmap_replay --bag DIR --out FILE --local-hz 5 --global-hz 2 [--clears on|off]
//       [--audit local_costmap] [--lidar /scan] [--camera /depth_marks]
//       --ros-args --params-file ... --params-file ...
// It prints one JSON line of counts (messages, delivered, dropped by the TF wait, updates,
// clears, wall_s) on stdout at the end.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/obstacle_layer.hpp"
#include "nav2_costmap_2d/static_layer.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rcl/time.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/serialization.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/LinearMath/Transform.h"
#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace
{

using nav2_costmap_2d::Costmap2DROS;
using nav2_costmap_2d::ObservationBuffer;
using nav2_costmap_2d::ObstacleLayer;
using nav2_costmap_2d::StaticLayer;
using sensor_msgs::msg::LaserScan;

// The two members the stock layers keep protected, reached the standard way (a pointer to a
// member named through a derived class): the source buffers in observation_sources order, and
// the map callback. Nothing is overridden; these classes are never instantiated.
struct ObstaclePeek : ObstacleLayer
{
  static std::vector<std::shared_ptr<ObservationBuffer>> & buffers(ObstacleLayer & layer)
  {
    return layer.*(&ObstaclePeek::observation_buffers_);
  }
};

struct StaticPeek : StaticLayer
{
  static void map(StaticLayer & layer, nav_msgs::msg::OccupancyGrid::SharedPtr grid)
  {
    (layer.*(&StaticPeek::incomingMap))(std::move(grid));
  }
};

/// One ObstacleLayer source: the buffer its topic's messages go into, and how they get there.
struct Source
{
  std::size_t costmap;
  std::string layer;
  ObstacleLayer * obstacle;
  std::shared_ptr<ObservationBuffer> buffer;
  bool inf_is_valid;
};

/// A message the layer's MessageFilter would still be holding: its frame did not resolve yet.
struct Pending
{
  LaserScan::SharedPtr scan;
  const Source * source;
  rcl_time_point_value_t deadline;
};

/// One costmap under replay: the stock node, its update period and the queue of its filters.
struct Costmap
{
  std::string name;
  std::shared_ptr<Costmap2DROS> ros;
  double period_s;
  double tolerance_s;
  bool active{false};
  rcl_time_point_value_t next_tick{0};
  std::deque<Pending> pending;
};

struct Counters
{
  std::uint64_t messages{0};
  std::uint64_t delivered{0};
  std::uint64_t waited{0};
  std::uint64_t dropped_timeout{0};
  std::uint64_t dropped_unrouted{0};
  std::uint64_t updates{0};
  std::uint64_t clears{0};
};

rcl_time_point_value_t stamp_ns(const builtin_interfaces::msg::Time & t)
{
  return static_cast<rcl_time_point_value_t>(t.sec) * 1000000000LL + t.nanosec;
}

/// Bag time becomes the node's ROS time: the override use_sim_time turned on, set by hand.
void set_time(Costmap2DROS & node, rcl_time_point_value_t ns)
{
  auto clock = node.get_clock();
  std::lock_guard<std::mutex> guard(clock->get_clock_mutex());
  rcl_clock_t * handle = clock->get_clock_handle();
  bool enabled = false;
  if (rcl_is_enabled_ros_time_override(handle, &enabled) != RCL_RET_OK ||
    (!enabled && rcl_enable_ros_time_override(handle) != RCL_RET_OK) ||
    rcl_set_ros_time_override(handle, ns) != RCL_RET_OK)
  {
    throw std::runtime_error("cannot set " + std::string(node.get_name()) + "'s ROS time");
  }
}

std::string string_param(Costmap2DROS & node, const std::string & name, const std::string & fallback)
{
  std::string value = fallback;
  if (node.has_parameter(name)) {
    value = node.get_parameter(name).as_string();
  }
  return value;
}

bool bool_param(Costmap2DROS & node, const std::string & name, bool fallback)
{
  bool value = fallback;
  if (node.has_parameter(name)) {
    value = node.get_parameter(name).as_bool();
  }
  return value;
}

double double_param(Costmap2DROS & node, const std::string & name, double fallback)
{
  double value = fallback;
  if (node.has_parameter(name)) {
    value = node.get_parameter(name).as_double();
  }
  return value;
}

/// Write-side of the snapshot file ros/replay/score.py reads (little-endian, packed):
///   'C' u8 costmap, f64 t, f64 robot x, y, yaw, f64 origin x, y, f64 resolution,
///       u32 size_x, u32 size_y, then size_x * size_y u8 costs, row-major from the origin;
///   'P' u8 sensor (0 lidar, 1 camera), f64 t, u32 n, then n * (f32 x, f32 y) in that frame.
class SnapshotWriter
{
public:
  explicit SnapshotWriter(const std::string & path)
  : out_(path, std::ios::binary) {}

  bool good() const {return out_.good();}

  void costmap(std::uint8_t index, double t, const geometry_msgs::msg::PoseStamped & pose,
    nav2_costmap_2d::Costmap2D & grid)
  {
    const double values[6] = {
      t, pose.pose.position.x, pose.pose.position.y, tf2::getYaw(pose.pose.orientation),
      grid.getOriginX(), grid.getOriginY()};
    const std::uint32_t size[2] = {grid.getSizeInCellsX(), grid.getSizeInCellsY()};
    const double resolution = grid.getResolution();
    put('C');
    put(index);
    out_.write(reinterpret_cast<const char *>(values), sizeof(values));
    out_.write(reinterpret_cast<const char *>(&resolution), sizeof(resolution));
    out_.write(reinterpret_cast<const char *>(size), sizeof(size));
    out_.write(reinterpret_cast<const char *>(grid.getCharMap()), size[0] * size[1]);
  }

  void points(std::uint8_t sensor, double t, const std::vector<float> & xy)
  {
    const std::uint32_t n = static_cast<std::uint32_t>(xy.size() / 2);
    put('P');
    put(sensor);
    out_.write(reinterpret_cast<const char *>(&t), sizeof(t));
    out_.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out_.write(reinterpret_cast<const char *>(xy.data()), xy.size() * sizeof(float));
  }

private:
  void put(std::uint8_t byte) {out_.write(reinterpret_cast<const char *>(&byte), 1);}

  std::ofstream out_;
};

/// A scan's returns (finite, within its own range_min..range_max: pepin_bringup.msgs.scan_arrays)
/// placed in ``frame`` through TF at the scan's own stamp into ``xy``; false when TF cannot.
bool placed_returns(
  const LaserScan & scan, tf2_ros::Buffer & tf, const std::string & frame, std::vector<float> & xy)
{
  geometry_msgs::msg::TransformStamped placement;
  try {
    placement = tf.lookupTransform(frame, scan.header.frame_id, tf2::TimePoint(
          std::chrono::nanoseconds(stamp_ns(scan.header.stamp))));
  } catch (const tf2::TransformException &) {
    return false;
  }
  tf2::Transform transform;
  tf2::fromMsg(placement.transform, transform);
  xy.reserve(scan.ranges.size() * 2);
  for (std::size_t i = 0; i < scan.ranges.size(); ++i) {
    const float r = scan.ranges[i];
    if (!std::isfinite(r) || r < scan.range_min || r > scan.range_max) {
      continue;
    }
    const double a = scan.angle_min + static_cast<double>(i) * scan.angle_increment;
    const tf2::Vector3 p = transform * tf2::Vector3(r * std::cos(a), r * std::sin(a), 0.0);
    xy.push_back(static_cast<float>(p.x()));
    xy.push_back(static_cast<float>(p.y()));
  }
  return true;
}

struct Options
{
  std::string bag;
  std::string out;
  double local_hz{5.0};
  double global_hz{2.0};
  std::string audit{"local_costmap"};
  std::string lidar{"/scan"};
  std::string camera{"/depth_marks"};
  bool clears{true};
};

Options parse(const std::vector<std::string> & args)
{
  Options o;
  if (args.size() % 2 == 0) {
    throw std::runtime_error("arguments come in --key value pairs");
  }
  for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
    const std::string & key = args[i];
    const std::string & value = args[i + 1];
    if (key == "--bag") {
      o.bag = value;
    } else if (key == "--out") {
      o.out = value;
    } else if (key == "--local-hz") {
      o.local_hz = std::stod(value);
    } else if (key == "--global-hz") {
      o.global_hz = std::stod(value);
    } else if (key == "--audit") {
      o.audit = value;
    } else if (key == "--lidar") {
      o.lidar = value;
    } else if (key == "--camera") {
      o.camera = value;
    } else if (key == "--clears") {
      o.clears = value != "off";
    } else {
      throw std::runtime_error("unknown argument " + key);
    }
  }
  if (o.bag.empty() || o.out.empty()) {
    throw std::runtime_error("--bag and --out are required");
  }
  return o;
}

class Replay
{
public:
  explicit Replay(Options options)
  : options_(std::move(options)), writer_(options_.out)
  {
    if (!writer_.good()) {
      throw std::runtime_error("cannot write " + options_.out);
    }
    add("local_costmap", options_.local_hz);
    add("global_costmap", options_.global_hz);
  }

  void run()
  {
    rosbag2_cpp::Reader reader;
    reader.open(options_.bag);
    std::map<std::string, std::string> types;
    for (const auto & topic : reader.get_all_topics_and_types()) {
      types[topic.name] = topic.type;
    }
    rclcpp::Serialization<tf2_msgs::msg::TFMessage> tf_serde;
    rclcpp::Serialization<LaserScan> scan_serde;
    rclcpp::Serialization<nav_msgs::msg::OccupancyGrid> map_serde;
    rclcpp::Serialization<std_msgs::msg::String> string_serde;
    const auto wall_start = std::chrono::steady_clock::now();
    bool first = true;
    rcl_time_point_value_t previous = 0;
    while (reader.has_next()) {
      auto bag_message = reader.read_next();
      const rcl_time_point_value_t t = bag_message->recv_timestamp;
      if (first) {
        first = false;
        for (auto & c : costmaps_) {
          c.next_tick = t;
        }
      } else if (t < previous) {
        throw std::runtime_error("the bag is not in receive order at " + bag_message->topic_name);
      }
      previous = t;
      advance(t);
      ++counters_.messages;
      rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
      const std::string & topic = bag_message->topic_name;
      const std::string & type = types[topic];
      if (type == "tf2_msgs/msg/TFMessage") {
        tf2_msgs::msg::TFMessage tf;
        tf_serde.deserialize_message(&serialized, &tf);
        transforms(tf, topic == "/tf_static");
      } else if (type == "sensor_msgs/msg/LaserScan") {
        auto scan = std::make_shared<LaserScan>();
        scan_serde.deserialize_message(&serialized, scan.get());
        observe(topic, scan, t);
      } else if (type == "nav_msgs/msg/OccupancyGrid") {
        auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
        map_serde.deserialize_message(&serialized, grid.get());
        map(topic, grid);
      } else if (topic == "/replay/clear" && options_.clears) {
        std_msgs::msg::String which;
        string_serde.deserialize_message(&serialized, &which);
        clear(which.data);
      }
    }
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "{\"messages\": " << counters_.messages << ", \"delivered\": " << counters_.delivered <<
      ", \"waited_for_tf\": " << counters_.waited << ", \"dropped_tf_timeout\": " <<
      counters_.dropped_timeout << ", \"unrouted\": " << counters_.dropped_unrouted <<
      ", \"updates\": " << counters_.updates << ", \"clears\": " << counters_.clears <<
      ", \"wall_s\": " << wall_s << "}" << std::endl;
  }

  void shutdown()
  {
    for (auto & c : costmaps_) {
      if (c.active) {
        c.ros->deactivate();
      }
      c.ros->cleanup();
      c.ros->shutdown();
    }
  }

private:
  void add(const std::string & name, double hz)
  {
    Costmap c;
    c.name = name;
    c.ros = std::make_shared<Costmap2DROS>(name, "/", name, true);
    c.period_s = hz > 0.0 ? 1.0 / hz : 0.0;
    set_time(*c.ros, 0);
    c.ros->configure();
    c.tolerance_s = double_param(*c.ros, "transform_tolerance", 0.3);
    routes(costmaps_.size(), *c.ros);
    costmaps_.push_back(std::move(c));
  }

  /// Every ObstacleLayer source and StaticLayer of one costmap, by the topic it subscribes to.
  void routes(std::size_t index, Costmap2DROS & ros)
  {
    for (const auto & layer : *ros.getLayeredCostmap()->getPlugins()) {
      const std::string name = layer->getName();
      if (auto obstacle = std::dynamic_pointer_cast<ObstacleLayer>(layer)) {
        std::istringstream sources(string_param(ros, name + ".observation_sources", ""));
        auto & buffers = ObstaclePeek::buffers(*obstacle);
        std::size_t i = 0;
        for (std::string source; sources >> source; ++i) {
          const std::string prefix = name + "." + source + ".";
          if (string_param(ros, prefix + "data_type", "LaserScan") != "LaserScan") {
            continue;  // PointCloud2 sources: none on this robot
          }
          const std::string topic = string_param(ros, prefix + "topic", source);
          sources_[topic].push_back(Source{index, name, obstacle.get(), buffers.at(i),
              bool_param(ros, prefix + "inf_is_valid", false)});
        }
      } else if (auto stat = std::dynamic_pointer_cast<StaticLayer>(layer)) {
        statics_[string_param(ros, name + ".map_topic", "map")].push_back(stat.get());
      }
    }
  }

  /// Runs every costmap tick due before bag time ``t``, then sets the clocks to ``t``.
  void advance(rcl_time_point_value_t t)
  {
    for (;;) {
      Costmap * due = nullptr;
      for (auto & c : costmaps_) {
        if (c.active && c.period_s > 0.0 && c.next_tick <= t &&
          (due == nullptr || c.next_tick < due->next_tick))
        {
          due = &c;
        }
      }
      if (due == nullptr) {
        break;
      }
      tick(*due);
    }
    now_ = t;
    for (auto & c : costmaps_) {
      set_time(*c.ros, t);
      expire(c, t);
    }
  }

  void tick(Costmap & c)
  {
    const rcl_time_point_value_t t = c.next_tick;
    for (auto & other : costmaps_) {
      set_time(*other.ros, t);
      expire(other, t);
    }
    c.ros->updateMap();
    ++counters_.updates;
    geometry_msgs::msg::PoseStamped pose;
    if (c.ros->getRobotPose(pose)) {
      const std::uint8_t index = static_cast<std::uint8_t>(&c - costmaps_.data());
      writer_.costmap(index, t * 1e-9, pose, *c.ros->getCostmap());
      if (c.name == options_.audit) {
        // The newest scan of each audited topic that TF can place at its own stamp (the very
        // newest often cannot yet: its odom -> base_link arrives a few ms after it).
        const std::string frame = c.ros->getGlobalFrameID();
        const std::pair<std::uint8_t, std::string> audited[] = {{0, options_.lidar}, {1, options_.camera}};
        for (const auto & [sensor, topic] : audited) {
          const auto & recent = recent_[topic];
          for (auto scan = recent.rbegin(); scan != recent.rend(); ++scan) {
            std::vector<float> xy;
            if (placed_returns(**scan, *c.ros->getTfBuffer(), frame, xy)) {
              writer_.points(sensor, t * 1e-9, xy);
              break;
            }
          }
        }
      }
    }
    c.next_tick += static_cast<rcl_time_point_value_t>(c.period_s * 1e9);
  }

  void transforms(const tf2_msgs::msg::TFMessage & tf, bool is_static)
  {
    for (auto & c : costmaps_) {
      for (const auto & transform : tf.transforms) {
        c.ros->getTfBuffer()->setTransform(transform, "replay", is_static);
      }
      if (!c.active) {
        activate(c);
      }
      retry(c);
    }
  }

  /// Activates a costmap once its global frame resolves to the robot. on_activate() waits for
  /// the first updateMap() the stock loop thread would run, so this thread runs it meanwhile.
  void activate(Costmap & c)
  {
    std::string error;
    if (!c.ros->getTfBuffer()->canTransform(c.ros->getGlobalFrameID(), c.ros->getBaseFrameID(),
      tf2::TimePointZero, &error))
    {
      return;
    }
    std::atomic<bool> done{false};
    std::thread activation([&]() {c.ros->activate(); done = true;});
    while (!done) {
      c.ros->updateMap();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    activation.join();
    c.active = true;
    while (c.next_tick < now_) {  // the first tick on the drive's own grid of update times
      c.next_tick += static_cast<rcl_time_point_value_t>(c.period_s * 1e9);
    }
  }

  void observe(const std::string & topic, const LaserScan::SharedPtr & scan, rcl_time_point_value_t t)
  {
    auto & recent = recent_[topic];
    recent.push_back(scan);
    if (recent.size() > kRecentScans) {
      recent.pop_front();
    }
    auto found = sources_.find(topic);
    if (found == sources_.end()) {
      ++counters_.dropped_unrouted;
      return;
    }
    for (const auto & source : found->second) {
      Costmap & c = costmaps_[source.costmap];
      if (!deliver(c, scan, source)) {
        ++counters_.waited;
        c.pending.push_back(Pending{scan, &source,
            t + static_cast<rcl_time_point_value_t>(c.tolerance_s * 1e9)});
      }
    }
  }

  /// The MessageFilter's test: the costmap's global frame resolves at the message's stamp.
  bool deliver(Costmap & c, const LaserScan::SharedPtr & scan, const Source & source)
  {
    if (!c.active) {
      return false;
    }
    const tf2::TimePoint stamp(std::chrono::nanoseconds(stamp_ns(scan->header.stamp)));
    if (!c.ros->getTfBuffer()->canTransform(c.ros->getGlobalFrameID(), scan->header.frame_id, stamp)) {
      return false;
    }
    if (source.inf_is_valid) {
      source.obstacle->laserScanValidInfCallback(scan, source.buffer);
    } else {
      source.obstacle->laserScanCallback(scan, source.buffer);
    }
    ++counters_.delivered;
    return true;
  }

  void retry(Costmap & c)
  {
    std::deque<Pending> still;
    for (auto & p : c.pending) {
      if (!deliver(c, p.scan, *p.source)) {
        still.push_back(p);
      }
    }
    c.pending.swap(still);
  }

  void expire(Costmap & c, rcl_time_point_value_t t)
  {
    while (!c.pending.empty() && c.pending.front().deadline < t) {
      c.pending.pop_front();
      ++counters_.dropped_timeout;
    }
  }

  void map(const std::string & topic, const nav_msgs::msg::OccupancyGrid::SharedPtr & grid)
  {
    auto found = statics_.find(topic);
    if (found == statics_.end()) {
      ++counters_.dropped_unrouted;
      return;
    }
    for (StaticLayer * layer : found->second) {
      StaticPeek::map(*layer, grid);
    }
  }

  void clear(const std::string & which)
  {
    for (auto & c : costmaps_) {
      if (c.active && c.name.rfind(which, 0) == 0) {
        c.ros->resetLayers();
        ++counters_.clears;
      }
    }
  }

  Options options_;
  SnapshotWriter writer_;
  std::vector<Costmap> costmaps_;
  std::map<std::string, std::vector<Source>> sources_;
  std::map<std::string, std::vector<StaticLayer *>> statics_;
  static constexpr std::size_t kRecentScans = 10;
  std::map<std::string, std::deque<LaserScan::SharedPtr>> recent_;
  Counters counters_;
  rcl_time_point_value_t now_{0};
};

}  // namespace

int main(int argc, char ** argv)
{
  const std::vector<std::string> args = rclcpp::init_and_remove_ros_arguments(argc, argv);
  int status = 0;
  try {
    Replay replay(parse(args));
    replay.run();
    replay.shutdown();
  } catch (const std::exception & e) {
    std::cerr << "costmap_replay: " << e.what() << std::endl;
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
