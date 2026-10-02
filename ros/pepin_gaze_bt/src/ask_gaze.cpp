// AskGaze: the behaviour tree's door to the gaze arbiter (pepin_bringup.gaze).
//
// One std_srvs/Trigger call to `service_name`, waited for exactly as long as the arbiter takes
// to answer it (the look's saccade, its still frames, the way home), never longer than
// `timeout_s`. Every decision is the arbiter's; this node only carries the answer: SUCCESS when
// the arbiter says so, FAILURE when it answers success=false ("back off first"), and SUCCESS
// whenever there is no answer to carry (the arbiter is not running, or did not answer in time),
// so a tree with AskGaze in it drives exactly as the tree without it when the arbiter is away.
// A halt (the goal cancelled mid-look) forgets the pending call.

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/bt_factory.h"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace pepin_gaze_bt
{

using Trigger = std_srvs::srv::Trigger;

class AskGaze : public BT::StatefulActionNode
{
public:
  AskGaze(const std::string & name, const BT::NodeConfig & config)
  : BT::StatefulActionNode(name, config)
  {
    node_ = config.blackboard->get<rclcpp::Node::SharedPtr>("node");
    // Its own group on its own executor, as Nav2's service nodes do: the answer is taken here,
    // on the tree's thread, and never by bt_navigator's executor.
    group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive, false);
    executor_.add_callback_group(group_, node_->get_node_base_interface());
    // The client is made with the tree, not at the first stall: by then discovery has had the
    // whole drive to find the arbiter, and service_is_ready() answers for what is there.
    connect(getInput<std::string>("service_name").value_or("/gaze/stall_look"));
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>(
        "service_name", "/gaze/stall_look", "the arbiter's std_srvs/Trigger service"),
      BT::InputPort<double>(
        "timeout_s", 10.0, "the longest this node waits for the answer, seconds"),
    };
  }

  BT::NodeStatus onStart() override
  {
    connect(getInput<std::string>("service_name").value_or("/gaze/stall_look"));
    if (!client_->service_is_ready()) {
      RCLCPP_INFO(
        node_->get_logger(), "AskGaze: %s is not up; the tree goes on without a look",
        service_.c_str());
      return BT::NodeStatus::SUCCESS;
    }
    timeout_ = rclcpp::Duration::from_seconds(getInput<double>("timeout_s").value_or(10.0));
    started_ = node_->now();
    auto sent = client_->async_send_request(std::make_shared<Trigger::Request>());
    pending_.emplace(sent.future.share(), sent.request_id);
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    const auto done = executor_.spin_until_future_complete(
      pending_->future, std::chrono::milliseconds(1));
    if (done == rclcpp::FutureReturnCode::SUCCESS) {
      const auto answer = pending_->future.get();
      pending_.reset();
      RCLCPP_INFO(
        node_->get_logger(), "AskGaze %s: %s", answer->success ? "SUCCESS" : "FAILURE",
        answer->message.c_str());
      return answer->success ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    }
    if (node_->now() - started_ > timeout_) {
      forget();
      RCLCPP_WARN(
        node_->get_logger(), "AskGaze: no answer from %s in %.1f s; the tree goes on",
        service_.c_str(), timeout_.seconds());
      return BT::NodeStatus::SUCCESS;
    }
    return BT::NodeStatus::RUNNING;
  }

  void onHalted() override {forget();}

private:
  void connect(const std::string & service)
  {
    if (!client_ || service != service_) {
      service_ = service;
      client_ = node_->create_client<Trigger>(service_, rclcpp::ServicesQoS(), group_);
    }
  }

  void forget()
  {
    if (pending_) {
      client_->remove_pending_request(pending_->request_id);
      pending_.reset();
    }
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  rclcpp::Client<Trigger>::SharedPtr client_;
  std::string service_;
  std::optional<rclcpp::Client<Trigger>::SharedFutureAndRequestId> pending_;
  rclcpp::Time started_;
  rclcpp::Duration timeout_{0, 0};
};

}  // namespace pepin_gaze_bt

BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<pepin_gaze_bt::AskGaze>("AskGaze");
}
