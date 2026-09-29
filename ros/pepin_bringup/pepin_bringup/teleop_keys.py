"""ROS 2 node: drive by the arrow keys from a laptop terminal, on /cmd_vel.

Runs on the LAPTOP, in the pepin-vslam container (ros/teleop.sh): a new process on the board
stalls the robot's link for 3-4 s, one here does not. The keys are :mod:`pepin.teleop`'s: a key
latches its twist (a terminal gives no key-up), Up/Down drive and Left/Right turn at full
speed, Shift+arrow at the slow one, space stops. The latched twist is republished at 10 Hz,
because the base's deadman stops the wheels 0.5 s after the last command. Ctrl-C sends a zero
twist and exits.
"""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist as TwistMsg
from rclpy.signals import SignalHandlerOptions

from pepin.kinematics import Twist
from pepin.teleop import HELP, DriveState, KeyReader, apply_key

PUBLISH_HZ = 10.0
STOP_REPEATS = 3  # the zero twist on exit goes out a few times: one message may be lost


def to_msg(twist: Twist) -> TwistMsg:
    """The ROS message for a body twist: forward speed in m/s, yaw rate in rad/s."""
    msg = TwistMsg()
    msg.linear.x = twist.linear
    msg.angular.z = twist.angular
    return msg


def main(args: list[str] | None = None) -> None:
    """Read the keys and publish the latched twist at 10 Hz until Ctrl-C, then stop the wheels."""
    # Python's own SIGINT handler: rclpy's would shut the context down before the zero twist.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("teleop_keys")
    publisher = node.create_publisher(TwistMsg, "/cmd_vel", 10)
    state = DriveState()
    print(HELP, flush=True)
    try:
        with KeyReader() as keys:
            while True:
                while (key := keys.read()) is not None:
                    state = apply_key(state, key)
                publisher.publish(to_msg(state.twist))
                time.sleep(1.0 / PUBLISH_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(STOP_REPEATS):
            publisher.publish(TwistMsg())
            time.sleep(0.05)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
