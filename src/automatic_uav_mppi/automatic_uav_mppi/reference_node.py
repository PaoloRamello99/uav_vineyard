#!/usr/bin/env python3
"""
Mission reference publisher (NED-FRD convention).

Publishes StateReference messages on /command/state_reference
for the MPPI rate controller node.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node

from automatic_uav_mppi.coordinates_conversion.Enu2Ned import Enu2NedConverter
from automatic_uav_mppi.mission.mission_reference import SerpentineMission
from quadrotor_msgs.msg import StateReference


class ReferenceNode(Node):
    def __init__(self):
        super().__init__("reference_node")

        # ── Mission parameters ──
        self.declare_parameter("mission.home", [0.0, 0.0, 0.0])
        self.declare_parameter("mission.first_row", [-10.0, 20.0])
        self.declare_parameter("mission.altitude", 2.5)
        self.declare_parameter("mission.row_length", 20.0)
        self.declare_parameter("mission.row_spacing", 2.5)
        self.declare_parameter("mission.num_rows", 10)
        self.declare_parameter("mission.v_ref", 1.0)
        self.declare_parameter("mission.T_takeoff", 4.0)
        self.declare_parameter("mission.T_landing", 4.0)
        self.declare_parameter("mission.T_settle", 3.0)

        # ── Controller-matching parameters ──
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("horizon", 25)
        self.declare_parameter("dt", 0.02)

        home = list(
            self.get_parameter("mission.home").get_parameter_value().double_array_value
        )
        first_row = list(
            self.get_parameter("mission.first_row")
            .get_parameter_value()
            .double_array_value
        )
        altitude = (
            self.get_parameter("mission.altitude").get_parameter_value().double_value
        )
        row_length = (
            self.get_parameter("mission.row_length").get_parameter_value().double_value
        )
        row_spacing = (
            self.get_parameter("mission.row_spacing").get_parameter_value().double_value
        )
        num_rows = (
            self.get_parameter("mission.num_rows").get_parameter_value().integer_value
        )
        v_ref = self.get_parameter("mission.v_ref").get_parameter_value().double_value
        T_takeoff = (
            self.get_parameter("mission.T_takeoff").get_parameter_value().double_value
        )
        T_landing = (
            self.get_parameter("mission.T_landing").get_parameter_value().double_value
        )
        T_settle = (
            self.get_parameter("mission.T_settle").get_parameter_value().double_value
        )

        publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        self.horizon = self.get_parameter("horizon").get_parameter_value().integer_value
        self.dt = self.get_parameter("dt").get_parameter_value().double_value

        self.mission = SerpentineMission(
            home=np.array(home),
            first_row=np.array(first_row),
            altitude=altitude,
            row_length=row_length,
            row_spacing=row_spacing,
            num_rows=num_rows,
            v_ref=v_ref,
            T_takeoff=T_takeoff,
            T_landing=T_landing,
            T_settle=T_settle,
        )

        # ── Publisher ──
        self.ref_pub = self.create_publisher(
            StateReference, "/command/state_reference", 10
        )

        self.t_start = None
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)

        self.get_logger().info("Reference node ready")
        self.get_logger().info(
            f"Mission phases: takeoff={self.mission.t1:.1f}s, "
            f"transit={self.mission.t2:.1f}s, serpentine={self.mission.t3:.1f}s, "
            f"return={self.mission.t4:.1f}s, settle={self.mission.t5:.1f}s, "
            f"land={self.mission.t6:.1f}s"
        )

    def timer_callback(self):
        if self.t_start is None:
            self.t_start = self.get_clock().now().nanoseconds * 1e-9

        t_now = self.get_clock().now().nanoseconds * 1e-9 - self.t_start

        # ── Determine UAV state label ──
        if t_now < self.mission.t1:
            uav_state = "TAKEOFF"
        elif t_now > self.mission.t6:
            uav_state = "HOLD"
        else:
            uav_state = ""

        # ── Build StateReference (NED-FRD) ──
        msg = StateReference()
        msg.uav_state.data = uav_state

        for k in range(self.horizon):
            t_k = t_now + k * self.dt
            ref_enu = self.mission.get_reference(t_k)
            ref_ned = Enu2NedConverter.enu_to_ned(ref_enu)

            pose = Pose()
            pose.position.x = float(ref_ned[0])
            pose.position.y = float(ref_ned[1])
            pose.position.z = float(ref_ned[2])
            pose.orientation.w = float(ref_ned[6])
            pose.orientation.x = float(ref_ned[7])
            pose.orientation.y = float(ref_ned[8])
            pose.orientation.z = float(ref_ned[9])
            msg.poses.append(pose)

            twist = Twist()
            twist.linear.x = float(ref_ned[3])
            twist.linear.y = float(ref_ned[4])
            twist.linear.z = float(ref_ned[5])
            twist.angular.x = float(ref_ned[10])
            twist.angular.y = float(ref_ned[11])
            twist.angular.z = float(ref_ned[12])
            msg.twists.append(twist)

        self.ref_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ReferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
