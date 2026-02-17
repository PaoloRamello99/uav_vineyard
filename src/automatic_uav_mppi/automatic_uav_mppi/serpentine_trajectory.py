#!/usr/bin/env python3
"""
serpentine_node.py
Geographic (position-based) serpentine trajectory generator
"""

import os
import yaml
import math

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty

from px4_msgs.msg import VehicleLocalPosition
from quadrotor_msgs.msg import StateReference
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped


class UAVState:
    IDLE = 0
    TAKEOFF = 1
    HOLD = 2
    FLIGHT = 3

class SerpentineState:
    STRAIGHT = 10
    TURN = 11
    FINISH = 12

class SerpentineTrajectory(Node):

    def __init__(self):  
        super().__init__("serpentine_trajectory")
        
        # --- Declare Parameters ---
        self.declare_parameter("home", [0.0, 0.0, 0.0])
        self.declare_parameter("first_row", [0.0, 0.0, 3.0])
        self.declare_parameter("initial_velocity", [0.0, 0.0, 0.0])
        self.declare_parameter("height", 3.0)
        self.declare_parameter("row_length", 20.0)
        self.declare_parameter("row_spacing", 2.5)
        self.declare_parameter("num_rows", 4)
        self.declare_parameter("v_ref", 1.0)
        self.declare_parameter("pub_frequency", 50.0)
        self.declare_parameter("trajectory_steps", 50)
        self.declare_parameter("takeoff_threshold", 0.3)
        self.declare_parameter("straight_threshold", 0.3)
        self.declare_parameter("theta_threshold", 0.3)
        self.declare_parameter("hold_time", 3.0)
        self.declare_parameter("orientation", True)
        self.declare_parameter("acc_ref", 1.0)

        # --- Read Parameters into Variables ---
        self.home = np.array(self.get_parameter("home").value, dtype=float)
        self.first_row = np.array(self.get_parameter("first_row").value, dtype=float)
        self.initial_velocity = np.array(self.get_parameter("initial_velocity").value, dtype=float)
        self.height = self.get_parameter("height").value
        self.row_length = self.get_parameter("row_length").value
        self.row_spacing = self.get_parameter("row_spacing").value
        self.num_rows = self.get_parameter("num_rows").value
        self.v_ref = self.get_parameter("v_ref").value
        self.pub_frequency = (self.get_parameter("pub_frequency").get_parameter_value().double_value)
        self.trajectory_steps = (self.get_parameter("trajectory_steps").get_parameter_value().integer_value)
        self.takeoff_threshold = self.get_parameter("takeoff_threshold").value
        self.straight_threshold = self.get_parameter("straight_threshold").value
        self.theta_threshold = self.get_parameter("theta_threshold").value
        self.hold_time = self.get_parameter("hold_time").get_parameter_value().double_value
        self.orientation = self.get_parameter("orientation").value
        self.acc_ref = self.get_parameter("acc_ref").value

        self.row_idx = 0
        self.radius_turn = self.row_spacing / 2
        self.t_start_straight = None
        self.t_start_turn = None
        
        # QoS      
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscribers ---
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos,
        )

        # --- Publishers ---
        self.pose_pub = self.create_publisher(PoseStamped, "/command/ref_pose", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "/command/ref_velocity", 10)
        self.state_ref_pub = self.create_publisher(
            StateReference, "/command/state_reference", 10
        )

        self.srv = self.create_service(
            Empty, "start_serpentine", self.start_serpentine_callback
        )


        # --- State ---
        self.t = 0
        self.flight_state = UAVState.IDLE
        self.current_position = self.home.copy()
        self.current_velocity = self.initial_velocity.copy()
        self.hold_start_time = None

        self.timer_period = 1.0 / self.pub_frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.flight_state = UAVState.TAKEOFF
        self.get_logger().info("Starting in TAKEOFF state automatically")
        
        # --- Precompute geographic serpentine ---
        self.row_idx = 0
        self.serpentine_state = SerpentineState.STRAIGHT


    # ======================================================================
    # CALLBACKS
    # ======================================================================

    def position_callback(self, msg: VehicleLocalPosition):
        self.current_position[0] = msg.x
        self.current_position[1] = -msg.y
        self.current_position[2] = -msg.z

        self.current_velocity[0] = msg.vx
        self.current_velocity[1] = -msg.vy
        self.current_velocity[2] = -msg.vz

        self.get_logger().debug(
            f"Current position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})"
        )

        if self.flight_state == UAVState.TAKEOFF:
            height_difference = abs(self.current_position[2] - self.height)

            if height_difference < self.takeoff_threshold:
                self.get_logger().info(
                    f"Takeoff height reached ({self.current_position[2]:.2f}m)! Transitioning to HOLD mode."
                )
                self.flight_state = UAVState.HOLD
                self.hold_start_time = self.get_clock().now()

    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2
            + (pos1[1] - pos2[1]) ** 2
            + (pos1[2] - pos2[2]) ** 2
        )

    def calculate_orientation(self, current_pos, next_pos):
        """Calculate quaternion to orient in direction of travel"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        yaw = np.arctan2(dy, dx)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        return cy, 0.0, 0.0, sy

    def timer_callback(self):
        """Main timer callback that handles state transitions and publishing"""
        if self.flight_state == UAVState.IDLE:
            pass
        elif self.flight_state == UAVState.TAKEOFF:
            self.publish_takeoff_waypoint()
        elif self.flight_state == UAVState.HOLD:
            self.publish_hold_position()
            self.check_hold_timeout()
        else:
            self.publish_serpentine_trajectory()

    def check_hold_timeout(self):
        """Check if hold time has elapsed and transition to FLIGHT"""
        if self.hold_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.hold_start_time).nanoseconds / 1e9

            if elapsed_time >= self.hold_time:
                self.get_logger().info(
                    f"Hold time of {self.hold_time}s elapsed. Starting serpentine trajectory."
                )
                self.flight_state = UAVState.FLIGHT

    def publish_hold_position(self):
        """Publish hold position at takeoff location"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.home[0])
        pose_msg.pose.position.y = float(self.home[1])
        pose_msg.pose.position.z = float(self.height)

        pose_msg.pose.orientation.w = 1.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0

        self.pose_pub.publish(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = 0.0
        vel_msg.twist.linear.y = 0.0
        vel_msg.twist.linear.z = 0.0
        self.vel_pub.publish(vel_msg)

        state_ref = StateReference()
        state_ref.uav_state = String(data="HOLD")

        for _ in range(self.trajectory_steps):
            pose = Pose()
            pose.position.x = float(self.home[0])
            pose.position.y = float(self.home[1])
            pose.position.z = float(self.height)
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            state_ref.poses.append(pose)

            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            state_ref.twists.append(twist)

        self.state_ref_pub.publish(state_ref)

        if self.hold_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.hold_start_time).nanoseconds / 1e9
            remaining_time = max(0.0, self.hold_time - elapsed_time)

            self.get_logger().debug(
                f"Holding position at ({self.home[0]:.2f}, {self.home[1]:.2f}, {self.height:.2f}), "
                f"remaining time: {remaining_time:.1f}s"
            )

    def publish_takeoff_waypoint(self):
        """Publish the takeoff waypoint using StateReference format"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.home[0])
        pose_msg.pose.position.y = float(self.home[1])
        pose_msg.pose.position.z = float(self.height)

        pose_msg.pose.orientation.w = 1.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0

        self.pose_pub.publish(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = 0.0
        vel_msg.twist.linear.y = 0.0
        vel_msg.twist.linear.z = 0.0
        self.vel_pub.publish(vel_msg)

        state_ref = StateReference()
        state_ref.uav_state = String(data="TAKEOFF")

        for _ in range(self.trajectory_steps):
            pose = Pose()
            pose.position.x = float(self.home[0])
            pose.position.y = float(self.home[1])
            pose.position.z = float(self.height)
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            state_ref.poses.append(pose)

            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            state_ref.twists.append(twist)

        self.state_ref_pub.publish(state_ref)

        dist_to_takeoff = self.calculate_distance(
            self.current_position, [self.home[0], self.home[1], self.height]
        )

        self.get_logger().debug(
            f"Publishing takeoff waypoint: ({self.home[0]:.2f}, {self.home[1]:.2f}, {self.height:.2f}), "
            f"Distance: {dist_to_takeoff:.2f}m"
        )

    def publish_serpentine_trajectory(self):
        """Publish the serpentine trajectory using StateReference format"""
        #x, y, z = self.calculate_serpentine_position(self.t)
        #vx, vy, vz = self.calculate_serpentine_velocity(self.t)

        x, y, z, vx, vy, vz = self.serpentine_reference(self.t)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)

        if self.orientation:
            dt = 0.02
            next_t = self.t + dt
            next_x, next_y, next_z, next_vx, next_vy, next_vz = self.serpentine_reference(next_t)

            w, qx, qy, qz = self.calculate_orientation(
                [x, y, z], [next_x, next_y, next_z]
            )

            pose_msg.pose.orientation.w = float(w)
            pose_msg.pose.orientation.x = float(qx)
            pose_msg.pose.orientation.y = float(qy)
            pose_msg.pose.orientation.z = float(qz)
        else:
            pose_msg.pose.orientation.w = 1.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0

        self.pose_pub.publish(pose_msg)

        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = "map"
        vel_msg.twist.linear.x = float(vx)
        vel_msg.twist.linear.y = float(vy)
        vel_msg.twist.linear.z = float(vz)
        self.vel_pub.publish(vel_msg)

        state_ref = StateReference()
        state_ref.uav_state = String(data="FLIGHT")

        for i in range(self.trajectory_steps):
            future_t = self.t + i * self.timer_period ######

            #future_x, future_y, future_z = self.calculate_serpentine_position(future_t)
            #future_vx, future_vy, future_vz = self.calculate_serpentine_velocity(
            #    future_t)

            future_x, future_y, future_z, future_vx, future_vy, future_vz = self.serpentine_reference(future_t)

            pose = Pose()
            pose.position.x = float(future_x)
            pose.position.y = float(future_y)
            pose.position.z = float(future_z)

            if self.orientation and i > 0:
                orient_t = future_t + 0.01
                orient_x, orient_y, orient_z, orient_vx, orient_vy, orient_vz = self.serpentine_reference(
                    orient_t
                )

                w, qx, qy, qz = self.calculate_orientation(
                    [future_x, future_y, future_z],
                    [orient_x, orient_y, orient_z],
                )

                pose.orientation.w = float(w)
                pose.orientation.x = float(qx)
                pose.orientation.y = float(qy)
                pose.orientation.z = float(qz)
            else:
                pose.orientation.w = 1.0
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0

            state_ref.poses.append(pose)

            twist = Twist()
            twist.linear.x = float(future_vx)
            twist.linear.y = float(future_vy)
            twist.linear.z = float(future_vz)
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0

            state_ref.twists.append(twist)

        self.state_ref_pub.publish(state_ref)

        self.t += self.timer_period #####

        self.get_logger().debug(
            f"Publishing position: ({x:.2f}, {y:.2f}, {z:.2f}), velocity: ({vx:.2f}, {vy:.2f}, {vz:.2f})"
        )

    def start_serpentine_callback(self, request, response):
        """Callback when the start_serpentine service is called"""
        self.get_logger().info("Received request to start serpentine trajectory")

        if self.flight_state == UAVState.IDLE:
            self.flight_state = UAVState.TAKEOFF
            self.get_logger().info("Starting takeoff sequence")
        elif self.flight_state == UAVState.TAKEOFF:
            self.flight_state = UAVState.HOLD
            self.hold_start_time = self.get_clock().now()
            self.get_logger().info("Transitioning from takeoff to hold mode")
        elif self.flight_state == UAVState.HOLD:
            self.flight_state = UAVState.FLIGHT
            self.get_logger().info("Transitioning from hold to serpentine trajectory")
        else:
            self.get_logger().info(
                f"Already in serpentine flight mode: {self.flight_state}"
            )

        return response




    def serpentine_reference(self, t):
        direction = 1 if self.row_idx % 2 == 0 else -1
        
        # --- STATE: STRAIGHT ---
        if self.serpentine_state == SerpentineState.STRAIGHT:
            if self.t_start_straight is None:
                self.t_start_straight = t

            dt_straight = t - self.t_start_straight

            if direction == 1:
                x_start = self.first_row[0]
                x_end = self.first_row[0] + self.row_length
            else:
                x_start = self.first_row[0] + self.row_length
                x_end = self.first_row[0]
            
            vx = direction * self.v_ref
            vy = 0.0
            vz = 0.0

            x = x_start + vx * dt_straight
            y = self.first_row[1] + self.row_idx * self.row_spacing
            z = self.height

            straight_difference = abs(self.current_position[0] - x_end)

            if straight_difference <= self.straight_threshold:
                self.serpentine_state = SerpentineState.TURN
                self.t_start_straight = None
                self.t_start_turn = None
                return x, y, z, vx, vy, vz
            else:
                return x,y,z, vx,vy,vz


        # --- STATE: TURN ---
        elif self.serpentine_state == SerpentineState.TURN:
            if self.t_start_turn is None:
                self.t_start_turn = t
            
            dt_turn = t - self.t_start_turn
            R = self.radius_turn
            omega = self.v_ref / R
            theta = omega * dt_turn

            y_c = self.first_row[1] + self.row_idx * self.row_spacing + R

            if direction == 1:
                x_c = self.first_row[0] + self.row_length
            else:
                x_c = self.first_row[0]

            x = x_c + direction * R * np.sin(theta)
            y = y_c - R * np.cos(theta)
            z = self.height
            vx = direction * self.v_ref * np.cos(theta)
            vy = self.v_ref * np.sin(theta)
            vz = 0.0


            if theta >= np.pi:
                self.row_idx += 1
                if self.row_idx >= self.num_rows:
                    self.serpentine_state = SerpentineState.FINISH
                    return self.current_position[0], self.current_position[1], self.height, 0.0, 0.0, 0.0  
                
                self.serpentine_state = SerpentineState.STRAIGHT
                self.t_start_turn = None
                self.t_start_straight = None
                return x,y,z, vx,vy,vz
            
            else:
                return x,y,z, vx,vy,vz
        
        else:
            self.serpentine_state = SerpentineState.FINISH
            return self.current_position[0], self.current_position[1], self.height, 0.0, 0.0, 0.0

    



def main(args=None):
    rclpy.init(args=args)

    serpentine_node = SerpentineTrajectory()

    try:
        rclpy.spin(serpentine_node)
    except KeyboardInterrupt:
        pass
    finally:
        serpentine_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()