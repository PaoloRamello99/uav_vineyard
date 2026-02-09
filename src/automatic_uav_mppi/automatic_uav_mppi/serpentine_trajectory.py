#!/usr/bin/env python3
"""
serpentine_node.py
Geographic (position-based) serpentine trajectory generator
"""

import os
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from quadrotor_msgs.msg import StateReference


CONFIG_FILE_PATH = "/workspaces/uav_vineyard/src/automatic_uav_mppi/automatic_uav_mppi/config/geometry_vineyard.yaml"


class UAVState:
    IDLE = 0
    TAKEOFF = 1
    HOLD = 2
    SERPENTINE = 3
    MISSION_COMPLETE = 4

class SerpentineState:
    STRAIGHT = 10
    TURN = 11
    FINISH = 12


class SerpentineTrajectory(Node):

    def __init__(self, config):
        super().__init__("serpentine_trajectory")
        self._parse_config(config)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscribers ---
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos,
        )

        # --- Publishers ---
        self.state_ref_pub = self.create_publisher(
            StateReference, "/command/state_reference", 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, "/command/ref_pose", 10
        )
        self.vel_pub = self.create_publisher(
            TwistStamped, "/command/ref_velocity", 10
        )

        # --- Timer ---
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.timer_callback
        )
        self.timer_period = 1.0 / self.publish_rate

        # --- State ---
        self.flight_state = UAVState.TAKEOFF
        self.current_position = self.home.copy()
        self.current_velocity = self.v_initial.copy()
        self.hold_start_time = None


        self.declare_parameter("orientation", True)
        self.orientation = self.get_parameter("orientation").get_parameter_value().bool_value
        

        # --- Precompute geographic serpentine ---
        self.row_idx = 0
        self.get_logger().info("Serpentine node started in TAKEOFF state")
        self.t_serpentine = 0.0
        self.serpentine_state = SerpentineState.STRAIGHT


    # ======================================================================
    # CALLBACKS
    # ======================================================================

    def position_callback(self, msg: VehicleLocalPosition):
        # PX4: NED → ENU
        self.current_position[0] = msg.y
        self.current_position[1] = msg.x
        self.current_position[2] = -msg.z

        self.current_velocity[0] = msg.vy
        self.current_velocity[1] = msg.vx
        self.current_velocity[2] = -msg.vz

        self.get_logger().debug(
            f"Current position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})"
        )

        if self.flight_state == UAVState.TAKEOFF:
            height_difference = abs(self.current_position[2] - self.altitude)

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
    
    def check_hold_timeout(self):
        """Check if hold time has elapsed and transition to FLIGHT"""
        if self.hold_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.hold_start_time).nanoseconds / 1e9

            if elapsed_time >= self.hold_time:
                self.flight_state = UAVState.SERPENTINE
                self.get_logger().info("HOLD completed → SERPENTINE")

    def publish_hold_position(self):
        """Publish hold position at takeoff location"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.home[0])
        pose_msg.pose.position.y = float(self.home[1])
        pose_msg.pose.position.z = float(self.home[2])

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

        for _ in range(self.horizon):
            pose = Pose()
            pose.position.x = float(self.home[0])
            pose.position.y = float(self.home[1])
            pose.position.z = float(self.home[2])
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
                f"Holding position at ({self.home[0]:.2f}, {self.home[1]:.2f}, {self.home[2]:.2f}), "
                f"remaining time: {remaining_time:.1f}s"
            )

    def publish_takeoff_waypoint(self):
        """Publish the takeoff waypoint using StateReference format"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.home[0])
        pose_msg.pose.position.y = float(self.home[1])
        pose_msg.pose.position.z = float(self.home[2])

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
        vel_msg.twist.linear.z = 3.0
        self.vel_pub.publish(vel_msg)

        state_ref = StateReference()
        state_ref.uav_state = String(data="TAKEOFF")

        for _ in range(self.horizon):
            pose = Pose()
            pose.position.x = float(self.home[0])
            pose.position.y = float(self.home[1])
            pose.position.z = float(self.home[2])
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
            self.current_position, [self.home[0], self.home[1], self.home[2]]
        )

        self.get_logger().debug(
            f"Publishing takeoff waypoint: ({self.home[0]:.2f}, {self.home[1]:.2f}, {self.home[2]:.2f}), "
            f"Distance: {dist_to_takeoff:.2f}m"
        )



    # ======================================================================
    # TIMER
    # ======================================================================

    def timer_callback(self):
        if self.flight_state == UAVState.TAKEOFF:
            self.publish_takeoff_waypoint()

        elif self.flight_state == UAVState.HOLD:
            self.publish_hold_position()
            self.check_hold_timeout()

        elif self.flight_state == UAVState.SERPENTINE:
            self.publish_serpentine_reference()

    # ==================================================================
    # SERPENTINE PUBLISH
    # ==================================================================

    def publish_serpentine_reference(self):

        x,y,z, vx,vy,vz = self.serpentine_reference(self.t_serpentine)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)

        if self.orientation:
            dt = 0.01
            next_t = self.t_serpentine + dt
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
        state_ref.uav_state = String(data="SERPENTINE")

        for i in range(self.horizon):
            future_t = self.t_serpentine + i*self.v_ref*self.timer_period

            future_x, future_y, future_z, future_vx, future_vy, future_vz = self.serpentine_reference(future_t)

            pose = Pose()
            pose.position.x = float(future_x)
            pose.position.y = float(future_y)
            pose.position.z = float(future_z)

            if self.orientation and i > 0:
                orient_t = future_t + 0.01
                orient_x, orient_y, orient_z, orient_vx, orient_vy, orient_vz = self.serpentine_reference(orient_t)

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

        v_xy = self.current_velocity[:2]
        speed_along_path = np.linalg.norm(v_xy)

        self.t_serpentine += speed_along_path * self.timer_period







    # ==================================================================
    # SERPENTINE CORE
    # ==================================================================

    def serpentine_reference(self, t):

        direction = 1 if self.row_idx % 2 == 0 else -1
        
        # --- STATE: STRAIGHT ---
        if self.serpentine_state == SerpentineState.STRAIGHT :
            
            x_end = self.first_row[0] + (self.row_length if direction == 1 else 0.0)
            
            has_reached_end = (self.current_position[0] >= x_end) if direction == 1 else (self.current_position[0] <= x_end)

            if not has_reached_end:
                return self.generate_straight(t)
            else:
                self.serpentine_state = SerpentineState.TURN
                t = 0.0
                return self.generate_turn(t) 

        # --- STATE: TURN ---
        elif self.serpentine_state == SerpentineState.TURN:

            next_row_y = self.first_row[1] + (self.row_idx + 1) * self.row_spacing

            has_finished_turn = self.current_position[1] >= next_row_y

            if not has_finished_turn:
                return self.generate_turn(t)
            else:
                self.row_idx += 1
                if self.row_idx >= self.num_rows:
                    self.flight_state = UAVState.MISSION_COMPLETE
                    return self.current_position[0], self.current_position[1], self.altitude, 0.0, 0.0, 0.0  
                
                self.serpentine_state = SerpentineState.STRAIGHT
                self.t_serpentine = 0.0 
                return self.generate_straight(t)
        
        else:
            return self.flight_state == UAVState.MISSION_COMPLETE
        


    # ==================================================================
    # GEOMETRY
    # ==================================================================

    def generate_straight(self, t):
        direction = 1 if self.row_idx % 2 == 0 else -1

        x0 = self.first_row[0]
        x_start = x0 if direction == 1 else x0 + self.row_length

        y = self.first_row[1] + self.row_idx * self.row_spacing

        vx = direction * self.v_ref
        vy = 0.0

        x = x_start + vx * t
        return x, y, self.altitude, vx, vy, 0.0

    def generate_turn(self, t):
        direction = 1 if self.row_idx % 2 == 0 else -1

        R = self.radius_turn
        omega = self.v_ref / R
        theta = omega * t  # 0 → pi

        x_end = self.first_row[0] + (self.row_length if direction == 1 else 0.0)
        y_end = self.first_row[1] + self.row_idx * self.row_spacing

        x_c = x_end
        y_c = y_end + R

        x = x_c + direction * R * np.sin(theta)
        y = y_c - R * np.cos(theta)

        vx = direction * self.v_ref * np.cos(theta)
        vy = self.v_ref * np.sin(theta)

        return x, y, self.altitude, vx, vy, 0.0

    
    def calculate_orientation(self, current_pos, next_pos):
        """Calculate quaternion to orient in direction of travel"""
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        yaw = np.arctan2(dy, dx)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        return cy, 0.0, 0.0, sy



    # ======================================================================
    # CONFIG
    # ======================================================================

    def _parse_config(self, config):
        self.home = np.array(config["home"], dtype=np.float32)
        self.first_row = np.array(config["first_row"], dtype=np.float32)
        self.altitude = float(config["altitude"])
        self.row_length = float(config["row_length"])
        self.row_spacing = float(config["row_spacing"])
        self.num_rows = int(config["num_rows"])
        self.v_ref = float(config["v_ref"])
        self.v_initial = np.array(config["v_initial"], dtype=np.float32)
        self.radius_turn = float(config["radius_turn"])
        self.publish_rate = float(config["publish_rate"])
        self.horizon = int(config["horizon"])
        self.takeoff_threshold = float(config["takeoff_threshold"])
        self.hold_time = float(config["hold_time"])



# ======================================================================
# MAIN
# ======================================================================

def main(args=None):
    rclpy.init(args=args)

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Config file not found: {CONFIG_FILE_PATH}")
        return

    with open(CONFIG_FILE_PATH, "r") as f:
        config = yaml.safe_load(f)

    node = SerpentineTrajectory(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
