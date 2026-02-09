## LEMNISCATE NODE

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty

from px4_msgs.msg import VehicleLocalPosition
from quadrotor_msgs.msg import StateReference


class UAVState:
    IDLE = 0
    TAKEOFF = 1
    HOLD = 2
    FLIGHT = 3


class LemniscateTrajectoryNode(Node):
    def __init__(self):
        super().__init__("lemniscate_trajectory_node")

        # Declare parameters
        self.declare_parameter("a", 10.0)
        self.declare_parameter("height", 3.0)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("speed", 0.5)
        self.declare_parameter("pub_frequency", 50.0)
        self.declare_parameter("orientation", True)
        self.declare_parameter("future_horizon", 1.0)
        self.declare_parameter("takeoff_x", 0.0)
        self.declare_parameter("takeoff_y", 0.0)
        self.declare_parameter("position_threshold", 0.3)
        self.declare_parameter("direction", 1.0)
        self.declare_parameter("random_direction_at_launch", True)
        self.declare_parameter("trajectory_steps", 50)
        self.declare_parameter("hold_time", 3.0)

        # Get parameters
        self.a = self.get_parameter("a").get_parameter_value().double_value
        self.height = self.get_parameter("height").get_parameter_value().double_value
        self.center_x = (
            self.get_parameter("center_x").get_parameter_value().double_value
        )
        self.center_y = (
            self.get_parameter("center_y").get_parameter_value().double_value
        )
        self.speed = self.get_parameter("speed").get_parameter_value().double_value
        self.pub_frequency = (
            self.get_parameter("pub_frequency").get_parameter_value().double_value
        )
        self.orientation = (
            self.get_parameter("orientation").get_parameter_value().bool_value
        )
        self.future_horizon = (
            self.get_parameter("future_horizon").get_parameter_value().double_value
        )
        self.takeoff_x = (
            self.get_parameter("takeoff_x").get_parameter_value().double_value
        )
        self.takeoff_y = (
            self.get_parameter("takeoff_y").get_parameter_value().double_value
        )
        self.position_threshold = (
            self.get_parameter("position_threshold").get_parameter_value().double_value
        )
        self.direction = (
            self.get_parameter("direction").get_parameter_value().double_value
        )
        random_dir_at_launch = (
            self.get_parameter("random_direction_at_launch")
            .get_parameter_value()
            .bool_value
        )
        self.trajectory_steps = (
            self.get_parameter("trajectory_steps").get_parameter_value().integer_value
        )
        self.hold_time = (
            self.get_parameter("hold_time").get_parameter_value().double_value
        )

        if random_dir_at_launch:
            self.direction = 1.0 if np.random.random() < 0.5 else -1.0
            self.get_logger().info(
                f"Randomly selected direction: {'clockwise' if self.direction > 0 else 'counter-clockwise'}"
            )

        self.pose_pub = self.create_publisher(PoseStamped, "/command/ref_pose", 10)
        self.vel_pub = self.create_publisher(TwistStamped, "/command/ref_velocity", 10)
        self.state_ref_pub = self.create_publisher(
            StateReference, "/command/state_reference", 10
        )

        self.srv = self.create_service(
            Empty, "start_lemniscate", self.start_lemniscate_callback
        )

        self.t = np.pi / 2
        self.flight_state = UAVState.IDLE
        self.current_position = [0.0, 0.0, 0.0]
        self.hold_start_time = None

        qos_profile = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos_profile,
        )

        self.timer_period = 1.0 / self.pub_frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.flight_state = UAVState.TAKEOFF
        self.get_logger().info("Starting in TAKEOFF state automatically")

        self.get_logger().info(
            f"Lemniscate Trajectory Node initialized with: "
            f"a={self.a}m, height={self.height}m, "
            f"center=({self.center_x}, {self.center_y}), speed={self.speed}rad/s, "
            f"direction={'clockwise' if self.direction > 0 else 'counter-clockwise'}, "
            f"future_horizon={self.future_horizon}s, "
            f"takeoff position=({self.takeoff_x}, {self.takeoff_y}, {self.height}), "
            f"hold time={self.hold_time}s"
        )

    def position_callback(self, msg):
        """Callback to receive the drone's current position"""
        self.current_position[0] = msg.x
        self.current_position[1] = -msg.y
        self.current_position[2] = -msg.z

        self.get_logger().debug(
            f"Current position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})"
        )

        if self.flight_state == UAVState.TAKEOFF:
            height_difference = abs(self.current_position[2] - self.height)

            if height_difference < self.position_threshold:
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

    def calculate_lemniscate_position(self, t):
        """Calculate position on lemniscate for given parameter t"""
        denominator = 1 + np.square(np.sin(t))
        x = self.center_x + self.a * np.cos(t) / denominator
        y = (
            self.center_y
            + self.direction * self.a * np.sin(t) * np.cos(t) / denominator
        )
        z = self.height
        return x, y, z

    def calculate_lemniscate_velocity(self, t):
        """Calculate velocity on lemniscate for given parameter t"""
        sin_t = np.sin(t)
        cos_t = np.cos(t)
        sin_t_sq = np.square(sin_t)
        denom = 1 + sin_t_sq
        denom_sq = np.square(denom)

        dx_dt = self.a * (-sin_t / denom - cos_t * 2 * sin_t * cos_t / denom_sq)

        term1 = (cos_t * cos_t - sin_t * sin_t) / denom
        term2 = sin_t * cos_t * 2 * sin_t * cos_t / denom_sq
        dy_dt = self.a * self.direction * (term1 - term2)

        vx = dx_dt * self.speed
        vy = dy_dt * self.speed
        vz = 0.0

        return vx, vy, vz

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
            self.publish_lemniscate_trajectory()

    def check_hold_timeout(self):
        """Check if hold time has elapsed and transition to FLIGHT"""
        if self.hold_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.hold_start_time).nanoseconds / 1e9

            if elapsed_time >= self.hold_time:
                self.get_logger().info(
                    f"Hold time of {self.hold_time}s elapsed. Starting lemniscate trajectory."
                )
                self.flight_state = UAVState.FLIGHT

    def publish_hold_position(self):
        """Publish hold position at takeoff location"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.takeoff_x)
        pose_msg.pose.position.y = float(self.takeoff_y)
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
            pose.position.x = float(self.takeoff_x)
            pose.position.y = float(self.takeoff_y)
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
                f"Holding position at ({self.takeoff_x:.2f}, {self.takeoff_y:.2f}, {self.height:.2f}), "
                f"remaining time: {remaining_time:.1f}s"
            )

    def publish_takeoff_waypoint(self):
        """Publish the takeoff waypoint using StateReference format"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.takeoff_x)
        pose_msg.pose.position.y = float(self.takeoff_y)
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
            pose.position.x = float(self.takeoff_x)
            pose.position.y = float(self.takeoff_y)
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
            self.current_position, [self.takeoff_x, self.takeoff_y, self.height]
        )

        self.get_logger().debug(
            f"Publishing takeoff waypoint: ({self.takeoff_x:.2f}, {self.takeoff_y:.2f}, {self.height:.2f}), "
            f"Distance: {dist_to_takeoff:.2f}m"
        )

    def publish_lemniscate_trajectory(self):
        """Publish the lemniscate trajectory using StateReference format"""
        x, y, z = self.calculate_lemniscate_position(self.t)
        vx, vy, vz = self.calculate_lemniscate_velocity(self.t)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)

        if self.orientation:
            dt = 0.01
            next_t = self.t + dt
            next_x, next_y, next_z = self.calculate_lemniscate_position(next_t)

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
            future_t = self.t + i * self.speed * self.timer_period

            future_x, future_y, future_z = self.calculate_lemniscate_position(future_t)
            future_vx, future_vy, future_vz = self.calculate_lemniscate_velocity(
                future_t
            )

            pose = Pose()
            pose.position.x = float(future_x)
            pose.position.y = float(future_y)
            pose.position.z = float(future_z)

            if self.orientation and i > 0:
                orient_t = future_t + 0.01
                orient_x, orient_y, orient_z = self.calculate_lemniscate_position(
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

        self.t += self.speed * self.timer_period

        if self.t >= 2 * math.pi:
            self.t -= 2 * math.pi

        self.get_logger().debug(
            f"Publishing position: ({x:.2f}, {y:.2f}, {z:.2f}), velocity: ({vx:.2f}, {vy:.2f}, {vz:.2f})"
        )

    def start_lemniscate_callback(self, request, response):
        """Callback when the start_lemniscate service is called"""
        self.get_logger().info("Received request to start lemniscate trajectory")

        if self.flight_state == UAVState.IDLE:
            self.flight_state = UAVState.TAKEOFF
            self.get_logger().info("Starting takeoff sequence")
        elif self.flight_state == UAVState.TAKEOFF:
            self.flight_state = UAVState.HOLD
            self.hold_start_time = self.get_clock().now()
            self.get_logger().info("Transitioning from takeoff to hold mode")
        elif self.flight_state == UAVState.HOLD:
            self.flight_state = UAVState.FLIGHT
            self.get_logger().info("Transitioning from hold to lemniscate trajectory")
        else:
            self.get_logger().info(
                f"Already in lemniscate flight mode: {self.flight_state}"
            )

        return response


def main(args=None):
    rclpy.init(args=args)

    lemniscate_node = LemniscateTrajectoryNode()

    try:
        rclpy.spin(lemniscate_node)
    except KeyboardInterrupt:
        pass
    finally:
        lemniscate_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()