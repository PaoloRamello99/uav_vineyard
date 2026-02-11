#!/usr/bin/env python3
############################################################################
# Offboard MPPI Rate Control - PX4 (Subscriber Version)
# Cleaned & Optimized
############################################################################

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

# Controller MPPI
from uav_control_py.controller.mppi.mppi_rate import MPPIRateController
from uav_control_py.config.config_loader import load_mppi_config

# Coordinate Conversion
from automatic_uav_mppi.coordinates_conversion.Ned2Enu import Ned2EnuConverter

# ROS 2 messages
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleCommand,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from quadrotor_msgs.msg import StateReference


class OffboardMPPI(Node):
    def __init__(self):
        super().__init__("offboard_mppi")

        # ================= QoS Profiles =================
        # Best effort is preferred for high-frequency telemetry/control topics
        qos_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ================= Subscribers =================
        # 1. Odometry (NED frame from PX4)
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.odometry_cb, qos_sub
        )

        # 2. Status (Arming/Mode)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",  # Verify if your PX4 version uses _v1
            self.vehicle_status_callback,
            qos_sub,
        )

        # 3. Reference Trajectory (ENU frame from Serpentine Node)
        self.ref_sub = self.create_subscription(
            StateReference, "/command/state_reference", self.reference_callback, 10
        )

        # ================= Publishers =================
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub
        )
        self.rate_pub = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos_pub
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_pub
        )

        # ================= MPPI Controller =================
        self.config = load_mppi_config()
        self.mppi = MPPIRateController(self.config)
        self.dt = float(self.config["dt"])
        self.H = self.mppi.horizon

        # ================= Internal State =================
        self.state_ned = np.zeros(13, dtype=np.float32)
        self.state_enu = np.zeros(13, dtype=np.float32)
        self.state_enu[6] = 1.0  # Initialize Quaternion w=1

        # We start with None to prevent arming before a trajectory exists
        self.ref_traj_enu = None

        # Control Limits
        self.max_thrust = self.config["max_thrust"]
        self.rate_min = np.array(self.config["angular_rate_min"])
        self.rate_max = np.array(self.config["angular_rate_max"])

        # Logic Flags
        self.offboard_setpoint_counter = 0
        self.offboard_ready = False
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        self.get_logger().info("🔥 Warming up MPPI controller...")
        dummy_state = np.zeros(13, dtype=np.float32)
        dummy_state[6] = 1.0
        dummy_ref = np.zeros((self.H, 13), dtype=np.float32)
        dummy_ref[:, 6] = 1.0
        self.mppi.get_control(dummy_state, dummy_ref)
        self.get_logger().info("✅ MPPI Warmup completed.")

        # Start Loop
        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("✅ Offboard MPPI node READY. Waiting for Trajectory...")

    # ================= Callbacks =================
    def vehicle_status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def odometry_cb(self, msg):
        """Receives Odometry (NED) and converts to ENU for MPPI"""
        self.state_ned[0:3] = msg.position
        self.state_ned[3:6] = msg.velocity
        self.state_ned[6:10] = msg.q
        self.state_ned[10:13] = msg.angular_velocity

        # Convert NED to ENU
        self.state_enu = Ned2EnuConverter.ned_to_enu(self.state_ned)

    def reference_callback(self, msg):
        """
        Receives the full horizon trajectory (Takeoff, Hold, or Serpentine)
        and converts it to the numpy format required by MPPI (Jax).
        """
        traj = np.zeros((self.H, 13), dtype=np.float32)

        n_points = len(msg.poses)
        if n_points == 0:
            return

        for i in range(self.H):
            # If the received horizon is shorter than MPPI horizon, repeat the last point
            k = min(i, n_points - 1)

            # Position
            traj[i, 0] = msg.poses[k].position.x
            traj[i, 1] = msg.poses[k].position.y
            traj[i, 2] = msg.poses[k].position.z

            # Linear Velocity
            traj[i, 3] = msg.twists[k].linear.x
            traj[i, 4] = msg.twists[k].linear.y
            traj[i, 5] = msg.twists[k].linear.z

            # Orientation (Quaternion)
            traj[i, 6] = msg.poses[k].orientation.w
            traj[i, 7] = msg.poses[k].orientation.x
            traj[i, 8] = msg.poses[k].orientation.y
            traj[i, 9] = msg.poses[k].orientation.z

            # Angular Velocity
            traj[i, 10] = msg.twists[k].angular.x
            traj[i, 11] = msg.twists[k].angular.y
            traj[i, 12] = msg.twists[k].angular.z

        self.ref_traj_enu = traj

    # ================= PX4 Helpers =================
    def publish_offboard_control_mode(self):
        """Publishes the heartbeat to keep Offboard mode active"""
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True  # We are controlling Body Rates + Thrust
        self.offboard_pub.publish(msg)

    def arm(self):
        """Sends the Arm command"""
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0  # 1 = Arm
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)

    def set_offboard_mode(self):
        """Switches PX4 to Offboard mode"""
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0
        cmd.param2 = 6.0  # 6 = OFFBOARD
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)

    def publish_rates(self, thrust_norm, p, q, r):
        """
        Publishes the control output.
        Note on Frames:
        MPPI outputs in ENU. PX4 expects NED body rates.
        ENU Roll Rate (+X) -> NED Roll Rate (+X)
        ENU Pitch Rate (+Y) -> NED Pitch Rate (-Y) (Inverted)
        ENU Yaw Rate (+Z) -> NED Yaw Rate (-Z) (Inverted)
        """
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        # RICC: It should be FRD (body)
        # https://docs.px4.io/main/en/msg_docs/VehicleRatesSetpoint
        msg.roll = float(p)
        msg.pitch = float(-q)  # Convert to NED
        msg.yaw = float(-r)  # Convert to NED

        # PX4 Frame: Z is Down. To fly UP, we need negative Z force in body frame.
        msg.thrust_body[:] = [0.0, 0.0, -thrust_norm]

        self.rate_pub.publish(msg)

    # ================= Main Loop =================
    def control_loop(self):
        self.publish_offboard_control_mode()

        # 1. Wait for valid trajectory
        if self.ref_traj_enu is None:
            if self.offboard_setpoint_counter % 20 == 0:
                self.get_logger().info("⏳ Waiting for Reference Trajectory...")
            self.offboard_setpoint_counter += 1
            return

        # 2. Arming Sequence
        if not self.offboard_ready:
            self.publish_offboard_control_mode()
            self.publish_rates(0.0, 0.0, 0.0, 0.0)
            self.arm()

            if self.offboard_setpoint_counter == int(1 / self.dt):
                self.set_offboard_mode()
                self.offboard_ready = True
                self.t_start = self.get_clock().now().nanoseconds * 1e-9

            self.offboard_setpoint_counter += 1
            return

        # 3. MPPI Optimization Step
        try:
            # Execute JAX-based MPPI
            u, comp_time, min_cost = self.mppi.get_control(
                self.state_enu, self.ref_traj_enu
            )

            # Unpack u = [Thrust(N), p, q, r]
            thrust, p, q, r = u

            # Saturation & Normalization
            thrust_norm = np.clip(thrust / self.max_thrust, 0.0, 1.0)
            rates = np.clip([p, q, r], self.rate_min, self.rate_max)

            # Publish
            self.publish_rates(thrust_norm, rates[0], rates[1], rates[2])

        except Exception as e:
            self.get_logger().error(f"MPPI Error: {e}")
            # Failsafe: Gentle hover thrust to prevent freefall during errors
            self.publish_rates(0.1, 0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardMPPI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
