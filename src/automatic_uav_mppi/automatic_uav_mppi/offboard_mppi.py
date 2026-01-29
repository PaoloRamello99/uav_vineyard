#!/usr/bin/env python3
############################################################################
# Offboard MPPI Rate Control - PX4
############################################################################

import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from automatic_uav_mppi.config.config_loader import load_mppi_config
from automatic_uav_mppi.controller.mppi.mppi_rate import MPPIRateController

# from automatic_uav_mppi.coordinates_conversion.Enu2Ned import Enu2NedConverter
from automatic_uav_mppi.coordinates_conversion.Ned2Enu import Ned2EnuConverter
from automatic_uav_mppi.mission.mission_reference import SerpentineMission
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleCommand,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleStatus,
)


class OffboardMPPI(Node):
    def __init__(self):
        super().__init__("offboard_mppi")
        # ================= QoS =================
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
        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.odometry_cb, qos_sub
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_sub,
        )
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            "fmu/out/vehicle_status_v1",
            self.vehicle_status_callback,
            qos_sub,
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

        # ================= MPPI =================
        self.config = load_mppi_config()
        self.mppi = MPPIRateController(self.config)
        self.dt = float(self.config["dt"])
        self.H = self.mppi.horizon

        self.get_logger().info("🔥 Warming up MPPI controller...")
        dummy_state = np.zeros(13, dtype=np.float32)
        dummy_state[6] = 1.0
        dummy_ref = np.zeros((self.H, 13), dtype=np.float32)
        dummy_ref[:, 6] = 1.0
        self.mppi.get_control(dummy_state, dummy_ref)
        self.get_logger().info("✅ MPPI Warmup completed.")

        self.get_logger().info("✅ Offboard MPPI node READY (PX4 compliant)")

        # ================= Mission =================
        self.mission = SerpentineMission()
        self.t_start = self.get_clock().now().nanoseconds * 1e-9  # mission clock

        # ================= State =================
        self.state_ned = np.zeros(13, dtype=np.float32)
        self.state_ned[6] = 1.0

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.hover_thrust_norm = (self.config["mass"] * self.config["g"]) / self.config[
            "max_thrust"
        ]
        self.offboard_ready = False
        self.timer = self.create_timer(self.dt, self.control_loop)

        # ================= Limits =================
        self.max_thrust = self.config["max_thrust"]
        self.rate_min = np.array(self.config["angular_rate_min"])
        self.rate_max = np.array(self.config["angular_rate_max"])

        # ================= Logging =================
        log_dir = "/workspaces/uav_vineyard/src/automatic_uav_mppi/files"
        os.makedirs(log_dir, exist_ok=True)

        self.log_path = os.path.join(log_dir, "mppi_control_log.txt")
        self.log_file = open(self.log_path, "w")

        # header
        self.log_file.write(
            "time   comp_time_s     min_cost|   "
            "thrust_N   p_cmd   q_cmd   r_cmd|  "
            "x_enu   y_enu   z_enu| "
            "ref_x   ref_y   ref_z|   "
            "vx_enu  vy_enu  vz_enu|    "
            "ref_vx  ref_vy ref_vz\n"
        )
        self.log_file.flush()

        self.log_counter = 0
        self.get_logger().info(f"📁 Logging MPPI data to {self.log_path}")

    # ================= Callbacks =================
    def vehicle_status_callback(self, msg):
        """Callback function for vehicle_status topic subscriber."""
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_pub.publish(msg)

    def arm(self):
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0  # arm
        cmd.param2 = 0.0
        cmd.param3 = 0.0
        cmd.param4 = 0.0
        cmd.param5 = 0.0
        cmd.param6 = 0.0
        cmd.param7 = 0.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)

    def set_offboard_mode(self):
        cmd = VehicleCommand()
        cmd.timestamp = self.get_clock().now().nanoseconds // 1000
        cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        cmd.param1 = 1.0  # PX4 custom mode
        cmd.param2 = 6.0  # OFFBOARD
        cmd.param3 = 0.0
        cmd.param4 = 0.0
        cmd.param5 = 0.0
        cmd.param6 = 0.0
        cmd.param7 = 0.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.cmd_pub.publish(cmd)

    def odometry_cb(self, msg):
        self.state_ned[0:3] = msg.position
        self.state_ned[3:6] = msg.velocity
        self.state_ned[6:10] = msg.q
        self.state_ned[10:13] = msg.angular_velocity

        # Convert NED to ENU
        self.state_enu = Ned2EnuConverter.ned_to_enu(self.state_ned)

    # ================= Publish =================
    def publish_rates(self, thrust_cmd, p, q, r):
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # microseconds
        msg.roll = p
        msg.pitch = -q
        msg.yaw = -r
        msg.thrust_body[:] = [0.0, 0.0, -thrust_cmd]
        self.rate_pub.publish(msg)

    # ================= Main Loop =================
    def control_loop(self):
        self.publish_offboard_control_mode()

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

        # -------- Current mission time --------
        t_now = self.get_clock().now().nanoseconds * 1e-9 - self.t_start

        # --- TEST BYPASS MPPI ---
        # 75% riesce a decollare
        # thrust_cmd = 19.62/self.max_thrust  # 75% thrust
        # self.publish_rates(thrust_cmd, 0.0, 0.0, 0.0)
        # self.get_logger().info(f"TEST BYPASS: Thrust Cmd {thrust_cmd}")

        # -------- Reference trajectory --------
        ref_enu = np.zeros((self.H, 13), dtype=np.float32)
        for k in range(self.H):
            t_k = t_now + k * self.dt
            ref_enu[k] = self.mission.get_reference(t_k)

        # -------- Convert ENU -> NED --------
        # ref_ned = np.array([Enu2NedConverter.enu_to_ned(x) for x in ref_enu])

        # -------- MPPI control --------
        # u, _, _ = self.mppi.get_control(self.state_ned, ref_ned)
        try:
            u, comp_time, min_cost = self.mppi.get_control(self.state_enu, ref_enu)
        except ValueError:
            ret = self.mppi.get_control(self.state_enu, ref_enu)
            u = ret[0]
            comp_time = 0.0
            min_cost = 0.0
        u, _, _ = self.mppi.get_control(self.state_enu, ref_enu)
        thrust, p, q, r = u

        # -------- Saturation --------
        thrust_cmd = np.clip(thrust / self.max_thrust, 0.0, 1.0)
        rates = np.clip([p, q, r], self.rate_min, self.rate_max)

        # -------- Publish rates --------
        self.publish_rates(thrust_cmd, rates[0], rates[1], rates[2])

        # -------- Log data --------
        if self.log_file:  # Safety check
            t_log = self.get_clock().now().nanoseconds * 1e-9

            x, y, z = self.state_enu[0:3]
            vx, vy, vz = self.state_enu[3:6]

            ref_x, ref_y, ref_z = ref_enu[0, 0:3]
            ref_vx, ref_vy, ref_vz = ref_enu[0, 3:6]

            self.log_file.write(
                f"{t_log:.4f}    {comp_time:.4f}     {min_cost:.4f}|  "
                f"{thrust:.3f}   {rates[0]:.3f}   {rates[1]:.3f}     {rates[2]:.3f}|  "
                f"{x:.3f}    {y:.3f}    {z:.3f}|     "
                f"{ref_x:.3f}    {ref_y:.3f}    {ref_z:.3f}|  "
                f"{vx:.3f}    {vy:.3f}    {vz:.3f}|   "
                f"{ref_vx:.3f}    {ref_vy:.3f}    {ref_vz:.3f}\n"
            )

        # flush every 20 lines
        self.log_counter += 1
        if self.log_counter % 20 == 0:
            self.log_file.flush()


# ================= Main =================
def main(args=None):
    rclpy.init(args=args)
    node = OffboardMPPI()
    rclpy.spin(node)
    node.log_file.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
