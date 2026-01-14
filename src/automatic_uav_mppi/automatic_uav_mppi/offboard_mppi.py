#!/usr/bin/env python3
############################################################################
# Offboard MPPI Rate Control - PX4
############################################################################

import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import (
    OffboardControlMode,
    VehicleRatesSetpoint,
    VehicleOdometry,
    VehicleAngularVelocity,
    VehicleStatus,
    VehicleAttitude,
    VehicleCommand
)

from uav_control_py.config.config_loader import load_mppi_config
from uav_control_py.controller.mppi.mppi_rate import MPPIRateController
from uav_control_py.mission.mission_reference import SerpentineMission


class OffboardMPPI(Node):

    def __init__(self):
        super().__init__("offboard_mppi")

        # ================= QoS =================
        qos_pub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             history=QoSHistoryPolicy.KEEP_LAST, 
                             depth=1
                             )
        qos_sub = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                             durability=QoSDurabilityPolicy.VOLATILE,
                             history=QoSHistoryPolicy.KEEP_LAST, 
                             depth=1
                             )

        # ================= Subscribers =================
        self.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", self.odometry_cb, qos_sub)
        self.create_subscription(VehicleAngularVelocity, "/fmu/out/vehicle_angular_velocity", self.angular_velocity_cb, qos_sub)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_cb, qos_sub)
        self.status_sub = self.create_subscription(VehicleStatus, 'fmu/out/vehicle_status', self.vehicle_status_callback, qos_sub)
        self.status_sub_v1 = self.create_subscription(VehicleStatus, 'fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_sub)
        
        # ================= Publishers =================
        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", qos_pub)
        self.rate_pub = self.create_publisher(VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", qos_pub)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos_pub)

        # ================= MPPI =================
        self.config = load_mppi_config()
        self.mppi = MPPIRateController(self.config)
        self.dt = float(self.config["dt"])
        self.H = self.mppi.horizon

        self.get_logger().info("✅ Offboard MPPI node READY (PX4 compliant)")

        # ================= Mission =================
        self.mission = SerpentineMission()
        self.t_start = self.get_clock().now().nanoseconds * 1e-9  # mission clock

        # ================= State =================
        self.state_ned = np.zeros(13, dtype=np.float32)
        self.state_received = False
        
        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.offboard_ready = False
        self.timer = self.create_timer(self.dt, self.control_loop)


        # ================= Limits =================
        self.max_thrust = self.config["max_thrust"]
        self.rate_min = np.array(self.config["angular_rate_min"])
        self.rate_max = np.array(self.config["angular_rate_max"])



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
        cmd.param1 = 1.0   # arm
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
        cmd.param1 = 1.0    # PX4 custom mode
        cmd.param2 = 6.0    # OFFBOARD
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
        

    def attitude_cb(self, msg):
        self.state_ned[6:10] = msg.q
        self.state_received = True

    def angular_velocity_cb(self, msg):
        self.state_ned[10:13] = msg.xyz

    
    # ================= Publish =================
    def publish_rates(self, thrust_n, p, q, r):
        msg = VehicleRatesSetpoint()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000  # microseconds
        msg.roll = p
        msg.pitch = q
        msg.yaw = r
        msg.thrust_body[:] = [0.0, 0.0, -thrust_n]
        self.rate_pub.publish(msg)

    @staticmethod
    def enu_to_ned(x_enu):
        R = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        x = np.zeros_like(x_enu)
        x[0:3] = R @ x_enu[0:3]
        x[3:6] = R @ x_enu[3:6]
        qw,qx,qy,qz = x_enu[6:10]
        x[6:10] = [qw, qy, qx, -qz]
        x[10:13] = x_enu[10:13]
        return x



    # ================= Main Loop =================
    def control_loop(self):
        
        if not self.offboard_ready:
            self.publish_offboard_control_mode()
            self.publish_rates(0.75, 0.0, 0.0, 0.0)
            
            if self.offboard_setpoint_counter == int(1/self.dt):
                self.set_offboard_mode()
                self.arm()
                self.offboard_ready = True

            self.offboard_setpoint_counter += 1
            return
        





        # -------- Current mission time --------
        t_now = self.get_clock().now().nanoseconds * 1e-9 - self.t_start

        # -------- Reference trajectory --------
        ref_enu = np.zeros((self.H, 13), dtype=np.float32)
        for k in range(self.H):
            t_k = t_now + k * self.dt
            ref_enu[k] = self.mission.get_reference(t_k)

        # -------- Convert ENU -> NED --------
        ref_ned = np.array([self.enu_to_ned(x) for x in ref_enu], dtype=np.float32)

        # -------- MPPI control --------
        u, _, _ = self.mppi.get_control(self.state_ned, ref_ned)
        thrust, p, q, r = u
        self.get_logger().info(f"MPPI:\n{u}")

        # -------- Saturation --------
        thrust_n = np.clip(thrust / self.max_thrust, 0.0, 1.0)
        rates = np.clip([p, q, r], self.rate_min, self.rate_max)

        # -------- Publish rates --------
        self.publish_rates(thrust_n, rates[0], rates[1], rates[2])



    


# ================= Main =================
def main(args=None):
    rclpy.init(args=args)
    node = OffboardMPPI()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
