import jax
import jax.numpy as jnp
import numpy as np
import rclpy
from rclpy.clock import Clock
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64

from automatic_uav_mppi.mppi_rate_new import MPPIRateController
from px4_msgs.msg import (
    OffboardControlMode,
    VehicleAngularVelocity,
    VehicleAttitude,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleRatesSetpoint,
    VehicleStatus,
)
from quadrotor_msgs.msg import AttitudeReference, StateReference


class UAVOffboardMPPI(Node):
    def __init__(self):
        super().__init__("uav_offboard_mppi")

        self.px4_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.in_takeoff_mode = False
        self.in_hold_mode = False

        self.declare_parameter("uav.mass", 2.0)
        self.declare_parameter("uav.inertia_xx", 0.02166)
        self.declare_parameter("uav.inertia_yy", 0.02166)
        self.declare_parameter("uav.inertia_zz", 0.04)
        self.declare_parameter("uav.motor_constant", 8.54858e-6)
        self.declare_parameter("uav.torque_constant", 0.016)
        self.declare_parameter("uav.arm_length", 0.25)
        self.declare_parameter("uav.max_motor_speed", 1000.0)

        self.declare_parameter("mppi_ctl_fq", 50.0)
        self.declare_parameter("use_future_reference", True)
        self.declare_parameter("use_custom_controller", False)
        self.declare_parameter("offboard.auto_arm", True)
        self.declare_parameter("offboard.auto_offboard", True)
        self.declare_parameter("offboard.setpoint_count", 10)
        self.declare_parameter("takeoff.auto_exit", True)
        self.declare_parameter("takeoff.height_tolerance", 0.2)
        self.declare_parameter("takeoff.velocity_tolerance", 0.2)

        self.declare_parameter("mppi.n_samples", 900)
        self.declare_parameter("mppi.horizon", 25)
        self.declare_parameter("mppi.temperature", 1e3)
        self.declare_parameter("mppi.ctrl_noise_scale", [4.0, 2.0, 2.0, 0.5])
        self.declare_parameter(
            "mppi.Q",
            [
                1e3,
                1e3,
                1e3,
                4e1,
                4e1,
                4e1,
                1e0,
                1e0,
                1e0,
                1e0,
                1e-1,
                1e-1,
                1e-1,
            ],
        )
        self.declare_parameter("mppi.R", [1e-2, 5e-2, 5e-2, 1e-1])
        self.declare_parameter("mppi.R_rate", [1e0, 2e-1, 2e-1, 5e0])

        self.declare_parameter("filter.enabled", True)
        self.declare_parameter("filter.window_length", 11)
        self.declare_parameter("filter.polyorder", 3)

        self.uav_mass = (
            self.get_parameter("uav.mass").get_parameter_value().double_value
        )

        i_xx = self.get_parameter("uav.inertia_xx").get_parameter_value().double_value
        i_yy = self.get_parameter("uav.inertia_yy").get_parameter_value().double_value
        i_zz = self.get_parameter("uav.inertia_zz").get_parameter_value().double_value

        self.motor_constant = (
            self.get_parameter("uav.motor_constant").get_parameter_value().double_value
        )
        self.torque_constant = (
            self.get_parameter("uav.torque_constant").get_parameter_value().double_value
        )
        self.arm_length = (
            self.get_parameter("uav.arm_length").get_parameter_value().double_value
        )
        self.max_motor_speed = (
            self.get_parameter("uav.max_motor_speed").get_parameter_value().double_value
        )

        self.max_thrust = 4.0 * self.motor_constant * np.power(self.max_motor_speed, 2)
        self.declare_parameter("mppi.control_min", [0.0, -3.0, -3.0, -1.0])
        self.declare_parameter("mppi.control_max", [self.max_thrust, 3.0, 3.0, 1.0])

        lx = self.arm_length / np.sqrt(2)
        ly = self.arm_length / np.sqrt(2)
        self.tau_max_x = (
            self.motor_constant * np.power(self.max_motor_speed, 2) * (ly + ly)
        )
        self.tau_max_y = (
            self.motor_constant * np.power(self.max_motor_speed, 2) * (lx + lx)
        )
        self.tau_max_z = (
            self.motor_constant
            * self.torque_constant
            * np.power(self.max_motor_speed, 2)
            * 2
        )

        self.uav_inertia = jnp.diag(jnp.array([i_xx, i_yy, i_zz]))
        self.get_logger().info("Successfully created JAX array for inertia")

        self.mppi_ctl_fq = (
            self.get_parameter("mppi_ctl_fq").get_parameter_value().double_value
        )
        self.mppi_timer_period = 1.0 / self.mppi_ctl_fq

        self.use_future_reference = (
            self.get_parameter("use_future_reference").get_parameter_value().bool_value
        )
        self.use_custom_controller = (
            self.get_parameter("use_custom_controller").get_parameter_value().bool_value
        )
        self.auto_arm = (
            self.get_parameter("offboard.auto_arm").get_parameter_value().bool_value
        )
        self.auto_offboard = (
            self.get_parameter("offboard.auto_offboard")
            .get_parameter_value()
            .bool_value
        )
        self.offboard_setpoint_count = (
            self.get_parameter("offboard.setpoint_count")
            .get_parameter_value()
            .integer_value
        )

        # Debug logging for auto arm/offboard settings
        self.get_logger().info(f"Auto arm enabled: {self.auto_arm}")
        self.get_logger().info(f"Auto offboard enabled: {self.auto_offboard}")
        self.get_logger().info(
            f"Offboard setpoint count: {self.offboard_setpoint_count}"
        )
        self.takeoff_auto_exit = (
            self.get_parameter("takeoff.auto_exit").get_parameter_value().bool_value
        )
        self.takeoff_height_tolerance = (
            self.get_parameter("takeoff.height_tolerance")
            .get_parameter_value()
            .double_value
        )
        self.takeoff_velocity_tolerance = (
            self.get_parameter("takeoff.velocity_tolerance")
            .get_parameter_value()
            .double_value
        )

        self.n_samples = (
            self.get_parameter("mppi.n_samples").get_parameter_value().integer_value
        )
        self.horizon = (
            self.get_parameter("mppi.horizon").get_parameter_value().integer_value
        )
        self.temperature = (
            self.get_parameter("mppi.temperature").get_parameter_value().double_value
        )
        self.ctrl_noise_scale = (
            self.get_parameter("mppi.ctrl_noise_scale")
            .get_parameter_value()
            .double_array_value
        )

        Q_list = self.get_parameter("mppi.Q").get_parameter_value().double_array_value
        R_list = self.get_parameter("mppi.R").get_parameter_value().double_array_value
        R_rate_list = (
            self.get_parameter("mppi.R_rate").get_parameter_value().double_array_value
        )

        self.Q = jnp.diag(jnp.array(Q_list))
        self.R = jnp.diag(jnp.array(R_list))
        self.R_rate = jnp.diag(jnp.array(R_rate_list))

        control_min = (
            self.get_parameter("mppi.control_min")
            .get_parameter_value()
            .double_array_value
        )
        control_max = (
            self.get_parameter("mppi.control_max")
            .get_parameter_value()
            .double_array_value
        )

        self.filter_enabled = (
            self.get_parameter("filter.enabled").get_parameter_value().bool_value
        )
        self.window_length = (
            self.get_parameter("filter.window_length")
            .get_parameter_value()
            .integer_value
        )
        self.polyorder = (
            self.get_parameter("filter.polyorder").get_parameter_value().integer_value
        )

        if self.window_length % 2 == 0:
            self.window_length += 1
            self.get_logger().warn(
                f"Window length must be odd, adjusted to {self.window_length}"
            )

        if self.polyorder >= self.window_length:
            self.polyorder = self.window_length - 1
            self.get_logger().warn(
                f"Polynomial order must be less than window length, adjusted to {self.polyorder}"
            )

        self.get_logger().info(
            f"Savitzky-Golay filter enabled: {self.filter_enabled}, window_length: {self.window_length}, polyorder: {self.polyorder}"
        )

        self.thrust_buffer = np.zeros(self.window_length)
        self.roll_rate_buffer = np.zeros(self.window_length)
        self.pitch_rate_buffer = np.zeros(self.window_length)
        self.yaw_rate_buffer = np.zeros(self.window_length)

        self.filtered_thrust = 0.0
        self.filtered_roll_rate = 0.0
        self.filtered_pitch_rate = 0.0
        self.filtered_yaw_rate = 0.0

        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_position_callback_,
            self.px4_qos,
        )

        self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback_,
            self.px4_qos,
        )

        self.create_subscription(
            VehicleAngularVelocity,
            "/fmu/out/vehicle_angular_velocity",
            self.angular_velocity_callback_,
            self.px4_qos,
        )

        self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.status_callback_,
            self.px4_qos,
        )

        self.create_subscription(
            StateReference,
            "/command/state_reference",
            self.status_ref_callback_,
            10,
        )

        self.offb_ctl_mode_pub_ = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", 10
        )
        self.vehicle_command_pub_ = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", 10
        )

        self.vehicle_rates_pub_ = self.create_publisher(
            VehicleRatesSetpoint, "/fmu/in/vehicle_rates_setpoint", 10
        )

        self.attitude_reference_pub_ = self.create_publisher(
            AttitudeReference, "/mppi/attitude_reference", 10
        )

        self.computation_time_pub_ = self.create_publisher(
            Float64, "~/computation_time_ms", 10
        )

        self.mppi_cost_pub_ = self.create_publisher(Float64, "~/mppi_cost", 10)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_DISARMED
        self.vehicle_status_received = False  # Track if we've received status

        self.dt = self.mppi_timer_period
        self.g = 9.81

        mppi_config = {
            "dt": self.dt,
            "n_samples": self.n_samples,
            "horizon": self.horizon,
            "temperature": self.temperature,
            "mass": self.uav_mass,
            "inertia": self.uav_inertia,
            "g": self.g,
            "tau": 1 / self.mppi_ctl_fq,
            "max_thrust": self.max_thrust,
            "control_min": jnp.array(control_min),
            "control_max": jnp.array(control_max),
            "ctrl_noise_scale": jnp.array(self.ctrl_noise_scale),
            "Q": self.Q,
            "R": self.R,
            "R_rate": self.R_rate,
        }

        self.get_logger().info("Initializing MPPI controller with JAX")
        self.mppi_controller = MPPIRateController(mppi_config)

        self.timer = self.create_timer(self.mppi_timer_period, self.ctl_callback_)

        self.pos_ = np.array([0.0, 0.0, 0.0])
        self.vel_ = np.array([0.0, 0.0, 0.0])
        self.att_ = np.array([1.0, 0.0, 0.0, 0.0])
        self.body_rates_ = np.array([0.0, 0.0, 0.0])

        self.ref_pos_ = np.array([0.0, 0.0, 3.0])
        self.ref_vel_ = np.array([0.0, 0.0, 0.0])
        self.ref_att_ = np.array([1.0, 0.0, 0.0, 0.0])

        self.has_future_trajectory = False
        self.future_positions = np.zeros((self.horizon, 3))
        self.future_velocities = np.zeros((self.horizon, 3))
        self.future_quaternions = np.zeros((self.horizon, 4))

        self.get_logger().info("MPPI controller initialization complete")
        self.get_logger().info(f"JAX is using: {jax.default_backend()}")
        self.get_logger().info(f"JAX devices: {jax.devices()}")

        self.control_counter = 0
        self.error_counter = 0
        self.arm_command_sent = False
        self.offboard_command_sent = False
        self.offboard_setpoint_counter = 0
        self.takeoff_complete = False

    def prepare_reference_trajectory(self):
        """Prepare reference trajectory for MPPI"""
        if not self.has_future_trajectory:
            ref_traj = np.zeros((self.horizon, 13))
            for i in range(self.horizon):
                ref_traj[i, :3] = self.ref_pos_
                ref_traj[i, 3:6] = self.ref_vel_
                ref_traj[i, 6:10] = self.ref_att_
                ref_traj[i, 10:] = np.zeros(3)
        else:
            ref_traj = np.zeros((self.horizon, 13))
            for i in range(min(self.horizon, len(self.future_positions))):
                ref_traj[i, :3] = self.future_positions[i]
                ref_traj[i, 3:6] = self.future_velocities[i]
                ref_traj[i, 6:10] = self.future_quaternions[i]
                ref_traj[i, 10:] = np.zeros(3)

            if len(self.future_positions) < self.horizon:
                for i in range(len(self.future_positions), self.horizon):
                    ref_traj[i] = ref_traj[len(self.future_positions) - 1]

        return jnp.array(ref_traj)

    def prepare_current_state(self):
        """Prepare the current state vector for MPPI"""
        return jnp.array(
            [
                self.pos_[0],
                self.pos_[1],
                self.pos_[2],
                self.vel_[0],
                self.vel_[1],
                self.vel_[2],
                self.att_[0],
                self.att_[1],
                self.att_[2],
                self.att_[3],
                self.body_rates_[0],
                self.body_rates_[1],
                self.body_rates_[2],
            ]
        )

    def normalize_thrust(self, thrust):
        """Convert raw thrust to normalized thrust [-1, 0] for PX4 compatibility."""
        thrust_normalized = 1.15 * thrust / self.max_thrust
        thrust_body = np.clip(-thrust_normalized, -1.0, 0.0)
        return float(thrust_body)

    def publish_offboard_control_mode(self):
        """Publish offboard control mode message with body rate control enabled."""
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)

        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        offboard_msg.thrust_and_torque = False
        offboard_msg.direct_actuator = False

        self.offb_ctl_mode_pub_.publish(offboard_msg)

    def apply_savgol_filter(self, new_value, buffer):
        """Apply Savitzky-Golay filter to smooth control inputs."""
        if not self.filter_enabled:
            return new_value, buffer

        buffer = np.roll(buffer, -1)
        buffer[-1] = new_value

        filtered_value = savgol_filter(
            buffer, self.window_length, self.polyorder, mode="nearest"
        )[-1]

        return filtered_value, buffer

    def local_position_callback_(self, msg):
        """Process vehicle local position updates."""
        self.get_logger().info(
            f"Received new vehicle local position: "
            f"x={msg.x:.4f}, "
            f"y={msg.y:.4f}, "
            f"z={msg.z:.4f}",
            once=True,
        )
        self.pos_[0] = msg.y
        self.pos_[1] = msg.x
        self.pos_[2] = -msg.z
        self.vel_[0] = msg.vy
        self.vel_[1] = msg.vx
        self.vel_[2] = -msg.vz

    def attitude_callback_(self, msg):
        self.get_logger().info(
            f"Received new vehicle attitude: "
            f"q0={msg.q[0]:.4f}, "
            f"q1={msg.q[1]:.4f}, "
            f"q2={msg.q[2]:.4f}, "
            f"q3={msg.q[3]:.4f}",
            once=True,
        )
        self.att_[0] = msg.q[0]
        self.att_[1] = msg.q[2]
        self.att_[2] = msg.q[1]
        self.att_[3] = -msg.q[3]

    def angular_velocity_callback_(self, msg):
        self.get_logger().info(
            f"Received new vehicle angular velocity: "
            f"p={msg.xyz[0]:.4f}, "
            f"q={msg.xyz[1]:.4f}, "
            f"r={msg.xyz[2]:.4f}",
            once=True,
        )
        self.body_rates_[0] = msg.xyz[1]
        self.body_rates_[1] = msg.xyz[0]
        self.body_rates_[2] = -msg.xyz[2]

    def status_callback_(self, msg):
        self.get_logger().info(
            f"Received new vehicle status: nav_state={msg.nav_state}, arm_state={msg.arming_state}",
            once=True,
        )
        if self.control_counter % 200 == 0:
            self.get_logger().debug(
                f"Status update: nav_state={msg.nav_state}, arm_state={msg.arming_state}"
            )

        # Reset command flags when state changes back to allow re-triggering
        if msg.arming_state != self.arm_state:
            if msg.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.arm_command_sent = False  # Reset when vehicle becomes disarmed
                self.get_logger().info("Vehicle disarmed - resetting arm command flag")

        if msg.nav_state != self.nav_state:
            if msg.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.offboard_command_sent = False  # Reset when leaving offboard mode
                self.get_logger().info(
                    "Vehicle left offboard mode - resetting offboard command flag"
                )

        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.vehicle_status_received = True

    def send_arm_command(self):
        cmd = VehicleCommand()
        cmd.timestamp = int(Clock().now().nanoseconds / 1000)
        cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        cmd.param1 = 1.0
        cmd.param2 = 0.0
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.vehicle_command_pub_.publish(cmd)
        self.arm_command_sent = True

    def send_offboard_command(self):
        cmd = VehicleCommand()
        cmd.timestamp = int(Clock().now().nanoseconds / 1000)
        cmd.command = VehicleCommand.VEHICLE_CMD_SET_NAV_STATE
        cmd.param1 = float(VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.vehicle_command_pub_.publish(cmd)
        self.offboard_command_sent = True

    def status_ref_callback_(self, msg):
        """Callback for state reference updates"""
        num_points = len(msg.poses)

        if num_points == 0:
            return

        uav_state = (
            msg.uav_state.data if hasattr(msg, "uav_state") and msg.uav_state else ""
        )

        if uav_state == "TAKEOFF" and not self.takeoff_complete:
            self.ref_pos_[0] = msg.poses[0].position.x
            self.ref_pos_[1] = msg.poses[0].position.y
            self.ref_pos_[2] = msg.poses[0].position.z

            self.in_takeoff_mode = True
            self.in_hold_mode = False
            self.get_logger().info("Received TAKEOFF state command", once=True)
            return
        elif uav_state == "TAKEOFF" and self.takeoff_complete:
            self.in_takeoff_mode = False
            self.in_hold_mode = False
        elif uav_state == "HOLD":
            self.ref_pos_[0] = msg.poses[0].position.x
            self.ref_pos_[1] = msg.poses[0].position.y
            self.ref_pos_[2] = msg.poses[0].position.z

            self.ref_att_[0] = msg.poses[0].orientation.w
            self.ref_att_[1] = msg.poses[0].orientation.x
            self.ref_att_[2] = msg.poses[0].orientation.y
            self.ref_att_[3] = msg.poses[0].orientation.z

            if len(msg.twists) > 0:
                self.ref_vel_[0] = msg.twists[0].linear.x
                self.ref_vel_[1] = msg.twists[0].linear.y
                self.ref_vel_[2] = msg.twists[0].linear.z

            self.in_takeoff_mode = False
            self.in_hold_mode = True
            self.get_logger().info("Received HOLD state command", once=True)
            return
        else:
            self.in_takeoff_mode = False
            self.in_hold_mode = False

        self.ref_pos_[0] = msg.poses[0].position.x
        self.ref_pos_[1] = msg.poses[0].position.y
        self.ref_pos_[2] = msg.poses[0].position.z

        self.ref_att_[0] = msg.poses[0].orientation.w
        self.ref_att_[1] = msg.poses[0].orientation.x
        self.ref_att_[2] = msg.poses[0].orientation.y
        self.ref_att_[3] = msg.poses[0].orientation.z

        if len(msg.twists) > 0:
            self.ref_vel_[0] = msg.twists[0].linear.x
            self.ref_vel_[1] = msg.twists[0].linear.y
            self.ref_vel_[2] = msg.twists[0].linear.z

        if num_points > 1:
            max_points = min(num_points, self.future_positions.shape[0])

            for i in range(max_points):
                self.future_positions[i, 0] = msg.poses[i].position.x
                self.future_positions[i, 1] = msg.poses[i].position.y
                self.future_positions[i, 2] = msg.poses[i].position.z

                self.future_quaternions[i, 0] = msg.poses[i].orientation.w
                self.future_quaternions[i, 1] = msg.poses[i].orientation.x
                self.future_quaternions[i, 2] = msg.poses[i].orientation.y
                self.future_quaternions[i, 3] = msg.poses[i].orientation.z

                if i < len(msg.twists):
                    self.future_velocities[i, 0] = msg.twists[i].linear.x
                    self.future_velocities[i, 1] = msg.twists[i].linear.y
                    self.future_velocities[i, 2] = msg.twists[i].linear.z
                else:
                    self.future_velocities[i] = np.zeros(3)

            if max_points < self.future_positions.shape[0]:
                for i in range(max_points, self.future_positions.shape[0]):
                    self.future_positions[i] = self.future_positions[max_points - 1]
                    self.future_velocities[i] = self.future_velocities[max_points - 1]
                    self.future_quaternions[i] = self.future_quaternions[max_points - 1]

            self.has_future_trajectory = True
        else:
            self.has_future_trajectory = False

    def ctl_callback_(self):
        """Main control callback"""
        self.control_counter += 1

        self.publish_offboard_control_mode()
        self.offboard_setpoint_counter += 1

        # Only proceed with auto arm/offboard if we've received vehicle status
        if not self.vehicle_status_received:
            if self.control_counter % 100 == 0:
                self.get_logger().warn(
                    "Waiting for vehicle status before auto arm/offboard"
                )
            # Continue with the rest of the callback but skip auto commands
        elif self.vehicle_status_received:
            # Auto arm logic
            if self.auto_arm and self.arm_state == VehicleStatus.ARMING_STATE_DISARMED:
                self.get_logger().info(
                    "Sending auto arm command", throttle_duration_sec=1.0
                )
                self.send_arm_command()

            # Auto offboard logic
            if (
                self.auto_offboard
                and self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
                and self.offboard_setpoint_counter >= self.offboard_setpoint_count
            ):
                self.get_logger().info(
                    f"Sending auto offboard command (counter={self.offboard_setpoint_counter})",
                    throttle_duration_sec=1.0,
                )
                self.send_offboard_command()

        if (
            self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD
            and not self.use_custom_controller
        ):
            rates_msg = VehicleRatesSetpoint()
            rates_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            rates_msg.roll = 0.0
            rates_msg.pitch = 0.0
            rates_msg.yaw = 0.0
            rates_msg.thrust_body = [0.0, 0.0, -0.5]
            self.vehicle_rates_pub_.publish(rates_msg)

        if (
            self.in_takeoff_mode
            and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            current_state = self.prepare_current_state()
            reference_trajectory = self.prepare_reference_trajectory()
            control_command, compute_time, min_cost = self.mppi_controller.get_control(
                current_state, reference_trajectory
            )

            cost_msg = Float64()
            cost_msg.data = min_cost
            self.mppi_cost_pub_.publish(cost_msg)

            height_error = self.ref_pos_[2] - self.pos_[2]
            height_vel_error = 0.0 - self.vel_[2]

            kp_z = 0.15
            kd_z = 0.08

            base_thrust = 0.75
            thrust_correction = kp_z * height_error + kd_z * height_vel_error
            takeoff_thrust = min(max(base_thrust + thrust_correction, 0.6), 0.9)

            if self.control_counter % 50 == 0:
                self.get_logger().info(
                    f"TAKEOFF PD: z_err={height_error:.2f}, z_vel={self.vel_[2]:.2f}, "
                    f"thrust={takeoff_thrust:.2f}"
                )

            att_ref_msg = AttitudeReference()
            att_ref_msg.header.stamp = self.get_clock().now().to_msg()
            att_ref_msg.thrust_body = [0.0, 0.0, -takeoff_thrust]

            att_ref_msg.attitude.w = self.att_[0]
            att_ref_msg.attitude.x = self.att_[1]
            att_ref_msg.attitude.y = self.att_[2]
            att_ref_msg.attitude.z = self.att_[3]

            att_ref_msg.angular_velocity.x = 0.0
            att_ref_msg.angular_velocity.y = 0.0
            att_ref_msg.angular_velocity.z = 0.0

            self.attitude_reference_pub_.publish(att_ref_msg)

            if not self.use_custom_controller:
                rates_msg = VehicleRatesSetpoint()
                rates_msg.timestamp = int(Clock().now().nanoseconds / 1000)
                rates_msg.roll = 0.0
                rates_msg.pitch = 0.0
                rates_msg.yaw = 0.0
                rates_msg.thrust_body = [0.0, 0.0, -takeoff_thrust]

                self.vehicle_rates_pub_.publish(rates_msg)

            if (
                self.takeoff_auto_exit
                and abs(height_error) <= self.takeoff_height_tolerance
                and abs(height_vel_error) <= self.takeoff_velocity_tolerance
            ):
                self.in_takeoff_mode = False
                self.in_hold_mode = False
                self.takeoff_complete = True
                self.get_logger().info(
                    "Takeoff complete - switching to MPPI control", once=True
                )

            return

        try:
            current_state = self.prepare_current_state()
            reference_trajectory = self.prepare_reference_trajectory()

            control_command, compute_time, min_cost = self.mppi_controller.get_control(
                current_state, reference_trajectory
            )

            cost_msg = Float64()
            cost_msg.data = min_cost
            self.mppi_cost_pub_.publish(cost_msg)

            compute_time_ms = compute_time * 1000

            computation_time_msg = Float64()
            computation_time_msg.data = compute_time_ms
            self.computation_time_pub_.publish(computation_time_msg)

            raw_thrust = float(control_command[0])
            raw_roll_rate = float(control_command[1])
            raw_pitch_rate = float(control_command[2])
            raw_yaw_rate = float(control_command[3])

            self.filtered_thrust, self.thrust_buffer = self.apply_savgol_filter(
                raw_thrust, self.thrust_buffer
            )
            self.filtered_roll_rate, self.roll_rate_buffer = self.apply_savgol_filter(
                raw_roll_rate, self.roll_rate_buffer
            )
            self.filtered_pitch_rate, self.pitch_rate_buffer = self.apply_savgol_filter(
                raw_pitch_rate, self.pitch_rate_buffer
            )
            self.filtered_yaw_rate, self.yaw_rate_buffer = self.apply_savgol_filter(
                raw_yaw_rate, self.yaw_rate_buffer
            )

            thrust = self.filtered_thrust
            roll_rate = self.filtered_roll_rate
            pitch_rate = self.filtered_pitch_rate
            yaw_rate = self.filtered_yaw_rate

            if self.in_hold_mode and self.control_counter % 50 == 0:
                self.get_logger().info(
                    f"HOLD MPPI: thrust={thrust:.2f}N, "
                    f"rates=[{roll_rate:.2f}, {pitch_rate:.2f}, {yaw_rate:.2f}]"
                )

            if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                thrust_normalized = self.normalize_thrust(thrust)

                att_ref_msg = AttitudeReference()
                att_ref_msg.header.stamp = self.get_clock().now().to_msg()
                att_ref_msg.thrust_body = [0.0, 0.0, thrust_normalized]

                dt = self.mppi_timer_period
                omega = np.array([roll_rate, -pitch_rate, -yaw_rate])

                current_rot = R.from_quat(
                    [self.att_[1], self.att_[2], self.att_[3], self.att_[0]]
                )

                delta_rot = R.from_rotvec(omega * dt)
                predicted_rot = delta_rot * current_rot

                predicted_quat = predicted_rot.as_quat()

                att_ref_msg.attitude.w = predicted_quat[3]
                att_ref_msg.attitude.x = predicted_quat[1]
                att_ref_msg.attitude.y = predicted_quat[0]
                att_ref_msg.attitude.z = -predicted_quat[2]

                att_ref_msg.angular_velocity.x = pitch_rate
                att_ref_msg.angular_velocity.y = roll_rate
                att_ref_msg.angular_velocity.z = -yaw_rate

                self.attitude_reference_pub_.publish(att_ref_msg)

                if not self.use_custom_controller:
                    rates_msg = VehicleRatesSetpoint()
                    rates_msg.timestamp = int(Clock().now().nanoseconds / 1000)
                    rates_msg.roll = pitch_rate
                    rates_msg.pitch = roll_rate
                    rates_msg.yaw = -yaw_rate
                    rates_msg.thrust_body = [0.0, 0.0, thrust_normalized]

                    self.vehicle_rates_pub_.publish(rates_msg)
                else:
                    self.get_logger().debug(
                        "Using custom controller mode - only publishing attitude reference",
                        once=True,
                    )

        except Exception as e:
            self.get_logger().error(f"MPPI control computation failed: {str(e)}")
            self.get_logger().warn("Using fallback zero control")
            self.error_counter += 1

            if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                rates_msg = VehicleRatesSetpoint()
                rates_msg.timestamp = int(Clock().now().nanoseconds / 1000)
                rates_msg.roll = 0.0
                rates_msg.pitch = 0.0
                rates_msg.yaw = 0.0
                hover_thrust = -0.5
                rates_msg.thrust_body = [0.0, 0.0, hover_thrust]
                self.vehicle_rates_pub_.publish(rates_msg)


def main(args=None):
    rclpy.init(args=args)

    quad_rate_mppi_node = UAVOffboardMPPI()

    try:
        rclpy.spin(quad_rate_mppi_node)
    except KeyboardInterrupt:
        pass
    finally:
        quad_rate_mppi_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()