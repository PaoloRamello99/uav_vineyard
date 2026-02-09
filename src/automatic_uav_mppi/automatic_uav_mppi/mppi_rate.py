#!/usr/bin/env python3

"""
MPPI Controller Implementation for UAV Control Using JAX

This module contains a Model Predictive Path Integral (MPPI) controller implementation
that uses JAX for efficient computation and automatic differentiation.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX logging
# logging.getLogger("jax").setLevel(logging.DEBUG)


class MPPIRateController:
    """
    Model Predictive Path Integral (MPPI) controller for UAV rate control

    This implementation uses JAX for efficient sampling and computation.
    """

    def __init__(self, config, logger=None):
        """
        Initialize the MPPI controller with configuration parameters

        Args:
            config: Dictionary containing controller parameters
            logger: Optional ROS logger to use for logging
        """
        # Store ROS logger if provided
        self.logger = logger

        # Extract parameters from config
        self.dt = config.get("dt", 0.02)  # Control timestep
        self.n_samples = config.get("n_samples", 900)  # Number of samples for MPPI
        self.horizon = config.get("horizon", 25)  # Prediction horizon
        self.temperature = config.get(
            "temperature", 1e-3
        )  # Temperature parameter for weighting

        # UAV physical parameters
        self.mass = config.get("mass", 2.0)
        self.inertia = config.get(
            "inertia", jnp.diag(jnp.array([0.02166, 0.02166, 0.04]))
        )
        self.inertia_inv = jnp.linalg.inv(self.inertia)  # Precompute inverse
        self.g = config.get("g", 9.81)  # Gravity
        self.tau = config.get("tau", 0.02)  # Time constant for body rate dynamics

        # Control limits
        self.max_thrust = config.get("max_thrust", 40.0)
        self.control_min = config.get("control_min", jnp.array([0.0, -3.0, -3.0, -1.0]))
        self.control_max = config.get(
            "control_max", jnp.array([self.max_thrust, 3.0, 3.0, 1.0])
        )

        # Cost weights
        self.Q = config.get(
            "Q",
            jnp.diag(
                jnp.array(
                    [
                        8e2,
                        8e2,
                        8e2,  # Position errors
                        4e1,
                        4e1,
                        4e1,  # Velocity errors
                        2e1,
                        2e1,
                        2e1,
                        2e1,  # Quaternion errors
                        2e1,
                        2e1,
                        2e1,  # Angular rate errors
                    ]
                )
            ),
        )

        self.R = config.get(
            "R", jnp.diag(jnp.array([0.01, 0.05, 0.05, 0.10]))
        )  # Control cost
        self.R_rate = config.get(
            "R_rate", jnp.diag(jnp.array([0.10, 0.30, 0.30, 0.50]))
        )  # Control rate cost

        # Noise scaling for control samples
        self.ctrl_noise_scale = config.get(
            "ctrl_noise_scale", jnp.array([4.0, 0.30, 0.30, 0.10])
        )

        # Initialize RNG key
        self.rng_key = jax.random.PRNGKey(0)

        # Initialize control sequence
        hover_thrust = self.mass * self.g
        self.control_sequence = np.zeros((self.horizon, 4))
        self.control_sequence[:, 0] = hover_thrust  # Set thrust to hover thrust

        # JIT compile functions for better performance
        try:
            self.jit_dynamics = jax.jit(self.jax_dynamics_3d)
            self.jit_mppi = jax.jit(self.jax_mppi_3d)
            self.jit_compiled = True
            print("MPPI controller initialized with JAX JIT compilation")
        except Exception as e:
            print(f"Error during JAX JIT compilation: {str(e)}")
            print("Falling back to non-JIT functions - performance may be degraded")
            self.jit_dynamics = self.jax_dynamics_3d
            self.jit_mppi = self.jax_mppi_3d
            self.jit_compiled = False

    def jax_dynamics_3d(self, state, control, dt=None, m=None, g=None):
        """
        JAX-compatible 3D quadrotor dynamics with quaternion attitude

        Args:
            state: Current state [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
            control: Control input [thrust, p_cmd, q_cmd, r_cmd]
            dt: Time step (can override self.dt)
            m: Mass (can override self.mass)
            g: Gravity (can override self.g)

        Returns:
            Next state after applying the control input
        """
        if dt is None:
            dt = self.dt
        if m is None:
            m = self.mass
        if g is None:
            g = self.g

        # Define the state derivative function for RK4 integration
        def state_derivative(state):
            # Unpack state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
            x, y, z, x_dot, y_dot, z_dot, qw, qx, qy, qz, p, q, r = state

            # Unpack control inputs: [thrust_cmd, p_cmd, q_cmd, r_cmd]
            T, p_cmd, q_cmd, r_cmd = control

            # Rotation matrix from quaternion - only compute the elements we need
            # Third column (for thrust direction)
            R13 = 2.0 * (qx * qz + qw * qy)
            R23 = 2.0 * (qy * qz - qw * qx)
            R33 = 1.0 - 2.0 * (qx * qx + qy * qy)

            # Translational dynamics
            # Thrust vector rotated from body to inertial frame
            thrust_inertial_x = T * R13 / m
            thrust_inertial_y = T * R23 / m
            thrust_inertial_z = T * R33 / m

            # Linear accelerations
            x_ddot = thrust_inertial_x
            y_ddot = thrust_inertial_y
            z_ddot = thrust_inertial_z - g

            # Quaternion derivative
            # Converting body rates to quaternion derivative using the relation:
            # q_dot = 0.5 * q ⊗ [0, omega]
            qw_dot = 0.5 * (-qx * p - qy * q - qz * r)
            qx_dot = 0.5 * (qw * p + qz * q - qy * r)
            qy_dot = 0.5 * (qw * q + qx * r - qz * p)
            qz_dot = 0.5 * (qw * r + qy * p - qx * q)

            # Body rate dynamics (first-order model)
            tau = self.tau  # Time constant

            use_full_dynamics = True
            if use_full_dynamics:
                p_dot_d = (p_cmd - p) / tau
                q_dot_d = (q_cmd - q) / tau
                r_dot_d = (r_cmd - r) / tau

                w_ang_d = jnp.array([p_dot_d, q_dot_d, r_dot_d])
                current_rates = jnp.array([p, q, r])

                tau_d = self.inertia @ w_ang_d + jnp.cross(
                    current_rates, self.inertia @ current_rates
                )

                p_dot, q_dot, r_dot = self.inertia_inv @ (
                    tau_d - jnp.cross(current_rates, self.inertia @ current_rates)
                )
            else:
                p_dot = (p_cmd - p) / tau
                q_dot = (q_cmd - q) / tau
                r_dot = (r_cmd - r) / tau

            # Return state derivatives
            return jnp.array(
                [
                    x_dot,
                    y_dot,
                    z_dot,  # Position derivatives
                    x_ddot,
                    y_ddot,
                    z_ddot,  # Velocity derivatives
                    qw_dot,
                    qx_dot,
                    qy_dot,
                    qz_dot,  # Quaternion derivatives
                    p_dot,
                    q_dot,
                    r_dot,  # Angular rate derivatives
                ]
            )

        # RK4 integration implementation
        k1 = state_derivative(state)
        k2 = state_derivative(state + dt * 0.5 * k1)
        k3 = state_derivative(state + dt * 0.5 * k2)
        k4 = state_derivative(state + dt * k3)
        next_state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Normalize quaternion for better numerical stability
        q_norm = jnp.sqrt(
            next_state[6] ** 2
            + next_state[7] ** 2
            + next_state[8] ** 2
            + next_state[9] ** 2
        )
        next_state = next_state.at[6:10].set(next_state[6:10] / q_norm)

        return next_state

    def jax_mppi_3d(
        self,
        state,
        U,
        rng_key,
        ref_traj=None,
        n_samples=None,
        N=None,
        Q=None,
        R=None,
        R_rate=None,
        pose_constraints_weight=1e6,
        temperature=None,
    ):
        """
        MPPI controller for 3D quadrotor with quaternion representation

        Args:
            state: Current state vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
            U: Initial control sequence (horizon x 4)
            rng_key: JAX random key for sampling
            ref_traj: Reference trajectory (horizon x 13)
            n_samples: Number of samples (override self.n_samples)
            N: Horizon length (override self.horizon)
            Q: State cost matrix (override self.Q)
            R: Control cost matrix (override self.R)
            R_rate: Control rate cost matrix (override self.R_rate)
            pose_constraints_weight: Weight for pose constraints
            temperature: Temperature parameter for weighting (override self.temperature)

        Returns:
            tuple: (optimal control input, shifted control sequence, all sampled sequences, min_cost)
        """

        # Use default parameters if not provided
        if n_samples is None:
            n_samples = self.n_samples
        if N is None:
            N = self.horizon
        if Q is None:
            Q = self.Q
        if R is None:
            R = self.R
        if R_rate is None:
            R_rate = self.R_rate
        if temperature is None:
            temperature = self.temperature

        # Define control input limits
        control_min = self.control_min
        control_max = self.control_max

        # Get noise scale for controls
        ctrl_noise_scale = self.ctrl_noise_scale

        # Create default reference trajectory if none provided
        if ref_traj is None:
            ref_traj = jnp.tile(
                jnp.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                (N, 1),
            )

        # Compute quaternion error for cost function
        def quaternion_error(q1, q2):
            """
            Compute quaternion error using the approximation method from Minarik et al. 2024

            Args:
                q1: Reference quaternion [qw, qx, qy, qz]
                q2: Current quaternion [qw, qx, qy, qz]

            Returns:
                The squared distance between quaternions according to the approximation metric
            """
            # Calculate the quaternion inner product
            inner_product = (
                q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
            )

            # Use the approximation: dq(q1, q2) = 1 - <q1, q2>^2
            # This handles q and -q representing the same rotation
            return 1.0 - inner_product**2

        # FIXME: Maybe not necessary
        def attitude_constraint(x):
            """Constraint to limit attitude deviation with smoother gradient"""
            # Use quaternion's w component to compute the total rotation angle
            # The angle is 2*arccos(qw) for a unit quaternion
            qw = x[6]
            # Computing deviation angle with better numerical properties
            deviation = 2.0 * jnp.arccos(jnp.clip(jnp.abs(qw), 0.0, 1.0))
            max_angle = jnp.deg2rad(30.0)  # 30 degrees maximum deviation

            # Quadratic penalty for smoother gradient near constraint boundary
            violation = deviation - max_angle
            return jnp.where(violation > 0, violation**2, 0.0)

        def scan_fn(carry, inputs):
            """Function to scan over time steps"""
            cost, sim_state, prev_u, step = carry
            u = inputs

            # Get reference state for current time step
            q_ref = ref_traj[step]

            # State tracking cost
            # Position and velocity error
            pos_vel_error = sim_state[:6] - q_ref[:6]
            pos_vel_cost = jnp.dot(pos_vel_error, jnp.dot(Q[:6, :6], pos_vel_error))

            # Quaternion error
            q_error = quaternion_error(q_ref[6:10], sim_state[6:10])
            att_error = 2.0 * jnp.arccos(jnp.clip(jnp.abs(q_error), 0.0, 1.0))
            att_cost = 10.0 * att_error**2

            # Angular rate error
            rate_error = sim_state[10:] - q_ref[10:]
            rate_cost = jnp.dot(rate_error, jnp.dot(Q[10:, 10:], rate_error))

            # Control cost
            ctrl_cost = jnp.dot(u, jnp.dot(R, u))

            # Enhanced control rate cost
            ctrl_rate_cost = jnp.dot(u - prev_u, jnp.dot(R_rate, u - prev_u))

            # Total stage cost
            stage_cost = (
                pos_vel_cost + att_cost + rate_cost + ctrl_cost + ctrl_rate_cost
            )

            # Apply control limits
            u_clipped = jnp.clip(u, control_min, control_max)

            # Simulate next state
            next_state = self.jit_dynamics(sim_state, u_clipped)

            # Add attitude constraint penalty
            constraint_cost = pose_constraints_weight * attitude_constraint(next_state)

            # Update cumulative cost
            new_cost = cost + stage_cost + constraint_cost

            return (new_cost, next_state, u_clipped, step + 1), None

        def single_sample_cost(rng_subkey):
            """Compute cost for a single trajectory sample"""
            # Generate noise for control samples
            noise = jax.random.normal(rng_subkey, (N, 4))
            noise_scaled = noise * ctrl_noise_scale
            u_seq = U + noise_scaled

            # Clip control sequences to respect limits
            u_seq = jnp.clip(u_seq, control_min, control_max)

            # Initial carry value
            carry_init = (0.0, state, u_seq[0], 0)

            # Simulate trajectory and compute cost
            (final_cost, final_state, _, _), _ = jax.lax.scan(
                scan_fn, carry_init, u_seq
            )

            return final_cost, u_seq

        # Generate multiple trajectory samples
        rng_keys = jax.random.split(rng_key, n_samples)
        all_costs, all_seqs = jax.vmap(single_sample_cost)(rng_keys)

        # Add shape verification
        expected_all_seqs_shape = (n_samples, N, 4)
        all_seqs_shape = all_seqs.shape
        if self.logger:
            self.logger.debug(f"all_seqs shape: {all_seqs_shape}")
        else:
            logging.debug(f"all_seqs shape: {all_seqs_shape}")
        if all_seqs_shape != expected_all_seqs_shape:
            if self.logger:
                self.logger.warn(
                    f"all_seqs shape mismatch: expected {expected_all_seqs_shape}, got {all_seqs_shape}"
                )
            else:
                logging.warning(
                    f"all_seqs shape mismatch: expected {expected_all_seqs_shape}, got {all_seqs_shape}"
                )

        # Compute optimal control using softmax weighting with normalized weights
        min_cost = jnp.min(all_costs)  # rho in Minarik 2024
        exp_costs = jnp.exp(
            -(1 / temperature) * (all_costs - min_cost)
        )  # omega in Minarik 2024

        # Normalize weights to sum to 1 (eta in Minarik 2024)
        eta = jnp.sum(exp_costs)
        weights = exp_costs / eta  # Normalized weights

        # Check weights shape
        expected_weights_shape = (n_samples,)
        weights_shape = weights.shape
        if weights_shape != expected_weights_shape:
            if self.logger:
                self.logger.warn(
                    f"weights shape mismatch: expected {expected_weights_shape}, got {weights_shape}"
                )
            else:
                logging.warning(
                    f"weights shape mismatch: expected {expected_weights_shape}, got {weights_shape}"
                )

        # Calculate weighted average control sequence
        best_U = jnp.sum(weights[:, jnp.newaxis, jnp.newaxis] * all_seqs, axis=0)

        # Apply control limits to final controls
        best_U = jnp.clip(best_U, control_min, control_max)

        # Shift control sequence for next iteration (receding horizon)
        new_U = jnp.roll(best_U, shift=-1, axis=0)
        new_U = new_U.at[-1].set(new_U[-2])  # Repeat last control input

        return best_U[0], new_U, all_seqs, min_cost

    def get_rng_key(self):
        """
        Get a new random key and update the stored key.

        Returns:
            A new JAX random subkey for randomization, falls back to NumPy if JAX fails
        """
        try:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            return subkey
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error generating random key: {str(e)}")
            return np.random.randint(0, 1000000)

    def get_control(self, state, ref_traj=None):
        """
        Get control input for the current state

        Args:
            state: Current state vector [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
            ref_traj: Reference trajectory (horizon x 13)

        Returns:
            tuple: (optimal control input, computation time in seconds, min cost value)
        """
        # Get new random key
        rng_key = self.get_rng_key()

        # Convert to JAX array if needed
        if isinstance(state, np.ndarray):
            state = jnp.array(state)

        if isinstance(self.control_sequence, np.ndarray):
            control_seq = jnp.array(self.control_sequence)
        else:
            control_seq = self.control_sequence

        # Time the computation
        import time

        start_time = time.time()

        # Call MPPI controller - now receiving the min_cost as well
        control_command, self.control_sequence, _, min_cost = self.jit_mppi(
            state, control_seq, rng_key, ref_traj
        )

        # Convert to numpy array
        if isinstance(control_command, jnp.ndarray):
            control_command = np.array(control_command)
        if isinstance(self.control_sequence, jnp.ndarray):
            self.control_sequence = np.array(self.control_sequence)
        if isinstance(min_cost, jnp.ndarray):
            min_cost = float(min_cost)

        end_time = time.time()
        compute_time = end_time - start_time

        return control_command, compute_time, min_cost