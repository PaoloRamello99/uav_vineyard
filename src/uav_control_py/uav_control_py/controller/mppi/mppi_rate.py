#!/usr/bin/env python3
"""
mppi_rate.py
Main controller class combining Dynamics, Costs, and MPPI logic.
"""

import functools
import logging
import time

# JAX Imports
import jax
import jax.numpy as jnp
import numpy as np

from . import costs
from . import quad_dynamics as dynamics

# Configure JAX logging
logging.getLogger("jax").setLevel(logging.INFO)


class MPPIRateController:
    """
    Model Predictive Path Integral (MPPI) controller for UAV rate control.
    """

    def __init__(self, config, logger=None):
        self.logger = logger
        self._parse_config(config)

        # Initialize RNG
        self.rng_key = jax.random.PRNGKey(0)

        # Initialize control sequence (Warm start)
        hover_thrust = self.mass * self.g
        self.control_sequence = jnp.zeros((self.horizon, 4), dtype=jnp.float32)
        self.control_sequence = self.control_sequence.at[:, 0].set(hover_thrust)

        # Pre-compile the main MPPI loop
        print("Compiling MPPI JAX functions...")

        # 1. Bake in the static arguments using partial
        # This creates a new function that strictly expects ONLY the dynamic JAX arrays
        mppi_step_configured = functools.partial(
            self._mppi_optimize_step,
            n_samples=self.n_samples,
            horizon=self.horizon,
            temperature=self.temperature,
            pose_constraint_weight=self.pose_constraint_weight,
        )

        # 2. JIT the configured function
        # Note: We no longer need static_argnames because we removed them from the signature!
        self.jit_mppi_step = jax.jit(mppi_step_configured)

        print("Compilation complete.")

    def _parse_config(self, config):
        """
        Parse configuration and set up parameters.
        Raises KeyError if a required parameter is missing.
        """
        print("Parsing MPPI configuration...")
        try:
            # --- STATIC ARGUMENTS (Must be Python int/float for JAX JIT) ---
            self.n_samples = int(config["n_samples"])
            self.horizon = int(config["horizon"])
            self.temperature = float(config["temperature"])
            self.pose_constraint_weight = float(1e6)  # Keep this as a hardcoded default

            # --- DYNAMIC ARGUMENTS (JAX Arrays on GPU) ---
            self.dt = jnp.float32(config["dt"])

            # Physics Parameters
            self.mass = jnp.float32(config["mass"])
            self.g = jnp.float32(config["g"])
            inertia_matrix = jnp.array(config["inertia"], dtype=jnp.float32)
            self.tau = jnp.float32(config["tau"])
            self.inertia_inv = jnp.linalg.inv(inertia_matrix)

            # Constraints
            min_thrust = jnp.float32(config["min_thrust"])
            self.max_thrust = jnp.float32(config["max_thrust"])

            angular_rate_min_list = config["angular_rate_min"]
            angular_rate_max_list = config["angular_rate_max"]

            self.control_min = jnp.array(
                [min_thrust] + angular_rate_min_list, dtype=jnp.float32
            )
            self.control_max = jnp.array(
                [self.max_thrust] + angular_rate_max_list, dtype=jnp.float32
            )

            self.ctrl_noise_scale = jnp.array(
                config["ctrl_noise_scale"], dtype=jnp.float32
            )

            # Cost Weights
            Q_diag = jnp.array(config["Q_diag"], dtype=jnp.float32)
            R_diag = jnp.array(config["R_diag"], dtype=jnp.float32)
            R_rate_diag = jnp.array(config["R_rate_diag"], dtype=jnp.float32)

            # Create matrices
            Q = jnp.diag(Q_diag)
            self.Q_pos = Q[:6, :6]
            self.Q_rate = Q[10:, 10:]
            self.R = jnp.diag(R_diag)
            self.R_rate = jnp.diag(R_rate_diag)

            print("Successfully parsed and validated all MPPI parameters.")

        except KeyError as e:
            raise KeyError(f"Missing required parameter in config: {e}")

    @staticmethod
    def _mppi_optimize_step(
        state,
        U_seq,
        key,
        ref_traj,
        dt,
        Q_pos,
        Q_rate,
        R,
        R_rate,
        mass,
        g,
        tau,
        inertia_inv,
        ctrl_noise_scale,
        control_min,
        control_max,
        # Static parameters
        n_samples,
        horizon,
        temperature,
        pose_constraint_weight,
    ):
        """
        The core JAX function that performs rollouts and computes the new control sequence.
        """
        # 1. Noise Generation
        noise = jax.random.normal(key, (n_samples, horizon, 4))
        noise_scaled = noise * ctrl_noise_scale

        # Broadcast U_seq to (n_samples, horizon, 4) and add noise
        u_perturbed = U_seq + noise_scaled
        u_perturbed = jnp.clip(u_perturbed, control_min, control_max)

        # CRITICAL FIX: Create a partially applied cost function with weight matrices bound
        cost_fn = functools.partial(
            costs.calculate_stage_cost, Q_pos=Q_pos, Q_rate=Q_rate, R=R, R_rate=R_rate
        )

        # 2. Rollout Function (lax.scan)
        def rollout_fn(carry, inputs):
            curr_state, prev_u, step_idx, running_cost = carry
            u_input, ref_state = inputs

            # Compute Cost using the partially applied function
            stage_cost = cost_fn(curr_state, u_input, prev_u, ref_state)

            # Dynamics Step
            next_state = dynamics.dynamics_rk4(
                curr_state, u_input, dt, mass, g, tau, inertia_inv
            )

            # Constraint Cost
            constr_cost = pose_constraint_weight * costs.attitude_constraint(next_state)

            new_carry = (
                next_state,
                u_input,
                step_idx + 1,
                running_cost + stage_cost + constr_cost,
            )
            return new_carry, None

        # 3. Vectorized Rollout (vmap over samples)
        def single_rollout(u_seq_sample):
            init_carry = (state, u_seq_sample[0], 0, 0.0)

            scan_inputs = (u_seq_sample, ref_traj)

            (final_state, _, _, total_cost), _ = jax.lax.scan(
                rollout_fn, init_carry, scan_inputs
            )
            return total_cost

        all_costs = jax.vmap(single_rollout)(u_perturbed)

        # 4. MPPI Weighting & Update
        min_cost = jnp.min(all_costs)
        exp_costs = jnp.exp(-(1.0 / temperature) * (all_costs - min_cost))
        weights = exp_costs / (jnp.sum(exp_costs) + 1e-10)

        # Weighted average of the perturbed sequences
        weighted_seqs = weights[:, None, None] * u_perturbed
        best_U = jnp.sum(weighted_seqs, axis=0)

        return best_U[0], best_U, min_cost

    def get_control(self, state, ref_traj=None):
        """
        Main public interface to get the next control input.
        """
        # 1. Ensure ref_traj is numpy (CPU side) for easy manipulation
        if ref_traj is None:
            ref_traj = np.zeros((self.horizon, 13), dtype=np.float32)
            ref_traj[:, 6] = 1.0
        else:
            ref_traj = np.array(ref_traj, dtype=np.float32)

        # 2. Manual Padding (CPU side)
        current_len = ref_traj.shape[0]
        if current_len < self.horizon:
            # Pad with the last state
            padding = np.tile(ref_traj[-1], (self.horizon - current_len, 1))
            ref_traj = np.vstack((ref_traj, padding))
        elif current_len > self.horizon:
            # Truncate
            ref_traj = ref_traj[: self.horizon]

        # 3. Send to JAX
        ref_traj = jnp.asarray(ref_traj, dtype=jnp.float32)

        # Handle Data Types
        state = jnp.asarray(state, dtype=jnp.float32)

        # RNG update
        self.rng_key, subkey = jax.random.split(self.rng_key)

        start_time = time.time()

        # Debug: Check for NaN/Inf before calling JIT
        if jnp.any(jnp.isnan(state)) or jnp.any(jnp.isinf(state)):
            print(f"ERROR: Invalid state before JIT: {state}")
        if jnp.any(jnp.isnan(self.control_sequence)) or jnp.any(
            jnp.isinf(self.control_sequence)
        ):
            print(
                f"ERROR: Invalid control_sequence before JIT: {self.control_sequence}"
            )

        # Run MPPI Optimization
        control_cmd, best_seq, min_cost = self.jit_mppi_step(
            state,
            self.control_sequence,
            subkey,
            ref_traj,
            self.dt,
            self.Q_pos,
            self.Q_rate,
            self.R,
            self.R_rate,
            self.mass,
            self.g,
            self.tau,
            self.inertia_inv,
            self.ctrl_noise_scale,
            self.control_min,
            self.control_max,
        )

        # Update internal state (Receding Horizon)
        self.control_sequence = jnp.roll(best_seq, shift=-1, axis=0)
        self.control_sequence = self.control_sequence.at[-1].set(best_seq[-1])

        comp_time = time.time() - start_time

        return np.array(control_cmd), comp_time, float(min_cost)