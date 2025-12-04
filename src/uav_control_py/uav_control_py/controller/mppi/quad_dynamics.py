#!/usr/bin/env python3
"""
dynamics.py
Pure JAX functions for UAV dynamics and RK4 integration.
"""

import jax.numpy as jnp


def dynamics_3d(state, control, mass, g, tau, inertia_inv):
    """
    Computes the state derivative x_dot = f(x, u).

    Args:
        state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
        control: [thrust, p_cmd, q_cmd, r_cmd]
        mass: UAV mass
        g: Gravity
        tau: Actuator time constant
        inertia_inv: Inverse of inertia matrix
    """
    # Unpack state
    x, y, z, x_dot, y_dot, z_dot, qw, qx, qy, qz, p, q, r = state

    # Unpack control
    T, p_cmd, q_cmd, r_cmd = control

    # Rotation matrix from quaternion (Body to Inertial)
    # Only computing 3rd column (z-axis) for thrust
    R13 = 2.0 * (qx * qz + qw * qy)
    R23 = 2.0 * (qy * qz - qw * qx)
    R33 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Translational dynamics
    x_ddot = (T * R13) / mass
    y_ddot = (T * R23) / mass
    z_ddot = (T * R33) / mass - g

    # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, omega]
    qw_dot = 0.5 * (-qx * p - qy * q - qz * r)
    qx_dot = 0.5 * (qw * p + qz * q - qy * r)
    qy_dot = 0.5 * (qw * q + qx * r - qz * p)
    qz_dot = 0.5 * (qw * r + qy * p - qx * q)

    # Body rate dynamics (First-order actuator response)
    p_dot = (p_cmd - p) / tau
    q_dot = (q_cmd - q) / tau
    r_dot = (r_cmd - r) / tau

    return jnp.array(
        [
            x_dot,
            y_dot,
            z_dot,
            x_ddot,
            y_ddot,
            z_ddot,
            qw_dot,
            qx_dot,
            qy_dot,
            qz_dot,
            p_dot,
            q_dot,
            r_dot,
        ]
    )


def dynamics_rk4(state, control, dt, mass, g, tau, inertia_inv):
    """
    Performs RK4 integration step.
    """
    k1 = dynamics_3d(state, control, mass, g, tau, inertia_inv)
    k2 = dynamics_3d(state + dt * 0.5 * k1, control, mass, g, tau, inertia_inv)
    k3 = dynamics_3d(state + dt * 0.5 * k2, control, mass, g, tau, inertia_inv)
    k4 = dynamics_3d(state + dt * k3, control, mass, g, tau, inertia_inv)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Normalize quaternion
    q_norm = jnp.sqrt(jnp.sum(next_state[6:10] ** 2))
    next_state = next_state.at[6:10].set(next_state[6:10] / q_norm)

    return next_state