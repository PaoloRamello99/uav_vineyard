#!/usr/bin/env python3
"""
costs.py
Cost functions and quaternion math helpers.
"""

import jax.numpy as jnp


def quaternion_error(q_ref, q_curr):
    """
    Compute quaternion error (1 - <q1, q2>^2).
    Approximation from Minarik et al. 2024.
    """
    q_ref = jnp.reshape(q_ref, (4,))
    q_curr = jnp.reshape(q_curr, (4,))

    inner_product = jnp.sum(q_ref * q_curr)
    inner_product = jnp.clip(inner_product, -1.0, 1.0)
    return 1.0 - inner_product**2


def attitude_constraint(state, max_angle_deg=30.0):
    """
    Soft constraint to limit tilt angle.
    """
    qw = state[6]
    deviation = 2.0 * jnp.arccos(jnp.clip(jnp.abs(qw), 0.0, 1.0))
    max_angle = jnp.deg2rad(max_angle_deg)
    violation = deviation - max_angle
    return jnp.where(violation > 0, violation**2, 0.0)


def calculate_stage_cost(state, u, prev_u, ref, Q_pos, Q_rate, R, R_rate):
    """
    Calculates the cost for a single time step.
    """
    # 1. Position and Velocity Cost
    pos_vel_err = state[:6] - ref[:6]
    pos_vel_cost = jnp.sum(pos_vel_err * jnp.dot(Q_pos, pos_vel_err))

    # 2. Attitude Cost
    q_err = quaternion_error(ref[6:10], state[6:10])
    att_cost = 10.0 * q_err

    # 3. Angular Rate Cost
    rate_err = state[10:13] - ref[10:13]
    rate_cost = jnp.sum(rate_err * jnp.dot(Q_rate, rate_err))

    # 4. Control Input Cost
    ctrl_cost = jnp.sum(u * jnp.dot(R, u))

    # 5. Control Rate Cost (Smoothness)
    u_diff = u - prev_u
    ctrl_rate_cost = jnp.sum(u_diff * jnp.dot(R_rate, u_diff))

    # Total Stage Cost
    return pos_vel_cost + att_cost + rate_cost + ctrl_cost + ctrl_rate_cost


def terminal_cost(state_T, ref_T, Q_pos_f, Q_rate_f, w_att_f):
    """
    Terminal cost function.
    """
    # 1. Position and Velocity Cost
    pos_vel_err = state_T[:6] - ref_T[:6]
    pos_vel_cost = jnp.sum(pos_vel_err * jnp.dot(Q_pos_f, pos_vel_err))

    # 2. Attitude Cost
    q_err = quaternion_error(ref_T[6:10], state_T[6:10])
    att_cost = w_att_f * q_err

    # 3. Angular Rate Cost
    rate_err = state_T[10:13] - ref_T[10:13]
    rate_cost = jnp.sum(rate_err * jnp.dot(Q_rate_f, rate_err))

    # Total Terminal Cost
    return pos_vel_cost + att_cost + rate_cost