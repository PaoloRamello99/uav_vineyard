#!/usr/bin/env python3
"""
simulation_mppi.py
Main entry point. Runs MPPI simulation and calls the visualization module.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Set JAX environment variables BEFORE importing JAX
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# Suppress JAX warnings about missing TPU/GPU
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_debug_nans", True)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Local Imports
from uav_control_py.config.config_loader import load_mppi_config
from uav_control_py.controller.mppi.mppi_rate import MPPIRateController
from uav_control_py.controller.mppi.quad_dynamics import dynamics_rk4
from visualization.plotter import MPPIVisualizer


def generate_lemniscate_reference(t, center=[0.0, 0.0, 2.0], scale=10.0, period=15.0):
    """Generates a figure-8 reference state."""
    theta = 2 * np.pi * t / period
    denom = 1.0 + np.sin(theta) ** 2

    # Position
    x = center[0] + scale * np.cos(theta) / denom
    y = center[1] + scale * np.sin(theta) * np.cos(theta) / denom
    z = center[2]

    # Velocity (Approximate)
    dt_small = 0.01
    theta_next = 2 * np.pi * (t + dt_small) / period
    denom_next = 1.0 + np.sin(theta_next) ** 2
    x_next = center[0] + scale * np.cos(theta_next) / denom_next
    y_next = center[1] + scale * np.sin(theta_next) * np.cos(theta_next) / denom_next

    vx = (x_next - x) / dt_small
    vy = (y_next - y) / dt_small
    vz = 0.0

    # Full state: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    return np.array(
        [x, y, z, vx, vy, vz, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )


def run_simulation():
    # 1. Setup
    print("Initializing MPPI Controller...")
    try:
        config = load_mppi_config()
        controller = MPPIRateController(config)
    except (FileNotFoundError, KeyError) as e:
        print("\n--- CONFIGURATION ERROR ---")
        print(f"Failed to load or parse 'mppi_config.yaml': {e}")
        print("Please ensure the file exists and all required parameters are set.")
        print("---------------------------\n")
        sys.exit(1)

    # Simulation settings
    T_sim = 60.0
    dt_sim = 0.02
    steps = int(T_sim / dt_sim)

    # Initial Condition
    current_state = np.array(
        [-1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    # Logging
    history = {"state": [], "ref": [], "control": [], "time": [], "cost": []}

    # 2. Main Loop
    print(f"Starting Simulation ({T_sim}s)...")
    start_time = time.time()

    for k in range(steps):
        t = k * dt_sim

        # Generate Horizon Reference (N x 13)
        ref_horizon = []
        for i in range(config["horizon"]):
            ref_horizon.append(generate_lemniscate_reference(t + i * dt_sim))
        ref_horizon = np.array(ref_horizon)

        # Get Control
        control, _, min_cost = controller.get_control(current_state, ref_horizon)

        # Update Physics (Simulate one step)
        # Convert to JAX for dynamics function
        state_jax = jnp.array(current_state)
        ctrl_jax = jnp.array(control)
        next_state_jax = dynamics_rk4(
            state_jax,
            ctrl_jax,
            dt_sim,
            controller.mass,
            controller.g,
            controller.tau,
            controller.inertia_inv,
        )

        current_state = np.array(next_state_jax)

        # Store Data
        history["state"].append(current_state)
        history["ref"].append(ref_horizon[0])
        history["control"].append(control)
        history["time"].append(t)
        history["cost"].append(min_cost)

        if k % 20 == 0:
            print(f"Step {k}/{steps} | Time: {t:.2f}s | Cost: {min_cost:.2f}")

    print(f"Simulation Complete. Total Time: {time.time() - start_time:.2f}s")

    # Convert to numpy arrays for plotting
    for key in history:
        history[key] = np.array(history[key])

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MPPI simulation and visualization."
    )
    # Master switch
    parser.add_argument("--no-vis", action="store_true", help="Skip all visualization.")

    # General plot options
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots interactively (blocks execution).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable saving plots and animations."
    )

    # Granular plot control
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable the 2D position tracking plot.",
    )
    parser.add_argument(
        "--no-3d", action="store_true", help="Disable the static 3D trajectory plot."
    )
    parser.add_argument(
        "--plot-control",
        action="store_true",
        help="Enable the control inputs plot (off by default).",
    )
    parser.add_argument(
        "--make-animation",
        action="store_true",
        help="Generate and save a 3D animation (off by default).",
    )

    args = parser.parse_args()

    # Run Simulation
    sim_data = run_simulation()

    # Run Visualization
    if not args.no_vis:
        print("\nStarting Visualization...")
        vis = MPPIVisualizer()

        save_enabled = not args.no_save
        show_enabled = args.show_plots

        # 1. Plot static graphs (position is on by default)
        if not args.no_history:
            vis.plot_history(
                sim_data, save=save_enabled, plot_control=args.plot_control
            )

        # 2. Plot 3D trajectory (on by default)
        if not args.no_3d:
            vis.plot_3d_trajectory(sim_data, save=save_enabled)

        # 3. Create Animation (off by default)
        if args.make_animation:
            vis.make_animation(sim_data, save=save_enabled)

        # Show all generated plots if enabled
        if show_enabled:
            print("[Visualizer] Showing plots...")
            plt.show()