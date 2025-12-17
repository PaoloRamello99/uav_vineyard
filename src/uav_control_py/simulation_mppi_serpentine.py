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


def generate_mission_reference(
    t,
    home=np.array([0.0, 0.0, 0.0]),
    first_row=np.array([-10.0, 20.0]),
    altitude=2.5,
    row_length=20.0,
    row_spacing=2.5,
    num_rows=10,
    v_ref=2.0,
    T_takeoff=3.0,
    T_landing=3.0,
):
    """
    Missione UAV con serpentina:
    - 1 filare = tratto rettilineo + 1 turn
    - direzione determinata dalla parità del filare
    """

    # =========================
    # PRE-COMPUTAZIONI
    # =========================
    R = row_spacing / 2.0

    T_row = row_length / v_ref
    T_turn = np.pi * R / v_ref
    T_cycle = T_row + T_turn

    T_serpentine = num_rows * T_cycle

    T_to_vineyard = np.linalg.norm(first_row - home[:2]) / v_ref
    T_return = T_to_vineyard

    t1 = T_takeoff
    t2 = t1 + T_to_vineyard
    t3 = t2 + T_serpentine
    t4 = t3 + T_return
    t5 = t4 + T_landing

    # =========================
    # 1) TAKEOFF
    # =========================
    if t < t1:
        z = home[2] + altitude * (t / T_takeoff)
        return np.array([home[0], home[1], z,
                         0.0, 0.0, altitude / T_takeoff,
                         1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # =========================
    # 2) TRANSITO AL VIGNETO
    # =========================
    elif t < t2:
        tau = (t - t1) / T_to_vineyard
        pos = home[:2] + tau * (first_row - home[:2])
        vel = v_ref * (first_row - home[:2]) / np.linalg.norm(first_row - home[:2])

        return np.array([pos[0], pos[1], altitude,
                         vel[0], vel[1], 0.0,
                         1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # =========================
    # 3) SERPENTINA
    # =========================
    elif t < t3:
        ts = t - t2

        row_idx = int(ts // T_cycle)
        row_idx = min(row_idx, num_rows - 1)

        tau = ts - row_idx * T_cycle

        y_row = first_row[1] + row_idx * row_spacing
        direction = 1 if row_idx % 2 == 0 else -1

        # ---- RETTILINEO ----
        if tau < T_row:
            s = v_ref * tau

            if direction == 1:
                x = first_row[0] + s
                vx = v_ref
            else:
                x = first_row[0] + row_length - s
                vx = -v_ref

            y = y_row
            vy = 0.0

        # ---- TURN (verso filare successivo) ----
        else:
            if row_idx == num_rows - 1:
                # ultimo filare
                s = row_length
                if direction == 1:
                    x = first_row[0] + s
                    vx = v_ref
                else:
                    x = first_row[0]
                    vx = -v_ref

                y = y_row
                vy = 0.0
            else:
                t_turn = tau - T_row
                theta = np.pi * t_turn / T_turn

                if direction == 1:
                    x_c = first_row[0] + row_length
                else:
                    x_c = first_row[0]

                y_c = y_row + R

                x = x_c + direction * R * np.sin(theta)
                y = y_c - R * np.cos(theta)

                vx = direction * v_ref * np.cos(theta)
                vy = v_ref * np.sin(theta)

        return np.array([x, y, altitude,
                         vx, vy, 0.0,
                         1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # =========================
    # 4) RITORNO A HOME
    # =========================
    elif t < t4:

        tau = (t - t3) / T_return
        tau = np.clip(tau, 0.0, 1.0)

        # ultima posizione serpentina
        last_row = num_rows - 1
        last_y = first_row[1] + last_row * row_spacing
        direction = 1 if last_row % 2 == 0 else -1

        if direction == 1:
            last_x = first_row[0] + row_length
        else:
            last_x = first_row[0]

        start = np.array([last_x, last_y])
        end   = home[:2]

        pos = start + tau * (end - start)
        dir_vec = end - start
        dir_vec /= np.linalg.norm(dir_vec)

        vel = v_ref * dir_vec

        return np.array([
            pos[0], pos[1], altitude,
            vel[0], vel[1], 0.0,
            1, 0, 0, 0, 0, 0, 0
        ], dtype=np.float32)


    # =========================
    # 5) LANDING (discesa verticale su HOME)
    # =========================
    elif t < t5:

        tau = (t - t4) / T_landing
        tau = np.clip(tau, 0.0, 1.0)

        z = altitude * (1 - tau)

        return np.array([
            home[0], home[1], z,
            0.0, 0.0, -altitude / T_landing,
            1, 0, 0, 0, 0, 0, 0
        ], dtype=np.float32)

    # =========================
    # FINE MISSIONE
    # =========================
    else:
        return np.array([home[0], home[1], home[2],
                        0.0, 0.0, 0.0,
                        1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    

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
    T_sim = 160.0
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
            ref_horizon.append(generate_mission_reference(t + i * dt_sim))
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