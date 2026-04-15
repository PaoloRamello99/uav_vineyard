# Autonomous UAV Navigation & Predictive Control Architecture

[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![PX4](https://img.shields.io/badge/PX4-Autopilot-brightgreen.svg)](https://px4.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![DevContainers](https://img.shields.io/badge/DevContainers-Supported-blue.svg)](https://containers.dev/)

## Overview
This repository contains the software architecture and control algorithms for the autonomous navigation of a multirotor Unmanned Aerial Vehicle (UAV) in complex, non-convex environments (specifically developed for vineyard obstacle avoidance and path tracking). 

The system bypasses traditional reactive PID control, utilizing a custom **Model Predictive Path Integral (MPPI)** controller to ensure safe, optimized, and predictive flight dynamics. The architecture heavily relies on **ROS 2** for high-level intelligence and mission logic, interfaced with **PX4 Autopilot** for low-level flight stabilization.

## Key Features
* **MPPI Predictive Control:** A Python-based stochastic optimal control algorithm that generates and evaluates thousands of forward-simulated trajectories in real-time (using Monte Carlo sampling) to bypass local minima and ensure strict obstacle avoidance.
* **Non-Linear Dynamics Modeling:** Incorporates a 4th-order Runge-Kutta (RK4) integrator for high-fidelity state prediction within the MPPI receding horizon.
* **Hierarchical Finite State Machine (FSM):** Dual-layer FSM managing the entire UAV lifecycle (Takeoff, Holding, Landing) and the dynamic mission trajectory (Continuous snake-like path generation with smooth semicircular turns).
* **Software-In-The-Loop (SITL) & Processor-In-The-Loop (PITL):** Full simulation pipeline validated in **Gazebo 3D** using custom `.sdf` environments. Performance and computational load tested on real hardware (NVIDIA Jetson Orin Nano).
* **Containerized Development:** The entire environment is containerized using **Docker** and **Dev Containers** to ensure strict reproducibility and dependency management across different machines.

## System Architecture
The software stack is decoupled to separate the mission intelligence from the hardware control:
1. **High-Level Control (ROS 2):** Python nodes calculate the reference trajectory, evaluate the MPPI cost functions, and publish normalized thrust and body rates.
2. **Middleware:** `MicroXRCE-DDS` acts as a bridge, translating ROS 2 topics (ENU/FLU frames) into PX4 `uORB` messages (NED/FRD frames) in real-time.
3. **Low-Level Control (PX4):** Operates in `Offboard` mode, receiving rate commands at 50 Hz and translating them into actuator outputs.

## Project Structure
```text
uav_vineyard/
├── .devcontainer/             # Docker and DevContainer configuration files
├── src/
│   ├── serpentine_node.py     # Trajectory generator and FSM logic
│   ├── uav_offboard_mppi.py   # ROS 2 node interfacing MPPI with PX4 via Offboard mode
│   ├── mppi_rate.py           # Core MPPI controller and stochastic sampling
│   ├── dynamics.py            # Mathematical model of the UAV (RK4 integration)
│   └── costs.py               # Cost functions (Tracking, Attitude, Control effort)
├── worlds/
│   └── vineyard_world.sdf     # Custom Gazebo 3D environment for SITL testing
└── launch/
    └── simulation_launch.py   # Automated launch file for Gazebo, PX4, and ROS 2 nodes
