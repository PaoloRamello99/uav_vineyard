#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


class MPPIVisualizer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.media_dir = os.path.join(base_dir, "media")
        self.img_dir = os.path.join(self.media_dir, "plots")
        self.vid_dir = os.path.join(self.media_dir, "videos")

        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.vid_dir, exist_ok=True)

    def plot_history(self, history, save=True, show=False, plot_control=False):
        """
        Plots position tracking and optionally control inputs.
        history: dict containing 'state', 'ref', 'control', 'time'
        """
        t = history["time"]
        state = history["state"]
        ref = history["ref"]

        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        labels = ["x", "y", "z"]

        for i in range(3):
            ax[i].plot(t, state[:, i], label="Actual", linewidth=2)
            ax[i].plot(t, ref[:, i], "r--", label="Reference", alpha=0.7)
            ax[i].set_ylabel(f"{labels[i]} (m)")
            ax[i].grid(True)

        ax[0].legend(loc="upper right")
        ax[2].set_xlabel("Time (s)")
        fig.suptitle("MPPI Position Tracking")
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.img_dir, "position_tracking.png")
            plt.savefig(save_path)
            print(f"[Visualizer] Saved plot to {save_path}")

        if plot_control:
            control = history["control"]
            fig2, ax2 = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
            ctrl_labels = ["Thrust (N)", "Roll Cmd", "Pitch Cmd", "Yaw Cmd"]

            for i in range(4):
                ax2[i].plot(t, control[:, i], "g-", linewidth=1.5)
                ax2[i].set_ylabel(ctrl_labels[i])
                ax2[i].grid(True)

            ax2[3].set_xlabel("Time (s)")
            fig2.suptitle("Control Inputs")
            plt.tight_layout()

            if save:
                save_path = os.path.join(self.img_dir, "control_inputs.png")
                plt.savefig(save_path)
                print(f"[Visualizer] Saved plot to {save_path}")

    def plot_3d_trajectory(self, history, save=True, show=False):
        """
        Creates a static 3D visualization of the trajectory.
        Shows the actual path vs reference path with start/end markers.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        state = history["state"]
        ref = history["ref"]

        ax.plot(
            ref[:, 0],
            ref[:, 1],
            ref[:, 2],
            "r--",
            linewidth=2,
            alpha=0.6,
            label="Reference",
        )

        ax.plot(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            "b-",
            linewidth=2,
            label="Actual",
        )

        ax.scatter(
            state[0, 0],
            state[0, 1],
            state[0, 2],
            c="g",
            marker="o",
            s=100,
            label="Start",
        )
        ax.scatter(
            state[-1, 0],
            state[-1, 1],
            state[-1, 2],
            c="r",
            marker="X",
            s=100,
            label="End",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Trajectory Visualization", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        max_range = (
            np.array(
                [
                    state[:, 0].max() - state[:, 0].min(),
                    state[:, 1].max() - state[:, 1].min(),
                    state[:, 2].max() - state[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (state[:, 0].max() + state[:, 0].min()) * 0.5
        mid_y = (state[:, 1].max() + state[:, 1].min()) * 0.5
        mid_z = (state[:, 2].max() + state[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=25, azim=45)

        plt.tight_layout()

        if save:
            save_path = os.path.join(self.img_dir, "trajectory_3d.png")
            plt.savefig(save_path, dpi=150)
            print(f"[Visualizer] Saved 3D trajectory plot to {save_path}")

    def make_animation(self, history, save=True, filename="simulation.mp4"):
        """Creates a 3D animation of the trajectory."""
        print("[Visualizer] Generating animation (this may take a moment)...")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        skip = 2
        states = history["state"][::skip]
        refs = history["ref"][::skip]
        times = history["time"][::skip]

        ax.plot(
            history["ref"][:, 0],
            history["ref"][:, 1],
            history["ref"][:, 2],
            "r--",
            alpha=0.3,
            linewidth=1.5,
            label="Reference Path",
        )

        (traj_line,) = ax.plot([], [], [], "b-", linewidth=2, label="Drone Path")
        (drone_point,) = ax.plot(
            [], [], [], "bo", markersize=10, markeredgecolor="k", markeredgewidth=1
        )
        (target_point,) = ax.plot(
            [], [], [], "r*", markersize=15, label="Current Target"
        )

        trail_length = 20
        trail_lines = []
        for i in range(trail_length):
            alpha = (i + 1) / trail_length * 0.5
            (line,) = ax.plot([], [], [], "b-", linewidth=1, alpha=alpha)
            trail_lines.append(line)

        time_text = ax.text2D(
            0.05, 0.95, "", transform=ax.transAxes, fontsize=12, fontweight="bold"
        )

        all_positions = np.vstack([states[:, :3], refs[:, :3]])
        margin = 0.5
        ax.set_xlim(
            all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin
        )
        ax.set_ylim(
            all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin
        )
        ax.set_zlim(
            all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin
        )

        ax.set_xlabel("X (m)", fontsize=11)
        ax.set_ylabel("Y (m)", fontsize=11)
        ax.set_zlabel("Z (m)", fontsize=11)
        ax.set_title("MPPI Controller Simulation", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax.view_init(elev=25, azim=45)

        def init():
            traj_line.set_data([], [])
            traj_line.set_3d_properties([])
            drone_point.set_data([], [])
            drone_point.set_3d_properties([])
            target_point.set_data([], [])
            target_point.set_3d_properties([])
            for line in trail_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            time_text.set_text("")
            return [traj_line, drone_point, target_point, time_text] + trail_lines

        def animate(frame):
            x, y, z = states[frame, 0], states[frame, 1], states[frame, 2]

            traj_line.set_data(states[:frame, 0], states[:frame, 1])
            traj_line.set_3d_properties(states[:frame, 2])

            drone_point.set_data([x], [y])
            drone_point.set_3d_properties([z])

            tx, ty, tz = refs[frame, 0], refs[frame, 1], refs[frame, 2]
            target_point.set_data([tx], [ty])
            target_point.set_3d_properties([tz])

            for i, line in enumerate(trail_lines):
                trail_start = max(0, frame - trail_length + i)
                trail_end = max(0, frame - trail_length + i + 1)
                if trail_start < frame:
                    line.set_data(
                        states[trail_start:trail_end, 0],
                        states[trail_start:trail_end, 1],
                    )
                    line.set_3d_properties(states[trail_start:trail_end, 2])

            time_text.set_text(f"Time: {times[frame]:.2f} s")

            return [traj_line, drone_point, target_point, time_text] + trail_lines

        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(states),
            interval=50,
            blit=False,
            repeat=True,
        )

        if save:
            save_path = os.path.join(self.vid_dir, filename)
            try:
                writer = FFMpegWriter(fps=20, bitrate=1800)
                anim.save(save_path, writer=writer)
                print(f"[Visualizer] Saved animation to {save_path}")
            except Exception as e:
                print(f"[Visualizer] Warning: Could not save animation: {e}")
                print(
                    "[Visualizer] Make sure ffmpeg is installed: 'sudo apt install ffmpeg'"
                )

        return anim