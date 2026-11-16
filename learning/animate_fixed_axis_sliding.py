"""Animate fixed-axis Coulomb sliding with translation and yaw.

Run ``python learning/animate_fixed_axis_sliding.py``; use ``--help`` for options.
Press space to pause or resume, or use ``--output`` to save an animated GIF.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from experiments.mecanum_common import RobotFootprint
from experiments.mecanum_physics import MecanumPhysicsParams
from experiments.mecanum_sliding import (
    individual_wheel_braking_deceleration,
    rollout_sliding_deceleration,
)
from learning.animate_fixed_axis_xy_sliding import wheel_forces


def wheel_contact_velocities(velocity, wheel_positions):
    """Return wheel-center velocities in body coordinates, including yaw."""
    return np.column_stack((
        velocity[0] - velocity[2] * wheel_positions[:, 1],
        velocity[1] + velocity[2] * wheel_positions[:, 0],
    ))


def simulate_sliding(initial_velocity=(1.0, 0.4, 0.8), deceleration=1.0, dt=0.001, max_time=5.0):
    """Run the shared fixed-step sliding model and return its sampled states."""
    velocity = np.asarray(initial_velocity, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("initial_velocity must contain finite vx, vy and yaw rate")
    if not np.all(np.isfinite([deceleration, dt, max_time])) or min(deceleration, dt, max_time) <= 0:
        raise ValueError("deceleration, dt and max_time must be finite and positive")
    params = MecanumPhysicsParams()
    wheel_braking = individual_wheel_braking_deceleration(deceleration, params)
    states, velocities, end_time, stopped = rollout_sliding_deceleration(
        velocity, wheel_braking, params=params, dt=dt, max_time=max_time
    )
    times = np.minimum(np.arange(len(states)) * dt, end_time)
    forces = np.array([wheel_forces(value, wheel_braking, params) for value in velocities])
    return times, states, velocities, forces, params, stopped


def create_animation(initial_velocity=(1.0, 0.4, 0.8), deceleration=1.0, fps=30,
                     playback_speed=0.35, dt=0.001, max_time=5.0):
    if fps <= 0 or not np.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("fps and playback speed must be positive")
    times, states, velocities, forces, params, stopped = simulate_sliding(
        initial_velocity, deceleration, dt, max_time
    )
    frame_times = np.arange(0.0, times[-1], playback_speed / fps)
    frame_indices = np.unique(np.concatenate((
        np.minimum(np.rint(frame_times / dt).astype(int), len(times) - 1), [len(times) - 1]
    )))
    footprint = RobotFootprint()
    corners = footprint.world_corners(0.0, 0.0, 0.0)
    wheels = np.array([
        [params.wb_hlength, params.wb_hwidth], [params.wb_hlength, -params.wb_hwidth],
        [-params.wb_hlength, params.wb_hwidth], [-params.wb_hlength, -params.wb_hwidth],
    ])
    directions = np.asarray(params.roller_directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    local_wheel_velocities = np.array([wheel_contact_velocities(value, wheels) for value in velocities])

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(3, 2, width_ratios=(1.35, 1.0))
    scene = figure.add_subplot(grid[:, 0])
    velocity_axis = figure.add_subplot(grid[0, 1])
    yaw_rate_axis = figure.add_subplot(grid[1, 1], sharex=velocity_axis)
    angle_axis = figure.add_subplot(grid[2, 1], sharex=velocity_axis)
    figure.suptitle("Fixed-axis Coulomb sliding with yaw", fontsize=14)

    margin = (np.hypot(footprint.length, footprint.width) / 2
              + max(np.linalg.norm(local_wheel_velocities, axis=2).max() / 2.5,
                    np.linalg.norm(forces, axis=2).max() / 250, 0.2) + 0.15)
    scene.set_xlim(states[:, 0].min() - margin, states[:, 0].max() + margin)
    scene.set_ylim(states[:, 1].min() - margin, states[:, 1].max() + margin)
    scene.set_aspect("equal", adjustable="box")
    scene.set_xlabel("world x [m]")
    scene.set_ylabel("world y [m]")
    scene.set_title("Robot, path and wheel contact forces")
    scene.grid(alpha=0.2)
    scene.plot(states[:, 0], states[:, 1], color="0.8", linestyle=":", linewidth=1)
    trail, = scene.plot([], [], color="tab:blue", linewidth=2)
    robot = Polygon(corners, closed=True, facecolor="#d8e8ed", edgecolor="#263b42", linewidth=2)
    scene.add_patch(robot)
    wheel_points, = scene.plot([], [], "s", color="#263b42", markersize=8)
    roller_lines = [scene.plot([], [], color="0.35", linestyle="--", linewidth=1.4)[0] for _ in wheels]
    wheel_labels = [scene.text(0, 0, label, fontsize=9, weight="bold") for label in ("FL", "FR", "RL", "RR")]
    force_arrows = scene.quiver(wheels[:, 0], wheels[:, 1], forces[0, :, 0], forces[0, :, 1],
                                color="tab:red", angles="xy", scale_units="xy", scale=250, width=0.008)
    velocity_arrow = scene.quiver([0], [0], [velocities[0, 0]], [velocities[0, 1]],
                                   color="tab:blue", angles="xy", scale_units="xy", scale=2.5, width=0.008)
    wheel_velocity_arrows = scene.quiver(
        wheels[:, 0], wheels[:, 1], local_wheel_velocities[0, :, 0], local_wheel_velocities[0, :, 1],
        color="tab:blue", alpha=0.35, angles="xy", scale_units="xy", scale=2.5, width=0.006,
    )
    info = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top", fontsize=10,
                      bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    scene.legend(handles=[
        Line2D([], [], color="tab:blue", linewidth=2, label="body velocity / path"),
        Line2D([], [], color="tab:blue", alpha=0.35, linewidth=2, label="wheel contact velocity"),
        Line2D([], [], color="tab:red", linewidth=2, label="wheel friction force"),
        Line2D([], [], color="0.35", linestyle="--", label="resisted roller axis"),
    ], loc="lower right", fontsize=9)

    velocity_axis.plot(times, velocities[:, 0], color="tab:blue", label="vx")
    velocity_axis.plot(times, velocities[:, 1], color="tab:orange", label="vy")
    velocity_axis.set_title("Body velocity")
    velocity_axis.set_ylabel("velocity [m/s]")
    velocity_axis.legend(loc="best")
    yaw_rate_axis.plot(times, velocities[:, 2], color="tab:purple", label="yaw rate")
    yaw_rate_axis.set_title("Yaw rate")
    yaw_rate_axis.set_ylabel("yaw rate [rad/s]")
    yaw_rate_axis.legend(loc="best")
    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))
    headings[speeds <= 1e-4] = np.nan
    angle_axis.plot(times, headings, color="tab:green", label="slide angle in body frame")
    angle_axis.plot(times, np.degrees(states[:, 2]), color="tab:brown", label="body yaw in world frame")
    angle_axis.set_title("Direction and orientation")
    angle_axis.set_xlabel("simulation time [s]")
    angle_axis.set_ylabel("angle [deg]")
    angle_axis.legend(loc="best", fontsize=9)
    cursors = [axis.axvline(0, color="black", linewidth=1) for axis in (velocity_axis, yaw_rate_axis, angle_axis)]
    for axis in (velocity_axis, yaw_rate_axis, angle_axis):
        axis.grid(alpha=0.2)
    figure.text(0.5, 0.015, "Fixed-step numerical rollout; wheel forces are effective model forces.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    def update(frame):
        index = frame_indices[frame]
        position = states[index, :2]
        yaw = states[index, 2]
        velocity = velocities[index]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        wheel_locations = wheels @ rotation.T + position
        world_directions = directions @ rotation.T
        world_forces = forces[index] @ rotation.T
        world_wheel_velocities = local_wheel_velocities[index] @ rotation.T
        world_velocity = rotation @ velocity[:2]
        robot.set_xy(corners @ rotation.T + position)
        trail.set_data(states[:index + 1, 0], states[:index + 1, 1])
        wheel_points.set_data(wheel_locations[:, 0], wheel_locations[:, 1])
        for wheel_index, (line, label) in enumerate(zip(roller_lines, wheel_labels)):
            endpoints = wheel_locations[wheel_index] + np.array([-0.10, 0.10])[:, None] * world_directions[wheel_index]
            line.set_data(endpoints[:, 0], endpoints[:, 1])
            label.set_position(wheel_locations[wheel_index] + [0.025, 0.035])
        force_arrows.set_offsets(wheel_locations)
        force_arrows.set_UVC(world_forces[:, 0], world_forces[:, 1])
        velocity_arrow.set_offsets(position.reshape(1, 2))
        velocity_arrow.set_UVC([world_velocity[0]], [world_velocity[1]])
        wheel_velocity_arrows.set_offsets(wheel_locations)
        wheel_velocity_arrows.set_UVC(world_wheel_velocities[:, 0], world_wheel_velocities[:, 1])
        for cursor in cursors:
            cursor.set_xdata([times[index], times[index]])
        slide_angle = "--" if np.isnan(headings[index]) else f"{headings[index]:.1f} deg"
        phase = "Stopped" if stopped and index == len(times) - 1 else (
            "Time limit reached" if index == len(times) - 1 and not stopped else "Sliding"
        )
        info.set_text(
            f"{phase}  |  t = {times[index]:.2f} s\n"
            f"vx = {velocity[0]:.3f}, vy = {velocity[1]:.3f} m/s\n"
            f"yaw rate = {velocity[2]:.3f} rad/s\n"
            f"slide angle = {slide_angle}   body yaw = {np.degrees(yaw):.1f} deg\n"
            f"wheel force = {np.linalg.norm(forces[index], axis=1).max():.1f} N per slipping wheel"
        )
        return (robot, trail, wheel_points, *roller_lines, *wheel_labels, force_arrows,
                velocity_arrow, wheel_velocity_arrows, *cursors, info)

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=1000 / fps, blit=False, repeat=True)
    paused = False

    def toggle_pause(event):
        nonlocal paused
        if event.key == " ":
            if paused:
                animation.resume()
            else:
                animation.pause()
            paused = not paused

    figure.canvas.mpl_connect("key_press_event", toggle_pause)
    return figure, animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vx", type=float, default=1.0, help="Initial body x velocity [m/s].")
    parser.add_argument("--vy", type=float, default=0.4, help="Initial body y velocity [m/s].")
    parser.add_argument("--yaw-rate", type=float, default=0.8, help="Initial yaw rate [rad/s].")
    parser.add_argument("--deceleration", type=float, default=1.0, help="Body-x braking calibration [m/s^2].")
    parser.add_argument("--dt", type=float, default=0.001, help="Numerical integration step [s].")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum simulation time [s].")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--speed", type=float, default=0.35, help="Simulation seconds per playback second.")
    parser.add_argument("--output", type=Path, help="Save an animated GIF instead of showing the window.")
    args = parser.parse_args()
    figure, animation = create_animation(
        (args.vx, args.vy, args.yaw_rate), args.deceleration, args.fps, args.speed, args.dt, args.max_time
    )
    if args.output:
        animation.save(args.output, writer=PillowWriter(fps=args.fps))
        print(f"Saved {args.output}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()