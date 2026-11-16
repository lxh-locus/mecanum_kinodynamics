"""Animate the fixed-axis Coulomb model for a zero-yaw sliding example.

Run from the repository root with ``python learning/animate_fixed_axis_sliding.py``.
Use ``--output learning/sliding.gif`` to save the animation instead.
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
    _sliding_contact_data,
    individual_wheel_braking_deceleration,
    sliding_deceleration_coulomb_fixed_axis,
)


INITIAL_VELOCITY = np.array([1.0, 0.4, 0.0])
TOLERANCE = 1e-9


def wheel_forces(velocity, wheel_braking, params):
    """Return the model's effective per-wheel forces in body coordinates."""
    contact = _sliding_contact_data(velocity, params, TOLERANCE)
    if contact is None:
        return np.zeros((4, 2))
    _, directions, _, projection = contact
    signs = np.sign(projection)
    signs[np.isclose(projection, 0.0, atol=TOLERANCE)] = 0.0
    return -params.body_mass * wheel_braking * signs[:, np.newaxis] * directions


def simulate_sliding(max_body_x_deceleration=1.0, fps=30, playback_speed=0.35):
    """Sample the exact two-stage solution, including transition and stop events."""
    if max_body_x_deceleration <= 0.0 or fps <= 0 or playback_speed <= 0.0:
        raise ValueError("deceleration, fps and playback speed must be positive")
    params = MecanumPhysicsParams()
    wheel_braking = individual_wheel_braking_deceleration(max_body_x_deceleration, params)
    initial_acceleration = sliding_deceleration_coulomb_fixed_axis(
        INITIAL_VELOCITY, wheel_braking, params
    )
    transition_time = (INITIAL_VELOCITY[1] - INITIAL_VELOCITY[0]) / (
        initial_acceleration[0] - initial_acceleration[1]
    )
    transition_velocity = INITIAL_VELOCITY + transition_time * initial_acceleration
    transition_velocity[0] = transition_velocity[1]
    diagonal_acceleration = sliding_deceleration_coulomb_fixed_axis(
        transition_velocity, wheel_braking, params
    )
    diagonal_duration = -transition_velocity[0] / diagonal_acceleration[0]
    stop_time = transition_time + diagonal_duration
    times = np.unique(np.concatenate((
        np.arange(0.0, stop_time, playback_speed / fps), [transition_time, stop_time]
    )))
    transition_position = (
        INITIAL_VELOCITY[:2] * transition_time
        + 0.5 * initial_acceleration[:2] * transition_time**2
    )
    elapsed_diagonal = np.maximum(times - transition_time, 0.0)
    elapsed_initial = np.minimum(times, transition_time)
    velocities = INITIAL_VELOCITY + elapsed_initial[:, None] * initial_acceleration
    velocities += elapsed_diagonal[:, None] * diagonal_acceleration
    velocities[-1] = 0.0
    positions = (
        elapsed_initial[:, None] * INITIAL_VELOCITY[:2]
        + 0.5 * elapsed_initial[:, None]**2 * initial_acceleration[:2]
    )
    diagonal = times >= transition_time
    positions[diagonal] = (
        transition_position + elapsed_diagonal[diagonal, None] * transition_velocity[:2]
        + 0.5 * elapsed_diagonal[diagonal, None]**2 * diagonal_acceleration[:2]
    )
    forces = np.array([wheel_forces(velocity, wheel_braking, params) for velocity in velocities])
    return times, positions, velocities, forces, transition_time, params


def create_animation(max_body_x_deceleration=1.0, fps=30, playback_speed=0.35):
    times, positions, velocities, forces, transition_time, params = simulate_sliding(
        max_body_x_deceleration, fps, playback_speed
    )
    figure = plt.figure(figsize=(14, 8))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    scene = figure.add_subplot(grid[:, 0])
    velocity_axis = figure.add_subplot(grid[0, 1])
    angle_axis = figure.add_subplot(grid[1, 1], sharex=velocity_axis)
    figure.suptitle("Fixed-axis Coulomb sliding | body x forward, body y left", fontsize=14)

    corners = RobotFootprint().world_corners(0.0, 0.0, 0.0)
    half_length, half_width = params.wb_hlength, params.wb_hwidth
    wheels = np.array([
        [half_length, half_width], [half_length, -half_width],
        [-half_length, half_width], [-half_length, -half_width],
    ])
    directions = np.asarray(params.roller_directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    scene.set_aspect("equal", adjustable="box")
    scene.set_xlim(-0.65, positions[-1, 0] + 0.8)
    scene.set_ylim(-0.5, positions[-1, 1] + 0.75)
    scene.set_xlabel("world x [m]")
    scene.set_ylabel("world y [m]")
    scene.set_title("Robot, path and wheel contact forces")
    scene.grid(alpha=0.2)
    scene.plot(positions[:, 0], positions[:, 1], color="0.8", linestyle=":", linewidth=1)
    trail, = scene.plot([], [], color="tab:blue", linewidth=2)
    robot = Polygon(corners, closed=True, facecolor="#d8e8ed", edgecolor="#263b42", linewidth=2)
    scene.add_patch(robot)
    wheel_points, = scene.plot([], [], "s", color="#263b42", markersize=8)
    roller_lines = [scene.plot([], [], color="0.35", linestyle="--", linewidth=1.4)[0] for _ in wheels]
    wheel_labels = [scene.text(0, 0, label, fontsize=9, weight="bold") for label in ("FL", "FR", "RL", "RR")]
    force_arrows = scene.quiver(
        wheels[:, 0], wheels[:, 1], forces[0, :, 0], forces[0, :, 1],
        color="tab:red", angles="xy", scale_units="xy", scale=250, width=0.008,
    )
    velocity_arrow = scene.quiver(
        [0], [0], [velocities[0, 0]], [velocities[0, 1]],
        color="tab:blue", angles="xy", scale_units="xy", scale=2.5, width=0.008,
    )
    wheel_velocity_arrows = scene.quiver(
        wheels[:, 0], wheels[:, 1], np.full(4, velocities[0, 0]), np.full(4, velocities[0, 1]),
        color="tab:blue", alpha=0.35, angles="xy", scale_units="xy", scale=2.5, width=0.006,
    )
    info = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top", fontsize=10,
                      bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9})
    scene.legend(handles=[
        Line2D([], [], color="tab:blue", linewidth=2, label="body velocity / path"),
        Line2D([], [], color="tab:blue", alpha=0.35, linewidth=2, label="velocity at each wheel"),
        Line2D([], [], color="tab:red", linewidth=2, label="wheel friction force"),
        Line2D([], [], color="0.35", linestyle="--", label="resisted roller axis"),
    ], loc="lower right", fontsize=9)

    velocity_axis.plot(times, velocities[:, 0], color="tab:blue", label="vx")
    velocity_axis.plot(times, velocities[:, 1], color="tab:orange", label="vy")
    velocity_axis.axvline(transition_time, color="0.5", linestyle="--", label="vx = vy")
    velocity_cursor = velocity_axis.axvline(0, color="black", linewidth=1)
    velocity_axis.set_title("Body velocity")
    velocity_axis.set_ylabel("velocity [m/s]")
    velocity_axis.legend(loc="upper right")
    velocity_axis.grid(alpha=0.2)
    headings = np.degrees(np.arctan2(velocities[:, 1], velocities[:, 0]))
    headings[-1] = np.nan
    angle_axis.plot(times, headings, color="tab:green", label="velocity angle in body frame")
    angle_axis.axhline(45, color="0.5", linestyle=":", label="45 degrees")
    angle_axis.axvline(transition_time, color="0.5", linestyle="--")
    angle_cursor = angle_axis.axvline(0, color="black", linewidth=1)
    angle_axis.set_title("Slide direction")
    angle_axis.set_xlabel("simulation time [s]")
    angle_axis.set_ylabel("angle from body x [deg]")
    angle_axis.set_ylim(0, 55)
    angle_axis.legend(loc="lower right")
    angle_axis.grid(alpha=0.2)
    figure.text(0.5, 0.015, "Force = body mass x each wheel's acceleration contribution; dashed axes show constrained directions.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))

    def update(frame):
        position = positions[frame]
        velocity = velocities[frame]
        wheel_locations = wheels + position
        robot.set_xy(corners + position)
        trail.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])
        wheel_points.set_data(wheel_locations[:, 0], wheel_locations[:, 1])
        for index, (line, label) in enumerate(zip(roller_lines, wheel_labels)):
            endpoints = wheel_locations[index] + np.array([-0.10, 0.10])[:, None] * directions[index]
            line.set_data(endpoints[:, 0], endpoints[:, 1])
            label.set_position(wheel_locations[index] + [0.025, 0.035])
        force_arrows.set_offsets(wheel_locations)
        force_arrows.set_UVC(forces[frame, :, 0], forces[frame, :, 1])
        velocity_arrow.set_offsets(position.reshape(1, 2))
        velocity_arrow.set_UVC([velocity[0]], [velocity[1]])
        wheel_velocity_arrows.set_offsets(wheel_locations)
        wheel_velocity_arrows.set_UVC(np.full(4, velocity[0]), np.full(4, velocity[1]))
        velocity_cursor.set_xdata([times[frame], times[frame]])
        angle_cursor.set_xdata([times[frame], times[frame]])
        angle = "--" if frame == len(times) - 1 else f"{headings[frame]:.1f} deg"
        phase = "Stopped" if frame == len(times) - 1 else (
            "Diagonal sliding" if times[frame] >= transition_time else "Body-x braking"
        )
        info.set_text(
            f"{phase}  |  t = {times[frame]:.2f} s\n"
            f"vx = {velocity[0]:.3f}, vy = {velocity[1]:.3f} m/s\n"
            f"slide angle = {angle}   body yaw = 0 deg\n"
            f"wheel force = {np.linalg.norm(forces[frame], axis=1).max():.1f} N per slipping wheel"
        )
        return (robot, trail, wheel_points, *roller_lines, *wheel_labels, force_arrows,
            velocity_arrow, wheel_velocity_arrows, velocity_cursor, angle_cursor, info)

    animation = FuncAnimation(figure, update, frames=len(times), interval=1000 / fps, blit=False, repeat=True)
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
    parser.add_argument("--deceleration", type=float, default=1.0, help="Initial body-x deceleration [m/s^2].")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second.")
    parser.add_argument("--speed", type=float, default=0.35, help="Simulation seconds per playback second.")
    parser.add_argument("--output", type=Path, help="Save an animated GIF instead of showing the window.")
    args = parser.parse_args()
    figure, animation = create_animation(args.deceleration, args.fps, args.speed)
    if args.output:
        animation.save(args.output, writer=PillowWriter(fps=args.fps))
        print(f"Saved {args.output}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()