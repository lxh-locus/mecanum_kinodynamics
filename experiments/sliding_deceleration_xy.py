"""Visualize sliding deceleration over the kinematically feasible xy velocities."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from mecanum_common import RobotFootprint
from mecanum_physics import (
    MecanumPhysicsParams,
    individual_wheel_braking_deceleration,
    inverse_kinematics,
    sliding_deceleration,
    sliding_deceleration_discrete_emperical,
)


def _directional_decelerations(
    params, max_wheel_velocity, angles, deceleration_fn, deceleration_kwargs
):
    """Evaluate braking at the kinematic boundary for a set of xy headings.

    Args:
        params: Mecanum physical parameters.
        max_wheel_velocity: Absolute wheel-speed limit in rad/s.
        angles: Heading angles in radians, measured from body +x toward +y.
        deceleration_fn: Function mapping body velocity to ``[ax, ay, alpha]``.
        deceleration_kwargs: Keyword arguments passed to ``deceleration_fn``.
    Returns:
        ``(speeds, magnitudes)`` arrays containing the kinematic boundary
        speed and directional braking deceleration for each angle.
    """
    speeds = []
    magnitudes = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        unit_wheel_velocity = inverse_kinematics(
            [direction[0], direction[1], 0.0], params=params
        )
        boundary_speed = max_wheel_velocity / np.max(np.abs(unit_wheel_velocity))
        velocity = np.array([*(boundary_speed * direction), 0.0], dtype=float)
        acceleration = deceleration_fn(velocity, **deceleration_kwargs)
        speeds.append(boundary_speed)
        magnitudes.append(-np.dot(acceleration[:2], direction))
    return np.asarray(speeds, dtype=float), np.asarray(magnitudes, dtype=float)


def plot_sliding_deceleration_xy(
    params=None,
    max_wheel_velocity=10.0,
    max_body_x_deceleration=4.0,
    range_scale=1.15,
    angle_sweep=16,
    deceleration_fn=sliding_deceleration,
    deceleration_kwargs=None,
    arrow_color="tab:red",
    title="Sliding Deceleration by Body-Velocity Direction",
    subtitle=None,
):
    """Plot directional deceleration arrows around the robot footprint.

    Args:
        params: Mecanum physical parameters, or ``None`` for defaults.
        max_wheel_velocity: Absolute wheel-speed limit in rad/s.
        max_body_x_deceleration: Total body-x braking limit in m/s^2.
        range_scale: Multiplier for the displayed arrow-length scale.
        angle_sweep: Number of evenly spaced velocity headings to evaluate.
        deceleration_fn: Function mapping body velocity to ``[ax, ay, alpha]``.
        deceleration_kwargs: Keyword arguments passed to ``deceleration_fn``.
            If omitted, the current roller sliding model is used.
        arrow_color: Matplotlib color for the deceleration arrows.
        title: First title line for the figure.
        subtitle: Optional second title line. If omitted, a default kinematic
            boundary subtitle is used.
    Returns:
        ``(figure, axis)`` containing the diagnostic plot.
    """
    if params is None:
        params = MecanumPhysicsParams()
    if max_wheel_velocity <= 0.0:
        raise ValueError("max_wheel_velocity must be positive")
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")
    if range_scale <= 0.0:
        raise ValueError("range_scale must be positive")
    if not isinstance(angle_sweep, (int, np.integer)) or angle_sweep <= 0:
        raise ValueError("angle_sweep must be a positive integer")

    if deceleration_kwargs is None:
        wheel_braking_deceleration = individual_wheel_braking_deceleration(
            max_body_x_deceleration,
            params=params,
        )
        deceleration_kwargs = {
            "wheel_braking_deceleration": wheel_braking_deceleration,
            "params": params,
        }
    else:
        deceleration_kwargs = dict(deceleration_kwargs)

    angles = np.arange(angle_sweep, dtype=float) * (2.0 * np.pi / angle_sweep)
    speeds, magnitudes = _directional_decelerations(
        params, max_wheel_velocity, angles, deceleration_fn, deceleration_kwargs
    )

    figure, axis = plt.subplots(figsize=(9, 9))
    footprint = RobotFootprint()
    robot = footprint.world_corners(x=0.0, y=0.0, theta=0.0)
    axis.add_patch(Polygon(robot, closed=True, facecolor="lightsteelblue", edgecolor="black", alpha=0.7))

    arrow_scale = range_scale * max(footprint.length, footprint.width) / np.max(magnitudes)
    arrow_x = np.cos(angles) * magnitudes * arrow_scale
    arrow_y = np.sin(angles) * magnitudes * arrow_scale
    axis.quiver(
        np.zeros_like(angles),
        np.zeros_like(angles),
        arrow_x,
        arrow_y,
        color=arrow_color,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
    )
    for angle, speed, magnitude, x, y in zip(angles, speeds, magnitudes, arrow_x, arrow_y):
        axis.text(
            x,
            y,
            f"{np.degrees(angle):g}°\na={magnitude:.2f}\nv={speed:.2f}",
            ha="center",
            va="center",
            fontsize=8,
        )

    if subtitle is None:
        subtitle = f"kinematic boundary, zero yaw rate, max wheel velocity = {max_wheel_velocity:g} rad/s"
    axis.set_title(f"{title}\n{subtitle}")
    axis.set_xlabel("body x forward [m]")
    axis.set_ylabel("body y left [m]")
    arrow_limit = arrow_scale * np.max(magnitudes)
    axis.set_xlim(-arrow_limit, arrow_limit)
    axis.set_ylim(-arrow_limit, arrow_limit)
    axis.set_aspect(1.0, adjustable="box")
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    return figure, axis


def main():
    """Parse command-line options and display the xy deceleration plot."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualize sliding deceleration magnitude over kinematically limited "
            "body vx/vy velocities with zero yaw rate."
        )
    )
    parser.add_argument("--max-wheel-velocity", type=float, default=10.0)
    parser.add_argument("--max-body-x-deceleration", type=float, default=4.0)
    parser.add_argument("--range-scale", type=float, default=1.15)
    parser.add_argument("--sweep-n-angles", type=int, default=16)
    args = parser.parse_args()

    plot_sliding_deceleration_xy(
        max_wheel_velocity=args.max_wheel_velocity,
        max_body_x_deceleration=args.max_body_x_deceleration,
        range_scale=args.range_scale,
        angle_sweep=args.sweep_n_angles,
        deceleration_fn=sliding_deceleration,
    )
    plot_sliding_deceleration_xy(
        max_wheel_velocity=args.max_wheel_velocity,
        max_body_x_deceleration=args.max_body_x_deceleration,
        range_scale=args.range_scale,
        angle_sweep=args.sweep_n_angles,
        deceleration_fn=sliding_deceleration_discrete_emperical,
        deceleration_kwargs={
            "cardinal_body_deceleration": args.max_body_x_deceleration,
            "diagonal_body_deceleration": 0.5 * args.max_body_x_deceleration,
            "diagonal_angle_half_width_degrees": 10.0,
        },
        arrow_color="tab:blue",
        title="Empirical Step Sliding Deceleration by Body-Velocity Direction",
        subtitle=(
            f"axes = {args.max_body_x_deceleration:g} m/s^2, diagonal = "
            f"{0.5 * args.max_body_x_deceleration:g} m/s^2, +/-10 deg"
        ),
    )
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
    finally:
        plt.close("all")
