"""Compare every mecanum sliding deceleration model over body-velocity headings."""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .mecanum_common import RobotFootprint
    from .mecanum_physics import MecanumPhysicsParams, inverse_kinematics
    from .mecanum_sliding import (
        individual_wheel_braking_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_fixed_axis,
        sliding_deceleration_coulomb_velocity_axis,
        sliding_deceleration_discrete_emperical,
    )
except ImportError:
    from mecanum_common import RobotFootprint
    from mecanum_physics import MecanumPhysicsParams, inverse_kinematics
    from mecanum_sliding import (
        individual_wheel_braking_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_fixed_axis,
        sliding_deceleration_coulomb_velocity_axis,
        sliding_deceleration_discrete_emperical,
    )


def sample_boundary_velocities(params, max_wheel_velocity, angle_sweep, yaw_rate):
    """Sample evenly spaced XY headings at the mecanum kinematic boundary."""
    angles = np.arange(angle_sweep, dtype=float) * (2.0 * np.pi / angle_sweep)
    velocities = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        unit_wheel_velocity = inverse_kinematics(direction, params=params)
        boundary_speed = max_wheel_velocity / np.max(np.abs(unit_wheel_velocity))
        velocities.append(np.array([*(boundary_speed * direction[:2]), yaw_rate]))
    return angles, np.asarray(velocities)


def sliding_methods(params, max_body_x_deceleration, diagonal_body_x_scale, diagonal_angle_half_width):
    """Return every ``sliding_*`` function and its calibrated arguments."""
    coulomb_wheel_braking = individual_wheel_braking_deceleration(
        max_body_x_deceleration, params=params
    )
    approx_wheel_braking = individual_wheel_braking_deceleration(
        max_body_x_deceleration, params=params, model="approx"
    )
    velocity_axis_wheel_braking = individual_wheel_braking_deceleration(
        max_body_x_deceleration, params=params, model="velocity_axis"
    )
    return (
        (
            "approx",
            sliding_deceleration_approx_model,
            {"wheel_braking_deceleration": approx_wheel_braking, "params": params},
        ),
        (
            "coulomb fixed axis",
            sliding_deceleration_coulomb_fixed_axis,
            {"wheel_braking_deceleration": coulomb_wheel_braking, "params": params},
        ),
        (
            "coulomb velocity axis",
            sliding_deceleration_coulomb_velocity_axis,
            {"wheel_braking_deceleration": velocity_axis_wheel_braking, "params": params},
        ),
        (
            "discrete empirical",
            sliding_deceleration_discrete_emperical,
            {
                "cardinal_body_deceleration": max_body_x_deceleration,
                "diagonal_body_deceleration": max_body_x_deceleration * diagonal_body_x_scale,
                "diagonal_angle_half_width_degrees": diagonal_angle_half_width,
                "params": params,
            },
        ),
    )


def plot_sliding_braking_forces(
    params=None,
    max_wheel_velocity=MecanumPhysicsParams().max_wheel_velocity,
    max_body_x_deceleration=4.0,
    diagonal_body_x_scale=0.5,
    diagonal_angle_half_width=10.0,
    angle_sweep=16,
    yaw_rate=0.0,
    range_scale=1.15,
):
    """Plot directional deceleration arrows for all sliding models."""
    if params is None:
        params = MecanumPhysicsParams()
    if max_wheel_velocity <= 0.0:
        raise ValueError("max_wheel_velocity must be positive")
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")
    if not 0.0 < diagonal_body_x_scale <= 1.0:
        raise ValueError("diagonal_body_x_scale must be in (0, 1]")
    if not 0.0 <= diagonal_angle_half_width <= 45.0:
        raise ValueError("diagonal_angle_half_width must be in [0, 45]")
    if not isinstance(angle_sweep, (int, np.integer)) or angle_sweep <= 0:
        raise ValueError("angle_sweep must be a positive integer")
    if range_scale <= 0.0:
        raise ValueError("range_scale must be positive")

    angles, velocities = sample_boundary_velocities(
        params, max_wheel_velocity, angle_sweep, yaw_rate
    )
    methods = sliding_methods(
        params, max_body_x_deceleration, diagonal_body_x_scale, diagonal_angle_half_width
    )
    accelerations = [
        np.asarray([method(velocity, **kwargs) for velocity in velocities])
        for _, method, kwargs in methods
    ]
    max_magnitude = max(np.linalg.norm(values[:, :2], axis=1).max() for values in accelerations)
    footprint = RobotFootprint()
    robot = footprint.world_corners(x=0.0, y=0.0, theta=0.0)
    arrow_scale = range_scale * max(footprint.length, footprint.width) / max_magnitude

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for axis, (title, _, _), values in zip(axes.flat, methods, accelerations):
        axis.add_patch(Polygon(robot, closed=True, facecolor="lightsteelblue", edgecolor="black", alpha=0.7))
        magnitudes = np.linalg.norm(values[:, :2], axis=1)
        arrow_x = np.cos(angles) * magnitudes * arrow_scale
        arrow_y = np.sin(angles) * magnitudes * arrow_scale
        axis.quiver(
            np.zeros_like(angles), np.zeros_like(angles), arrow_x, arrow_y,
            color="tab:red", angles="xy", scale_units="xy", scale=1.0, width=0.004,
        )
        for angle, value, x, y in zip(angles, values, arrow_x, arrow_y):
            axis.text(
                x, y,
                f"{np.degrees(angle):g}°\n|a|={np.linalg.norm(value[:2]):.2f}\nα={value[2]:.2f}",
                ha="center", va="center", fontsize=7,
            )
        axis.set_title(title)
        axis.set_aspect(1.0, adjustable="box")
        axis.grid(True, alpha=0.2)

    for axis in axes[1, :]:
        axis.set_xlabel("body x forward [m]")
    for axis in axes[:, 0]:
        axis.set_ylabel("body y left [m]")
    limit = arrow_scale * max_magnitude * range_scale
    for axis in axes.flat:
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
    figure.suptitle(
        "Mecanum Sliding Braking Deceleration by Body-Velocity Direction\n"
        f"kinematic boundary, yaw rate = {yaw_rate:g} rad/s, "
        f"max wheel velocity = {max_wheel_velocity:g} rad/s"
    )
    figure.tight_layout()
    return figure, axes


def main():
    """Parse options and display or save the four-model directional comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wheel-velocity", type=float, default=MecanumPhysicsParams().max_wheel_velocity)
    parser.add_argument("--max-body-x-deceleration", type=float, default=4.0)
    parser.add_argument("--diagonal-body-x-scale", type=float, default=0.5)
    parser.add_argument("--diagonal-angle-half-width", type=float, default=10.0)
    parser.add_argument("--sweep-n-angles", type=int, default=16)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--range-scale", type=float, default=1.15)
    parser.add_argument("--output", type=str, help="Save the comparison plot to this PNG path.")
    args = parser.parse_args()

    figure, _ = plot_sliding_braking_forces(
        max_wheel_velocity=args.max_wheel_velocity,
        max_body_x_deceleration=args.max_body_x_deceleration,
        diagonal_body_x_scale=args.diagonal_body_x_scale,
        diagonal_angle_half_width=args.diagonal_angle_half_width,
        angle_sweep=args.sweep_n_angles,
        yaw_rate=args.yaw_rate,
        range_scale=args.range_scale,
    )
    if args.output:
        figure.savefig(args.output, dpi=160)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
    finally:
        plt.close("all")
