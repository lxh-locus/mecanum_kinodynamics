"""
dynamic_stopping_distance_rollout.py

Companion to kinematic_boundary_rollout.py: instead of rolling out constant
twists over a fixed horizon, each sampled boundary velocity is braked to rest
in minimum time under the torque-limited acceleration zonotope, and the
resulting stopping paths / chassis footprints are plotted.

The minimum stopping time is the Minkowski functional of the acceleration
zonotope evaluated at -v0 (see stopping_distance_polytope.py); the applied
body acceleration is then the constant -v0 / t_stop.
"""
import argparse
import sys

from kinematic_boundary_rollout import sample_boundary_velocities
import matplotlib.pyplot as plt

from dynamic_stopping_distance_rollout_limited import (
    plot_stopping_rollouts,
    rollout_stopping_twist,
    stopping_time,
)
from mecanum_common import Mecanum
from stopping_distance_polytope import _compute_acceleration_zonotope


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample [vx, vy, omega] on the kinematic polytope boundary, brake each to "
            "rest in minimum time under the torque-limited acceleration zonotope, and "
            "plot the stopping paths with oriented chassis rectangles."
        )
    )
    parser.add_argument("--max-wheel-velocity", type=float, default=10.0, help="Wheel-speed limit [rad/s].")
    parser.add_argument("--max-torque", type=float, default=3.5, help="Wheel-torque limit [N*m].")
    parser.add_argument(
        "--sampling-degree",
        type=int,
        default=0,
        help="Surface sampling degree; degree 0 samples only polytope vertices.",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=20,
        help="Draw a chassis rectangle every N trajectory samples.",
    )
    parser.add_argument(
        "--sampling-method",
        choices=["shrink", "bisect"],
        default="shrink",
        help="Face sampling method: shrink face contours or recursively bisect faces.",
    )
    parser.add_argument(
        "--show-all-rectangles",
        action="store_true",
        help="Draw chassis rectangles throughout each rollout instead of final-only view.",
    )
    args = parser.parse_args()

    if args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    if args.max_torque <= 0.0:
        raise ValueError("max-torque must be positive")
    if args.sampling_degree < 0:
        raise ValueError("sampling-degree must be non-negative")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.rectangle_stride < 1:
        raise ValueError("rectangle-stride must be at least 1")

    model = Mecanum()
    boundary_velocities = sample_boundary_velocities(
        model=model,
        max_wheel_velocity=args.max_wheel_velocity,
        sampling_degree=args.sampling_degree,
        sampling_method=args.sampling_method,
    )

    accel_hull = _compute_acceleration_zonotope(model, args.max_torque)

    plot_stopping_rollouts(
        model=model,
        boundary_velocities=boundary_velocities,
        accel_hull=accel_hull,
        dt=args.dt,
        rectangle_stride=args.rectangle_stride,
        show_final_only=(not args.show_all_rectangles),
        title_suffix="",
    )
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
