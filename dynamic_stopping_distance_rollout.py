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

from kinematic_boundary_rollout import (
    sample_boundary_velocities_bisect,
    sample_boundary_velocities_random,
)
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
    parser.add_argument("--samples", type=int, default=8, help="Number of boundary velocity samples (random method).")
    parser.add_argument("--bisect-tier", type=int, default=0, help="Omega-axis bisection tier (bisect method).")
    parser.add_argument(
        "--n-spacing",
        type=int,
        default=1,
        help="Evenly spaced points per cross-section edge (bisect method).",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=20,
        help="Draw a chassis rectangle every N trajectory samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for boundary sampling.")
    parser.add_argument(
        "--sampling-method",
        choices=["bisect", "random"],
        default="bisect",
        help="Boundary sampling method: deterministic omega-slice bisection or random face sampling.",
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
    if args.samples < 1:
        raise ValueError("samples must be at least 1")
    if args.bisect_tier < 0:
        raise ValueError("bisect-tier must be non-negative")
    if args.n_spacing < 1:
        raise ValueError("n-spacing must be at least 1")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.rectangle_stride < 1:
        raise ValueError("rectangle-stride must be at least 1")

    model = Mecanum()
    if args.sampling_method == "random":
        boundary_velocities = sample_boundary_velocities_random(
            model=model,
            max_wheel_velocity=args.max_wheel_velocity,
            total_samples=args.samples,
            seed=args.seed,
        )
    else:
        boundary_velocities = sample_boundary_velocities_bisect(
            model=model,
            max_wheel_velocity=args.max_wheel_velocity,
            bisect_tier=args.bisect_tier,
            n_spacing=args.n_spacing,
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
