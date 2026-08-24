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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from kinematic_boundary_rollout import (
    _rectangle_corners,
    sample_boundary_velocities_bisect,
    sample_boundary_velocities_random,
)
from mecanum_common import Mecanum
from stopping_distance_polytope import _compute_acceleration_zonotope, _minkowski_functional


def stopping_time(accel_hull, bodyv):
    """Minimum time to bring body velocity `bodyv` to rest [s]."""
    v0 = np.asarray(bodyv, dtype=float)
    if np.allclose(v0, 0.0):
        return 0.0
    return float(_minkowski_functional(accel_hull, -v0[np.newaxis, :])[0])


def rollout_stopping_twist(vx, vy, omega, accel_hull, dt):
    """Roll out a planar pose trajectory while braking to rest in minimum time."""
    v0 = np.array([vx, vy, omega], dtype=float)
    t_stop = stopping_time(accel_hull, v0)
    if t_stop <= 0.0:
        return np.zeros((1, 3), dtype=float), 0.0

    accel = -v0 / t_stop
    steps = int(np.ceil(t_stop / dt))
    states = np.zeros((steps + 1, 3), dtype=float)  # [x, y, theta]

    for k in range(steps):
        t = min(k * dt, t_stop)
        # Final partial step keeps the trajectory from overshooting t_stop.
        step = min(dt, t_stop - t)
        vx_k, vy_k, omega_k = v0 + accel * t

        x, y, theta = states[k]
        c = np.cos(theta)
        s = np.sin(theta)

        states[k + 1, 0] = x + step * (c * vx_k - s * vy_k)
        states[k + 1, 1] = y + step * (s * vx_k + c * vy_k)
        states[k + 1, 2] = theta + step * omega_k

    return states, t_stop


def plot_stopping_rollouts(
    model,
    boundary_velocities,
    accel_hull,
    dt=0.005,
    rectangle_stride=20,
    show_final_only=True,
):
    """Plot stopping paths and chassis rectangles for sampled boundary velocities."""
    fig, ax = plt.subplots(figsize=(10, 10))

    trajectory_color = "tab:red"
    rectangle_color = "tab:blue"
    trajectory_alpha = 0.25
    final_outline_alpha = 0.9
    intermediate_outline_alpha = 0.12
    stop_distances = []
    stop_times = []

    for cmd in boundary_velocities:
        vx, vy, omega = cmd
        states, t_stop = rollout_stopping_twist(vx, vy, omega, accel_hull=accel_hull, dt=dt)

        stop_times.append(t_stop)
        stop_distances.append(float(np.linalg.norm(states[-1, :2])))

        ax.plot(states[:, 0], states[:, 1], color=trajectory_color, linewidth=1.2, alpha=trajectory_alpha)

        if show_final_only:
            idxs = [states.shape[0] - 1]
        else:
            idxs = list(range(0, states.shape[0], rectangle_stride))
            if idxs[-1] != states.shape[0] - 1:
                idxs.append(states.shape[0] - 1)

        for j, idx in enumerate(idxs):
            x, y, theta = states[idx]
            corners = _rectangle_corners(
                x=x,
                y=y,
                theta=theta,
                half_length=model.wb_hlength,
                half_width=model.wb_hwidth,
            )
            alpha = intermediate_outline_alpha if (not show_final_only and j < len(idxs) - 1) else final_outline_alpha
            poly = Polygon(corners, closed=True, fill=False, edgecolor=rectangle_color, linewidth=0.9, alpha=alpha)
            ax.add_patch(poly)

    start_corners = _rectangle_corners(
        x=0.0,
        y=0.0,
        theta=0.0,
        half_length=model.wb_hlength,
        half_width=model.wb_hwidth,
    )
    ax.add_patch(Polygon(start_corners, closed=True, fill=False, edgecolor="black", linewidth=1.4))

    ax.set_title(
        "Minimum-Time Stopping Rollouts with Oriented Chassis Footprint\n"
        f"max d_trans = {max(stop_distances):.3f} m, max t_stop = {max(stop_times):.3f} s"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


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
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
