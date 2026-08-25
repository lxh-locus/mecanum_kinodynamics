"""
dynamic_stopping_distance_rollout_limited.py

Variant of dynamic_stopping_distance_rollout.py that samples boundary
velocities from the velocity-limited (box-truncated) kinematic polytope
defined in kinematic_boundary_rollout_limited.py, then brakes each sampled
velocity to rest in minimum time under the torque-limited acceleration
zonotope. A second figure shows the truncated 3D kinematic polytope, as in
kinematic_velocity_limit_box.py.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from kinematic_boundary_rollout_limited import (
    _polytope_faces,
    _rectangle_corners,
    compute_truncated_extents,
    compute_wheel_velocity_limits,
    plot_truncated_polytope,
    sample_boundary_velocities,
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
        "Minimum-Time Stopping Rollouts with Oriented Chassis Footprint (Limited)\n"
        f"max d_trans = {max(stop_distances):.3f} m, max t_stop = {max(stop_times):.3f} s"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample [vx, vy, omega] on the velocity-limited kinematic polytope boundary, "
            "brake each to rest in minimum time under the torque-limited acceleration "
            "zonotope, and plot the stopping paths with oriented chassis rectangles "
            "alongside the truncated 3D polytope."
        )
    )
    parser.add_argument("--max-wheel-velocity", type=float, default=10.0, help="Wheel-speed limit [rad/s].")
    parser.add_argument("--max-torque", type=float, default=3.5, help="Wheel-torque limit [N*m].")
    parser.add_argument(
        "--sampling-degree",
        type=int,
        default=0,
        help=(
            "Surface sampling degree. Degree 0 samples only the polytope vertices. "
            "Degree d also samples each face's contour shrunk (in-plane, towards its "
            "centroid) by every new dyadic factor k/2**d for odd k, i.e. degree 1 adds "
            "the 0.5-scaled contour, degree 2 adds 0.25/0.75, degree 3 adds "
            "0.125/0.375/0.625/0.875, and so on."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=20,
        help="Draw a chassis rectangle every N trajectory samples.",
    )
    parser.add_argument(
        "--show-all-rectangles",
        action="store_true",
        help="Draw chassis rectangles throughout each rollout instead of final-only view.",
    )
    parser.add_argument("--vx-min", type=float, default=0.5, help="Minimum body vx limit [m/s].")
    parser.add_argument("--vx-max", type=float, default=1.0, help="Maximum body vx limit [m/s].")
    parser.add_argument("--vy-min", type=float, default=-0.1, help="Minimum body vy limit [m/s].")
    parser.add_argument("--vy-max", type=float, default=0.1, help="Maximum body vy limit [m/s].")
    parser.add_argument("--omega-min", type=float, default=0.5, help="Minimum body omega limit [rad/s].")
    parser.add_argument("--omega-max", type=float, default=2.0, help="Maximum body omega limit [rad/s].")
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
    if args.vx_min > args.vx_max:
        raise ValueError("vx-min must be less than or equal to vx-max")
    if args.vy_min > args.vy_max:
        raise ValueError("vy-min must be less than or equal to vy-max")
    if args.omega_min > args.omega_max:
        raise ValueError("omega-min must be less than or equal to omega-max")

    vx_range = (args.vx_min, args.vx_max)
    vy_range = (args.vy_min, args.vy_max)
    omega_range = (args.omega_min, args.omega_max)

    model = Mecanum()

    max_vx, max_vy, max_omega = compute_wheel_velocity_limits(model, args.max_wheel_velocity)
    print(f"Kinematic body-velocity limits from max-wheel-velocity={args.max_wheel_velocity:g} rad/s:")
    print(f"  vx    in [{-max_vx:.4f}, {max_vx:.4f}] m/s")
    print(f"  vy    in [{-max_vy:.4f}, {max_vy:.4f}] m/s")
    print(f"  omega in [{-max_omega:.4f}, {max_omega:.4f}] rad/s")

    print("User-requested vx/vy/omega bounds:")
    print(f"  vx    in [{vx_range[0]:.4f}, {vx_range[1]:.4f}] m/s")
    print(f"  vy    in [{vy_range[0]:.4f}, {vy_range[1]:.4f}] m/s")
    print(f"  omega in [{omega_range[0]:.4f}, {omega_range[1]:.4f}] rad/s")

    vertices, faces = _polytope_faces(model, args.max_wheel_velocity, vx_range, vy_range, omega_range)

    (vx_lo, vx_hi), (vy_lo, vy_hi), (omega_lo, omega_hi) = compute_truncated_extents(vertices)
    print("Truncated polytope extents (wheel limit intersected with requested vx/vy/omega bounds):")
    print(f"  vx    in [{vx_lo:.4f}, {vx_hi:.4f}] m/s")
    print(f"  vy    in [{vy_lo:.4f}, {vy_hi:.4f}] m/s")
    print(f"  omega in [{omega_lo:.4f}, {omega_hi:.4f}] rad/s")

    boundary_velocities = sample_boundary_velocities(vertices, faces, sampling_degree=args.sampling_degree)
    extra_points = boundary_velocities[vertices.shape[0]:]

    accel_hull = _compute_acceleration_zonotope(model, args.max_torque)

    plot_stopping_rollouts(
        model=model,
        boundary_velocities=boundary_velocities,
        accel_hull=accel_hull,
        dt=args.dt,
        rectangle_stride=args.rectangle_stride,
        show_final_only=(not args.show_all_rectangles),
    )

    plot_truncated_polytope(vertices, faces, vx_range, vy_range, omega_range, extra_points=extra_points)

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
