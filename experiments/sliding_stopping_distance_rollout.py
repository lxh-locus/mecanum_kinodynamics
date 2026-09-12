"""
sliding_stopping_distance_rollout.py

Sample body velocities on a box-truncated kinematic wheel-speed polytope, then
roll each velocity forward while the platform decelerates under the sliding
roller model. This is the sliding-friction counterpart to
dynamic_stopping_distance_rollout_limited.py.
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .kinematic_boundary_rollout_limited import (
        _polytope_faces,
        compute_truncated_extents,
        compute_wheel_velocity_limits,
        plot_truncated_polytope,
        sample_boundary_velocities,
        sample_boundary_velocities_face_bisection,
    )
    from .mecanum_common import Mecanum
    from .mecanum_physics import MecanumPhysicsParams
    from .mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_sliding_deceleration_coulomb,
        sliding_deceleration_coulomb_fixed_axis,
    )
    from .sliding_deceleration_xy import plot_sliding_deceleration_xy
except ImportError:
    from kinematic_boundary_rollout_limited import (
        _polytope_faces,
        compute_truncated_extents,
        compute_wheel_velocity_limits,
        plot_truncated_polytope,
        sample_boundary_velocities,
        sample_boundary_velocities_face_bisection,
    )
    from mecanum_common import Mecanum
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_sliding_deceleration_coulomb,
        sliding_deceleration_coulomb_fixed_axis,
    )
    from sliding_deceleration_xy import plot_sliding_deceleration_xy


def plot_sliding_stopping_rollouts(
    model,
    params,
    boundary_velocities,
    wheel_braking_deceleration,
    dt=0.005,
    max_time=5.0,
    rectangle_stride=20,
    show_final_only=True,
    speed_tolerance=1e-4,
    yaw_rate_tolerance=1e-4,
):
    """Plot stopping paths for boundary velocities under sliding deceleration."""
    figure, axis = plt.subplots(figsize=(10, 10))

    stop_times = []
    stop_distances = []
    stopped_count = 0

    for command in boundary_velocities:
        states, _, stop_time, stopped = rollout_sliding_deceleration_coulomb(
            command,
            wheel_braking_deceleration=wheel_braking_deceleration,
            params=params,
            dt=dt,
            max_time=max_time,
            speed_tolerance=speed_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )

        stopped_count += int(stopped)
        stop_times.append(stop_time)
        stop_distances.append(float(np.linalg.norm(states[-1, :2])))

        color = "tab:red" if stopped else "tab:orange"
        axis.plot(states[:, 0], states[:, 1], color=color, linewidth=1.2, alpha=0.28)

        if show_final_only:
            indices = [states.shape[0] - 1]
        else:
            indices = list(range(0, states.shape[0], rectangle_stride))
            if indices[-1] != states.shape[0] - 1:
                indices.append(states.shape[0] - 1)

        for idx_num, state_idx in enumerate(indices):
            x, y, theta = states[state_idx]
            corners = model.footprint.world_corners(x=x, y=y, theta=theta)
            alpha = 0.12 if (not show_final_only and idx_num < len(indices) - 1) else 0.9
            axis.add_patch(
                Polygon(corners, closed=True, fill=False, edgecolor="tab:blue", linewidth=0.9, alpha=alpha)
            )

    start_corners = model.footprint.world_corners(x=0.0, y=0.0, theta=0.0)
    axis.add_patch(Polygon(start_corners, closed=True, fill=False, edgecolor="black", linewidth=1.4))

    max_distance = max(stop_distances) if stop_distances else 0.0
    max_stop_time = max(stop_times) if stop_times else 0.0
    axis.set_title(
        "Sliding-Deceleration Rollouts from Kinematic Boundary Velocities\n"
        f"stopped {stopped_count}/{len(boundary_velocities)} within {max_time:g} s, "
        f"max d_trans = {max_distance:.3f} m, max t = {max_stop_time:.3f} s"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    return figure, axis


def main():
    """Parse command-line options and display sliding stopping rollouts."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample [vx, vy, omega] on a box-truncated kinematic wheel-speed "
            "polytope and roll each body velocity while sliding_deceleration "
            "brakes the robot."
        )
    )
    parser.add_argument(
        "--max-wheel-velocity",
        type=float,
        default=MecanumPhysicsParams().max_wheel_velocity,
        help="Wheel-speed limit [rad/s].",
    )
    parser.add_argument(
        "--max-body-x-deceleration",
        type=float,
        default=0.9,
        help="Desired total body-x sliding deceleration used to calibrate each wheel [m/s^2].",
    )
    parser.add_argument(
        "--sampling-degree",
        type=int,
        default=0,
        help=(
            "Surface sampling degree. Degree 0 samples only the truncated "
            "polytope vertices; higher degrees add points on each face."
        ),
    )
    parser.add_argument(
        "--sampling-method",
        choices=["shrink", "bisect"],
        default="shrink",
        help=(
            "Face sampling method for --sampling-degree > 0: 'shrink' scales "
            "each face contour toward its centroid; 'bisect' recursively "
            "bisects each face."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum rollout time [s].")
    parser.add_argument("--speed-tolerance", type=float, default=1e-4, help="Translational stop threshold [m/s].")
    parser.add_argument("--yaw-rate-tolerance", type=float, default=1e-4, help="Yaw-rate stop threshold [rad/s].")
    parser.add_argument(
        "--direction-range-scale",
        type=float,
        default=1.15,
        help="Arrow scale multiplier for the body-velocity direction deceleration figure.",
    )
    parser.add_argument(
        "--sweep-n-angles",
        type=int,
        default=16,
        help="Number of directions shown in the body-velocity direction deceleration figure.",
    )
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
    parser.add_argument(
        "--hide-requested-bounds-box",
        action="store_true",
        help="Suppress the translucent user-requested vx/vy/omega bounds box in the 3D polytope figure.",
    )
    parser.add_argument("--vx-min", type=float, default=0.5, help="Minimum body vx limit [m/s].")
    parser.add_argument("--vx-max", type=float, default=1.0, help="Maximum body vx limit [m/s].")
    parser.add_argument("--vy-min", type=float, default=-0.1, help="Minimum body vy limit [m/s].")
    parser.add_argument("--vy-max", type=float, default=0.5, help="Maximum body vy limit [m/s].")
    parser.add_argument("--omega-min", type=float, default=0.5, help="Minimum body omega limit [rad/s].")
    parser.add_argument("--omega-max", type=float, default=2.0, help="Maximum body omega limit [rad/s].")
    args = parser.parse_args()

    if args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    if args.max_body_x_deceleration <= 0.0:
        raise ValueError("max-body-x-deceleration must be positive")
    if args.sampling_degree < 0:
        raise ValueError("sampling-degree must be non-negative")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.max_time <= 0.0:
        raise ValueError("max-time must be positive")
    if args.speed_tolerance <= 0.0:
        raise ValueError("speed-tolerance must be positive")
    if args.yaw_rate_tolerance <= 0.0:
        raise ValueError("yaw-rate-tolerance must be positive")
    if args.direction_range_scale <= 0.0:
        raise ValueError("direction-range-scale must be positive")
    if args.sweep_n_angles < 1:
        raise ValueError("sweep-n-angles must be at least 1")
    if args.rectangle_stride < 1:
        raise ValueError("rectangle-stride must be at least 1")
    if args.vx_min > args.vx_max:
        raise ValueError("vx-min must be less than or equal to vx-max")
    if args.vy_min > args.vy_max:
        raise ValueError("vy-min must be less than or equal to vy-max")
    if args.omega_min > args.omega_max:
        raise ValueError("omega-min must be less than or equal to omega-max")

    params = MecanumPhysicsParams()
    model = Mecanum(params=params)
    vx_range = (args.vx_min, args.vx_max)
    vy_range = (args.vy_min, args.vy_max)
    omega_range = (args.omega_min, args.omega_max)

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

    sampling_fn = sample_boundary_velocities if args.sampling_method == "shrink" else sample_boundary_velocities_face_bisection
    boundary_velocities = sampling_fn(vertices, faces, sampling_degree=args.sampling_degree)
    extra_points = boundary_velocities[vertices.shape[0]:]

    wheel_braking_deceleration = individual_wheel_braking_deceleration(
        args.max_body_x_deceleration,
        params=params,
    )
    print(f"Sampled {len(boundary_velocities)} boundary body velocities")
    print(f"Per-wheel sliding deceleration: {wheel_braking_deceleration:.4f} m/s^2")

    plot_sliding_stopping_rollouts(
        model=model,
        params=params,
        boundary_velocities=boundary_velocities,
        wheel_braking_deceleration=wheel_braking_deceleration,
        dt=args.dt,
        max_time=args.max_time,
        rectangle_stride=args.rectangle_stride,
        show_final_only=(not args.show_all_rectangles),
        speed_tolerance=args.speed_tolerance,
        yaw_rate_tolerance=args.yaw_rate_tolerance,
    )
    plot_truncated_polytope(
        vertices,
        faces,
        vx_range,
        vy_range,
        omega_range,
        extra_points=extra_points,
        show_requested_bounds_box=(not args.hide_requested_bounds_box),
    )
    plot_sliding_deceleration_xy(
        params=params,
        max_wheel_velocity=args.max_wheel_velocity,
        max_body_x_deceleration=args.max_body_x_deceleration,
        range_scale=args.direction_range_scale,
        angle_sweep=args.sweep_n_angles,
        deceleration_fn=sliding_deceleration_coulomb_fixed_axis,
    )
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)