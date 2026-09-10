"""
analysis_sliding_stopping_distance.py

Roll out every braking strategy for one initial body velocity and compare
them side by side: body velocity vs time on the left, and the resulting
swept-footprint path on the right. This mirrors the bottom-left (velocity
profile) and top-right (rollout detail) panels of inspect_rollout_archives.py.

Strategies compared:
- sliding roller friction (sliding_stopping_distance_rollout.rollout_sliding_deceleration)
- independent axis braking (fieldset_generator_barebones/generate_rollouts.py's rollout)
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .mecanum_common import Mecanum
    from .mecanum_physics import MecanumPhysicsParams, individual_wheel_braking_deceleration
    from .sliding_stopping_distance_rollout import rollout_sliding_deceleration
except ImportError:
    from mecanum_common import Mecanum
    from mecanum_physics import MecanumPhysicsParams, individual_wheel_braking_deceleration
    from sliding_stopping_distance_rollout import rollout_sliding_deceleration


def _independent_axis_brake_profile(initial_speed, deceleration, dt):
    """Ramp an initial axis speed to zero at a constant deceleration.

    Ports fieldset_generator_barebones/generate_rollouts.py's ``brake_profile``,
    omitting its response-time hold so braking starts immediately.
    """
    speed = abs(float(initial_speed))
    if speed == 0.0:
        return np.zeros(1, dtype=float)
    deceleration = abs(float(deceleration))
    if deceleration == 0.0:
        raise ValueError("deceleration must be nonzero")
    sign = np.sign(initial_speed)
    profile = []
    while speed >= deceleration * dt:
        speed -= deceleration * dt
        profile.append(sign * speed)
    profile.append(0.0)
    return np.asarray(profile, dtype=float)


def rollout_independent_axis_braking(body_velocity, brake_deceleration, dt):
    """Roll out pose and body velocity under independent per-axis braking.

    Ports fieldset_generator_barebones/generate_rollouts.py's ``rollout``,
    omitting its response-time hold so braking starts immediately.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        brake_deceleration: Per-axis deceleration magnitudes ``[dx, dy, dyaw]``.
        dt: Integration step in seconds.
    Returns:
        ``(states, velocities)``. ``states`` are ``[x, y, theta]`` rows, and
        ``velocities`` are ``[vx, vy, yaw_rate]`` rows aligned with ``states``.
    """
    profiles = [
        _independent_axis_brake_profile(body_velocity[axis], brake_deceleration[axis], dt)
        for axis in range(3)
    ]
    max_length = max(len(profile) for profile in profiles)
    velocities = np.array(
        [np.pad(profile, (0, max_length - len(profile))) for profile in profiles], dtype=float
    ).T
    states = np.zeros((max_length + 1, 3), dtype=float)
    for index, (vx, vy, yaw_rate) in enumerate(velocities):
        x, y, theta = states[index]
        c = np.cos(theta)
        s = np.sin(theta)
        states[index + 1] = (x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, theta + yaw_rate * dt)
    return states, velocities


def plot_velocity_profiles(axis, strategies, dt):
    """Draw vx/vy/omega vs time for every strategy."""
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    styles = (("vx", "-"), ("vy", "--"), ("\u03c9", ":"))
    for label, color, _, velocities in strategies:
        times = np.arange(len(velocities)) * dt
        for axis_index, (name, style) in enumerate(styles):
            axis.plot(
                times,
                velocities[:, axis_index],
                color=color,
                linestyle=style,
                linewidth=1.4,
                label=f"{label} {name}",
            )
    axis.set_title("body velocity vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("vx, vy [m/s]   \u03c9 [rad/s]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="x-small", ncol=len(strategies))


def plot_rollout_paths(axis, strategies, footprint, max_footprints):
    """Draw each strategy's path with swept footprints and a shared start footprint."""
    for label, color, states, _ in strategies:
        stride = max(1, len(states) // max_footprints)
        for x, y, theta in states[::stride]:
            axis.add_patch(
                Polygon(
                    footprint.world_corners(x, y, theta),
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.6,
                    alpha=0.35,
                )
            )
        x, y, theta = states[-1]
        axis.add_patch(
            Polygon(footprint.world_corners(x, y, theta), closed=True, fill=False, edgecolor=color, linewidth=1.6)
        )
        axis.plot(states[:, 0], states[:, 1], color=color, linewidth=1.8, label=label, zorder=3)

    start_corners = footprint.world_corners(0.0, 0.0, 0.0)
    axis.add_patch(Polygon(start_corners, closed=True, facecolor="lightsteelblue", edgecolor="black", alpha=0.75))
    axis.set_title("rollout path and swept footprints", fontsize="medium")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")


def main():
    """Parse command-line options and display the braking-strategy comparison."""
    parser = argparse.ArgumentParser(
        description="Roll out every braking strategy for one initial body velocity and compare them."
    )
    parser.add_argument("--vx", type=float, default=1.0, help="Initial body vx [m/s].")
    parser.add_argument("--vy", type=float, default=0.5, help="Initial body vy [m/s].")
    parser.add_argument("--omega", type=float, default=1.0, help="Initial body yaw rate [rad/s].")
    parser.add_argument(
        "--max-body-x-deceleration",
        type=float,
        default=0.9,
        help="Desired total body-x sliding deceleration, used to calibrate the sliding-roller model [m/s^2].",
    )
    parser.add_argument(
        "--brake-deceleration-x", type=float, default=0.9, help="Independent-axis vx braking deceleration [m/s^2]."
    )
    parser.add_argument(
        "--brake-deceleration-y", type=float, default=0.9, help="Independent-axis vy braking deceleration [m/s^2]."
    )
    parser.add_argument(
        "--brake-deceleration-yaw",
        type=float,
        default=3.0,
        help="Independent-axis yaw-rate braking deceleration [rad/s^2].",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum rollout time [s].")
    parser.add_argument(
        "--max-footprints", type=int, default=40, help="Maximum swept footprint outlines drawn per strategy."
    )
    args = parser.parse_args()

    if args.max_body_x_deceleration <= 0.0:
        raise ValueError("max-body-x-deceleration must be positive")
    if args.brake_deceleration_x <= 0.0 or args.brake_deceleration_y <= 0.0 or args.brake_deceleration_yaw <= 0.0:
        raise ValueError("brake-deceleration values must be positive")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.max_time <= 0.0:
        raise ValueError("max-time must be positive")
    if args.max_footprints < 1:
        raise ValueError("max-footprints must be at least one")

    initial_velocity = np.array([args.vx, args.vy, args.omega], dtype=float)
    params = MecanumPhysicsParams()
    model = Mecanum(params=params)

    wheel_braking_deceleration = individual_wheel_braking_deceleration(args.max_body_x_deceleration, params=params)
    slide_states, slide_velocities, stop_time, stopped = rollout_sliding_deceleration(
        initial_velocity,
        wheel_braking_deceleration=wheel_braking_deceleration,
        params=params,
        dt=args.dt,
        max_time=args.max_time,
    )
    print(f"sliding roller friction: stop_time={stop_time:.3f} s, stopped={stopped}")

    axis_states, axis_velocities = rollout_independent_axis_braking(
        initial_velocity,
        brake_deceleration=(args.brake_deceleration_x, args.brake_deceleration_y, args.brake_deceleration_yaw),
        dt=args.dt,
    )
    print(f"independent axis braking: stop_time={len(axis_velocities) * args.dt:.3f} s")

    strategies = [
        ("sliding roller friction", "tab:red", slide_states, slide_velocities),
        ("independent axis braking", "tab:blue", axis_states, axis_velocities),
    ]

    figure, (velocity_axis, path_axis) = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(
        f"Braking Strategy Comparison: vx={args.vx:g} m/s, vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    plot_velocity_profiles(velocity_axis, strategies, args.dt)
    plot_rollout_paths(path_axis, strategies, model.footprint, args.max_footprints)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
