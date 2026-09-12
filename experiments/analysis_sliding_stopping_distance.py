"""
analysis_sliding_stopping_distance.py

Roll out every braking strategy for one initial body velocity and compare
them side by side: body velocity vs time on the left, and the resulting
swept-footprint path on the right. This mirrors the bottom-left (velocity
profile) and top-right (rollout detail) panels of inspect_rollout_archives.py.

Strategies compared:
- sliding roller friction, coulomb model (mecanum_sliding.rollout_sliding_deceleration_coulomb_fixed_axis)
- sliding roller friction, velocity-axis Coulomb model (mecanum_sliding.sliding_deceleration_coulomb_velocity_axis)
- sliding roller friction, continuous approximation (mecanum_sliding.rollout_sliding_deceleration_approx)
- independent axis braking (mecanum_sliding.rollout_independent_axis_braking)
- discrete empirical heading-dependent braking (mecanum_sliding.rollout_discrete_empirical_deceleration)
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .mecanum_common import Mecanum
    from .mecanum_physics import MecanumPhysicsParams
    from .mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_discrete_empirical_deceleration,
        rollout_independent_axis_braking,
        rollout_sliding_deceleration_coulomb,
        rollout_sliding_deceleration_approx,
        sliding_deceleration_coulomb_velocity_axis,
    )
except ImportError:
    from mecanum_common import Mecanum
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_discrete_empirical_deceleration,
        rollout_independent_axis_braking,
        rollout_sliding_deceleration_coulomb,
        rollout_sliding_deceleration_approx,
        sliding_deceleration_coulomb_velocity_axis,
    )


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


def plot_velocity_and_energy(axis, strategies, dt, params):
    """Draw vx/vy/omega vs time, plus a twin-axis kinetic-energy power trace, for every strategy."""
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
    axis.set_title("body velocity and power vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("vx, vy [m/s]   \u03c9 [rad/s]")
    axis.grid(True, alpha=0.25)

    power_axis = axis.twinx()
    power_axis.axhline(0.0, color="0.7", linewidth=0.6, linestyle=":")
    for label, color, _, velocities in strategies:
        times = np.arange(len(velocities)) * dt
        # translational + rotational kinetic energy, using the same body model as the rollout
        energy = 0.5 * params.body_mass * np.sum(velocities[:, :2] ** 2, axis=1) + 0.5 * params.body_yaw_inertia * velocities[:, 2] ** 2
        # power is the rate of change of kinetic energy (negative while braking dissipates it)
        power = np.gradient(energy, dt) if len(times) > 1 else np.zeros_like(energy)
        power_axis.plot(times, power, color=color, linestyle="-.", linewidth=1.6, alpha=0.6, label=f"{label} power")
    power_axis.set_ylabel("power [W]")

    lines, labels = axis.get_legend_handles_labels()
    power_lines, power_labels = power_axis.get_legend_handles_labels()
    axis.legend(
        lines + power_lines,
        labels + power_labels,
        loc="best",
        fontsize="x-small",
        ncol=len(strategies),
    )


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
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["sliding-coulomb", "sliding-coulomb-velocity-axis", "sliding-approx", "independent-axis", "discrete-empirical", "all"],
        default=["sliding-coulomb", "independent-axis"],
        help=(
            "Braking strategies to roll out and display. 'sliding' is the roller "
            "friction-circle Coulomb model, 'sliding-approx' is its continuous "
            "alignment-weighted approximation, 'independent-axis' is the "
            "box-bounded per-axis braking model, 'discrete-empirical' is the "
            "heading-dependent model, and 'all' selects every strategy. "
            "Default: sliding independent-axis."
        ),
    )
    parser.add_argument("--vx", type=float, default=1.0, help="Initial body vx [m/s].")
    parser.add_argument("--vy", type=float, default=0.5, help="Initial body vy [m/s].")
    parser.add_argument("--omega", type=float, default=0.8, help="Initial body yaw rate [rad/s].")
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
    parser.add_argument(
        "--cardinal-body-deceleration",
        type=float,
        default=0.9,
        help="Discrete-empirical deceleration outside diagonal bands [m/s^2].",
    )
    parser.add_argument(
        "--diagonal-body-deceleration",
        type=float,
        default=0.7,
        help="Discrete-empirical deceleration inside diagonal bands [m/s^2].",
    )
    parser.add_argument(
        "--diagonal-angle-half-width-degrees",
        type=float,
        default=5.0,
        help="Half-width around each 45-degree diagonal direction [deg].",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integration step [s].")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum rollout time [s].")
    parser.add_argument(
        "--max-footprints", type=int, default=40, help="Maximum swept footprint outlines drawn per strategy."
    )
    args = parser.parse_args()

    selected = set(args.strategies)
    if "all" in selected:
        selected = {"sliding-coulomb", "sliding-coulomb-velocity-axis", "sliding-approx", "independent-axis", "discrete-empirical"}
    if args.max_body_x_deceleration <= 0.0:
        raise ValueError("max-body-x-deceleration must be positive")
    if args.brake_deceleration_x <= 0.0 or args.brake_deceleration_y <= 0.0 or args.brake_deceleration_yaw <= 0.0:
        raise ValueError("brake-deceleration values must be positive")
    if "discrete-empirical" in selected:
        if args.cardinal_body_deceleration <= 0.0 or args.diagonal_body_deceleration <= 0.0:
            raise ValueError("cardinal-body-deceleration and diagonal-body-deceleration must be positive")
        if args.diagonal_body_deceleration > args.cardinal_body_deceleration:
            raise ValueError("diagonal-body-deceleration must not exceed cardinal-body-deceleration")
        if not (0.0 <= args.diagonal_angle_half_width_degrees <= 45.0):
            raise ValueError("diagonal-angle-half-width-degrees must be in [0, 45]")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.max_time <= 0.0:
        raise ValueError("max-time must be positive")
    if args.max_footprints < 1:
        raise ValueError("max-footprints must be at least one")

    initial_velocity = np.array([args.vx, args.vy, args.omega], dtype=float)
    params = MecanumPhysicsParams()
    model = Mecanum(params=params)

    strategies = []
    if "independent-axis" in selected:
        axis_states, axis_velocities = rollout_independent_axis_braking(
            initial_velocity,
            brake_deceleration=(args.brake_deceleration_x, args.brake_deceleration_y, args.brake_deceleration_yaw),
            dt=args.dt,
        )
        print(f"independent axis braking: stop_time={len(axis_velocities) * args.dt:.3f} s")
        strategies.append(("independent axis braking", "tab:blue", axis_states, axis_velocities))

    if "sliding-coulomb" in selected:
        wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params
        )
        slide_states, slide_velocities, stop_time, stopped = rollout_sliding_deceleration_coulomb(
            initial_velocity,
            wheel_braking_deceleration=wheel_braking_deceleration,
            params=params,
            dt=args.dt,
            max_time=args.max_time,
        )
        print(f"sliding roller friction (Coulomb): stop_time={stop_time:.3f} s, stopped={stopped}")
        strategies.append(("sliding roller friction (Coulomb)", "tab:red", slide_states, slide_velocities))

    if "sliding-coulomb-velocity-axis" in selected:
        wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params
        )
        velocity_axis_states, velocity_axis_velocities, velocity_axis_stop_time, velocity_axis_stopped = (
            rollout_sliding_deceleration_coulomb(
                initial_velocity,
                wheel_braking_deceleration=wheel_braking_deceleration,
                params=params,
                dt=args.dt,
                max_time=args.max_time,
                deceleration_fn=sliding_deceleration_coulomb_velocity_axis,
            )
        )
        print(
            "sliding roller friction (Coulomb, velocity axis): "
            f"stop_time={velocity_axis_stop_time:.3f} s, stopped={velocity_axis_stopped}"
        )
        strategies.append(
            (
                "sliding roller friction (Coulomb, velocity axis)",
                "tab:orange",
                velocity_axis_states,
                velocity_axis_velocities,
            )
        )

    if "sliding-approx" in selected:
        approx_wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params, model="approx"
        )
        approx_states, approx_velocities, approx_stop_time, approx_stopped = rollout_sliding_deceleration_approx(
            initial_velocity,
            wheel_braking_deceleration=approx_wheel_braking_deceleration,
            params=params,
            dt=args.dt,
            max_time=args.max_time,
        )
        print(f"sliding roller friction (approx): stop_time={approx_stop_time:.3f} s, stopped={approx_stopped}")
        strategies.append(("sliding roller friction (approx)", "tab:purple", approx_states, approx_velocities))

    if "discrete-empirical" in selected:
        empirical_states, empirical_velocities, empirical_stop_time, empirical_stopped = (
            rollout_discrete_empirical_deceleration(
                initial_velocity,
                cardinal_body_deceleration=args.cardinal_body_deceleration,
                diagonal_body_deceleration=args.diagonal_body_deceleration,
                diagonal_angle_half_width_degrees=args.diagonal_angle_half_width_degrees,
                dt=args.dt,
                max_time=args.max_time,
            )
        )
        print(f"discrete empirical: stop_time={empirical_stop_time:.3f} s, stopped={empirical_stopped}")
        strategies.append(("discrete empirical", "tab:green", empirical_states, empirical_velocities))

    figure, (velocity_axis, path_axis) = plt.subplots(1, 2, figsize=(14, 6))
    figure.suptitle(
        f"Braking Strategy Comparison: vx={args.vx:g} m/s, vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    plot_velocity_profiles(velocity_axis, strategies, args.dt)
    plot_rollout_paths(path_axis, strategies, model.footprint, args.max_footprints)
    figure.tight_layout()

    energy_figure, energy_axis = plt.subplots(figsize=(8, 6))
    energy_figure.suptitle(
        f"Velocity and Energy: vx={args.vx:g} m/s, vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    plot_velocity_and_energy(energy_axis, strategies, args.dt, params)
    energy_figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
