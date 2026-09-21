"""
analysis_sliding_stopping_distance.py

Roll out every braking strategy for one initial body velocity and compare
them side by side: body velocity vs time on the left, and the resulting
swept-footprint path on the right. This mirrors the bottom-left (velocity
profile) and top-right (rollout detail) panels of inspect_rollout_archives.py.

Strategies compared:
- sliding roller friction, fixed-axis Coulomb model (mecanum_sliding.rollout_sliding_deceleration)
- sliding roller friction, velocity-axis Coulomb model (mecanum_sliding.sliding_deceleration_coulomb_velocity_axis)
- sliding roller friction, continuous approximation (mecanum_sliding.rollout_sliding_deceleration with a different deceleration function)
- independent axis braking (mecanum_sliding.rollout_independent_axis_braking)
- discrete empirical heading-dependent braking (mecanum_sliding.rollout_discrete_empirical_deceleration)
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

try:
    from .mecanum_common import Mecanum
    from .mecanum_physics import MecanumPhysicsParams
    from .mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_discrete_empirical_deceleration,
        rollout_independent_axis_braking,
        rollout_sliding_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_velocity_axis,
    )
except ImportError:
    from mecanum_common import Mecanum
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_discrete_empirical_deceleration,
        rollout_independent_axis_braking,
        rollout_sliding_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_velocity_axis,
    )


def plot_global_velocity_profiles(axis, strategies, dt, params=None):
    """Draw world-frame vx/vy/omega vs time for every strategy."""
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    styles = (("vx", "-"), ("vy", "--"), ("\u03c9", ":"))
    for label, color, states, velocities in strategies:
        times = np.arange(len(velocities)) * dt
        headings = states[: len(velocities), 2]
        cos_headings = np.cos(headings)
        sin_headings = np.sin(headings)
        global_velocities = np.column_stack(
            [
                cos_headings * velocities[:, 0] - sin_headings * velocities[:, 1],
                sin_headings * velocities[:, 0] + cos_headings * velocities[:, 1],
                velocities[:, 2],
            ]
        )
        for axis_index, (name, style) in enumerate(styles):
            axis.plot(
                times,
                global_velocities[:, axis_index],
                color=color,
                linestyle=style,
                linewidth=1.4,
                label=f"{label} {name}",
            )
    power_axis = None
    if params is not None:
        power_axis = axis.twinx()
        power_axis.axhline(0.0, color="0.7", linewidth=0.6, linestyle=":")
        for label, color, _, velocities in strategies:
            times = np.arange(len(velocities)) * dt
            energy = 0.5 * params.body_mass * np.sum(velocities[:, :2] ** 2, axis=1) + 0.5 * params.body_yaw_inertia * velocities[:, 2] ** 2
            power = np.gradient(energy, dt) if len(times) > 1 else np.zeros_like(energy)
            power_axis.plot(
                times,
                power,
                color=color,
                linestyle="-.",
                linewidth=1.6,
                alpha=0.6,
                label=f"{label} power",
            )
        power_axis.set_ylabel("power [W]")

    axis.set_title("global velocity and power vs time" if params is not None else "global velocity vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("global vx, vy [m/s]   \u03c9 [rad/s]")
    axis.grid(True, alpha=0.25)
    velocity_handles, velocity_labels = axis.get_legend_handles_labels()
    if power_axis is None:
        return velocity_handles, velocity_labels
    power_handles, power_labels = power_axis.get_legend_handles_labels()
    return velocity_handles + power_handles, velocity_labels + power_labels


def plot_body_velocity_and_power(axis, strategies, dt, params):
    """Draw body-frame velocity and kinetic-energy power for every strategy."""
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    styles = (("vx", "-"), ("vy", "--"), ("\u03c9", ":"))
    for label, color, _, velocities in strategies:
        times = np.arange(len(velocities)) * dt
        for velocity_index, (name, style) in enumerate(styles):
            axis.plot(
                times,
                velocities[:, velocity_index],
                color=color,
                linestyle=style,
                linewidth=1.4,
                label=f"{label} {name}",
            )
    axis.set_title("body velocity and power vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("body vx, vy [m/s]   \u03c9 [rad/s]")
    axis.grid(True, alpha=0.25)

    power_axis = axis.twinx()
    power_axis.axhline(0.0, color="0.7", linewidth=0.6, linestyle=":")
    for label, color, _, velocities in strategies:
        times = np.arange(len(velocities)) * dt
        energy = 0.5 * params.body_mass * np.sum(velocities[:, :2] ** 2, axis=1) + 0.5 * params.body_yaw_inertia * velocities[:, 2] ** 2
        power = np.gradient(energy, dt) if len(times) > 1 else np.zeros_like(energy)
        power_axis.plot(times, power, color=color, linestyle="-.", linewidth=1.6, alpha=0.6, label=f"{label} power")
    power_axis.set_ylabel("power [W]")

    velocity_handles, velocity_labels = axis.get_legend_handles_labels()
    power_handles, power_labels = power_axis.get_legend_handles_labels()
    return velocity_handles + power_handles, velocity_labels + power_labels


def plot_contact_velocity_profiles(axes, strategies, dt, params):
    """Draw each wheel's contact-speed magnitude and angle for every strategy."""
    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    wheel_names = ("front-left", "front-right", "rear-left", "rear-right")
    angle_axes = []
    for wheel_index, (axis, wheel_name) in enumerate(zip(axes.flat, wheel_names)):
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        angle_axis = axis.twinx()
        angle_axes.append(angle_axis)
        angle_axis.axhline(0.0, color="0.7", linewidth=0.6, linestyle=":")
        roller_angle = np.degrees(np.arctan2(*params.roller_directions[wheel_index][::-1]))
        rolling_angles = (roller_angle - 90.0, roller_angle + 90.0)
        wheel_x, wheel_y = wheel_positions[wheel_index]
        angle_traces = []
        for label, color, _, velocities in strategies:
            times = np.arange(len(velocities)) * dt
            contact_velocities = np.column_stack(
                [
                    velocities[:, 0] - velocities[:, 2] * wheel_y,
                    velocities[:, 1] + velocities[:, 2] * wheel_x,
                ]
            )
            speed = np.linalg.norm(contact_velocities, axis=1)
            angle = np.degrees(np.arctan2(contact_velocities[:, 1], contact_velocities[:, 0]))
            angle_traces.append(angle)
            axis.plot(times, speed, color=color, linestyle="-", linewidth=1.3, label=f"{label} magnitude")
            angle_axis.plot(
                times,
                angle,
                color=color,
                linestyle="--",
                linewidth=1.3,
                label=f"{label} angle",
            )
        angle_data = np.concatenate(angle_traces)
        angle_center = 0.5 * (np.min(angle_data) + np.max(angle_data))
        angle_half_range = max(
            np.max(np.abs(angle_data - angle_center)),
            np.max(np.abs(np.asarray(rolling_angles) - angle_center)),
        )
        angle_axis.set_ylim(angle_center - angle_half_range, angle_center + angle_half_range)
        for rolling_angle in rolling_angles:
            angle_axis.axhline(
                rolling_angle,
                color="black",
                linewidth=1.0,
                label="perpendicular to roller axis" if wheel_index == 0 and rolling_angle == rolling_angles[0] else "",
            )
        axis.set_title(f"{wheel_name} contact velocity", fontsize="medium")
        axis.set_xlabel("time [s]")
        axis.set_ylabel("magnitude [m/s]")
        angle_axis.set_ylabel("angle [deg]")
        axis.grid(True, alpha=0.25)
    magnitude_handles, magnitude_labels = axes.flat[0].get_legend_handles_labels()
    angle_handles, angle_labels = angle_axes[0].get_legend_handles_labels()
    return magnitude_handles + angle_handles, magnitude_labels + angle_labels


def plot_roller_positions_and_contact_velocities(
    axes,
    strategies,
    params,
    max_steps=40,
    roller_length=0.1,
    roller_width=0.025,
):
    """Draw sampled roller poses and contact velocities in the global frame.

    The four panels show the rollers in front-left, front-right, rear-left,
    rear-right order. For every strategy, the roller pose and contact velocity
    are drawn at sampled rollout timesteps. Contact velocities are transformed
    from the body frame into the global inertial frame using each state's yaw.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if roller_length <= 0.0 or roller_width <= 0.0:
        raise ValueError("roller_length and roller_width must be positive")
    if axes.size != 4:
        raise ValueError("axes must contain four subplot axes")

    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    wheel_names = ("front-left", "front-right", "rear-left", "rear-right")
    contact_velocity_scale = 0.25
    for strategy_index, (label, color, states, velocities) in enumerate(strategies):
        for axis in axes.flat:
            axis.plot(
                states[:, 0],
                states[:, 1],
                color=color,
                linewidth=1.8,
                alpha=0.3,
                zorder=0,
            )
        stride = max(1, len(states) // max_steps)
        for sample_index in range(0, len(states), stride):
            pose = np.asarray(states[sample_index], dtype=float)
            velocity = np.asarray(velocities[min(sample_index, len(velocities) - 1)], dtype=float)
            cosine = np.cos(pose[2])
            sine = np.sin(pose[2])
            body_to_global = np.array([[cosine, -sine], [sine, cosine]])
            global_wheel_positions = wheel_positions @ body_to_global.T + pose[:2]
            body_contact_velocities = np.column_stack(
                [
                    velocity[0] - velocity[2] * wheel_positions[:, 1],
                    velocity[1] + velocity[2] * wheel_positions[:, 0],
                ]
            )
            global_contact_velocities = body_contact_velocities @ body_to_global.T

            for wheel_index, axis in enumerate(axes.flat):
                wheel_x, wheel_y = global_wheel_positions[wheel_index]
                body_roller_direction = np.asarray(params.roller_directions[wheel_index], dtype=float)
                body_roller_direction /= np.linalg.norm(body_roller_direction)
                roller_direction = body_roller_direction @ body_to_global.T
                roller_angle = np.degrees(np.arctan2(roller_direction[1], roller_direction[0]))
                axis.add_patch(
                    Rectangle(
                        (wheel_x - roller_length / 2.0, wheel_y - roller_width / 2.0),
                        roller_length,
                        roller_width,
                        angle=roller_angle,
                        rotation_point="center",
                        fill=False,
                        edgecolor=color,
                        linewidth=0.6,
                        alpha=0.35,
                        label="roller" if strategy_index == 0 and sample_index == 0 and wheel_index == 0 else "",
                    )
                )
                contact_velocity = global_contact_velocities[wheel_index]
                contact_speed = np.linalg.norm(contact_velocity)
                line_length = min(contact_velocity_scale * contact_speed, roller_width)
                contact_direction = (
                    contact_velocity / contact_speed if contact_speed > 0.0 else np.zeros(2)
                )
                line_end = np.array([wheel_x, wheel_y]) + line_length * contact_direction
                axis.plot(
                    [wheel_x, line_end[0]],
                    [wheel_y, line_end[1]],
                    color=color,
                    linewidth=1.0,
                    zorder=3,
                    label=label if wheel_index == 0 and sample_index == 0 else "",
                )
                axis.plot(wheel_x, wheel_y, ".", color=color, markersize=3)

    for axis, wheel_name in zip(axes.flat, wheel_names):
        axis.set_title(wheel_name, fontsize="medium")
        axis.set_xlabel("global x [m]")
        axis.set_ylabel("global y [m]")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)

    return axes.flat[0].get_legend_handles_labels()


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
    return axis.get_legend_handles_labels()


def main():
    """Parse command-line options and display the braking-strategy comparison."""
    parser = argparse.ArgumentParser(
        description="Roll out every braking strategy for one initial body velocity and compare them."
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "sliding-coulomb-fixed-axis",
            "sliding-coulomb-velocity-axis",
            "sliding-coulomb-approx",
            "independent-axis",
            "discrete-empirical",
            "all",
        ],
        default=["sliding-coulomb-velocity-axis", "independent-axis"],
        help=(
            "Braking strategies to roll out and display. 'sliding-coulomb-fixed-axis' is the "
            "roller friction-circle Coulomb model resolved about the fixed roller axis, "
            "'sliding-coulomb-velocity-axis' resolves it about the contact velocity axis, "
            "'sliding-coulomb-approx' is the continuous alignment-weighted approximation, "
            "'independent-axis' is the box-bounded per-axis braking model, "
            "'discrete-empirical' is the heading-dependent model, and 'all' selects every "
            "strategy. Default: sliding-coulomb-velocity-axis independent-axis."
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
        selected = {
            "sliding-coulomb-fixed-axis",
            "sliding-coulomb-velocity-axis",
            "sliding-coulomb-approx",
            "independent-axis",
            "discrete-empirical",
        }
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

    if "sliding-coulomb-fixed-axis" in selected:
        wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params, model="fixed-axis"
        )
        slide_states, slide_velocities, stop_time, stopped = rollout_sliding_deceleration(
            initial_velocity,
            wheel_braking_deceleration=wheel_braking_deceleration,
            params=params,
            dt=args.dt,
            max_time=args.max_time,
        )
        print(f"sliding roller friction (Coulomb, fixed axis): stop_time={stop_time:.3f} s, stopped={stopped}")
        strategies.append(
            ("sliding roller friction (Coulomb, fixed axis)", "tab:red", slide_states, slide_velocities)
        )

    if "sliding-coulomb-velocity-axis" in selected:
        wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params, model="velocity-axis"
        )
        velocity_axis_states, velocity_axis_velocities, velocity_axis_stop_time, velocity_axis_stopped = (
            rollout_sliding_deceleration(
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

    if "sliding-coulomb-approx" in selected:
        approx_wheel_braking_deceleration = individual_wheel_braking_deceleration(
            args.max_body_x_deceleration, params=params, model="approx"
        )
        approx_states, approx_velocities, approx_stop_time, approx_stopped = rollout_sliding_deceleration(
            initial_velocity,
            wheel_braking_deceleration=approx_wheel_braking_deceleration,
            params=params,
            dt=args.dt,
            max_time=args.max_time,
            deceleration_fn=sliding_deceleration_approx_model,
            deceleration_kwargs={
                "wheel_braking_deceleration": approx_wheel_braking_deceleration,
                "params": params,
                "diagonal_angle_half_width_degrees": args.diagonal_angle_half_width_degrees,
            },
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
    velocity_handles, velocity_labels = plot_global_velocity_profiles(velocity_axis, strategies, args.dt)
    path_handles, path_labels = plot_rollout_paths(path_axis, strategies, model.footprint, args.max_footprints)
    figure.legend(
        velocity_handles,
        velocity_labels,
        loc="lower center",
        bbox_to_anchor=(0.25, 0.01),
        fontsize="x-small",
        ncol=3,
        frameon=False,
    )
    figure.legend(
        path_handles,
        path_labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.01),
        fontsize="small",
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 0.93))

    energy_figure, (global_energy_axis, power_axis) = plt.subplots(1, 2, figsize=(14, 6))
    energy_figure.suptitle(
        f"Global Velocity and Power: vx={args.vx:g} m/s, vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    global_energy_handles, global_energy_labels = plot_global_velocity_profiles(
        global_energy_axis, strategies, args.dt, params
    )
    power_handles, power_labels = plot_body_velocity_and_power(power_axis, strategies, args.dt, params)
    energy_figure.legend(
        global_energy_handles,
        global_energy_labels,
        loc="lower center",
        bbox_to_anchor=(0.25, 0.01),
        fontsize="x-small",
        ncol=3,
        frameon=False,
    )
    energy_figure.legend(
        power_handles,
        power_labels,
        loc="lower center",
        bbox_to_anchor=(0.75, 0.01),
        fontsize="small",
        ncol=2,
        frameon=False,
    )
    energy_figure.tight_layout(rect=(0.0, 0.16, 1.0, 0.93))

    contact_figure, contact_axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    contact_figure.suptitle(
        f"Wheel Contact Velocities: vx={args.vx:g} m/s, vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    contact_handles, contact_labels = plot_contact_velocity_profiles(contact_axes, strategies, args.dt, params)
    contact_figure.legend(
        contact_handles,
        contact_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize="x-small",
        ncol=3,
        frameon=False,
    )
    contact_figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))

    roller_figure, roller_axes = plt.subplots(2, 2, figsize=(12, 8))
    roller_figure.suptitle(
        f"Rollout Roller Positions and Contact Velocities: vx={args.vx:g} m/s, "
        f"vy={args.vy:g} m/s, \u03c9={args.omega:g} rad/s"
    )
    roller_handles, roller_labels = plot_roller_positions_and_contact_velocities(
        roller_axes, strategies, params, max_steps=args.max_footprints
    )
    roller_figure.legend(
        roller_handles,
        roller_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize="small",
        ncol=2,
        frameon=False,
    )
    roller_figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.93))

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
