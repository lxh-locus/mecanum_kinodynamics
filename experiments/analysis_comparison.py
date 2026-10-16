"""Compare recorded odom and EKF rollouts with predicted braking trajectories."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from . import analysis_raw_data_rollout as raw
    from .mecanum_common import RobotFootprint
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
    import analysis_raw_data_rollout as raw
    from mecanum_common import RobotFootprint
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        individual_wheel_braking_deceleration,
        rollout_discrete_empirical_deceleration,
        rollout_independent_axis_braking,
        rollout_sliding_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_velocity_axis,
    )


STRATEGY_COLORS = {
    "sliding-coulomb-fixed-axis": "tab:red",
    "sliding-coulomb-velocity-axis": "tab:cyan",
    "sliding-coulomb-approx": "tab:purple",
    "independent-axis": "tab:brown",
    "discrete-empirical": "tab:green",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="PlotJuggler-style CSV recording.")
    parser.add_argument(
        "--strategies", nargs="+", choices=[*STRATEGY_COLORS, "all"],
        default=["sliding-coulomb-approx"],
        help="Braking strategies to compare (default: sliding-coulomb-approx).",
    )
    parser.add_argument(
        "--initial-source", choices=["ekf", "odom"], default="ekf",
        help="Velocity source at the selected start for every strategy (default: ekf).",
    )
    parser.add_argument(
        "--time-window", nargs=2, type=float, metavar=("START", "STOP"),
        help="Comparison interval in seconds from the recording start; default: common odom/IMU interval.",
    )
    parser.add_argument(
        "--start-at-brake", action="store_true",
        help="Start at the first brake_engaged rising edge in the selected interval, if the topic exists.",
    )
    parser.add_argument("--imu-bias-window", type=float, default=0.2)
    parser.add_argument("--no-motion-bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-motion-threshold", type=float, default=0.01)
    parser.add_argument("--no-motion-window", type=float, default=1.0)
    parser.add_argument("--footprint-stride", type=int, default=1)
    parser.add_argument("--energy", action="store_true", help="Include energy and power proxies (default: off).")
    parser.add_argument("--output", type=Path, help="Save figures instead of displaying them.")
    parser.add_argument("--dt", type=float, default=0.005, help="Strategy integration step [s].")
    parser.add_argument(
        "--max-time", type=float,
        help="Cap predicted duration [s]; default: the full selected interval. Predictions never exceed the interval.",
    )
    parser.add_argument("--max-body-x-deceleration", type=float, default=0.9)
    parser.add_argument("--brake-deceleration-x", type=float, default=0.9)
    parser.add_argument("--brake-deceleration-y", type=float, default=0.9)
    parser.add_argument("--brake-deceleration-yaw", type=float, default=3.0)
    parser.add_argument("--cardinal-body-deceleration", type=float, default=0.9)
    parser.add_argument("--diagonal-body-deceleration", type=float, default=0.7)
    parser.add_argument("--diagonal-angle-half-width-degrees", type=float, default=5.0)
    args = parser.parse_args(argv)
    for name in (
        "dt", "max_body_x_deceleration", "brake_deceleration_x", "brake_deceleration_y",
        "brake_deceleration_yaw", "cardinal_body_deceleration", "diagonal_body_deceleration",
    ):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    for name in ("imu_bias_window", "no_motion_threshold", "no_motion_window"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    if args.max_time is not None and (not np.isfinite(args.max_time) or args.max_time <= 0.0):
        parser.error("--max-time must be positive and finite")
    if args.footprint_stride < 1:
        parser.error("--footprint-stride must be at least 1")
    if args.time_window is not None and (
        not np.all(np.isfinite(args.time_window)) or args.time_window[0] >= args.time_window[1]
    ):
        parser.error("--time-window requires finite START < STOP")
    if not 0.0 <= args.diagonal_angle_half_width_degrees <= 45.0:
        parser.error("--diagonal-angle-half-width-degrees must be in [0, 45]")
    if "all" in args.strategies or "discrete-empirical" in args.strategies:
        if args.diagonal_body_deceleration > args.cardinal_body_deceleration:
            parser.error("--diagonal-body-deceleration must not exceed --cardinal-body-deceleration")
    return args


def window_rollout(label, color, times, twists, start, stop):
    """Interpolate exact interval boundaries and rebase an integrated recorded path."""
    states = raw.integrate_body_twist(times, twists)
    selected_times = np.concatenate(([start], times[(times > start) & (times < stop)], [stop]))
    selected_twists = np.column_stack([
        np.interp(selected_times, times, twists[:, index]) for index in range(3)
    ])
    selected_states = np.column_stack([
        np.interp(selected_times, times, states[:, index]) for index in range(3)
    ])
    return label, color, selected_times, selected_twists, raw.rebase_states(selected_states)


def load_recorded_rollouts(args):
    """Apply the raw-data bias/EKF pipeline before selecting a common comparison interval."""
    odom_times, odom_twists = raw.load_topic(args.csv, raw.ODOM_SUFFIXES, "/odom/header/stamp")
    imu_times, imu_values = raw.load_topic(args.csv, raw.IMU_SUFFIXES, "/imu/header/stamp")
    origin_time = min(odom_times[0], imu_times[0])
    odom_times = odom_times - origin_time
    imu_times = imu_times - origin_time
    start = max(odom_times[0], imu_times[0])
    stop = min(odom_times[-1], imu_times[-1])
    if args.time_window is not None:
        start = max(start, args.time_window[0])
        stop = min(stop, args.time_window[1])
    if start >= stop:
        raise ValueError("Selected interval has no common odom and IMU coverage")

    brake_signal = raw.load_optional_signal(args.csv, raw.BRAKE_ENGAGED_SUFFIX)
    braking = np.zeros(len(imu_times), dtype=bool)
    if brake_signal is not None:
        brake_times, brake_values = brake_signal
        brake_times = brake_times - origin_time
        previous = np.searchsorted(brake_times, imu_times, side="right") - 1
        available = previous >= 0
        braking[available] = brake_values[previous[available]] > 0.5
    if args.start_at_brake:
        if brake_signal is None:
            print("Warning: brake_engaged topic is not available; using the normal rollout start")
        else:
            brake_start = raw.find_rising_edge(brake_times, brake_values, start, stop)
            if brake_start is None:
                raise ValueError(f"No brake_engaged low-to-high transition found in [{start}, {stop}] s")
            start = brake_start
            if start >= stop:
                raise ValueError("Brake transition leaves no comparison interval")
            print(f"Starting rollout at brake_engaged transition: {start:.6f} s")

    initial = imu_times - imu_times[0] <= args.imu_bias_window
    imu_bias = imu_values[initial].mean(axis=0)
    if args.no_motion_bias:
        stationary = raw.no_motion_mask(
            imu_times, odom_times, odom_twists, args.no_motion_threshold, args.no_motion_window
        )
        if not np.any(stationary):
            raise ValueError(
                "No IMU samples fall in odom no-motion periods; raise --no-motion-threshold "
                "or lower --no-motion-window"
            )
        imu_bias[:2] = imu_values[stationary, :2].mean(axis=0)
    print(f"IMU bias: ax={imu_bias[0]:.4f} m/s^2, ay={imu_bias[1]:.4f} m/s^2, omega={imu_bias[2]:.5f} rad/s")
    ekf_twists = raw.fuse_imu_odom_velocity(
        imu_times, imu_values - imu_bias, odom_times, odom_twists, braking
    )
    print(f"Comparison interval: [{start:.6f}, {stop:.6f}] s")
    return [
        window_rollout("odom velocity integrated", raw.ODOM_COLOR, odom_times, odom_twists, start, stop),
        window_rollout("imu EKF fused", raw.EKF_COLOR, imu_times, ekf_twists, start, stop),
    ]


def build_strategy_rollouts(initial_velocity, start, stop, args):
    """Adapt the stopping strategies to the recorded-rollout plotting format."""
    selected = set(STRATEGY_COLORS) if "all" in args.strategies else set(args.strategies)
    params = MecanumPhysicsParams()
    horizon = stop - start if args.max_time is None else min(stop - start, args.max_time)
    rollouts = []
    for name, color in STRATEGY_COLORS.items():
        if name not in selected:
            continue
        if name == "independent-axis":
            parameter_summary = (
                f"brake deceleration: x={args.brake_deceleration_x:.3f} m/s^2, "
                f"y={args.brake_deceleration_y:.3f} m/s^2, "
                f"yaw={args.brake_deceleration_yaw:.3f} rad/s^2"
            )
            states, velocities = rollout_independent_axis_braking(
                initial_velocity,
                (args.brake_deceleration_x, args.brake_deceleration_y, args.brake_deceleration_yaw),
                args.dt,
            )
            velocities = np.vstack((initial_velocity, velocities))
            elapsed = np.arange(len(states)) * args.dt
            stopped = elapsed[-1] <= horizon
            duration = min(elapsed[-1], horizon)
            clipped = np.concatenate((elapsed[elapsed < duration], [duration]))
            states = np.column_stack([np.interp(clipped, elapsed, states[:, index]) for index in range(3)])
            velocities = np.column_stack([
                np.interp(clipped, elapsed, velocities[:, index]) for index in range(3)
            ])
            elapsed = clipped
        else:
            options = dict(params=params, dt=args.dt, max_time=horizon)
            if name == "discrete-empirical":
                parameter_summary = (
                    f"body deceleration: cardinal={args.cardinal_body_deceleration:.3f} m/s^2, "
                    f"diagonal={args.diagonal_body_deceleration:.3f} m/s^2; "
                    f"diagonal angle half-width={args.diagonal_angle_half_width_degrees:.3f} deg"
                )
                states, velocities, duration, stopped = rollout_discrete_empirical_deceleration(
                    initial_velocity,
                    cardinal_body_deceleration=args.cardinal_body_deceleration,
                    diagonal_body_deceleration=args.diagonal_body_deceleration,
                    diagonal_angle_half_width_degrees=args.diagonal_angle_half_width_degrees,
                    **options,
                )
            else:
                model = name.removeprefix("sliding-coulomb-")
                wheel_braking = individual_wheel_braking_deceleration(
                    args.max_body_x_deceleration, params=params, model=model
                )
                parameter_summary = (
                    f"per-wheel deceleration={wheel_braking:.3f} m/s^2; "
                    f"max body x deceleration={args.max_body_x_deceleration:.3f} m/s^2"
                )
                if model == "velocity-axis":
                    options["deceleration_fn"] = sliding_deceleration_coulomb_velocity_axis
                elif model == "approx":
                    parameter_summary += (
                        f"; diagonal angle half-width={args.diagonal_angle_half_width_degrees:.3f} deg"
                    )
                    options["deceleration_fn"] = sliding_deceleration_approx_model
                    options["deceleration_kwargs"] = dict(
                        wheel_braking_deceleration=wheel_braking, params=params,
                        diagonal_angle_half_width_degrees=args.diagonal_angle_half_width_degrees,
                    )
                states, velocities, duration, stopped = rollout_sliding_deceleration(
                    initial_velocity, wheel_braking_deceleration=wheel_braking, **options
                )
            elapsed = np.minimum(np.arange(len(states)) * args.dt, horizon)
        print(f"{name}: duration={duration:.3f} s, stopped={stopped}; {parameter_summary}")
        rollouts.append((name, color, start + elapsed, velocities, states))
    return rollouts


def comparison_figure_title(args):
    selected = set(STRATEGY_COLORS) if "all" in args.strategies else set(args.strategies)
    wheel_decelerations = []
    params = MecanumPhysicsParams()
    for name in STRATEGY_COLORS:
        if name in selected and name.startswith("sliding-coulomb-"):
            model = name.removeprefix("sliding-coulomb-")
            value = individual_wheel_braking_deceleration(
                args.max_body_x_deceleration, params=params, model=model
            )
            wheel_decelerations.append((model, value))
    if not wheel_decelerations:
        return args.csv.name
    if len(wheel_decelerations) == 1:
        model, value = wheel_decelerations[0]
        detail = f"per-wheel deceleration ({model}): {value:.3f} m/s^2"
    else:
        values = ", ".join(f"{model}={value:.3f}" for model, value in wheel_decelerations)
        detail = f"per-wheel deceleration [m/s^2]: {values}"
    return f"{args.csv.name}\n{detail}"


def main(argv=None):
    args = parse_args(argv)
    recorded = load_recorded_rollouts(args)
    source = recorded[1 if args.initial_source == "ekf" else 0]
    initial_velocity = source[3][0].copy()
    start, stop = source[2][[0, -1]]
    print(
        f"Strategy initial velocity ({args.initial_source}) at t={start:.6f} s: "
        f"vx={initial_velocity[0]:+.4f} m/s, vy={initial_velocity[1]:+.4f} m/s, "
        f"omega={initial_velocity[2]:+.4f} rad/s"
    )
    rollouts = recorded + build_strategy_rollouts(initial_velocity, start, stop, args)
    figure, (velocity_axis, path_axis) = plt.subplots(1, 2, figsize=(15, 7))
    raw.plot_body_velocities(velocity_axis, rollouts)
    raw.plot_rollout_paths(path_axis, rollouts, RobotFootprint(), args.footprint_stride)
    title = comparison_figure_title(args)
    figure.suptitle(title)
    figure.tight_layout()
    if args.energy:
        energy_figure, energy_axis = plt.subplots(figsize=(10, 6))
        raw.plot_energy_analogs(energy_axis, rollouts)
        energy_figure.suptitle(title)
        energy_figure.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved {args.output}")
        if args.energy:
            energy_output = args.output.with_name(f"{args.output.stem}_energy{args.output.suffix}")
            energy_figure.savefig(energy_output, dpi=150, bbox_inches="tight")
            print(f"Saved {energy_output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()