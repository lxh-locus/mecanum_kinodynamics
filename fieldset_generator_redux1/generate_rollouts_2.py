#!/usr/bin/env python3
"""Render JSON-configured fieldset rollouts using the sliding mecanum backend.

The fieldset config supplies the velocity envelope, sensor response delay, and
padded robot footprint. Braking and pose integration use the same sliding
kinodynamic model as experiments/sliding_stopping_distance_rollout.py.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.kinematic_boundary_rollout_limited import (
    _polytope_faces,
    compute_truncated_extents,
    compute_wheel_velocity_limits,
    sample_boundary_velocities,
    sample_boundary_velocities_face_bisection,
)
from experiments.mecanum_common import Mecanum
from experiments.mecanum_physics import MecanumPhysicsParams, individual_wheel_braking_deceleration
from experiments.sliding_stopping_distance_rollout import rollout_sliding_deceleration


DEFAULT_CONFIG = Path(__file__).with_name("fieldset_config.json")


def load_config(path):
    """Load the rollout-relevant portions of a fieldset JSON configuration."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "robot_revision" not in config or "field_configs" not in config:
        raise ValueError("Configuration must contain robot_revision and field_configs")
    return config


def footprint_corners(footprint, x, y, yaw):
    """Return the padded, offset rectangular footprint at a world-frame pose."""
    length = float(footprint["length"])
    width = float(footprint["width"])
    padding = float(footprint.get("padding", 0.0))
    x_offset = float(footprint.get("x_offset", 0.0))
    x_front = length / 2.0 + x_offset + padding
    x_rear = -length / 2.0 + x_offset - padding
    y_left = width / 2.0 + padding
    y_right = -width / 2.0 - padding
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [x + c * x_front - s * y_left, y + s * x_front + c * y_left],
            [x + c * x_front - s * y_right, y + s * x_front + c * y_right],
            [x + c * x_rear - s * y_right, y + s * x_rear + c * y_right],
            [x + c * x_rear - s * y_left, y + s * x_rear + c * y_left],
        ],
        dtype=float,
    )


def response_time(robot_revision, generator_params):
    """Return the robot delay plus the slowest configured sensor response time."""
    sensor_times = [sensor["sensor_type"].get("response_time", 0.0) for sensor in robot_revision.get("sensors", [])]
    return float(generator_params.get("robot_response_time", 0.0)) + max(sensor_times, default=0.0)


def velocity_ranges_from_field(field):
    """Return the field's requested vx, vy, and yaw ranges from its JSON limits."""
    velocity = field["dynamic_limit"]["velocity"]
    ranges = tuple((float(velocity["min"][axis]), float(velocity["max"][axis])) for axis in ("x", "y", "yaw"))
    if any(lower > upper for lower, upper in ranges):
        raise ValueError(f"Field '{field['name']}' has an invalid velocity range")
    return ranges


def max_wheel_velocity_from_config(robot_revision, params):
    """Convert the robot-level maximum x speed in the JSON to wheel speed."""
    max_body_x_velocity = float(robot_revision["dynamic_limit"]["velocity"]["max"]["x"])
    if max_body_x_velocity <= 0.0:
        raise ValueError("robot_revision dynamic_limit velocity max x must be positive")
    return max_body_x_velocity / params.wheel_radius


def integrate_response_delay(initial_velocity, duration, dt):
    """Integrate the constant-velocity response period before sliding braking begins."""
    steps = int(duration // dt)
    poses = np.zeros((steps + 1, 3), dtype=float)
    vx, vy, yaw_rate = np.asarray(initial_velocity, dtype=float)
    for index in range(steps):
        x, y, yaw = poses[index]
        c = np.cos(yaw)
        s = np.sin(yaw)
        poses[index + 1] = (x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + yaw_rate * dt)
    return poses


def rollout_with_response_delay(initial_velocity, wheel_braking_deceleration, params, delay, dt, max_time):
    """Hold velocity during response delay, then append a sliding-model braking rollout."""
    response_poses = integrate_response_delay(initial_velocity, delay, dt)
    braking_poses, _, stop_time, stopped = rollout_sliding_deceleration(
        initial_velocity,
        wheel_braking_deceleration=wheel_braking_deceleration,
        params=params,
        dt=dt,
        max_time=max_time,
    )
    x, y, yaw = response_poses[-1]
    c = np.cos(yaw)
    s = np.sin(yaw)
    shifted_braking = braking_poses.copy()
    shifted_braking[:, 0] = x + c * braking_poses[:, 0] - s * braking_poses[:, 1]
    shifted_braking[:, 1] = y + s * braking_poses[:, 0] + c * braking_poses[:, 1]
    shifted_braking[:, 2] += yaw
    return np.vstack((response_poses, shifted_braking[1:])), stop_time, stopped


def save_rollout_data(output_path, initial_velocities, results, dt, delay, field_name, wheel_braking):
    """Save variable-length pose histories as a compressed, padded NumPy archive."""
    trajectories = [trajectory for trajectory, _, _ in results]
    lengths = np.array([trajectory.shape[0] for trajectory in trajectories], dtype=int)
    poses = np.full((len(trajectories), lengths.max(), 3), np.nan, dtype=float)
    for index, trajectory in enumerate(trajectories):
        poses[index, :lengths[index]] = trajectory
    np.savez_compressed(
        output_path,
        format_version=1,
        model="sliding_kinodynamic",
        field_name=field_name,
        initial_velocities=np.asarray(initial_velocities, dtype=float),
        poses=poses,
        lengths=lengths,
        stop_times=np.asarray([stop_time for _, stop_time, _ in results], dtype=float),
        stopped=np.asarray([stopped for _, _, stopped in results], dtype=bool),
        dt=float(dt),
        response_time=float(delay),
        wheel_braking_deceleration=float(wheel_braking),
    )


def plot_field_rollouts(
    robot_revision,
    field,
    generator_params,
    boundary_velocities,
    velocity_ranges,
    dt,
    max_time,
    output_path,
    data_output_path=None,
):
    """Render field boundary rollouts using the sliding mecanum braking model."""
    params = MecanumPhysicsParams()
    body_x_braking = abs(float(robot_revision["brake_deceleration"]["x"]))
    wheel_braking = individual_wheel_braking_deceleration(body_x_braking, params=params)
    delay = response_time(robot_revision, generator_params)
    results = [
        rollout_with_response_delay(velocity, wheel_braking, params, delay, dt, max_time)
        for velocity in boundary_velocities
    ]
    if data_output_path is not None:
        save_rollout_data(
            data_output_path,
            boundary_velocities,
            results,
            dt,
            delay,
            field["name"],
            wheel_braking,
        )

    figure, axis = plt.subplots(figsize=(10, 10))
    stopped_count = 0
    for trajectory, _, stopped in results:
        stopped_count += int(stopped)
        color = "tab:red" if stopped else "tab:orange"
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.8, alpha=0.35)
        x, y, yaw = trajectory[-1]
        axis.add_patch(Polygon(footprint_corners(robot_revision["footprint"], x, y, yaw), closed=True, fill=False,
                               edgecolor="tab:blue", linewidth=0.6, alpha=0.35))

    axis.add_patch(Polygon(footprint_corners(robot_revision["footprint"], 0.0, 0.0, 0.0), closed=True,
                           facecolor="lightsteelblue", edgecolor="black", alpha=0.75))
    vx_range, vy_range, omega_range = velocity_ranges
    axis.set_title(
        f"{field['name']}: Sliding-Kinodynamic Fieldset Rollouts\n"
        f"vx [{vx_range[0]:g}, {vx_range[1]:g}] m/s, "
        f"vy [{vy_range[0]:g}, {vy_range[1]:g}] m/s, "
        f"yaw [{omega_range[0]:g}, {omega_range[1]:g}] rad/s\n"
        f"response delay {delay:.3f} s, stopped {stopped_count}/{len(results)} within {max_time:g} s"
    )
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return len(results), delay, wheel_braking, stopped_count


def main():
    parser = argparse.ArgumentParser(description="Render fieldset JSON rollouts with sliding mecanum kinodynamics.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG, help="Input fieldset JSON file.")
    parser.add_argument("--field", action="append", help="Field-config name to render; repeat to select several fields.")
    parser.add_argument(
        "--max-wheel-velocity",
        type=float,
        help="Wheel-speed limit [rad/s]; defaults to robot_revision dynamic_limit velocity max x / wheel radius.",
    )
    parser.add_argument("--sampling-degree", type=int, default=3, help="Surface sampling degree; degree 0 samples only polytope vertices.")
    parser.add_argument(
        "--sampling-method",
        choices=["shrink", "bisect"],
        default="shrink",
        help="Face sampling method: shrink face contours or recursively bisect faces.",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Integrator step [s].")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum sliding-braking duration [s].")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("rollout_output"),
        help="Directory for generated PNG files.",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        default=Path(__file__).with_name("rollout_data"),
        help="Directory for compressed rollout .npz archives.",
    )
    parser.add_argument(
        "--save-rollout-data",
        action="store_true",
        help="Save rollout pose histories as compressed .npz archives.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.sampling_degree < 0:
        raise ValueError("sampling-degree must be non-negative")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.max_time <= 0.0:
        raise ValueError("max-time must be positive")
    params = MecanumPhysicsParams()
    model = Mecanum(params=params)
    max_wheel_velocity = args.max_wheel_velocity
    if max_wheel_velocity is None:
        max_wheel_velocity = max_wheel_velocity_from_config(config["robot_revision"], params)
    if max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    max_vx, max_vy, max_omega = compute_wheel_velocity_limits(model, max_wheel_velocity)
    print(f"Kinematic limits: vx +/-{max_vx:.4f} m/s, vy +/-{max_vy:.4f} m/s, omega +/-{max_omega:.4f} rad/s")

    fields = config["field_configs"]
    if args.field:
        selected_names = set(args.field)
        fields = [field for field in fields if field["name"] in selected_names]
        missing = selected_names - {field["name"] for field in fields}
        if missing:
            raise ValueError(f"Unknown field names: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rollout_data:
        args.data_output_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        velocity_ranges = velocity_ranges_from_field(field)
        vertices, faces = _polytope_faces(model, max_wheel_velocity, *velocity_ranges)
        sampling_fn = sample_boundary_velocities if args.sampling_method == "shrink" else sample_boundary_velocities_face_bisection
        boundary_velocities = sampling_fn(vertices, faces, sampling_degree=args.sampling_degree)
        (vx_lo, vx_hi), (vy_lo, vy_hi), (omega_lo, omega_hi) = compute_truncated_extents(vertices)
        print(f"{field['name']} requested limits: vx {velocity_ranges[0]}, vy {velocity_ranges[1]}, omega {velocity_ranges[2]}")
        print(f"  truncated extents: vx [{vx_lo:.4f}, {vx_hi:.4f}], vy [{vy_lo:.4f}, {vy_hi:.4f}], omega [{omega_lo:.4f}, {omega_hi:.4f}]")
        print(f"  sampled {len(boundary_velocities)} kinematically feasible boundary velocities")
        output_path = args.output_dir / f"{field['name']}_sliding_rollouts.png"
        data_output_path = (
            args.data_output_dir / f"{field['name']}_sliding_rollouts.npz" if args.save_rollout_data else None
        )
        count, delay, wheel_braking, stopped_count = plot_field_rollouts(
            config["robot_revision"],
            field,
            config["generator_params"],
            boundary_velocities,
            velocity_ranges,
            args.dt,
            args.max_time,
            output_path,
            data_output_path,
        )
        print(
            f"{field['name']}: {count} rollouts, response delay {delay:.3f} s, "
            f"wheel braking {wheel_braking:.4f} m/s^2, stopped {stopped_count}/{count} -> "
            f"{output_path}{f', {data_output_path}' if data_output_path is not None else ''}"
        )


if __name__ == "__main__":
    main()
