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

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from mecanum_physics import MecanumPhysicsParams, individual_wheel_braking_deceleration
from sliding_stopping_distance_rollout import rollout_sliding_deceleration


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


def sample_boundary_velocities(dynamic_limit, instances):
    """Mirror fieldset_generator's boundary-only velocity sampling pattern."""
    velocity = dynamic_limit["velocity"]
    x_values = np.linspace(velocity["min"]["x"], velocity["max"]["x"], instances["x"])
    y_values = np.linspace(velocity["min"]["y"], velocity["max"]["y"], instances["y"])
    yaw_values = np.linspace(velocity["min"]["yaw"], velocity["max"]["yaw"], instances["yaw"])
    velocities = []
    for yaw_rate in yaw_values:
        for x_velocity in x_values:
            velocities.extend(((x_velocity, y_values[0], yaw_rate), (x_velocity, y_values[-1], yaw_rate)))
        for y_velocity in y_values:
            velocities.extend(((x_values[0], y_velocity, yaw_rate), (x_values[-1], y_velocity, yaw_rate)))
    return velocities


def response_time(robot_revision, generator_params):
    """Return the robot delay plus the slowest configured sensor response time."""
    sensor_times = [sensor["sensor_type"].get("response_time", 0.0) for sensor in robot_revision.get("sensors", [])]
    return float(generator_params.get("robot_response_time", 0.0)) + max(sensor_times, default=0.0)


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


def plot_field_rollouts(robot_revision, field, generator_params, dt, max_time, output_path):
    """Render field boundary rollouts using the sliding mecanum braking model."""
    instances = {
        "x": int(generator_params.get("x_vel_instances", 4)),
        "y": int(generator_params.get("y_vel_instances", 4)),
        "yaw": int(generator_params.get("yaw_rate_instances", 4)),
    }
    if any(count < 2 for count in instances.values()):
        raise ValueError("Each velocity instance count must be at least two")

    params = MecanumPhysicsParams()
    body_x_braking = abs(float(robot_revision["brake_deceleration"]["x"]))
    wheel_braking = individual_wheel_braking_deceleration(body_x_braking, params=params)
    delay = response_time(robot_revision, generator_params)
    results = [
        rollout_with_response_delay(velocity, wheel_braking, params, delay, dt, max_time)
        for velocity in sample_boundary_velocities(field["dynamic_limit"], instances)
    ]

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
    velocity_limits = field["dynamic_limit"]["velocity"]
    axis.set_title(
        f"{field['name']}: Sliding-Kinodynamic Fieldset Rollouts\n"
        f"vx [{velocity_limits['min']['x']:g}, {velocity_limits['max']['x']:g}] m/s, "
        f"vy [{velocity_limits['min']['y']:g}, {velocity_limits['max']['y']:g}] m/s, "
        f"yaw [{velocity_limits['min']['yaw']:g}, {velocity_limits['max']['yaw']:g}] rad/s\n"
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
    parser.add_argument("--dt", type=float, help="Integrator step [s]; defaults to generator_params.integrator_dt.")
    parser.add_argument("--max-time", type=float, default=5.0, help="Maximum sliding-braking duration [s].")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("rollout_output"),
        help="Directory for generated PNG files.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    generator_params = config["generator_params"]
    dt = float(args.dt if args.dt is not None else generator_params["integrator_dt"])
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.max_time <= 0.0:
        raise ValueError("max-time must be positive")
    fields = config["field_configs"]
    if args.field:
        selected_names = set(args.field)
        fields = [field for field in fields if field["name"] in selected_names]
        missing = selected_names - {field["name"] for field in fields}
        if missing:
            raise ValueError(f"Unknown field names: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        output_path = args.output_dir / f"{field['name']}_sliding_rollouts.png"
        count, delay, wheel_braking, stopped_count = plot_field_rollouts(
            config["robot_revision"], field, generator_params, dt, args.max_time, output_path
        )
        print(
            f"{field['name']}: {count} rollouts, response delay {delay:.3f} s, "
            f"wheel braking {wheel_braking:.4f} m/s^2, stopped {stopped_count}/{count} -> {output_path}"
        )


if __name__ == "__main__":
    main()
