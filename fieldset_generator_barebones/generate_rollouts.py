#!/usr/bin/env python3
"""Render fieldset-generator-style stopping rollouts from a JSON configuration.

This intentionally small mock-up reads only rollout-relevant configuration:
the padded robot footprint, velocity limits, braking deceleration, generator
sampling settings, and the slowest configured sensor response time.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


DEFAULT_CONFIG = Path(__file__).with_name("fieldset_config.json")


def load_config(path):
    """Load a fieldset configuration without its production schema dependency."""
    with Path(path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if "robot_revision" not in config or "field_configs" not in config:
        raise ValueError("Configuration must contain robot_revision and field_configs")
    return config


def footprint_corners(footprint, x, y, yaw):
    """Return a padded, offset rectangular footprint at a world-frame pose."""
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


def linspace_limits(velocity_limits, axis, instances):
    """Sample one configured velocity axis, including both endpoint limits."""
    return np.linspace(
        float(velocity_limits["min"][axis]),
        float(velocity_limits["max"][axis]),
        num=instances,
    )


def sample_boundary_velocities(dynamic_limit, instances):
    """Mirror fieldset_generator's boundary-only velocity sampling pattern."""
    velocity = dynamic_limit["velocity"]
    x_values = linspace_limits(velocity, "x", instances["x"])
    y_values = linspace_limits(velocity, "y", instances["y"])
    yaw_values = linspace_limits(velocity, "yaw", instances["yaw"])
    velocities = []
    for yaw_rate in yaw_values:
        for x_velocity in x_values:
            velocities.extend(((x_velocity, y_values[0], yaw_rate), (x_velocity, y_values[-1], yaw_rate)))
        for y_velocity in y_values:
            velocities.extend(((x_values[0], y_velocity, yaw_rate), (x_values[-1], y_velocity, yaw_rate)))
    return velocities


def brake_profile(initial_velocity, deceleration, response_time, dt):
    """Hold an initial velocity through response time, then decelerate to zero."""
    speed = abs(float(initial_velocity))
    if speed == 0.0:
        return np.zeros(1, dtype=float)
    deceleration = abs(float(deceleration))
    if deceleration == 0.0:
        raise ValueError("Configured braking deceleration must be nonzero")
    sign = np.sign(initial_velocity)
    response_steps = int(response_time // dt)
    profile = [initial_velocity] * response_steps
    while speed >= deceleration * dt:
        speed -= deceleration * dt
        profile.append(sign * speed)
    profile.append(0.0)
    return np.asarray(profile, dtype=float)


def rollout(initial_velocity, brake_deceleration, response_time, dt):
    """Return a world-frame pose history for one body-frame initial velocity."""
    profiles = [
        brake_profile(initial_velocity[index], brake_deceleration[axis], response_time, dt)
        for index, axis in enumerate(("x", "y", "yaw"))
    ]
    max_length = max(len(profile) for profile in profiles)
    velocities = np.array(
        [np.pad(profile, (0, max_length - len(profile))) for profile in profiles], dtype=float
    ).T
    poses = np.zeros((max_length + 1, 3), dtype=float)
    for index, (vx, vy, yaw_rate) in enumerate(velocities):
        x, y, yaw = poses[index]
        c = np.cos(yaw)
        s = np.sin(yaw)
        poses[index + 1] = (x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + yaw_rate * dt)
    return poses


def response_time(robot_revision, generator_params):
    """Return robot plus slowest sensor response time, as used for safety rollout."""
    sensor_times = [sensor["sensor_type"].get("response_time", 0.0) for sensor in robot_revision.get("sensors", [])]
    return float(generator_params.get("robot_response_time", 0.0)) + max(sensor_times, default=0.0)


def plot_field_rollouts(robot_revision, field, generator_params, dt, output_path):
    """Render boundary rollouts and initial/final padded body footprints for one field."""
    instances = {
        "x": int(generator_params.get("x_vel_instances", 4)),
        "y": int(generator_params.get("y_vel_instances", 4)),
        "yaw": int(generator_params.get("yaw_rate_instances", 4)),
    }
    if any(count < 2 for count in instances.values()):
        raise ValueError("Each velocity instance count must be at least two")
    delay = response_time(robot_revision, generator_params)
    trajectories = [
        rollout(velocity, robot_revision["brake_deceleration"], delay, dt)
        for velocity in sample_boundary_velocities(field["dynamic_limit"], instances)
    ]

    figure, axis = plt.subplots(figsize=(10, 10))
    for trajectory in trajectories:
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:red", linewidth=0.8, alpha=0.35)
        x, y, yaw = trajectory[-1]
        axis.add_patch(Polygon(footprint_corners(robot_revision["footprint"], x, y, yaw), closed=True, fill=False,
                                 edgecolor="tab:blue", linewidth=0.6, alpha=0.35))

    axis.add_patch(Polygon(footprint_corners(robot_revision["footprint"], 0.0, 0.0, 0.0), closed=True,
                           facecolor="lightsteelblue", edgecolor="black", alpha=0.75))
    velocity_limits = field["dynamic_limit"]["velocity"]
    axis.set_title(
        f"{field['name']}: Fieldset Stopping Rollouts\n"
        f"vx [{velocity_limits['min']['x']:g}, {velocity_limits['max']['x']:g}] m/s, "
        f"vy [{velocity_limits['min']['y']:g}, {velocity_limits['max']['y']:g}] m/s, "
        f"yaw [{velocity_limits['min']['yaw']:g}, {velocity_limits['max']['yaw']:g}] rad/s"
    )
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return len(trajectories), delay


def main():
    parser = argparse.ArgumentParser(description="Render stopping rollouts from a fieldset-generator JSON config.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG, help="Input fieldset JSON file.")
    parser.add_argument("--field", action="append", help="Field-config name to render; repeat to select several fields.")
    parser.add_argument("--dt", type=float, help="Integrator step [s]; defaults to generator_params.integrator_dt.")
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
    fields = config["field_configs"]
    if args.field:
        selected_names = set(args.field)
        fields = [field for field in fields if field["name"] in selected_names]
        missing = selected_names - {field["name"] for field in fields}
        if missing:
            raise ValueError(f"Unknown field names: {', '.join(sorted(missing))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for field in fields:
        output_path = args.output_dir / f"{field['name']}_rollouts.png"
        count, delay = plot_field_rollouts(config["robot_revision"], field, generator_params, dt, output_path)
        print(f"{field['name']}: {count} rollouts, response delay {delay:.3f} s -> {output_path}")


if __name__ == "__main__":
    main()