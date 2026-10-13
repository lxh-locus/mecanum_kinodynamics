"""
analysis_comparison.py

Roll out a recorded trajectory from a PlotJuggler-style CSV export (e.g.
data/v2-27361-shelved/Fast2.csv) two ways and draw the robot footprint at
every sample of each:

- odom: integrate the body-frame twist (linear x, linear y, angular z) into the global frame.
- imu:  integrate gyro z for heading, rotate the body-frame linear acceleration (x, y) into
        the global frame, and double-integrate it. Gyro bias is the mean over an initial window
        where the robot is assumed stationary (--imu-bias-window). Linear-acceleration bias is,
        by default, the mean over IMU samples where odom reports no motion throughout a window
        centered on the sample (--no-motion-bias, --no-motion-window); --no-no-motion-bias uses
        the initial window instead. The robot is
        assumed to start at rest.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

try:
    from .mecanum_common import RobotFootprint
except ImportError:
    from mecanum_common import RobotFootprint


ODOM_SUFFIXES = (
    "/odom/twist/twist/linear/x",
    "/odom/twist/twist/linear/y",
    "/odom/twist/twist/angular/z",
)
ODOM_POSE_SUFFIXES = (
    "/odom/pose/pose/position/x",
    "/odom/pose/pose/position/y",
    "/odom/pose/pose/orientation/yaw_deg",
)
IMU_SUFFIXES = (
    "/imu/linear_acceleration/x",
    "/imu/linear_acceleration/y",
    "/imu/angular_velocity/z",
)
BRAKE_ENGAGED_SUFFIX = "/brake_engaged/data"
ODOM_COLOR = "tab:blue"
IMU_COLOR = "tab:red"
KALMAN_Q = np.diag([0.20**2, 0.20**2, 0.05**2])
KALMAN_R = np.diag([0.10**2, 0.10**2, 0.05**2, 0.03**2])


def _find_column(header, suffix):
    matches = [name for name in header if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one column ending in '{suffix}', found {matches}")
    return matches[0]


def fuse_imu_odom_velocity(imu_times, imu_values, odom_times, odom_twists):
    """Fuse bias-corrected IMU acceleration/gyro with odom twist; return body velocities at IMU timestamps.

    Odom is trusted at a constant KALMAN_R, so this does not yet down-weight odom during braking
    or wheel slip.
    """
    imu_times = np.asarray(imu_times, dtype=float)
    imu_values = np.asarray(imu_values, dtype=float)
    odom_times = np.asarray(odom_times, dtype=float)
    odom_twists = np.asarray(odom_twists, dtype=float)
    if imu_times.ndim != 1 or odom_times.ndim != 1:
        raise ValueError("imu_times and odom_times must be one-dimensional")
    if imu_values.shape != (len(imu_times), 3):
        raise ValueError("imu_values must have shape (len(imu_times), 3) for ax, ay, and gyro z")
    if odom_twists.shape != (len(odom_times), 3):
        raise ValueError("odom_twists must have shape (len(odom_times), 3) for vx, vy, and omega")
    if len(imu_times) == 0 or len(odom_times) == 0:
        raise ValueError("IMU and odom inputs must each contain at least one sample")
    if np.any(np.diff(imu_times) <= 0.0) or np.any(np.diff(odom_times) <= 0.0):
        raise ValueError("imu_times and odom_times must be strictly increasing")

    odom_on_imu = np.column_stack(
        [np.interp(imu_times, odom_times, odom_twists[:, index]) for index in range(3)]
    )
    observation_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    identity = np.eye(3)
    state = odom_on_imu[0].copy()
    covariance = KALMAN_R[:3, :3].copy()
    fused = np.empty((len(imu_times), 3), dtype=float)

    for index, timestamp in enumerate(imu_times):
        if index:
            dt = timestamp - imu_times[index - 1]
            # An accelerometer reads proper acceleration, so body velocity obeys dv/dt = a - omega x v.
            omega_input = imu_values[index - 1, 2]
            transition = np.array(
                [
                    [1.0, omega_input * dt, 0.0],
                    [-omega_input * dt, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            state = transition @ state + np.array([*imu_values[index - 1, :2], 0.0]) * dt
            covariance = transition @ covariance @ transition.T + KALMAN_Q * dt

        measurement = np.array([*odom_on_imu[index], imu_values[index, 2]])
        innovation = measurement - observation_matrix @ state
        innovation_covariance = observation_matrix @ covariance @ observation_matrix.T + KALMAN_R
        gain = np.linalg.solve(innovation_covariance, observation_matrix @ covariance).T
        state += gain @ innovation
        residual_transform = identity - gain @ observation_matrix
        covariance = (
            residual_transform @ covariance @ residual_transform.T + gain @ KALMAN_R @ gain.T
        )
        fused[index] = state

    return fused


def load_topic(path, suffixes, stamp_suffix):
    """Return (absolute times [s], values) for rows that carry every column in `suffixes`."""
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = [_find_column(reader.fieldnames, suffix) for suffix in suffixes]
        stamp_column = next((name for name in reader.fieldnames if name.endswith(stamp_suffix)), "__time")
        times = []
        values = []
        for row in reader:
            # Topics are interleaved; skip rows that belong to other messages.
            if any(row[column] == "" for column in columns):
                continue
            times.append(float(row[stamp_column] or row["__time"]))
            values.append([float(row[column]) for column in columns])

    if len(times) < 2:
        raise ValueError(f"Need at least two samples for {stamp_suffix} in {path}, found {len(times)}")
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    return times[order], np.asarray(values, dtype=float)[order]


def load_optional_signal(path, suffix):
    """Return (absolute times [s], values), or None if the CSV has no matching column."""
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        matches = [name for name in reader.fieldnames if name.endswith(suffix)]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(f"Expected at most one column ending in '{suffix}', found {matches}")
        signal_column = matches[0]
        times = []
        values = []
        for row in reader:
            if row[signal_column] == "":
                continue
            times.append(float(row["__time"]))
            values.append(float(row[signal_column]))

    if not times:
        return np.array([], dtype=float), np.array([], dtype=float)
    times = np.asarray(times, dtype=float)
    order = np.argsort(times, kind="stable")
    return times[order], np.asarray(values, dtype=float)[order]


def find_rising_edge(times, values, start, stop):
    """Return the first low-to-high transition timestamp inside [start, stop], if any."""
    if len(times) < 2:
        return None
    high = values > 0.5
    rising_edges = times[1:][~high[:-1] & high[1:]]
    in_window = (rising_edges >= start) & (rising_edges <= stop)
    matches = rising_edges[in_window]
    return float(matches[0]) if len(matches) else None


def integrate_body_twist(times, twists):
    """Integrate body-frame twists into global poses [x, y, theta], starting at the origin."""
    states = np.zeros((len(times), 3), dtype=float)
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        vx, vy, omega = twists[k]
        x, y, theta = states[k]
        c = np.cos(theta)
        s = np.sin(theta)
        states[k + 1] = [x + dt * (c * vx - s * vy), y + dt * (s * vx + c * vy), theta + dt * omega]
    return states


def no_motion_mask(imu_times, odom_times, odom_twists, threshold, window):
    """Flag IMU samples where every odom twist within +/- window/2 has all components within `threshold`."""
    odom_still = np.all(np.abs(odom_twists) <= threshold, axis=1)
    # Prefix count of moving odom samples gives O(1) "any motion in window" lookups.
    moving_count = np.concatenate([[0], np.cumsum(~odom_still)])
    start = np.searchsorted(odom_times, imu_times - window / 2.0, side="left")
    stop = np.searchsorted(odom_times, imu_times + window / 2.0, side="right")
    return (stop > start) & (moving_count[stop] == moving_count[start])


def integrate_imu(times, imu, bias):
    """Integrate bias-corrected IMU [ax, ay, omega] into global poses and body-frame twists."""
    corrected = imu - bias

    states = np.zeros((len(times), 3), dtype=float)
    global_velocity = np.zeros((len(times), 2), dtype=float)
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        ax, ay, omega = corrected[k]
        x, y, theta = states[k]
        c = np.cos(theta)
        s = np.sin(theta)
        vx, vy = global_velocity[k]
        global_velocity[k + 1] = [vx + dt * (c * ax - s * ay), vy + dt * (s * ax + c * ay)]
        states[k + 1] = [x + dt * vx, y + dt * vy, theta + dt * omega]

    c = np.cos(states[:, 2])
    s = np.sin(states[:, 2])
    body_twists = np.column_stack(
        [
            c * global_velocity[:, 0] + s * global_velocity[:, 1],
            -s * global_velocity[:, 0] + c * global_velocity[:, 1],
            corrected[:, 2],
        ]
    )
    return states, body_twists


def rebase_states(states, origin=None):
    """Express poses relative to the first pose, so the sequence starts at the origin with zero heading."""
    x0, y0, theta0 = states[0] if origin is None else origin
    c = np.cos(theta0)
    s = np.sin(theta0)
    dx = states[:, 0] - x0
    dy = states[:, 1] - y0
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy, states[:, 2] - theta0])


def state_at(times, states, timestamp):
    """Linearly interpolate a pose at a timestamp in the integrated state sequence."""
    return np.array([np.interp(timestamp, times, states[:, index]) for index in range(states.shape[1])])


def plot_body_velocities(axis, rollouts):
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    for label, color, times, twists, _ in rollouts:
        for index, (name, style) in enumerate((("vx", "-"), ("vy", "--"), ("\u03c9", ":"))):
            axis.plot(times, twists[:, index], color=color, linestyle=style, linewidth=1.4, label=f"{label} {name}")
    axis.set_title("body velocity vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("body vx, vy [m/s]   \u03c9 [rad/s]")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")


def plot_rollout_paths(axis, rollouts, footprint, stride):
    for label, color, _, _, states in rollouts:
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
    axis.add_patch(
        Polygon(footprint.world_corners(0.0, 0.0, 0.0), closed=True, facecolor="lightsteelblue", edgecolor="black", alpha=0.75)
    )
    axis.set_title("rollout paths and footprints", fontsize="medium")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")


def main():
    parser = argparse.ArgumentParser(
        description="Roll out a recorded CSV from odom twist and from IMU data, with robot footprints."
    )
    parser.add_argument("csv", type=Path, help="CSV export in the format of data/v2-27361-shelved/Fast2.csv.")
    parser.add_argument(
        "--footprint-stride",
        type=int,
        default=1,
        help="Draw a footprint every N samples (default: 1, every sample).",
    )
    parser.add_argument(
        "--imu-bias-window",
        type=float,
        default=0.2,
        help="Initial stationary duration [s] used to estimate IMU gyro bias, and accel bias "
        "when --no-no-motion-bias is set (default: 0.2).",
    )
    parser.add_argument(
        "--no-motion-bias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate IMU linear-acceleration bias from all samples where odom reports no motion (default: on).",
    )
    parser.add_argument(
        "--no-motion-threshold",
        type=float,
        default=0.01,
        help="Max |odom vx|, |vy| [m/s] and |omega| [rad/s] counted as no motion (default: 0.01).",
    )
    parser.add_argument(
        "--no-motion-window",
        type=float,
        default=1.0,
        help="An IMU sample counts as no motion only if odom is still over this window [s] centered on it "
        "(default: 1.0), which excludes braking transients right after odom reaches zero.",
    )
    parser.add_argument("--output", type=Path, help="Save the figure to this path instead of showing it.")
    parser.add_argument(
        "--time-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help="Display only START to STOP seconds from the start of the file (e.g. 12 14). Bias estimation and "
        "integration still run over the whole file; the shown rollouts are re-expressed so the robot is at "
        "the origin with zero heading at START. Default: the whole file.",
    )
    parser.add_argument(
        "--start-at-brake",
        action="store_true",
        help="Start the displayed rollout at the first brake_engaged low-to-high transition in --time-window. "
        "Without --time-window, search the whole file. If the topic is absent, keep the normal start.",
    )
    args = parser.parse_args()
    if args.footprint_stride < 1:
        raise ValueError("footprint-stride must be at least 1")
    if args.imu_bias_window < 0.0:
        raise ValueError("imu-bias-window must be non-negative")
    if args.no_motion_threshold < 0.0:
        raise ValueError("no-motion-threshold must be non-negative")
    if args.no_motion_window < 0.0:
        raise ValueError("no-motion-window must be non-negative")
    if args.time_window is not None and args.time_window[0] >= args.time_window[1]:
        raise ValueError("time-window START must be less than STOP")

    odom_times, odom_twists = load_topic(args.csv, ODOM_SUFFIXES, "/odom/header/stamp")
    odom_pose_times, odom_pose_values = load_topic(args.csv, ODOM_POSE_SUFFIXES, "/odom/header/stamp")
    imu_times, imu_values = load_topic(args.csv, IMU_SUFFIXES, "/imu/header/stamp")
    t0 = min(odom_times[0], imu_times[0])
    odom_times = odom_times - t0
    odom_pose_times = odom_pose_times - t0
    odom_pose_states = odom_pose_values.copy()
    odom_pose_states[:, 2] = np.unwrap(np.radians(odom_pose_states[:, 2]))
    imu_times = imu_times - t0

    display_start = args.time_window[0] if args.time_window is not None else None
    display_stop = args.time_window[1] if args.time_window is not None else None
    brake_start = None
    if args.start_at_brake:
        brake_signal = load_optional_signal(args.csv, BRAKE_ENGAGED_SUFFIX)
        if brake_signal is None:
            print("Warning: brake_engaged topic is not available; using the normal rollout start")
        else:
            brake_times, brake_values = brake_signal
            search_start = display_start if display_start is not None else -np.inf
            search_stop = display_stop if display_stop is not None else np.inf
            brake_start = find_rising_edge(brake_times - t0, brake_values, search_start, search_stop)
            if brake_start is None:
                raise ValueError(
                    f"No brake_engaged low-to-high transition found in [{search_start}, {search_stop}] s"
                )
            display_start = brake_start
            print(f"Starting rollout at brake_engaged transition: {brake_start:.6f} s")

    initial = imu_times - imu_times[0] <= args.imu_bias_window
    if not np.any(initial):
        raise ValueError("imu-bias-window contains no IMU samples")
    imu_bias = imu_values[initial].mean(axis=0)
    if args.no_motion_bias:
        stationary = no_motion_mask(
            imu_times, odom_times, odom_twists, args.no_motion_threshold, args.no_motion_window
        )
        if not np.any(stationary):
            raise ValueError(
                "No IMU samples fall in odom no-motion periods; raise --no-motion-threshold or lower --no-motion-window"
            )
        imu_bias[:2] = imu_values[stationary, :2].mean(axis=0)
        print(f"IMU accel bias from {np.count_nonzero(stationary)}/{len(imu_times)} odom no-motion samples")
    print(f"IMU bias: ax={imu_bias[0]:.4f} m/s^2, ay={imu_bias[1]:.4f} m/s^2, omega={imu_bias[2]:.5f} rad/s")

    if display_start is not None or display_stop is not None:
        odom_in = np.ones(len(odom_times), dtype=bool)
        odom_pose_in = np.ones(len(odom_pose_times), dtype=bool)
        imu_in = np.ones(len(imu_times), dtype=bool)
        if display_start is not None:
            odom_in &= odom_times >= display_start
            odom_pose_in &= odom_pose_times >= display_start
            imu_in &= imu_times >= display_start
        if display_stop is not None:
            odom_in &= odom_times <= display_stop
            odom_pose_in &= odom_pose_times <= display_stop
            imu_in &= imu_times <= display_stop
        if (
            np.count_nonzero(odom_in) < 2
            or np.count_nonzero(odom_pose_in) < 2
            or np.count_nonzero(imu_in) < 2
        ):
            raise ValueError(
                f"Selected rollout interval [{display_start}, {display_stop}] s needs at least two odom, "
                "odom-pose, and IMU samples"
            )
    else:
        odom_in = np.ones(len(odom_times), dtype=bool)
        odom_pose_in = np.ones(len(odom_pose_times), dtype=bool)
        imu_in = np.ones(len(imu_times), dtype=bool)

    odom_states = integrate_body_twist(odom_times, odom_twists)
    imu_states, imu_twists = integrate_imu(imu_times, imu_values, imu_bias)
    odom_origin = state_at(odom_times, odom_states, brake_start) if brake_start is not None else None
    imu_origin = state_at(imu_times, imu_states, brake_start) if brake_start is not None else None
    odom_pose_origin = (
        state_at(odom_pose_times, odom_pose_states, brake_start) if brake_start is not None else None
    )
    rollouts = [
        (
            "odom velocity integrated",
            ODOM_COLOR,
            odom_times[odom_in],
            odom_twists[odom_in],
            rebase_states(odom_states[odom_in], odom_origin),
        ),
        ("imu", IMU_COLOR, imu_times[imu_in], imu_twists[imu_in], rebase_states(imu_states[imu_in], imu_origin)),
    ]
    path_rollouts = [
        *rollouts,
        (
            "odom position recorded",
            "tab:green",
            odom_pose_times[odom_pose_in],
            np.zeros((np.count_nonzero(odom_pose_in), 3)),
            rebase_states(odom_pose_states[odom_pose_in], odom_pose_origin),
        ),
    ]
    for label, _, times, _, states in rollouts:
        print(
            f"{label}: {len(times)} samples over {times[-1] - times[0]:.3f} s; final pose "
            f"x={states[-1, 0]:.4f} m, y={states[-1, 1]:.4f} m, theta={np.degrees(states[-1, 2]):.2f} deg"
        )

    fig, (velocity_axis, path_axis) = plt.subplots(1, 2, figsize=(15, 7))
    plot_body_velocities(velocity_axis, rollouts)
    plot_rollout_paths(path_axis, path_rollouts, RobotFootprint(), args.footprint_stride)
    fig.suptitle(args.csv.name)
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
