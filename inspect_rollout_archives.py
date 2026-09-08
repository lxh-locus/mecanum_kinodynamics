#!/usr/bin/env python3
"""Interactively overlay rollout archives and inspect velocity-matched rollouts.

Click any trajectory endpoint in the overlay panel to select that rollout. The
detail panel then draws that rollout together with the nearest-initial-velocity
rollout from every other archive, including their swept footprints.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from overlay_rollout_archives import footprint_corners, load_archive

MAX_DETAIL_FOOTPRINTS = 80


def load_inspection_archive(path):
    """Load an archive plus the initial velocities needed for matching."""
    field_name, poses, lengths, footprint_vertices, footprint_indices = load_archive(path)
    archive = np.load(path, allow_pickle=False)
    if "initial_velocities" not in archive.files:
        raise ValueError(f"{path} is missing required key: initial_velocities")
    initial_velocities = archive["initial_velocities"]
    if initial_velocities.shape != (poses.shape[0], 3):
        raise ValueError(f"{path} initial_velocities must have shape (rollout, 3)")
    dt = float(archive["dt"]) if "dt" in archive.files else float("nan")
    if dt <= 0.0:
        raise ValueError(f"{path} dt must be positive")
    return {
        "path": Path(path),
        "label": Path(path).stem,
        "field_name": field_name,
        "poses": poses,
        "lengths": lengths,
        "footprint_vertices": footprint_vertices,
        "footprint_indices": footprint_indices,
        "initial_velocities": initial_velocities,
        "dt": dt,
    }


def trajectory(archive, index):
    """Return the unpadded pose history of one rollout."""
    return archive["poses"][index, : archive["lengths"][index]]


def endpoints(archive):
    """Return the final (x, y) of every rollout in an archive."""
    return np.array([trajectory(archive, index)[-1, :2] for index in range(len(archive["lengths"]))])


def body_velocities(archive, index):
    """Approximate body-frame (vx, vy, omega) by forward-differencing the pose history."""
    path = trajectory(archive, index)
    dt = archive["dt"]
    if len(path) < 2 or not np.isfinite(dt):
        return np.zeros(0), np.zeros((0, 3))
    deltas = np.diff(path, axis=0) / dt
    yaw = path[:-1, 2]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # rotate the world-frame difference back into the body frame at each step
    velocities = np.column_stack(
        (
            cos_yaw * deltas[:, 0] + sin_yaw * deltas[:, 1],
            -sin_yaw * deltas[:, 0] + cos_yaw * deltas[:, 1],
            deltas[:, 2],
        )
    )
    return np.arange(len(velocities)) * dt, velocities


def match_index(archive, velocity, omega_weight):
    """Return the rollout whose initial velocity is closest to the target."""
    scale = np.array([1.0, 1.0, omega_weight])
    distances = np.linalg.norm((archive["initial_velocities"] - velocity) * scale, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def draw_overlay(axis, archives, colors, alpha, line_width):
    """Draw every trajectory and a pickable endpoint scatter per archive."""
    pickable = []
    for archive, color in zip(archives, colors):
        for index in range(len(archive["lengths"])):
            path = trajectory(archive, index)
            axis.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                alpha=alpha,
                linewidth=line_width,
                label=archive["label"] if index == 0 else None,
                zorder=1,
            )
        terminal = endpoints(archive)
        scatter = axis.scatter(
            terminal[:, 0],
            terminal[:, 1],
            s=14,
            color=color,
            edgecolors="black",
            linewidths=0.3,
            alpha=min(1.0, alpha + 0.35),
            picker=5,
            zorder=3,
        )
        pickable.append(scatter)
    axis.plot(0.0, 0.0, marker="s", color="black", markersize=6, zorder=4, label="start")
    axis.set_title("Click a trajectory endpoint")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    return pickable


def draw_rollout(axis, archive, index, color, label):
    """Draw one rollout's trajectory, swept footprints, and terminal pose."""
    path = trajectory(archive, index)
    vertices = archive["footprint_vertices"][archive["footprint_indices"][index]]
    stride = max(1, len(path) // MAX_DETAIL_FOOTPRINTS)
    for x, y, yaw in path[::stride]:
        axis.add_patch(
            Polygon(
                footprint_corners(vertices, x, y, yaw),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=0.6,
                alpha=0.35,
            )
        )
    x, y, yaw = path[-1]
    axis.add_patch(
        Polygon(
            footprint_corners(vertices, x, y, yaw),
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=1.6,
        )
    )
    axis.add_patch(
        Polygon(
            footprint_corners(vertices, *path[0]),
            closed=True,
            facecolor="lightsteelblue",
            edgecolor=color,
            alpha=0.5,
        )
    )
    axis.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, label=label, zorder=3)


def draw_velocity(axis, entries):
    """Draw the body-frame initial velocity vectors of the selected rollouts."""
    axis.clear()
    axis.set_title("initial velocity (body frame)", fontsize="medium")
    axis.set_xlabel("vx [m/s]")
    axis.set_ylabel("vy [m/s]")
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    axis.axvline(0.0, color="0.7", linewidth=0.8)
    limit = 0.1
    for row, (color, velocity, _) in enumerate(entries):
        vx, vy, omega = velocity
        axis.annotate(
            "",
            xy=(vx, vy),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2.0},
        )
        # stacked in axes coords so near-identical velocity vectors do not overprint
        axis.annotate(
            f"ω={omega:+.3f}",
            xy=(0.02, 0.96 - 0.07 * row),
            xycoords="axes fraction",
            color=color,
            fontsize="small",
        )
        limit = max(limit, abs(vx), abs(vy))
    limit *= 1.3
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)


def draw_velocity_profile(axis, entries):
    """Draw finite-difference body-frame velocity histories for the selected rollouts."""
    axis.clear()
    axis.set_title("differentiated body velocity vs time", fontsize="medium")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("vx, vy [m/s]   ω [rad/s]")
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    styles = (("vx", "-"), ("vy", "--"), ("ω", ":"))
    for color, _, info in entries:
        archive, index = info[0], info[1]
        times, velocities = body_velocities(archive, index)
        if len(times) == 0:
            continue
        for axis_index, (name, style) in enumerate(styles):
            axis.plot(
                times,
                velocities[:, axis_index],
                color=color,
                linestyle=style,
                linewidth=1.4,
                label=f"{archive['label']} {name}",
            )
    axis.grid(True, alpha=0.25)
    if entries:
        axis.legend(loc="best", fontsize="x-small", ncol=len(entries))


def describe(entries):
    """Build the text summary for the selected and matched rollouts."""
    lines = []
    for _, velocity, info in entries:
        archive, index, distance, is_selection = info
        path = trajectory(archive, index)
        heading = "selected" if is_selection else f"match (Δv={distance:.3f})"
        lines.append(
            f"{archive['label']} [{heading}]\n"
            f"  rollout {index}  vx={velocity[0]:+.3f} vy={velocity[1]:+.3f} ω={velocity[2]:+.3f}\n"
            f"  steps {len(path)}  end ({path[-1, 0]:+.3f}, {path[-1, 1]:+.3f}) yaw {path[-1, 2]:+.3f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively inspect velocity-matched rollouts across archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: --archive base.npz tab:blue --archive sliding.npz tab:red",
    )
    parser.add_argument(
        "--archive",
        action="append",
        nargs=2,
        metavar=("PATH", "COLOR"),
        required=True,
        help="Rollout archive path and color; repeat per archive.",
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Overlay trajectory opacity in (0, 1].")
    parser.add_argument("--line-width", type=float, default=0.7, help="Overlay trajectory line width.")
    parser.add_argument(
        "--omega-weight",
        type=float,
        default=1.0,
        help="Weight applied to the yaw-rate term when matching initial velocities.",
    )
    args = parser.parse_args()

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if args.line_width <= 0.0:
        raise ValueError("line-width must be positive")
    if args.omega_weight < 0.0:
        raise ValueError("omega-weight must be non-negative")

    archives = [load_inspection_archive(path) for path, _ in args.archive]
    colors = [color for _, color in args.archive]
    for archive, color in zip(archives, colors):
        print(f"{archive['path']}: {len(archive['lengths'])} {archive['field_name']} rollouts, color {color}")

    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.6, 1.0])
    overlay_axis = figure.add_subplot(grid[0, 0])
    profile_axis = figure.add_subplot(grid[1, 0])
    detail_axis = figure.add_subplot(grid[0, 1])
    velocity_axis = figure.add_subplot(grid[1, 1])

    scatters = draw_overlay(overlay_axis, archives, colors, args.alpha, args.line_width)
    scatter_archive = {id(scatter): position for position, scatter in enumerate(scatters)}
    detail_axis.set_title("select an endpoint to inspect a rollout")
    detail_axis.set_aspect("equal", adjustable="box")
    detail_axis.grid(True, alpha=0.25)
    draw_velocity(velocity_axis, [])
    draw_velocity_profile(profile_axis, [])
    summary = figure.text(0.52, 0.015, "", fontsize="small", va="bottom", family="monospace")

    def on_pick(event):
        position = scatter_archive.get(id(event.artist))
        if position is None or len(event.ind) == 0:
            return
        selected_archive = archives[position]
        selected_index = int(event.ind[0])
        velocity = selected_archive["initial_velocities"][selected_index]

        entries = [(colors[position], velocity, (selected_archive, selected_index, 0.0, True))]
        for other, color in enumerate(colors):
            if other == position:
                continue
            matched_index, distance = match_index(archives[other], velocity, args.omega_weight)
            entries.append(
                (
                    color,
                    archives[other]["initial_velocities"][matched_index],
                    (archives[other], matched_index, distance, False),
                )
            )

        detail_axis.clear()
        for color, _, (archive, index, distance, is_selection) in entries:
            label = archive["label"] if is_selection else f"{archive['label']} (Δv={distance:.3f})"
            draw_rollout(detail_axis, archive, index, color, label)
        detail_axis.set_title(f"rollout {selected_index} of {selected_archive['label']}", fontsize="medium")
        detail_axis.set_xlabel("world x [m]")
        detail_axis.set_ylabel("world y [m]")
        detail_axis.relim()
        detail_axis.autoscale_view()
        detail_axis.set_aspect("equal", adjustable="box")
        detail_axis.grid(True, alpha=0.25)
        detail_axis.legend(loc="best", fontsize="small")

        draw_velocity(velocity_axis, entries)
        draw_velocity_profile(profile_axis, entries)
        summary.set_text(describe(entries))
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("pick_event", on_pick)
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)
