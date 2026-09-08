#!/usr/bin/env python3
"""Overlay trajectories from one or more rollout .npz archives in a PNG image."""
import argparse
import sys
from pathlib import Path

import matplotlib

if __name__ == "__main__" and "--show-figure" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union


def load_archive(path):
    """Load and validate the padded pose histories stored in a rollout archive."""
    archive = np.load(path, allow_pickle=False)
    required_keys = {
        "field_name",
        "poses",
        "lengths",
        "footprint_vertices",
        "footprint_names",
        "trajectory_footprint_indices",
    }
    missing_keys = required_keys - set(archive.files)
    if missing_keys:
        raise ValueError(f"{path} is missing required keys: {', '.join(sorted(missing_keys))}")

    poses = archive["poses"]
    lengths = archive["lengths"]
    if poses.ndim != 3 or poses.shape[2] != 3:
        raise ValueError(f"{path} poses must have shape (rollout, step, 3)")
    if lengths.shape != (poses.shape[0],):
        raise ValueError(f"{path} lengths must have one value per rollout")
    if np.any(lengths < 1) or np.any(lengths > poses.shape[1]):
        raise ValueError(f"{path} contains invalid trajectory lengths")
    footprint_vertices = archive["footprint_vertices"]
    footprint_names = archive["footprint_names"]
    trajectory_footprint_indices = archive["trajectory_footprint_indices"]
    if footprint_vertices.ndim != 3 or footprint_vertices.shape[1:] != (4, 2):
        raise ValueError(f"{path} footprint_vertices must have shape (footprint, 4, 2)")
    if footprint_names.shape != (footprint_vertices.shape[0],):
        raise ValueError(f"{path} footprint_names must have one value per footprint")
    if trajectory_footprint_indices.shape != (poses.shape[0],):
        raise ValueError(f"{path} trajectory_footprint_indices must have one value per rollout")
    if np.any(trajectory_footprint_indices < 0) or np.any(trajectory_footprint_indices >= len(footprint_vertices)):
        raise ValueError(f"{path} contains invalid trajectory footprint indices")
    return archive["field_name"].item(), poses, lengths, footprint_vertices, trajectory_footprint_indices


def footprint_corners(footprint_vertices, x, y, yaw):
    """Transform body-frame footprint vertices to a world-frame pose."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return footprint_vertices @ rotation.T + np.array([x, y])


def polygon_parts(geometry):
    """Yield polygon components from a Polygon, MultiPolygon, or collection."""
    if geometry.geom_type == "Polygon":
        yield geometry
    else:
        for part in geometry.geoms:
            yield from polygon_parts(part)


def plot_fused_footprints(axis, trajectories, footprint_vertices, footprint_indices, color, label, alpha, line_width):
    """Union and draw every saved footprint pose in one archive."""
    chunk_size = 256
    fused_chunks = []
    footprint_polygons = []
    for trajectory, footprint_index in zip(trajectories, footprint_indices):
        for x, y, yaw in trajectory:
            footprint_polygons.append(
                ShapelyPolygon(footprint_corners(footprint_vertices[footprint_index], x, y, yaw))
            )
            if len(footprint_polygons) == chunk_size:
                fused_chunks.append(unary_union(footprint_polygons))
                footprint_polygons.clear()
    if footprint_polygons:
        fused_chunks.append(unary_union(footprint_polygons))
    fused_footprint = unary_union(fused_chunks)
    for index, polygon in enumerate(polygon_parts(fused_footprint)):
        exterior = np.asarray(polygon.exterior.coords)
        axis.plot(
            exterior[:, 0],
            exterior[:, 1],
            color=color,
            linewidth=line_width,
            alpha=alpha,
        )
        for interior in polygon.interiors:
            hole = np.asarray(interior.coords)
            axis.plot(hole[:, 0], hole[:, 1], color=color, linewidth=line_width, alpha=alpha)
    hull = np.asarray(fused_footprint.convex_hull.exterior.coords)
    axis.plot(
        hull[:, 0],
        hull[:, 1],
        color=color,
        linewidth=line_width,
        alpha=alpha,
        linestyle="--",
    )


def plot_archive(axis, path, color, label, alpha, line_width, show_trajectories, show_footprints, fuse_footprints):
    """Plot selected trajectory and terminal-footprint data from one archive."""
    field_name, poses, lengths, footprint_vertices, footprint_indices = load_archive(path)
    trajectories = []
    trajectory_footprint_indices = []
    for index, length in enumerate(lengths):
        trajectory = poses[index, :length]
        if show_footprints and fuse_footprints:
            trajectories.append(trajectory)
            trajectory_footprint_indices.append(footprint_indices[index])
        if show_trajectories:
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                alpha=alpha,
                linewidth=line_width,
                label=label if index == 0 else None,
            )
        if show_footprints:
            if not fuse_footprints:
                x, y, yaw = trajectory[-1]
                axis.add_patch(
                    Polygon(
                        footprint_corners(footprint_vertices[footprint_indices[index]], x, y, yaw),
                        closed=True,
                        fill=False,
                        edgecolor=color,
                        linewidth=line_width,
                        alpha=alpha,
                    )
                )
    if show_footprints and fuse_footprints:
        plot_fused_footprints(
            axis,
            trajectories,
            footprint_vertices,
            trajectory_footprint_indices,
            color,
            label,
            alpha,
            line_width,
        )
    return field_name, len(lengths), footprint_vertices


def main():
    parser = argparse.ArgumentParser(
        description="Overlay padded rollout .npz archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Color examples: tab:blue, tab:orange, tab:green, tab:red, black, or '#RRGGBB'.\n"
            "Example: --archive base.npz tab:red --archive sliding.npz '#1f77b4'"
        ),
    )
    parser.add_argument(
        "--archive",
        action="append",
        nargs=2,
        metavar=("PATH", "COLOR"),
        required=True,
        help="Rollout archive path and color; repeat per archive. Use named colors such as tab:blue, tab:orange, tab:green, tab:red, black, or quoted hex '#RRGGBB'.",
    )
    parser.add_argument("--output", type=Path, default=Path("rollout_overlay.png"), help="Output PNG path.")
    parser.add_argument("--show-figure", action="store_true", help="Display the saved overlay figure interactively.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Trajectory opacity in [0, 1].")
    parser.add_argument("--line-width", type=float, default=0.8, help="Trajectory line width.")
    trajectory_group = parser.add_mutually_exclusive_group()
    trajectory_group.add_argument("--show-trajectories", dest="show_trajectories", action="store_true", default=True)
    trajectory_group.add_argument("--hide-trajectories", dest="show_trajectories", action="store_false")
    footprint_group = parser.add_mutually_exclusive_group()
    footprint_group.add_argument("--show-footprints", dest="show_footprints", action="store_true", default=True)
    footprint_group.add_argument("--hide-footprints", dest="show_footprints", action="store_false")
    parser.add_argument(
        "--fuse-footprints",
        action="store_true",
        help="Union every saved footprint pose in each archive into one swept geometry before drawing.",
    )
    args = parser.parse_args()

    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if args.line_width <= 0.0:
        raise ValueError("line-width must be positive")
    if args.fuse_footprints and not args.show_footprints:
        raise ValueError("fuse-footprints requires show-footprints")

    figure, axis = plt.subplots(figsize=(10, 10))
    for archive_path, color in args.archive:
        path = Path(archive_path)
        field_name, count, footprint_vertices = plot_archive(
            axis,
            path,
            color,
            path.stem,
            args.alpha,
            args.line_width,
            args.show_trajectories,
            args.show_footprints,
            args.fuse_footprints,
        )
        if args.show_footprints:
            for footprint_index, footprint_vertices_for_start in enumerate(footprint_vertices):
                axis.add_patch(
                    Polygon(
                        footprint_corners(footprint_vertices_for_start, 0.0, 0.0, 0.0),
                        closed=True,
                        facecolor="lightsteelblue",
                        edgecolor="black",
                        alpha=0.75,
                        label="start footprint" if path == Path(args.archive[0][0]) and footprint_index == 0 else None,
                    )
                )
        print(f"{path}: {count} {field_name} rollouts, color {color}")

    if args.show_trajectories:
        axis.plot(0.0, 0.0, marker="s", color="black", markersize=5, label="start")
    axis.set_title("Rollout Archive Overlay")
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.relim()
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    if args.show_figure:
        plt.show()
    plt.close(figure)
    print(f"Saved overlay to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        plt.close("all")
        sys.exit(130)