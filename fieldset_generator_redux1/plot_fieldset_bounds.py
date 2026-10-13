#!/usr/bin/env python3
"""Plot the config-derived kinematic velocity polytope with fieldset velocity boxes overlaid.

--fields accepts exact names, 'all', or shell-style wildcards (*, ?, [seq]); quote
wildcards so the shell does not expand them. Examples:
    --fields 'fast*'          non-appliance fast fields
    --fields '*fast*'         every fast field, appliance included
    --fields 'appliance_*'    every appliance field
    --fields 'fast?*'         fast_* variants but not plain 'fast' (? matches exactly one character)
    --fields stationary '*strafe*'

--exclude takes the same pattern syntax and removes matches from the --fields selection:
    --exclude 'appliance_*'                   all non-appliance fields
    --fields '*fast*' --exclude '*reverse'    forward fast fields, appliance included
"""
import argparse
import fnmatch
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.kinematic_boundary_rollout_limited import _polytope_faces, compute_wheel_velocity_limits
from experiments.mecanum_common import Mecanum
from experiments.mecanum_physics import MecanumPhysicsParams

DEFAULT_CONFIG = Path(__file__).with_name("fieldset_config.json")
AXES = ("x", "y", "yaw")
BOX_FACE_INDICES = ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7])
BOX_EDGE_INDICES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def velocity_ranges(dynamic_limit):
    velocity = dynamic_limit["velocity"]
    return tuple((float(velocity["min"][axis]), float(velocity["max"][axis])) for axis in AXES)


def box_vertices(ranges):
    (x0, x1), (y0, y1), (w0, w1) = ranges
    return np.array(
        [
            [x0, y0, w0], [x1, y0, w0], [x1, y1, w0], [x0, y1, w0],
            [x0, y0, w1], [x1, y0, w1], [x1, y1, w1], [x0, y1, w1],
        ],
        dtype=float,
    )


def draw_box(ax, ranges, color, style, alpha):
    verts = box_vertices(ranges)
    if style == "polytope":
        ax.add_collection3d(
            Poly3DCollection(
                [verts[idx] for idx in BOX_FACE_INDICES],
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                linewidth=0.8,
            )
        )
    else:
        ax.add_collection3d(
            Line3DCollection([verts[list(edge)] for edge in BOX_EDGE_INDICES], colors=color, linewidths=1.4)
        )


def main():
    parser = argparse.ArgumentParser(description="Overlay fieldset velocity boxes on the kinematic velocity polytope.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG, help="Input fieldset JSON file.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["all"],
        help="Field-config names or wildcard patterns (e.g. 'fast*', 'appliance_*') to overlay, or 'all' (default).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Field-config names or wildcard patterns to remove from the --fields selection.",
    )
    parser.add_argument(
        "--style",
        choices=["polytope", "wireframe"],
        default="wireframe",
        help="Render field boxes as translucent filled polytopes or edge-only wireframes.",
    )
    parser.add_argument("--box-alpha", type=float, default=0.15, help="Face alpha for --style polytope.")
    parser.add_argument("--z-scale", type=float, default=0.5, help="Visual compression of the omega axis.")
    parser.add_argument(
        "--max-wheel-velocity",
        type=float,
        default=None,
        help="Wheel-speed limit [rad/s]; defaults to robot_revision dynamic_limit velocity max x / wheel radius.",
    )
    parser.add_argument("--output", type=Path, help="Save the figure to this path instead of showing it.")
    parser.add_argument("--list", action="store_true", help="List available field names and exit.")
    args = parser.parse_args()
    if args.z_scale <= 0.0:
        raise ValueError("z-scale must be positive")
    if args.max_wheel_velocity is not None and args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")

    with args.config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    robot_revision = config["robot_revision"]
    field_configs = config["field_configs"]

    if args.list:
        for field in field_configs:
            print(field["name"])
        return

    if "all" in args.fields:
        fields = field_configs
    else:
        unmatched = [
            pattern
            for pattern in args.fields
            if not any(fnmatch.fnmatchcase(field["name"], pattern) for field in field_configs)
        ]
        if unmatched:
            raise ValueError(f"No fields match: {', '.join(unmatched)}")
        fields = [
            field
            for field in field_configs
            if any(fnmatch.fnmatchcase(field["name"], pattern) for pattern in args.fields)
        ]
    fields = [
        field
        for field in fields
        if not any(fnmatch.fnmatchcase(field["name"], pattern) for pattern in args.exclude)
    ]
    if not fields:
        raise ValueError("No fields left to plot after applying --exclude")

    params = MecanumPhysicsParams()
    model = Mecanum(params=params)
    max_wheel_velocity = args.max_wheel_velocity
    if max_wheel_velocity is None:
        max_wheel_velocity = float(robot_revision["dynamic_limit"]["velocity"]["max"]["x"]) / params.wheel_radius
    print(f"Max wheel velocity: {max_wheel_velocity:.4f} rad/s")
    # Box limits well outside the wheel-speed extents so the full polytope is untruncated.
    loose_ranges = [(-2.0 * limit, 2.0 * limit) for limit in compute_wheel_velocity_limits(model, max_wheel_velocity)]
    vertices, faces = _polytope_faces(model, max_wheel_velocity, *loose_ranges)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(faces, facecolor="lightgray", edgecolor="dimgray", alpha=0.2, linewidth=0.8)
    )

    cmap = plt.get_cmap("tab20" if len(fields) > 10 else "tab10")
    handles = [Patch(facecolor="lightgray", edgecolor="dimgray", label="kinematic bounds")]
    extent_points = [vertices]
    for i, field in enumerate(fields):
        color = cmap(i % cmap.N)
        ranges = velocity_ranges(field["dynamic_limit"])
        draw_box(ax, ranges, color, args.style, args.box_alpha)
        extent_points.append(box_vertices(ranges))
        handles.append(Patch(facecolor=color, edgecolor=color, label=field["name"]))

    extent_points = np.vstack(extent_points)
    lo = extent_points.min(axis=0)
    hi = extent_points.max(axis=0)
    pad = 0.1 * np.maximum(hi - lo, 1e-6)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    aspect = hi - lo + 2 * pad
    aspect[2] *= args.z_scale
    ax.set_box_aspect(aspect)
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.set_zlabel("omega [rad/s]")
    ax.set_title(f"Kinematic bounds with fieldset velocity boxes ({args.style})")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize="small")
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
