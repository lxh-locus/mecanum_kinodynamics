import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

from mecanum_common import Mecanum


def _compute_acceleration_zonotope(model, max_torque):
    """
    Compute the convex hull of the feasible body-acceleration set.

    The feasible set is the image of the torque hypercube [-M, M]^4
    under the linear map G (bodya_from_wheeltorque_matrix), which is
    a zonotope in (ax, ay, alpha) space.
    """
    G = model._bodya_from_wheeltorque_matrix()
    corners = np.array(list(product([-max_torque, max_torque], repeat=4)))
    image_points = (G @ corners.T).T  # shape (16, 3)
    _, unique_idx = np.unique(np.round(image_points, 10), axis=0, return_index=True)
    unique_points = image_points[unique_idx]
    hull = ConvexHull(unique_points)
    return unique_points, hull


def _build_face_patches(vertices, hull):
    """Return triangle vertex arrays for every ConvexHull simplex face."""
    return [vertices[simplex] for simplex in hull.simplices]


def _sample_feasible_bodya(model, max_torque, hull, range_scale, samples):
    """
    Sample body-acceleration space and keep points inside the zonotope.

    Utilization is the L-inf norm of the minimum-norm (pseudoinverse) torque
    solution, normalised by max_torque. This is a lower bound on the true
    minimum torque utilization and serves as a colormap proxy.
    """
    G = model._bodya_from_wheeltorque_matrix()
    G_pinv = np.linalg.pinv(G)

    lims = np.max(np.abs(hull.points), axis=0) * range_scale  # [lax, lay, lalpha]

    ax_vals = np.linspace(-lims[0], lims[0], samples)
    ay_vals = np.linspace(-lims[1], lims[1], samples)
    alpha_vals = np.linspace(-lims[2], lims[2], samples)

    ax_g, ay_g, alpha_g = np.meshgrid(ax_vals, ay_vals, alpha_vals, indexing="ij")
    points = np.column_stack([ax_g.ravel(), ay_g.ravel(), alpha_g.ravel()])

    # Point is inside hull iff hull.equations @ [p, 1]^T <= 0 for all half-spaces.
    eqs = hull.equations
    inside = np.all(points @ eqs[:, :3].T + eqs[:, 3] <= 1e-9, axis=1)
    feasible = points[inside]

    torques_pinv = feasible @ G_pinv.T  # min-norm torque for each point
    usage = np.max(np.abs(torques_pinv), axis=1) / max_torque

    return feasible, usage, lims


def _plot_acceleration_limits(
    model,
    max_torque,
    range_scale=1.15,
    volume_samples=24,
    max_volume_points=45000,
):
    """Plot zonotope boundary patches and sampled feasible acceleration volume."""
    vertices, hull = _compute_acceleration_zonotope(model, max_torque)
    patches = _build_face_patches(vertices, hull)
    feasible, usage, lims = _sample_feasible_bodya(
        model, max_torque, hull, range_scale, volume_samples
    )

    fig = plt.figure(figsize=(16, 7))
    ax_patches = fig.add_subplot(121, projection="3d")
    ax_volume = fig.add_subplot(122, projection="3d")

    # --- Left: zonotope boundary patches ---
    collection = Poly3DCollection(
        patches,
        alpha=0.35,
        facecolor="tab:cyan",
        edgecolor="black",
        linewidth=0.6,
    )
    ax_patches.add_collection3d(collection)
    ax_patches.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], s=14, c="black", alpha=0.8
    )
    ax_patches.set_title("Body-Acceleration Limit Polytope")
    ax_patches.set_xlabel("ax [m/s²]")
    ax_patches.set_ylabel("ay [m/s²]")
    ax_patches.set_zlabel("α [rad/s²]")
    ax_patches.set_xlim(-lims[0], lims[0])
    ax_patches.set_ylim(-lims[1], lims[1])
    ax_patches.set_zlim(-lims[2], lims[2])

    # --- Right: sampled feasible volume (dark theme) ---
    if feasible.shape[0] > max_volume_points:
        rng = np.random.default_rng(0)
        chosen = rng.choice(feasible.shape[0], size=max_volume_points, replace=False)
        feasible = feasible[chosen]
        usage = usage[chosen]

    scatter = ax_volume.scatter(
        feasible[:, 0],
        feasible[:, 1],
        feasible[:, 2],
        c=usage,
        cmap="inferno",
        s=8,
        alpha=0.72,
        edgecolors="none",
    )
    ax_volume.set_facecolor("#0e1117")
    ax_volume.xaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
    ax_volume.yaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
    ax_volume.zaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
    ax_volume.xaxis.pane.set_edgecolor("#8b949e")
    ax_volume.yaxis.pane.set_edgecolor("#8b949e")
    ax_volume.zaxis.pane.set_edgecolor("#8b949e")
    ax_volume.tick_params(colors="#e6edf3")
    ax_volume.set_title("Feasible Body-Acceleration Volume (Sampled)")
    ax_volume.set_xlabel("ax [m/s²]")
    ax_volume.set_ylabel("ay [m/s²]")
    ax_volume.set_zlabel("α [rad/s²]")
    ax_volume.title.set_color("#f0f6fc")
    ax_volume.xaxis.label.set_color("#e6edf3")
    ax_volume.yaxis.label.set_color("#e6edf3")
    ax_volume.zaxis.label.set_color("#e6edf3")
    ax_volume.set_xlim(-lims[0], lims[0])
    ax_volume.set_ylim(-lims[1], lims[1])
    ax_volume.set_zlim(-lims[2], lims[2])
    colorbar = fig.colorbar(scatter, ax=ax_volume, fraction=0.04, pad=0.07)
    colorbar.set_label("min-norm torque / torque limit")
    colorbar.outline.set_edgecolor("#8b949e")
    colorbar.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3")
    colorbar.ax.yaxis.label.set_color("#e6edf3")

    fig.suptitle(
        f"Mecanum Body-Acceleration Limits  (max wheel torque = {max_torque} N·m)",
        y=0.97,
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot 3D body-acceleration limit polytope for a mecanum drive "
            "given a maximum wheel torque."
        )
    )
    parser.add_argument(
        "--max-torque",
        type=float,
        default=3.5,
        help="Maximum absolute wheel torque in N·m (default: 3.5).",
    )
    parser.add_argument(
        "--range-scale",
        type=float,
        default=1.15,
        help="Axis range multiplier for the plot window (default: 1.15).",
    )
    parser.add_argument(
        "--volume-samples",
        type=int,
        default=24,
        help="Sampling resolution per axis for feasible-volume points (default: 24).",
    )
    parser.add_argument(
        "--max-volume-points",
        type=int,
        default=45000,
        help="Max points drawn in the volume plot after random downsampling (default: 45000).",
    )
    args = parser.parse_args()

    if args.max_torque <= 0.0:
        raise ValueError("max-torque must be positive")
    if args.range_scale <= 0.0:
        raise ValueError("range-scale must be positive")
    if args.volume_samples < 2:
        raise ValueError("volume-samples must be at least 2")
    if args.max_volume_points < 1:
        raise ValueError("max-volume-points must be at least 1")

    model = Mecanum()
    _plot_acceleration_limits(
        model=model,
        max_torque=args.max_torque,
        range_scale=args.range_scale,
        volume_samples=args.volume_samples,
        max_volume_points=args.max_volume_points,
    )


if __name__ == "__main__":
    main()
