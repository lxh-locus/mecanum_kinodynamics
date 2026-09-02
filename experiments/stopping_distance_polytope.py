"""
stopping_distance_polytope.py

For each feasible starting body velocity (constrained by max wheel speed),
compute the minimum stopping distance under constant maximum deceleration
(constrained by max wheel torque), then plot the result.

Stopping distance derivation
------------------------------
Under constant deceleration a applied for time t*, with v(t*) = 0:

    a = -v0 / t*
    displacement = integral_0^{t*} v(t) dt = v0*t* + 0.5*a*t*^2 = 0.5*v0*t*

Translational stopping distance (Euclidean distance of center of mass):

    d_trans = 0.5 * ||[vx, vy]|| * t*(v0)

Angular stopping displacement:

    theta_stop = 0.5 * |vw| * t*(v0)

The minimum stopping time t*(v0) is the Minkowski functional of the
acceleration zonotope Z evaluated at -v0 (same as stopping_time_polytope.py):

    t*(v0) = max_i { n_i · (-v0) / (-d_i) }

where (n_i, d_i) are the half-space equations of Z (scipy convention:
n_i · x + d_i <= 0).
"""
import argparse
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from mecanum_common import Mecanum


def _preferred_colormap():
    """Prefer winter; fall back to a built-in map when unavailable."""
    names = plt.colormaps()
    if "winter" in names:
        return "winter"
    if "cmc.winter" in names:
        return "cmc.winter"
    return "magma"


# ---------------------------------------------------------------------------
# Zonotope helpers
# ---------------------------------------------------------------------------

def _compute_acceleration_zonotope(model, max_torque):
    """Convex hull of the feasible body-acceleration zonotope."""
    G = model._bodya_from_wheeltorque_matrix()
    corners = np.array(list(product([-max_torque, max_torque], repeat=4)))
    image_pts = (G @ corners.T).T
    _, idx = np.unique(np.round(image_pts, 10), axis=0, return_index=True)
    return ConvexHull(image_pts[idx])


def _minkowski_functional(hull, points):
    """
    Minkowski functional of the convex body described by `hull` at each row of
    `points`.  Returns mu such that points/mu lie on the boundary of the body.

    scipy hull convention:  n_i · x + d_i <= 0  (half-space).
    Functional:  mu(p) = max_i { (n_i · p) / (-d_i) }
    """
    eqs = hull.equations          # (F, 4): [n | d]
    n = eqs[:, :3]                # (F, 3)
    d = eqs[:, 3]                 # (F,)
    dot = points @ n.T            # (N, F)
    return np.max(dot / (-d)[np.newaxis, :], axis=1)


# ---------------------------------------------------------------------------
# Kinematic feasibility sampling
# ---------------------------------------------------------------------------

def _sample_feasible_bodyv(model, max_wheel_velocity, range_scale, samples):
    """Sample body-velocity space, return points satisfying |wheel_i| <= w_max."""
    l_plus_w = model.wb_hlength + model.wb_hwidth
    radius = model.wheel_radius
    w_max = max_wheel_velocity

    max_vx = radius * w_max
    max_vy = radius * w_max
    max_vw = radius * w_max / l_plus_w

    vx_vals = np.linspace(-range_scale * max_vx, range_scale * max_vx, samples)
    vy_vals = np.linspace(-range_scale * max_vy, range_scale * max_vy, samples)
    vw_vals = np.linspace(-range_scale * max_vw, range_scale * max_vw, samples)

    vx_g, vy_g, vw_g = np.meshgrid(vx_vals, vy_vals, vw_vals, indexing="ij")
    vx, vy, vw = vx_g.ravel(), vy_g.ravel(), vw_g.ravel()

    wheel = np.column_stack([
        (vx - vy - l_plus_w * vw) / radius,
        (vx + vy + l_plus_w * vw) / radius,
        (vx + vy - l_plus_w * vw) / radius,
        (vx - vy + l_plus_w * vw) / radius,
    ])
    feasible = np.all(np.abs(wheel) <= w_max + 1e-9, axis=1)
    pts = np.column_stack([vx[feasible], vy[feasible], vw[feasible]])
    return pts, (max_vx, max_vy, max_vw)


# ---------------------------------------------------------------------------
# Stopping distance computation
# ---------------------------------------------------------------------------

def _stopping_distances(hull, bodyv_points):
    """
    For each starting body velocity v0, compute:
      t_stop  - minimum stopping time  [s]
      d_trans - translational stopping distance  ||[vx, vy]|| / 2 * t_stop  [m]
      d_rot   - angular stopping displacement    |vw| / 2 * t_stop            [rad]
    """
    t_stop = _minkowski_functional(hull, -bodyv_points)
    v_trans = np.linalg.norm(bodyv_points[:, :2], axis=1)
    v_rot = np.abs(bodyv_points[:, 2])
    d_trans = 0.5 * v_trans * t_stop
    d_rot = 0.5 * v_rot * t_stop
    return t_stop, d_trans, d_rot


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_stopping_distances(
    model,
    max_wheel_velocity,
    max_torque,
    range_scale=1.05,
    samples=22,
    max_points=50000,
):
    """Compute and plot translational and angular stopping distances.

    Args:
        model: Mecanum model providing physical parameters.
        max_wheel_velocity: Absolute wheel-speed limit in rad/s.
        max_torque: Absolute wheel-torque limit in N m.
        range_scale: Multiplier for the plotted velocity range.
        samples: Grid samples per velocity axis.
        max_points: Maximum number of plotted points.
    Returns:
        ``None``; displays the generated matplotlib figure.
    """
    accel_hull = _compute_acceleration_zonotope(model, max_torque)
    bodyv_pts, (max_vx, max_vy, max_vw) = _sample_feasible_bodyv(
        model, max_wheel_velocity, range_scale, samples
    )
    t_stop, d_trans, d_rot = _stopping_distances(accel_hull, bodyv_pts)
    cmap_name = _preferred_colormap()

    if bodyv_pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(bodyv_pts.shape[0], size=max_points, replace=False)
        bodyv_pts = bodyv_pts[idx]
        t_stop = t_stop[idx]
        d_trans = d_trans[idx]
        d_rot = d_rot[idx]

    v_trans = np.linalg.norm(bodyv_pts[:, :2], axis=1)

    fig = plt.figure(figsize=(16, 7))

    # ---- Left: 3D body-velocity space colored by translational stopping distance ----
    ax3d = fig.add_subplot(121, projection="3d")
    sc3d = ax3d.scatter(
        bodyv_pts[:, 0], bodyv_pts[:, 1], bodyv_pts[:, 2],
        c=d_trans,
        cmap=cmap_name,
        s=6,
        alpha=0.65,
        edgecolors="none",
    )
    ax3d.set_xlabel("vx [m/s]")
    ax3d.set_ylabel("vy [m/s]")
    ax3d.set_zlabel("vw [rad/s]")
    ax3d.set_title("Body-Velocity Space Colored by Translational Stopping Distance")
    ax3d.set_xlim(-range_scale * max_vx, range_scale * max_vx)
    ax3d.set_ylim(-range_scale * max_vy, range_scale * max_vy)
    ax3d.set_zlim(-range_scale * max_vw, range_scale * max_vw)
    cb3d = fig.colorbar(sc3d, ax=ax3d, fraction=0.04, pad=0.07)
    cb3d.set_label("Translational stopping distance d_trans [m]")

    # ---- Right: d_trans and theta_stop vs translational speed ----
    ax2d = fig.add_subplot(122)

    sc_trans = ax2d.scatter(
        v_trans, d_trans,
        c=np.abs(bodyv_pts[:, 2]),
        cmap=cmap_name,
        s=10,
        alpha=0.45,
        edgecolors="none",
        label="Translational distance d_trans [m]",
        zorder=2,
    )
    ax2d.scatter(
        v_trans, d_rot,
        c="steelblue",
        s=6,
        alpha=0.25,
        edgecolors="none",
        marker='X',
        label="Angular displacement theta_stop [rad]",
        zorder=1,
    )

    # Reference quadratic: d = v² / (2 * max_decel_trans)
    # Max translational decel = max of row-0 of G times M_max, summed: 4 * k_linear * M_max
    G = model._bodya_from_wheeltorque_matrix()
    a_max_trans = np.sum(np.abs(G[0])) * max_torque
    v_ref = np.linspace(0, np.max(v_trans) * 1.05, 200)
    d_ref = v_ref ** 2 / (2.0 * a_max_trans)
    ax2d.plot(
        v_ref,
        d_ref,
        "k--",
        linewidth=1.2,
        alpha=0.8,
        label=(
            "Reference (pure vx): "
            f"d = v^2/(2 a_max), a_max = {a_max_trans:.2f} m/s^2"
        ),
    )

    ax2d.set_xlabel("Translational speed ||[vx, vy]|| [m/s]")
    ax2d.set_ylabel("Stopping metric (m for d_trans, rad for theta_stop)")
    ax2d.set_title("Stopping Metrics vs Translational Speed (color = |vw| [rad/s])")
    ax2d.legend(fontsize=8, framealpha=0.6)
    cb2d = fig.colorbar(sc_trans, ax=ax2d, fraction=0.04, pad=0.04)
    cb2d.set_label("|vw| [rad/s]")

    fig.suptitle(
        f"Mecanum Stopping Distance and Rotation  "
        f"(w_max = {max_wheel_velocity} rad/s,  M_max = {max_torque} N·m)",
        y=0.99,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot minimum stopping distances for a mecanum drive, given "
            "kinematic (wheel speed) and dynamic (wheel torque) limits."
        )
    )
    parser.add_argument(
        "--max-wheel-velocity",
        type=float,
        default=10.0,
        help="Maximum absolute wheel velocity in rad/s (default: 10.0).",
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
        default=1.05,
        help="Axis range multiplier relative to kinematic limits (default: 1.05).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=23,
        help="Sampling resolution per axis in body-velocity space (default: 23).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Max points drawn after random downsampling (default: 50000).",
    )
    args = parser.parse_args()

    if args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    if args.max_torque <= 0.0:
        raise ValueError("max-torque must be positive")
    if args.range_scale <= 0.0:
        raise ValueError("range-scale must be positive")
    if args.samples < 2:
        raise ValueError("samples must be at least 2")
    if args.max_points < 1:
        raise ValueError("max-points must be at least 1")

    model = Mecanum()
    _plot_stopping_distances(
        model=model,
        max_wheel_velocity=args.max_wheel_velocity,
        max_torque=args.max_torque,
        range_scale=args.range_scale,
        samples=args.samples,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
