"""
stopping_time_polytope.py

For each starting body velocity on the kinematic feasibility boundary (defined by
max wheel speed), compute the minimum stopping time under constant maximum
deceleration (defined by max wheel torque), then plot the result in body-velocity
space colored by stopping time.

Minimum stopping time derivation
---------------------------------
Under constant deceleration a, the body velocity reaches zero at time t* = -v0/a.
The optimal a is the one in the acceleration zonotope Z (image of the torque
hypercube under G) that minimises t* = ||v0|| subject to a*t* = -v0.

Equivalently, for a given v0, the minimum t* such that -v0/t* lies inside Z is:

    t*(v0) = max_i { (-n_i · v0) / b_i }

where (n_i, b_i) are the half-space normals/offsets of Z (hull.equations rows,
with the convention n_i · x <= b_i).  This is the Minkowski functional of Z
evaluated at -v0, computed without any LP.
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
# Zonotope helpers (same approach as dynamic_acceleration_limit_box.py)
# ---------------------------------------------------------------------------

def _compute_acceleration_zonotope(model, max_torque):
    """Convex hull of the feasible body-acceleration zonotope."""
    G = model._bodya_from_wheeltorque_matrix()
    corners = np.array(list(product([-max_torque, max_torque], repeat=4)))
    image_pts = (G @ corners.T).T
    _, idx = np.unique(np.round(image_pts, 10), axis=0, return_index=True)
    unique_pts = image_pts[idx]
    return ConvexHull(unique_pts)


def _minkowski_functional(hull, points):
    """
    For each row p in `points`, return the Minkowski functional of the convex
    body described by `hull`:

        mu(p) = max_i { (n_i · p) / b_i }   where hull.equations = [n_i | b_i]
                                              and  n_i · x + b_i <= 0  (scipy sign)

    The scipy ConvexHull equation convention is: n_i · x + b_i <= 0, so the
    half-space is {x : n_i · x <= -b_i}.  With that sign:

        mu(p) = max_i { (n_i · p) / (-b_i) }  over faces with n_i · p > 0.

    Points inside the body have mu <= 1; on the boundary mu = 1.
    """
    eqs = hull.equations          # shape (F, 4):  [n | d],  n·x + d <= 0
    n = eqs[:, :3]                # normals,  shape (F, 3)
    d = eqs[:, 3]                 # offsets,  n·x <= -d

    # dot[i, f] = n_f · p_i
    dot = points @ n.T            # (N, F)

    # Minkowski functional: max over active faces (those with positive dot)
    rhs = -d                      # (-d_f) > 0 for all f since hull encloses origin
    ratio = dot / rhs[np.newaxis, :]   # (N, F)
    return np.max(ratio, axis=1)  # (N,)


# ---------------------------------------------------------------------------
# Kinematic feasibility: sample body velocities inside wheel-speed polytope
# ---------------------------------------------------------------------------

def _sample_feasible_bodyv(model, max_wheel_velocity, range_scale, samples):
    """Sample body-velocity space and keep points satisfying |wheel_i| <= w_max."""
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
    vx = vx_g.ravel()
    vy = vy_g.ravel()
    vw = vw_g.ravel()

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
# Stopping time computation
# ---------------------------------------------------------------------------

def _stopping_times(hull, bodyv_points):
    """
    For each starting body velocity v0, compute minimum stopping time under
    constant deceleration inside the acceleration zonotope.

    t*(v0) = Minkowski functional of the zonotope evaluated at -v0.

    Zero-velocity points are assigned t* = 0.
    """
    neg_v = -bodyv_points
    return _minkowski_functional(hull, neg_v)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_stopping_times(
    model,
    max_wheel_velocity,
    max_torque,
    range_scale=1.05,
    samples=22,
    max_points=50000,
):
    accel_hull = _compute_acceleration_zonotope(model, max_torque)
    bodyv_pts, (max_vx, max_vy, max_vw) = _sample_feasible_bodyv(
        model, max_wheel_velocity, range_scale, samples
    )
    t_stop = _stopping_times(accel_hull, bodyv_pts)
    cmap_name = _preferred_colormap()

    fig = plt.figure(figsize=(14, 7))

    # ---- Left: stopping time in body-velocity space ----
    ax_main = fig.add_subplot(121, projection="3d")

    if bodyv_pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(bodyv_pts.shape[0], size=max_points, replace=False)
        plot_pts = bodyv_pts[idx]
        plot_t = t_stop[idx]
    else:
        plot_pts = bodyv_pts
        plot_t = t_stop

    sc = ax_main.scatter(
        plot_pts[:, 0], plot_pts[:, 1], plot_pts[:, 2],
        c=plot_t,
        cmap=cmap_name,
        s=6,
        alpha=0.65,
        edgecolors="none",
    )
    ax_main.set_xlabel("vx [m/s]")
    ax_main.set_ylabel("vy [m/s]")
    ax_main.set_zlabel("vw [rad/s]")
    ax_main.set_title("Stopping Time over Kinematic Feasibility Volume")
    ax_main.set_xlim(-range_scale * max_vx, range_scale * max_vx)
    ax_main.set_ylim(-range_scale * max_vy, range_scale * max_vy)
    ax_main.set_zlim(-range_scale * max_vw, range_scale * max_vw)

    cb_main = fig.colorbar(sc, ax=ax_main, fraction=0.04, pad=0.07)
    cb_main.set_label("t* [s]")

    # ---- Right: stopping time vs speed magnitude ----
    ax_scatter = fig.add_subplot(122)
    speed = np.linalg.norm(plot_pts[:, :2], axis=1)   # translational speed

    ax_scatter.scatter(speed, plot_t, c=np.abs(plot_pts[:, 2]),
                       cmap=cmap_name, s=3, alpha=0.4, edgecolors="none")
    ax_scatter.set_xlabel("Translational speed |[vx, vy]| [m/s]")
    ax_scatter.set_ylabel("Minimum stopping time [s]")
    ax_scatter.set_title("Stopping Time vs Translational Speed\n"
                         "(color = |vw| [rad/s])")
    cb2 = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(0, np.max(np.abs(plot_pts[:, 2]))),
            cmap=cmap_name,
        ),
        ax=ax_scatter,
        fraction=0.04,
        pad=0.04,
    )
    cb2.set_label("|vw| [rad/s]")

    fig.suptitle(
        f"Mecanum Stopping Times  "
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
            "Plot minimum stopping times for a mecanum drive, given "
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
    _plot_stopping_times(
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
