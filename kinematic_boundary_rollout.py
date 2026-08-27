import argparse
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from mecanum_common import Mecanum
from kinematic_boundary_rollout_limited import (
    _rectangle_corners,
    plot_boundary_rollouts,
    rollout_constant_twist,
)
from sampling_methods import sample_boundary_velocities_bisect as _sample_boundary_velocities_bisect
from sampling_methods import sample_boundary_velocities_random as _sample_boundary_velocities_random


def _build_inequalities(model, max_wheel_velocity):
    """Build A, b for the kinematic polytope A @ [vx, vy, omega] <= b."""
    l_plus_w = model.wb_hlength + model.wb_hwidth
    radius = model.wheel_radius
    rhs = radius * max_wheel_velocity

    forms = np.array(
        [
            [1.0, -1.0, -l_plus_w],
            [1.0, 1.0, l_plus_w],
            [1.0, 1.0, -l_plus_w],
            [1.0, -1.0, l_plus_w],
        ],
        dtype=float,
    )

    # |f_i(v)| <= rhs  <=>  f_i(v) <= rhs and -f_i(v) <= rhs
    A = np.vstack([forms, -forms])
    b = np.full(A.shape[0], rhs, dtype=float)
    return A, b


def _compute_polytope_vertices(A, b, atol=1e-9):
    """Compute all vertices for the 3D polytope A @ x <= b."""
    vertices = []
    for i, j, k in combinations(range(A.shape[0]), 3):
        M = np.vstack([A[i], A[j], A[k]])
        if np.isclose(np.linalg.det(M), 0.0, atol=atol):
            continue

        x = np.linalg.solve(M, np.array([b[i], b[j], b[k]], dtype=float))
        if np.all(A @ x <= b + atol):
            if not any(np.allclose(x, v, atol=1e-8) for v in vertices):
                vertices.append(x)

    if not vertices:
        return np.empty((0, 3), dtype=float)
    return np.array(vertices, dtype=float)


def _build_face_polygons(vertices, A, b, atol=1e-8):
    """Return ordered polygon points for each active boundary face."""
    faces = []
    for idx in range(A.shape[0]):
        n = A[idx]
        d = b[idx]
        mask = np.isclose(vertices @ n, d, atol=atol)
        pts = vertices[mask]
        if pts.shape[0] < 3:
            continue

        center = np.mean(pts, axis=0)
        n_unit = n / np.linalg.norm(n)

        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if np.abs(np.dot(n_unit, ref)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)

        u = np.cross(n_unit, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(n_unit, u)

        rel = pts - center
        angles = np.arctan2(rel @ v, rel @ u)
        ordered = pts[np.argsort(angles)]
        faces.append(ordered)

    return faces


def sample_boundary_velocities_random(model, max_wheel_velocity, total_samples=8, seed=0):
    """Sample [vx, vy, omega] commands on the kinematic polytope boundary."""
    A, b = _build_inequalities(model, max_wheel_velocity)
    vertices = _compute_polytope_vertices(A, b)
    faces = _build_face_polygons(vertices, A, b)
    return _sample_boundary_velocities_random(vertices, faces, total_samples=total_samples, seed=seed)


def sample_boundary_velocities_bisect(model, max_wheel_velocity, bisect_tier=0, n_spacing=1):
    """
    Deterministic boundary sampler that bisects only along the omega axis.

    `bisect_tier` t splits [-omega_max, omega_max] into 2^t sub-intervals per side,
    yielding omega levels k * omega_max / 2^t for k in (-2^t, 2^t); the degenerate
    endpoints +/-omega_max are skipped. Each omega slice is a convex polygon in
    (vx, vy); `n_spacing` evenly spaced points are taken per polygon edge, starting
    at each vertex, giving 4 * n_spacing points per slice for this polytope.
    """
    A, b = _build_inequalities(model, max_wheel_velocity)
    vertices = _compute_polytope_vertices(A, b)
    return _sample_boundary_velocities_bisect(A, b, vertices, bisect_tier=bisect_tier, n_spacing=n_spacing)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample [vx, vy, omega] commands from the kinematic polytope boundary, "
            "roll out constant-twist trajectories, and plot oriented chassis rectangles."
        )
    )
    parser.add_argument("--max-wheel-velocity", type=float, default=10.0, help="Wheel-speed limit [rad/s].")
    parser.add_argument("--samples", type=int, default=8, help="Number of boundary velocity samples (random method).")
    parser.add_argument("--bisect-tier", type=int, default=0, help="Omega-axis bisection tier (bisect method).")
    parser.add_argument(
        "--n-spacing",
        type=int,
        default=1,
        help="Evenly spaced points per cross-section edge (bisect method).",
    )
    parser.add_argument("--horizon", type=float, default=3.0, help="Rollout horizon [s].")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration step [s].")
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=40,
        help="Draw a chassis rectangle every N trajectory samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for boundary sampling.")
    parser.add_argument(
        "--sampling-method",
        choices=["bisect", "random"],
        default="bisect",
        help="Boundary sampling method: deterministic recursive bisection or random face sampling.",
    )
    parser.add_argument(
        "--show-all-rectangles",
        action="store_true",
        help="Draw chassis rectangles throughout each rollout instead of final-only view.",
    )
    args = parser.parse_args()

    if args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    if args.samples < 1:
        raise ValueError("samples must be at least 1")
    if args.bisect_tier < 0:
        raise ValueError("bisect-tier must be non-negative")
    if args.n_spacing < 1:
        raise ValueError("n-spacing must be at least 1")
    if args.horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.rectangle_stride < 1:
        raise ValueError("rectangle-stride must be at least 1")

    model = Mecanum()
    if args.sampling_method == "random":
        boundary_velocities = sample_boundary_velocities_random(
                    model=model,
                    max_wheel_velocity=args.max_wheel_velocity,
                    total_samples=args.samples,
                    seed=args.seed,
                )
    else:
        boundary_velocities = sample_boundary_velocities_bisect(
                    model=model,
                    max_wheel_velocity=args.max_wheel_velocity,
                    bisect_tier=args.bisect_tier,
                    n_spacing=args.n_spacing,
                )

    plot_boundary_rollouts(
        model=model,
        boundary_velocities=boundary_velocities,
        horizon=args.horizon,
        dt=args.dt,
        rectangle_stride=args.rectangle_stride,
        show_final_only=(not args.show_all_rectangles),
        title_suffix="",
        show_start=False,
    )
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
