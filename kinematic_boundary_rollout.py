import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from mecanum_common import Mecanum


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


def _sample_points_on_face(face_points, samples, rng):
    """Sample points approximately uniformly on a convex face polygon."""
    center = np.mean(face_points, axis=0)
    sampled = []

    for i in range(face_points.shape[0]):
        p0 = face_points[i]
        p1 = face_points[(i + 1) % face_points.shape[0]]

        # Sample barycentric coordinates over triangle (center, p0, p1).
        for _ in range(max(1, samples // face_points.shape[0])):
            r1 = np.sqrt(rng.random())
            r2 = rng.random()
            a = 1.0 - r1
            b = r1 * (1.0 - r2)
            c = r1 * r2
            sampled.append(a * center + b * p0 + c * p1)

    if len(sampled) == 0:
        return np.empty((0, 3), dtype=float)
    return np.array(sampled, dtype=float)


def sample_boundary_velocities_random(model, max_wheel_velocity, total_samples=8, seed=0):
    """Sample [vx, vy, omega] commands on the kinematic polytope boundary."""
    A, b = _build_inequalities(model, max_wheel_velocity)
    vertices = _compute_polytope_vertices(A, b)
    if vertices.shape[0] == 0:
        raise ValueError("Could not compute polytope vertices for current parameters")

    faces = _build_face_polygons(vertices, A, b)
    if len(faces) == 0:
        raise ValueError("Could not compute polytope boundary faces")

    rng = np.random.default_rng(seed)

    # Split requested samples across faces, then trim to exact count.
    per_face = max(1, int(np.ceil(total_samples / len(faces))))
    all_pts = []
    for face in faces:
        pts = _sample_points_on_face(face, per_face, rng)
        if pts.shape[0] > 0:
            all_pts.append(pts)

    boundary_pts = np.vstack(all_pts)

    # Keep unique-ish points and shuffle before truncating.
    rounded = np.round(boundary_pts, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    boundary_pts = boundary_pts[np.sort(unique_idx)]
    rng.shuffle(boundary_pts)

    if boundary_pts.shape[0] < total_samples:
        return boundary_pts
    return boundary_pts[:total_samples]


def _ray_to_polytope_boundary(origin, direction, A, b, atol=1e-12):
    """Intersect ray origin + t*direction (t>=0) with boundary of A @ x <= b."""
    ad = A @ direction
    ac = A @ origin

    valid = ad > atol
    if not np.any(valid):
        # Fallback: reverse direction if forward ray does not exit.
        direction = -direction
        ad = A @ direction
        valid = ad > atol
        if not np.any(valid):
            raise ValueError("Could not find exiting ray direction for boundary intersection")

    t_candidates = (b[valid] - ac[valid]) / ad[valid]
    t = np.min(t_candidates)
    return origin + t * direction


def _slice_polygon_vertices(A, b, omega, atol=1e-9):
    """Return CCW-ordered (vx, vy) vertices of the polytope cross-section at `omega`."""
    A2 = A[:, :2]
    b2 = b - A[:, 2] * omega

    verts = []
    for i, j in combinations(range(A2.shape[0]), 2):
        M = np.vstack([A2[i], A2[j]])
        if np.isclose(np.linalg.det(M), 0.0, atol=atol):
            continue

        p = np.linalg.solve(M, np.array([b2[i], b2[j]], dtype=float))
        if np.all(A2 @ p <= b2 + 1e-8):
            if not any(np.allclose(p, q, atol=1e-8) for q in verts):
                verts.append(p)

    if not verts:
        return np.empty((0, 2), dtype=float)

    pts = np.array(verts, dtype=float)
    center = np.mean(pts, axis=0)
    rel = pts - center
    return pts[np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))]


def sample_boundary_velocities_bisect(model, max_wheel_velocity, bisect_tier=0, n_spacing=1):
    """
    Deterministic boundary sampler that bisects only along the omega axis.

    `bisect_tier` t splits [-omega_max, omega_max] into 2^t sub-intervals per side,
    yielding omega levels k * omega_max / 2^t for k in (-2^t, 2^t); the degenerate
    endpoints +/-omega_max are skipped. Each omega slice is a convex polygon in
    (vx, vy); `n_spacing` evenly spaced points are taken per polygon edge, starting
    at each vertex, giving 4 * n_spacing points per slice for this polytope.
    """
    if bisect_tier < 0:
        raise ValueError("bisect_tier must be non-negative")
    if n_spacing < 1:
        raise ValueError("n_spacing must be at least 1")

    A, b = _build_inequalities(model, max_wheel_velocity)
    vertices = _compute_polytope_vertices(A, b)
    if vertices.shape[0] == 0:
        raise ValueError("Could not compute polytope vertices for current parameters")

    omega_max = float(np.max(vertices[:, 2]))
    divisions = 2 ** bisect_tier

    points = []
    for k in range(-(divisions - 1), divisions):
        omega = omega_max * k / divisions
        poly = _slice_polygon_vertices(A, b, omega)
        if poly.shape[0] == 0:
            continue

        if poly.shape[0] < 3:
            for p in poly:
                points.append([p[0], p[1], omega])
            continue

        for idx in range(poly.shape[0]):
            p0 = poly[idx]
            p1 = poly[(idx + 1) % poly.shape[0]]
            for s in range(n_spacing):
                q = p0 + (s / n_spacing) * (p1 - p0)
                points.append([q[0], q[1], omega])

    if not points:
        raise ValueError("Omega-slice sampler produced no boundary points")

    return np.array(points, dtype=float)


def rollout_constant_twist(vx, vy, omega, horizon, dt):
    """Roll out a planar pose trajectory from constant body-frame twist."""
    steps = int(np.ceil(horizon / dt))
    states = np.zeros((steps + 1, 3), dtype=float)  # [x, y, theta]

    for k in range(steps):
        x, y, theta = states[k]
        c = np.cos(theta)
        s = np.sin(theta)

        x_dot = c * vx - s * vy
        y_dot = s * vx + c * vy

        states[k + 1, 0] = x + dt * x_dot
        states[k + 1, 1] = y + dt * y_dot
        states[k + 1, 2] = theta + dt * omega

    return states


def _rectangle_corners(x, y, theta, half_length, half_width):
    """Return 4 world-frame corners for the oriented chassis rectangle."""
    # Local corners in counter-clockwise order.
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )

    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    world = local @ R.T
    world[:, 0] += x
    world[:, 1] += y
    return world


def plot_boundary_rollouts(
    model,
    boundary_velocities,
    horizon=3.0,
    dt=0.01,
    rectangle_stride=30,
    show_final_only=True,
):
    """Plot rollouts and chassis rectangles for sampled boundary velocities."""
    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = plt.cm.turbo(np.linspace(0.0, 1.0, boundary_velocities.shape[0]))

    for i, cmd in enumerate(boundary_velocities):
        vx, vy, omega = cmd
        states = rollout_constant_twist(vx, vy, omega, horizon=horizon, dt=dt)

        ax.plot(states[:, 0], states[:, 1], color=cmap[i], linewidth=1.2, alpha=0.9)

        if show_final_only:
            idxs = [states.shape[0] - 1]
        else:
            idxs = list(range(0, states.shape[0], rectangle_stride))
            if idxs[-1] != states.shape[0] - 1:
                idxs.append(states.shape[0] - 1)

        for j, idx in enumerate(idxs):
            x, y, theta = states[idx]
            corners = _rectangle_corners(
                x=x,
                y=y,
                theta=theta,
                half_length=model.wb_hlength,
                half_width=model.wb_hwidth,
            )
            alpha = 0.12 if (not show_final_only and j < len(idxs) - 1) else 0.35
            poly = Polygon(corners, closed=True, fill=False, edgecolor=cmap[i], linewidth=0.9, alpha=alpha)
            ax.add_patch(poly)

    ax.set_title("Boundary-Velocity Rollouts with Oriented Chassis Footprint")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


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
    )


if __name__ == "__main__":
    main()
