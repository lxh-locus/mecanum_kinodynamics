"""
kinematic_boundary_rollout_limited.py

Variant of kinematic_boundary_rollout.py that additionally truncates the
kinematic polytope with user-supplied box limits on [vx, vy, omega] before
sampling boundary velocities and rolling out constant-twist trajectories.
A second figure shows the truncated 3D polytope (boundary patches and
vertices), similar to the plot in kinematic_velocity_limit_box.py.
"""
import argparse
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mecanum_common import Mecanum


def _build_inequalities(model, max_wheel_velocity, vx_range, vy_range, omega_range):
    """Build A, b for the kinematic polytope A @ [vx, vy, omega] <= b,
    truncated by box limits vx_range/vy_range/omega_range."""
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

    vx_min, vx_max = vx_range
    vy_min, vy_max = vy_range
    omega_min, omega_max = omega_range

    box_A = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    box_b = np.array([vx_max, -vx_min, vy_max, -vy_min, omega_max, -omega_min], dtype=float)

    A = np.vstack([A, box_A])
    b = np.concatenate([b, box_b])
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


def _polytope_faces(model, max_wheel_velocity, vx_range, vy_range, omega_range):
    """Compute vertices and ordered face polygons for the truncated polytope."""
    A, b = _build_inequalities(model, max_wheel_velocity, vx_range, vy_range, omega_range)
    vertices = _compute_polytope_vertices(A, b)
    if vertices.shape[0] == 0:
        raise ValueError("Could not compute polytope vertices for current parameters")

    faces = _build_face_polygons(vertices, A, b)
    if len(faces) == 0:
        raise ValueError("Could not compute polytope boundary faces")

    return vertices, faces


def compute_wheel_velocity_limits(model, max_wheel_velocity):
    """Return (max_vx, max_vy, max_omega) reachable from `max_wheel_velocity` alone."""
    l_plus_w = model.wb_hlength + model.wb_hwidth
    radius = model.wheel_radius
    max_vx = radius * max_wheel_velocity
    max_vy = radius * max_wheel_velocity
    max_omega = radius * max_wheel_velocity / l_plus_w
    return max_vx, max_vy, max_omega


def compute_truncated_extents(vertices):
    """Return per-axis (min, max) extents of the truncated polytope's vertices."""
    return tuple(zip(vertices.min(axis=0), vertices.max(axis=0)))


def sample_boundary_velocities(vertices, faces, sampling_degree=0):
    """
    Sample [vx, vy, omega] points on the truncated polytope surface.

    Degree 0 yields the polytope vertices. Degree d >= 1 additionally yields,
    for every face, that face's contour scaled in-plane towards its centroid
    by each new dyadic factor k / 2**d for odd k in [1, 2**d) -- i.e. degree 1
    adds the 0.5-scaled contour, degree 2 adds the 0.25- and 0.75-scaled
    contours, degree 3 adds 0.125/0.375/0.625/0.875, and so on. Results from
    all degrees up to and including `sampling_degree` are combined.
    """
    if sampling_degree < 0:
        raise ValueError("sampling_degree must be non-negative")

    points = [v for v in vertices]

    for d in range(1, sampling_degree + 1):
        scales = [k / (2 ** d) for k in range(1, 2 ** d, 2)]
        for scale in scales:
            for face_pts in faces:
                center = np.mean(face_pts, axis=0)
                shrunk = center + scale * (face_pts - center)
                points.extend(shrunk)

    points = np.array(points, dtype=float)
    rounded = np.round(points, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _planar_polygon_area(pts):
    """Area of a planar (possibly non-convex-safe) polygon given as 3D points."""
    n = pts.shape[0]
    total = np.zeros(3, dtype=float)
    for i in range(n):
        total += np.cross(pts[i], pts[(i + 1) % n])
    return 0.5 * np.linalg.norm(total)


def _split_polygon_by_line(face_pts, point, direction, normal):
    """Split a convex, planar, ordered polygon by the infinite line through
    `point` with in-plane `direction`. Returns (p1, p2, area_a, area_b) for
    the two crossing points and the areas of the two resulting halves, or
    None if the line does not cross exactly two edges."""
    n = face_pts.shape[0]
    values = np.array([np.dot(normal, np.cross(direction, p - point)) for p in face_pts])
    if np.any(np.abs(values) < 1e-9):
        return None

    signs = np.sign(values)
    crossings = []
    for i in range(n):
        j = (i + 1) % n
        if signs[i] != signs[j]:
            t = values[i] / (values[i] - values[j])
            crossings.append((i, face_pts[i] + t * (face_pts[j] - face_pts[i])))

    if len(crossings) != 2:
        return None

    (i1, p1), (i2, p2) = crossings
    if i1 > i2:
        i1, p1, i2, p2 = i2, p2, i1, p1

    poly_a = np.vstack([p1[np.newaxis, :], face_pts[i1 + 1 : i2 + 1], p2[np.newaxis, :]])
    poly_b = np.vstack([p2[np.newaxis, :], face_pts[i2 + 1 :], face_pts[: i1 + 1], p1[np.newaxis, :]])
    return p1, p2, _planar_polygon_area(poly_a), _planar_polygon_area(poly_b)


def _bisecting_line_for_face(face_pts, normal, angle_samples=180, origin_bias=0.25):
    """Find a line through the face centroid that splits its area
    approximately evenly, biased towards lines radial from the world origin.

    Returns (p1, p2) -- the line's two intersection points with the face
    boundary -- chosen by a grid search over in-plane angles that trades off
    area balance against alignment with the ray from (0, 0, 0) through the
    face centroid.
    """
    n = face_pts.shape[0]
    centroid = np.mean(face_pts, axis=0)
    total_area = max(_planar_polygon_area(face_pts), 1e-12)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    radial_in_plane = centroid - np.dot(centroid, normal) * normal
    radial_norm = np.linalg.norm(radial_in_plane)
    theta_radial = np.arctan2(np.dot(radial_in_plane, v), np.dot(radial_in_plane, u)) if radial_norm > 1e-9 else 0.0

    best_cost = None
    best_pair = None
    for i in range(angle_samples):
        theta = np.pi * i / angle_samples
        direction = np.cos(theta) * u + np.sin(theta) * v
        result = _split_polygon_by_line(face_pts, centroid, direction, normal)
        if result is None:
            continue

        p1, p2, area_a, area_b = result
        area_imbalance = abs(area_a - area_b) / total_area
        angular_dist = abs(((theta - theta_radial + np.pi / 2) % np.pi) - np.pi / 2) / (np.pi / 2)
        cost = area_imbalance + origin_bias * angular_dist
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pair = (p1, p2)

    if best_pair is None:
        # Degenerate fallback: bisect via opposite-edge midpoints.
        k = n // 2
        best_pair = (0.5 * (face_pts[0] + face_pts[1]), 0.5 * (face_pts[k % n] + face_pts[(k + 1) % n]))

    return best_pair


def _face_normal(face_pts):
    """Return a unit normal for a planar, ordered polygon `face_pts`."""
    v1 = face_pts[1] - face_pts[0]
    for i in range(2, face_pts.shape[0]):
        n = np.cross(v1, face_pts[i] - face_pts[0])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            return n / norm
    raise ValueError("Could not compute a face normal; face points are collinear")


def _face_basis(normal):
    """Return an orthonormal in-plane (u, v) basis for a plane with `normal`."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _ray_polygon_boundary_distance(face_pts, point, direction, normal):
    """Distance from `point` (assumed interior to convex polygon `face_pts`)
    to the first edge hit by the ray from `point` along in-plane `direction`."""
    u, v = _face_basis(normal)

    def to2d(p):
        return np.array([np.dot(p, u), np.dot(p, v)])

    p2 = to2d(point)
    d2 = np.array([np.dot(direction, u), np.dot(direction, v)])

    n = face_pts.shape[0]
    best_t = None
    for i in range(n):
        j = (i + 1) % n
        a2 = to2d(face_pts[i])
        edge2 = to2d(face_pts[j]) - a2

        A = np.array([[d2[0], -edge2[0]], [d2[1], -edge2[1]]])
        det = np.linalg.det(A)
        if abs(det) < 1e-12:
            continue

        t, s = np.linalg.solve(A, a2 - p2)
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9:
            if best_t is None or t < best_t:
                best_t = t

    if best_t is None:
        raise ValueError("Ray from point did not intersect the face boundary")
    return best_t


def _perpendicular_bisect_points(point, direction, arm_length_a, arm_length_b, depth, normal, out_points, face_pts):
    """Iteratively branch an H-tree-like cross pattern of sample points.

    At each step, a new line through the current point, perpendicular to the
    current direction, is split at that point into two arms of length
    `arm_length_a` and `arm_length_b` (which need not be equal); the centers
    of those two arms become new sample points. Each new point then branches
    again, perpendicular to the line that produced it, with its own arm
    lengths ray-cast from that point to the face boundary, until `depth` is
    exhausted.
    """
    stack = [(point, direction, arm_length_a, arm_length_b, depth)]
    while stack:
        pt, dirn, len_a, len_b, d = stack.pop()
        if d <= 0:
            continue

        perp = np.cross(normal, dirn)
        perp = perp / np.linalg.norm(perp)

        p1 = pt + (len_a / 2.0) * perp
        p2 = pt - (len_b / 2.0) * perp
        out_points.append(p1)
        out_points.append(p2)

        if d - 1 > 0:
            next_perp = np.cross(normal, perp)
            next_perp = next_perp / np.linalg.norm(next_perp)

            len_a1 = _ray_polygon_boundary_distance(face_pts, p1, next_perp, normal)
            len_b1 = _ray_polygon_boundary_distance(face_pts, p1, -next_perp, normal)
            stack.append((p1, perp, len_a1, len_b1, d - 1))

            len_a2 = _ray_polygon_boundary_distance(face_pts, p2, next_perp, normal)
            len_b2 = _ray_polygon_boundary_distance(face_pts, p2, -next_perp, normal)
            stack.append((p2, perp, len_a2, len_b2, d - 1))


def sample_boundary_velocities_bisected(vertices, faces, sampling_degree=0):
    """
    Sample [vx, vy, omega] points on the truncated polytope surface using a
    branching, H-tree-like cross pattern per face.

    Degree 0 yields only the polytope vertices. Degree 1 additionally yields,
    for every face, the midpoint of a line through the face centroid that
    splits its area approximately evenly, biased towards lines radial from
    the world origin. Degree d >= 2 then branches from that point: a new
    line, perpendicular to its parent and half as long, is bisected at the
    parent point, and the centers of its two halves become new points; each
    of those branches again the same way, perpendicular to its own parent
    line, down to depth d (i.e. 2**d - 1 points per face at degree d).
    """
    if sampling_degree < 0:
        raise ValueError("sampling_degree must be non-negative")

    points = [v for v in vertices]

    for face_pts in faces:
        n = face_pts.shape[0]
        if sampling_degree < 1 or n < 3:
            continue

        normal = _face_normal(face_pts)
        edge_mid_0, edge_mid_k = _bisecting_line_for_face(face_pts, normal)
        center = 0.5 * (edge_mid_0 + edge_mid_k)
        points.append(center)

        if sampling_degree >= 2:
            direction = edge_mid_k - edge_mid_0
            direction = direction / np.linalg.norm(direction)
            perp = np.cross(normal, direction)
            perp = perp / np.linalg.norm(perp)
            arm_length_a = _ray_polygon_boundary_distance(face_pts, center, perp, normal)
            arm_length_b = _ray_polygon_boundary_distance(face_pts, center, -perp, normal)
            _perpendicular_bisect_points(
                center, direction, arm_length_a, arm_length_b, sampling_degree - 1, normal, points, face_pts
            )

    points = np.array(points, dtype=float)
    rounded = np.round(points, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


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

    trajectory_color = "tab:red"
    rectangle_color = "tab:blue"
    trajectory_alpha = 0.25
    final_outline_alpha = 0.9
    intermediate_outline_alpha = 0.12

    for cmd in boundary_velocities:
        vx, vy, omega = cmd
        states = rollout_constant_twist(vx, vy, omega, horizon=horizon, dt=dt)

        ax.plot(states[:, 0], states[:, 1], color=trajectory_color, linewidth=1.2, alpha=trajectory_alpha)

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
            alpha = intermediate_outline_alpha if (not show_final_only and j < len(idxs) - 1) else final_outline_alpha
            poly = Polygon(corners, closed=True, fill=False, edgecolor=rectangle_color, linewidth=0.9, alpha=alpha)
            ax.add_patch(poly)

    start_corners = _rectangle_corners(
        x=0.0,
        y=0.0,
        theta=0.0,
        half_length=model.wb_hlength,
        half_width=model.wb_hwidth,
    )
    ax.add_patch(Polygon(start_corners, closed=True, fill=False, edgecolor="black", linewidth=1.4))

    ax.set_title("Boundary-Velocity Rollouts with Oriented Chassis Footprint (Limited)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_truncated_polytope(vertices, faces, vx_range, vy_range, omega_range, extra_points=None):
    """Plot the truncated 3D kinematic polytope's boundary patches and vertices.

    `extra_points`, if given, are additional sampled points (e.g. from
    --sampling-degree) drawn in gray with alpha=0.5.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:cyan",
        "tab:olive",
    ]

    for i, face_pts in enumerate(faces):
        patch = Poly3DCollection(
            [face_pts],
            alpha=0.35,
            facecolor=colors[i % len(colors)],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_collection3d(patch)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=14, c="black", alpha=0.8)

    if extra_points is not None and len(extra_points) > 0:
        extra_points = np.asarray(extra_points, dtype=float)
        ax.scatter(
            extra_points[:, 0], extra_points[:, 1], extra_points[:, 2], s=10, c="gray", alpha=0.7
        )

    ax.set_title("Velocity-Limited Kinematic Polytope")
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.set_zlabel("omega [rad/s]")

    pad_x = 0.1 * max(vx_range[1] - vx_range[0], 1e-6)
    pad_y = 0.1 * max(vy_range[1] - vy_range[0], 1e-6)
    pad_w = 0.1 * max(omega_range[1] - omega_range[0], 1e-6)
    ax.set_xlim(vx_range[0] - pad_x, vx_range[1] + pad_x)
    ax.set_ylim(vy_range[0] - pad_y, vy_range[1] + pad_y)
    ax.set_zlim(omega_range[0] - pad_w, omega_range[1] + pad_w)

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample [vx, vy, omega] points on the surface of the velocity-limited "
            "kinematic polytope, roll out constant-twist trajectories, and plot oriented "
            "chassis rectangles alongside the truncated 3D polytope."
        )
    )
    parser.add_argument("--max-wheel-velocity", type=float, default=10.0, help="Wheel-speed limit [rad/s].")
    parser.add_argument(
        "--sampling-degree",
        type=int,
        default=0,
        help=(
            "Surface sampling degree. Degree 0 samples only the polytope vertices. "
            "Degree d also samples each face's contour shrunk (in-plane, towards its "
            "centroid) by every new dyadic factor k/2**d for odd k, i.e. degree 1 adds "
            "the 0.5-scaled contour, degree 2 adds 0.25/0.75, degree 3 adds "
            "0.125/0.375/0.625/0.875, and so on."
        ),
    )
    parser.add_argument(
        "--sampling-method",
        choices=["shrink", "bisect"],
        default="shrink",
        help=(
            "Face sampling method for --sampling-degree > 0: 'shrink' scales each "
            "face's contour towards its centroid; 'bisect' recursively bisects each "
            "face and samples the midpoint of each bisecting line."
        ),
    )
    parser.add_argument("--horizon", type=float, default=3.0, help="Rollout horizon [s].")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration step [s].")
    parser.add_argument(
        "--rectangle-stride",
        type=int,
        default=40,
        help="Draw a chassis rectangle every N trajectory samples.",
    )
    parser.add_argument(
        "--show-all-rectangles",
        action="store_true",
        help="Draw chassis rectangles throughout each rollout instead of final-only view.",
    )
    parser.add_argument("--vx-min", type=float, default=0.5, help="Minimum body vx limit [m/s].")
    parser.add_argument("--vx-max", type=float, default=1.0, help="Maximum body vx limit [m/s].")
    parser.add_argument("--vy-min", type=float, default=-0.1, help="Minimum body vy limit [m/s].")
    parser.add_argument("--vy-max", type=float, default=0.1, help="Maximum body vy limit [m/s].")
    parser.add_argument("--omega-min", type=float, default=0.5, help="Minimum body omega limit [rad/s].")
    parser.add_argument("--omega-max", type=float, default=2.0, help="Maximum body omega limit [rad/s].")
    args = parser.parse_args()

    if args.max_wheel_velocity <= 0.0:
        raise ValueError("max-wheel-velocity must be positive")
    if args.sampling_degree < 0:
        raise ValueError("sampling-degree must be non-negative")
    if args.horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.rectangle_stride < 1:
        raise ValueError("rectangle-stride must be at least 1")
    if args.vx_min > args.vx_max:
        raise ValueError("vx-min must be less than or equal to vx-max")
    if args.vy_min > args.vy_max:
        raise ValueError("vy-min must be less than or equal to vy-max")
    if args.omega_min > args.omega_max:
        raise ValueError("omega-min must be less than or equal to omega-max")

    vx_range = (args.vx_min, args.vx_max)
    vy_range = (args.vy_min, args.vy_max)
    omega_range = (args.omega_min, args.omega_max)

    model = Mecanum()

    max_vx, max_vy, max_omega = compute_wheel_velocity_limits(model, args.max_wheel_velocity)
    print(f"Kinematic body-velocity limits from max-wheel-velocity={args.max_wheel_velocity:g} rad/s:")
    print(f"  vx    in [{-max_vx:.4f}, {max_vx:.4f}] m/s")
    print(f"  vy    in [{-max_vy:.4f}, {max_vy:.4f}] m/s")
    print(f"  omega in [{-max_omega:.4f}, {max_omega:.4f}] rad/s")

    print("User-requested vx/vy/omega bounds:")
    print(f"  vx    in [{vx_range[0]:.4f}, {vx_range[1]:.4f}] m/s")
    print(f"  vy    in [{vy_range[0]:.4f}, {vy_range[1]:.4f}] m/s")
    print(f"  omega in [{omega_range[0]:.4f}, {omega_range[1]:.4f}] rad/s")

    vertices, faces = _polytope_faces(model, args.max_wheel_velocity, vx_range, vy_range, omega_range)

    (vx_lo, vx_hi), (vy_lo, vy_hi), (omega_lo, omega_hi) = compute_truncated_extents(vertices)
    print("Truncated polytope extents (wheel limit intersected with requested vx/vy/omega bounds):")
    print(f"  vx    in [{vx_lo:.4f}, {vx_hi:.4f}] m/s")
    print(f"  vy    in [{vy_lo:.4f}, {vy_hi:.4f}] m/s")
    print(f"  omega in [{omega_lo:.4f}, {omega_hi:.4f}] rad/s")

    sampling_fn = sample_boundary_velocities if args.sampling_method == "shrink" else sample_boundary_velocities_bisected
    boundary_velocities = sampling_fn(vertices, faces, sampling_degree=args.sampling_degree)
    extra_points = boundary_velocities[vertices.shape[0]:]

    plot_boundary_rollouts(
        model=model,
        boundary_velocities=boundary_velocities,
        horizon=args.horizon,
        dt=args.dt,
        rectangle_stride=args.rectangle_stride,
        show_final_only=(not args.show_all_rectangles),
    )

    plot_truncated_polytope(vertices, faces, vx_range, vy_range, omega_range, extra_points=extra_points)

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
