"""
velocity_polytope.py

Self-contained kinematic velocity-polytope construction and boundary sampling.
This strips the pieces of experiments/kinematic_boundary_rollout_limited.py and
experiments/sampling_methods.py that fieldset_generator_redux1/
generate_rollouts_2.py relies on, so redux2 scripts need no imports from the
experiments package for boundary-velocity sampling.

The polytope functions take a parameter bundle (e.g.
``mecanum_fixed_axis.MecanumPhysicsParams``) and only read its ``wb_hlength``,
``wb_hwidth``, and ``wheel_radius`` attributes.

Provided here:
- ``compute_wheel_velocity_limits``: body-velocity axis limits from a wheel
  speed limit.
- ``polytope_faces``: vertices and ordered face polygons of the wheel-speed
  polytope truncated by vx/vy/omega box limits.
- ``compute_truncated_extents``: per-axis extents of the truncated polytope.
- ``sample_boundary_velocities``: vertex/edge/shrunk-contour face sampling.
"""
from itertools import combinations

import numpy as np


def _build_inequalities(params, max_wheel_velocity, vx_range, vy_range, omega_range):
    """Build A, b for the kinematic polytope A @ [vx, vy, omega] <= b,
    truncated by box limits vx_range/vy_range/omega_range."""
    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius
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
        faces.append(pts[np.argsort(angles)])

    return faces


def polytope_faces(params, max_wheel_velocity, vx_range, vy_range, omega_range):
    """Compute vertices and ordered face polygons for the truncated polytope.

    Args:
        params: Parameter bundle exposing ``wb_hlength``, ``wb_hwidth``, and
            ``wheel_radius``.
        max_wheel_velocity: Wheel-speed limit in rad/s.
        vx_range: ``(min, max)`` body vx bounds in m/s.
        vy_range: ``(min, max)`` body vy bounds in m/s.
        omega_range: ``(min, max)`` body omega bounds in rad/s.
    Returns:
        ``(vertices, faces)`` where ``vertices`` is an ``(N, 3)`` array and
        ``faces`` is a list of ordered ``(K, 3)`` polygon arrays.
    Raises:
        ValueError: If no vertices or boundary faces exist for the parameters.
    """
    A, b = _build_inequalities(params, max_wheel_velocity, vx_range, vy_range, omega_range)
    vertices = _compute_polytope_vertices(A, b)
    if vertices.shape[0] == 0:
        raise ValueError("Could not compute polytope vertices for current parameters")

    faces = _build_face_polygons(vertices, A, b)
    if len(faces) == 0:
        raise ValueError("Could not compute polytope boundary faces")
    return vertices, faces


def compute_wheel_velocity_limits(params, max_wheel_velocity):
    """Return body-velocity axis limits from the wheel-speed limit."""
    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius
    max_vx = radius * max_wheel_velocity
    max_vy = radius * max_wheel_velocity
    max_omega = radius * max_wheel_velocity / l_plus_w
    return max_vx, max_vy, max_omega


def compute_truncated_extents(vertices):
    """Return per-axis (min, max) extents of the truncated polytope."""
    return tuple(zip(vertices.min(axis=0), vertices.max(axis=0)))


def _edge_bisection_points(p_start, p_end, sampling_degree):
    """Return dyadic bisection points along a segment.

    Args:
        p_start: Segment start point.
        p_end: Segment end point.
        sampling_degree: Number of recursive dyadic levels.
    Returns:
        A list of points strictly inside the segment.
    """
    points = []
    for d in range(1, sampling_degree + 1):
        for k in range(1, 2 ** d, 2):
            frac = k / (2 ** d)
            points.append(p_start + frac * (p_end - p_start))
    return points


def sample_boundary_velocities(vertices, faces, sampling_degree=0):
    """Sample vertices, edge points, and shrunk face contours.

    Args:
        vertices: ``(N, 3)`` polytope vertices.
        faces: Ordered polygon arrays describing the boundary faces.
        sampling_degree: Number of dyadic edge and contour levels.
    Returns:
        An ``(M, 3)`` array of deduplicated surface samples.
    """
    if sampling_degree < 0:
        raise ValueError("sampling_degree must be non-negative")

    points = [v for v in vertices]
    if sampling_degree >= 1:
        for face_pts in faces:
            n = face_pts.shape[0]
            for i in range(n):
                j = (i + 1) % n
                points.extend(_edge_bisection_points(face_pts[i], face_pts[j], sampling_degree))

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


__all__ = [
    "polytope_faces",
    "compute_wheel_velocity_limits",
    "compute_truncated_extents",
    "sample_boundary_velocities",
]
