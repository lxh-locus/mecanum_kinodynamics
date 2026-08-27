"""Sampling methods for kinematic velocity-polytope boundaries."""
from itertools import combinations

import numpy as np


def _sample_points_on_face(face_points, samples, rng):
    """Sample points approximately uniformly on a convex face polygon."""
    center = np.mean(face_points, axis=0)
    sampled = []

    for i in range(face_points.shape[0]):
        p0 = face_points[i]
        p1 = face_points[(i + 1) % face_points.shape[0]]

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


def sample_boundary_velocities_random(vertices, faces, total_samples=8, seed=0):
    """Sample points approximately uniformly on supplied boundary faces."""
    if vertices.shape[0] == 0:
        raise ValueError("Could not compute polytope vertices for current parameters")
    if len(faces) == 0:
        raise ValueError("Could not compute polytope boundary faces")

    rng = np.random.default_rng(seed)
    per_face = max(1, int(np.ceil(total_samples / len(faces))))
    all_pts = []
    for face in faces:
        pts = _sample_points_on_face(face, per_face, rng)
        if pts.shape[0] > 0:
            all_pts.append(pts)

    boundary_pts = np.vstack(all_pts)
    rounded = np.round(boundary_pts, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    boundary_pts = boundary_pts[np.sort(unique_idx)]
    rng.shuffle(boundary_pts)

    if boundary_pts.shape[0] < total_samples:
        return boundary_pts
    return boundary_pts[:total_samples]


def _slice_polygon_vertices(A, b, omega, atol=1e-9):
    """Return CCW-ordered (vx, vy) vertices of a polytope slice."""
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


def sample_boundary_velocities_bisect(A, b, vertices, bisect_tier=0, n_spacing=1):
    """Sample a polytope boundary at evenly spaced omega slices."""
    if bisect_tier < 0:
        raise ValueError("bisect_tier must be non-negative")
    if n_spacing < 1:
        raise ValueError("n_spacing must be at least 1")
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


def _edge_bisection_points(p_start, p_end, sampling_degree):
    """Return dyadic bisection points along a segment."""
    points = []
    for d in range(1, sampling_degree + 1):
        for k in range(1, 2 ** d, 2):
            frac = k / (2 ** d)
            points.append(p_start + frac * (p_end - p_start))
    return points


def sample_boundary_velocities(vertices, faces, sampling_degree=0):
    """Sample vertices, edge points, and shrunk face contours."""
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


def _face_normal(face_pts):
    """Return a unit normal for a planar, ordered polygon."""
    v1 = face_pts[1] - face_pts[0]
    for i in range(2, face_pts.shape[0]):
        n = np.cross(v1, face_pts[i] - face_pts[0])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            return n / norm
    raise ValueError("Could not compute a face normal; face points are collinear")


def _face_basis(normal):
    """Return an orthonormal in-plane basis for a plane."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _project_face_to_2d(face_pts, normal):
    """Project an ordered 3D face into local 2D coordinates."""
    u, v = _face_basis(normal)
    anchor = face_pts[0]
    rel = face_pts - anchor
    poly_2d = np.stack([rel @ u, rel @ v], axis=1)
    return anchor, u, v, poly_2d


def _point_to_2d(point, anchor, u, v):
    rel = point - anchor
    return np.array([np.dot(rel, u), np.dot(rel, v)])


def _points_to_3d(points_2d, anchor, u, v):
    points_2d = np.asarray(points_2d, dtype=float)
    return anchor + np.outer(points_2d[:, 0], u) + np.outer(points_2d[:, 1], v)


def _polygon_area_2d(poly_2d):
    """Return the area of a 2D polygon."""
    n = poly_2d.shape[0]
    total = 0.0
    for i in range(n):
        x1, y1 = poly_2d[i]
        x2, y2 = poly_2d[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def _rotate90(v):
    return np.array([-v[1], v[0]])


def _line_polygon_intersections_2d(poly_2d, point_2d, direction_2d):
    """Return sorted intersections of an infinite line and polygon."""
    n = poly_2d.shape[0]
    hits = []
    for i in range(n):
        j = (i + 1) % n
        a = poly_2d[i]
        edge = poly_2d[j] - a

        matrix = np.array([[direction_2d[0], -edge[0]], [direction_2d[1], -edge[1]]])
        det = np.linalg.det(matrix)
        if abs(det) < 1e-12:
            continue

        t, s = np.linalg.solve(matrix, a - point_2d)
        if -1e-9 <= s <= 1 + 1e-9:
            hits.append((t, a + s * edge, i))

    hits.sort(key=lambda hit: hit[0])
    deduped = []
    for hit in hits:
        if deduped and np.allclose(hit[1], deduped[-1][1], atol=1e-9):
            continue
        deduped.append(hit)
    return deduped


def _split_polygon_by_line_2d(poly_2d, point_2d, direction_2d):
    """Split a convex polygon by an infinite line."""
    hits = _line_polygon_intersections_2d(poly_2d, point_2d, direction_2d)
    if len(hits) != 2:
        return None

    (_, p1, i1), (_, p2, i2) = hits
    if i1 > i2:
        i1, p1, i2, p2 = i2, p2, i1, p1

    poly_a = np.vstack([p1[np.newaxis, :], poly_2d[i1 + 1 : i2 + 1], p2[np.newaxis, :]])
    poly_b = np.vstack([p2[np.newaxis, :], poly_2d[i2 + 1 :], poly_2d[: i1 + 1], p1[np.newaxis, :]])
    return p1, p2, _polygon_area_2d(poly_a), _polygon_area_2d(poly_b)


def _bisecting_line_for_face_2d(poly_2d, origin_2d, angle_samples=180, origin_bias=0.25):
    """Find an approximately area-bisecting line through a polygon centroid."""
    n = poly_2d.shape[0]
    centroid = np.mean(poly_2d, axis=0)
    total_area = max(_polygon_area_2d(poly_2d), 1e-12)

    radial = centroid - origin_2d
    radial_norm = np.linalg.norm(radial)
    theta_radial = np.arctan2(radial[1], radial[0]) if radial_norm > 1e-9 else 0.0

    best_cost = None
    best_pair = None
    for i in range(angle_samples):
        theta = np.pi * i / angle_samples
        direction = np.array([np.cos(theta), np.sin(theta)])
        result = _split_polygon_by_line_2d(poly_2d, centroid, direction)
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
        k = n // 2
        best_pair = (0.5 * (poly_2d[0] + poly_2d[1]), 0.5 * (poly_2d[k % n] + poly_2d[(k + 1) % n]))
    return best_pair


def _ray_polygon_boundary_distance_2d(poly_2d, point_2d, direction_2d):
    """Return the first boundary distance along a ray."""
    for t, _, _ in _line_polygon_intersections_2d(poly_2d, point_2d, direction_2d):
        if t > 1e-9:
            return t
    raise ValueError("Ray from point did not intersect the face boundary")


def _perpendicular_bisect_points_2d(point, direction, arm_length_a, arm_length_b, depth, poly_2d, out_points):
    """Recursively add perpendicular face-interior bisection points."""
    stack = [(point, direction, arm_length_a, arm_length_b, depth)]
    while stack:
        pt, dirn, len_a, len_b, d = stack.pop()
        if d <= 0:
            continue

        perp = _rotate90(dirn)
        children = [pt + (len_a / 2.0) * perp, pt - (len_b / 2.0) * perp]
        out_points.extend(children)

        if d - 1 > 0:
            next_perp = _rotate90(perp)
            for child_pt in children:
                next_len_a = _ray_polygon_boundary_distance_2d(poly_2d, child_pt, next_perp)
                next_len_b = _ray_polygon_boundary_distance_2d(poly_2d, child_pt, -next_perp)
                stack.append((child_pt, perp, next_len_a, next_len_b, d - 1))


def sample_boundary_velocities_bisected(vertices, faces, sampling_degree=0):
    """Sample truncated-polytope faces with recursive interior bisection."""
    if sampling_degree < 0:
        raise ValueError("sampling_degree must be non-negative")

    points = [v for v in vertices]
    for face_pts in faces:
        n = face_pts.shape[0]
        if sampling_degree < 1 or n < 3:
            continue

        for i in range(n):
            j = (i + 1) % n
            points.extend(_edge_bisection_points(face_pts[i], face_pts[j], sampling_degree))

        normal = _face_normal(face_pts)
        anchor, u, v, poly_2d = _project_face_to_2d(face_pts, normal)
        origin_2d = _point_to_2d(np.zeros(3), anchor, u, v)

        edge_mid_0, edge_mid_k = _bisecting_line_for_face_2d(poly_2d, origin_2d)
        center = 0.5 * (edge_mid_0 + edge_mid_k)
        points_2d = [center]

        if sampling_degree >= 2:
            direction = edge_mid_k - edge_mid_0
            direction = direction / np.linalg.norm(direction)
            perp = _rotate90(direction)
            arm_length_a = _ray_polygon_boundary_distance_2d(poly_2d, center, perp)
            arm_length_b = _ray_polygon_boundary_distance_2d(poly_2d, center, -perp)
            _perpendicular_bisect_points_2d(
                center, direction, arm_length_a, arm_length_b, sampling_degree - 1, poly_2d, points_2d
            )

        points.extend(_points_to_3d(points_2d, anchor, u, v))

    points = np.array(points, dtype=float)
    rounded = np.round(points, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_idx)]
