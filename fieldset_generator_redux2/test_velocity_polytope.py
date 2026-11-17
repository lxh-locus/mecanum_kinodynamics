import numpy as np
import pytest

from mecanum_fixed_axis import MecanumPhysicsParams
from velocity_polytope import (
    _build_inequalities,
    compute_truncated_extents,
    compute_wheel_velocity_limits,
    polytope_faces,
    sample_boundary_velocities,
)

PARAMS = MecanumPhysicsParams()
MAX_WHEEL_VELOCITY = 26.25
VX_RANGE = (-1.5, 2.0)
VY_RANGE = (-0.3, 0.3)
OMEGA_RANGE = (-0.9, 0.9)


def _polytope():
    return polytope_faces(PARAMS, MAX_WHEEL_VELOCITY, VX_RANGE, VY_RANGE, OMEGA_RANGE)


def test_compute_wheel_velocity_limits_scales_by_radius_and_geometry():
    max_vx, max_vy, max_omega = compute_wheel_velocity_limits(PARAMS, MAX_WHEEL_VELOCITY)

    expected_translation = PARAMS.wheel_radius * MAX_WHEEL_VELOCITY
    np.testing.assert_allclose(max_vx, expected_translation)
    np.testing.assert_allclose(max_vy, expected_translation)
    np.testing.assert_allclose(
        max_omega, expected_translation / (PARAMS.wb_hlength + PARAMS.wb_hwidth)
    )


def test_polytope_faces_vertices_satisfy_all_inequalities():
    vertices, faces = _polytope()

    assert vertices.shape[0] > 0
    assert len(faces) > 0
    A, b = _build_inequalities(PARAMS, MAX_WHEEL_VELOCITY, VX_RANGE, VY_RANGE, OMEGA_RANGE)
    assert np.all(vertices @ A.T <= b + 1e-8)
    # Every vertex lies on at least three boundary planes.
    active = np.isclose(vertices @ A.T, b, atol=1e-8)
    assert np.all(active.sum(axis=1) >= 3)


def test_polytope_faces_face_points_lie_on_polytope_boundary():
    vertices, faces = _polytope()

    A, b = _build_inequalities(PARAMS, MAX_WHEEL_VELOCITY, VX_RANGE, VY_RANGE, OMEGA_RANGE)
    for face_pts in faces:
        assert face_pts.shape[0] >= 3
        active = np.isclose(face_pts @ A.T, b, atol=1e-8)
        # All points of one face share at least one common boundary plane.
        assert np.any(np.all(active, axis=0))


def test_polytope_faces_raises_for_infeasible_box():
    with pytest.raises(ValueError, match="vertices"):
        polytope_faces(PARAMS, MAX_WHEEL_VELOCITY, (10.0, 11.0), VY_RANGE, OMEGA_RANGE)


def test_compute_truncated_extents_matches_vertex_min_max():
    vertices, _ = _polytope()

    extents = compute_truncated_extents(vertices)

    for axis, (low, high) in enumerate(extents):
        assert low == vertices[:, axis].min()
        assert high == vertices[:, axis].max()
        assert low <= high


def test_sample_boundary_velocities_degree_zero_returns_vertices():
    vertices, faces = _polytope()

    samples = sample_boundary_velocities(vertices, faces, sampling_degree=0)

    np.testing.assert_allclose(np.sort(samples, axis=0), np.sort(vertices, axis=0))


def test_sample_boundary_velocities_higher_degree_adds_unique_boundary_points():
    vertices, faces = _polytope()

    degree_one = sample_boundary_velocities(vertices, faces, sampling_degree=1)
    degree_two = sample_boundary_velocities(vertices, faces, sampling_degree=2)

    assert degree_one.shape[0] > vertices.shape[0]
    assert degree_two.shape[0] > degree_one.shape[0]
    rounded = np.round(degree_two, decimals=8)
    assert np.unique(rounded, axis=0).shape[0] == degree_two.shape[0]
    A, b = _build_inequalities(PARAMS, MAX_WHEEL_VELOCITY, VX_RANGE, VY_RANGE, OMEGA_RANGE)
    assert np.all(degree_two @ A.T <= b + 1e-8)
    # Every sample lies on at least one boundary plane.
    active = np.isclose(degree_two @ A.T, b, atol=1e-8)
    assert np.all(active.any(axis=1))


def test_sample_boundary_velocities_rejects_negative_degree():
    vertices, faces = _polytope()

    with pytest.raises(ValueError, match="non-negative"):
        sample_boundary_velocities(vertices, faces, sampling_degree=-1)
