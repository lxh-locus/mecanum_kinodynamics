import numpy as np
import pytest

from mecanum_common import RobotFootprint
from mecanum_physics import (
    MecanumPhysicsParams,
    exact_dynamics_coeffs,
    forward_dynamics_exact,
    forward_dynamics_linear,
    forward_dynamics_matrix_linear,
    forward_kinematics,
    inverse_dynamics_exact,
    inverse_dynamics_linear,
    inverse_kinematics,
    params_from_model,
    relax_wheel_velocity_to_constraint,
    wheel_constraint_violation,
)


def test_robot_footprint_world_corners_use_nominal_body_dimensions():
    footprint = RobotFootprint()

    corners = footprint.world_corners(x=1.0, y=2.0, theta=np.pi / 2.0)

    np.testing.assert_allclose(
        corners,
        [
            [1.0 - footprint.width / 2.0, 2.0 + footprint.length / 2.0],
            [1.0 + footprint.width / 2.0, 2.0 + footprint.length / 2.0],
            [1.0 + footprint.width / 2.0, 2.0 - footprint.length / 2.0],
            [1.0 - footprint.width / 2.0, 2.0 - footprint.length / 2.0],
        ],
        atol=1e-12,
    )


def test_params_from_model_copies_physical_attributes():
    class Model:
        wb_hwidth = 1
        wb_hlength = 2
        wheel_radius = 3
        body_mass = 4
        wheel_spin_inertia = 5
        body_yaw_inertia = 6
        max_wheel_velocity = 7

    params = params_from_model(Model())

    assert params == MecanumPhysicsParams(
        wb_hwidth=1.0,
        wb_hlength=2.0,
        wheel_radius=3.0,
        body_mass=4.0,
        wheel_spin_inertia=5.0,
        body_yaw_inertia=6.0,
        max_wheel_velocity=7.0,
    )


def test_params_from_model_defaults_max_wheel_velocity_when_absent():
    class Model:
        wb_hwidth = 1
        wb_hlength = 2
        wheel_radius = 3
        body_mass = 4
        wheel_spin_inertia = 5
        body_yaw_inertia = 6

    params = params_from_model(Model())

    assert params.max_wheel_velocity == MecanumPhysicsParams().max_wheel_velocity


def test_wheel_velocity_constraint_projection_removes_violation():
    wheel_velocity = np.array([1.0, 2.0, 4.0, 8.0])

    relaxed = relax_wheel_velocity_to_constraint(wheel_velocity)

    np.testing.assert_allclose(wheel_constraint_violation(relaxed), 0.0)
    np.testing.assert_allclose(relaxed, [3.25, 4.25, 1.75, 5.75])


def test_forward_kinematics_rejects_incompatible_wheel_speeds_in_strict_mode():
    with pytest.raises(ValueError, match="constraint broken"):
        forward_kinematics([1.0, 2.0, 3.0, 5.0], strict=True)


def test_forward_kinematics_relaxes_incompatible_wheel_speeds_when_not_strict():
    wheel_velocity = np.array([1.0, 2.0, 3.0, 5.0])

    actual = forward_kinematics(wheel_velocity, strict=False)
    expected = forward_kinematics(relax_wheel_velocity_to_constraint(wheel_velocity))

    np.testing.assert_allclose(actual, expected)


def test_kinematics_round_trip_for_compatible_wheel_speeds():
    body_velocity = np.array([1.2, -0.4, 0.7])
    wheel_velocity = inverse_kinematics(body_velocity)

    actual = forward_kinematics(wheel_velocity)


    np.testing.assert_allclose(wheel_constraint_violation(wheel_velocity), 0.0, atol=1e-12)
    np.testing.assert_allclose(actual, body_velocity)


def test_inverse_kinematics_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        inverse_kinematics([1.0, 2.0])


def test_linear_dynamics_matches_matrix_product():
    torque = np.array([1.0, -2.0, 3.0, -4.0])

    actual = forward_dynamics_linear(torque)
    expected = forward_dynamics_matrix_linear() @ torque

    np.testing.assert_allclose(actual, expected)


def test_linear_inverse_dynamics_round_trip_to_body_acceleration():
    body_acceleration = np.array([0.8, -0.35, 1.1])
    torque = inverse_dynamics_linear(body_acceleration)

    actual = forward_dynamics_linear(torque)

    np.testing.assert_allclose(actual, body_acceleration, atol=1e-12)


def test_exact_dynamics_coeffs_are_finite_scalars():
    coeffs = exact_dynamics_coeffs()

    assert len(coeffs) == 3
    assert np.isfinite(coeffs).all()


def test_exact_inverse_dynamics_round_trip_to_body_acceleration():
    body_acceleration = np.array([0.4, -0.2, 0.6])
    wheel_velocity = inverse_kinematics([1.0, 0.5, -0.25])
    torque = inverse_dynamics_exact(body_acceleration, wheel_velocity)

    actual = forward_dynamics_exact(torque, wheel_velocity)

    np.testing.assert_allclose(actual, body_acceleration, atol=1e-12)


def test_dynamics_functions_reject_wrong_shapes():
    with pytest.raises(ValueError, match="wheel_torque"):
        forward_dynamics_linear([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="body_accel"):
        inverse_dynamics_linear([1.0, 2.0])

    with pytest.raises(ValueError, match="wheel_torque"):
        forward_dynamics_exact([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="body_accel"):
        inverse_dynamics_exact([1.0, 2.0], [1.0, 2.0, 3.0, 4.0])