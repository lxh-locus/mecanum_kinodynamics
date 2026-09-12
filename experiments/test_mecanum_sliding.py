import numpy as np
import pytest

from mecanum_physics import MecanumPhysicsParams
from mecanum_sliding import (
    individual_wheel_braking_deceleration,
    sliding_deceleration_approx_model,
    sliding_deceleration_coulomb_fixed_axis,
    sliding_deceleration_coulomb_velocity_axis,
    sliding_deceleration_discrete_emperical,
)


def test_individual_wheel_braking_deceleration_calibrates_body_x_limit():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    np.testing.assert_allclose(wheel_braking, np.sqrt(2.0))


def test_individual_wheel_braking_deceleration_calibrates_approx_model_body_x_limit():
    wheel_braking = individual_wheel_braking_deceleration(4.0, model="approx")

    np.testing.assert_allclose(wheel_braking, 2.0)

    actual = sliding_deceleration_approx_model([1.0, 0.0, 0.0], wheel_braking)
    np.testing.assert_allclose(actual, [-4.0, 0.0, 0.0], atol=1e-12)


def test_individual_wheel_braking_deceleration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        individual_wheel_braking_deceleration(0.0)

    with pytest.raises(ValueError, match="model"):
        individual_wheel_braking_deceleration(4.0, model="unknown")

    params = MecanumPhysicsParams(roller_directions=((0.0, 0.0),) * 4)
    with pytest.raises(ValueError, match="nonzero"):
        individual_wheel_braking_deceleration(4.0, params=params)


def test_sliding_deceleration_returns_zero_for_stationary_body():
    actual = sliding_deceleration_coulomb_fixed_axis([0.0, 0.0, 0.0], 1.0)

    np.testing.assert_allclose(actual, np.zeros(3))


def test_sliding_deceleration_calibrates_body_x_braking_without_yaw():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    actual = sliding_deceleration_coulomb_fixed_axis([1.0, 0.0, 0.0], wheel_braking)

    np.testing.assert_allclose(actual, [-4.0, 0.0, 0.0], atol=1e-12)


def test_sliding_deceleration_velocity_axis_calibrates_body_x_braking_without_yaw():
    wheel_braking = 1.0

    actual = sliding_deceleration_coulomb_velocity_axis([1.0, 0.0, 0.0], wheel_braking)

    np.testing.assert_allclose(actual, [-4.0, 0.0, 0.0], atol=1e-12)


def test_sliding_deceleration_velocity_axis_returns_zero_for_stationary_body():
    actual = sliding_deceleration_coulomb_velocity_axis([0.0, 0.0, 0.0], 1.0)

    np.testing.assert_allclose(actual, np.zeros(3))


def test_sliding_deceleration_velocity_axis_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        sliding_deceleration_coulomb_velocity_axis([1.0], 1.0)

    with pytest.raises(ValueError, match="positive"):
        sliding_deceleration_coulomb_velocity_axis([1.0, 0.0], 0.0)

    with pytest.raises(ValueError, match="tolerance"):
        sliding_deceleration_coulomb_velocity_axis([1.0, 0.0], 1.0, tolerance=0.0)


def test_sliding_deceleration_includes_yaw_moment_from_contact_forces():
    actual = sliding_deceleration_coulomb_fixed_axis([0.0, 0.0, 1.0], 1.0)

    assert actual[0] == pytest.approx(0.0, abs=1e-12)
    assert actual[1] == pytest.approx(0.0, abs=1e-12)
    assert actual[2] < 0.0


def test_sliding_deceleration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        sliding_deceleration_coulomb_fixed_axis([1.0], 1.0)

    with pytest.raises(ValueError, match="positive"):
        sliding_deceleration_coulomb_fixed_axis([1.0, 0.0], 0.0)

    with pytest.raises(ValueError, match="tolerance"):
        sliding_deceleration_coulomb_fixed_axis([1.0, 0.0], 1.0, tolerance=0.0)


def test_sliding_deceleration_discrete_emperical_uses_cardinal_value_on_axes():
    actual_x = sliding_deceleration_discrete_emperical([2.0, 0.0], 4.0, 2.0, 5.0)
    actual_y = sliding_deceleration_discrete_emperical([0.0, -3.0], 4.0, 2.0, 5.0)

    np.testing.assert_allclose(actual_x, [-4.0, 0.0, 0.0])
    np.testing.assert_allclose(actual_y, [0.0, 4.0, 0.0])


def test_sliding_deceleration_discrete_emperical_uses_diagonal_value_inside_band():
    actual = sliding_deceleration_discrete_emperical([1.0, 1.0], 4.0, 2.0, 5.0)

    np.testing.assert_allclose(actual, [-np.sqrt(2.0), -np.sqrt(2.0), 0.0])


def test_sliding_deceleration_discrete_emperical_step_changes_at_band_edge():
    inside_angle = np.radians(40.0)
    outside_angle = np.radians(39.0)
    inside_velocity = [np.cos(inside_angle), np.sin(inside_angle)]
    outside_velocity = [np.cos(outside_angle), np.sin(outside_angle)]

    inside = sliding_deceleration_discrete_emperical(inside_velocity, 4.0, 2.0, 5.0)
    outside = sliding_deceleration_discrete_emperical(outside_velocity, 4.0, 2.0, 5.0)

    assert np.linalg.norm(inside[:2]) == pytest.approx(2.0)
    assert np.linalg.norm(outside[:2]) == pytest.approx(4.0)


def test_sliding_deceleration_discrete_emperical_repeats_diagonal_band_by_quadrant():
    angle = np.radians(135.0)

    actual = sliding_deceleration_discrete_emperical([np.cos(angle), np.sin(angle)], 4.0, 2.0, 5.0)

    np.testing.assert_allclose(actual, [np.sqrt(2.0), -np.sqrt(2.0), 0.0])


def test_sliding_deceleration_discrete_emperical_returns_zero_for_zero_translation():
    actual = sliding_deceleration_discrete_emperical([0.0, 0.0, 1.0], 4.0, 2.0, 5.0)

    np.testing.assert_allclose(actual, np.zeros(3))


def test_sliding_deceleration_discrete_emperical_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        sliding_deceleration_discrete_emperical([1.0], 4.0, 2.0, 5.0)

    with pytest.raises(ValueError, match="cardinal"):
        sliding_deceleration_discrete_emperical([1.0, 0.0], 0.0, 2.0, 5.0)

    with pytest.raises(ValueError, match="diagonal"):
        sliding_deceleration_discrete_emperical([1.0, 0.0], 4.0, 0.0, 5.0)

    with pytest.raises(ValueError, match="not exceed"):
        sliding_deceleration_discrete_emperical([1.0, 0.0], 2.0, 4.0, 5.0)

    with pytest.raises(ValueError, match=r"\[0, 45\]"):
        sliding_deceleration_discrete_emperical([1.0, 0.0], 4.0, 2.0, 46.0)

    with pytest.raises(ValueError, match="tolerance"):
        sliding_deceleration_discrete_emperical([1.0, 0.0], 4.0, 2.0, 5.0, tolerance=0.0)
