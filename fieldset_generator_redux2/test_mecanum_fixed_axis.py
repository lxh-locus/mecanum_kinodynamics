import numpy as np
import pytest

from mecanum_fixed_axis import (
    MecanumPhysicsParams,
    append_response_delay,
    fixed_axis_braking_rollout,
    individual_wheel_braking_deceleration,
    integrate_response_delay,
    rollout_sliding_deceleration,
    sliding_deceleration_coulomb_fixed_axis,
)


def test_individual_wheel_braking_deceleration_calibrates_body_x_limit():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    np.testing.assert_allclose(wheel_braking, np.sqrt(2.0))


def test_individual_wheel_braking_deceleration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        individual_wheel_braking_deceleration(0.0)

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


def test_sliding_deceleration_is_symmetric_across_cardinal_directions():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    np.testing.assert_allclose(
        sliding_deceleration_coulomb_fixed_axis([-1.0, 0.0, 0.0], wheel_braking),
        [4.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sliding_deceleration_coulomb_fixed_axis([0.0, 1.0, 0.0], wheel_braking),
        [0.0, -4.0, 0.0],
        atol=1e-12,
    )


def test_sliding_deceleration_diagonal_motion_brakes_two_wheels_without_yaw():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    actual = sliding_deceleration_coulomb_fixed_axis([1.0, 1.0, 0.0], wheel_braking)

    # Two rollers free-roll on the diagonal, so only two wheels brake and the
    # symmetric pair produces no net yaw moment.
    np.testing.assert_allclose(actual, [-2.0, -2.0, 0.0], atol=1e-12)


def test_sliding_deceleration_accepts_translational_velocity_only():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    with_yaw_row = sliding_deceleration_coulomb_fixed_axis([1.0, 0.2, 0.0], wheel_braking)
    without_yaw_row = sliding_deceleration_coulomb_fixed_axis([1.0, 0.2], wheel_braking)

    np.testing.assert_allclose(with_yaw_row, without_yaw_row)


def test_sliding_deceleration_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        sliding_deceleration_coulomb_fixed_axis([1.0, 0.0, 0.0], 0.0)

    with pytest.raises(ValueError, match="shape"):
        sliding_deceleration_coulomb_fixed_axis([1.0], 1.0)


def test_rollout_stops_pure_x_motion_at_constant_deceleration_distance():
    wheel_braking = individual_wheel_braking_deceleration(4.0)
    initial_speed = 1.0

    states, velocities, stop_time, stopped = rollout_sliding_deceleration(
        [initial_speed, 0.0, 0.0], wheel_braking, dt=0.001
    )

    assert stopped
    np.testing.assert_allclose(stop_time, initial_speed / 4.0, atol=5e-3)
    np.testing.assert_allclose(velocities[-1], np.zeros(3), atol=1e-12)
    # v^2 / (2 a) stopping distance for constant deceleration.
    np.testing.assert_allclose(states[-1, 0], initial_speed**2 / 8.0, atol=5e-3)
    np.testing.assert_allclose(states[-1, 1:], 0.0, atol=1e-12)


def test_rollout_starts_stopped_for_stationary_initial_velocity():
    states, velocities, stop_time, stopped = rollout_sliding_deceleration(
        [0.0, 0.0, 0.0], 1.0
    )

    assert stopped
    assert stop_time == 0.0
    assert states.shape == (1, 3)
    assert velocities.shape == (1, 3)


def test_rollout_reports_not_stopped_when_max_time_is_too_short():
    wheel_braking = individual_wheel_braking_deceleration(4.0)

    _, velocities, stop_time, stopped = rollout_sliding_deceleration(
        [2.0, 0.0, 0.0], wheel_braking, dt=0.005, max_time=0.1
    )

    assert not stopped
    assert stop_time == 0.1
    assert np.linalg.norm(velocities[-1][:2]) > 0.0


def test_rollout_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        rollout_sliding_deceleration([1.0, 0.0], 1.0)

    with pytest.raises(ValueError, match="dt"):
        rollout_sliding_deceleration([1.0, 0.0, 0.0], 1.0, dt=0.0)

    with pytest.raises(ValueError, match="max_time"):
        rollout_sliding_deceleration([1.0, 0.0, 0.0], 1.0, max_time=0.0)


def test_integrate_response_delay_holds_constant_velocity():
    # Binary-exact dt so duration // dt yields exactly 8 steps.
    poses = integrate_response_delay([1.0, 0.5, 0.0], duration=0.5, dt=0.0625)

    assert poses.shape == (9, 3)
    np.testing.assert_allclose(poses[-1], [0.5, 0.25, 0.0], atol=1e-12)


def test_append_response_delay_shifts_braking_poses_by_delay_motion():
    braking_poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    trajectory = append_response_delay([1.0, 0.0, 0.0], braking_poses, delay=0.5, dt=0.0625)

    # 8 delay steps plus the appended braking pose (shared pose deduplicated).
    assert trajectory.shape == (10, 3)
    np.testing.assert_allclose(trajectory[8], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(trajectory[-1], [0.6, 0.0, 0.0], atol=1e-12)


def test_append_response_delay_rotates_braking_poses_into_final_delay_frame():
    braking_poses = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    yaw_rate = np.pi / 2.0

    trajectory = append_response_delay([0.0, 0.0, yaw_rate], braking_poses, delay=1.0, dt=0.01)

    final_yaw = trajectory[-2, 2]
    expected = [np.cos(final_yaw), np.sin(final_yaw), final_yaw]
    np.testing.assert_allclose(trajectory[-1], expected, atol=1e-12)


def test_fixed_axis_braking_rollout_composes_delay_and_braking():
    initial_velocity = [1.0, -0.4, 0.3]
    delay = 0.333
    dt = 0.005

    trajectory, stop_time, stopped, wheel_braking = fixed_axis_braking_rollout(
        initial_velocity, 4.0, delay, dt=dt
    )

    np.testing.assert_allclose(wheel_braking, np.sqrt(2.0))
    braking_poses, _, expected_stop_time, expected_stopped = rollout_sliding_deceleration(
        initial_velocity, wheel_braking, dt=dt
    )
    expected = append_response_delay(initial_velocity, braking_poses, delay, dt)
    np.testing.assert_allclose(trajectory, expected)
    assert stop_time == expected_stop_time
    assert stopped == expected_stopped
