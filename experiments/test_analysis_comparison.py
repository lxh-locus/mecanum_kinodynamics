import numpy as np

from experiments.analysis_comparison import (
    KALMAN_Q,
    KALMAN_R,
    find_rising_edge,
    COPILOT_fuse_imu_odom_velocity,
    fuse_imu_odom_velocity,
    rebase_states,
    state_at,
)


def test_find_rising_edge_selects_first_transition_inside_window():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

    assert find_rising_edge(times, values, 1.5, 4.0) == 3.0


def test_find_rising_edge_does_not_treat_initial_high_as_transition():
    times = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, 1.0, 0.0])

    assert find_rising_edge(times, values, 0.0, 2.0) is None


def test_rebase_states_uses_interpolated_event_pose_as_origin():
    times = np.array([0.0, 2.0])
    states = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    origin = state_at(times, states, 1.0)

    actual = rebase_states(states[1:], origin)

    np.testing.assert_allclose(origin, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(actual, [[1.0, 0.0, 0.0]])


def test_fuse_imu_odom_velocity_preserves_matching_constant_measurements():
    imu_times = np.array([0.0, 1.0, 2.0])
    imu_values = np.tile([0.0, 0.0, 0.0], (3, 1))
    odom_times = np.array([0.0, 2.0])
    odom_twists = np.tile([0.4, -0.2, 0.0], (2, 1))

    fused = COPILOT_fuse_imu_odom_velocity(imu_times, imu_values, odom_times, odom_twists)

    np.testing.assert_allclose(fused, np.tile([0.4, -0.2, 0.0], (3, 1)), atol=1e-12)


def test_fuse_imu_odom_velocity_accounts_for_rotating_body_frame():
    # Steady circular motion: a perfect accelerometer reads the centripetal term omega x v.
    times = np.arange(0.0, 2.0, 0.01)
    imu_values = np.tile([0.0, 1.0, 1.0], (len(times), 1))
    odom_twists = np.tile([1.0, 0.0, 1.0], (len(times), 1))

    fused = COPILOT_fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    np.testing.assert_allclose(fused[-1], [1.0, 0.0, 1.0], atol=1e-9)


def test_fuse_imu_odom_velocity_weights_both_omega_sources_by_variance():
    times = np.array([0.0, 0.5, 1.0])
    imu_values = np.tile([0.0, 0.0, 1.2], (3, 1))
    odom_twists = np.tile([0.0, 0.0, 1.0], (3, 1))

    fused = COPILOT_fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    assert 1.0 < fused[-1, 2] < 1.2


def test_fuse_imu_odom_velocity_predicts_from_linear_acceleration():
    times = np.array([0.0, 1.0])
    imu_values = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    odom_twists = np.zeros((2, 3))

    fused = COPILOT_fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    assert fused[1, 0] > fused[0, 0]


def test_active_fuse_imu_odom_velocity_initializes_from_first_odom_measurement():
    imu_times = np.array([0.0])
    imu_values = np.zeros((1, 3))
    odom_twists = np.array([[0.6, -0.35, 0.2]])

    fused = fuse_imu_odom_velocity(imu_times, imu_values, imu_times, odom_twists)

    np.testing.assert_allclose(fused[0], odom_twists[0], atol=1e-7)


def test_active_fuse_imu_odom_velocity_accepts_braking_mask_without_changing_estimate():
    times = np.array([0.0, 0.01, 0.02])
    imu_values = np.zeros((3, 3))
    odom_twists = np.tile([0.3, -0.1, 0.0], (3, 1))

    without_mask = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)
    with_mask = fuse_imu_odom_velocity(times, imu_values, times, odom_twists, [False, True, True])

    np.testing.assert_allclose(with_mask, without_mask)


def test_active_fuse_imu_odom_velocity_uses_first_measurement_once():
    times = np.array([0.0, 0.01])
    imu_values = np.zeros((2, 3))
    odom_twists = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    process_variance = KALMAN_Q[0, 0] * 0.01
    measurement_variance = KALMAN_R[0, 0]
    expected_gain = (measurement_variance + process_variance) / (
        2.0 * measurement_variance + process_variance
    )
    np.testing.assert_allclose(fused[1, 0], expected_gain, atol=1e-6)


def test_active_fuse_imu_odom_velocity_uses_acceleration_input():
    times = np.array([0.0, 0.01])
    imu_values = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    odom_twists = np.zeros((2, 3))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    assert fused[1, 0] > fused[0, 0]


def test_active_fuse_imu_odom_velocity_preserves_steady_circular_motion():
    times = np.arange(0.0, 2.0, 0.01)
    imu_values = np.tile([0.0, 1.0, 1.0], (len(times), 1))
    odom_twists = np.tile([1.0, 0.0, 1.0], (len(times), 1))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    np.testing.assert_allclose(fused[-1], [1.0, 0.0, 1.0], atol=1e-7)


def test_active_fuse_imu_odom_velocity_handles_lateral_velocity_during_turn():
    times = np.arange(0.0, 2.0, 0.01)
    vx, vy, omega = 0.4, -0.2, 0.3
    imu_values = np.tile([-omega * vy, omega * vx, omega], (len(times), 1))
    odom_twists = np.tile([vx, vy, omega], (len(times), 1))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    np.testing.assert_allclose(fused[-1], [vx, vy, omega], atol=1e-7)