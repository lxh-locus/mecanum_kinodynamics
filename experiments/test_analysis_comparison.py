import numpy as np

from experiments.analysis_comparison import (
    find_rising_edge,
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

    fused = fuse_imu_odom_velocity(imu_times, imu_values, odom_times, odom_twists)

    np.testing.assert_allclose(fused, np.tile([0.4, -0.2, 0.0], (3, 1)), atol=1e-12)


def test_fuse_imu_odom_velocity_accounts_for_rotating_body_frame():
    # Steady circular motion: a perfect accelerometer reads the centripetal term omega x v.
    times = np.arange(0.0, 2.0, 0.01)
    imu_values = np.tile([0.0, 1.0, 1.0], (len(times), 1))
    odom_twists = np.tile([1.0, 0.0, 1.0], (len(times), 1))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    np.testing.assert_allclose(fused[-1], [1.0, 0.0, 1.0], atol=1e-9)


def test_fuse_imu_odom_velocity_weights_both_omega_sources_by_variance():
    times = np.array([0.0, 0.5, 1.0])
    imu_values = np.tile([0.0, 0.0, 1.2], (3, 1))
    odom_twists = np.tile([0.0, 0.0, 1.0], (3, 1))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    assert 1.0 < fused[-1, 2] < 1.2


def test_fuse_imu_odom_velocity_predicts_from_linear_acceleration():
    times = np.array([0.0, 1.0])
    imu_values = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    odom_twists = np.zeros((2, 3))

    fused = fuse_imu_odom_velocity(times, imu_values, times, odom_twists)

    assert fused[1, 0] > fused[0, 0]