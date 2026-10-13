import numpy as np

from experiments.analysis_comparison import find_rising_edge, rebase_states, state_at


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