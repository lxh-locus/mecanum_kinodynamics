import numpy as np
import pytest

from experiments.analysis_comparison import (
    STRATEGY_COLORS,
    build_strategy_rollouts,
    comparison_figure_title,
    load_recorded_rollouts,
    main,
    parse_args,
    raw,
    window_rollout,
)


def test_comparison_defaults_to_approx_and_ekf_without_energy():
    args = parse_args(["recording.csv"])

    assert args.strategies == ["sliding-coulomb-approx"]
    assert args.initial_source == "ekf"
    assert not args.energy


def test_comparison_title_includes_selected_per_wheel_deceleration():
    args = parse_args(["recording.csv"])

    title = comparison_figure_title(args)

    assert title == "recording.csv\nper-wheel deceleration (approx): 0.225 m/s^2"


def test_window_rollout_interpolates_exact_start_and_stop_and_rebases_heading():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    twists = np.array([[1.0, 0.0, 0.4], [2.0, 0.0, 0.4], [3.0, 0.0, 0.4], [4.0, 0.0, 0.4]])

    _, _, selected_times, selected_twists, states = window_rollout(
        "odom", "blue", times, twists, 1.5, 2.5
    )

    np.testing.assert_allclose(selected_times, [1.5, 2.0, 2.5])
    np.testing.assert_allclose(selected_twists[0], [2.5, 0.0, 0.4])
    np.testing.assert_allclose(states[0], np.zeros(3))
    np.testing.assert_allclose(states[:, 2], [0.0, 0.2, 0.4])


@pytest.mark.parametrize("strategy", STRATEGY_COLORS)
def test_every_strategy_shares_initial_conditions_and_stays_in_window(strategy, capsys):
    args = parse_args(["recording.csv", "--strategies", strategy, "--dt", "0.01"])
    initial = np.array([0.8, -0.2, 0.3])

    rollouts = build_strategy_rollouts(initial, 12.3, 12.337, args)

    assert len(rollouts) == 1
    label, _, times, velocities, states = rollouts[0]
    assert label == strategy
    np.testing.assert_allclose(velocities[0], initial)
    np.testing.assert_allclose(states[0], np.zeros(3))
    np.testing.assert_allclose(times[[0, -1]], [12.3, 12.337])
    assert np.all(np.diff(times) > 0.0)
    assert len(times) == len(velocities) == len(states)
    output = capsys.readouterr().out
    assert f"{strategy}: duration=" in output
    assert "stopped=" in output
    if strategy.startswith("sliding-coulomb-"):
        assert "per-wheel deceleration=" in output
        assert "max body x deceleration=0.900 m/s^2" in output


@pytest.mark.parametrize("strategy, options, expected", [
    (
        "independent-axis",
        ["--brake-deceleration-x", "1.1", "--brake-deceleration-y", "1.2",
         "--brake-deceleration-yaw", "2.3"],
        "brake deceleration: x=1.100 m/s^2, y=1.200 m/s^2, yaw=2.300 rad/s^2",
    ),
    (
        "discrete-empirical",
        ["--cardinal-body-deceleration", "1.2", "--diagonal-body-deceleration", "0.8",
         "--diagonal-angle-half-width-degrees", "7"],
        "body deceleration: cardinal=1.200 m/s^2, diagonal=0.800 m/s^2; "
        "diagonal angle half-width=7.000 deg",
    ),
    (
        "sliding-coulomb-approx",
        ["--max-body-x-deceleration", "1.2", "--diagonal-angle-half-width-degrees", "7"],
        "per-wheel deceleration=0.300 m/s^2; max body x deceleration=1.200 m/s^2; "
        "diagonal angle half-width=7.000 deg",
    ),
])
def test_strategy_printout_includes_configured_braking_parameters(strategy, options, expected, capsys):
    args = parse_args(["recording.csv", "--strategies", strategy, *options])

    build_strategy_rollouts(np.array([0.8, -0.2, 0.3]), 0.0, 0.02, args)

    assert expected in capsys.readouterr().out


def test_all_strategies_handle_rest_and_max_time():
    args = parse_args(["recording.csv", "--strategies", "all", "--max-time", "0.023"])
    rollouts = build_strategy_rollouts(np.zeros(3), 1.0, 2.0, args)

    assert {rollout[0] for rollout in rollouts} == set(STRATEGY_COLORS)
    for _, _, times, velocities, states in rollouts:
        assert times[0] == 1.0
        assert times[-1] <= 1.023
        np.testing.assert_allclose(velocities, 0.0)
        np.testing.assert_allclose(states, 0.0)


@pytest.fixture
def recorded_topics(monkeypatch):
    odom_times = np.array([10.0, 11.0, 12.0, 13.0])
    imu_times = np.array([10.1, 11.1, 12.1, 13.1])
    odom_twists = np.tile([1.0, 0.2, 0.3], (4, 1))
    ekf_twists = np.tile([0.8, 0.1, 0.2], (4, 1))
    monkeypatch.setattr(
        raw, "load_topic",
        lambda path, suffixes, stamp: (odom_times, odom_twists)
        if suffixes == raw.ODOM_SUFFIXES else (imu_times, np.zeros((4, 3))),
    )
    monkeypatch.setattr(raw, "load_optional_signal", lambda *args: None)
    monkeypatch.setattr(raw, "fuse_imu_odom_velocity", lambda *args: ekf_twists)


def test_recorded_rollouts_use_same_exact_interval_without_pure_imu(recorded_topics):
    args = parse_args(["recording.csv", "--time-window", "0.5", "2.5", "--no-no-motion-bias"])

    rollouts = load_recorded_rollouts(args)

    assert [rollout[0] for rollout in rollouts] == ["odom velocity integrated", "imu EKF fused"]
    for _, _, times, _, states in rollouts:
        np.testing.assert_allclose(times[[0, -1]], [0.5, 2.5])
        np.testing.assert_allclose(states[0], np.zeros(3))


def test_brake_start_uses_exact_transition(recorded_topics, monkeypatch):
    monkeypatch.setattr(
        raw, "load_optional_signal",
        lambda *args: (np.array([10.0, 11.5, 12.0]), np.array([0.0, 1.0, 0.0])),
    )
    args = parse_args([
        "recording.csv", "--time-window", "0.5", "2.5", "--start-at-brake", "--no-no-motion-bias",
    ])

    rollouts = load_recorded_rollouts(args)

    for _, _, times, _, states in rollouts:
        assert times[0] == 1.5
        np.testing.assert_allclose(states[0], np.zeros(3))


@pytest.mark.parametrize("source, expected", [("ekf", [0.8, 0.1, 0.2]), ("odom", [1.0, 0.2, 0.3])])
def test_main_uses_selected_initial_source_and_only_comparison_plots(
    source, expected, recorded_topics, monkeypatch
):
    import matplotlib.pyplot as plt

    captured = []
    monkeypatch.setattr(raw, "plot_body_velocities", lambda axis, rollouts: captured.extend(rollouts))
    monkeypatch.setattr(raw, "plot_rollout_paths", lambda *args: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    before = set(plt.get_fignums())
    try:
        main([
            "recording.csv", "--time-window", "0.5", "2.5", "--no-no-motion-bias",
            "--initial-source", source,
        ])
        assert len(set(plt.get_fignums()) - before) == 1
        assert len(captured) == 3
        assert captured[-1][0] == "sliding-coulomb-approx"
        np.testing.assert_allclose(captured[-1][3][0], expected)
        assert captured[-1][2][0] == 0.5
    finally:
        for figure_number in set(plt.get_fignums()) - before:
            plt.close(figure_number)