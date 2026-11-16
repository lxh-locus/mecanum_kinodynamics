import numpy as np

from experiments.mecanum_common import RobotFootprint
from experiments.mecanum_physics import MecanumPhysicsParams
from experiments.mecanum_sliding import (
    individual_wheel_braking_deceleration,
    sliding_deceleration_coulomb_fixed_axis,
)
from learning.animate_fixed_axis_xy_sliding import create_animation, simulate_sliding


def test_two_stage_solution_and_wheel_forces_match_fixed_axis_model():
    times, positions, velocities, forces, transition_time, params = simulate_sliding()
    transition_index = np.flatnonzero(times == transition_time)[0]
    wheel_braking = individual_wheel_braking_deceleration(1.0, params)

    np.testing.assert_allclose(transition_time, 0.6)
    np.testing.assert_allclose(times[-1], 1.4)
    np.testing.assert_allclose(velocities[transition_index], [0.4, 0.4, 0.0])
    np.testing.assert_allclose(velocities[transition_index:-1, 0], velocities[transition_index:-1, 1])
    np.testing.assert_allclose(velocities[-1], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[-1], [0.58, 0.40])
    assert np.count_nonzero(np.linalg.norm(forces[0], axis=1)) == 4
    assert np.count_nonzero(np.linalg.norm(forces[transition_index], axis=1)) == 2
    assert np.count_nonzero(forces[-1]) == 0
    for velocity, wheel_force in zip(velocities, forces):
        acceleration = sliding_deceleration_coulomb_fixed_axis(velocity, wheel_braking, params)
        np.testing.assert_allclose(wheel_force.sum(axis=0) / params.body_mass, acceleration[:2], atol=1e-12)


def test_space_key_pauses_and_resumes_animation(monkeypatch):
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent

    figure, animation = create_animation()
    calls = []
    monkeypatch.setattr(animation, "pause", lambda: calls.append("pause"))
    monkeypatch.setattr(animation, "resume", lambda: calls.append("resume"))
    try:
        figure.canvas.draw()
        figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", figure.canvas, key=" "))
        figure.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", figure.canvas, key=" "))
        assert calls == ["pause", "resume"]
    finally:
        plt.close(figure)


def test_animation_uses_default_footprint_and_wheelbase():
    import matplotlib.pyplot as plt

    figure, animation = create_animation()
    try:
        figure.canvas.draw()
        scene = figure.axes[0]
        corners = RobotFootprint().world_corners(0, 0, 0)
        np.testing.assert_allclose(scene.patches[0].get_xy()[:4], corners)
        params = MecanumPhysicsParams()
        wheels = np.array([
            [params.wb_hlength, params.wb_hwidth], [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth], [-params.wb_hlength, -params.wb_hwidth],
        ])
        np.testing.assert_allclose(scene.lines[2].get_xdata(), wheels[:, 0])
        np.testing.assert_allclose(scene.lines[2].get_ydata(), wheels[:, 1])
        np.testing.assert_allclose(scene.collections[0].get_offsets(), wheels)
    finally:
        plt.close(figure)


def test_wheel_velocity_arrows_follow_body_motion():
    import matplotlib.pyplot as plt

    times, positions, velocities, _, transition_time, _ = simulate_sliding()
    figure, animation = create_animation()
    try:
        figure.canvas.draw()
        arrows = figure.axes[0].collections[2]
        initial_offsets = arrows.get_offsets().copy()
        for index in (0, np.flatnonzero(times == transition_time)[0], len(times) - 1):
            animation._func(index)
            np.testing.assert_allclose(arrows.get_offsets(), initial_offsets + positions[index])
            np.testing.assert_allclose(arrows.U, np.full(4, velocities[index, 0]))
            np.testing.assert_allclose(arrows.V, np.full(4, velocities[index, 1]))
        assert arrows.get_alpha() < 1.0
    finally:
        plt.close(figure)