
from dataclasses import dataclass

import numpy as np
import pytest

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


@dataclass(frozen=True)
class RobotFootprint:
    """Rectangular robot body footprint in the body frame (x forward, y left)."""

    length: float = 0.812 # 2*0.065 buffer unadded
    width: float = 0.567 # 2*0.065 buffer unadded

    def __post_init__(self):
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("Footprint length and width must be positive")

    def world_corners(self, x, y, theta):
        """Return counter-clockwise footprint corners at the given world pose."""
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array(
            [
                [x + c * half_length - s * half_width, y + s * half_length + c * half_width],
                [x + c * half_length + s * half_width, y + s * half_length - c * half_width],
                [x - c * half_length + s * half_width, y - s * half_length - c * half_width],
                [x - c * half_length - s * half_width, y - s * half_length + c * half_width],
            ],
            dtype=float,
        )


class Mecanum:
    """
    Nominal model
    - 1 index is front left, 2 is front right, etc
    - front roller axes point in (slip out)
    - rear roller axes point out (slip in)
    - Body frame is x forward, y left, z up (FLU)
    """

    def __init__(self, model=None, params=None):
        """
        Initialize model constants from either:
        - `model`: an existing model object via params_from_model, or
        - `params`: a MecanumPhysicsParams bundle, or
        - default MecanumPhysicsParams values.
        """
        if (model is not None) and (params is not None):
            raise ValueError("Provide either model or params, not both")

        if model is not None:
            physics_params = params_from_model(model)
        elif params is not None:
            physics_params = params
        else:
            physics_params = MecanumPhysicsParams()

        self.wb_hwidth = float(physics_params.wb_hwidth)  # wheel base half-width, m
        self.wb_hlength = float(physics_params.wb_hlength)  # wheel base half-length, m
        self.wheel_radius = float(physics_params.wheel_radius)  # wheel radius, m
        self.body_mass = float(physics_params.body_mass)  # total platform mass, kg
        self.wheel_spin_inertia = float(physics_params.wheel_spin_inertia)  # wheel inertia about spin axis, kg*m^2
        self.body_yaw_inertia = float(physics_params.body_yaw_inertia)  # platform yaw inertia about COM, kg*m^2
        self.footprint = RobotFootprint()

    def _physics_params(self):
        return params_from_model(self)

    @staticmethod
    def _constraint_violation(w1, w2, w3, w4):
        """Return value for the no-slip relative wheel velocity constraint."""
        return wheel_constraint_violation([w1, w2, w3, w4])

    @staticmethod
    def _relax_to_constraint(w1, w2, w3, w4):
        """Project wheel speeds to the closest set that satisfies w1+w2-w3-w4=0."""
        relaxed = relax_wheel_velocity_to_constraint([w1, w2, w3, w4])
        return tuple(relaxed)

    def bodyv_from_wheelv(self, w1, w2, w3, w4, strict=True):
        """
        Calculate body velocity from wheel velocities
        "strict" enforces relative wheel velocity constraint
        """
        return forward_kinematics(
            [w1, w2, w3, w4],
            params=self._physics_params(),
            strict=strict,
        )

    def wheelv_from_bodyv(self, vx, vy, vw):
        """
        Calculate wheel velocities from body velocities
        """
        return inverse_kinematics([vx, vy, vw], params=self._physics_params())

    def _bodya_from_wheeltorque_matrix(self):
        """
        Return linear map G such that body_accel = G @ wheel_torque.

        This uses the Zeidis 2019 approximate dynamic model (Eq. 44-45,
        differentiated in time): [ax, ay, alpha]^T = G [M1..M4]^T.
        """
        return forward_dynamics_matrix_linear(self._physics_params())

    def bodya_from_wheeltorque(self, m1, m2, m3, m4):
        """
        Calculate body acceleration [ax, ay, alpha] from wheel torques [M1..M4].
        """
        return forward_dynamics_linear(
            [m1, m2, m3, m4],
            params=self._physics_params(),
        )

    def wheeltorque_from_bodya(self, ax, ay, alpha):
        """
        Calculate wheel torques [M1..M4] from desired body acceleration.

        The system is underdetermined (3 equations, 4 unknowns), so this returns
        the minimum-norm torque solution via Moore-Penrose pseudoinverse.
        """
        return inverse_dynamics_linear(
            [ax, ay, alpha],
            params=self._physics_params(),
        )

    def _exact_dynamics_coeffs(self):
        """Return coefficients (k2, A2, C2) from Zeidis exact Eq. (66)."""
        return exact_dynamics_coeffs(self._physics_params())

    def _prepare_exact_wheel_state(self, w1, w2, w3, w4, strict):
        """Validate/project wheel state before exact dynamics evaluation."""
        wheel_velocity = np.array([w1, w2, w3, w4], dtype=float)
        violation = wheel_constraint_violation(wheel_velocity)
        if strict and not np.isclose(violation, 0.0):
            raise ValueError(
                "Relative wheel velocity constraint broken, strict following enforced. "
                f"w1 + w2 - w3 - w4 = {violation}"
            )
        if (not strict) and (not np.isclose(violation, 0.0)):
            wheel_velocity = relax_wheel_velocity_to_constraint(wheel_velocity)
        return tuple(wheel_velocity)

    def bodya_from_wheeltorque_exact(self, m1, m2, m3, m4, w1, w2, w3, w4, strict=True):
        """
        Exact Zeidis model: body acceleration from wheel torques and wheel speeds.

        Inputs w1..w4 are wheel angular velocities (rad/s), used in the nonlinear
        non-holonomic terms of Eq. (65).
        """
        return forward_dynamics_exact(
            [m1, m2, m3, m4],
            [w1, w2, w3, w4],
            params=self._physics_params(),
            strict=strict,
        )

    def wheeltorque_from_bodya_exact(self, ax, ay, alpha, w1, w2, w3, w4, strict=True):
        """
        Exact Zeidis model inverse: wheel torques for desired body acceleration.

        This is underdetermined (3 equations, 4 torques); returns minimum-norm
        solution consistent with Eq. (65).
        """
        return inverse_dynamics_exact(
            [ax, ay, alpha],
            [w1, w2, w3, w4],
            params=self._physics_params(),
            strict=strict,
        )


def test_body_to_wheel_to_body_roundtrip():
    model = Mecanum()
    bodyv = np.array([1.2, -0.35, 0.8])
    wheelv = model.wheelv_from_bodyv(*bodyv)

    recovered_bodyv = model.bodyv_from_wheelv(*wheelv, strict=True)

    assert np.allclose(recovered_bodyv, bodyv)


def test_body_to_wheel_to_body_roundtrip_multiple_cases_strict_on():
    model = Mecanum()
    bodyv_cases = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.2, -0.35, 0.8]),
        np.array([-0.9, 0.4, -1.1]),
    ]

    for bodyv in bodyv_cases:
        wheelv = model.wheelv_from_bodyv(*bodyv)
        recovered_bodyv = model.bodyv_from_wheelv(*wheelv, strict=True)
        assert np.allclose(recovered_bodyv, bodyv)


def test_wheel_to_body_to_wheel_roundtrip_constraint_satisfied():
    model = Mecanum()
    wheelv = np.array([5.0, 3.0, 4.0, 4.0])
    bodyv = model.bodyv_from_wheelv(*wheelv, strict=True)

    recovered_wheelv = model.wheelv_from_bodyv(*bodyv)

    assert np.allclose(recovered_wheelv, wheelv)


def test_wheel_to_body_to_wheel_roundtrip_strict_off_keeps_valid_wheelv():
    model = Mecanum()
    wheelv = np.array([6.0, -2.0, 1.5, 2.5])

    bodyv = model.bodyv_from_wheelv(*wheelv, strict=False)
    recovered_wheelv = model.wheelv_from_bodyv(*bodyv)

    assert np.allclose(recovered_wheelv, wheelv)


def test_bodyv_from_wheelv_strict_on_raises_on_constraint_violation():
    model = Mecanum()

    with pytest.raises(ValueError):
        model.bodyv_from_wheelv(1.0, 2.0, 3.0, 4.0, strict=True)


def test_bodyv_from_wheelv_strict_off_relaxes_to_valid_solution():
    model = Mecanum()
    wheelv = np.array([1.0, 2.0, 3.0, 4.0])

    relaxed_bodyv = model.bodyv_from_wheelv(*wheelv, strict=False)

    projected_wheelv = np.array(model._relax_to_constraint(*wheelv))
    projected_bodyv = model.bodyv_from_wheelv(*projected_wheelv, strict=True)

    assert np.allclose(relaxed_bodyv, projected_bodyv)


def test_roundtrip_with_invalid_wheelv_strict_off_returns_projected_wheelv():
    model = Mecanum()
    invalid_wheelv = np.array([1.0, 2.0, 3.0, 4.0])

    bodyv = model.bodyv_from_wheelv(*invalid_wheelv, strict=False)
    recovered_wheelv = model.wheelv_from_bodyv(*bodyv)

    projected_wheelv = np.array(model._relax_to_constraint(*invalid_wheelv))

    assert np.allclose(recovered_wheelv, projected_wheelv)


def test_bodya_from_wheeltorque_decoupled_patterns():
    model = Mecanum()

    ax_only = model.bodya_from_wheeltorque(1.0, 1.0, 1.0, 1.0)
    assert np.isclose(ax_only[1], 0.0)
    assert np.isclose(ax_only[2], 0.0)
    assert ax_only[0] > 0.0

    ay_only = model.bodya_from_wheeltorque(-1.0, 1.0, 1.0, -1.0)
    assert np.isclose(ay_only[0], 0.0)
    assert np.isclose(ay_only[2], 0.0)
    assert ay_only[1] > 0.0

    alpha_only = model.bodya_from_wheeltorque(-1.0, 1.0, -1.0, 1.0)
    assert np.isclose(alpha_only[0], 0.0)
    assert np.isclose(alpha_only[1], 0.0)
    assert alpha_only[2] > 0.0


def test_bodya_to_torque_to_bodya_roundtrip_min_norm_inverse():
    model = Mecanum()
    target_bodya = np.array([0.7, -0.4, 1.1])

    wheel_torque = model.wheeltorque_from_bodya(*target_bodya)
    recovered_bodya = model.bodya_from_wheeltorque(*wheel_torque)

    assert np.allclose(recovered_bodya, target_bodya)


def test_exact_bodya_to_torque_to_bodya_roundtrip_min_norm_inverse():
    model = Mecanum()
    wheelv = model.wheelv_from_bodyv(0.8, -0.3, 0.2)
    target_bodya = np.array([0.6, -0.25, 0.9])

    wheel_torque = model.wheeltorque_from_bodya_exact(*target_bodya, *wheelv, strict=True)
    recovered_bodya = model.bodya_from_wheeltorque_exact(*wheel_torque, *wheelv, strict=True)

    assert np.allclose(recovered_bodya, target_bodya)


def test_exact_strict_on_raises_on_constraint_violation():
    model = Mecanum()

    with pytest.raises(ValueError):
        model.bodya_from_wheeltorque_exact(1.0, 2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 4.0, strict=True)


def test_exact_strict_off_relaxes_same_as_projected_state():
    model = Mecanum()
    invalid_wheelv = np.array([1.0, 2.0, 3.0, 4.0])
    torques = np.array([0.8, -0.6, 1.2, -0.4])

    bodya_relaxed = model.bodya_from_wheeltorque_exact(*torques, *invalid_wheelv, strict=False)
    projected_wheelv = np.array(model._relax_to_constraint(*invalid_wheelv))
    bodya_projected = model.bodya_from_wheeltorque_exact(*torques, *projected_wheelv, strict=True)

    assert np.allclose(bodya_relaxed, bodya_projected)