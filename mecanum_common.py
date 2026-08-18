
import numpy as np
import pytest

class Mecanum:
    """
    Nominal model
    - 1 index is front left, 2 is front right, etc
    - front roller axes point in (slip out)
    - rear roller axes point out (slip in)
    - Body frame is x forward, y left, z up (FLU)
    """

    def __init__(self):
        self.wb_hwidth = 0.25  # wheel base half-width, m
        self.wb_hlength = 0.3  # wheel base half-length, m
        self.wheel_radius = 0.1  # wheel radius, m
        self.body_mass = 100.0  # total platform mass, kg
        self.wheel_spin_inertia = 0.02  # wheel inertia about spin axis, kg*m^2
        self.body_yaw_inertia = 1.2  # platform yaw inertia about COM, kg*m^2

    @staticmethod
    def _constraint_violation(w1, w2, w3, w4):
        """Return value for the no-slip relative wheel velocity constraint."""
        return w1 + w2 - w3 - w4

    @staticmethod
    def _relax_to_constraint(w1, w2, w3, w4):
        """Project wheel speeds to the closest set that satisfies w1+w2-w3-w4=0."""
        shift = (w1 + w2 - w3 - w4) / 4.0
        return (w1 - shift, w2 - shift, w3 + shift, w4 + shift)

    def bodyv_from_wheelv(self, w1, w2, w3, w4, strict=True):
        """
        Calculate body velocity from wheel velocities
        "strict" enforces relative wheel velocity constraint
        """
        violation = self._constraint_violation(w1, w2, w3, w4)
        if strict and not np.isclose(violation, 0.0):
            raise ValueError(
                "Relative wheel velocity constraint broken, strict following enforced. "
                f"w1 + w2 - w3 - w4 = {violation}"
            )
        if (not strict) and (not np.isclose(violation, 0.0)):
            w1, w2, w3, w4 = self._relax_to_constraint(w1, w2, w3, w4)

        bodyv_x = self.wheel_radius / 2.0 * (w1 + w2)  # forward
        bodyv_y = self.wheel_radius / 2.0 * (w3 - w1)  # left
        bodyv_w = self.wheel_radius / (2.0 * (self.wb_hwidth + self.wb_hlength)) * (w2 - w3)

        return np.array([bodyv_x, bodyv_y, bodyv_w])

    def wheelv_from_bodyv(self, vx, vy, vw):
        """
        Calculate wheel velocities from body velocities
        """
        l_plus_w = self.wb_hlength + self.wb_hwidth
        return np.array([
            vx - vy - l_plus_w * vw,
            vx + vy + l_plus_w * vw,
            vx + vy - l_plus_w * vw,
            vx - vy + l_plus_w * vw
        ]) / self.wheel_radius

    def _bodya_from_wheeltorque_matrix(self):
        """
        Return linear map G such that body_accel = G @ wheel_torque.

        This uses the Zeidis 2019 approximate dynamic model (Eq. 44-45,
        differentiated in time): [ax, ay, alpha]^T = G [M1..M4]^T.
        """
        l_plus_w = self.wb_hlength + self.wb_hwidth
        radius = self.wheel_radius

        linear_denom = self.body_mass * radius * radius + 4.0 * self.wheel_spin_inertia
        yaw_denom = self.body_yaw_inertia * radius * radius + 4.0 * self.wheel_spin_inertia * l_plus_w * l_plus_w

        k_linear = radius / linear_denom
        k_yaw = radius / yaw_denom

        return np.array([
            [k_linear, k_linear, k_linear, k_linear],
            [-k_linear, k_linear, k_linear, -k_linear],
            [-k_yaw, k_yaw, -k_yaw, k_yaw],
        ])

    def bodya_from_wheeltorque(self, m1, m2, m3, m4):
        """
        Calculate body acceleration [ax, ay, alpha] from wheel torques [M1..M4].
        """
        torques = np.array([m1, m2, m3, m4], dtype=float)
        return self._bodya_from_wheeltorque_matrix() @ torques

    def wheeltorque_from_bodya(self, ax, ay, alpha):
        """
        Calculate wheel torques [M1..M4] from desired body acceleration.

        The system is underdetermined (3 equations, 4 unknowns), so this returns
        the minimum-norm torque solution via Moore-Penrose pseudoinverse.
        """
        body_accel = np.array([ax, ay, alpha], dtype=float)
        gain = self._bodya_from_wheeltorque_matrix()
        return np.linalg.pinv(gain) @ body_accel

    def _exact_dynamics_coeffs(self):
        """Return coefficients (k2, A2, C2) from Zeidis exact Eq. (66)."""
        l_plus_w = self.wb_hlength + self.wb_hwidth
        radius = self.wheel_radius
        ms = self.body_mass
        j1 = self.wheel_spin_inertia
        jc = self.body_yaw_inertia

        a = ms * radius * radius / 8.0 + jc * radius * radius / (16.0 * l_plus_w * l_plus_w) + j1
        b = jc * radius * radius / (16.0 * l_plus_w * l_plus_w)
        c = ms * radius * radius / 8.0 - jc * radius * radius / (16.0 * l_plus_w * l_plus_w)

        k2 = radius * (b + c) / (2.0 * l_plus_w * (a + c))
        a2 = (3.0 * a + 4.0 * b - c) / (4.0 * (a + c) * (a + 2.0 * b - c))
        c2 = (a + 4.0 * b - 3.0 * c) / (4.0 * (a + c) * (a + 2.0 * b - c))
        return k2, a2, c2

    def _prepare_exact_wheel_state(self, w1, w2, w3, w4, strict):
        """Validate/project wheel state before exact dynamics evaluation."""
        violation = self._constraint_violation(w1, w2, w3, w4)
        if strict and not np.isclose(violation, 0.0):
            raise ValueError(
                "Relative wheel velocity constraint broken, strict following enforced. "
                f"w1 + w2 - w3 - w4 = {violation}"
            )
        if (not strict) and (not np.isclose(violation, 0.0)):
            w1, w2, w3, w4 = self._relax_to_constraint(w1, w2, w3, w4)
        return w1, w2, w3, w4

    def bodya_from_wheeltorque_exact(self, m1, m2, m3, m4, w1, w2, w3, w4, strict=True):
        """
        Exact Zeidis model: body acceleration from wheel torques and wheel speeds.

        Inputs w1..w4 are wheel angular velocities (rad/s), used in the nonlinear
        non-holonomic terms of Eq. (65).
        """
        w1, w2, w3, w4 = self._prepare_exact_wheel_state(w1, w2, w3, w4, strict=strict)

        k2, a2, c2 = self._exact_dynamics_coeffs()
        h = 0.5 * (a2 - c2)

        nl_1 = k2 * (w2 + w3) * (w2 - w3)
        nl_23 = k2 * (w3 - 2.0 * w1 - w2) * (w2 - w3)

        wdd_1 = nl_1 + a2 * m1 - h * (m2 - m3) + c2 * m4
        wdd_2 = nl_23 + a2 * m2 - h * (m1 - m4) + c2 * m3
        wdd_3 = nl_23 + a2 * m3 + h * (m1 - m4) + c2 * m2

        l_plus_w = self.wb_hlength + self.wb_hwidth
        radius = self.wheel_radius
        ax = radius / 2.0 * (wdd_1 + wdd_2)
        ay = radius / 2.0 * (wdd_3 - wdd_1)
        alpha = radius / (2.0 * l_plus_w) * (wdd_2 - wdd_3)
        return np.array([ax, ay, alpha])

    def wheeltorque_from_bodya_exact(self, ax, ay, alpha, w1, w2, w3, w4, strict=True):
        """
        Exact Zeidis model inverse: wheel torques for desired body acceleration.

        This is underdetermined (3 equations, 4 torques); returns minimum-norm
        solution consistent with Eq. (65).
        """
        w1, w2, w3, w4 = self._prepare_exact_wheel_state(w1, w2, w3, w4, strict=strict)

        k2, a2, c2 = self._exact_dynamics_coeffs()
        h = 0.5 * (a2 - c2)

        nl_1 = k2 * (w2 + w3) * (w2 - w3)
        nl_23 = k2 * (w3 - 2.0 * w1 - w2) * (w2 - w3)

        l_plus_w = self.wb_hlength + self.wb_hwidth
        radius = self.wheel_radius
        wdd_1 = (ax - ay - l_plus_w * alpha) / radius
        wdd_2 = (ax + ay + l_plus_w * alpha) / radius
        wdd_3 = (ax + ay - l_plus_w * alpha) / radius

        rhs = np.array([
            wdd_1 - nl_1,
            wdd_2 - nl_23,
            wdd_3 - nl_23,
        ])
        gain = np.array([
            [a2, -h, h, c2],
            [-h, a2, c2, h],
            [h, c2, a2, -h],
        ])
        return np.linalg.pinv(gain) @ rhs


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