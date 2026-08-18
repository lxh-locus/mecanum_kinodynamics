"""
Explicit forward and inverse dynamics helpers for a mecanum platform.

This module mirrors the dynamic models used in `mecanum_common.py` but exposes
standalone functions rather than class methods.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MecanumPhysicsParams:
    """Physical parameters for the mecanum dynamic model."""

    wb_hwidth: float = 0.25
    wb_hlength: float = 0.30
    wheel_radius: float = 0.10
    body_mass: float = 100.0
    wheel_spin_inertia: float = 0.02
    body_yaw_inertia: float = 1.2


def params_from_model(model) -> MecanumPhysicsParams:
    """Build a parameter bundle from an existing `Mecanum` model object."""
    return MecanumPhysicsParams(
        wb_hwidth=float(model.wb_hwidth),
        wb_hlength=float(model.wb_hlength),
        wheel_radius=float(model.wheel_radius),
        body_mass=float(model.body_mass),
        wheel_spin_inertia=float(model.wheel_spin_inertia),
        body_yaw_inertia=float(model.body_yaw_inertia),
    )


def wheel_constraint_violation(wheel_velocity):
    """Return the wheel-speed compatibility residual: w1 + w2 - w3 - w4."""
    w1, w2, w3, w4 = np.asarray(wheel_velocity, dtype=float)
    return w1 + w2 - w3 - w4


def relax_wheel_velocity_to_constraint(wheel_velocity):
    """Project wheel speeds to the closest vector satisfying w1+w2-w3-w4 = 0."""
    w1, w2, w3, w4 = np.asarray(wheel_velocity, dtype=float)
    shift = (w1 + w2 - w3 - w4) / 4.0 # An alternative would be normalize somehow?
    return np.array([w1 - shift, w2 - shift, w3 + shift, w4 + shift], dtype=float)


def _prepare_wheel_velocity(wheel_velocity, strict):
    w = np.asarray(wheel_velocity, dtype=float)
    if w.shape != (4,):
        raise ValueError("wheel_velocity must have shape (4,)")

    violation = wheel_constraint_violation(w)
    if strict and (not np.isclose(violation, 0.0)):
        raise ValueError(
            "Relative wheel velocity constraint broken, strict following enforced. "
            f"w1 + w2 - w3 - w4 = {violation}"
        )
    if (not strict) and (not np.isclose(violation, 0.0)):
        w = relax_wheel_velocity_to_constraint(w)
    return w


def forward_kinematics(
    wheel_velocity,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    strict=True,
):
    """
    Compute body velocity from wheel velocities.

    Inputs:
    - wheel_velocity: [w1, w2, w3, w4]
    Returns:
    - [vx, vy, yaw_rate]
    """
    w1, w2, w3, _ = _prepare_wheel_velocity(wheel_velocity, strict=strict)
    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius

    vx = radius / 2.0 * (w1 + w2)
    vy = radius / 2.0 * (w3 - w1)
    yaw_rate = radius / (2.0 * l_plus_w) * (w2 - w3)
    return np.array([vx, vy, yaw_rate], dtype=float)


def inverse_kinematics(body_velocity, params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """
    Compute wheel velocities from body velocity.

    The inverse is underdefined if wheel-speed compatibility is not enforced
    (4 wheel speeds from a 3-DOF body velocity). This function returns the
    standard no-slip compatible solution that satisfies w1+w2-w3-w4 = 0.

    Inputs:
    - body_velocity: [vx, vy, yaw_rate]
    Returns:
    - [w1, w2, w3, w4]
    """
    body_v = np.asarray(body_velocity, dtype=float)
    if body_v.shape != (3,):
        raise ValueError("body_velocity must have shape (3,)")
    vx, vy, yaw_rate = body_v

    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius
    return np.array(
        [
            vx - vy - l_plus_w * yaw_rate,
            vx + vy + l_plus_w * yaw_rate,
            vx + vy - l_plus_w * yaw_rate,
            vx - vy + l_plus_w * yaw_rate,
        ],
        dtype=float,
    ) / radius


def forward_dynamics_matrix_linear(params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """
    Return linear map G such that body_accel = G @ wheel_torque.

    body_accel = [ax, ay, alpha]
    wheel_torque = [M1, M2, M3, M4]
    """
    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius

    linear_denom = params.body_mass * radius * radius + 4.0 * params.wheel_spin_inertia
    yaw_denom = (
        params.body_yaw_inertia * radius * radius
        + 4.0 * params.wheel_spin_inertia * l_plus_w * l_plus_w
    )

    k_linear = radius / linear_denom
    k_yaw = radius / yaw_denom

    return np.array(
        [
            [k_linear, k_linear, k_linear, k_linear],
            [-k_linear, k_linear, k_linear, -k_linear],
            [-k_yaw, k_yaw, -k_yaw, k_yaw],
        ],
        dtype=float,
    )


def forward_dynamics_linear(wheel_torque, params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """Approximate forward dynamics: wheel torques -> body accelerations."""
    torque = np.asarray(wheel_torque, dtype=float)
    if torque.shape != (4,):
        raise ValueError("wheel_torque must have shape (4,)")
    return forward_dynamics_matrix_linear(params) @ torque


def inverse_dynamics_linear(body_accel, params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """
    Approximate inverse dynamics: body accelerations -> wheel torques.

    Because the mapping is underdetermined (3 equations, 4 unknowns), this
    returns the minimum-norm solution using a pseudoinverse.
    """
    accel = np.asarray(body_accel, dtype=float)
    if accel.shape != (3,):
        raise ValueError("body_accel must have shape (3,)")
    gain = forward_dynamics_matrix_linear(params)
    return np.linalg.pinv(gain) @ accel


def exact_dynamics_coeffs(params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """Return (k2, A2, C2) coefficients from Zeidis exact dynamics (Eq. 66)."""
    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius

    a = (
        params.body_mass * radius * radius / 8.0
        + params.body_yaw_inertia * radius * radius / (16.0 * l_plus_w * l_plus_w)
        + params.wheel_spin_inertia
    )
    b = params.body_yaw_inertia * radius * radius / (16.0 * l_plus_w * l_plus_w)
    c = (
        params.body_mass * radius * radius / 8.0
        - params.body_yaw_inertia * radius * radius / (16.0 * l_plus_w * l_plus_w)
    )

    k2 = radius * (b + c) / (2.0 * l_plus_w * (a + c))
    a2 = (3.0 * a + 4.0 * b - c) / (4.0 * (a + c) * (a + 2.0 * b - c))
    c2 = (a + 4.0 * b - 3.0 * c) / (4.0 * (a + c) * (a + 2.0 * b - c))
    return k2, a2, c2


def forward_dynamics_exact(
    wheel_torque,
    wheel_velocity,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    strict=True,
):
    """
    Exact nonlinear forward dynamics: wheel torques + wheel speeds -> body accel.

    Inputs:
    - wheel_torque: [M1, M2, M3, M4]
    - wheel_velocity: [w1, w2, w3, w4]
    Returns:
    - [ax, ay, alpha]
    """
    torque = np.asarray(wheel_torque, dtype=float)
    if torque.shape != (4,):
        raise ValueError("wheel_torque must have shape (4,)")
    w1, w2, w3, _ = _prepare_wheel_velocity(wheel_velocity, strict=strict)
    m1, m2, m3, m4 = torque

    k2, a2, c2 = exact_dynamics_coeffs(params)
    h = 0.5 * (a2 - c2)

    nl_1 = k2 * (w2 + w3) * (w2 - w3)
    nl_23 = k2 * (w3 - 2.0 * w1 - w2) * (w2 - w3)

    wdd_1 = nl_1 + a2 * m1 - h * (m2 - m3) + c2 * m4
    wdd_2 = nl_23 + a2 * m2 - h * (m1 - m4) + c2 * m3
    wdd_3 = nl_23 + a2 * m3 + h * (m1 - m4) + c2 * m2

    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius
    ax = radius / 2.0 * (wdd_1 + wdd_2)
    ay = radius / 2.0 * (wdd_3 - wdd_1)
    alpha = radius / (2.0 * l_plus_w) * (wdd_2 - wdd_3)
    return np.array([ax, ay, alpha], dtype=float)


def inverse_dynamics_exact(
    body_accel,
    wheel_velocity,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    strict=True,
):
    """
    Exact nonlinear inverse dynamics: desired body accel + wheel speeds -> torques.

    This is underdetermined, so the minimum-norm wheel torque solution is
    returned via pseudoinverse.
    """
    accel = np.asarray(body_accel, dtype=float)
    if accel.shape != (3,):
        raise ValueError("body_accel must have shape (3,)")
    ax, ay, alpha = accel

    w1, w2, w3, _ = _prepare_wheel_velocity(wheel_velocity, strict=strict)

    k2, a2, c2 = exact_dynamics_coeffs(params)
    h = 0.5 * (a2 - c2)

    nl_1 = k2 * (w2 + w3) * (w2 - w3)
    nl_23 = k2 * (w3 - 2.0 * w1 - w2) * (w2 - w3)

    l_plus_w = params.wb_hlength + params.wb_hwidth
    radius = params.wheel_radius
    wdd_1 = (ax - ay - l_plus_w * alpha) / radius
    wdd_2 = (ax + ay + l_plus_w * alpha) / radius
    wdd_3 = (ax + ay - l_plus_w * alpha) / radius

    rhs = np.array([
        wdd_1 - nl_1,
        wdd_2 - nl_23,
        wdd_3 - nl_23,
    ])
    gain = np.array(
        [
            [a2, -h, h, c2],
            [-h, a2, c2, h],
            [h, c2, a2, -h],
        ],
        dtype=float,
    )
    return np.linalg.pinv(gain) @ rhs


__all__ = [
    "MecanumPhysicsParams",
    "params_from_model",
    "wheel_constraint_violation",
    "relax_wheel_velocity_to_constraint",
    "forward_kinematics",
    "inverse_kinematics",
    "forward_dynamics_matrix_linear",
    "forward_dynamics_linear",
    "inverse_dynamics_linear",
    "exact_dynamics_coeffs",
    "forward_dynamics_exact",
    "inverse_dynamics_exact",
]
