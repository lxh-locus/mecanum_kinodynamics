"""
Explicit forward and inverse dynamics helpers for a mecanum platform.

This module mirrors the dynamic models used in `mecanum_common.py` but exposes
standalone functions rather than class methods.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MecanumPhysicsParams:
    """Physical parameters for the mecanum dynamic model.

    Attributes:
        wb_hwidth: Half wheel-base width in meters.
        wb_hlength: Half wheel-base length in meters.
        wheel_radius: Wheel radius in meters.
        body_mass: Platform mass in kilograms.
        wheel_spin_inertia: Per-wheel spin inertia in kg m^2.
        body_yaw_inertia: Platform yaw inertia in kg m^2.
    """

    wb_hwidth: float = 0.2405
    wb_hlength: float = 0.25
    wheel_radius: float = 0.10
    body_mass: float = 100.0
    wheel_spin_inertia: float = 0.08
    body_yaw_inertia: float = 1.2


def params_from_model(model) -> MecanumPhysicsParams:
    """Build a parameter bundle from an existing model object.

    Args:
        model: Object exposing the six physical model attributes.
    Returns:
        A new immutable ``MecanumPhysicsParams`` instance.
    """
    return MecanumPhysicsParams(
        wb_hwidth=float(model.wb_hwidth),
        wb_hlength=float(model.wb_hlength),
        wheel_radius=float(model.wheel_radius),
        body_mass=float(model.body_mass),
        wheel_spin_inertia=float(model.wheel_spin_inertia),
        body_yaw_inertia=float(model.body_yaw_inertia),
    )


def wheel_constraint_violation(wheel_velocity):
    """Return the wheel-speed compatibility residual.

    Args:
        wheel_velocity: Iterable ``[w1, w2, w3, w4]`` in rad/s.
    Returns:
        Scalar residual ``w1 + w2 - w3 - w4``.
    """
    w1, w2, w3, w4 = np.asarray(wheel_velocity, dtype=float)
    return w1 + w2 - w3 - w4


def relax_wheel_velocity_to_constraint(wheel_velocity):
    """Project wheel speeds to the closest compatible vector.

    Args:
        wheel_velocity: Iterable ``[w1, w2, w3, w4]`` in rad/s.
    Returns:
        A length-four NumPy array satisfying ``w1 + w2 - w3 - w4 = 0``.
    """
    w1, w2, w3, w4 = np.asarray(wheel_velocity, dtype=float)
    shift = (w1 + w2 - w3 - w4) / 4.0 # An alternative would be normalize somehow?
    return np.array([w1 - shift, w2 - shift, w3 + shift, w4 + shift], dtype=float)


def _prepare_wheel_velocity(wheel_velocity, strict):
    """Validate a wheel velocity vector and optionally project it.

    Args:
        wheel_velocity: Candidate length-four wheel velocity vector.
        strict: If true, reject compatibility violations; otherwise project them.
    Returns:
        A compatible length-four floating-point NumPy array.
    """
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

    Args:
        wheel_velocity: Wheel speeds ``[w1, w2, w3, w4]`` in rad/s.
        params: Physical model parameters.
        strict: Whether to reject incompatible wheel speeds.
    Returns:
        Body velocity ``[vx, vy, yaw_rate]`` in m/s, m/s, and rad/s.
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

    Args:
        body_velocity: Body velocity ``[vx, vy, yaw_rate]``.
        params: Physical model parameters.
    Returns:
        Compatible wheel speeds ``[w1, w2, w3, w4]`` in rad/s.
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

    Args:
        params: Physical model parameters.
    Returns:
        A ``(3, 4)`` matrix mapping wheel torque ``[M1, M2, M3, M4]`` to
        body acceleration ``[ax, ay, alpha]``.
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
    """Compute approximate body acceleration from wheel torques.

    Args:
        wheel_torque: Torque vector ``[M1, M2, M3, M4]`` in N m.
        params: Physical model parameters.
    Returns:
        Body acceleration ``[ax, ay, alpha]``.
    """
    torque = np.asarray(wheel_torque, dtype=float)
    if torque.shape != (4,):
        raise ValueError("wheel_torque must have shape (4,)")
    return forward_dynamics_matrix_linear(params) @ torque


def inverse_dynamics_linear(body_accel, params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """
    Approximate inverse dynamics: body accelerations -> wheel torques.

    Because the mapping is underdetermined (3 equations, 4 unknowns), this
    returns the minimum-norm solution using a pseudoinverse.

    Args:
        body_accel: Desired acceleration ``[ax, ay, alpha]``.
        params: Physical model parameters.
    Returns:
        Minimum-norm wheel torque vector ``[M1, M2, M3, M4]``.
    """
    accel = np.asarray(body_accel, dtype=float)
    if accel.shape != (3,):
        raise ValueError("body_accel must have shape (3,)")
    gain = forward_dynamics_matrix_linear(params)
    return np.linalg.pinv(gain) @ accel


def individual_wheel_braking_deceleration(max_body_x_deceleration):
    """Convert a body-x deceleration limit to an individual-wheel value.

    The four roller axes are diagonal, with two wheels aligned to each axis
    family. If every wheel provides the same axis-constrained deceleration
    ``d_wheel``, their body-x contributions sum to ``2 * d_wheel``. Therefore,
    the equal per-wheel value corresponding to a desired body-x limit is half
    that limit.

    Args:
        max_body_x_deceleration: Positive total body-x deceleration in m/s^2.
    Returns:
        The equal axis-constrained braking deceleration for one wheel in m/s^2.
    Raises:
        ValueError: If ``max_body_x_deceleration`` is not positive.
    """
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")
    return max_body_x_deceleration / 2.0


def sliding_deceleration(body_velocity, wheel_braking_deceleration, tolerance=1e-9):
    """Generate a planar deceleration using a roller friction-circle model.

    The contact force at each wheel is resolved in the roller-axis frame. The
    component along ``roller_direction`` is the braking component; the
    perpendicular component is the free-rolling direction and is therefore
    not resisted. A friction circle limits the contact force to each wheel's
    share of the available braking budget. This gives a continuous
    slip-angle response: axis alignment gives maximum braking, while a
    90-degree slip angle gives zero braking for that roller.

    ``wheel_braking_deceleration`` is the axis-constrained braking value for
    one wheel, in acceleration units. Use
    ``individual_wheel_braking_deceleration`` to obtain it from a desired
    total body-x deceleration. This is still a reduced model: it assumes equal
    load sharing, ignores yaw moment from contact forces, and uses a hard
    Coulomb-style friction-circle limit rather than a tire brush or measured
    slip-angle curve.

    Args:
        body_velocity: Translational body velocity ``[vx, vy]`` or full planar
            velocity ``[vx, vy, yaw_rate]``. Only ``vx`` and ``vy`` determine
            sliding braking.
        wheel_braking_deceleration: Positive axis-constrained braking
            deceleration for one wheel in m/s^2.
        tolerance: Absolute translational-speed threshold below which the
            returned deceleration is treated as zero.
    Returns:
        A length-three NumPy array ``[ax, ay, alpha]`` in m/s^2 and rad/s^2.
        The result is zero for zero translational velocity.
    Raises:
        ValueError: If the velocity has an unsupported shape or either scalar
            parameter is not positive.
    """
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape not in ((2,), (3,)):
        raise ValueError("body_velocity must have shape (2,) or (3,)")
    if wheel_braking_deceleration <= 0.0:
        raise ValueError("wheel_braking_deceleration must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    translation = velocity[:2]
    speed = np.linalg.norm(translation)
    if speed <= tolerance:
        return np.zeros(3, dtype=float)

    direction = translation / speed
    roller_directions = np.array(
        [
            [1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )
    roller_directions /= np.linalg.norm(roller_directions, axis=1)[:, np.newaxis]
    rolling_projection = roller_directions @ direction
    rolling_projection = np.clip(rolling_projection, -1.0, 1.0)

    # Resolve the velocity into each roller-axis frame. The desired braking
    # force is restricted to that axis, with signed magnitude proportional to
    # the velocity projection, so it is maximal at zero slip and zero at a
    # 90-degree slip angle. The supplied value is already the individual-wheel
    # axis braking value.
    axis_acceleration = -wheel_braking_deceleration * rolling_projection
    friction_limit = np.full(roller_directions.shape[0], wheel_braking_deceleration)
    axis_acceleration = np.clip(axis_acceleration, -friction_limit, friction_limit)
    acceleration = np.sum(axis_acceleration[:, np.newaxis] * roller_directions, axis=0)
    return np.array([acceleration[0], acceleration[1], 0.0], dtype=float)


def exact_dynamics_coeffs(params: MecanumPhysicsParams = MecanumPhysicsParams()):
    """Return exact-model coefficients from Zeidis Eq. 66.

    Args:
        params: Physical model parameters.
    Returns:
        Tuple ``(k2, A2, C2)`` used by the nonlinear dynamics equations.
    """
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

    Args:
        wheel_torque: Wheel torques ``[M1, M2, M3, M4]`` in N m.
        wheel_velocity: Wheel speeds ``[w1, w2, w3, w4]`` in rad/s.
        params: Physical model parameters.
        strict: Whether to reject incompatible wheel speeds.
    Returns:
        Body acceleration ``[ax, ay, alpha]``.
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

    Args:
        body_accel: Desired body acceleration ``[ax, ay, alpha]``.
        wheel_velocity: Wheel speeds ``[w1, w2, w3, w4]`` in rad/s.
        params: Physical model parameters.
        strict: Whether to reject incompatible wheel speeds.
    Returns:
        Minimum-norm wheel torque vector ``[M1, M2, M3, M4]``.
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
    "individual_wheel_braking_deceleration",
    "sliding_deceleration",
    "exact_dynamics_coeffs",
    "forward_dynamics_exact",
    "inverse_dynamics_exact",
]
