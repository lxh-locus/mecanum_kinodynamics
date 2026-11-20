"""
mecanum_fixed_axis.py

Self-contained sliding dynamics and trajectory rollout for the fixed-axis
Coulomb mecanum braking model. This strips the pieces of
experiments/mecanum_common.py, experiments/mecanum_physics.py, and
experiments/mecanum_sliding.py that fieldset_generator_redux1/
generate_rollouts_2.py relies on down to the single
``sliding_deceleration_coulomb_fixed_axis`` model, so redux2 scripts need no
imports from the experiments package for braking physics.

Provided here:
- ``MecanumPhysicsParams``: the physical parameters the fixed-axis model uses.
- ``individual_wheel_braking_deceleration``: body-x to per-wheel calibration.
- ``sliding_deceleration_coulomb_fixed_axis``: the roller friction-circle model.
- ``rollout_sliding_deceleration``: pose/velocity integration under braking.
- ``integrate_response_delay`` / ``append_response_delay``: constant-velocity
  sensor-delay segment handling.
- ``fixed_axis_braking_rollout``: one-call delay + braking trajectory rollout.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MecanumPhysicsParams:
    """Physical parameters for the fixed-axis sliding mecanum model.

    Attributes:
        wb_hwidth: Half wheel-base width in meters.
        wb_hlength: Half wheel-base length in meters.
        wheel_radius: Wheel radius in meters.
        body_mass: Platform mass in kilograms.
        body_yaw_inertia: Platform yaw inertia in kg m^2.
        roller_directions: Four roller rotation-axis directions in the body
            frame, ordered as front-left, front-right, rear-left, rear-right.
            The body frame is FLU: x forward, y left, and z up. Each direction
            is a 2D ``(x, y)`` vector; its sign selects an axis orientation.
    """

    wb_hwidth: float = 0.2405
    wb_hlength: float = 0.25
    wheel_radius: float = 3 * 0.0254
    body_mass: float = 100.0
    body_yaw_inertia: float = 8.174  # 1/12*100*(.812^2 + .567^2)
    roller_directions: tuple = (
        (1.0, -1.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, -1.0),
    )


def _normalized_roller_directions(params):
    # TODO: Would it make sense to embedded this in the physics model above, since mecanum is [almost] always 45deg tilts?
    # for example, on initialization, set to ( (cos(45deg), sin(-45deg)), etc... )
    """Return the validated, unit-norm (4, 2) roller-direction array."""
    roller_directions = np.asarray(params.roller_directions, dtype=float)
    if roller_directions.shape != (4, 2):
        raise ValueError("params.roller_directions must have shape (4, 2)")
    roller_norms = np.linalg.norm(roller_directions, axis=1)
    if np.any(roller_norms <= 0.0):
        raise ValueError("params.roller_directions must contain nonzero vectors")
    return roller_directions / roller_norms[:, np.newaxis]


def individual_wheel_braking_deceleration(
    max_body_x_deceleration,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
):
    """Convert a body-x deceleration limit to an individual-wheel value.

    Calibrates for ``sliding_deceleration_coulomb_fixed_axis``, whose hard
    per-wheel sign switch contributes a full ``|roller_dir_x|`` body-x
    component per wheel.

    Args:
        max_body_x_deceleration: Positive total body-x deceleration in m/s^2.
        params: Physical model parameters containing the four body-frame roller
            directions. The current calibration assumes their layout is
            symmetric.
    Returns:
        The equal axis-constrained braking deceleration for one wheel in m/s^2.
    Raises:
        ValueError: If ``max_body_x_deceleration`` is not positive.
    """
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")
    roller_directions = _normalized_roller_directions(params)
    body_x_gain = np.sum(np.abs(roller_directions[:, 0]))
    return max_body_x_deceleration / body_x_gain


def _sliding_contact_data(body_velocity, params, tolerance):
    """Build normalized roller and contact-velocity data for the sliding model.

    Args:
        body_velocity: Translational body velocity ``[vx, vy]`` or full planar
            velocity ``[vx, vy, yaw_rate]``. The yaw rate contributes to each
            wheel's local contact velocity when present.
        params: Mecanum physical parameters containing wheel half-dimensions
            and the four body-frame roller directions.
        tolerance: Positive threshold used to identify negligible body and
            wheel-contact velocities.
    Returns:
        ``None`` when both translational speed and yaw rate are within
        ``tolerance``. Otherwise, a tuple
        ``(wheel_positions, roller_directions, rolling_projection)`` where each
        array has four rows, one per wheel:

        - ``wheel_positions`` has shape ``(4, 2)`` and contains each wheel's
          ``[x, y]`` body-frame position.
        - ``roller_directions`` has shape ``(4, 2)`` and contains normalized
          roller-axis directions.
        - ``rolling_projection`` has shape ``(4,)`` and contains the clipped
          dot product of each roller direction with its contact direction.
    Raises:
        ValueError: If ``body_velocity`` does not have shape ``(2,)`` or
            ``(3,)``, ``tolerance`` is not positive, or the roller directions
            do not have shape ``(4, 2)`` or contain a zero vector.
    """
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape not in ((2,), (3,)):
        raise ValueError("body_velocity must have shape (2,) or (3,)")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    translation = velocity[:2]
    yaw_rate = velocity[2] if velocity.shape == (3,) else 0.0
    speed = np.linalg.norm(translation)
    if speed <= tolerance and abs(yaw_rate) <= tolerance:
        return None

    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    roller_directions = _normalized_roller_directions(params)

    contact_velocities = np.column_stack(
        [
            translation[0] - yaw_rate * wheel_positions[:, 1],
            translation[1] + yaw_rate * wheel_positions[:, 0],
        ]
    )
    contact_speeds = np.linalg.norm(contact_velocities, axis=1)
    contact_directions = np.zeros_like(contact_velocities)
    nonzero_contacts = contact_speeds > tolerance
    contact_directions[nonzero_contacts] = (
        contact_velocities[nonzero_contacts]
        / contact_speeds[nonzero_contacts, np.newaxis]
    )
    rolling_projection = np.sum(roller_directions * contact_directions, axis=1)
    rolling_projection = np.clip(rolling_projection, -1.0, 1.0) # TODO: Maybe remove since only care about sign
    return wheel_positions, roller_directions, rolling_projection


def _body_acceleration_from_contacts(contact_accelerations, wheel_positions, params):
    """Sum wheel contact accelerations and convert their moment to yaw acceleration."""
    acceleration = np.sum(contact_accelerations, axis=0)
    yaw_acceleration = (
        params.body_mass
        * np.sum(
            wheel_positions[:, 0] * contact_accelerations[:, 1]
            - wheel_positions[:, 1] * contact_accelerations[:, 0]
        )
        / params.body_yaw_inertia
    )
    return np.array([acceleration[0], acceleration[1], yaw_acceleration], dtype=float)


def sliding_deceleration_coulomb_fixed_axis(
    body_velocity,
    wheel_braking_deceleration,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    tolerance=1e-9,
):
    """Generate a planar deceleration using a roller coulomb model.

    The contact force at each wheel is resolved in the roller-axis frame. The
    component along ``roller_direction`` is the braking component; the
    perpendicular component is the free-rolling direction and is therefore
    not resisted. A friction limit caps each wheel's axis force at its
    individual braking value. A slipping roller contributes its full axis
    braking value; a roller with zero velocity along its axis is fully rolling
    and contributes no braking. This is a Coulomb-style sliding model: each
    slipping roller applies a fixed-magnitude braking response based only on
    the sign of the velocity projected onto its resisted axis. As a result,
    the response changes abruptly when that projection crosses zero instead
    of varying smoothly with slip angle.

    Upshot: Per-wheel braking magnitudes are fixed, and braking directions
    are fixed along the roller axis.

    ``wheel_braking_deceleration`` is the axis-constrained braking value for
    one wheel, in acceleration units. Use
    ``individual_wheel_braking_deceleration`` to obtain it from a desired
    total body-x deceleration. This is still a reduced model: it assumes equal
    load sharing, includes yaw moment only from the resolved contact forces,
    and uses a hard Coulomb-style sliding limit rather than a tire brush or
    measured slip-angle curve.

    Args:
        body_velocity: Translational body velocity ``[vx, vy]`` or full planar
            velocity ``[vx, vy, yaw_rate]``. The yaw rate contributes to each
            wheel's local contact velocity when present.
        wheel_braking_deceleration: Positive axis-constrained braking
            deceleration for one wheel in m/s^2.
        params: Physical model parameters used for wheel locations, body mass,
            and yaw inertia.
        tolerance: Absolute translational-speed threshold below which the
            returned deceleration is treated as zero.
    Returns:
        A length-three NumPy array ``[ax, ay, alpha]`` in m/s^2 and rad/s^2.
        The result is zero for zero translational velocity.
    Raises:
        ValueError: If the velocity has an unsupported shape or a scalar
            parameter is not positive.
    """
    if wheel_braking_deceleration <= 0.0:
        raise ValueError("wheel_braking_deceleration must be positive")
    contact_data = _sliding_contact_data(body_velocity, params, tolerance)
    if contact_data is None:
        return np.zeros(3, dtype=float)
    wheel_positions, roller_directions, rolling_projection = contact_data

    # A slipping roller supplies its full Coulomb braking value along its axis.
    # A zero projection means it is fully rolling, so it supplies no braking.
    axis_acceleration = -wheel_braking_deceleration * np.sign(rolling_projection)
    axis_acceleration[np.isclose(rolling_projection, 0.0, atol=tolerance)] = 0.0
    contact_accelerations = axis_acceleration[:, np.newaxis] * roller_directions
    return _body_acceleration_from_contacts(contact_accelerations, wheel_positions, params)


def _is_stopped(body_velocity, speed_tolerance, yaw_rate_tolerance):
    """Return true when translational and yaw speeds are both negligible."""
    velocity = np.asarray(body_velocity, dtype=float)
    return np.linalg.norm(velocity[:2]) <= speed_tolerance and abs(velocity[2]) <= yaw_rate_tolerance


def _advance_body_velocity(body_velocity, body_acceleration, dt, speed_tolerance, yaw_rate_tolerance):
    """Advance body velocity and clamp small Coulomb-friction sign crossings."""
    velocity = np.asarray(body_velocity, dtype=float)
    acceleration = np.asarray(body_acceleration, dtype=float)
    next_velocity = velocity + dt * acceleration

    opposes_motion = velocity * acceleration < 0.0
    crossed_zero = np.signbit(velocity) != np.signbit(next_velocity)
    next_velocity[opposes_motion & crossed_zero] = 0.0

    if np.linalg.norm(next_velocity[:2]) <= speed_tolerance:
        next_velocity[:2] = 0.0
    if abs(next_velocity[2]) <= yaw_rate_tolerance:
        next_velocity[2] = 0.0
    return next_velocity


def rollout_sliding_deceleration(
    body_velocity,
    wheel_braking_deceleration,
    params=None,
    dt=0.005,
    max_time=5.0,
    speed_tolerance=1e-4,
    yaw_rate_tolerance=1e-4,
):
    """Roll out pose and body velocity under fixed-axis Coulomb braking.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        wheel_braking_deceleration: Per-wheel roller-axis braking deceleration,
            calibrated via ``individual_wheel_braking_deceleration``.
        params: Mecanum physical parameters, or ``None`` for defaults.
        dt: Integration step in seconds.
        max_time: Maximum rollout duration in seconds.
        speed_tolerance: Translational stopping threshold in m/s.
        yaw_rate_tolerance: Yaw-rate stopping threshold in rad/s.
    Returns:
        ``(poses, velocities, stop_time, stopped)``. ``poses`` are
        ``[x, y, theta]`` rows, and ``velocities`` are ``[vx, vy, yaw_rate]`` rows.
    """
    if params is None:
        params = MecanumPhysicsParams()
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("body_velocity must have shape (3,)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")
    if speed_tolerance <= 0.0:
        raise ValueError("speed_tolerance must be positive")
    if yaw_rate_tolerance <= 0.0:
        raise ValueError("yaw_rate_tolerance must be positive")

    max_steps = int(np.ceil(max_time / dt))
    poses = [np.zeros(3, dtype=float)]
    velocities = [velocity.copy()]
    stopped = _is_stopped(velocity, speed_tolerance, yaw_rate_tolerance)
    stop_time = 0.0 if stopped else max_time

    for step_idx in range(max_steps):
        if stopped:
            break

        t = step_idx * dt
        step = min(dt, max_time - t)
        if step <= 0.0:
            break

        acceleration = sliding_deceleration_coulomb_fixed_axis(
            velocity, wheel_braking_deceleration, params=params
        )
        # Convert force-derived acceleration into the derivative of body-frame
        # velocity by including the rotating-frame transport term.
        body_frame_acceleration = acceleration.copy()
        body_frame_acceleration[0] += velocity[2] * velocity[1]
        body_frame_acceleration[1] -= velocity[2] * velocity[0]
        x, y, theta = poses[-1]
        c = np.cos(theta)
        s = np.sin(theta)
        poses.append(
            np.array(
                [
                    x + step * (c * velocity[0] - s * velocity[1]),
                    y + step * (s * velocity[0] + c * velocity[1]),
                    theta + step * velocity[2],
                ],
                dtype=float,
            )
        )

        velocity = _advance_body_velocity(
            velocity,
            body_frame_acceleration,
            step,
            speed_tolerance=speed_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )
        velocities.append(velocity.copy())

        stopped = _is_stopped(velocity, speed_tolerance, yaw_rate_tolerance)
        if stopped:
            stop_time = t + step

    return np.asarray(poses), np.asarray(velocities), stop_time, stopped


def integrate_response_delay(initial_velocity, duration, dt):
    """Integrate the constant-velocity response period before sliding braking begins."""
    steps = int(duration // dt)
    poses = np.zeros((steps + 1, 3), dtype=float)
    vx, vy, yaw_rate = np.asarray(initial_velocity, dtype=float)
    for index in range(steps):
        x, y, yaw = poses[index]
        c = np.cos(yaw)
        s = np.sin(yaw)
        poses[index + 1] = (x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + yaw_rate * dt)
    return poses


def prepend_response_delay(initial_velocity, braking_poses, delay, dt):
    """Hold velocity during response delay, then prepend braking poses."""
    response_poses = integrate_response_delay(initial_velocity, delay, dt)
    x, y, yaw = response_poses[-1]
    c = np.cos(yaw)
    s = np.sin(yaw)
    shifted_braking = braking_poses.copy()
    shifted_braking[:, 0] = x + c * braking_poses[:, 0] - s * braking_poses[:, 1]
    shifted_braking[:, 1] = y + s * braking_poses[:, 0] + c * braking_poses[:, 1]
    shifted_braking[:, 2] += yaw
    return np.vstack((response_poses, shifted_braking[1:]))


def fixed_axis_braking_rollout(
    initial_velocity,
    single_wheel_braking_deceleration,
    delay,
    params=None,
    dt=0.005,
    max_time=5.0,
):
    """Run the fixed-axis braking rollout with a leading response delay.

    Holds ``initial_velocity`` for ``delay`` seconds, then brakes with the
    fixed-axis Coulomb model calibrated from the body-x deceleration limit.

    Args:
        initial_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        body_x_braking_deceleration: Positive total body-x deceleration in m/s^2.
        single_wheel_braking_deceleration: Positive individual wheel deceleration in m/s^2
        delay: Response delay before braking begins, in seconds.
        params: Mecanum physical parameters, or ``None`` for defaults.
        dt: Integration step in seconds.
        max_time: Maximum sliding-braking duration in seconds.
    Returns:
        ``(trajectory, stop_time, stopped, wheel_braking)``. ``trajectory`` is
        the full ``[x, y, theta]`` pose history including the delay segment,
        ``stop_time`` is the braking duration (excluding the delay),
        ``stopped`` reports whether braking finished within ``max_time``, and
        ``wheel_braking`` is the calibrated per-wheel deceleration in m/s^2.
    """
    if params is None:
        params = MecanumPhysicsParams()

    braking_poses, _, stop_time, stopped = rollout_sliding_deceleration(
        initial_velocity, single_wheel_braking_deceleration, params=params, dt=dt, max_time=max_time
    )
    trajectory = prepend_response_delay(initial_velocity, braking_poses, delay, dt)
    return trajectory, stop_time, stopped, single_wheel_braking_deceleration


__all__ = [
    "MecanumPhysicsParams",
    "individual_wheel_braking_deceleration",
    "sliding_deceleration_coulomb_fixed_axis",
    "rollout_sliding_deceleration",
    "integrate_response_delay",
    "prepend_response_delay",
    "fixed_axis_braking_rollout",
]
