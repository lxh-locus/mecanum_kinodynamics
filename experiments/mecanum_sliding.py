"""
mecanum_sliding.py

Sliding-friction braking models for a mecanum platform, and the pose/velocity
rollout loops that integrate each model forward from an initial body
velocity. Consolidates functions previously scattered across
mecanum_physics.py, sliding_stopping_distance_rollout.py, and
analysis_sliding_stopping_distance.py.
"""
import numpy as np

try:
    from .mecanum_physics import MecanumPhysicsParams
except ImportError:
    from mecanum_physics import MecanumPhysicsParams


def individual_wheel_braking_deceleration(
    max_body_x_deceleration,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    model="coulomb",
):
    """Convert a body-x deceleration limit to an individual-wheel value.

    The four roller axes are diagonal, and the required per-wheel calibration
    depends on which sliding model will consume the result:
    - ``"coulomb"`` (default) calibrates for ``sliding_deceleration_coulomb_model``,
      whose hard per-wheel sign switch contributes a full ``|roller_dir_x|``
      body-x component per wheel.
    - ``"approx"`` calibrates for ``sliding_deceleration_approx_model``, whose
      continuous alignment weighting contributes a squared ``roller_dir_x**2``
      body-x component per wheel.

    Args:
        max_body_x_deceleration: Positive total body-x deceleration in m/s^2.
        params: Physical model parameters containing the four body-frame roller
            directions. The current calibration assumes their layout is
            symmetric.
        model: Which sliding model to calibrate for, ``"coulomb"`` or ``"approx"``.
    Returns:
        The equal axis-constrained braking deceleration for one wheel in m/s^2.
    Raises:
        ValueError: If ``max_body_x_deceleration`` is not positive or ``model``
            is not ``"coulomb"`` or ``"approx"``.
    """
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")
    if model not in ("coulomb", "approx"):
        raise ValueError("model must be 'coulomb' or 'approx'")
    roller_directions = np.asarray(params.roller_directions, dtype=float)
    if roller_directions.shape != (4, 2):
        raise ValueError("params.roller_directions must have shape (4, 2)")
    roller_norms = np.linalg.norm(roller_directions, axis=1)
    if np.any(roller_norms <= 0.0):
        raise ValueError("params.roller_directions must contain nonzero vectors")
    roller_directions /= roller_norms[:, np.newaxis]
    if model == "coulomb":
        body_x_gain = np.sum(np.abs(roller_directions[:, 0]))
    else:
        body_x_gain = np.sum(roller_directions[:, 0] ** 2)
    return max_body_x_deceleration / body_x_gain


def sliding_deceleration_approx_model(
    body_velocity,
    wheel_braking_deceleration,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    tolerance=1e-9,
):
    """Generate a planar deceleration using a roller friction-circle model.

    The contact force at each wheel is resolved in the roller-axis frame. Each
    wheel's braking contribution along its own ``roller_direction`` is weighted
    continuously by that axis's alignment with the wheel's contact-point slip
    direction (``roller_direction`` dot ``contact_direction``); the
    perpendicular component is the free-rolling direction and is therefore not
    resisted. For this platform's symmetric 45-degree roller layout, that
    continuous weighting makes the four wheels' contributions sum to an
    isotropic result: the resultant deceleration always points opposite the
    slip direction, with constant magnitude regardless of heading, instead of
    snapping between roller-axis-aligned directions as a hard Coulomb sign
    switch would.

    ``wheel_braking_deceleration`` is the axis-constrained braking value for
    one wheel, in acceleration units. Use
    ``individual_wheel_braking_deceleration(..., model="approx")`` to obtain it
    from a desired total body-x deceleration. This is still a reduced model:
    it assumes equal load sharing and includes yaw moment only from the
    resolved contact forces.

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
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape not in ((2,), (3,)):
        raise ValueError("body_velocity must have shape (2,) or (3,)")
    if wheel_braking_deceleration <= 0.0:
        raise ValueError("wheel_braking_deceleration must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    translation = velocity[:2]
    yaw_rate = velocity[2] if velocity.shape == (3,) else 0.0
    speed = np.linalg.norm(translation)
    if speed <= tolerance and abs(yaw_rate) <= tolerance:
        return np.zeros(3, dtype=float)

    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    roller_directions = np.asarray(params.roller_directions, dtype=float)
    if roller_directions.shape != (4, 2):
        raise ValueError("params.roller_directions must have shape (4, 2)")
    roller_norms = np.linalg.norm(roller_directions, axis=1)
    if np.any(roller_norms <= 0.0):
        raise ValueError("params.roller_directions must contain nonzero vectors")
    roller_directions /= roller_norms[:, np.newaxis]
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
        contact_velocities[nonzero_contacts] / contact_speeds[nonzero_contacts, np.newaxis]
    )
    rolling_projection = np.sum(roller_directions * contact_directions, axis=1)
    rolling_projection = np.clip(rolling_projection, -1.0, 1.0)

    # Continuous alignment weighting (rather than a hard Coulomb sign switch)
    # blends braking smoothly across body x/y as heading rotates; a wheel
    # whose contact velocity is perpendicular to its roller axis naturally
    # contributes zero, since its projection is already zero.
    axis_acceleration = -wheel_braking_deceleration * rolling_projection
    contact_accelerations = axis_acceleration[:, np.newaxis] * roller_directions
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


def sliding_deceleration_coulomb_model(
    body_velocity,
    wheel_braking_deceleration,
    params: MecanumPhysicsParams = MecanumPhysicsParams(),
    tolerance=1e-9,
):
    """Generate a planar deceleration using a roller friction-circle model.

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

    ``wheel_braking_deceleration`` is the axis-constrained braking value for
    one wheel, in acceleration units. Use
    ``individual_wheel_braking_deceleration(..., model="coulomb")`` to obtain it
    from a desired total body-x deceleration. This is still a reduced model: it
    assumes equal load sharing, includes yaw moment only from the resolved
    contact forces, and uses a hard Coulomb-style sliding limit rather than a
    tire brush or measured slip-angle curve.

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
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape not in ((2,), (3,)):
        raise ValueError("body_velocity must have shape (2,) or (3,)")
    if wheel_braking_deceleration <= 0.0:
        raise ValueError("wheel_braking_deceleration must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    translation = velocity[:2]
    yaw_rate = velocity[2] if velocity.shape == (3,) else 0.0
    speed = np.linalg.norm(translation)
    if speed <= tolerance and abs(yaw_rate) <= tolerance:
        return np.zeros(3, dtype=float)

    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    roller_directions = np.asarray(params.roller_directions, dtype=float)
    if roller_directions.shape != (4, 2):
        raise ValueError("params.roller_directions must have shape (4, 2)")
    roller_norms = np.linalg.norm(roller_directions, axis=1)
    if np.any(roller_norms <= 0.0):
        raise ValueError("params.roller_directions must contain nonzero vectors")
    roller_directions /= roller_norms[:, np.newaxis]
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
        contact_velocities[nonzero_contacts] / contact_speeds[nonzero_contacts, np.newaxis]
    )
    rolling_projection = np.sum(roller_directions * contact_directions, axis=1)
    rolling_projection = np.clip(rolling_projection, -1.0, 1.0)

    # A slipping roller supplies its full Coulomb braking value along its axis.
    # A zero projection means it is fully rolling, so it supplies no braking.
    axis_acceleration = -wheel_braking_deceleration * np.sign(rolling_projection)
    axis_acceleration[np.isclose(rolling_projection, 0.0, atol=tolerance)] = 0.0
    contact_accelerations = axis_acceleration[:, np.newaxis] * roller_directions
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


def sliding_deceleration_discrete_emperical(
    body_velocity,
    cardinal_body_deceleration,
    diagonal_body_deceleration,
    diagonal_angle_half_width_degrees,
    tolerance=1e-9,
):
    """Generate empirical stepwise sliding deceleration by velocity heading.

    The model applies ``cardinal_body_deceleration`` for headings near the body
    x/y axes and ``diagonal_body_deceleration`` for headings inside the repeated
    45-degree diagonal bands. ``diagonal_angle_half_width_degrees`` sets the
    symmetric half-width around every diagonal direction; for example, 5 degrees
    applies the diagonal value from 40 to 50 degrees, and likewise in every
    quadrant. There is no smooth transition between the two values.

    Args:
        body_velocity: Translational body velocity ``[vx, vy]`` or full planar
            velocity ``[vx, vy, yaw_rate]``. Only the translational direction is
            used by this empirical model.
        cardinal_body_deceleration: Positive body deceleration used outside the
            diagonal bands, in m/s^2.
        diagonal_body_deceleration: Positive body deceleration used inside the
            diagonal bands, in m/s^2. This must not exceed the cardinal value.
        diagonal_angle_half_width_degrees: Half-width around each 45-degree
            diagonal direction, in degrees. Must be in ``[0, 45]``.
        tolerance: Translational-speed threshold below which the returned
            deceleration is treated as zero.
    Returns:
        A length-three NumPy array ``[ax, ay, alpha]`` in m/s^2 and rad/s^2.
        ``alpha`` is zero because this empirical model only specifies body-x/y
        deceleration values.
    Raises:
        ValueError: If the velocity shape is unsupported or a scalar parameter
            is outside its valid range.
    """
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape not in ((2,), (3,)):
        raise ValueError("body_velocity must have shape (2,) or (3,)")
    if cardinal_body_deceleration <= 0.0:
        raise ValueError("cardinal_body_deceleration must be positive")
    if diagonal_body_deceleration <= 0.0:
        raise ValueError("diagonal_body_deceleration must be positive")
    if diagonal_body_deceleration > cardinal_body_deceleration:
        raise ValueError("diagonal_body_deceleration must not exceed cardinal_body_deceleration")
    if not (0.0 <= diagonal_angle_half_width_degrees <= 45.0):
        raise ValueError("diagonal_angle_half_width_degrees must be in [0, 45]")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    translation = velocity[:2]
    speed = np.linalg.norm(translation)
    if speed <= tolerance:
        return np.zeros(3, dtype=float)

    heading_degrees = np.degrees(np.arctan2(translation[1], translation[0])) % 90.0
    distance_from_diagonal = abs(heading_degrees - 45.0)
    if distance_from_diagonal <= diagonal_angle_half_width_degrees:
        deceleration = diagonal_body_deceleration
    else:
        deceleration = cardinal_body_deceleration

    acceleration_xy = -deceleration * translation / speed
    return np.array([acceleration_xy[0], acceleration_xy[1], 0.0], dtype=float)


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


def rollout_sliding_deceleration_coulomb(
    body_velocity,
    wheel_braking_deceleration,
    params=None,
    dt=0.005,
    max_time=5.0,
    speed_tolerance=1e-4,
    yaw_rate_tolerance=1e-4,
):
    """Roll out pose and body velocity under the Coulomb sliding deceleration model.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        wheel_braking_deceleration: Per-wheel roller-axis braking deceleration,
            calibrated via ``individual_wheel_braking_deceleration(..., model="coulomb")``.
        params: Mecanum physical parameters, or ``None`` for defaults.
        dt: Integration step in seconds.
        max_time: Maximum rollout duration in seconds.
        speed_tolerance: Translational stopping threshold in m/s.
        yaw_rate_tolerance: Yaw-rate stopping threshold in rad/s.
    Returns:
        ``(states, velocities, stop_time, stopped)``. ``states`` are
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
    states = [np.zeros(3, dtype=float)]
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

        acceleration = sliding_deceleration_coulomb_model(
            velocity,
            wheel_braking_deceleration=wheel_braking_deceleration,
            params=params,
        )
        x, y, theta = states[-1]
        c = np.cos(theta)
        s = np.sin(theta)
        states.append(
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
            acceleration,
            step,
            speed_tolerance=speed_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )
        velocities.append(velocity.copy())

        stopped = _is_stopped(velocity, speed_tolerance, yaw_rate_tolerance)
        if stopped:
            stop_time = t + step

    return np.asarray(states), np.asarray(velocities), stop_time, stopped


def rollout_sliding_deceleration_approx(
    body_velocity,
    wheel_braking_deceleration,
    params=None,
    dt=0.005,
    max_time=5.0,
    speed_tolerance=1e-4,
    yaw_rate_tolerance=1e-4,
):
    """Roll out pose and body velocity under the continuous sliding-approx model.

    Mirrors ``rollout_sliding_deceleration``'s integration loop, but calls
    ``sliding_deceleration_approx_model`` instead of the Coulomb sign-switch
    model.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        wheel_braking_deceleration: Per-wheel roller-axis braking deceleration,
            calibrated via ``individual_wheel_braking_deceleration(..., model="approx")``.
        params: Mecanum physical parameters, or ``None`` for defaults.
        dt: Integration step in seconds.
        max_time: Maximum rollout duration in seconds.
        speed_tolerance: Translational stopping threshold in m/s.
        yaw_rate_tolerance: Yaw-rate stopping threshold in rad/s.
    Returns:
        ``(states, velocities, stop_time, stopped)``, matching
        ``rollout_sliding_deceleration``'s return shape.
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
    states = [np.zeros(3, dtype=float)]
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

        acceleration = sliding_deceleration_approx_model(
            velocity,
            wheel_braking_deceleration=wheel_braking_deceleration,
            params=params,
        )
        x, y, theta = states[-1]
        c = np.cos(theta)
        s = np.sin(theta)
        states.append(
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
            acceleration,
            step,
            speed_tolerance=speed_tolerance,
            yaw_rate_tolerance=yaw_rate_tolerance,
        )
        velocities.append(velocity.copy())

        stopped = _is_stopped(velocity, speed_tolerance, yaw_rate_tolerance)
        if stopped:
            stop_time = t + step

    return np.asarray(states), np.asarray(velocities), stop_time, stopped


def rollout_discrete_empirical_deceleration(
    body_velocity,
    cardinal_body_deceleration,
    diagonal_body_deceleration,
    diagonal_angle_half_width_degrees,
    dt=0.005,
    max_time=5.0,
    speed_tolerance=1e-4,
):
    """Roll out pose and body velocity under the discrete empirical deceleration model.

    Mirrors ``rollout_sliding_deceleration``'s integration loop, but calls
    ``sliding_deceleration_discrete_emperical`` instead of the roller
    friction-circle model. Yaw rate is held constant because that model only
    specifies body-x/y deceleration.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        cardinal_body_deceleration: Positive deceleration outside diagonal bands [m/s^2].
        diagonal_body_deceleration: Positive deceleration inside diagonal bands [m/s^2].
        diagonal_angle_half_width_degrees: Half-width around each diagonal direction [deg].
        dt: Integration step in seconds.
        max_time: Maximum rollout duration in seconds.
        speed_tolerance: Translational stopping threshold in m/s.
    Returns:
        ``(states, velocities, stop_time, stopped)``, matching
        ``rollout_sliding_deceleration``'s return shape.
    """
    velocity = np.asarray(body_velocity, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("body_velocity must have shape (3,)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if max_time <= 0.0:
        raise ValueError("max_time must be positive")
    if speed_tolerance <= 0.0:
        raise ValueError("speed_tolerance must be positive")

    max_steps = int(np.ceil(max_time / dt))
    states = [np.zeros(3, dtype=float)]
    velocities = [velocity.copy()]
    stopped = np.linalg.norm(velocity[:2]) <= speed_tolerance
    stop_time = 0.0 if stopped else max_time

    for step_idx in range(max_steps):
        if stopped:
            break

        t = step_idx * dt
        step = min(dt, max_time - t)
        if step <= 0.0:
            break

        acceleration = sliding_deceleration_discrete_emperical(
            velocity[:2],
            cardinal_body_deceleration=cardinal_body_deceleration,
            diagonal_body_deceleration=diagonal_body_deceleration,
            diagonal_angle_half_width_degrees=diagonal_angle_half_width_degrees,
        )
        x, y, theta = states[-1]
        c = np.cos(theta)
        s = np.sin(theta)
        states.append(
            np.array(
                [
                    x + step * (c * velocity[0] - s * velocity[1]),
                    y + step * (s * velocity[0] + c * velocity[1]),
                    theta + step * velocity[2],
                ],
                dtype=float,
            )
        )

        translation = velocity[:2] + step * acceleration[:2]
        opposes_motion = velocity[:2] * acceleration[:2] < 0.0
        crossed_zero = np.signbit(velocity[:2]) != np.signbit(translation)
        translation[opposes_motion & crossed_zero] = 0.0
        if np.linalg.norm(translation) <= speed_tolerance:
            translation[:] = 0.0
        velocity = np.array([translation[0], translation[1], velocity[2]], dtype=float)
        velocities.append(velocity.copy())

        stopped = np.linalg.norm(velocity[:2]) <= speed_tolerance
        if stopped:
            stop_time = t + step

    return np.asarray(states), np.asarray(velocities), stop_time, stopped


def _independent_axis_brake_profile(initial_speed, deceleration, dt):
    """Ramp an initial axis speed to zero at a constant deceleration.

    Ports fieldset_generator_barebones/generate_rollouts.py's ``brake_profile``,
    omitting its response-time hold so braking starts immediately.
    """
    speed = abs(float(initial_speed))
    if speed == 0.0:
        return np.zeros(1, dtype=float)
    deceleration = abs(float(deceleration))
    if deceleration == 0.0:
        raise ValueError("deceleration must be nonzero")
    sign = np.sign(initial_speed)
    profile = []
    while speed >= deceleration * dt:
        speed -= deceleration * dt
        profile.append(sign * speed)
    profile.append(0.0)
    return np.asarray(profile, dtype=float)


def rollout_independent_axis_braking(body_velocity, brake_deceleration, dt):
    """Roll out pose and body velocity under independent per-axis braking.

    Ports fieldset_generator_barebones/generate_rollouts.py's ``rollout``,
    omitting its response-time hold so braking starts immediately. Unlike the
    roller friction-circle models above, this model brakes each body axis
    (vx, vy, yaw rate) independently rather than resolving forces through the
    wheels' roller-axis geometry.

    Args:
        body_velocity: Initial ``[vx, vy, yaw_rate]`` body velocity.
        brake_deceleration: Per-axis deceleration magnitudes ``[dx, dy, dyaw]``.
        dt: Integration step in seconds.
    Returns:
        ``(states, velocities)``. ``states`` are ``[x, y, theta]`` rows, and
        ``velocities`` are ``[vx, vy, yaw_rate]`` rows aligned with ``states``.
    """
    profiles = [
        _independent_axis_brake_profile(body_velocity[axis], brake_deceleration[axis], dt)
        for axis in range(3)
    ]
    max_length = max(len(profile) for profile in profiles)
    velocities = np.array(
        [np.pad(profile, (0, max_length - len(profile))) for profile in profiles], dtype=float
    ).T
    states = np.zeros((max_length + 1, 3), dtype=float)
    for index, (vx, vy, yaw_rate) in enumerate(velocities):
        x, y, theta = states[index]
        c = np.cos(theta)
        s = np.sin(theta)
        states[index + 1] = (x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, theta + yaw_rate * dt)
    return states, velocities


__all__ = [
    "individual_wheel_braking_deceleration",
    "sliding_deceleration_approx_model",
    "sliding_deceleration_coulomb_model",
    "sliding_deceleration_discrete_emperical",
    "rollout_sliding_deceleration_coulomb",
    "rollout_sliding_deceleration_approx",
    "rollout_discrete_empirical_deceleration",
    "rollout_independent_axis_braking",
]
