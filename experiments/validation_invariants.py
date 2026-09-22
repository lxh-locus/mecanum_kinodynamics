"""Randomized validation of sliding-braking physical invariants.

This is the second layer of the validation plan. It checks the fixed-axis,
approximate, and velocity-axis roller braking models for zero response at rest,
finite bounded accelerations, non-increasing kinetic energy, and per-wheel
braking limits.
"""

import argparse

import numpy as np

try:
    from .mecanum_physics import MecanumPhysicsParams
    from .mecanum_sliding import (
        _body_acceleration_from_contacts,
        _bump_function_constructor,
        _sliding_contact_data,
        individual_wheel_braking_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_fixed_axis,
        sliding_deceleration_coulomb_velocity_axis,
    )
except ImportError:
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        _body_acceleration_from_contacts,
        _bump_function_constructor,
        _sliding_contact_data,
        individual_wheel_braking_deceleration,
        sliding_deceleration_approx_model,
        sliding_deceleration_coulomb_fixed_axis,
        sliding_deceleration_coulomb_velocity_axis,
    )


def _contact_accelerations(model_name, body_velocity, wheel_braking, params, tolerance):
    """Return per-wheel contact accelerations for a roller model."""
    contact_data = _sliding_contact_data(body_velocity, params, tolerance)
    if contact_data is None:
        return np.zeros((4, 2), dtype=float), None

    _, roller_directions, contact_directions, rolling_projection = contact_data
    active = ~np.isclose(rolling_projection, 0.0, atol=tolerance)

    if model_name == "fixed-axis":
        axis_acceleration = -wheel_braking * np.sign(rolling_projection)
        axis_acceleration[~active] = 0.0
        return axis_acceleration[:, np.newaxis] * roller_directions, contact_data

    if model_name == "approx":
        bump_func = _bump_function_constructor(0.0, np.sin(np.radians(10.0)))
        bump_values = np.array(
            [0.0 if value <= 0.0 else bump_func(value) for value in np.abs(rolling_projection)]
        )
        axis_acceleration = -wheel_braking * bump_values
        return axis_acceleration[:, np.newaxis] * contact_directions, contact_data

    if model_name == "velocity-axis":
        contact_accelerations = -wheel_braking * contact_directions
        contact_accelerations[~active] = 0.0
        return contact_accelerations, contact_data

    raise ValueError(f"Unknown model: {model_name}")


def _model_response(model_name, body_velocity, wheel_braking, params):
    functions = {
        "fixed-axis": sliding_deceleration_coulomb_fixed_axis,
        "approx": sliding_deceleration_approx_model,
        "velocity-axis": sliding_deceleration_coulomb_velocity_axis,
    }
    return functions[model_name](body_velocity, wheel_braking, params=params)


def validate_invariants(
    sample_count=1000,
    seed=0,
    params=MecanumPhysicsParams(),
    max_body_x_deceleration=1.5,
    tolerance=1e-9,
    numerical_tolerance=1e-12,
):
    """Validate braking invariants on seeded random body velocities.

    Returns a summary containing the largest observed acceleration and energy
    rate. Raises ``AssertionError`` if a model violates an invariant.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if max_body_x_deceleration <= 0.0:
        raise ValueError("max_body_x_deceleration must be positive")

    rng = np.random.default_rng(seed)
    body_velocities = rng.uniform(-1.0, 1.0, size=(sample_count, 3))
    body_velocities[:, :2] *= 2.0
    body_velocities[:, 2] *= 3.0
    body_velocities[0, :] = 0.0

    model_names = ("fixed-axis", "approx", "velocity-axis")
    wheel_braking = {
        name: individual_wheel_braking_deceleration(
            max_body_x_deceleration, params=params, model=name
        )
        for name in model_names
    }
    max_acceleration = {name: 0.0 for name in model_names}
    max_energy_rate = {name: -np.inf for name in model_names}
    max_wheel_limit = {name: 0.0 for name in model_names}

    wheel_positions = np.array(
        [
            [params.wb_hlength, params.wb_hwidth],
            [params.wb_hlength, -params.wb_hwidth],
            [-params.wb_hlength, params.wb_hwidth],
            [-params.wb_hlength, -params.wb_hwidth],
        ],
        dtype=float,
    )
    acceleration_bound = 4.0 * max(wheel_braking.values())
    yaw_acceleration_bound = (
        params.body_mass
        * np.sum(np.linalg.norm(wheel_positions, axis=1))
        * max(wheel_braking.values())
        / params.body_yaw_inertia
    )

    for body_velocity in body_velocities:
        for model_name in model_names:
            braking = wheel_braking[model_name]
            acceleration = _model_response(model_name, body_velocity, braking, params)
            contact_accelerations, contact_data = _contact_accelerations(
                model_name, body_velocity, braking, params, tolerance
            )

            assert np.isfinite(acceleration).all()
            assert np.linalg.norm(acceleration[:2]) <= acceleration_bound + numerical_tolerance
            assert abs(acceleration[2]) <= yaw_acceleration_bound + numerical_tolerance

            if contact_data is None:
                np.testing.assert_allclose(acceleration, 0.0, atol=numerical_tolerance)
                np.testing.assert_allclose(contact_accelerations, 0.0, atol=numerical_tolerance)
                continue

            actual_positions, roller_directions, contact_directions, rolling_projection = contact_data
            np.testing.assert_allclose(
                acceleration,
                _body_acceleration_from_contacts(
                    contact_accelerations, actual_positions, params
                ),
                atol=numerical_tolerance,
            )

            translation = body_velocity[:2]
            yaw_rate = body_velocity[2]
            contact_velocities = np.column_stack(
                [
                    translation[0] - yaw_rate * actual_positions[:, 1],
                    translation[1] + yaw_rate * actual_positions[:, 0],
                ]
            )
            body_power = (
                params.body_mass * np.dot(body_velocity[:2], acceleration[:2])
                + params.body_yaw_inertia * body_velocity[2] * acceleration[2]
            )
            contact_power = params.body_mass * np.sum(
                contact_velocities * contact_accelerations
            )
            assert body_power <= numerical_tolerance
            assert contact_power <= numerical_tolerance
            np.testing.assert_allclose(body_power, contact_power, atol=numerical_tolerance)

            if model_name == "velocity-axis":
                wheel_usage = np.linalg.norm(contact_accelerations, axis=1)
            else:
                wheel_usage = np.abs(
                    np.sum(contact_accelerations * roller_directions, axis=1)
                )
            assert np.all(wheel_usage <= braking + numerical_tolerance)

            max_acceleration[model_name] = max(
                max_acceleration[model_name], float(np.linalg.norm(acceleration))
            )
            max_energy_rate[model_name] = max(max_energy_rate[model_name], float(body_power))
            max_wheel_limit[model_name] = max(
                max_wheel_limit[model_name], float(np.max(wheel_usage))
            )

    return {
        "sample_count": sample_count,
        "seed": seed,
        "max_acceleration": max_acceleration,
        "max_energy_rate": max_energy_rate,
        "max_wheel_usage": max_wheel_limit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = validate_invariants(sample_count=args.sample_count, seed=args.seed)
    for model_name in ("fixed-axis", "approx", "velocity-axis"):
        print(
            f"{model_name}: max acceleration={summary['max_acceleration'][model_name]:.3e}, "
            f"max energy rate={summary['max_energy_rate'][model_name]:.3e}, "
            f"max wheel usage={summary['max_wheel_usage'][model_name]:.3e}"
        )


if __name__ == "__main__":
    main()