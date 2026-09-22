"""Randomized validation of the mecanum kinematics round trip.

This is the first layer of the validation plan: inverse kinematics must produce
compatible wheel speeds, and forward kinematics must recover the input body
velocity.
"""

import argparse

import numpy as np

try:
    from .mecanum_physics import (
        MecanumPhysicsParams,
        forward_kinematics,
        inverse_kinematics,
        wheel_constraint_violation,
    )
except ImportError:
    from mecanum_physics import (
        MecanumPhysicsParams,
        forward_kinematics,
        inverse_kinematics,
        wheel_constraint_violation,
    )


def validate_kinematics(
    sample_count=1000,
    seed=0,
    velocity_scales=(2.0, 2.0, 3.0),
    params=MecanumPhysicsParams(),
    atol=1e-12,
    rtol=1e-12,
):
    """Validate inverse/forward kinematics on seeded random body velocities.

    Returns a summary containing the largest body-velocity round-trip error and
    wheel compatibility residual. Raises ``AssertionError`` on failure.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if len(velocity_scales) != 3 or np.any(np.asarray(velocity_scales) <= 0.0):
        raise ValueError("velocity_scales must contain three positive values")

    rng = np.random.default_rng(seed)
    body_velocities = rng.uniform(
        low=-np.asarray(velocity_scales, dtype=float),
        high=np.asarray(velocity_scales, dtype=float),
        size=(sample_count, 3),
    )

    max_body_error = 0.0
    max_constraint_residual = 0.0
    for body_velocity in body_velocities:
        wheel_velocity = inverse_kinematics(body_velocity, params=params)
        recovered_body_velocity = forward_kinematics(wheel_velocity, params=params)
        constraint_residual = wheel_constraint_violation(wheel_velocity)

        np.testing.assert_allclose(
            recovered_body_velocity,
            body_velocity,
            atol=atol,
            rtol=rtol,
            err_msg=f"Kinematics round trip failed for {body_velocity}",
        )
        np.testing.assert_allclose(
            constraint_residual,
            0.0,
            atol=atol,
            rtol=rtol,
            err_msg=f"Wheel compatibility failed for {body_velocity}",
        )

        max_body_error = max(
            max_body_error,
            float(np.max(np.abs(recovered_body_velocity - body_velocity))),
        )
        max_constraint_residual = max(max_constraint_residual, abs(float(constraint_residual)))

    return {
        "sample_count": sample_count,
        "seed": seed,
        "max_body_error": max_body_error,
        "max_constraint_residual": max_constraint_residual,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = validate_kinematics(sample_count=args.sample_count, seed=args.seed)
    print(
        f"Validated {summary['sample_count']} samples (seed={summary['seed']}): "
        f"max body error={summary['max_body_error']:.3e}, "
        f"max wheel residual={summary['max_constraint_residual']:.3e}"
    )


if __name__ == "__main__":
    main()