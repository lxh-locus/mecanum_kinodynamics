"""Plot the 3**4 fixed-axis wheel-braking combinations in body acceleration space.

This is the independent-wheel combinatorial set, not a claim that every
combination can arise from a single rigid-body sliding velocity. Panels separate
yaw magnitudes so coincident xy arrows remain distinguishable.
"""
import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

try:
    from .mecanum_physics import MecanumPhysicsParams
    from .mecanum_sliding import (
        _body_acceleration_from_contacts,
        _sliding_contact_data,
        individual_wheel_braking_deceleration,
    )
except ImportError:
    from mecanum_physics import MecanumPhysicsParams
    from mecanum_sliding import (
        _body_acceleration_from_contacts,
        _sliding_contact_data,
        individual_wheel_braking_deceleration,
    )


def enumerate_fixed_axis_vectors(wheel_braking_deceleration, params=None):
    """Return wheel signs (81, 4) and body accelerations (81, 3).

    Wheel order is front-left, front-right, rear-left, rear-right. Each sign
    multiplies the normalized roller axis and the per-wheel braking value.
    """
    if not np.isfinite(wheel_braking_deceleration) or wheel_braking_deceleration <= 0.0:
        raise ValueError("wheel_braking_deceleration must be finite and positive")
    if params is None:
        params = MecanumPhysicsParams()
    wheel_positions, roller_directions, _, _ = _sliding_contact_data(
        [1.0, 0.0, 0.0], params, 1e-9
    )
    wheel_signs = np.array(list(product((-1, 0, 1), repeat=4)), dtype=int)
    accelerations = np.array([
        _body_acceleration_from_contacts(
            wheel_braking_deceleration * signs[:, np.newaxis] * roller_directions,
            wheel_positions,
            params,
        )
        for signs in wheel_signs
    ])
    return wheel_signs, accelerations


def plot_fixed_axis_vectors(wheel_braking_deceleration=1.0, params=None):
    """Return a figure and axes with xy arrows grouped by absolute yaw acceleration.

    Identical xy vectors within a yaw-magnitude panel are drawn once, with
    endpoint labels giving their wheel-combination multiplicity.
    """
    _, accelerations = enumerate_fixed_axis_vectors(wheel_braking_deceleration, params)
    yaw_magnitudes = np.round(np.abs(accelerations[:, 2]), decimals=12)
    levels = np.unique(yaw_magnitudes)
    columns = min(3, len(levels))
    rows = (len(levels) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(5 * columns, 4.5 * rows),
        squeeze=False, sharex=True, sharey=True, layout="constrained",
    )
    limit = 1.25 * np.max(np.abs(accelerations[:, :2]))
    for axis, level in zip(axes.flat, levels):
        selected = accelerations[yaw_magnitudes == level, :2]
        vectors, counts = np.unique(np.round(selected, decimals=12), axis=0, return_counts=True)
        axis.quiver(
            np.zeros(len(vectors)), np.zeros(len(vectors)),
            vectors[:, 0], vectors[:, 1], color="black", angles="xy",
            scale_units="xy", scale=1.0, width=0.006,
        )
        stationary = np.all(vectors == 0.0, axis=1)
        if np.any(stationary):
            axis.scatter([0.0], [0.0], color="black", zorder=3)
        for vector, count in zip(vectors, counts):
            axis.annotate(
                f"{count}x", vector, xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
        axis.set_title(f"|alpha| = {level:.4g} rad/s^2 ({len(selected)} combinations)")
        axis.set_xlabel("body ax [m/s^2]")
        axis.set_ylabel("body ay [m/s^2]")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
    for axis in list(axes.flat)[len(levels):]:
        axis.set_visible(False)
    figure.suptitle(
        "Fixed-axis Coulomb braking: all 81 wheel combinations\n"
        f"per-wheel a = {wheel_braking_deceleration:g} m/s^2; "
        "endpoint labels count coincident combinations"
    )
    return figure, axes


def main():
    """Display or save the combinatorial fixed-axis deceleration plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    braking = parser.add_mutually_exclusive_group()
    braking.add_argument(
        "--wheel-braking-deceleration", type=float, default=1.0,
        help="Per-wheel axis braking magnitude in m/s^2 (default: 1).",
    )
    braking.add_argument(
        "--max-body-x-deceleration", type=float,
        help="Instead calibrate the wheel magnitude from body-x braking in m/s^2.",
    )
    parser.add_argument("--output", help="Save the figure to this image path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window.")
    args = parser.parse_args()
    wheel_braking = args.wheel_braking_deceleration
    if args.max_body_x_deceleration is not None:
        wheel_braking = individual_wheel_braking_deceleration(args.max_body_x_deceleration)
    figure, _ = plot_fixed_axis_vectors(wheel_braking)
    if args.output:
        figure.savefig(args.output, dpi=180)
    if not args.no_show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()