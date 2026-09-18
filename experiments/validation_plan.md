# Physical Model Validation Plan

This plan validates the mecanum sliding-braking model in layers: kinematics, mathematical invariants, numerical behavior, and agreement with physical measurements.

## 1. Validate Kinematics First

Before judging braking behavior:

- Generate random body velocities.
- Apply `inverse_kinematics()`.
- Feed the wheel speeds into `forward_kinematics()`.
- Confirm that the original body velocity is recovered.
- Verify that the wheel compatibility residual is near zero.

This isolates geometry and roller-sign errors from braking-model errors.

## 2. Check Physical Invariants

For random body velocities, test every deceleration model:

- Zero velocity produces zero acceleration.
- Accelerations are finite and bounded.
- Braking never adds kinetic energy.

For body velocity $v = (v_x, v_y, omega)$ and acceleration $a = (a_x, a_y, alpha)$, the kinetic-energy rate should satisfy:

$$
P = m(v_x a_x + v_y a_y) + I_z \omega \alpha <= 0
$$

An equivalent contact-level check is:

$$
P = \sum_i v_{contact,i} \cdot a_{contact,i} <= 0
$$

Any positive value indicates that the model is adding energy.

Also check per-wheel limits:

- Fixed-axis Coulomb: resisted-axis acceleration magnitude is no greater than the wheel limit.
- Approximate model: the same bound holds, with smooth variation as heading changes.
- Velocity-axis model: contact acceleration magnitude is no greater than the wheel limit.

## 3. Test Predictable Cases

For pure body-$x$ velocity, verify that calibration produces the requested total deceleration:

```python
wheel_braking = individual_wheel_braking_deceleration(
    0.9,
    model="fixed-axis",
)
acceleration = sliding_deceleration_coulomb_fixed_axis(
    [1.0, 0.0, 0.0],
    wheel_braking,
)
```

Check that `acceleration[0]` is approximately `-0.9`.

Repeat for:

- Pure positive and negative body $x$ velocity
- Pure positive and negative body $y$ velocity
- Diagonal translation
- Pure positive and negative yaw
- Translation combined with yaw
- Velocities near roller-axis alignment boundaries

Expected behavior:

- The fixed-axis model has discontinuities when a roller projection crosses zero.
- The approximate model varies smoothly.
- The velocity-axis model opposes each local contact velocity.

## 4. Check Numerical Convergence

Run identical initial conditions with several integration steps, for example:

```text
dt = 0.02
dt = 0.01
dt = 0.005
dt = 0.001
```

Compare:

- Stopping time
- Stopping distance
- Final pose
- Maximum kinetic-energy increase

These values should converge as `dt` decreases. Persistent disagreement indicates a problem with zero-crossing handling, discontinuous Coulomb sign changes, or integration accuracy.

## 5. Compare Against Physical Data

Record repeated braking trials over a grid of:

- Initial $v_x$, $v_y$, and $omega$
- Travel headings
- Payload masses
- Floor surfaces
- Braking commands or motor-current limits

Measure body pose and velocity, then compare:

- Initial deceleration
- Stopping time
- Stopping distance
- Final yaw
- Translation/yaw coupling
- Direction-dependent variation

Fit the wheel-braking parameter using only part of the trials, then evaluate the remaining trials without retuning. This separates calibration from validation.

## 6. Record Model Limitations

The current reduced model assumes equal wheel load sharing and represents each wheel's braking contribution as an acceleration-like quantity. It does not explicitly model:

- Normal-load transfer
- Individual wheel torque or current limits
- Wheel slip ratio
- Roller bearing friction
- Wheel spin inertia during sliding
- Floor-dependent friction
- Compliance or contact-patch deformation

A useful validation criterion is whether one calibrated wheel-braking value predicts different headings, yaw rates, payloads, and surfaces. Matching only pure forward braking demonstrates calibration, not broad physical validation.
