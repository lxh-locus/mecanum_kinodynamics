# Sliding Coulomb Approximate Rollout

## Scope

This document describes the execution stack and physics used when
`fieldset_generator_redux1/generate_rollouts_2.py` is invoked with:

```text
--strategy sliding-coulomb-approx
```

The description follows the current implementation from configuration loading
through one trajectory rollout. The later operation that combines many
trajectories and constructs a convex hull is intentionally left as a stub.

The implementation is a reduced planar model. Body velocity is represented as

```text
[vx, vy, omega]
```

where `vx` and `vy` are body-frame translational velocities and `omega` is body
yaw rate. Pose is represented as world-frame

```text
[x, y, theta]
```

with `theta` the world yaw angle.

## Entry Point and Runtime Arguments

The script parses the following relevant arguments:

```bash
python generate_rollouts_2.py \
    fieldset_config.json \
    --strategy sliding-coulomb-approx
```

The current defaults in `generate_rollouts_2.py` are:

- `--sampling-degree 3`
- `--sampling-method shrink`
- `--dt 0.005` seconds
- `--max-time 5.0` seconds
- the configured JSON file as the input configuration

The JSON contains `generator_params.integrator_dt`, currently `0.001`, but this
script does not read that field for this strategy. The command-line `--dt`
default of `0.005` controls both the response-delay integration and the
braking rollout unless an explicit `--dt` is supplied.

The selected strategy is checked against `STRATEGIES`, then translated to the
filename label `coulomb-approx` for generated output names.

## Configuration Loading

`load_config()` parses the JSON and requires:

- `robot_revision`
- `field_configs`

The selected robot revision supplies the robot footprint, velocity limits,
braking configuration, sensors, and robot subtype. Each field supplies its own
requested velocity box and footprint override, if any.

For the current `redux1/fieldset_config.json` example:

```text
robot brake_deceleration.x = -0.9 m/s^2
robot_response_time       = 0.05 s
sensor response times     = 0.153 s for the configured sensors
```

The response delay used by the rollout is therefore:

```text
delay = 0.05 + max(0.153, 0.153, 0.153) = 0.203 s
```

The delay is not a braking model parameter. It is a constant-velocity period
representing the robot and sensor response before braking begins.

## Physical Parameters

`main()` constructs a default `MecanumPhysicsParams` object and passes it to
`Mecanum`. The important default parameters are:

```text
half wheelbase width   = 0.2405 m
half wheelbase length  = 0.25 m
wheel radius           = 3 * 0.0254 m
body mass              = 100 kg
body yaw inertia       = 8.174 kg m^2
```

The default roller-axis directions are normalized by the sliding model. Their
signs are:

```text
front-left   ( 1, -1)
front-right  ( 1,  1)
rear-left    ( 1,  1)
rear-right   ( 1, -1)
```

After normalization, each is a diagonal direction at 45 degrees in the body
frame.

## Kinematic Velocity Envelope

Before any braking rollout, the script generates admissible initial body
velocities for each field.

### Field velocity ranges

`velocity_ranges_from_field()` extracts the field's requested ranges:

```text
(vx_min, vx_max)
(vy_min, vy_max)
(omega_min, omega_max)
```

For example, the `medium` field has its own asymmetric or symmetric limits;
the field values, rather than the robot-level velocity limits, define the box
used for that field.

### Wheel-speed limit

`max_wheel_velocity_from_config()` converts the robot-level maximum x velocity
to a wheel angular velocity:

$$
\omega_{wheel,max} = \frac{v_{x,max}}{r}
$$

where $r$ is the wheel radius. The value is positive and is used to impose the
mecanum wheel-speed constraints.

### Truncated kinematic polytope

`_polytope_faces()` constructs half-spaces in
$[v_x, v_y, \omega]$ space. The mecanum constraints are equivalent to bounds
on four signed combinations of body velocity:

```text
vx - vy - (L + W) omega
vx + vy + (L + W) omega
vx + vy - (L + W) omega
vx - vy + (L + W) omega
```

Each combination is bounded in magnitude by $r\omega_{wheel,max}$. The six
corresponding inequalities are augmented with the six requested field-box
inequalities. Their intersection is a convex, truncated velocity polytope.

All candidate vertices are found by intersecting triples of half-space planes,
solving the resulting 3-by-3 systems, and retaining feasible points. Active
faces are then identified from those vertices and ordered as polygons.

### Boundary sampling

With the default `sampling-method shrink`,
`sample_boundary_velocities()` returns:

- every polytope vertex;
- dyadic points along face edges;
- dyadically shrunk contours inside each face.

With sampling degree 3, the surface is therefore sampled at multiple levels,
not just at the vertices. The result is a finite list of body velocities. Each
sample becomes an independent initial condition for the braking rollout.

## Per-Sample Strategy Dispatch

For each sampled velocity, `plot_field_rollouts()` calls:

```python
strategy_rollout(
    velocity,
    robot_revision,
    generator_params,
    params,
    delay,
    dt,
    max_time,
    "sliding-coulomb-approx",
)
```

The selected branch performs three main operations.

### 1. Calibrate per-wheel braking

The robot-level x braking magnitude is read as:

```python
body_x_braking = abs(float(brake["x"]))
```

For the current configuration this is `0.9 m/s^2`.

The approximate model calls:

```python
wheel_braking = individual_wheel_braking_deceleration(
    body_x_braking,
    params=params,
    model="approx",
)
```

For the approximate and velocity-axis models, the current calibration divides
by the number of wheels:

$$
 a_{wheel} = \frac{a_{x,body}}{4}
$$

Thus `0.9 m/s^2` produces `0.225 m/s^2` per wheel. This is an acceleration-like
quantity, not a torque or a directly measured tire force.

The calibration is chosen so that pure positive body-x motion produces the
requested aggregate body-x braking in the symmetric four-wheel case. It is not
an independent physical identification of each wheel's normal force or motor
limit.

### 2. Select the approximation width

The generator reads:

```python
angle_width = float(
    generator_params.get("diagonal_angle_half_width_degrees", 10.0)
)
```

The current JSON does not define this key, so the default is 10 degrees. This
width controls the transition from no resisted roller-axis component to the
full per-wheel braking value.

### 3. Call the shared rollout integrator

The strategy dispatch passes the approximate deceleration function and its
parameters into `rollout_sliding_deceleration()`:

```python
rollout_sliding_deceleration(
    initial_velocity,
    wheel_braking,
    params=params,
    dt=dt,
    max_time=max_time,
    deceleration_fn=sliding_deceleration_approx_model,
    deceleration_kwargs={
        "wheel_braking_deceleration": wheel_braking,
        "params": params,
        "diagonal_angle_half_width_degrees": angle_width,
    },
)
```

The returned values are:

```text
braking_poses, braking_velocities, stop_time, stopped
```

The strategy then appends the response-delay trajectory and returns the full
trajectory together with the stop metadata and calibrated wheel value.

## Approximate Sliding Physics

### Wheel contact geometry

For each wheel, `_sliding_contact_data()` constructs its body-frame position:

```text
front-left   (+L, +W)
front-right  (+L, -W)
rear-left    (-L, +W)
rear-right   (-L, -W)
```

For body velocity $[v_x, v_y, \omega]$, the local contact velocity at wheel
position $(x_i, y_i)$ is:

$$
\mathbf{v}_{c,i} =
\begin{bmatrix}
 v_x - \omega y_i \\
 v_y + \omega x_i
\end{bmatrix}
$$

Each non-negligible contact velocity is normalized:

$$
\hat{\mathbf{v}}_{c,i} =
\frac{\mathbf{v}_{c,i}}{\|\mathbf{v}_{c,i}\|}
$$

The roller direction is normalized as
$\hat{\mathbf{r}}_i$. The signed roller-axis projection is:

$$
 p_i = \hat{\mathbf{r}}_i \cdot \hat{\mathbf{v}}_{c,i}
$$

The projection is clipped to $[-1,1]$. A contact whose velocity is below the
`tolerance` threshold receives a zero contact direction.

If both translational speed and yaw rate are below tolerance, contact data is
`None` and the model returns exactly zero acceleration.

### Smooth alignment weighting

The approximate model uses the magnitude $|p_i|$ to determine how much each
wheel contributes. It constructs a smooth bump function over:

$$
[0, \sin(\delta)]
$$

where $\delta$ is the configured diagonal half-width, 10 degrees by default.
The bump function is 0 at or below the lower endpoint and 1 at or above the
upper endpoint, with a smooth transition inside. One form of a bump function is as follows:

$$
b(x) = \left( \frac{x-a}{b-a}\right)
$$
where $a$ is the start of the interval, 0 in this case, and $b$ is the end, $\sin(\delta)$. $g(x)$ is:

$$
g(x) = \frac{f(x)}{f(x) + x(1-x)}
$$
where
$$
f(x)=\begin{cases}
 \text{exp}(-\frac{1}{x}) & \text{if } x \gt 0  \\
 0 & x \leq 0 
\end{cases}
$$

Let $b(|p_i|)$ denote this weighting. The scalar axis braking response is:

$$
 a_{axis,i} = -a_{wheel}\,b(|p_i|)
$$

The resulting contact acceleration is applied opposite the local contact
velocity direction:

$$
\mathbf{a}_{c,i} = a_{axis,i}\hat{\mathbf{v}}_{c,i}
$$

Thus a wheel contributes no braking when its contact direction is perpendicular
to its roller axis, and its full configured braking magnitude when the roller
axis is sufficiently aligned. The response changes continuously as heading
changes, unlike the fixed-axis Coulomb model's sign switch.

The implementation uses the same contact acceleration to produce both body
translation and yaw acceleration:

$$
\begin{bmatrix}
 a_x \\
 a_y
\end{bmatrix}
= \sum_i \mathbf{a}_{c,i}
$$

$$
\alpha =
\frac{m}{I_z}
\sum_i (x_i a_{c,i,y} - y_i a_{c,i,x})
$$

The returned acceleration is `[ax, ay, alpha]`. The factor $m/I_z$ appears in
the yaw equation because the reduced helper receives acceleration-like contact
contributions and converts their moment into angular acceleration.

For the symmetric diagonal layout and translational motion, the approximation
is intended to produce a smooth, approximately isotropic opposing response.
With yaw, each wheel sees a different local contact velocity, so the summed
response can include translation/yaw coupling.

## Time Integration and Frame Conventions

`rollout_sliding_deceleration()` stores states beginning at:

```text
pose_0     = [0, 0, 0]
velocity_0 = initial_velocity
```

At each timestep, it first evaluates the approximate contact model using the
current body-frame velocity.

### Rotating body-frame velocity derivative

The contact model returns force-derived body acceleration. Because the velocity
components are expressed in a rotating body frame, the rollout adds the
transport terms:

$$
\dot v_x = a_x + \omega v_y
$$

$$
\dot v_y = a_y - \omega v_x
$$

$$
\dot\omega = \alpha
$$

In code this is:

```python
body_frame_acceleration[0] += velocity[2] * velocity[1]
body_frame_acceleration[1] -= velocity[2] * velocity[0]
```

These terms do not represent an additional braking force. They convert the
physical acceleration into the derivative of body-frame velocity.

### World-frame pose update

The current body velocity is mapped to world position rate using the current
yaw:

$$
\dot x = \cos(\theta)v_x - \sin(\theta)v_y
$$

$$
\dot y = \sin(\theta)v_x + \cos(\theta)v_y
$$

$$
\dot\theta = \omega
$$

The implementation uses an explicit Euler step of size `step`, normally `dt`:

```python
x_next     = x     + step * x_dot
y_next     = y     + step * y_dot
theta_next = theta + step * omega
```

The pose is advanced using the velocity at the beginning of the step. The
velocity is then advanced using the body-frame acceleration and the same step.

### Zero crossing and stopping

The velocity update detects a component crossing zero while acceleration
opposes that component. Such components are clamped to zero rather than being
allowed to numerically reverse direction under Coulomb braking.

The rollout reports `stopped=True` when both conditions hold:

```text
norm([vx, vy]) <= 1e-4 m/s
abs(omega)     <= 1e-4 rad/s
```

If stopping does not occur before `max_time`, the rollout returns
`stopped=False` and `stop_time=max_time`.

## Appending Sensor/Robot Response Delay

The braking integrator starts from pose `[0, 0, 0]`, so its output is local to
the beginning of braking. `append_response_delay()` first integrates the
initial velocity for the delay duration with no braking.

For each delay step, it uses the same body-to-world mapping:

$$
\begin{bmatrix}\dot x\\\dot y\end{bmatrix}
= R(\theta)\begin{bmatrix}v_x\\v_y\end{bmatrix}
$$

where $R(\theta)$ is the planar rotation matrix. Yaw advances as
$\theta \leftarrow \theta + \omega\,dt$.

The final delay pose becomes the origin for the braking trajectory. Each
braking pose $(x_b,y_b,\theta_b)$ is transformed into the world frame using
that delay endpoint $(x_d,y_d,\theta_d)$:

$$
\begin{bmatrix}x\\y\end{bmatrix}
=
\begin{bmatrix}x_d\\y_d\end{bmatrix}
+ R(\theta_d)\begin{bmatrix}x_b\\y_b\end{bmatrix}
$$

$$
\theta = \theta_d + \theta_b
$$

The initial duplicate braking pose is omitted when concatenating, so the final
trajectory is one continuous pose history: constant-velocity response delay,
then approximate sliding braking.

One implementation detail is that `integrate_response_delay()` uses
`int(duration // dt)`. A fractional remainder shorter than one time step is
not integrated, so the actual represented delay can be slightly shorter than
the requested delay.

## Plotting and Optional Archive Output

For each full trajectory, `plot_field_rollouts()`:

1. plots world x/y trajectory;
2. colors stopped trajectories red and unfinished trajectories orange;
3. selects the final pose;
4. draws the configured padded footprint at that final pose;
5. adds the initial footprint at the world origin;
6. writes a PNG to `rollout_output` by default.

If `--save-rollout-data` is supplied, `save_rollout_data()` also pads the
variable-length pose histories with `NaN` and writes a compressed `.npz` archive
containing initial velocities, poses, lengths, stop times, stopped flags, time
step, response time, wheel braking value, and footprint metadata.

The filename label for this strategy is `coulomb-approx`, for example:

```text
<field>_coulomb-approx_rollouts.png
<field>_coulomb-approx_rollouts.npz
```

## Important Modeling Limitations

The approximate rollout should be read as a validation and planning model, not
as a full wheel/motor/contact simulation. It assumes or omits:

- equal wheel load sharing;
- a single calibrated per-wheel braking value;
- explicit wheel torque and current limits;
- normal-load transfer;
- slip ratio and wheel spin dynamics;
- roller bearing friction;
- floor-dependent friction;
- compliance and contact-patch deformation;
- measurement uncertainty in response delay and braking values.

The initial velocities are sampled from a kinematic wheel-speed envelope, while
braking is governed by the reduced sliding model. Therefore a generated
trajectory is only as physically credible as the calibration and the validity
of those assumptions across the sampled headings, yaw rates, payloads, and
surfaces.
