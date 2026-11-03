# Fixed-Axis Sliding Braking on a Mecanum Robot

This note explains the reduced Coulomb model implemented by
[`sliding_deceleration_coulomb_fixed_axis`](experiments/mecanum_sliding.py#L246),
why its initial body acceleration can lie along only the dominant body axis,
and what the `DiagFast2` recording does and does not show.

## The roller constraint

A Mecanum wheel is not a conventional tire that can generate arbitrary
horizontal traction. Each small roller spins freely about its own axle. The
vectors in `params.roller_directions` point along those roller axles, projected
into the robot's body plane. A roller's free rolling motion is perpendicular
to its axle: spinning the roller moves its contact surface only in directions
perpendicular to that axle. It cannot cancel relative contact motion along the
axle. If $\hat{\mathbf{r}}$ is the axle direction and $\mathbf{u}$ is the
contact velocity, decompose the planar motion as

$$
\mathbf{u}=(\mathbf{u}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}
+\left[\mathbf{u}-(\mathbf{u}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}\right].
$$

The second term is perpendicular to the axle and can be accommodated by
roller rotation. The first term is along the axle and is the constrained slip
component. In ideal no-slip contact, the constraint is
$\mathbf{u}\cdot\hat{\mathbf{r}}=0$. During sliding, the relative slip along
the axle is $s\hat{\mathbf{r}}$, where $s=\mathbf{u}\cdot\hat{\mathbf{r}}$.
Coulomb friction opposes that slip, so its constrained-direction component is

$$
\mathbf{F}_{\mathrm{friction},\parallel}
=-\mu N\,\operatorname{sgn}(s)\,\hat{\mathbf{r}},
$$

with friction coefficient $\mu$ and normal load $N$. This explains why that
component of the ground reaction points along the axle: the slip being
resisted is along the axle. For sticking contact, the corresponding ideal
constraint reaction also acts along $\hat{\mathbf{r}}$; its magnitude is set
by the force needed to maintain the constraint, up to the available friction.
The model projects contact velocity onto `roller_directions` and applies its
effective per-wheel braking acceleration along those same directions.

The reference in [`zeidis2019.txt`](zeidis2019.txt) states this ideal contact
condition as $\mathbf{V}_P\cdot\boldsymbol{\gamma}=0$, where
$\boldsymbol{\gamma}$ is the roller-axis direction (Eq. 1). This expresses the
same no-slip constraint. The fixed-axis sliding model extends it by assuming
no braking along the perpendicular free-rolling direction and Coulomb braking
along the axle direction.

This is an idealization, not a claim that a real wheel has exactly zero
resistance in its free direction. Roller-bearing drag, deformation, gaps
between rollers, contact-patch effects, and uneven normal loads all add
behavior this model omits.

## What the function computes

For wheel $i$, with body-frame position $\mathbf{p}_i=(p_{x,i},p_{y,i})$ and
body velocity $(v_x,v_y,\omega)$, its local contact velocity is

$$
\mathbf{u}_i=
\begin{pmatrix}
v_x-\omega p_{y,i}\\
v_y+\omega p_{x,i}
\end{pmatrix}.
$$

Let $\hat{\mathbf{r}}_i$ be that wheel's normalized constrained direction.
The signed contact motion along it is

$$
s_i=\mathbf{u}_i\cdot\hat{\mathbf{r}}_i.
$$

The fixed-axis Coulomb rule assigns each slipping wheel a full braking
contribution with the opposite sign:

$$
\mathbf{a}_i=-b_w\,\operatorname{sgn}(s_i)\,\hat{\mathbf{r}}_i,
$$

where $b_w$ is the per-wheel braking value passed to the function. The code
sets the contribution to zero when $s_i$ is within its tolerance of zero.
This is a hard Coulomb switch: the magnitude does not grow gradually with
slip speed, and the response changes abruptly when a projection changes sign.
The resulting body translation is the sum of the four per-wheel contributions;
their moments about the center also contribute to yaw acceleration. See the
projection and summation in [`mecanum_sliding.py`](experiments/mecanum_sliding.py#L158)
and the fixed-axis sign rule in
[`mecanum_sliding.py`](experiments/mecanum_sliding.py#L304).

## Relation to friction-circle intuition

Yes: friction opposes relative slip at the ground contact. The important
distinction is which relative motion is slipping. The wheel-center contact
velocity $\mathbf{u}_i$ is not, by itself, the residual slip of the roller
against the ground. In the ideal free-roller picture, the roller spins to
accommodate the component perpendicular to its axle. The remaining slip is
$s_i\hat{\mathbf{r}}_i$, so kinetic friction opposes that residual slip along
the axle. This is what the fixed-axis model assumes.

A conventional isotropic friction circle instead bounds the combined planar
force, for example $\|\mathbf{F}\|\leq\mu N$. Once sliding, an isotropic
Coulomb force points opposite the full residual slip vector. A Mecanum roller
is anisotropic because its passive spin provides a low-resistance direction;
the friction-circle bound alone does not specify how force should vary between
that direction and the constrained axle direction. Also, ideal Coulomb kinetic
friction does not gradually increase with slip angle: it has full magnitude
for any nonzero slip and changes direction with the slip.

The separate `sliding_deceleration_approx_model` makes a smooth, phenomenological
choice between these limits. It uses

$$
c_i=|\hat{\mathbf{u}}_i\cdot\hat{\mathbf{r}}_i|,
\qquad
\mathbf{a}^{\mathrm{approx}}_i=-b_w B(c_i)\hat{\mathbf{u}}_i,
$$

where $B$ is the code's bump function: zero for perpendicular contact motion,
then smoothly increasing to one as the motion aligns with the axle. Since the
response is along $-\hat{\mathbf{u}}_i$, this approximation does oppose the
full local contact-velocity direction. It is not the same law as the fixed-axis
model, and its smooth angle-dependent weighting is a modeling choice rather
than a result implied by the standard friction-circle inequality.

## Why braking can point along the major body axis

With zero yaw rate, the default directions are two copies each of

$$
\hat{\mathbf{r}}_- = \frac{1}{\sqrt{2}}(1,-1),
\qquad
\hat{\mathbf{r}}_+ = \frac{1}{\sqrt{2}}(1,1).
$$

Their projections are $(v_x-v_y)/\sqrt{2}$ and $(v_x+v_y)/\sqrt{2}$,
respectively. If $v_x>0$ and $|v_y|<v_x$, both projections are positive. Each
wheel therefore brakes along $-\hat{\mathbf{r}}_i$. The two $y$ components
cancel by symmetry, while all four $x$ components point backward:

$$
\sum_{i=1}^4 \mathbf{a}_i
= -4b_w\frac{1}{\sqrt{2}}(1,0)
= (-2\sqrt{2}\,b_w,0).
$$

So the individual wheel contributions are diagonal; it is their symmetric sum
that points along body $-x$. The robot can still have nonzero $v_y$ at that
instant. In this hard-switch model the lateral speed is not immediately
reduced by a lateral resultant, even though the robot is moving diagonally.
When $|v_y|>|v_x|$, the two projection groups have opposite signs instead,
and their $x$ components cancel while their $y$ components add. The net
translation then points along the opposing major $y$ direction. At a diagonal
boundary, one group can have zero projection and contribute no braking.

Yaw changes each wheel's local contact velocity through $\omega\times
\mathbf{p}_i$. It can change the four projection signs independently and
produce a yaw moment, so the simple cancellation above is the symmetric
translation case, not a universal guarantee for every rotating motion.

## Check against the requested recording

The requested command was run:

```bash
python experiments/analysis_raw_data_rollout.py data/v2-19823-bare/DiagFast2_full.csv --time-window 6 7.4 --start-at-brake
```

It found a `brake_engaged` rising edge at $6.542295$ s. The first aligned
sample used for the displayed rollout is at $6.5708$ s. The printed fused
body velocity there is $(v_x,v_y,\omega)=(0.9303,-0.4816,0.0289)$ in SI units;
the corresponding odometry estimate is $(0.9293,-0.4403,0.0537)$. Thus both
translational components are present, with $|v_x|>|v_y|$.

Using the default wheel geometry and the fused sample above in the fixed-axis
function gives the four signed projections

$$
(0.9521,\ 0.3116,\ 0.2941,\ 0.9540),
$$
all positive. For an illustrative input $b_w=1\,\mathrm{m/s^2}$, the model's
four per-wheel acceleration contributions are

$$
(-0.7071,0.7071),\quad
(-0.7071,-0.7071),\quad
(-0.7071,-0.7071),\quad
(-0.7071,0.7071),
$$
which sum to $(-2.8284,0)$ with zero yaw acceleration for this sample. This
numerical check demonstrates why the model initially brakes along body $x$
although the robot also has substantial body-$y$ velocity. The chosen
$b_w=1$ is only an illustration; the command does not estimate or specify the
model's braking parameter.

Important limitation: `analysis_raw_data_rollout.py` integrates recorded odom
and IMU signals and prints their starting velocities; it does not call
`sliding_deceleration_coulomb_fixed_axis`. Its result is useful for choosing a
realistic initial velocity and checking the model's projections, but it is not
an experimental validation of the model's predicted deceleration. Over the
first printed 30 ms interval, odometry changes from
$(0.9545,-0.4678)$ to $(0.9293,-0.4403)$ m/s, while the fused estimate changes
from $(0.9551,-0.4682)$ to $(0.9303,-0.4816)$ m/s. Those are measured/filter
outputs, affected by sampling, estimation, and the actual robot dynamics; they
should not be read as the fixed-axis model's acceleration trace.

## Scope and limitations

The model is useful as a deliberately simple sliding-friction approximation:
fixed per-wheel braking magnitude, direction constrained by the roller
geometry, and a sign-based Coulomb switch. It assumes the wheel directions
and loads are symmetric, treats each wheel contribution with a single
calibrated value, and does not model a measured tire/roller friction curve,
load transfer, or residual resistance in the free-rolling direction. It should
therefore explain the directional behavior qualitatively, not be mistaken for
a full contact-patch model.
