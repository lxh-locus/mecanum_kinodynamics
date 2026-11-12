# Mecanum Physics

This document describes the kinematic and dynamic models implemented in
[`experiments/mecanum_physics.py`](../experiments/mecanum_physics.py). The body
frame is FLU: $x$ forward, $y$ left, and $z$ up. Wheels are ordered front-left,
front-right, rear-left, rear-right and denoted $1$ through $4$. Let
$r$ be the wheel radius, $l$ the half-length of the wheelbase, and $w$ its
half-width; the code uses $L = l+w$. The paper denotes these half-dimensions
by $\rho$ (half-length) and $l$ (half-width), respectively, so its
$\rho+l$ corresponds to $L$ here.

## 1. Forward and Inverse Kinematics

The body velocity is $(v_x,v_y,\omega)$, where $\omega$ is the yaw rate, and
the wheel angular velocities are $(\dot\phi_1,\dot\phi_2,\dot\phi_3,\dot\phi_4)$.
The implemented forward map is

$$
\begin{aligned}
v_x &= \frac{r}{2}(\dot\phi_1+\dot\phi_2),\\
v_y &= \frac{r}{2}(\dot\phi_3-\dot\phi_1),\\
\omega &= \frac{r}{2L}(\dot\phi_2-\dot\phi_3).
\end{aligned}
$$

Equivalently, $\mathbf{v}=J\dot{\boldsymbol\phi}$, with

$$
J=\frac{r}{2}
\begin{bmatrix}
1&1&0&0\\
-1&0&1&0\\
0&L^{-1}&-L^{-1}&0
\end{bmatrix}.
$$

Four wheel speeds describe a three-degree-of-freedom body velocity, so they
must satisfy the compatibility constraint

$$
\dot\phi_1+\dot\phi_2-\dot\phi_3-\dot\phi_4=0.
$$

If this constraint is maintained over time, differentiating it gives the
corresponding wheel-acceleration constraint. It is not automatically true for
an arbitrary four-wheel acceleration model.

`forward_kinematics` checks this constraint by default. With `strict=False`,
it projects an incompatible input to the closest compatible vector in the
Euclidean norm before applying the map.

The inverse kinematics returns the compatible wheel-speed solution

$$
\begin{aligned}
\dot\phi_1 &= \frac{v_x-v_y-L\omega}{r}, &
\dot\phi_2 &= \frac{v_x+v_y+L\omega}{r},\\
\dot\phi_3 &= \frac{v_x+v_y-L\omega}{r}, &
\dot\phi_4 &= \frac{v_x-v_y+L\omega}{r}.
\end{aligned}
$$

These equations produce wheel speeds satisfying the constraint and invert the
forward map. The model's roller direction metadata documents the wheel
configuration, while these functions use the fixed equations above directly.

## 2. Forward and Reverse Dynamics

The dynamics functions map wheel torques
$\mathbf{M}=(M_1,M_2,M_3,M_4)$ to body acceleration
$\mathbf{a}=(a_x,a_y,\alpha)$, or solve for torques given the desired body
acceleration. The linear model ignores velocity-dependent coupling and uses
the matrix $G$:

$$
\mathbf{a}=G\mathbf{M},\qquad
G=\begin{bmatrix}
k_t&k_t&k_t&k_t\\
-k_t&k_t&k_t&-k_t\\
-k_\psi&k_\psi&-k_\psi&k_\psi
\end{bmatrix},
$$

where

$$
k_t=\frac{r}{m r^2+4I_w},\qquad
k_\psi=\frac{rL}{I_z r^2+4I_wL^2}.
$$

Here $m$ is body mass, $I_w$ is the spin inertia of one wheel, and $I_z$ is
body yaw inertia. These gains are the body-acceleration form of the paper's
approximate model (Eqs. 44-45), with this document's $L=l+w$ corresponding to
the paper's $\rho+l$. `forward_dynamics_linear` evaluates this matrix product.
`inverse_dynamics_linear` solves the underdetermined $3\times4$ system with
the Moore-Penrose pseudoinverse, returning the minimum-Euclidean-norm torque
vector:

$$
\mathbf{M}=G^+\mathbf{a}.
$$

The exact model also accounts for velocity-dependent terms. Its coefficients
are computed from

$$
\begin{aligned}
A &= \frac{m r^2}{8}+\frac{I_zr^2}{16L^2}+I_w,&
B &= \frac{I_zr^2}{16L^2},&
C &= \frac{m r^2}{8}-\frac{I_zr^2}{16L^2},\\
k_2 &= \frac{r(B+C)}{2L(A+C)},&
A_2 &= \frac{3A+4B-C}{4(A+C)(A+2B-C)},&
C_2 &= \frac{A+4B-3C}{4(A+C)(A+2B-C)},\\
h &= \frac{A_2-C_2}{2}.&&&
\end{aligned}
$$

For compatible wheel velocities, define the nonlinear contributions

$$
N_1=k_2(\dot\phi_2+\dot\phi_3)(\dot\phi_2-\dot\phi_3),\qquad
N_{23}=k_2(\dot\phi_3-2\dot\phi_1-\dot\phi_2)(\dot\phi_2-\dot\phi_3).
$$

The exact implementation first computes three wheel-acceleration channels:

$$
\begin{aligned}
\ddot\phi_1 &= N_1+A_2M_1-h(M_2-M_3)+C_2M_4,\\
\ddot\phi_2 &= N_{23}+A_2M_2-h(M_1-M_4)+C_2M_3,\\
\ddot\phi_3 &= N_{23}+A_2M_3+h(M_1-M_4)+C_2M_2.
\end{aligned}
$$

Those channels are converted to body acceleration using the same structure as
the forward kinematics:

$$
a_x=\frac{r}{2}(\ddot\phi_1+\ddot\phi_2),\qquad
a_y=\frac{r}{2}(\ddot\phi_3-\ddot\phi_1),\qquad
\alpha=\frac{r}{2L}(\ddot\phi_2-\ddot\phi_3).
$$

In matrix form, let
$\ddot{\mathbf{q}}=[\ddot\phi_1,\ddot\phi_2,\ddot\phi_3]^T$ collect the
three acceleration channels computed by the implementation, and let
$\mathbf{M}=[M_1,M_2,M_3,M_4]^T$ be the four wheel torques. The output
$\mathbf{a}=[a_x,a_y,\alpha]^T$ is the body-frame acceleration, with
longitudinal acceleration $a_x$, lateral acceleration $a_y$, and yaw angular
acceleration $\alpha$. Define

$$
\mathbf{n}=\begin{bmatrix}N_1\\N_{23}\\N_{23}\end{bmatrix},\qquad
Q=\begin{bmatrix}
A_2&-h&h&C_2\\
-h&A_2&C_2&h\\
h&C_2&A_2&-h
\end{bmatrix},\qquad
R=\frac{r}{2}\begin{bmatrix}
1&1&0\\
-1&0&1\\
0&L^{-1}&-L^{-1}
\end{bmatrix}.
$$

Then the exact forward implementation is

$$
\ddot{\mathbf{q}}=\mathbf{n}+Q\mathbf{M},\qquad
\mathbf{a}=R\ddot{\mathbf{q}}=R(\mathbf{n}+Q\mathbf{M}).
$$

Here $\ddot{\mathbf{q}}$ contains only the three acceleration channels used
in the code, not all four wheel accelerations. For the inverse, the code
reconstructs those channels from body acceleration with

$$
H=\frac{1}{r}\begin{bmatrix}
1&-1&-L\\
1&1&L\\
1&1&-L
\end{bmatrix}=R^{-1},\qquad
\ddot{\mathbf{q}}_{\mathrm{desired}}=R^{-1}\mathbf{a}=H\mathbf{a}.
$$

It then removes the velocity-dependent contribution and solves for the
minimum-norm torques:

$$
\mathbf{M}=Q^+(\ddot{\mathbf{q}}_{\mathrm{desired}}-\mathbf{n})
=Q^+(H\mathbf{a}-\mathbf{n}),
$$

where $Q^+$ is the Moore-Penrose pseudoinverse. This is the matrix form of
the `numpy.linalg.pinv` operation in `inverse_dynamics_exact`.

The paper's exact formulation (Eqs. 65-67) obtains the fourth wheel speed from
$\dot\phi_4=\dot\phi_1+\dot\phi_2-\dot\phi_3$; differentiating this relation
gives the acceleration compatibility constraint. The approximate four-wheel
acceleration model is different: its Eq. 69 gives

$$
(A-2B-C)(\ddot\phi_1+\ddot\phi_2-\ddot\phi_3-\ddot\phi_4)
=M_1+M_2-M_3-M_4,
$$

where $A-2B-C=I_w$. Thus that approximate model preserves acceleration
compatibility only when $M_1+M_2=M_3+M_4$. This module's exact dynamics
computes only the first three acceleration channels, with the fourth implied
by the exact kinematic constraint.

`forward_dynamics_exact` takes both wheel torques and current wheel
velocities. `inverse_dynamics_exact` first converts the requested body
acceleration back to the three acceleration channels, removes $N_1$ and
$N_{23}$, then solves the resulting $3\times4$ torque system with a
pseudoinverse. As in the linear inverse, this selects a minimum-norm torque
solution. Both exact functions validate wheel-speed compatibility by default;
`strict=False` applies the same projection described in the kinematics
section.

## 3. Walkthrough of `mecanum_physics.py`

- **`MecanumPhysicsParams`** is a frozen dataclass holding geometry, mass,
  inertias, a default maximum wheel speed, and roller direction metadata. The
  default geometry is $r=3\times0.0254$ m, $l=0.25$ m, and $w=0.2405$ m;
  defaults for mass and inertias are 100 kg, 0.08 kg m$^2$ per wheel, and
  8.174 kg m$^2$ in yaw.
- **`params_from_model`** copies the geometry and dynamics values from a model
  object, converting them to floats. If the object has no maximum wheel speed,
  it uses the dataclass default. Other dataclass fields retain their defaults.
- **`wheel_constraint_violation`** evaluates
  $\dot\phi_1+\dot\phi_2-\dot\phi_3-\dot\phi_4$.
- **`relax_wheel_velocity_to_constraint`** subtracts one quarter of this
  residual from the first two wheels and adds it to the last two. This is the
  orthogonal projection onto the compatible-wheel-speed hyperplane.
- **`_prepare_wheel_velocity`** is the shared internal shape and compatibility
  check. It requires exactly four values, raises an error for an incompatible
  vector in strict mode, and otherwise applies the projection.
- **`forward_kinematics`** prepares the wheel vector and evaluates the three
  body-velocity equations in Section 1. The fourth wheel is used for
  compatibility checking, not directly in the three returned expressions.
- **`inverse_kinematics`** validates that the body velocity has three values
  and returns the four compatible wheel speeds from the inverse equations.
- **`forward_dynamics_matrix_linear`** builds the $3\times4$ matrix $G$ from
  wheel radius, body mass, wheel spin inertia, yaw inertia, and $L$.
- **`forward_dynamics_linear`** validates a four-element torque vector and
  returns $G\mathbf{M}$.
- **`inverse_dynamics_linear`** validates a three-element body acceleration
  and returns $G^+\mathbf{a}$ using `numpy.linalg.pinv`.
- **`exact_dynamics_coeffs`** computes and returns $(k_2,A_2,C_2)$ for the
  nonlinear model from the physical parameters.
- **`forward_dynamics_exact`** validates torque and wheel-speed inputs,
  computes the nonlinear and torque-dependent acceleration channels, then
  transforms them to $(a_x,a_y,\alpha)$.
- **`inverse_dynamics_exact`** validates the requested acceleration and wheel
  speeds, reconstructs the acceleration channels, subtracts the nonlinear
  terms, and applies a pseudoinverse to recover the minimum-norm torques.
- **`__all__`** lists the parameter type and public helpers exported by this
  module. The underscored `_prepare_wheel_velocity` helper is intentionally
  internal.

The `max_wheel_velocity` and `roller_directions` fields are stored as model
metadata but are not used by the kinematics or dynamics calculations in this
module. Input shape checks raise `ValueError` rather than attempting to infer
or reshape malformed vectors.