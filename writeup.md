# Mecanum Drive Kinematic and Dynamic Limits

This document derives the mathematics behind the four figure-generation scripts in
this repository.  All models use the `Mecanum` class defined in `mecanum_common.py`.

---

## Notation and model parameters

| Symbol | Description | Default |
|--------|-------------|---------|
| $R$ | Wheel radius | 0.1 m |
| $\rho$ | Wheel-base half-length | 0.3 m |
| $l$ | Wheel-base half-width | 0.25 m |
| $m_s$ | Total platform mass | 20 kg |
| $J_1$ | Wheel spin inertia (per wheel) | 0.02 kg·m² |
| $J_C$ | Platform yaw inertia about CoM | 1.2 kg·m² |
| $v_x, v_y$ | Body-frame forward / lateral velocity | m/s |
| $\dot\psi$ | Body yaw rate ($v_\omega$ in code) | rad/s |
| $\varphi_i$ | Angular velocity of wheel $i$ | rad/s |
| $M_i$ | Torque applied to wheel $i$ | N·m |
| $\rho + l$ | Referred to as $\ell$ in equations below | m |

Wheel numbering: 1 = front-left, 2 = front-right, 3 = rear-left, 4 = rear-right.
Body frame: $x$ forward, $y$ left, $z$ up (FLU).

---

## 1. Kinematics — wheel velocities and body velocities

### 1.1 Forward kinematics: wheel speeds → body velocity

From the no-slip constraint (Zeidis 2019, Eq. 37), when the compatibility condition
$\varphi_1 + \varphi_2 = \varphi_3 + \varphi_4$ holds:

$$v_x = \frac{R}{2}(\varphi_1 + \varphi_2)$$

$$v_y = \frac{R}{2}(\varphi_3 - \varphi_1)$$

$$\dot\psi = \frac{R}{2(\rho+l)}(\varphi_2 - \varphi_3)$$

### 1.2 Inverse kinematics: body velocity → wheel speeds

Inverting the above:

$$\varphi_1 = \frac{1}{R}\bigl(v_x - v_y - (\rho+l)\,\dot\psi\bigr)$$

$$\varphi_2 = \frac{1}{R}\bigl(v_x + v_y + (\rho+l)\,\dot\psi\bigr)$$

$$\varphi_3 = \frac{1}{R}\bigl(v_x + v_y - (\rho+l)\,\dot\psi\bigr)$$

$$\varphi_4 = \frac{1}{R}\bigl(v_x - v_y + (\rho+l)\,\dot\psi\bigr)$$

In matrix form: $\boldsymbol\varphi = J\,\mathbf{v}$, where

$$J = \frac{1}{R}\begin{pmatrix}1&-1&-(\rho+l)\\1&1&(\rho+l)\\1&1&-(\rho+l)\\1&-1&(\rho+l)\end{pmatrix},
\qquad \mathbf{v} = \begin{pmatrix}v_x\\v_y\\\dot\psi\end{pmatrix}$$

### 1.3 Compatibility constraint

For the four wheel-speed equations to be consistent with three body-velocity
degrees of freedom, the following must hold:

$$\varphi_1 + \varphi_2 - \varphi_3 - \varphi_4 = 0$$

This is enforced (or projected to) by `bodyv_from_wheelv` via the `strict` flag.

---

## 2. Kinematic velocity limit box (`kinematic_velocity_limit_box.py`)

### 2.1 Half-space description

Given a maximum wheel speed $\omega_\text{max}$, each wheel constraint
$|\varphi_i| \le \omega_\text{max}$ becomes one pair of linear inequalities in
$(v_x, v_y, \dot\psi)$.  Using the inverse-kinematics rows:

| Wheel | Constraint $\le R\,\omega_\text{max}$ |
|-------|---------------------------------------|
| 1 | $v_x - v_y - (\rho+l)\dot\psi \le R\,\omega_\text{max}$ |
| 2 | $v_x + v_y + (\rho+l)\dot\psi \le R\,\omega_\text{max}$ |
| 3 | $v_x + v_y - (\rho+l)\dot\psi \le R\,\omega_\text{max}$ |
| 4 | $v_x - v_y + (\rho+l)\dot\psi \le R\,\omega_\text{max}$ |

Each row also contributes a $\ge -R\,\omega_\text{max}$ bound, giving **8 half-spaces**
total:

$$A\,\mathbf{v} \le b, \qquad b = R\,\omega_\text{max}\,\mathbf{1}_8$$

$$A = \begin{pmatrix}J\\-J\end{pmatrix}$$

$\mathbf{1}_8 \in \mathbb{R}^8$ denotes the column vector of all ones, so
$b = R\,\omega_\text{max}\,\mathbf{1}_8$ places the same scalar bound on every
half-space: four upper bounds ($\varphi_i \le \omega_\text{max}$) and four lower
bounds ($-\varphi_i \le \omega_\text{max}$, i.e.\ $\varphi_i \ge -\omega_\text{max}$).

### 2.2 Feasible polytope geometry

The intersection of the 8 half-spaces is a bounded **convex polytope** in
$(v_x,\, v_y,\, \dot\psi)$ space.  Its vertices are found by solving every
$\binom{8}{3} = 56$ triples of active constraints, retaining only feasible
intersection points.

### 2.3 Axis-wise maximum extents

Setting all but one component of $\mathbf{v}$ to zero:

$$|v_x|_\text{max} = |v_y|_\text{max} = R\,\omega_\text{max}$$

$$|\dot\psi|_\text{max} = \frac{R\,\omega_\text{max}}{\rho + l}$$

These maxima cannot be achieved simultaneously due to coupling.

### 2.4 Colormap — wheel utilization

The second subplot colors each sampled feasible point by

$$u = \frac{\max_i |\varphi_i|}{\omega_\text{max}} \in [0, 1]$$

---

## 3. Approximate dynamic model (`mecanum_common.py`, `_bodya_from_wheeltorque_matrix`)

The approximate model (Zeidis 2019, Eqs. 44–45, differentiated in time) gives a
**linear** map from wheel torques to body accelerations:

$$\begin{pmatrix}\dot v_x\\\dot v_y\\\ddot\psi\end{pmatrix} = G\begin{pmatrix}M_1\\M_2\\M_3\\M_4\end{pmatrix}$$

Differentiating the kinematic constraint equations with respect to time and
inserting the Lagrangian equations of motion yields:

$$G = \begin{pmatrix}
k_\text{lin} & k_\text{lin} & k_\text{lin} & k_\text{lin}\\
-k_\text{lin} & k_\text{lin} & k_\text{lin} & -k_\text{lin}\\
-k_\text{yaw} & k_\text{yaw} & -k_\text{yaw} & k_\text{yaw}
\end{pmatrix}$$

where

$$k_\text{lin} = \frac{R}{m_s R^2 + 4J_1}, \qquad k_\text{yaw} = \frac{R}{J_C R^2 + 4J_1(\rho+l)^2}$$

### Inverse — minimum-norm torques

The system $G\mathbf{M} = \mathbf{a}$ is underdetermined (3 equations, 4 unknowns).
The minimum Euclidean-norm solution is

$$\mathbf{M}^* = G^+\,\mathbf{a}$$

where $G^+ = G^T(GG^T)^{-1}$ is the Moore–Penrose pseudoinverse.

---

## 4. Exact dynamic model (`mecanum_common.py`, `bodya_from_wheeltorque_exact`)

The exact (Chaplygin) equations of motion (Zeidis 2019, Eq. 65) are nonlinear due
to the non-holonomic constraint coupling.  The wheel angular accelerations are:

$$\ddot\varphi_1 = k_2(\varphi_2+\varphi_3)(\varphi_2-\varphi_3) + A_2 M_1 - H(M_2 - M_3) + C_2 M_4$$

$$\ddot\varphi_2 = k_2(\varphi_3 - 2\varphi_1 - \varphi_2)(\varphi_2-\varphi_3) + A_2 M_2 - H(M_1 - M_4) + C_2 M_3$$

$$\ddot\varphi_3 = k_2(\varphi_3 - 2\varphi_1 - \varphi_2)(\varphi_2-\varphi_3) + A_2 M_3 + H(M_1 - M_4) + C_2 M_2$$

where $H = \tfrac{1}{2}(A_2 - C_2)$ and the state-independent coefficients are
(Zeidis Eq. 66):

$$A = \frac{m_s R^2}{8} + \frac{J_C R^2}{16(\rho+l)^2} + J_1, \quad
  B = \frac{J_C R^2}{16(\rho+l)^2}, \quad
  C = \frac{m_s R^2}{8} - \frac{J_C R^2}{16(\rho+l)^2}$$

$$k_2 = \frac{R(B+C)}{2(\rho+l)(A+C)}, \quad
  A_2 = \frac{3A+4B-C}{4(A+C)(A+2B-C)}, \quad
  C_2 = \frac{A+4B-3C}{4(A+C)(A+2B-C)}$$

Body accelerations follow from the exact kinematics (Zeidis Eq. 67):

$$\dot v_x = \frac{R}{2}(\ddot\varphi_1 + \ddot\varphi_2), \quad
  \dot v_y = \frac{R}{2}(\ddot\varphi_3 - \ddot\varphi_1), \quad
  \ddot\psi = \frac{R}{2(\rho+l)}(\ddot\varphi_2 - \ddot\varphi_3)$$

The nonlinear terms vanish when $\varphi_2 = \varphi_3$ (pure translation) or
$M_1 + M_2 = M_3 + M_4$ (compatibility condition, Zeidis Eq. 68), in which case
the exact and approximate models coincide.

---

## 5. Dynamic acceleration limit box (`dynamic_acceleration_limit_box.py`)

### 5.1 Feasible acceleration set — the zonotope

With the approximate linear model $\mathbf{a} = G\mathbf{M}$ and symmetric torque
bounds $|M_i| \le M_\text{max}$, the feasible body-acceleration set is the image
of the $\ell^\infty$ ball in torque space under $G$:

$$\mathcal{Z} = \{G\mathbf{M} : \|\mathbf{M}\|_\infty \le M_\text{max}\}
             = \bigoplus_{i=1}^{4} [-M_\text{max},\,M_\text{max}]\,\mathbf{g}_i$$

where $\mathbf{g}_i$ is the $i$-th column of $G$.  This is a **zonotope** —
the Minkowski sum of four line segments.  It has at most $2\binom{4}{2} = 12$
faces (each a parallelogram), and the convex hull of the $2^4 = 16$ corner images

$$\bigl\{G\,\mathbf{c} : \mathbf{c} \in \{-M_\text{max}, +M_\text{max}\}^4\bigr\}$$

is computed via `scipy.spatial.ConvexHull`.

### 5.2 Axis-wise maximum extents

$$|\dot v_x|_\text{max} = 4\,k_\text{lin}\,M_\text{max}$$

$$|\dot v_y|_\text{max} = 4\,k_\text{lin}\,M_\text{max}$$

$$|\ddot\psi|_\text{max} = 4\,k_\text{yaw}\,M_\text{max}$$

As with velocities, these single-axis maxima cannot be achieved simultaneously.

### 5.3 Colormap — torque utilization

Points inside the zonotope are colored by the $\ell^\infty$ norm of the
**minimum-norm** (pseudoinverse) torque solution:

$$u = \frac{\|G^+\mathbf{a}\|_\infty}{M_\text{max}}$$

This is a lower bound on true minimum $\ell^\infty$ utilization (the actual minimum
$\ell^\infty$-norm torque solution could differ due to null-space freedom).

---

## 6. Stopping time (`stopping_time_polytope.py`)

### 6.1 Problem statement

Given initial body velocity $\mathbf{v}_0 \in \mathbb{R}^3$ and a constant
deceleration $\mathbf{a}$ chosen from the acceleration zonotope $\mathcal{Z}$,
find the minimum time $t^*$ to bring the robot to rest.

### 6.2 Derivation

Under constant deceleration the velocity evolves as
$\mathbf{v}(t) = \mathbf{v}_0 + \mathbf{a}\,t$.
Setting $\mathbf{v}(t^*) = \mathbf{0}$ requires $\mathbf{a} = -\mathbf{v}_0 / t^*$.
The constraint $\mathbf{a} \in \mathcal{Z}$ then becomes:

$$\frac{-\mathbf{v}_0}{t^*} \in \mathcal{Z}
\quad\Longleftrightarrow\quad
t^* \ge \mu_\mathcal{Z}(-\mathbf{v}_0)$$

where $\mu_\mathcal{Z}$ is the **Minkowski functional** (gauge) of $\mathcal{Z}$.
The minimum stopping time is therefore

$$\boxed{t^*(\mathbf{v}_0) = \mu_\mathcal{Z}(-\mathbf{v}_0)}$$

### 6.3 Efficient computation via half-spaces

The zonotope $\mathcal{Z}$ is described by $F$ half-spaces
$\{\mathbf{x} : \mathbf{n}_f \cdot \mathbf{x} + d_f \le 0\}$
(scipy convention).  Since $\mathcal{Z}$ contains the origin, $d_f < 0$ for all
faces.  The Minkowski functional evaluates to:

$$\mu_\mathcal{Z}(\mathbf{p}) = \max_f \frac{\mathbf{n}_f \cdot \mathbf{p}}{-d_f}$$

Applied to $\mathbf{p} = -\mathbf{v}_0$ this gives $t^*$ via a single matrix
multiply and row-wise maximum — **no LP required**.

### 6.4 Homogeneity

$\mu_\mathcal{Z}$ is positively homogeneous of degree 1:

$$t^*(\lambda\,\mathbf{v}_0) = \lambda\,t^*(\mathbf{v}_0), \quad \lambda > 0$$

Stopping time is therefore **linear** in the magnitude of the initial velocity
(for a fixed direction).

---

## 7. Stopping distance (`stopping_distance_polytope.py`)

### 7.1 Displacement under constant deceleration

Under optimal constant deceleration $\mathbf{a} = -\mathbf{v}_0 / t^*$, the
center-of-mass displacement from start to rest is

$$\Delta\mathbf{x} = \int_0^{t^*} \mathbf{v}(t)\,dt
  = \int_0^{t^*} \mathbf{v}_0\!\left(1 - \frac{t}{t^*}\right)dt
  = \frac{1}{2}\,\mathbf{v}_0\,t^*$$

### 7.2 Translational stopping distance

The Euclidean distance traveled by the center of mass is

$$\boxed{d_\text{trans}(\mathbf{v}_0)
  = \tfrac{1}{2}\,\bigl\|[v_x,\, v_y]\bigr\|\cdot t^*(\mathbf{v}_0)}$$

### 7.3 Angular stopping displacement

$$\boxed{\theta_\text{stop}(\mathbf{v}_0)
  = \tfrac{1}{2}\,|v_\omega|\cdot t^*(\mathbf{v}_0)}$$

Here $|v_\omega|$ (rendered in the scripts as `|vw| [rad/s]`) is the absolute
value of body yaw rate.  It measures how fast the platform is rotating,
independent of clockwise/counterclockwise sign.  In the stopping plots, this
quantity is used as a color channel to show rotational-speed magnitude while
the vertical axis shows stopping time or translational stopping distance.

### 7.4 Quadratic dependence on speed

Because $t^*$ is linear in $\|\mathbf{v}_0\|$ (Section 6.4), and
$\|[v_x, v_y]\|$ is linear in speed, stopping distance is **quadratic** in
translational speed for a fixed direction:

$$d_\text{trans} = \frac{\|[v_x,\,v_y]\|^2}{2\,\hat{\mathbf{v}}^\top G_\text{eff}\,\hat{\mathbf{v}}}$$

where $G_\text{eff}$ encapsulates the directional deceleration capacity.  The
reference curve shown in the right subplot uses the pure-$v_x$ maximum
deceleration $a_\text{max} = 4\,k_\text{lin}\,M_\text{max}$:

$$d_\text{ref}(v) = \frac{v^2}{2\,a_\text{max}}$$

---

## 8. Relationship between the four scripts

```
mecanum_common.py
  ├── wheelv_from_bodyv / bodyv_from_wheelv
  │       │
  │       └──► kinematic_velocity_limit_box.py
  │               Polytope P_v = {v : |J v|_∞ ≤ ω_max}
  │               8 half-spaces, analytic vertices
  │
  ├── _bodya_from_wheeltorque_matrix  (G)
  │       │
  │       └──► dynamic_acceleration_limit_box.py
  │               Zonotope Z = {G M : |M|_∞ ≤ M_max}
  │               16 image corners, ConvexHull
  │
  Z (zonotope) + P_v (kinematic feasibility)
       │
       ├──► stopping_time_polytope.py
       │       t*(v₀) = μ_Z(−v₀)   [Minkowski functional]
       │
       └──► stopping_distance_polytope.py
               d_trans = ½ ‖[vx,vy]‖ · t*(v₀)
               θ_stop  = ½ |vω|     · t*(v₀)
```

---

## References

I. Zeidis and K. Zimmermann, "Dynamics of a four-wheeled mobile robot with
Mecanum wheels," *ZAMM — Journal of Applied Mathematics and Mechanics*, vol. 99,
no. 12, 2019. doi: 10.1002/zamm.201900173
