# Mecanum Kinematics and Dynamics

## Running Tests

From the repository root, activate the conda environment and run the pytest file:

```bash
conda_activate
pytest -q test_mecanum_physics.py
```

You can also run it as a single command:

```bash
conda_activate && pytest -q test_mecanum_physics.py
```

## Sliding Deceleration Rollout Demo

Run the demo that samples body velocities on a box-truncated kinematic
wheel-speed polytope and rolls them forward while applying the sliding
deceleration model:

```bash
conda_activate && python sliding_stopping_distance_rollout.py
```

Useful options:

```bash
python sliding_stopping_distance_rollout.py --sampling-degree 1 --sampling-method shrink
python sliding_stopping_distance_rollout.py --vx-min 0.5 --vx-max 1.0 --vy-min -0.1 --vy-max 0.1 --omega-min 0.5 --omega-max 2.0
python sliding_stopping_distance_rollout.py --max-body-x-deceleration 4.0 --max-time 5.0
python sliding_stopping_distance_rollout.py --sweep-n-angles 24 --direction-range-scale 1.25
python sliding_stopping_distance_rollout.py --hide-requested-bounds-box
```

## Recorded Trajectory Comparison

Compare integrated odom and EKF-filtered trajectories with predicted braking
paths over a recording's selected time window:

```bash
conda_activate
python experiments/analysis_comparison.py data/v2-27361-shelved/Fast2.csv --time-window 0 1
```

The default strategy is `sliding-coulomb-approx`. Use `--strategies all` or
provide one or more strategy names from `--help`. Every prediction starts at
the exact selected start time with zero pose and the interpolated EKF body
velocity; `--initial-source odom` uses odom velocity instead. The interval is
limited to common odom/IMU coverage, and predictions end at rest or at the
interval's end. `--max-time` can further cap their duration.

Use `--start-at-brake` to begin at the first brake-engaged rising edge within
the window. Bias estimation and EKF processing still use the full recording,
with the same bias options as the raw-data rollout script. Pure IMU rollouts
are omitted. Energy graphs are off unless `--energy` is supplied.
`--output tmp/comparison.png` saves the figure instead of opening it.
