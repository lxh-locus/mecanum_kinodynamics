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
```
