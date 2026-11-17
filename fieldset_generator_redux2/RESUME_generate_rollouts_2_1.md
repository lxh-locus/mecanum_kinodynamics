# Resume notes: creating `generate_rollouts_2_1.py`

Working notes for continuing the redux2 stripping effort. The goal of this
folder is a minimal, self-contained fieldset rollout generator that keeps only
the **fixed-axis Coulomb sliding model** and the **shrunk-contour boundary
sampler**, with zero imports from the `experiments/` package.

## Status

- [x] `mecanum_fixed_axis.py` — sliding dynamics + trajectory rollout (done)
- [x] `velocity_polytope.py` — polytope construction + boundary sampling (done)
- [x] `test_mecanum_fixed_axis.py`, `test_velocity_polytope.py` — 24 tests passing
- [ ] `generate_rollouts_2_1.py` — stripped port of
      `fieldset_generator_redux1/generate_rollouts_2.py` (**next step**)

Both modules were verified bit-identical against the `experiments/` originals
(deceleration over 200 random states, full rollouts, delay helpers, the
`strategy_rollout` fixed-axis path, polytope faces/extents, shrink sampler at
degrees 0–3) before the verification script was discarded.

## Module APIs available

### `mecanum_fixed_axis.py`

| Symbol | Notes vs. `experiments/` original |
| --- | --- |
| `MecanumPhysicsParams` | Dropped `wheel_spin_inertia` and `max_wheel_velocity` fields |
| `individual_wheel_braking_deceleration(max_body_x_deceleration, params)` | Fixed-axis calibration only; no `model=` arg |
| `sliding_deceleration_coulomb_fixed_axis(body_velocity, wheel_braking_deceleration, params, tolerance)` | Unchanged |
| `rollout_sliding_deceleration(body_velocity, wheel_braking_deceleration, params, dt, max_time, ...)` | Hardcoded to fixed-axis; no `deceleration_fn`/`deceleration_kwargs` |
| `integrate_response_delay(initial_velocity, duration, dt)` | Ported from `generate_rollouts_2.py` |
| `append_response_delay(initial_velocity, braking_poses, delay, dt)` | Ported from `generate_rollouts_2.py` |
| `fixed_axis_braking_rollout(initial_velocity, body_x_braking_deceleration, delay, params, dt, max_time)` | New convenience; returns `(trajectory, stop_time, stopped, wheel_braking)` — equivalent to `strategy_rollout(...)`'s `"sliding-coulomb-fixed-axis"` path including the delay hold |

### `velocity_polytope.py`

| Symbol | Notes vs. `experiments/` original |
| --- | --- |
| `polytope_faces(params, max_wheel_velocity, vx_range, vy_range, omega_range)` | Public rename of `_polytope_faces`; takes `params` (reads only `wb_hlength`, `wb_hwidth`, `wheel_radius`) instead of a `Mecanum` model |
| `compute_wheel_velocity_limits(params, max_wheel_velocity)` | Takes `params` instead of model |
| `compute_truncated_extents(vertices)` | Unchanged |
| `sample_boundary_velocities(vertices, faces, sampling_degree)` | Shrunk-contour sampler only; face-bisection sampler and its helpers were removed |

## Plan for `generate_rollouts_2_1.py`

Port `fieldset_generator_redux1/generate_rollouts_2.py`, replacing all
`experiments.*` imports with the two local modules, and stripping:

1. **Strategies**: keep only the fixed-axis path. Delete `STRATEGIES`,
   `STRATEGY_FILENAME_LABELS`, the `--strategy` flag, and the
   `strategy_rollout` dispatcher — call `fixed_axis_braking_rollout` directly
   (it already folds in `append_response_delay`). Filename label stays
   `fixed-axis` (archives named `{field}_fixed-axis_rollouts.npz/.png`).
2. **Sampling**: delete the `--sampling-method` flag (`bisect` is gone); always
   use `sample_boundary_velocities`.
3. **Model object**: no `Mecanum` — pass `MecanumPhysicsParams()` straight to
   `polytope_faces` / `compute_wheel_velocity_limits`.
4. **sys.path hack**: not needed if the script lives beside the modules and is
   run from this folder; keep a `Path(__file__).parent` insert only if it
   should be runnable from the repo root.

Keep (config-driven, still needed):
- `load_config`, `footprint_corners`, `footprint_for_field`,
  `footprint_name_for_field`, `response_time`, `velocity_ranges_from_field`,
  `max_wheel_velocity_from_config` (uses `params.wheel_radius`)
- `save_rollout_data` — preserve the **format_version=3** `.npz` schema exactly
  (keys: `format_version`, `model`, `field_name`, `initial_velocities`,
  `poses`, `lengths`, `stop_times`, `stopped`, `dt`, `response_time`,
  `wheel_braking_deceleration`, `footprint_vertices`, `footprint_names`,
  `trajectory_footprint_indices`) so `inspect_rollout_archives.py` /
  `overlay_*_rollout_archives.py` keep working; `model` field should remain
  `"sliding-coulomb-fixed-axis"` for archive compatibility
- `plot_field_rollouts` and the CLI (`--field`, `--max-wheel-velocity`,
  `--sampling-degree`, `--dt`, `--max-time`, `--output-dir`,
  `--data-output-dir`, `--save-rollout-data`), matplotlib `Agg` backend
- Default config: `fieldset_config.json` in this folder (already present;
  `generator_params.rollout_strategy` can be ignored or validated as
  fixed-axis only)

## Validation after writing the script

- `cd fieldset_generator_redux2 && python -m pytest` (24 tests)
- Smoke run: `python generate_rollouts_2_1.py --save-rollout-data` and diff a
  resulting `.npz` against the redux1 output for the same config/field with
  `--strategy sliding-coulomb-fixed-axis --sampling-method shrink` to confirm
  identical poses/stop_times.
