import argparse
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mecanum_common import Mecanum


def _build_limit_planes(model, max_wheel_velocity):
	"""Return plane coefficients for |wheel_velocity| <= max_wheel_velocity."""
	l_plus_w = model.wb_hlength + model.wb_hwidth
	radius = model.wheel_radius
	rhs = radius * max_wheel_velocity

	# Each row is (a, b, c, d) for a*vx + b*vy + c*vw = d.
	wheel_forms = [
		(1.0, -1.0, -l_plus_w),
		(1.0, 1.0, l_plus_w),
		(1.0, 1.0, -l_plus_w),
		(1.0, -1.0, l_plus_w),
	]

	planes = []
	for a, b, c in wheel_forms:
		planes.append((a, b, c, rhs))
		planes.append((a, b, c, -rhs))
	return planes


def _build_inequalities(model, max_wheel_velocity):
	"""Build A, b for half-space system A @ [vx, vy, vw] <= b."""
	l_plus_w = model.wb_hlength + model.wb_hwidth
	radius = model.wheel_radius
	rhs = radius * max_wheel_velocity

	forms = np.array([
		[1.0, -1.0, -l_plus_w],
		[1.0, 1.0, l_plus_w],
		[1.0, 1.0, -l_plus_w],
		[1.0, -1.0, l_plus_w],
	])

	# |f_i(x)| <= rhs  <=>  f_i(x) <= rhs and -f_i(x) <= rhs.
	A = np.vstack([forms, -forms])
	b = np.full(A.shape[0], rhs)
	return A, b


def _compute_polytope_vertices(A, b, atol=1e-9):
	"""Compute all vertices of the 3D polytope given A @ x <= b."""
	vertices = []
	for i, j, k in combinations(range(A.shape[0]), 3):
		M = np.vstack([A[i], A[j], A[k]])
		det = np.linalg.det(M)
		if np.isclose(det, 0.0, atol=atol):
			continue

		x = np.linalg.solve(M, np.array([b[i], b[j], b[k]]))
		if np.all(A @ x <= b + atol):
			if not any(np.allclose(x, v, atol=1e-8) for v in vertices):
				vertices.append(x)

	if not vertices:
		return np.empty((0, 3))
	return np.array(vertices)


def _build_face_patches(vertices, A, b, atol=1e-8):
	"""Return ordered polygon vertices for each active half-space boundary."""
	faces = []
	for idx in range(A.shape[0]):
		n = A[idx]
		d = b[idx]
		on_face_mask = np.isclose(vertices @ n, d, atol=atol)
		face_pts = vertices[on_face_mask]
		if face_pts.shape[0] < 3:
			continue

		center = np.mean(face_pts, axis=0)
		n_unit = n / np.linalg.norm(n)
		ref = np.array([1.0, 0.0, 0.0])
		if np.abs(np.dot(n_unit, ref)) > 0.95:
			ref = np.array([0.0, 1.0, 0.0])

		u = np.cross(n_unit, ref)
		u = u / np.linalg.norm(u)
		v = np.cross(n_unit, u)

		rel = face_pts - center
		angles = np.arctan2(rel @ v, rel @ u)
		ordered_pts = face_pts[np.argsort(angles)]
		faces.append((idx, ordered_pts))

	return faces


def _sample_feasible_bodyv(model, max_wheel_velocity, range_scale, samples):
	"""Sample body velocity space and keep points satisfying wheel-speed limits."""
	l_plus_w = model.wb_hlength + model.wb_hwidth
	radius = model.wheel_radius

	max_vx = radius * max_wheel_velocity
	max_vy = radius * max_wheel_velocity
	max_vw = radius * max_wheel_velocity / l_plus_w

	vx_vals = np.linspace(-range_scale * max_vx, range_scale * max_vx, samples)
	vy_vals = np.linspace(-range_scale * max_vy, range_scale * max_vy, samples)
	vw_vals = np.linspace(-range_scale * max_vw, range_scale * max_vw, samples)
	vx_grid, vy_grid, vw_grid = np.meshgrid(vx_vals, vy_vals, vw_vals, indexing="ij")

	vx = vx_grid.ravel()
	vy = vy_grid.ravel()
	vw = vw_grid.ravel()

	wheelv = np.column_stack([
		(vx - vy - l_plus_w * vw) / radius,
		(vx + vy + l_plus_w * vw) / radius,
		(vx + vy - l_plus_w * vw) / radius,
		(vx - vy + l_plus_w * vw) / radius,
	])

	feasible_mask = np.all(np.abs(wheelv) <= max_wheel_velocity + 1e-9, axis=1)
	feasible_points = np.column_stack([vx[feasible_mask], vy[feasible_mask], vw[feasible_mask]])
	usage = np.max(np.abs(wheelv[feasible_mask]), axis=1) / max_wheel_velocity

	return feasible_points, usage, (max_vx, max_vy, max_vw)


def _plot_limit_planes_and_volume(
	model,
	max_wheel_velocity,
	range_scale=1.15,
	plane_samples=31,
	volume_samples=24,
	max_volume_points=45000,
):
	"""Plot boundary patches and sampled feasible volume in (vx, vy, vw) space."""
	A, b = _build_inequalities(model, max_wheel_velocity)
	vertices = _compute_polytope_vertices(A, b)
	faces = _build_face_patches(vertices, A, b)
	if vertices.shape[0] == 0 or len(faces) == 0:
		raise ValueError("Could not build limit polytope patches for current parameters")

	feasible_points, usage, (max_vx, max_vy, max_vw) = _sample_feasible_bodyv(
		model=model,
		max_wheel_velocity=max_wheel_velocity,
		range_scale=range_scale,
		samples=volume_samples,
	)

	fig = plt.figure(figsize=(16, 7))
	ax_planes = fig.add_subplot(121, projection="3d")
	ax_volume = fig.add_subplot(122, projection="3d")

	colors = [
		"tab:blue",
		"tab:orange",
		"tab:green",
		"tab:red",
		"tab:purple",
		"tab:brown",
		"tab:pink",
		"tab:gray",
	]

	for idx, face_pts in faces:
		patch = Poly3DCollection(
			[face_pts],
			alpha=0.35,
			facecolor=colors[idx % len(colors)],
			edgecolor="black",
			linewidth=0.8,
		)
		ax_planes.add_collection3d(patch)

	ax_planes.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=14, c="black", alpha=0.8)

	ax_planes.set_title("Boundary Patches")
	ax_planes.set_xlabel("vx [m/s]")
	ax_planes.set_ylabel("vy [m/s]")
	ax_planes.set_zlabel("vw [rad/s]")

	ax_planes.set_xlim(-range_scale * max_vx, range_scale * max_vx)
	ax_planes.set_ylim(-range_scale * max_vy, range_scale * max_vy)
	ax_planes.set_zlim(-range_scale * max_vw, range_scale * max_vw)

	if feasible_points.shape[0] > max_volume_points:
		rng = np.random.default_rng(0)
		chosen = rng.choice(feasible_points.shape[0], size=max_volume_points, replace=False)
		feasible_points = feasible_points[chosen]
		usage = usage[chosen]

	scatter = ax_volume.scatter(
		feasible_points[:, 0],
		feasible_points[:, 1],
		feasible_points[:, 2],
		c=usage,
		cmap="inferno",
		s=8,
		alpha=0.72,
		edgecolors="none",
	)
	ax_volume.set_facecolor("#0e1117")
	ax_volume.xaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
	ax_volume.yaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
	ax_volume.zaxis.pane.set_facecolor((0.10, 0.12, 0.16, 0.95))
	ax_volume.xaxis.pane.set_edgecolor("#8b949e")
	ax_volume.yaxis.pane.set_edgecolor("#8b949e")
	ax_volume.zaxis.pane.set_edgecolor("#8b949e")
	ax_volume.tick_params(colors="#e6edf3")
	ax_volume.set_title("Feasible Body-Velocity Volume (Sampled)")
	ax_volume.set_xlabel("vx [m/s]")
	ax_volume.set_ylabel("vy [m/s]")
	ax_volume.set_zlabel("vw [rad/s]")
	ax_volume.title.set_color("#f0f6fc")
	ax_volume.xaxis.label.set_color("#e6edf3")
	ax_volume.yaxis.label.set_color("#e6edf3")
	ax_volume.zaxis.label.set_color("#e6edf3")
	ax_volume.set_xlim(-range_scale * max_vx, range_scale * max_vx)
	ax_volume.set_ylim(-range_scale * max_vy, range_scale * max_vy)
	ax_volume.set_zlim(-range_scale * max_vw, range_scale * max_vw)
	colorbar = fig.colorbar(scatter, ax=ax_volume, fraction=0.04, pad=0.07)
	colorbar.set_label("max(|wheel velocity|) / wheel limit")
	colorbar.outline.set_edgecolor("#8b949e")
	colorbar.ax.yaxis.set_tick_params(color="#e6edf3", labelcolor="#e6edf3")
	colorbar.ax.yaxis.label.set_color("#e6edf3")

	# Custom legend entries for surfaces.
	legend_handles = [
		plt.Line2D([0], [0], linestyle="", marker="s", color=colors[i % len(colors)], label=f"face {i + 1}")
		for i in range(A.shape[0])
	]
	ax_planes.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
	fig.suptitle("Mecanum Body-Velocity Limits from Max Wheel Velocity", y=0.97)
	plt.tight_layout()
	plt.show()


def main():
	parser = argparse.ArgumentParser(
		description="Plot 3D body-velocity limit planes for a mecanum drive given max wheel velocity."
	)
	parser.add_argument(
		"--max-wheel-velocity",
		type=float,
		default=10.0,
		help="Maximum absolute wheel velocity in rad/s (default: 10.0).",
	)
	parser.add_argument(
		"--range-scale",
		type=float,
		default=1.15,
		help="Axis range multiplier for the plot window (default: 1.15).",
	)
	parser.add_argument(
		"--samples",
		type=int,
		default=31,
		help="Grid resolution per axis for each plane (default: 31).",
	)
	parser.add_argument(
		"--volume-samples",
		type=int,
		default=24,
		help="Sampling resolution per axis for feasible-volume points (default: 24).",
	)
	parser.add_argument(
		"--max-volume-points",
		type=int,
		default=45000,
		help="Max points drawn in the volume plot after random downsampling (default: 45000).",
	)
	args = parser.parse_args()

	if args.max_wheel_velocity <= 0.0:
		raise ValueError("max-wheel-velocity must be positive")
	if args.range_scale <= 0.0:
		raise ValueError("range-scale must be positive")
	if args.samples < 2:
		raise ValueError("samples must be at least 2")
	if args.volume_samples < 2:
		raise ValueError("volume-samples must be at least 2")
	if args.max_volume_points < 1:
		raise ValueError("max-volume-points must be at least 1")

	model = Mecanum()
	_plot_limit_planes_and_volume(
		model=model,
		max_wheel_velocity=args.max_wheel_velocity,
		range_scale=args.range_scale,
		plane_samples=args.samples,
		volume_samples=args.volume_samples,
		max_volume_points=args.max_volume_points,
	)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(130)
