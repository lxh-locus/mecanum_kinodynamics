"""Reusable Matplotlib animation helpers for planar robot trajectories."""

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


class Footprint(Protocol):
    """Interface required to draw a robot footprint at a world pose."""

    def world_corners(self, x: float, y: float, theta: float) -> np.ndarray:
        """Return the footprint's world-frame corner coordinates."""


@dataclass(frozen=True)
class Trajectory:
    """A labeled planar trajectory with a pose at each timestamp.

    ``poses`` has columns ``(x, y, theta)`` in world coordinates. Timestamps
    must be finite and strictly increasing; headings may wrap at +/- pi.
    """

    label: str
    color: str
    times: np.ndarray
    poses: np.ndarray
    _unwrapped_heading: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        poses = np.asarray(self.poses, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a non-empty one-dimensional array")
        if poses.shape != (times.size, 3):
            raise ValueError("poses must have shape (len(times), 3)")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(poses)):
            raise ValueError("times and poses must contain only finite values")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not self.label:
            raise ValueError("label must not be empty")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "_unwrapped_heading", np.unwrap(poses[:, 2]))

    def pose_at(self, timestamp: float) -> np.ndarray:
        """Interpolate a pose, clamping timestamps outside this trajectory."""
        if timestamp <= self.times[0] or self.times.size == 1:
            return self.poses[0].copy()
        if timestamp >= self.times[-1]:
            return self.poses[-1].copy()

        upper = int(np.searchsorted(self.times, timestamp, side="right"))
        lower = upper - 1
        fraction = (timestamp - self.times[lower]) / (self.times[upper] - self.times[lower])
        position = self.poses[lower, :2] + fraction * (
            self.poses[upper, :2] - self.poses[lower, :2]
        )
        heading = self._unwrapped_heading[lower] + fraction * (
            self._unwrapped_heading[upper] - self._unwrapped_heading[lower]
        )
        return np.array([position[0], position[1], heading], dtype=float)

    def path_until(self, timestamp: float) -> np.ndarray:
        """Return the path through ``timestamp`` for drawing a progressive trail."""
        if timestamp < self.times[0]:
            return np.empty((0, 2), dtype=float)

        count = int(np.searchsorted(self.times, timestamp, side="right"))
        path = self.poses[:count, :2]
        current_position = self.pose_at(timestamp)[:2]
        if path.size == 0 or not np.allclose(path[-1], current_position):
            path = np.vstack((path, current_position))
        return path


def _fit_path_axis(axis: Axes, trajectories: Sequence[Trajectory], footprint: Footprint) -> None:
    corners = [
        np.asarray(footprint.world_corners(*pose), dtype=float)
        for trajectory in trajectories
        for pose in trajectory.poses
    ]
    all_corners = np.concatenate(corners, axis=0)
    lower = np.min(all_corners, axis=0)
    upper = np.max(all_corners, axis=0)
    span = np.maximum(upper - lower, 0.1)
    padding = 0.05 * span
    axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    axis.set_aspect("equal", adjustable="box")


def animate_trajectories(
    path_axis: Axes,
    trajectories: Sequence[Trajectory],
    footprint: Footprint,
    *,
    time_axes: Sequence[Axes] = (),
    fps: float = 30.0,
    playback_speed: float = 1.0,
    repeat: bool = False,
    blit: bool = False,
    fit_limits: bool = True,
) -> FuncAnimation:
    """Animate robot footprints, path trails, and optional synchronized cursors.

    Each trajectory may use its own irregular timestamp grid. The animation
    interpolates poses on one shared timeline and holds a finished trajectory
    at its final pose. ``time_axes`` should be time-series axes in the same
    figure as ``path_axis``; a vertical cursor is advanced on each axis.

    Keep the returned ``FuncAnimation`` object alive for as long as the
    animation should be displayed. Call its ``save`` method to export it.
    """
    trajectories = tuple(trajectories)
    time_axes = tuple(time_axes)
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive and finite")
    if any(axis.figure is not path_axis.figure for axis in time_axes):
        raise ValueError("path_axis and time_axes must belong to the same figure")

    start_time = min(trajectory.times[0] for trajectory in trajectories)
    end_time = max(trajectory.times[-1] for trajectory in trajectories)
    frame_count = max(1, int(np.ceil((end_time - start_time) * fps / playback_speed)) + 1)
    frame_times = np.linspace(start_time, end_time, frame_count)

    if fit_limits:
        _fit_path_axis(path_axis, trajectories, footprint)

    patches = []
    trails = []
    for trajectory in trajectories:
        initial_pose = trajectory.pose_at(start_time)
        patch = Polygon(
            footprint.world_corners(*initial_pose),
            closed=True,
            fill=False,
            edgecolor=trajectory.color,
            linewidth=1.8,
            label=trajectory.label,
        )
        patch.set_visible(start_time >= trajectory.times[0])
        path_axis.add_patch(patch)
        patches.append(patch)

        trail, = path_axis.plot([], [], color=trajectory.color, linewidth=1.8, zorder=3)
        trails.append(trail)

    cursors = [axis.axvline(start_time, color="0.25", linewidth=1.0, linestyle="--") for axis in time_axes]
    clock = path_axis.text(
        0.02,
        0.98,
        "",
        transform=path_axis.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )

    animated_artists: tuple[Polygon | Line2D, ...] = (*patches, *trails, *cursors, clock)

    def update(frame_index: int):
        timestamp = float(frame_times[frame_index])
        for trajectory, patch, trail in zip(trajectories, patches, trails):
            pose = trajectory.pose_at(timestamp)
            patch.set_xy(footprint.world_corners(*pose))
            patch.set_visible(timestamp >= trajectory.times[0])
            path = trajectory.path_until(timestamp)
            if path.size:
                trail.set_data(path[:, 0], path[:, 1])
            else:
                trail.set_data([], [])
        for cursor in cursors:
            cursor.set_xdata([timestamp, timestamp])
        clock.set_text(f"t = {timestamp:.3f} s   x{playback_speed:g}")
        return animated_artists

    def initialize():
        return update(0)

    return FuncAnimation(
        path_axis.figure,
        update,
        frames=frame_count,
        init_func=initialize,
        interval=1000.0 / fps,
        repeat=repeat,
        blit=blit,
    )


__all__ = ["Trajectory", "animate_trajectories"]