#!/usr/bin/env python3
"""Create standard and fused overlays for matched barebones and redux archives."""
import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_BAREBONES_DIR = REPOSITORY_ROOT / "fieldset_generator_barebones" / "rollout_data"
DEFAULT_REDUX_DIR = REPOSITORY_ROOT / "fieldset_generator_redux1" / "rollout_data"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "rollout_overlays"
OVERLAY_SCRIPT = REPOSITORY_ROOT / "overlay_rollout_archives.py"


def archives_by_field(directory, suffix):
    """Return archive paths indexed by their field name for one generator."""
    archives = {}
    for path in directory.glob(f"*{suffix}"):
        field_name = path.name.removesuffix(suffix)
        if field_name in archives:
            raise ValueError(f"Duplicate archive for field '{field_name}' in {directory}")
        archives[field_name] = path
    return archives


def main():
    parser = argparse.ArgumentParser(description="Batch overlay matched barebones and redux rollout archives.")
    parser.add_argument("--barebones-dir", type=Path, default=DEFAULT_BAREBONES_DIR)
    parser.add_argument("--redux-dir", type=Path, default=DEFAULT_REDUX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print paired archives without rendering images.")
    args = parser.parse_args()

    barebones = archives_by_field(args.barebones_dir, "_rollouts.npz")
    redux = archives_by_field(args.redux_dir, "_sliding_rollouts.npz")
    common_fields = sorted(barebones.keys() & redux.keys())
    if not common_fields:
        raise ValueError("No matching barebones and redux rollout archives found")
    unmatched_barebones = sorted(barebones.keys() - redux.keys())
    unmatched_redux = sorted(redux.keys() - barebones.keys())
    if unmatched_barebones:
        print(f"Skipping barebones-only fields: {', '.join(unmatched_barebones)}")
    if unmatched_redux:
        print(f"Skipping redux-only fields: {', '.join(unmatched_redux)}")

    for field_name in common_fields:
        standard_output = args.output_dir / f"{field_name}_rollout.png"
        fused_output = args.output_dir / f"{field_name}_rollout_fused.png"
        print(f"{field_name}: {barebones[field_name].name} + {redux[field_name].name}")
        if args.dry_run:
            continue
        args.output_dir.mkdir(parents=True, exist_ok=True)
        overlay_args = [
            sys.executable,
            str(OVERLAY_SCRIPT),
            "--archive",
            str(barebones[field_name]),
            "tab:blue",
            "--archive",
            str(redux[field_name]),
            "tab:red",
        ]
        subprocess.run([*overlay_args, "--output", str(standard_output)], check=True)
        subprocess.run([*overlay_args, "--fuse-footprints", "--output", str(fused_output)], check=True)


if __name__ == "__main__":
    main()