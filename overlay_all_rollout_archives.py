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
REDUX_STRATEGY_LABELS = {
    "sliding-coulomb-fixed-axis": "fixed-axis",
    "sliding-coulomb-velocity-axis": "velocity-axis",
    "sliding-coulomb-approx": "coulomb-approx",
    "independent-axis": "independent-axis",
    "discrete-empirical": "discrete-empirical",
}


def archives_by_field(directory, suffix):
    """Return archive paths indexed by their field name for one generator."""
    archives = {}
    for path in directory.glob(f"*{suffix}"):
        field_name = path.name.removesuffix(suffix)
        if field_name in archives:
            raise ValueError(f"Duplicate archive for field '{field_name}' in {directory}")
        archives[field_name] = path
    return archives


def strategy_archive_labels(strategy):
    """Return the canonical archive label followed by its legacy spelling."""
    label = REDUX_STRATEGY_LABELS[strategy]
    legacy_label = label.replace("-", "_")
    return (label,) if legacy_label == label else (label, legacy_label)


def main():
    parser = argparse.ArgumentParser(description="Batch overlay matched barebones and redux rollout archives.")
    parser.add_argument("--barebones-dir", type=Path, default=DEFAULT_BAREBONES_DIR)
    parser.add_argument("--redux-dir", type=Path, default=DEFAULT_REDUX_DIR)
    parser.add_argument(
        "--redux-strategy",
        choices=tuple(REDUX_STRATEGY_LABELS),
        help="Redux strategy archive to collate; auto-detects the only available strategy when omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Print paired archives without rendering images.")
    args = parser.parse_args()

    barebones = archives_by_field(args.barebones_dir, "_rollouts.npz")
    available_strategies = {
        strategy
        for strategy in REDUX_STRATEGY_LABELS
        if any(
            args.redux_dir.glob(f"*_{label}_rollouts.npz")
            for label in strategy_archive_labels(strategy)
        )
    }
    if args.redux_strategy is None and len(available_strategies) > 1:
        available_strategies = sorted(available_strategies)
        raise ValueError(
            "Multiple redux strategies are available: "
            + ", ".join(available_strategies)
            + ". Pick one with --redux-strategy <strategy>."
        )
    selected_strategy = args.redux_strategy
    if selected_strategy is None and len(available_strategies) == 1:
        selected_strategy = next(iter(available_strategies))
        print(f"Auto-detected redux strategy: {selected_strategy}")
    if selected_strategy is None:
        selected_strategy = "sliding-coulomb-fixed-axis"
    redux = {}
    for strategy_label in strategy_archive_labels(selected_strategy):
        redux = archives_by_field(args.redux_dir, f"_{strategy_label}_rollouts.npz")
        if redux:
            break
    if not redux:
        legacy_redux = archives_by_field(args.redux_dir, "_sliding_rollouts.npz")
        if legacy_redux:
            print("Using legacy redux archive names ending in '_sliding_rollouts.npz'.")
            redux = legacy_redux
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