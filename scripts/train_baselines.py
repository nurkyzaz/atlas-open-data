#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_anomaly.config import expand_samples, load_config
from atlas_anomaly.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR
from atlas_anomaly.io import load_tables
from atlas_anomaly.models import fit_all_baselines, save_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline anomaly-detection models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--group", action="append", default=["normal_train"], help="Groups used as ordinary training data")
    parser.add_argument("--sample", action="append", default=[], help="Explicit ordinary sample keys")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--false-positive-rate", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sample_keys = expand_samples(config, args.group, args.sample)
    ordinary = load_tables(args.data_dir, sample_keys)
    if len(ordinary) == 0:
        raise ValueError("The ordinary training tables contain zero rows. Build at least one normal sample table first.")

    train_frame = ordinary[ordinary["split"] == "train"].reset_index(drop=True)
    val_frame = ordinary[ordinary["split"] == "val"].reset_index(drop=True)
    test_frame = ordinary[ordinary["split"] == "test"].reset_index(drop=True)

    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        shuffled = ordinary.sample(frac=1.0, random_state=args.random_state).reset_index(drop=True)
        train_end = max(1, int(0.7 * len(shuffled)))
        val_end = max(train_end + 1, int(0.85 * len(shuffled)))
        train_frame = shuffled.iloc[:train_end].reset_index(drop=True)
        val_frame = shuffled.iloc[train_end:val_end].reset_index(drop=True)
        test_frame = shuffled.iloc[val_end:].reset_index(drop=True)
        if len(test_frame) == 0 and len(shuffled) >= 3:
            test_frame = shuffled.iloc[-1:].reset_index(drop=True)
            val_frame = shuffled.iloc[train_end:-1].reset_index(drop=True)

    print(f"ordinary rows: total={len(ordinary)} train={len(train_frame)} val={len(val_frame)} test={len(test_frame)}")
    bundles = fit_all_baselines(
        train_frame=train_frame,
        val_frame=val_frame,
        false_positive_rate=args.false_positive_rate,
        random_state=args.random_state,
    )

    for model_name, bundle in bundles.items():
        bundle["test_rows"] = len(test_frame)
        path = f"{args.model_dir}/{model_name}.pkl"
        save_bundle(path, bundle)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
