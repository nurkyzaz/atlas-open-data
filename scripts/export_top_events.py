#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from atlas_anomaly.models import bundle_scores, load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the highest-score rows from a table.")
    parser.add_argument("--model", required=True, help="Path to a trained model .pkl file")
    parser.add_argument("--table", required=True, help="Path to a Parquet table")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", default=None, help="Optional output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.model)
    frame = pd.read_parquet(args.table)
    if len(frame.columns) == 0 or len(frame) == 0:
        raise ValueError(
            "The input table is empty. Rebuild the sample table first and make sure the ROOT read succeeded."
        )
    scores = bundle_scores(bundle, frame)
    ranked = frame.copy()
    ranked["anomaly_score"] = scores
    ranked = ranked.sort_values("anomaly_score", ascending=False).head(args.top_k)

    output_path = Path(args.output) if args.output is not None else Path(args.table).with_suffix(".top_events.csv")
    ranked.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
