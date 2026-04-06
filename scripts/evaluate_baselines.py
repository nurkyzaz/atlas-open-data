#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from atlas_anomaly.config import expand_samples, load_config
from atlas_anomaly.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_DIR, DEFAULT_MODEL_DIR, DEFAULT_RESULTS_DIR
from atlas_anomaly.io import load_tables
from atlas_anomaly.models import bundle_scores, compare_scores, load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline anomaly-detection models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--normal-group", action="append", default=["normal_train"])
    parser.add_argument("--robustness-group", action="append", default=["normal_generator_check"])
    parser.add_argument("--anomaly-group", action="append", default=["anomaly_proxy"])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


def plot_score_histogram(normal_scores, compare_scores_array, title: str, outfile: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=40, alpha=0.7, label="ordinary test", color="#2563eb")
    plt.hist(compare_scores_array, bins=40, alpha=0.7, label="comparison sample", color="#dc2626")
    plt.xlabel("Anomaly score")
    plt.ylabel("Rows")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    normal_samples = expand_samples(config, args.normal_group, [])
    robustness_samples = expand_samples(config, args.robustness_group, [])
    anomaly_samples = expand_samples(config, args.anomaly_group, [])

    ordinary = load_tables(args.data_dir, normal_samples)
    ordinary_test = ordinary[ordinary["split"] == "test"].reset_index(drop=True)
    robustness = load_tables(args.data_dir, robustness_samples) if robustness_samples else None
    if robustness is not None and len(robustness) == 0:
        robustness = None

    metrics: dict[str, dict] = {}

    for model_path in sorted(Path(args.model_dir).glob("*.pkl")):
        bundle = load_bundle(model_path)
        model_name = bundle["model_name"]
        threshold = bundle["threshold"]

        model_metrics: dict[str, dict] = {}
        normal_scores = bundle_scores(bundle, ordinary_test)

        if robustness is not None and len(robustness) > 0:
            robustness_scores = bundle_scores(bundle, robustness)
            model_metrics["robustness"] = {
                "mean_score": float(robustness_scores.mean()),
                "flag_rate": float((robustness_scores >= threshold).mean()),
            }
            plot_score_histogram(
                normal_scores,
                robustness_scores,
                title=f"{model_name}: ordinary test vs alternate-generator ttbar",
                outfile=results_dir / f"{model_name}_robustness.png",
            )

        for sample_key in anomaly_samples:
            sample_path = Path(args.data_dir) / f"{sample_key}.parquet"
            if not sample_path.exists():
                print(f"skipping {sample_key} because {sample_path} does not exist")
                continue
            sample_frame = load_tables(args.data_dir, [sample_key])
            if len(sample_frame) == 0:
                print(f"skipping {sample_key} because the table has zero rows")
                continue
            anomaly_scores = bundle_scores(bundle, sample_frame)
            comparison = compare_scores(normal_scores, anomaly_scores, threshold)
            comparison["rows"] = int(len(sample_frame))
            comparison["label"] = str(sample_frame["sample_label"].iloc[0])
            model_metrics[sample_key] = comparison
            plot_score_histogram(
                normal_scores,
                anomaly_scores,
                title=f"{model_name}: ordinary test vs {sample_key}",
                outfile=results_dir / f"{model_name}_{sample_key}.png",
            )

        metrics[model_name] = model_metrics

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
