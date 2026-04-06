#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from atlas_anomaly.config import expand_samples, load_config
from atlas_anomaly.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_DIR
from atlas_anomaly.io import get_sample_metadata, get_sample_urls, read_root_arrays
from atlas_anomaly.physics import build_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build flat event tables from ATLAS Open Data PHYSLITE files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--group", action="append", default=[], help="Sample group name from configs/samples.json")
    parser.add_argument("--sample", action="append", default=[], help="Explicit sample key to build")
    parser.add_argument("--max-files-per-sample", type=int, default=1)
    parser.add_argument("--max-events-per-file", type=int, default=2000)
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--retries", type=int, default=3, help="How many times to retry a failed remote ROOT read.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Base wait in seconds before retrying a failed read.")
    parser.add_argument("--sleep-between-files", type=float, default=3.0, help="Pause in seconds between files to reduce HTTP rate limiting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sample_keys = expand_samples(config, args.group, args.sample)
    if not sample_keys:
        raise ValueError("No samples were selected. Use --group or --sample.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_key in sample_keys:
        metadata = get_sample_metadata(sample_key)
        urls = get_sample_urls(sample_key, max_files=args.max_files_per_sample)
        role = config["samples"].get(sample_key, {}).get("role", "unclassified")
        sample_label = config["samples"].get(sample_key, {}).get("label", metadata.get("process"))
        sample_info = {
            "sample_label": sample_label,
            "sample_role": role,
            "process": metadata.get("process"),
            "physics_short": metadata.get("physics_short"),
            "cross_section_pb": metadata.get("cross_section_pb"),
        }

        print(f"Building sample {sample_key}: {metadata.get('physics_short')}")
        print(f"  process      = {metadata.get('process')}")
        print(f"  files        = {len(urls)}")
        print(f"  events/file  = {args.max_events_per_file}")

        rows = []
        for file_index, url in enumerate(urls):
            print(f"  reading file {file_index + 1}/{len(urls)}")
            try:
                arrays = read_root_arrays(
                    url,
                    max_events=args.max_events_per_file,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
            except Exception as exc:
                print(f"  skipping file {file_index + 1} because reading failed: {exc}")
                continue
            for event_index in range(len(arrays["event_number"])):
                row = build_row(event_index, arrays, sample_key, sample_info, file_index)
                if row is not None:
                    rows.append(row)
            if file_index + 1 < len(urls) and args.sleep_between_files > 0:
                print(f"  sleeping {args.sleep_between_files:.1f}s before the next file")
                time.sleep(args.sleep_between_files)

        frame = pd.DataFrame(rows)
        output_path = output_dir / f"{sample_key}.parquet"
        if len(frame) == 0:
            if output_path.exists():
                print(f"  built zero rows, so I kept the existing file unchanged: {output_path}")
            else:
                print(f"  built zero rows, so no output file was written for {sample_key}")
            continue
        frame.to_parquet(output_path, index=False)
        print(f"  wrote {len(frame)} selected rows to {output_path}")


if __name__ == "__main__":
    main()
