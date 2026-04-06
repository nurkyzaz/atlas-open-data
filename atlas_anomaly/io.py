from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import List, Optional, Union

import atlasopenmagic as atom
import pandas as pd
import uproot

from atlas_anomaly.constants import BRANCH_ALIASES, DEFAULT_DATA_DIR


def get_sample_metadata(sample_key: str) -> dict:
    return atom.get_metadata(sample_key)


def get_sample_urls(sample_key: str, max_files: Optional[int] = None) -> List[str]:
    urls = atom.get_urls(sample_key, protocol="https", cache=True)
    cleaned = [url.replace("simplecache::", "") for url in urls]
    if max_files is not None:
        return cleaned[:max_files]
    return cleaned


def read_root_arrays(
    url: str,
    max_events: Optional[int] = None,
    retries: int = 3,
    retry_sleep: float = 10.0,
) -> dict:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with uproot.open({url: "CollectionTree"}) as tree:
                arrays = tree.arrays(
                    list(BRANCH_ALIASES),
                    aliases=BRANCH_ALIASES,
                    entry_stop=max_events,
                    library="np",
                )
            return arrays
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            sleep_seconds = retry_sleep * attempt
            print(
                f"    read attempt {attempt}/{retries} failed: {exc}. "
                f"Waiting {sleep_seconds:.1f}s before retry."
            )
            time.sleep(sleep_seconds)
    raise last_exc


def stable_bucket(value: str, modulo: int = 100) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def assign_split(frame: pd.DataFrame) -> pd.Series:
    unique_sources = frame["source_uid"].nunique()
    split_key = "source_uid" if unique_sources >= 3 else "row_uid"
    buckets = frame[split_key].map(lambda value: stable_bucket(str(value)))
    return pd.Series(
        ["train" if x < 70 else "val" if x < 85 else "test" for x in buckets],
        index=frame.index,
        name="split",
    )


def load_tables(data_dir: Optional[Union[str, Path]], sample_keys: List[str]) -> pd.DataFrame:
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    frames = []
    for sample_key in sample_keys:
        path = base_dir / f"{sample_key}.parquet"
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    nonempty_frames = [frame for frame in frames if len(frame.columns) > 0]
    if not nonempty_frames:
        return pd.DataFrame()
    frame = pd.concat(nonempty_frames, ignore_index=True)
    if "split" not in frame.columns and "source_uid" in frame.columns and "row_uid" in frame.columns:
        frame["split"] = assign_split(frame)
    return frame
