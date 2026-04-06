from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from atlas_anomaly.constants import DEFAULT_CONFIG_PATH


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    return json.loads(path.read_text())


def expand_samples(config: dict, groups: Optional[List[str]], samples: Optional[List[str]]) -> List[str]:
    selected: list[str] = []

    if groups:
        for group in groups:
            selected.extend(config["groups"].get(group, []))

    if samples:
        selected.extend(samples)

    seen = set()
    ordered = []
    for sample in selected:
        if sample not in seen:
            ordered.append(sample)
            seen.add(sample)
    return ordered
