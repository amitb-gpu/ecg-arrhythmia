from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
