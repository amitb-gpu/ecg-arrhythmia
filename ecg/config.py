from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_dir(cfg: dict, run_name: str) -> Path:
    """Checkpoints/logits for one run: runs/<experiment>/<run_name>."""
    return ROOT / cfg["paths"]["runs_dir"] / cfg["experiment"] / run_name


def results_dir(cfg: dict, run_name: str) -> Path:
    """Committed figures/tables: results/<experiment>/<run_name>."""
    return ROOT / cfg["paths"]["results_dir"] / cfg["experiment"] / run_name
