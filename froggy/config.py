"""YAML-based config persistence for froggy.

Stores user preferences at ``<froggy_home>/config.yaml``.
All functions are pure helpers — no CLI or Rich dependencies.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .paths import froggy_home


def config_path() -> Path:
    """Return the path to the froggy config file."""
    return froggy_home() / "config.yaml"


def load_config() -> dict:
    """Load the config file and return its contents as a dict.

    Returns an empty dict when the file is missing, empty, or contains
    non-dict YAML (never raises on bad input).
    """
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = yaml.safe_load(p.read_text())
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_config(data: dict) -> None:
    """Write *data* as YAML to the config file, creating parent dirs if needed."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, default_flow_style=False))


def get_config(key: str, default=None):
    """Return the value for *key* from the config, or *default* if absent."""
    return load_config().get(key, default)


def set_config(key: str, value) -> None:
    """Set a single *key* to *value* in the config file.

    Preserves all other existing keys. Creates the file if it does not exist.
    """
    data = load_config()
    data[key] = value
    save_config(data)
