"""Managed home directory resolution for froggy.

Provides functions to resolve the froggy home directory (~/.froggy) and
its subdirectories. The home location can be overridden via the FROGGY_HOME
environment variable.

Only stdlib imports — no froggy-internal dependencies to avoid circular imports.
"""

from __future__ import annotations

import os
from pathlib import Path


def froggy_home() -> Path:
    """Return the froggy home directory.

    Reads ``FROGGY_HOME`` env var; falls back to ``~/.froggy``.
    """
    env = os.environ.get("FROGGY_HOME")
    if env:
        return Path(env)
    return Path.home() / ".froggy"


def models_dir() -> Path:
    """Return the managed models directory (``<froggy_home>/models``)."""
    return froggy_home() / "models"


def ensure_froggy_home() -> Path:
    """Create the froggy home and models directories if they don't exist.

    Returns the home path. Idempotent — safe to call multiple times.
    """
    home = froggy_home()
    home.mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)
    return home
