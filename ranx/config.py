"""Configuration system for ranx library."""

import os
from typing import Optional

# Global configuration state
_USE_NUMBA: Optional[bool] = None


def use_numba() -> bool:
    """
    Check if Numba should be used for performance optimizations.

    Returns True by default, but can be disabled via:
    1. Environment variable: RANX_USE_NUMBA=false
    2. Programmatically: set_numba_enabled(False)

    Returns:
        bool: True if Numba should be used, False otherwise
    """
    global _USE_NUMBA
    if _USE_NUMBA is None:
        env_value = os.environ.get("RANX_USE_NUMBA", "true").lower()
        _USE_NUMBA = env_value not in ("false", "0", "no", "off")
    return _USE_NUMBA


def set_numba_enabled(enabled: bool) -> None:
    """
    Programmatically enable or disable Numba usage.

    Args:
        enabled: True to enable Numba, False to disable
    """
    global _USE_NUMBA
    _USE_NUMBA = enabled


def reset_numba_config() -> None:
    """
    Reset Numba configuration to default (reads from environment again).

    Primarily used for testing purposes.
    """
    global _USE_NUMBA
    _USE_NUMBA = None
