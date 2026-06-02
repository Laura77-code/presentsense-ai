"""Small helpers for the PresentSense Streamlit app.

The main app intentionally keeps most UI logic in app.py so the project remains
simple for a university final project. This module exists for future app helpers.
"""

from __future__ import annotations

from pathlib import Path


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()
