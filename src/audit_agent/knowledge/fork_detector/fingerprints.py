"""Compute and look up normalized source code fingerprints.

normalize_source() strips comments and collapses whitespace so that
two files with cosmetic differences produce the same hash.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize_source(source: str) -> str:
    """Strip Solidity comments and normalise whitespace for hashing."""
    # Remove // line comments
    source = re.sub(r"//[^\n]*", "", source)
    # Remove /* ... */ block comments (non-greedy, multi-line)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Collapse whitespace
    source = re.sub(r"\s+", " ", source).strip()
    return source


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of the normalized source of a file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return compute_text_hash(text)


def compute_text_hash(source: str) -> str:
    """Return the SHA-256 hex digest of the normalized source string."""
    normalised = normalize_source(source)
    return hashlib.sha256(normalised.encode()).hexdigest()


def hash_project_files(directory: Path, glob: str = "**/*.sol") -> dict[str, str]:
    """Compute normalized hashes for all matching files in a directory.

    Returns: {relative_path_str -> hash}
    """
    result: dict[str, str] = {}
    for path in sorted(directory.glob(glob)):
        rel = str(path.relative_to(directory))
        result[rel] = compute_file_hash(path)
    return result
