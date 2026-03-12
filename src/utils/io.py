# src/utils/io.py
# File I/O utilities

import json
from pathlib import Path


def ensure_dir(path):
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path (str or Path)
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data, indent=2):
    """
    Write data to JSON file with UTF-8 encoding.

    Args:
        path: Output file path
        JSON-serializable object
        indent: Indentation level (default: 2)
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json(path):
    """
    Read JSON file with UTF-8 encoding.

    Args:
        path: Input file path

    Returns:
        Parsed JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)