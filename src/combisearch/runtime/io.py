"""Small runtime IO helpers."""

from __future__ import annotations

import json
import logging
import os
import pkgutil
from pathlib import Path
from typing import Any, Union

from combisearch.config.paths import COMBISEARCH_DATA_DIR, resolve_data_path


def read_json(file_name: Union[str, Path]) -> Any:
    with open(file_name, 'r') as f:
        return json.load(f)


def read_json_from_data_dir(file_name: Union[str, Path]) -> Any:
    resolved_path = resolve_data_path(file_name)
    try:
        return read_json(resolved_path)
    except FileNotFoundError:
        logging.error(
            f"data file not found. Use absolute paths or set {COMBISEARCH_DATA_DIR} "
            "correctly. "
            f"({COMBISEARCH_DATA_DIR}={os.environ.get(COMBISEARCH_DATA_DIR)}, "
            f"resolved_path={resolved_path})"
        )
        raise


def read_package_text(relative_path: str) -> str:
    """Read a package-bundled file as a string (e.g. 'domain/multiwoz/db/schema.json')."""
    return pkgutil.get_data("combisearch", relative_path).decode("utf-8")


def read_json_resource(relative_path: str) -> Any:
    return json.loads(read_package_text(relative_path))
