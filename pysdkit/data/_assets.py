# -*- coding: utf-8 -*-
"""Resolve packaged array / text assets under ``pysdkit/data``."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REAL_WORLD_DIR = DATA_DIR / "real_world"


def data_file(name: str) -> Path:
    """
    Return the path to a file shipped in ``pysdkit/data``.

    ``.npy`` records live in ``pysdkit/data/real_world``.  Other assets
    (e.g. ``texture.txt``) stay at the package root.  ``name`` may be a
    bare file name or a path relative to ``pysdkit/data``.

    :param name: File name relative to ``pysdkit/data``.
    :return: Absolute path to the packaged file.
    :raises FileNotFoundError: If the asset is missing from the install.
    """
    relative = Path(name)
    candidates = []
    if relative.suffix.lower() == ".npy":
        candidates.append(REAL_WORLD_DIR / relative.name)
        candidates.append(DATA_DIR / relative)
        if relative.parts and relative.parts[0] != "real_world":
            candidates.append(DATA_DIR / "real_world" / relative)
    else:
        candidates.append(DATA_DIR / relative)

    for path in candidates:
        if path.is_file():
            return path

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Missing packaged data: {}. Reinstall PySDKit or restore "
        "pysdkit/data (looked in: {})".format(name, tried)
    )
