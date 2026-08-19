# -*- coding: utf-8 -*-
"""Resolve packaged array / text assets under ``pysdkit/data``."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


def data_file(name: str) -> Path:
    """
    Return the path to a file shipped in ``pysdkit/data``.

    :param name: File name relative to ``pysdkit/data``.
    :return: Absolute path to the packaged file.
    :raises FileNotFoundError: If the asset is missing from the install.
    """
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            "Missing packaged data: {}. Reinstall PySDKit or restore "
            "pysdkit/data/{}".format(path, name)
        )
    return path
