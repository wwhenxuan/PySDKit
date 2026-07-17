# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 16:18:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Discover and run the full PySDKit unit-test suite.

NOTE:
    This file is intentionally named without the ``test_`` prefix so that
    ``unittest discover -p "test_*.py"`` does not import it.  A module-level
    ``load_tests`` hook would otherwise recurse during discovery.

Usage (from repository root):
    python -m pysdkit.tests.run_all
or:
    python pysdkit/tests/run_all.py
"""
from __future__ import annotations

import os
import sys
import unittest
import warnings
from typing import Optional

warnings.filterwarnings("ignore")


def _tests_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def build_suite(
    loader: Optional[unittest.TestLoader] = None,
    pattern: str = "test_*.py",
) -> unittest.TestSuite:
    """Load all unit tests under pysdkit/tests (including subpackages)."""
    if loader is None:
        loader = unittest.defaultTestLoader
    return loader.discover(start_dir=_tests_dir(), pattern=pattern)


def main() -> int:
    """Run the discovered suite and return a process exit code."""
    repo_root = os.path.abspath(os.path.join(_tests_dir(), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    suite = build_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
