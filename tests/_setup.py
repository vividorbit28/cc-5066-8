"""Shared helpers for the test suite."""
import os
import sys

# Make the repository root importable when tests run directly (``python3
# tests/test_x.py``) as well as through a test runner.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Small renders keep the search-driven tests fast while still exercising the
# full multi-scale loss and witness pipeline.
FAST_SIZE = 40
FAST_CONFIGS = 3
