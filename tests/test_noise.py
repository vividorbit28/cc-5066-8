"""Determinism and value-range contracts for the noise primitives."""
import unittest

import numpy as np

from tests._setup import *  # noqa: F401,F403  (sys.path bootstrap)
from material_matcher import noise


class TestNoise(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.x = rng.uniform(-3, 3, (32, 32))
        self.y = rng.uniform(-3, 3, (32, 32))

    def test_deterministic_for_seed(self):
        a = noise.fbm(self.x, self.y, seed=5)
        b = noise.fbm(self.x, self.y, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = noise.fbm(self.x, self.y, seed=1)
        b = noise.fbm(self.x, self.y, seed=2)
        self.assertGreater(np.abs(a - b).mean(), 1e-4)

    def test_fbm_in_unit_range_for_all_kinds(self):
        for kind in ("gradient", "value", "cellular"):
            out = noise.fbm(self.x, self.y, seed=3, kind=kind, octaves=4)
            self.assertGreaterEqual(out.min(), 0.0)
            self.assertLessEqual(out.max(), 1.0)
            self.assertTrue(np.isfinite(out).all())

    def test_cellular_non_constant(self):
        out = noise.cellular_noise(self.x, self.y, seed=4)
        self.assertGreater(out.std(), 1e-3)

    def test_octave_counts_distinct(self):
        o1 = noise.fbm(self.x, self.y, seed=4, octaves=1)
        o5 = noise.fbm(self.x, self.y, seed=4, octaves=5)
        # More octaves add detail; the two fields should not coincide.
        self.assertGreater(np.abs(o1 - o5).mean(), 1e-3)


if __name__ == "__main__":
    unittest.main()
