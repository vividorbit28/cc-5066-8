"""Multi-scale loss: self-similarity is zero and mean-colour collapse is punished."""
import unittest

import numpy as np

from tests._setup import *  # noqa: F401,F403
from material_matcher.render import (appearance_summary, loss_terms,
                                     multiscale_loss, render_graph)
from material_matcher.templates import build_chain, procedural_target


class TestLoss(unittest.TestCase):
    def test_self_loss_is_zero(self):
        target, _ = procedural_target(size=32, seed=2)
        g = build_chain(seed=2)
        loss, detail, render, mean_warn = multiscale_loss(g, target)
        self.assertLess(loss, 1e-9)
        np.testing.assert_allclose(render, target, atol=1e-9)
        self.assertFalse(mean_warn)

    def test_identical_arrays_zero_components(self):
        a = np.random.default_rng(0).random((24, 24, 3))
        total, comp = loss_terms(a, a)
        self.assertAlmostEqual(total, 0.0, places=8)
        for key in ("pixel", "histogram", "gradient", "spectrum"):
            self.assertEqual(comp[key], 0.0)
        self.assertFalse(comp["mean_colour_collapse"])

    def test_mean_colour_collapse_is_penalised(self):
        rng = np.random.default_rng(1)
        textured = rng.random((32, 32, 3))
        flat = np.ones_like(textured) * textured.mean(axis=(0, 1))
        total, comp = loss_terms(flat, textured)
        self.assertTrue(comp["mean_colour_collapse"])
        # The explicit 0.15 penalty must be present in the total.
        weights = {"pixel": 1, "histogram": .5, "gradient": .5,
                   "spectrum": .8}
        base = sum(comp[k] * weights[k]
                   for k in ("pixel", "histogram", "gradient", "spectrum"))
        self.assertAlmostEqual(total, base + 0.15, places=6)

    def test_spectrum_distinguishes_frequency(self):
        target, _ = procedural_target(
            {"noise.scale": 9.0}, size=40, seed=4)
        g = build_chain(seed=4)
        g.set_param("noise", "scale", 2.0)
        _, comp = loss_terms(render_graph(g, 40), target)
        self.assertGreater(comp["spectrum"], 0.02)

    def test_appearance_summary_shape(self):
        target, _ = procedural_target(size=24, seed=1)
        s = appearance_summary(target)
        self.assertEqual(len(s["mean_rgb"]), 3)
        self.assertIn("luma_std", s)


if __name__ == "__main__":
    unittest.main()
