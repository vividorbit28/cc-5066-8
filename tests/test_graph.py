"""Graph construction, evaluation, validation and JSON round-trip."""
import json
import unittest

import numpy as np

from tests._setup import *  # noqa: F401,F403
from material_matcher.graph import MaterialGraph, spec_index
from material_matcher.templates import build_chain, build_blend_graph


class TestGraph(unittest.TestCase):
    def test_chain_evaluates_to_rgb(self):
        g = build_chain(seed=2)
        x, y = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24))
        img = g.evaluate(x, y)
        self.assertEqual(img.shape, (24, 24, 3))
        self.assertTrue(((img >= 0) & (img <= 1)).all())

    def test_blend_evaluates_without_broadcast_leak(self):
        g = build_blend_graph()
        x, y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
        img = g.evaluate(x, y)
        self.assertEqual(img.shape, (16, 16, 3))

    def test_json_round_trip_preserves_params(self):
        g = build_chain(seed=7)
        g.set_param("noise", "scale", 8.5)
        g.set_param("output", "space", "world")
        restored = MaterialGraph.from_dict(json.loads(json.dumps(g.to_dict())))
        self.assertEqual(restored.get_param("noise", "scale"), 8.5)
        self.assertEqual(restored.get_param("output", "space"), "world")
        x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        np.testing.assert_allclose(restored.evaluate(x, y), g.evaluate(x, y))

    def test_string_discrete_clamp(self):
        g = build_chain()
        g.set_param("noise", "kind", "value")
        self.assertEqual(g.get_param("noise", "kind"), "value")
        g.set_param("output", "space", "world")
        self.assertEqual(g.get_param("output", "space"), "world")

    def test_numeric_index_maps_to_string_choice(self):
        # Regression: a vector-encoded integer index must resolve to the
        # matching string choice, not be silently dropped (which previously
        # left a hand-written "truth" stuck on the default kind).
        g = build_chain()
        g.set_param("noise", "kind", 1)
        self.assertEqual(g.get_param("noise", "kind"), "value")
        g.set_param("noise", "kind", 2)
        self.assertEqual(g.get_param("noise", "kind"), "cellular")
        g.set_param("output", "space", 1)
        self.assertEqual(g.get_param("output", "space"), "world")

    def test_numeric_discrete_clamp_rounds(self):
        g = build_chain()
        g.set_param("noise", "octaves", 3.4)
        self.assertEqual(g.get_param("noise", "octaves"), 3)

    def test_continuous_clamp_to_range(self):
        g = build_chain()
        g.set_param("noise", "scale", 999.0)
        self.assertLessEqual(g.get_param("noise", "scale"), 12.0)

    def test_validate_rejects_cycle_and_bad_arity(self):
        good = build_chain()
        self.assertTrue(good.validate())
        broken = MaterialGraph()
        broken.add("noise", "n")
        broken.add("output", "o", inputs=("n",))
        # Force a dangling input by mutating the node directly.
        broken.nodes["o"].inputs = ["missing"]
        with self.assertRaises(ValueError):
            broken.validate()

    def test_unknown_param_rejected(self):
        g = build_chain()
        with self.assertRaises(ValueError):
            g.add("noise", "bad", {"nope": 1.0})

    def test_param_vector_discrete_encoding(self):
        g = build_chain()
        g.set_param("noise", "kind", "cellular")  # index 2
        vec = g.param_vector()
        kind_idx = next(i for i, (nid, spec) in enumerate(spec_index(g))
                        if nid == "noise" and spec.name == "kind")
        self.assertEqual(vec[kind_idx], 2.0)


if __name__ == "__main__":
    unittest.main()
