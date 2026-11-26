"""Witness decal: object/world divergence, boundary error, second scale."""
import unittest

import numpy as np

from tests._setup import FAST_SIZE
from material_matcher.decals import (boundary_discontinuity,
                                     render_in_space, run_witness)
from material_matcher.templates import build_chain, procedural_target


class TestWitnessDecal(unittest.TestCase):
    def test_object_space_is_invariant_under_translation(self):
        g = build_chain(seed=3)
        base = render_in_space(g, FAST_SIZE, 1.0, "object")
        moved = render_in_space(g, FAST_SIZE, 1.0, "object", 0.4, 0.25)
        # Locked to the mesh: moving the world leaves the object decal identical.
        np.testing.assert_allclose(moved, base, atol=1e-12)

    def test_world_space_changes_under_translation(self):
        g = build_chain(seed=3)
        base = render_in_space(g, FAST_SIZE, 1.0, "world")
        moved = render_in_space(g, FAST_SIZE, 1.0, "world", 0.4, 0.25)
        self.assertGreater(np.abs(moved - base).mean(), 1e-3)

    def test_boundary_discontinuity_nonneg_and_bounded(self):
        img = np.random.default_rng(0).random((FAST_SIZE, FAST_SIZE, 3))
        err = boundary_discontinuity(img, border=6)
        self.assertGreaterEqual(err, 0.0)
        self.assertLessEqual(err, 1.0)

    def test_witness_resolves_object_when_pattern_is_locked(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=3)
        g_obj = build_chain(seed=3, space="object")
        g_wld = build_chain(seed=3, space="world")
        report = run_witness(g_obj, g_wld, target,
                             shift=(0.4, 0.25), size=FAST_SIZE)
        # Object-space decal does not move, so its motion residual is zero.
        self.assertEqual(report.object_motion_residual, 0.0)
        self.assertGreater(report.world_motion_residual, 0.0)
        self.assertEqual(report.resolved_space, "object")
        # A second-scale observation is always reported to guard aliasing.
        self.assertGreater(report.second_scale_uv, 1.0)
        self.assertGreaterEqual(report.second_scale_motion_residual, 0.0)

    def test_parameter_acceptance_and_rejection(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=3)
        g_obj = build_chain(seed=3, space="object")
        g_wld = build_chain(seed=3, space="world")
        # Make the two graphs genuinely differ on one continuous parameter.
        g_wld.set_param("noise", "scale", 11.0)
        report = run_witness(g_obj, g_wld, target,
                             accept_threshold=0.05, size=FAST_SIZE)
        rejected_names = {p["param"] for p in report.rejected_params}
        self.assertIn("noise.scale", rejected_names)
        # Most parameters still match and are accepted.
        self.assertGreater(len(report.accepted_params),
                           len(report.rejected_params))

    def test_report_is_json_serializable(self):
        import json
        target, _ = procedural_target(size=FAST_SIZE, seed=2)
        report = run_witness(build_chain(seed=2, space="object"),
                             build_chain(seed=2, space="world"),
                             target, size=FAST_SIZE)
        payload = report.to_dict()
        json.dumps(payload)  # raises if not serializable
        self.assertIn("accepted_params", payload)


if __name__ == "__main__":
    unittest.main()
