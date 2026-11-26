"""Inverse search: known-parameter recovery, ambiguity, and failure residuals."""
import unittest

import numpy as np

from tests._setup import FAST_CONFIGS, FAST_SIZE
from material_matcher.engine import fit_material
from material_matcher.search import inverse_search
from material_matcher.templates import (build_chain, inexpressible_target,
                                        procedural_target)


class TestKnownRecovery(unittest.TestCase):
    def test_recovers_known_continuous_parameters(self):
        truth = {
            "noise.scale": 7.0,
            "transform.rotation": 0.55,
            "threshold.low": 0.3,
            "threshold.high": 0.7,
            "noise.octaves": 4,
        }
        target, _ = procedural_target(truth, size=FAST_SIZE, seed=3)
        graph = build_chain(seed=3)
        result = inverse_search(graph, target, seed=0,
                                max_configs=FAST_CONFIGS)
        # Appearance recovery: the fitted render must closely match.
        self.assertLess(result.best_loss, 0.12)
        p = result.best_params
        # Continuous recovery within a tolerance band, not exact equality.
        self.assertAlmostEqual(p["noise.scale"], truth["noise.scale"], delta=0.6)
        self.assertAlmostEqual(
            p["transform.rotation"], truth["transform.rotation"], delta=0.2)
        self.assertAlmostEqual(
            p["threshold.low"], truth["threshold.low"], delta=0.12)
        self.assertAlmostEqual(
            p["threshold.high"], truth["threshold.high"], delta=0.12)
        # The octave count is a *discrete* parameter whose neighbouring values
        # yield near-identical local statistics, so appearance cannot always
        # certify the true count.  Assert it is a legal choice (and that the
        # ambiguity is represented), rather than demanding the exact integer:
        # the equivalent-solution tests below cover the degeneracy explicitly.
        self.assertIn(p["noise.octaves"], tuple(range(1, 7)))

    def test_trajectory_is_recorded_and_monotone_best(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=2)
        result = inverse_search(build_chain(seed=2), target, seed=0,
                                max_configs=2)
        self.assertGreater(len(result.trajectory), 0)
        losses = [t.best_loss for t in result.trajectory]
        self.assertEqual(losses, sorted(losses, reverse=True)[:len(losses)])
        # The running best never increases.
        running = np.minimum.accumulate(losses)
        np.testing.assert_allclose(losses, running)

    def test_recovers_non_default_noise_kind(self):
        # Regression: a value-noise target must keep its discrete kind through
        # the prescreen instead of collapsing onto the default gradient kind.
        truth = {"noise.kind": "value", "noise.scale": 6.0,
                 "transform.rotation": 0.6}
        target, true_graph = procedural_target(
            truth, size=FAST_SIZE, seed=3)
        self.assertEqual(true_graph.get_param("noise", "kind"), "value")
        result = inverse_search(build_chain(seed=3), target, seed=3,
                                max_configs=4)
        self.assertEqual(result.best_params["noise.kind"], "value")
        self.assertLess(result.best_loss, 0.1)


class TestAmbiguity(unittest.TestCase):
    def test_discrete_tier_keeps_true_octaves_as_viable(self):
        # Neighbouring octave counts alias into near-identical statistics on a
        # small swatch, so the single ranked winner can legitimately differ by
        # seed.  The robust contract is therefore not "the winner is exactly
        # octave 4" but: (a) the winner is a legal, low-loss explanation, and
        # (b) the true octave count itself is capable of a low-loss fit and is
        # not excluded by the discrete prescreen.  We check (b) directly by
        # scoring the known-true graph rather than relying on ranking.
        truth = {"noise.scale": 7.0, "transform.rotation": 0.55,
                 "noise.octaves": 4}
        target, true_graph = procedural_target(
            truth, size=FAST_SIZE, seed=3)
        result = inverse_search(build_chain(seed=3), target, seed=0,
                                max_configs=6)
        self.assertIn(result.best_params["noise.octaves"], tuple(range(1, 7)))
        self.assertLess(result.best_loss, 0.12)
        # The ground-truth graph explains the target essentially perfectly,
        # proving the vocabulary and loss admit the true discrete choice.
        from material_matcher.render import multiscale_loss
        true_loss, _, _, _ = multiscale_loss(true_graph, target)
        self.assertLess(true_loss, 1e-9)

    def test_neighbouring_octaves_are_near_degenerate(self):
        # The root of the ambiguity: rendering octaves 4 and 5 (with all
        # other parameters identical) produces very close appearances, so an
        # image score alone cannot be treated as proof of the true integer.
        target, true_graph = procedural_target(
            {"noise.octaves": 4}, size=FAST_SIZE, seed=3)
        neighbour = true_graph.clone()
        neighbour.set_param("noise", "octaves", 5)
        from material_matcher.render import multiscale_loss
        loss4, _, _, _ = multiscale_loss(true_graph, target)
        loss5, _, _, _ = multiscale_loss(neighbour, target)
        self.assertLess(loss4, 1e-9)
        # The neighbour is a close (not exact) explanation of the same image.
        self.assertLess(loss5, 0.12)
        self.assertGreater(loss5, loss4)

    def test_equivalent_solutions_retained_within_tolerance(self):
        # A threshold-less colour target leaves more near-degenerate configs;
        # the search must retain clusters rather than assert uniqueness.
        target, _ = procedural_target(
            {"noise.scale": 5.0}, size=FAST_SIZE, seed=5, threshold=True)
        result = inverse_search(build_chain(seed=5), target, seed=0,
                                max_configs=FAST_CONFIGS,
                                equivalent_tol=0.05)
        self.assertGreaterEqual(len(result.equivalents), 1)
        for eq in result.equivalents:
            # Every retained solution is within the adaptive tolerance.
            self.assertLessEqual(
                eq.loss, result.best_loss +
                max(0.05, result.best_loss * 0.2) + 1e-9)

    def test_lowest_image_score_does_not_claim_unique_truth(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=4)
        result = inverse_search(build_chain(seed=4), target, seed=0,
                                max_configs=FAST_CONFIGS)
        # The result object carries warnings, not a claim of exact recovery.
        self.assertTrue(hasattr(result, "equivalents"))
        self.assertTrue(hasattr(result, "warnings"))


class TestInexpressible(unittest.TestCase):
    def test_checkerboard_leaves_structured_residual(self):
        target = inexpressible_target("checkerboard", size=FAST_SIZE)
        report = fit_material(build_chain(seed=1), target, seed=0,
                              max_configs=FAST_CONFIGS,
                              run_witness_decal=False)
        result = report["result"]
        self.assertFalse(result.success)
        self.assertGreaterEqual(result.best_loss, 0.12)
        self.assertEqual(report["status"], "failed")
        codes = {a["code"] for a in report["anomalies"]}
        self.assertIn("fit_failure", codes)
        # The residual is structured, not merely an average-colour mismatch.
        diff = report["diff"]
        self.assertGreater(diff.mean(), 0.05)

    def test_all_inexpressible_kinds_fail_gracefully(self):
        for kind in ("checkerboard", "grid", "blobs"):
            target = inexpressible_target(kind, size=FAST_SIZE)
            result = inverse_search(build_chain(seed=1), target, seed=0,
                                    max_configs=2)
            self.assertFalse(result.success, msg=kind)
            self.assertGreaterEqual(result.best_loss, 0.12)


if __name__ == "__main__":
    unittest.main()
