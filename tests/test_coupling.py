"""Parameter-coupling diagnostics: evidence chain, attribution, rerun.

These tests drive the coupling probes directly on controlled graphs instead
of running the full inverse search, so they stay fast while still
exercising every verdict and the bounded evidence log.  No probe receives
ground-truth parameters: targets are passed in as rendered pixels only.
"""
import json
import unittest

import numpy as np

from tests._setup import FAST_SIZE
from material_matcher import coupling as cmod
from material_matcher.coupling import (
    RunTracker, build_equivalent_pair, diagnose_stagnation,
    make_observation_mask, perturbation_surface, sensitivity_scan)
from material_matcher.templates import build_chain
from material_matcher.render import render_graph


def _isotropic_target(scale=8.0, size=FAST_SIZE, seed=3):
    tg = build_chain(seed=seed)
    tg.set_param("noise", "scale", scale)
    return render_graph(tg, size=size), tg


class TestSensitivityScan(unittest.TestCase):
    def test_scan_centres_on_assignment_and_restores_graph(self):
        target, tg = _isotropic_target()
        g = tg.clone()
        before = g.param_vector().copy()
        base_loss, recs = sensitivity_scan(
            g, target, size=FAST_SIZE, steps=3)
        after = g.param_vector()
        np.testing.assert_allclose(after, before)
        names = {r.param for r in recs}
        self.assertIn("noise.scale", names)
        # Every sweep is centred, so the middle delta is the base point.
        scale_rec = next(r for r in recs if r.param == "noise.scale")
        self.assertEqual(len(scale_rec.losses), 3)
        self.assertAlmostEqual(scale_rec.losses[1], base_loss, places=8)

    def test_exact_generator_has_zero_base_loss(self):
        target, tg = _isotropic_target()
        base_loss, _ = sensitivity_scan(
            tg.clone(), target, size=FAST_SIZE, steps=3)
        self.assertLess(base_loss, 1e-9)


class TestEquivalentPair(unittest.TestCase):
    def test_degenerate_family_matches_front_but_cannot_be_attributed(self):
        target, _ = _isotropic_target()
        graph = build_chain(seed=3)
        pair = build_equivalent_pair(
            graph, target, size=FAST_SIZE, second_uv=0.5)
        # Both branches reproduce the front view within tolerance.
        self.assertTrue(pair.front_equivalent)
        self.assertLessEqual(pair.front_error_a, pair.front_tolerance)
        self.assertLessEqual(pair.front_error_b, pair.front_tolerance)
        # The frequency/sampling degeneracy survives the camera move, so no
        # branch may be written to a material preset.
        self.assertEqual(pair.attribution, "ambiguous")
        self.assertEqual(pair.preset_eligible, "none")
        # Locking is explicit and symmetric across the two branches.
        self.assertIn("noise.scale", pair.locked)
        self.assertIn("transform.scale_x", pair.locked)
        # The two fitted branches genuinely differ on the named parameters.
        self.assertNotAlmostEqual(
            pair.branch_a_params["noise.scale"],
            pair.branch_b_params["noise.scale"])

    def test_pair_is_json_serialisable(self):
        target, _ = _isotropic_target()
        pair = build_equivalent_pair(
            build_chain(seed=3), target, size=FAST_SIZE)
        json.dumps(pair.to_dict())

    def test_second_view_separates_branches_when_not_degenerate(self):
        # An anisotropic sampling transform breaks the isotropic ratio
        # invariant, so the two branches can no longer predict every view.
        tg = build_chain(seed=3)
        tg.set_param("noise", "scale", 6.0)
        tg.set_param("transform", "scale_x", 1.5)
        tg.set_param("transform", "scale_y", 0.8)
        target = render_graph(tg, size=FAST_SIZE)
        pair = build_equivalent_pair(
            build_chain(seed=3), target, size=FAST_SIZE,
            second_uv=0.5, front_tolerance=0.05, second_tolerance=0.05)
        # A definite attribution (or explicit non-eligibility), never a
        # silent preset write.
        self.assertIn(pair.attribution,
                      ("frequency", "sampling", "ambiguous"))
        if pair.attribution != "ambiguous":
            self.assertIn(pair.preset_eligible,
                          ("frequency", "sampling", "none"))
            winner = (pair.second_error_a if pair.attribution == "frequency"
                      else pair.second_error_b)
            loser = (pair.second_error_b if pair.attribution == "frequency"
                     else pair.second_error_a)
            self.assertLess(winner, loser)


class TestPerturbationSurface(unittest.TestCase):
    def test_frequency_sampling_pair_is_coupled_along_ratio_manifold(self):
        target, tg = _isotropic_target()
        g = tg.clone()
        surf = perturbation_surface(
            g, target, "noise.scale", "transform.scale_x",
            follow_params=["transform.scale_y"],
            size=FAST_SIZE, steps=4)
        self.assertEqual(surf.verdict, "coupled")
        # Single-axis moves change the loss; the compensating joint move
        # preserves the S/T invariant and stays at (numerical) zero.
        self.assertGreater(surf.axis_peak_a, 1e-3)
        self.assertGreater(surf.axis_peak_b, 1e-3)
        self.assertLessEqual(surf.compensating_peak, 5e-3)
        self.assertGreater(surf.opposing_peak, surf.compensating_peak)

    def test_derivative_error_when_axes_flat_but_joint_moves_loss(self):
        tg = build_chain(seed=3)
        target = render_graph(tg, size=FAST_SIZE)
        g = tg.clone()
        surf = perturbation_surface(
            g, target, "threshold.low", "threshold.high",
            size=FAST_SIZE, steps=3, span_frac=0.004,
            flat_tol=3e-3, coupling_tol=3e-3)
        self.assertEqual(surf.verdict, "derivative_error")
        self.assertGreater(surf.compensating_peak, 3e-3)

    def test_surface_restores_graph_and_is_serialisable(self):
        target, tg = _isotropic_target()
        g = tg.clone()
        before = g.param_vector().copy()
        surf = perturbation_surface(
            g, target, "noise.scale", "transform.scale_x",
            follow_params=["transform.scale_y"], size=FAST_SIZE, steps=3)
        np.testing.assert_allclose(g.param_vector(), before)
        payload = surf.to_dict()
        self.assertEqual(payload["param_a"], "noise.scale")
        json.dumps(payload)


class TestStagnation(unittest.TestCase):
    def test_converged_when_generator_is_exact(self):
        target, tg = _isotropic_target()
        verdict = diagnose_stagnation(
            tg.clone(), target, size=FAST_SIZE)
        self.assertEqual(verdict.verdict, "converged")
        self.assertLess(verdict.full_error, 1e-9)

    def test_deficient_observation_distinguished_from_stagnation(self):
        target, tg = _isotropic_target()
        g = build_chain(seed=3)
        g.set_param("transform", "offset_x", 1.5)
        mask = make_observation_mask(FAST_SIZE, band=FAST_SIZE // 3)
        verdict = diagnose_stagnation(
            g, target, mask=mask, size=FAST_SIZE, deficiency_tol=0.6)
        self.assertEqual(verdict.verdict, "deficient_observation")
        self.assertLess(verdict.observed_fraction, 0.6)
        # The full-swatch error must exceed the observed-only error.
        self.assertGreater(verdict.full_error, verdict.observed_error)

    def test_non_identifiable_when_structure_axes_all_flat(self):
        target, _ = _isotropic_target()
        g = build_chain(seed=3)
        g.set_param("threshold", "low", 0.985)
        g.set_param("threshold", "high", 0.99)
        structure = ["noise.scale", "noise.lacunarity", "noise.gain",
                     "transform.scale_x", "transform.scale_y",
                     "transform.rotation", "threshold.low",
                     "threshold.high", "threshold.softness"]
        base_loss, recs = sensitivity_scan(
            g, target, size=FAST_SIZE, params=structure, flat_tol=1e-3)
        self.assertTrue(all(r.flat for r in recs))
        self.assertGreater(base_loss, 0.02)

    def test_stagnation_keeps_rerunnable_initial_values(self):
        target, _ = _isotropic_target()
        g = build_chain(seed=3)
        g.set_param("noise", "scale", 2.0)
        initial = g.clone()
        verdict = diagnose_stagnation(
            g, target, initial_graph=initial, size=FAST_SIZE)
        self.assertEqual(verdict.verdict, "stagnation")
        self.assertEqual(len(verdict.initial_vector),
                         len(initial.param_vector()))
        np.testing.assert_allclose(
            verdict.initial_vector, initial.param_vector())
        self.assertGreater(verdict.initial_loss, verdict.full_error - 1e-9)

    def test_mask_full_coverage_matches_unmasked_error(self):
        target, tg = _isotropic_target()
        g = build_chain(seed=3)
        full = diagnose_stagnation(g, target, size=FAST_SIZE)
        masked = diagnose_stagnation(
            g, target, mask=np.ones((FAST_SIZE, FAST_SIZE)),
            size=FAST_SIZE)
        self.assertEqual(masked.observed_fraction, 1.0)
        self.assertAlmostEqual(masked.observed_error, full.full_error,
                               places=9)


class TestRunTracker(unittest.TestCase):
    def _run(self, tracker, run_id):
        target, tg = _isotropic_target()
        tracker.start_run(run_id, graph=build_chain(seed=3), target=target,
                          size=FAST_SIZE)
        for stage, loss in (("core", 0.3), ("core", 0.2), ("color", 0.1)):
            tracker.record_trajectory(run_id, stage, loss, evaluations=5)
        _, recs = sensitivity_scan(
            tg.clone(), target, size=FAST_SIZE, steps=3, stage="final")
        tracker.record_sensitivity(run_id, recs, stage="final")
        tracker.record_anomaly(run_id, "parameter_coupling", "test",
                               stage="final", evidence={"k": 1})
        tracker.finish_run(run_id, tg.clone(), target)
        return target, tg

    def test_bounded_log_caps_entries_per_stage(self):
        tracker = RunTracker(cap_per_stage=3, anomaly_cap=4)
        target, tg = self._run(tracker, "r1")
        for _ in range(10):
            tracker.record_trajectory("r1", "core", 0.05, evaluations=9)
        run = tracker.get_run("r1")
        self.assertLessEqual(len(run["trajectory"]["core"]), 3)
        for code in range(10):
            tracker.record_anomaly("r1", f"c{code}", "m")
        self.assertLessEqual(len(run["anomalies"]), 4)
        # Every bucket is capped independently; the total stays bounded by
        # the per-stage cap times the number of buckets.
        traj_total = sum(len(v) for v in run["trajectory"].values())
        sens_total = sum(len(v) for v in run["sensitivity"].values())
        self.assertLessEqual(traj_total, 3 * len(run["trajectory"]))
        self.assertLessEqual(sens_total, 3 * len(run["sensitivity"]))
        self.assertEqual(tracker.log_size("r1"),
                         traj_total + sens_total + len(run["anomalies"])
                         + len(run["coupling_surfaces"]))

    def test_rerun_verification_matches_on_same_graph(self):
        tracker = RunTracker()
        target, tg = self._run(tracker, "r1")
        check = tracker.rerun_verification("r1", tg.clone(), target)
        self.assertTrue(check.matches, check.reason)
        self.assertLessEqual(check.loss_drift, check.tolerances["loss"])

    def test_rerun_verification_flags_drift(self):
        tracker = RunTracker()
        target, tg = self._run(tracker, "r1")
        altered = tg.clone()
        altered.set_param("noise", "scale", 2.0)
        check = tracker.rerun_verification(
            "r1", altered, target, loss_tol=1e-4)
        self.assertFalse(check.matches)
        self.assertGreater(check.loss_drift, 1e-4)

    def test_stage_localised_anomalies(self):
        tracker = RunTracker()
        self._run(tracker, "r1")
        tracker.record_anomaly("r1", "x", "core-msg", stage="core")
        tracker.record_anomaly("r1", "y", "final-msg", stage="final")
        self.assertEqual(
            {a["code"] for a in tracker.stage_anomalies("r1", "core")},
            {"x"})
        self.assertEqual(
            {a["code"] for a in tracker.stage_anomalies("r1", "final")},
            {"parameter_coupling", "y"})

    def test_run_record_is_json_serialisable(self):
        tracker = RunTracker()
        self._run(tracker, "r1")
        json.dumps(tracker.get_run("r1"))


if __name__ == "__main__":
    unittest.main()
