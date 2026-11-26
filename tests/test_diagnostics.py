"""Diagnostic anomaly flags and fit-status classification."""
import unittest

from tests._setup import FAST_SIZE
from material_matcher.diagnostics import diagnose, fit_status
from material_matcher.search import SearchResult, inverse_search
from material_matcher.templates import build_chain, inexpressible_target


def _minimal_result(graph, loss, equivalents=None, warnings=None,
                    components=None):
    return SearchResult(
        best_params={}, best_loss=loss,
        best_components=components or {}, graph=graph,
        trajectory=[], equivalents=equivalents or [],
        evaluations=0, warnings=warnings or [], success=loss < 0.12)


class TestDiagnostics(unittest.TestCase):
    def test_status_thresholds(self):
        g = build_chain()
        self.assertEqual(fit_status(_minimal_result(g, 0.20)), "failed")
        self.assertEqual(fit_status(_minimal_result(g, 0.09)), "partial")
        self.assertEqual(fit_status(_minimal_result(g, 0.03)), "fitted")

    def test_fit_failure_flag(self):
        g = build_chain()
        anomalies = diagnose(_minimal_result(g, 0.5), None)
        codes = {a["code"] for a in anomalies}
        self.assertIn("fit_failure", codes)

    def test_spectrum_mismatch_flag(self):
        g = build_chain()
        comps = {"scale_1": {"spectrum": 0.3, "gradient": 0.01,
                             "pixel": 0.1, "histogram": 0.02,
                             "mean_colour_collapse": False}}
        anomalies = diagnose(_minimal_result(g, 0.05, components=comps), None)
        self.assertIn("spectrum_mismatch", {a["code"] for a in anomalies})

    def test_parameter_at_bound_flag(self):
        g = build_chain()
        g.set_param("noise", "scale", 12.0)  # sits exactly on the high bound
        anomalies = diagnose(_minimal_result(g, 0.03), None)
        bound = next(a for a in anomalies if a["code"] == "parameter_at_bound")
        self.assertIn("noise.scale", bound["params"])

    def test_ambiguity_flag_from_equivalents(self):
        from material_matcher.search import EquivalentSolution
        g = build_chain()
        eqs = [EquivalentSolution({}, 0.03, {}, 0),
               EquivalentSolution({}, 0.04, {}, 1)]
        anomalies = diagnose(_minimal_result(g, 0.03, equivalents=eqs), None)
        self.assertIn("parameter_ambiguity", {a["code"] for a in anomalies})

    def test_inexpressible_pipeline_flags_failure(self):
        target = inexpressible_target("grid", size=FAST_SIZE)
        result = inverse_search(build_chain(seed=1), target, seed=0,
                                max_configs=2)
        anomalies = diagnose(result, target)
        codes = {a["code"] for a in anomalies}
        self.assertIn("fit_failure", codes)


if __name__ == "__main__":
    unittest.main()
