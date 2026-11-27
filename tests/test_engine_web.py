"""Engine pipeline, background JobManager, and the web authoring HTTP API."""
import json
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from tests._setup import FAST_CONFIGS, FAST_SIZE
from material_matcher import web_app
from material_matcher.engine import JobManager, edit_target, fit_material
from material_matcher.templates import build_chain, procedural_target

# Coupling-driven HTTP tests use a smaller swatch and a single discrete
# configuration so the background fit stays well inside the wait window;
# the coupling probes themselves are asserted independently of search depth.
COUPLING_SIZE = 28
COUPLING_CONFIGS = 1


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def _get(url):
    with urllib.request.urlopen(url) as resp:
        return resp.status, resp.read(), resp.headers


class TestEngine(unittest.TestCase):
    def test_edit_target_photometric_only(self):
        target, _ = procedural_target(size=24, seed=1)
        edited = edit_target(target, brightness=0.1, contrast=0.2,
                             hue_shift=0.5)
        self.assertEqual(edited.shape, target.shape)
        self.assertTrue(((edited >= 0) & (edited <= 1)).all())
        # A zero edit must reproduce the original.
        np_out = edit_target(target)
        import numpy as np
        np.testing.assert_allclose(np_out, target, atol=1e-12)

    def test_job_manager_runs_in_background(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=3)
        mgr = JobManager()
        jid = mgr.start(build_chain(seed=3), target, seed=0,
                        max_configs=2, run_witness_decal=False)
        deadline = time.time() + 180
        while time.time() < deadline:
            snap = mgr.snapshot(jid)
            if snap["state"] in ("done", "error"):
                break
            time.sleep(0.5)
        snap = mgr.snapshot(jid)
        self.assertEqual(snap["state"], "done")
        self.assertTrue(snap["has_report"])
        # Live trajectory points were captured during the run.
        self.assertGreater(len(snap["trajectory"]), 0)
        report = mgr.get(jid)["report"]
        self.assertIn("fitted_graph", report)

    def test_fit_report_contains_graph_not_baked_texture(self):
        target, _ = procedural_target(size=FAST_SIZE, seed=2)
        report = fit_material(build_chain(seed=2), target, seed=0,
                              max_configs=2)
        # The deliverable is a serialisable node graph with typed nodes, not a
        # single image node carrying the target pixels.
        gd = report["fitted_graph"].to_dict()
        types = {n["type"] for n in gd["nodes"]}
        self.assertIn("noise", types)
        self.assertIn("output", types)
        # No node references an external image file.
        for n in gd["nodes"]:
            self.assertNotIn("image_path", n["params"])


class TestWebAPI(unittest.TestCase):
    server = None
    thread = None
    base = None

    @classmethod
    def setUpClass(cls):
        # Fresh isolated session so the tests do not depend on each other.
        web_app.SESSION.reset_target(kind="procedural", seed=3,
                                     size=FAST_SIZE)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_index_served(self):
        status, body, headers = _get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("Material Matcher", body.decode())

    def test_state_and_target_png(self):
        status, state = _post(self.base + "/api/target",
                              {"kind": "procedural", "size": FAST_SIZE,
                               "seed": 3})
        self.assertEqual(status, 200)
        self.assertEqual(state["state"]["size"], FAST_SIZE)
        status, png, headers = _get(self.base + "/api/target.png")
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_full_fit_flow_through_http(self):
        _post(self.base + "/api/target",
              {"kind": "procedural", "size": FAST_SIZE, "seed": 3})
        _, started = _post(self.base + "/api/fit",
                           {"max_configs": FAST_CONFIGS})
        self.assertIn("job_id", started)
        report = self._wait_for_job()
        self.assertEqual(report["status"], "fitted")
        self.assertLess(report["best_loss"], 0.12)
        self.assertIn("graph", report)
        self.assertIn("witness", report)
        # Equivalent-solution structure is always present.
        self.assertIsInstance(report["equivalents"], list)

    def test_inexpressible_reports_failure_via_http(self):
        _post(self.base + "/api/target",
              {"kind": "inexpressible", "inexpressible": "checkerboard",
               "size": FAST_SIZE, "seed": 1})
        _post(self.base + "/api/fit", {"max_configs": 2})
        report = self._wait_for_job()
        self.assertEqual(report["status"], "failed")
        codes = {a["code"] for a in report["anomalies"]}
        self.assertIn("fit_failure", codes)

    def test_result_images_and_graph_download(self):
        _post(self.base + "/api/target",
              {"kind": "procedural", "size": FAST_SIZE, "seed": 2})
        _post(self.base + "/api/fit", {"max_configs": 2})
        self._wait_for_job()
        status, images = _get_json(self.base + "/api/result_images")
        for key in ("target", "render", "diff_amplified",
                    "witness_object", "witness_world"):
            self.assertIn(key, images)
        # The downloadable graph is valid JSON describing nodes.
        _, job_state = _get_json(self.base + "/api/job")
        graph_json = job_state["report"]["graph_json"]
        self.assertIn("nodes", graph_json)

    def test_fit_report_carries_coupling_evidence(self):
        _post(self.base + "/api/target",
              {"kind": "procedural", "size": COUPLING_SIZE, "seed": 2})
        _post(self.base + "/api/fit", {"max_configs": COUPLING_CONFIGS})
        report = self._wait_for_job(timeout=120)
        coupling = report["coupling"]
        self.assertIsNotNone(coupling)
        # The evidence chain is present and uses observable quantities only.
        self.assertIn("equivalent_pair", coupling)
        self.assertTrue(coupling["equivalent_pair"]["locked"])
        self.assertIn("preset_eligible", coupling["equivalent_pair"])
        self.assertTrue(coupling["coupling_surfaces"])
        self.assertEqual(
            coupling["coupling_surfaces"][0]["param_a"], "noise.scale")
        self.assertIn("stagnation", coupling)
        self.assertTrue(coupling["sensitivity"])
        # A coupling-derived anomaly is evidence-backed, never a bare flag.
        coupling_flags = [a for a in report["anomalies"]
                          if a["code"] in ("parameter_coupling",
                                           "derivative_error",
                                           "coupled_parameter_family")]
        for flag in coupling_flags:
            self.assertIn("evidence", flag)
            self.assertEqual(flag["evidence"]["stage"], "final")

    def test_coupling_diagnostic_endpoints(self):
        _post(self.base + "/api/target",
              {"kind": "procedural", "size": COUPLING_SIZE, "seed": 2})
        _, pair = _post(self.base + "/api/diagnose/pair",
                        {"second_uv": 0.5})
        self.assertIn("attribution", pair)
        self.assertIn(pair["attribution"],
                      ("frequency", "sampling", "ambiguous"))
        _, surface = _post(self.base + "/api/diagnose/surface", {"steps": 4})
        self.assertIn(surface["verdict"],
                      ("coupled", "derivative_error", "independent"))
        _, stagnation = _post(self.base + "/api/diagnose/stagnation", {})
        self.assertIn(stagnation["verdict"],
                      ("stagnation", "non_identifiable",
                       "deficient_observation", "converged"))

    def test_mask_endpoint_and_deficient_rerun(self):
        _post(self.base + "/api/target",
              {"kind": "procedural", "size": COUPLING_SIZE, "seed": 2})
        _, mask_ack = _post(self.base + "/api/mask",
                            {"kind": "band", "band": COUPLING_SIZE // 4})
        self.assertLess(mask_ack["observed_fraction"], 1.0)
        status, png, headers = _get(self.base + "/api/mask.png")
        self.assertEqual(headers["Content-Type"], "image/png")
        _, stagnation = _post(self.base + "/api/diagnose/stagnation", {})
        self.assertIn(stagnation["verdict"],
                      ("stagnation", "non_identifiable",
                       "deficient_observation", "converged"))
        # A fit under the mask records a run the rerun endpoint can verify.
        _post(self.base + "/api/fit", {"max_configs": COUPLING_CONFIGS})
        self._wait_for_job(timeout=120)
        _, rerun = _post(self.base + "/api/diagnose/rerun", {})
        self.assertIn("matches", rerun)
        self.assertIn("loss_drift", rerun)
        # Restore full observation for the other tests sharing the session.
        _post(self.base + "/api/mask", {"kind": "none"})

    def _wait_for_job(self, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, snap = _get_json(self.base + "/api/job")
            if snap.get("state") in ("done", "error"):
                if snap.get("state") == "error":
                    raise AssertionError(snap.get("error"))
                return snap["report"]
            time.sleep(0.7)
        self.fail("job did not finish in time")


def _get_json(url):
    status, body, _ = _get(url)
    return status, json.loads(body)


if __name__ == "__main__":
    unittest.main()
