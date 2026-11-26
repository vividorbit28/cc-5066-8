"""Stdlib-only web server for the texture authoring UI.

The browser page is the primary material-matching surface: the user picks or
edits a target appearance, launches an inverse solve, watches the live run
state and optimisation trajectory, and inspects the recovered editable
material graph alongside the render difference and witness-decal verdict.
Only the Python standard library is used (``http.server``); numpy/Pillow
live behind the engine API.

Run with::

    python3 -m material_matcher.web_app [--port 8000]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .decals import render_in_space
from .engine import JobManager, edit_target, make_graph_template
from .graph import NODE_LABELS, REGISTRY, MaterialGraph
from .render import encode_png, render_graph
from .templates import inexpressible_target, procedural_target

# A compact, fast default so the interactive loop stays responsive.  The
# CLI/demo path exercises the heavier settings.
DEFAULT_SIZE = 64
DEFAULT_MAX_CONFIGS = 4


class Session:
    """Holds the current target, template and the active background job."""

    def __init__(self):
        # Reentrant so a handler that already holds the lock can call a
        # helper (e.g. ``_state``) that reads session state under the lock.
        self.lock = threading.RLock()
        self.jobs = JobManager()
        self.size = DEFAULT_SIZE
        self.target = None
        self.target_kind = "procedural"          # procedural | inexpressible
        self.template = "chain"
        self.seed = 3
        self.space = "object"
        self.graph = None
        self.truth = None                        # known ground truth, if any
        self.last_job_id = None
        self.reset_target()

    # -- target management --------------------------------------------------
    def reset_target(self, kind="procedural", inexpressible=None,
                     truth=None, seed=None, size=None):
        self.target_kind = kind
        self.seed = 3 if seed is None else seed
        self.size = self.size if size is None else size
        if kind == "inexpressible":
            self.target = inexpressible_target(
                inexpressible or "checkerboard", size=self.size)
            self.truth = None
        else:
            self.truth = truth if truth is not None else {
                "noise.scale": 7.0,
                "transform.rotation": 0.55,
                "threshold.low": 0.3,
                "threshold.high": 0.7,
                "noise.octaves": 4,
            }
            self.target, _ = procedural_target(
                self.truth, size=self.size, seed=self.seed)
        self.graph = make_graph_template(
            self.template, seed=self.seed, space=self.space)

    def apply_edits(self, brightness, contrast, hue_deg):
        with self.lock:
            self.target = edit_target(
                self.target, brightness=brightness, contrast=contrast,
                hue_shift=np.deg2rad(hue_deg))

    def start_job(self, max_configs):
        with self.lock:
            if self.target is None:
                raise RuntimeError("no target loaded")
            graph = self.graph.clone()
            target = self.target.copy()
        job_id = self.jobs.start(
            graph, target, seed=self.seed,
            max_configs=max_configs)
        with self.lock:
            self.last_job_id = job_id
        return job_id

    def report(self):
        if not self.last_job_id:
            return None
        return self.jobs.get(self.last_job_id)


SESSION = Session()


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def graph_payload(graph: MaterialGraph):
    """Structured, editable view of a graph: nodes grouped with their specs."""
    nodes = []
    for nid in graph.topo_order():
        node = graph.nodes[nid]
        specs = []
        for spec in REGISTRY[node.type]:
            raw = node.params[spec.name]
            specs.append({
                "name": spec.name,
                "kind": spec.kind,
                "value": (raw if isinstance(raw, str)
                          else (int(raw) if isinstance(raw, (int, np.integer))
                                else float(raw))),
                "low": spec.low,
                "high": spec.high,
                "choices": list(spec.choices),
            })
        nodes.append({"id": nid, "type": node.type,
                      "label": NODE_LABELS[node.type],
                      "inputs": list(node.inputs), "params": specs})
    return {"output_id": graph.output_id, "nodes": nodes}


def witness_payload(w):
    if w is None:
        return None
    d = w.to_dict()
    d["boundary_images_available"] = True
    return d


def report_payload(job):
    report = job["report"]
    result = report["result"]
    return {
        "status": report["status"],
        "best_loss": result.best_loss,
        "best_components": result.best_components,
        "best_params": result.best_params,
        "evaluations": result.evaluations,
        "success": result.success,
        "warnings": result.warnings,
        "anomalies": report["anomalies"],
        "equivalents": [
            {"cluster_id": e.cluster_id, "loss": e.loss,
             "components": e.components, "params": e.params}
            for e in result.equivalents],
        "witness": witness_payload(report["witness"]),
        "graph": graph_payload(report["fitted_graph"]),
        "graph_json": report["fitted_graph"].to_dict(),
        "target_summary": report["target_summary"],
        "render_summary": report["render_summary"],
        "ground_truth": SESSION.truth,
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "MaterialMatcher/1.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers ------------------------------------------------------------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_png(self, arr):
        body = encode_png(arr)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- routing ------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html"):
                return self._serve_page()
            if path == "/api/state":
                return self._send_json(self._state())
            if path == "/api/target.png":
                with SESSION.lock:
                    return self._send_png(SESSION.target)
            if path == "/api/template.png":
                with SESSION.lock:
                    img = render_graph(SESSION.graph, SESSION.size)
                return self._send_png(img)
            if path == "/api/job":
                return self._job_status()
            if path == "/api/result_images":
                return self._result_images()
            self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # surface server errors to the UI
            self._send_json({"error": repr(exc)}, 500)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            payload = self._read_json()
            if path == "/api/target":
                return self._set_target(payload)
            if path == "/api/edit":
                return self._edit_target(payload)
            if path == "/api/fit":
                return self._start_fit(payload)
            self._send_json({"error": "not found"}, 404)
        except Exception as exc:
            self._send_json({"error": repr(exc)}, 500)

    # -- GET responses ------------------------------------------------------
    def _serve_page(self):
        body = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _state(self):
        with SESSION.lock:
            return {
                "size": SESSION.size,
                "target_kind": SESSION.target_kind,
                "template": SESSION.template,
                "seed": SESSION.seed,
                "space": SESSION.space,
                "ground_truth": SESSION.truth,
                "graph": graph_payload(SESSION.graph),
                "last_job_id": SESSION.last_job_id,
            }

    def _job_status(self):
        job_id = SESSION.last_job_id
        if not job_id:
            return self._send_json({"state": "idle"})
        snap = SESSION.jobs.snapshot(job_id)
        if snap is None:
            return self._send_json({"state": "idle"})
        if snap["state"] == "done":
            job = SESSION.jobs.get(job_id)
            snap["report"] = report_payload(job)
        return self._send_json(snap)

    def _result_images(self):
        job = SESSION.report()
        if not job or job["state"] != "done":
            return self._send_json({"error": "no completed run"}, 409)
        # A single JSON document with embedded PNGs keeps the UI to one fetch.
        report = job["report"]

        def b64(arr):
            return base64.b64encode(encode_png(arr)).decode("ascii")

        w = report["witness"]
        images = {
            "target": b64(SESSION.target),
            "render": b64(report["render"]),
            "diff": b64(report["diff"]),
            "diff_amplified": b64(report["diff_amplified"]),
        }
        if w is not None:
            size = SESSION.size
            g_obj = report["fitted_graph"].clone()
            g_wld = report["fitted_graph"].clone()
            g_obj.set_param(g_obj.output_id, "space", "object")
            g_wld.set_param(g_wld.output_id, "space", "world")
            sx, sy = w.shift
            images["witness_object"] = b64(
                render_in_space(g_obj, size, 1.0, "object", sx, sy))
            images["witness_world"] = b64(
                render_in_space(g_wld, size, 1.0, "world", sx, sy))
        return self._send_json(images)

    # -- POST responses -----------------------------------------------------
    def _set_target(self, payload):
        kind = payload.get("kind", "procedural")
        size = int(payload.get("size", DEFAULT_SIZE))
        seed = int(payload.get("seed", 3))
        SESSION.template = payload.get("template", "chain")
        SESSION.space = payload.get("space", "object")
        if kind == "inexpressible":
            SESSION.size = size
            SESSION.reset_target(
                kind="inexpressible",
                inexpressible=payload.get("inexpressible", "checkerboard"),
                seed=seed)
        else:
            SESSION.size = size
            SESSION.reset_target(kind="procedural", seed=seed)
        with SESSION.lock:
            SESSION.last_job_id = None
        return self._send_json({"ok": True, "state": self._state()})

    def _edit_target(self, payload):
        SESSION.apply_edits(
            float(payload.get("brightness", 0.0)),
            float(payload.get("contrast", 0.0)),
            float(payload.get("hue", 0.0)))
        with SESSION.lock:
            SESSION.last_job_id = None
        return self._send_json({"ok": True})

    def _start_fit(self, payload):
        max_configs = int(payload.get("max_configs", DEFAULT_MAX_CONFIGS))
        job_id = SESSION.start_job(max_configs)
        return self._send_json({"job_id": job_id})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Material matcher web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Material matcher UI on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


# The front-end lives beside this module so it stays editable as plain HTML.
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_WEB_DIR, "web", "index.html"),
          encoding="utf-8") as _fh:
    INDEX_HTML = _fh.read()


if __name__ == "__main__":
    main()
