"""High-level fitting pipeline shared by the CLI and the web UI."""
from __future__ import annotations

import threading
import time
import uuid

import numpy as np

from . import coupling as cmod
from .decals import run_witness
from .diagnostics import diagnose, fit_status
from .graph import MaterialGraph
from .render import multiscale_loss, appearance_summary
from .search import inverse_search
from .templates import build_chain, build_blend_graph


def make_graph_template(template="chain", seed=1, space="object"):
    if template == "blend":
        return build_blend_graph(seed_a=seed, seed_b=seed + 8, space=space)
    return build_chain(seed=seed, space=space)


def edit_target(target, *, brightness=0.0, contrast=0.0, hue_shift=0.0):
    """Modify target appearance inside the authoring UI.

    All edits are simple photometric adjustments; the optimiser still only
    ever receives the resulting pixels, never the original image file.
    """
    img = target.copy()
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    if contrast:
        img = (img - lum[..., None]) * (1.0 + contrast) + lum[..., None]
    if brightness:
        img = img + brightness
    if hue_shift:
        # Rotate RGB around the grey axis (approximate hue rotation).
        axis = np.ones(3) / np.sqrt(3)
        cos_a, sin_a = np.cos(hue_shift), np.sin(hue_shift)
        proj = img @ axis
        orth = img - proj[..., None] * axis
        perp = np.cross(axis, orth)
        img = (proj[..., None] * axis + cos_a * orth + sin_a * perp)
    return np.clip(img, 0.0, 1.0)


def coupling_analysis(graph, target, *, uv_scale=1.0, size=None,
                      second_uv=0.5, mask=None):
    """Run the parameter-coupling evidence chain on a fitted graph.

    All three probes reuse small local renders around the fitted assignment,
    so the added cost is a fixed, bounded number of renders independent of
    the search budget.  Nothing here receives ground-truth parameters.
    """
    size = size or target.shape[0]
    has_transform = any(n.type == "transform" for n in graph.nodes.values())
    pair = None
    if has_transform:
        pair = cmod.build_equivalent_pair(
            graph, target, front_uv=uv_scale, second_uv=second_uv,
            size=size, mask=mask)

    surfaces = []
    if has_transform:
        noise_id = next(nid for nid, n in graph.nodes.items()
                        if n.type == "noise")
        xform_id = next(nid for nid, n in graph.nodes.items()
                        if n.type == "transform")
        # A compact 3x4 sweep keeps the perturbation render count bounded.
        # scale_y follows scale_x so the isotropic sampling invariant holds.
        surfaces.append(cmod.perturbation_surface(
            graph, target, f"{noise_id}.scale",
            f"{xform_id}.scale_x",
            follow_params=[f"{xform_id}.scale_y"],
            size=size, uv_scale=uv_scale, steps=4))

    _, sensitivity = cmod.sensitivity_scan(
        graph, target, size=size, uv_scale=uv_scale, steps=3, stage="final")
    stagnation = cmod.diagnose_stagnation(
        graph, target, mask=mask, size=size, uv_scale=uv_scale,
        stage="final")
    return {
        "equivalent_pair": pair,
        "coupling_surfaces": surfaces,
        "sensitivity": sensitivity,
        "stagnation": stagnation,
    }


def fit_material(graph: MaterialGraph, target, *,
                 seed=0, run_witness_decal=True, uv_scale=1.0,
                 progress=None, factors=(1, 2, 4), max_configs=4,
                 run_coupling=True, observation_mask=None, tracker=None,
                 run_id=None):
    """Run the full inverse pipeline and return a serialisable report."""
    def cb(event, payload):
        if progress and event == "iteration":
            progress(payload)

    if tracker is not None and run_id is not None:
        tracker.start_run(run_id, graph=graph, target=target,
                          uv_scale=uv_scale, size=target.shape[0],
                          factors=factors)
        def tracking_cb(event, payload):
            cb(event, payload)
            if event == "iteration":
                tracker.record_trajectory(
                    run_id, payload.get("stage", ""),
                    payload["best_loss"], payload["evaluations"],
                    payload.get("components"))
        search_cb = tracking_cb
    else:
        search_cb = cb

    result = inverse_search(
        graph, target, seed=seed, callback=search_cb, max_configs=max_configs,
        factors=factors, uv_scale=uv_scale)

    fitted = result.graph
    loss, comp, render, mean_warn = multiscale_loss(
        fitted, target, uv_scale=uv_scale, factors=factors,
        mask=observation_mask)
    diff = np.abs(render - target)

    witness = None
    if run_witness_decal:
        # Two fitting-equivalent graphs: one per coordinate space.
        g_obj = fitted.clone()
        g_wld = fitted.clone()
        g_obj.set_param(g_obj.output_id, "space", "object")
        g_wld.set_param(g_wld.output_id, "space", "world")
        witness = run_witness(g_obj, g_wld, target, size=target.shape[0],
                              uv_scale=uv_scale)

    coupling = None
    if run_coupling:
        coupling = coupling_analysis(
            fitted, target, uv_scale=uv_scale, size=target.shape[0],
            mask=observation_mask)
        if tracker is not None and run_id is not None:
            if coupling["equivalent_pair"] is not None:
                tracker.record_equivalent_pair(
                    run_id, coupling["equivalent_pair"])
            for surface in coupling["coupling_surfaces"]:
                tracker.record_coupling_surface(run_id, surface)
            tracker.record_sensitivity(run_id, coupling["sensitivity"],
                                       stage="final")
            if coupling["stagnation"].verdict in (
                    "stagnation", "non_identifiable",
                    "deficient_observation"):
                tracker.record_anomaly(
                    run_id, f"coupling_{coupling['stagnation'].verdict}",
                    coupling["stagnation"].reason, stage="final",
                    evidence={"full_error":
                              coupling["stagnation"].full_error,
                              "observed_fraction":
                              coupling["stagnation"].observed_fraction})
            tracker.finish_run(run_id, fitted, target, factors=factors)

    anomalies = diagnose(result, target, witness, coupling=coupling)
    return {
        "result": result,
        "fitted_graph": fitted,
        "render": render,
        "diff": diff,
        "diff_amplified": np.clip(diff * 3.0, 0.0, 1.0),
        "witness": witness,
        "coupling": coupling,
        "anomalies": anomalies,
        "status": fit_status(result),
        "target_summary": appearance_summary(target),
        "render_summary": appearance_summary(render),
    }


# ---------------------------------------------------------------------------
# Background job management (the UI shows live run state)
# ---------------------------------------------------------------------------
class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def start(self, graph, target, **kwargs):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "state": "running", "started_at": time.time(),
            "trajectory": [], "report": None, "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        def progress(payload):
            with self._lock:
                job["trajectory"].append(payload)

        def worker():
            try:
                report = fit_material(graph, target, progress=progress,
                                      **kwargs)
                with self._lock:
                    job["report"] = report
                    job["state"] = "done"
            except Exception as exc:  # surfaced to the UI, not swallowed
                with self._lock:
                    job["error"] = repr(exc)
                    job["state"] = "error"

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id):
        job = self.get(job_id)
        if not job:
            return None
        with self._lock:
            return {
                "id": job["id"], "state": job["state"],
                "error": job["error"],
                "trajectory": list(job["trajectory"]),
                "has_report": job["report"] is not None,
            }
