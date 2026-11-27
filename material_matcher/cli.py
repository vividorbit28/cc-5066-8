"""Command-line entry point.

The CLI is a convenience for scripting and headless tests; the material
matching interaction lives in the web authoring UI (``web_app.py``).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .engine import (fit_material, make_graph_template, edit_target,
                     coupling_analysis)
from .graph import MaterialGraph
from .render import load_target, save_png, render_graph
from .templates import procedural_target, inexpressible_target, build_chain
from . import coupling as cmod


def _write_report(report, outdir):
    os.makedirs(outdir, exist_ok=True)
    result = report["result"]
    save_png(report["render"], os.path.join(outdir, "render.png"))
    save_png(report["diff_amplified"], os.path.join(outdir, "diff.png"))
    with open(os.path.join(outdir, "fitted_graph.json"), "w") as fh:
        json.dump(report["fitted_graph"].to_dict(), fh, indent=2)

    payload = {
        "status": report["status"],
        "best_loss": result.best_loss,
        "best_components": result.best_components,
        "best_params": result.best_params,
        "equivalent_solutions": [
            {"cluster_id": e.cluster_id, "loss": e.loss, "params": e.params}
            for e in result.equivalents],
        "evaluations": result.evaluations,
        "warnings": result.warnings,
        "anomalies": report["anomalies"],
        "witness": report["witness"].to_dict() if report["witness"] else None,
        "coupling": _coupling_summary(report.get("coupling")),
        "target_summary": report["target_summary"],
        "render_summary": report["render_summary"],
        "trajectory": [
            {"iteration": t.iteration, "evaluations": t.evaluations,
             "best_loss": t.best_loss, "mean_loss": t.mean_loss,
             "elapsed": t.elapsed, "stage": t.stage}
            for t in result.trajectory],
    }
    with open(os.path.join(outdir, "report.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def _coupling_summary(coupling):
    """JSON-safe coupling evidence for the headless report."""
    if coupling is None:
        return None
    pair = coupling.get("equivalent_pair")
    return {
        "equivalent_pair": pair.to_dict() if pair is not None else None,
        "coupling_surfaces": [s.to_dict()
                              for s in coupling.get("coupling_surfaces", [])],
        "sensitivity": [s.to_dict()
                        for s in coupling.get("sensitivity", [])],
        "stagnation": (coupling["stagnation"].to_dict()
                       if coupling.get("stagnation") is not None else None),
    }


def cmd_demo(args):
    if args.inexpressible:
        target = inexpressible_target(args.inexpressible, size=args.size)
        graph = build_chain(seed=1)
    else:
        truth = {
            "noise.scale": 6.0, "noise.kind": "value",
            "transform.rotation": 0.6,
            "threshold.low": 0.3, "threshold.high": 0.72,
            "colormap.stop1_r": 0.8,
        }
        target, _true_graph = procedural_target(
            truth, size=args.size, seed=args.seed)
        graph = build_chain(seed=args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    save_png(target, os.path.join(args.outdir, "target.png"))

    report = fit_material(
        graph, target, seed=args.seed)
    payload = _write_report(report, args.outdir)
    print(json.dumps({k: payload[k] for k in
                      ("status", "best_loss", "evaluations")}, indent=2))
    print(f"report written to {args.outdir}")


def cmd_fit(args):
    target = load_target(args.target, size=args.size)
    graph = make_graph_template(args.template, seed=args.seed)
    if args.brightness or args.contrast or args.hue_shift:
        target = edit_target(target, brightness=args.brightness,
                             contrast=args.contrast,
                             hue_shift=np.deg2rad(args.hue_shift))
    report = fit_material(graph, target, seed=args.seed)
    payload = _write_report(report, args.outdir)
    print(json.dumps({k: payload[k] for k in
                      ("status", "best_loss", "evaluations")}, indent=2))
    print(f"report written to {args.outdir}")


def cmd_render(args):
    with open(args.graph) as fh:
        graph = MaterialGraph.from_dict(json.load(fh))
    img = render_graph(graph, size=args.size)
    save_png(img, args.output)
    print(f"wrote {args.output}")


def cmd_coupling(args):
    """Headless coupling diagnostics; the web UI remains the main entry."""
    if args.target:
        target = load_target(args.target, size=args.size)
        graph = make_graph_template(args.template, seed=args.seed)
    else:
        target, true_graph = procedural_target(size=args.size, seed=args.seed)
        graph = build_chain(seed=args.seed)
    mask = None
    if args.mask_band:
        mask = cmod.make_observation_mask(args.size, band=args.mask_band)
    elif args.mask_missing:
        mask = cmod.make_observation_mask(
            args.size, missing_fraction=args.mask_missing)
    analysis = coupling_analysis(
        graph, target, size=args.size, second_uv=args.second_uv, mask=mask)
    pair = analysis["equivalent_pair"]
    stagnation = analysis["stagnation"]
    summary = {
        "equivalent_pair": pair.to_dict() if pair else None,
        "surface_verdicts": [
            {"pair": (s.param_a, s.param_b), "verdict": s.verdict,
             "compensating_peak": s.compensating_peak,
             "axis_peak_a": s.axis_peak_a, "axis_peak_b": s.axis_peak_b}
            for s in analysis["coupling_surfaces"]],
        "stagnation": stagnation.to_dict(),
        "flat_parameters": [s.param for s in analysis["sensitivity"]
                            if s.flat],
    }
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "coupling.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({
        "attribution": pair.attribution if pair else None,
        "preset_eligible": pair.preset_eligible if pair else None,
        "surface_verdicts": [s["verdict"]
                             for s in summary["surface_verdicts"]],
        "stagnation": stagnation.verdict,
    }, indent=2))
    print(f"coupling diagnostics written to {args.outdir}/coupling.json")


def build_parser():
    p = argparse.ArgumentParser(
        prog="material-matcher",
        description="Inverse-fit procedural material graphs to textures.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run on a synthetic known-parameter "
                                    "target or an inexpressible pattern")
    d.add_argument("--outdir", default="workdir/cli_demo")
    d.add_argument("--size", type=int, default=96)
    d.add_argument("--seed", type=int, default=3)
    d.add_argument("--inexpressible",
                   choices=("checkerboard", "grid", "blobs"))
    d.set_defaults(func=cmd_demo)

    f = sub.add_parser("fit", help="fit a graph to an image file")
    f.add_argument("target")
    f.add_argument("--outdir", default="workdir/cli_fit")
    f.add_argument("--template", default="chain",
                   choices=("chain", "blend"))
    f.add_argument("--size", type=int, default=96)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--brightness", type=float, default=0.0)
    f.add_argument("--contrast", type=float, default=0.0)
    f.add_argument("--hue-shift", type=float, default=0.0)
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("render", help="render a saved graph JSON to PNG")
    r.add_argument("graph")
    r.add_argument("output")
    r.add_argument("--size", type=int, default=128)
    r.set_defaults(func=cmd_render)

    c = sub.add_parser("coupling",
                       help="headless parameter-coupling diagnostics "
                            "(the web UI is the primary interface)")
    c.add_argument("target", nargs="?",
                   help="optional target image; defaults to a synthetic one")
    c.add_argument("--outdir", default="workdir/cli_coupling")
    c.add_argument("--template", default="chain",
                   choices=("chain", "blend"))
    c.add_argument("--size", type=int, default=64)
    c.add_argument("--seed", type=int, default=3)
    c.add_argument("--second-uv", type=float, default=0.5,
                   dest="second_uv")
    c.add_argument("--mask-band", type=int, default=0, dest="mask_band")
    c.add_argument("--mask-missing", type=float, default=0.0,
                   dest="mask_missing")
    c.set_defaults(func=cmd_coupling)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
