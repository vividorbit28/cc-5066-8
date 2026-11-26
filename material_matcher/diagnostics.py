"""Diagnostic flags linking material parameters to render-difference errors.

The UI calls :func:`diagnose` on every search so problems are surfaced as
named anomalies with severity instead of a single opaque image score.
"""
from __future__ import annotations

from .graph import spec_index

# Empirical thresholds; loss is a weighted sum of L1 terms in [0, 1].
FAIL_LOSS = 0.12
POOR_LOSS = 0.07
BOUND_EPS = 1e-3


def _at_bound(graph, nid, spec):
    v = graph.get_param(nid, spec.name)
    return spec.kind == "continuous" and (
        abs(v - spec.low) <= BOUND_EPS or abs(v - spec.high) <= BOUND_EPS)


def diagnose(result, target, witness=None):
    """Return a list of anomaly dicts for a SearchResult."""
    anomalies = []
    g = result.graph
    comps = result.best_components or {}

    if result.best_loss >= FAIL_LOSS:
        anomalies.append({
            "severity": "error",
            "code": "fit_failure",
            "message": f"best loss {result.best_loss:.4f} exceeds failure "
                       f"threshold {FAIL_LOSS}; target contains structure "
                       "this node vocabulary cannot reproduce.",
        })
    elif result.best_loss >= POOR_LOSS:
        anomalies.append({
            "severity": "warning",
            "code": "poor_fit",
            "message": f"loss {result.best_loss:.4f} is high; inspect the "
                       "render difference for structured residuals.",
        })

    nested = [v for v in comps.values() if isinstance(v, dict)]
    if nested:
        dominant = max(
            (c for scale in nested for c in scale.items()
             if c[0] != "mean_colour_collapse"),
            key=lambda kv: kv[1], default=(None, 0.0))
    else:
        dominant = (None, 0.0)
    if dominant[0] == "spectrum" and dominant[1] > 0.05:
        anomalies.append({
            "severity": "warning",
            "code": "spectrum_mismatch",
            "message": "radial power spectrum dominates the residual: "
                       "pattern scale/frequency is wrong even though "
                       "average colours may match.",
            "value": dominant[1],
        })
    if dominant[0] == "gradient" and dominant[1] > 0.05:
        anomalies.append({
            "severity": "warning",
            "code": "gradient_mismatch",
            "message": "edge/gradient statistics dominate the residual: "
                       "threshold softness or contrast likely wrong.",
            "value": dominant[1],
        })
    if any(isinstance(v, dict) and v.get("mean_colour_collapse")
           for v in comps.values()):
        anomalies.append({
            "severity": "warning",
            "code": "mean_colour_collapse",
            "message": "candidate matched mean colour while losing texture "
                       "variation; this is penalised, not accepted.",
        })

    bounded = []
    for nid, spec in spec_index(g):
        if _at_bound(g, nid, spec):
            bounded.append(f"{nid}.{spec.name}")
    if bounded:
        anomalies.append({
            "severity": "warning",
            "code": "parameter_at_bound",
            "message": "parameters saturated at search range limits; the "
                       "true value may lie outside the current range.",
            "params": bounded,
        })

    if len(result.equivalents) > 1:
        anomalies.append({
            "severity": "info",
            "code": "parameter_ambiguity",
            "message": f"{len(result.equivalents)} parameter clusters fit "
                       "almost equally well; they are kept as equivalent "
                       "solutions rather than claiming a unique ground truth.",
            "clusters": len(result.equivalents),
        })

    for msg in result.warnings:
        if msg.startswith("mean_colour_collapse"):
            continue  # already raised above
        anomalies.append({"severity": "info", "code": "search_note",
                          "message": msg})

    if witness is not None:
        if witness.alias_warning:
            anomalies.append({
                "severity": "warning",
                "code": "period_aliasing",
                "message": "world-space decal looked static at the first "
                           "scale because the translation aliases the "
                           "pattern period; the second-scale observation "
                           "broke the tie.",
            })
        if witness.rejected_params:
            anomalies.append({
                "severity": "info",
                "code": "witness_rejected_params",
                "message": "witness decal refuses to certify these "
                           "parameters as identical across the two "
                           "coordinate spaces.",
                "params": [p["param"] for p in witness.rejected_params],
            })
    return anomalies


def fit_status(result):
    if result.best_loss >= FAIL_LOSS:
        return "failed"
    if result.best_loss >= POOR_LOSS:
        return "partial"
    return "fitted"
