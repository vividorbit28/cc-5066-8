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


def diagnose(result, target, witness=None, coupling=None):
    """Return a list of anomaly dicts for a SearchResult.

    Every flag is backed by a measured quantity carried in its ``evidence``
    payload; coupling flags are only raised when the corresponding probe
    actually observed the condition, so an unrun or inconclusive probe never
    produces a report (no false positives).
    """
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

    if coupling is not None:
        anomalies.extend(_coupling_anomalies(coupling))
    return anomalies


def _coupling_anomalies(coupling):
    """Turn measured coupling probes into evidence-backed anomaly records."""
    out = []
    pair = coupling.get("equivalent_pair")
    if pair is not None and pair.front_equivalent:
        evidence = {
            "front_error_a": pair.front_error_a,
            "front_error_b": pair.front_error_b,
            "second_error_a": pair.second_error_a,
            "second_error_b": pair.second_error_b,
            "front_uv": pair.front_uv,
            "second_uv": pair.second_uv,
            "locked": list(pair.locked),
            "preset_eligible": pair.preset_eligible,
            "stage": "final",
        }
        if pair.attribution == "ambiguous":
            out.append({
                "severity": "warning",
                "code": "coupled_parameter_family",
                "message": "the surface-frequency and sampling-scale "
                           "branches match at the front view and predict the "
                           "second view equally; the parameter family is "
                           "retained but no branch is written to a preset.",
                "evidence": evidence,
            })
        else:
            out.append({
                "severity": "info",
                "code": "coupling_attributed",
                "message": f"second-view pixel error attributes the "
                           f"appearance to the {pair.attribution} branch; "
                           f"preset eligibility: {pair.preset_eligible}. "
                           + pair.reason,
                "evidence": evidence,
            })

    for surface in coupling.get("coupling_surfaces", []):
        if surface.verdict == "coupled":
            out.append({
                "severity": "warning",
                "code": "parameter_coupling",
                "message": surface.reason,
                "evidence": {
                    "param_a": surface.param_a, "param_b": surface.param_b,
                    "axis_peak_a": surface.axis_peak_a,
                    "axis_peak_b": surface.axis_peak_b,
                    "compensating_peak": surface.compensating_peak,
                    "opposing_peak": surface.opposing_peak,
                    "stage": "final",
                },
            })
        elif surface.verdict == "derivative_error":
            out.append({
                "severity": "warning",
                "code": "derivative_error",
                "message": surface.reason,
                "evidence": {
                    "param_a": surface.param_a, "param_b": surface.param_b,
                    "axis_peak_a": surface.axis_peak_a,
                    "axis_peak_b": surface.axis_peak_b,
                    "compensating_peak": surface.compensating_peak,
                    "stage": "final",
                },
            })

    stagnation = coupling.get("stagnation")
    if stagnation is not None and stagnation.verdict != "converged":
        severity = {"stagnation": "warning",
                    "non_identifiable": "warning",
                    "deficient_observation": "info"}.get(
            stagnation.verdict, "info")
        out.append({
            "severity": severity,
            "code": f"coupling_{stagnation.verdict}",
            "message": stagnation.reason,
            "evidence": {
                "observed_fraction": stagnation.observed_fraction,
                "observed_error": stagnation.observed_error,
                "full_error": stagnation.full_error,
                "initial_loss": stagnation.initial_loss,
                "stage": stagnation.stage or "final",
            },
        })
    return out


def fit_status(result):
    if result.best_loss >= FAIL_LOSS:
        return "failed"
    if result.best_loss >= POOR_LOSS:
        return "partial"
    return "fitted"
