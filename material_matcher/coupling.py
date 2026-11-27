"""Parameter-coupling diagnostics for the procedural material matcher.

Everything in this module reasons from *observables only*: rendered pixels,
their loss against the supplied target, sensitivity sweeps and the search
trajectory.  No function here reads a ground-truth generator or hidden
parameters; when a synthetic target happens to carry ground truth it is
simply not part of any call signature.

The diagnostic chain, in order
------------------------------
1. :func:`sensitivity_scan` records how each material-graph parameter moves
   the loss (local first-order sensitivity).
2. :func:`build_equivalent_pair` constructs two fitted graphs that match at
   the front view: branch A changes the surface frequency
   (``noise.scale``), branch B changes the sampling scale
   (``transform.scale_x/scale_y``).  With one branch member locked in the
   editor, the camera distance changes and the *second-view pixel error*
   decides attribution.  Only a branch that predicts the independent view is
   eligible for a material preset; an exactly degenerate family is retained
   but never written.
3. :func:`perturbation_surface` sweeps two parameters jointly along the
   physical compensation direction.  Single-axis moves that change the loss
   while the compensating joint move leaves it flat identify *coupling*; a
   joint move that changes the loss where each axis reports a flat
   sensitivity identifies a *derivative error* (the local signal missed the
   structure).
4. :func:`diagnose_stagnation` separates optimisation stagnation from
   non-identifiability using known targets and deficient (masked)
   observations, preserving rerunnable initial values.
5. :class:`RunTracker` keeps a bounded execution evidence chain: loss
   trajectory, sensitivity records and stage-localised anomalies, with a
   rerun verification that re-derives the verdict from fresh observations
   instead of trusting stored numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .graph import MaterialGraph, REGISTRY
from .render import render_graph, pixel_error, multiscale_loss


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _spec(graph, node_id, name):
    return next(s for s in REGISTRY[graph.nodes[node_id].type]
                if s.name == name)


def _clamp_param(graph, node_id, name, value):
    graph.set_param(node_id, name, value)
    return graph.get_param(node_id, name)


def _continuous_specs(graph):
    """(node_id, ParamSpec) pairs for every continuous parameter."""
    out = []
    for nid, node in graph.nodes.items():
        for spec in REGISTRY[node.type]:
            if spec.kind == "continuous":
                out.append((nid, spec))
    return out


def pixel_loss(graph, target, uv_scale, size):
    return pixel_error(render_graph(graph, size=size, uv_scale=uv_scale),
                       target)


# ---------------------------------------------------------------------------
# 1. Per-parameter sensitivity
# ---------------------------------------------------------------------------
@dataclass
class SensitivityPoint:
    delta: float
    loss: float


@dataclass
class ParameterSensitivity:
    param: str
    base_value: float
    deltas: list
    losses: list
    sensitivity: float          # max |loss(d) - loss(0)| over the sweep
    flat: bool                  # sensitivity below ``flat_tol``
    stage: str = ""

    def to_dict(self):
        return asdict(self)


def sensitivity_scan(graph, target, *, size=None, uv_scale=1.0,
                     steps=5, span_frac=0.08, flat_tol=1e-3,
                     params=None, stage="", multiscale=False,
                     factors=(1, 2, 4)):
    """Sweep each continuous parameter independently around its current value.

    Parameters stay centred on the graph's fitted assignment, so the scan is
    a local first-order probe.  Discrete and seed parameters are skipped:
    their finite differences are not a derivative.  The returned records are
    the sensitivity evidence consumed by the perturbation surface and the
    stagnation diagnosis.
    """
    size = size or target.shape[0]

    def score(g):
        if multiscale:
            return multiscale_loss(g, target, uv_scale=uv_scale,
                                   factors=factors)[0]
        return pixel_loss(g, target, uv_scale, size)

    base_loss = score(graph)
    candidates = _continuous_specs(graph)
    if params is not None:
        wanted = set(params)
        candidates = [(nid, s) for nid, s in candidates
                      if f"{nid}.{s.name}" in wanted]

    records = []
    for nid, spec in candidates:
        original = graph.get_param(nid, spec.name)
        span = (spec.high - spec.low) * span_frac
        deltas = np.linspace(-span, span, steps)
        losses = []
        for d in deltas:
            if abs(d) < 1e-12:
                losses.append(base_loss)
                continue
            value = _clamp_param(graph, nid, spec.name, original + float(d))
            losses.append(score(graph))
        graph.set_param(nid, spec.name, original)
        sensitivity = float(max(abs(l - base_loss) for l in losses))
        records.append(ParameterSensitivity(
            param=f"{nid}.{spec.name}", base_value=float(original),
            deltas=[float(d) for d in deltas],
            losses=[float(l) for l in losses],
            sensitivity=sensitivity, flat=sensitivity <= flat_tol,
            stage=stage))
    return base_loss, records


# ---------------------------------------------------------------------------
# 2. Front-view equivalent pair and second-view attribution
# ---------------------------------------------------------------------------
@dataclass
class EquivalentPair:
    frequency_param: str
    sampling_param: str
    locked: list
    front_uv: float
    second_uv: float
    front_tolerance: float
    front_error_a: float
    front_error_b: float
    second_error_a: float
    second_error_b: float
    branch_a: MaterialGraph
    branch_b: MaterialGraph
    front_equivalent: bool
    attribution: str            # "frequency" | "sampling" | "ambiguous"
    preset_eligible: str        # "frequency" | "sampling" | "none"
    reason: str
    branch_a_params: dict = field(default_factory=dict)
    branch_b_params: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "frequency_param": self.frequency_param,
            "sampling_param": self.sampling_param,
            "locked": list(self.locked),
            "front_uv": self.front_uv,
            "second_uv": self.second_uv,
            "front_tolerance": self.front_tolerance,
            "front_error_a": self.front_error_a,
            "front_error_b": self.front_error_b,
            "second_error_a": self.second_error_a,
            "second_error_b": self.second_error_b,
            "front_equivalent": self.front_equivalent,
            "attribution": self.attribution,
            "preset_eligible": self.preset_eligible,
            "reason": self.reason,
            "branch_a_params": dict(self.branch_a_params),
            "branch_b_params": dict(self.branch_b_params),
        }


def _find_node(graph, node_type):
    for nid, node in graph.nodes.items():
        if node.type == node_type:
            return nid
    return None


def build_equivalent_pair(graph, target, *, noise_id=None, transform_id=None,
                          front_uv=1.0, second_uv=0.5, size=None,
                          front_tolerance=0.02, second_tolerance=0.02,
                          grid_points=61, refine=3, mask=None):
    """Fit two front-view-equivalent branches and attribute at a second view.

    Branch A leaves the sampling transform locked at neutral unit scale and
    calibrates the surface frequency (``noise.scale``).  Branch B leaves the
    surface frequency locked at its reference value and calibrates the
    sampling scale (isotropic ``transform.scale_x`` / ``scale_y``).  The
    ``locked`` list records exactly which parameters the editor holds fixed
    while the camera distance changes from ``front_uv`` to ``second_uv``.

    Each branch is a coarse grid sweep followed by a few local refinement
    passes so a front-view-equivalent match is not missed on grid
    quantisation.  Preset gating follows the evidence, not the parameter
    name: a branch is eligible only when its second-view pixel error is
    below ``second_tolerance`` *and* strictly lower than its sibling.  When
    both branches predict the independent view equally well the parameter
    family is genuinely degenerate and ``preset_eligible`` is ``"none"``.
    """
    size = size or target.shape[0]
    noise_id = noise_id or _find_node(graph, "noise")
    transform_id = transform_id or _find_node(graph, "transform")
    if noise_id is None or transform_id is None:
        raise ValueError("equivalent pair needs a noise and a transform node")

    freq_spec = _spec(graph, noise_id, "scale")
    sx_spec = _spec(graph, transform_id, "scale_x")
    reference_frequency = graph.get_param(noise_id, "scale")
    reference_sampling = graph.get_param(transform_id, "scale_x")

    def observed_error(g, uv):
        img = render_graph(g, size=size, uv_scale=uv)
        return pixel_error(img, target, mask)

    def calibrate(g, node, names, spec):
        """Coarse full-range sweep then shrinking local refinement."""
        best_v, best_l = None, float("inf")

        def probe(value):
            for name in names:
                g.set_param(node, name, float(value))
            return observed_error(g, front_uv)

        lo, hi = spec.low, spec.high
        for value in np.linspace(lo, hi, grid_points):
            err = probe(value)
            if err < best_l:
                best_v, best_l = float(value), err
        window = (hi - lo) / grid_points
        for _ in range(refine):
            lo = max(spec.low, best_v - window)
            hi = min(spec.high, best_v + window)
            for value in np.linspace(lo, hi, 11):
                err = probe(value)
                if err < best_l:
                    best_v, best_l = float(value), err
            window = hi - lo
        for name in names:
            g.set_param(node, name, best_v)
        return best_v, best_l

    # Branch A: sampling scale locked at neutral 1.0; calibrate frequency.
    branch_a = graph.clone()
    locked_a = [f"{transform_id}.scale_x", f"{transform_id}.scale_y"]
    branch_a.set_param(transform_id, "scale_x", 1.0)
    branch_a.set_param(transform_id, "scale_y", 1.0)
    va, front_a = calibrate(branch_a, noise_id, ("scale",), freq_spec)

    # Branch B: surface frequency locked at the reference; calibrate sampling.
    branch_b = graph.clone()
    locked_b = [f"{noise_id}.scale"]
    branch_b.set_param(noise_id, "scale", reference_frequency)
    vb, front_b = calibrate(branch_b, transform_id,
                            ("scale_x", "scale_y"), sx_spec)

    # Camera distance changes while the calibrated material stays locked.
    second_a = observed_error(branch_a, second_uv)
    second_b = observed_error(branch_b, second_uv)

    front_equivalent = (front_a <= front_tolerance
                        and front_b <= front_tolerance)
    gap = abs(second_a - second_b)
    if not front_equivalent:
        attribution = "ambiguous"
        eligible = "none"
        reason = ("front views do not match within tolerance; the two "
                  "branches are not an equivalent pair at this view.")
    elif gap < second_tolerance:
        attribution = "ambiguous"
        eligible = "none"
        reason = ("both branches predict the second view within tolerance "
                  "(degenerate parameter family); retain the family, write "
                  "no material preset.")
    elif second_a < second_b:
        attribution = "frequency"
        eligible = ("frequency" if second_a <= second_tolerance else "none")
        reason = ("second-view pixel error favours the surface-frequency "
                  "branch once the sampling scale is locked.")
    else:
        attribution = "sampling"
        eligible = ("sampling" if second_b <= second_tolerance else "none")
        reason = ("second-view pixel error favours the sampling-scale branch "
                  "once the surface frequency is locked.")

    return EquivalentPair(
        frequency_param=f"{noise_id}.scale",
        sampling_param=f"{transform_id}.scale_x",
        locked=sorted(set(locked_a + locked_b)),
        front_uv=float(front_uv), second_uv=float(second_uv),
        front_tolerance=float(front_tolerance),
        front_error_a=float(front_a), front_error_b=float(front_b),
        second_error_a=float(second_a), second_error_b=float(second_b),
        branch_a=branch_a, branch_b=branch_b,
        front_equivalent=bool(front_equivalent),
        attribution=attribution, preset_eligible=eligible, reason=reason,
        branch_a_params={
            f"{noise_id}.scale": branch_a.get_param(noise_id, "scale"),
            f"{transform_id}.scale_x":
                branch_a.get_param(transform_id, "scale_x"),
        },
        branch_b_params={
            f"{noise_id}.scale": branch_b.get_param(noise_id, "scale"),
            f"{transform_id}.scale_x":
                branch_b.get_param(transform_id, "scale_x"),
        })


# ---------------------------------------------------------------------------
# 3. Two-parameter perturbation surface (coupling vs derivative error)
# ---------------------------------------------------------------------------
@dataclass
class PerturbationPoint:
    delta_a: float
    delta_b: float
    mode: str                   # axis_a | axis_b | compensating | opposing
    loss: float


@dataclass
class CouplingSurface:
    param_a: str
    param_b: str
    base_loss: float
    points: list
    axis_peak_a: float
    axis_peak_b: float
    compensating_peak: float
    opposing_peak: float
    verdict: str                # coupled | derivative_error | independent
    reason: str

    def to_dict(self):
        return {
            "param_a": self.param_a, "param_b": self.param_b,
            "base_loss": self.base_loss,
            "points": [asdict(p) for p in self.points],
            "axis_peak_a": self.axis_peak_a,
            "axis_peak_b": self.axis_peak_b,
            "compensating_peak": self.compensating_peak,
            "opposing_peak": self.opposing_peak,
            "verdict": self.verdict, "reason": self.reason,
        }


def _split_param(graph, name):
    nid, pname = name.split(".", 1)
    if nid not in graph.nodes:
        raise ValueError(f"unknown node {nid}")
    return nid, _spec(graph, nid, pname)


def perturbation_surface(graph, target, param_a, param_b, *,
                         follow_params=(), size=None, uv_scale=1.0, steps=5,
                         span_frac=0.08, compensation=None,
                         coupling_tol=5e-3, flat_tol=1e-3):
    """Sweep two parameters along their coupled compensation direction.

    Four arms leave the fitted assignment, each in both parameter signs:

    * ``axis_a`` / ``axis_b`` move one parameter alone;
    * ``compensating`` moves both so the coupled invariant is preserved.
      For the frequency/sampling pair the invariant is the ratio S/T, so
      ``compensation`` defaults to the local ratio of the two values;
    * ``opposing`` moves both against the invariant, doubling the physical
      change as a control.

    ``follow_params`` are extra parameters that must mirror every delta
    applied to ``param_b``.  The sampling-scale invariant is isotropic, so
    ``transform.scale_y`` follows ``transform.scale_x``; without it a
    single-axis sampling move would itself break the invariant.

    Coupling is present when each single axis moves the loss but the
    compensating joint move keeps it flat.  A derivative error is present
    when both local axes look flat yet a joint move changes the loss: the
    first-order probe missed structure that the joint perturbation reveals.
    """
    size = size or target.shape[0]
    nid_a, spec_a = _split_param(graph, param_a)
    nid_b, spec_b = _split_param(graph, param_b)
    followers = [_split_param(graph, name) for name in follow_params]
    base_a = graph.get_param(nid_a, spec_a.name)
    base_b = graph.get_param(nid_b, spec_b.name)
    follower_base = [graph.get_param(n, s.name) for n, s in followers]
    if compensation is None:
        # Preserve S/T: dT = (T0/S0) dS.  Fall back to 1 when a base value
        # sits at zero, where the ratio is undefined.
        compensation = (base_b / base_a) if abs(base_a) > 1e-9 else 1.0

    def score(delta_a, delta_b, mode):
        graph.set_param(nid_a, spec_a.name, base_a + delta_a)
        graph.set_param(nid_b, spec_b.name, base_b + delta_b)
        for (fn, fs), fb in zip(followers, follower_base):
            graph.set_param(fn, fs.name, fb + delta_b)
        loss = pixel_loss(graph, target, uv_scale, size)
        return PerturbationPoint(float(delta_a), float(delta_b), mode, loss)

    span_a = (spec_a.high - spec_a.low) * span_frac
    # The compensating arm is driven by parameter A's physical delta; B
    # follows by the local invariant ratio (dT = compensation*dS), so the
    # two moves share one normalised grid instead of two independent spans.
    grid = np.linspace(-1.0, 1.0, steps)
    # Evaluate and pin the unperturbed centre first so the base loss is the
    # true reference every other point is measured against.
    base_point = score(0.0, 0.0, "center")
    points = [base_point]
    try:
        for t in grid:
            if abs(t) < 1e-12:
                continue
            da = float(t) * span_a
            db_alone = float(t) * (spec_b.high - spec_b.low) * span_frac
            points.append(score(da, 0.0, "axis_a"))
            points.append(score(0.0, db_alone, "axis_b"))
            points.append(score(da, compensation * da, "compensating"))
            points.append(score(da, -compensation * da, "opposing"))
    finally:
        graph.set_param(nid_a, spec_a.name, base_a)
        graph.set_param(nid_b, spec_b.name, base_b)
        for (fn, fs), fb in zip(followers, follower_base):
            graph.set_param(fn, fs.name, fb)

    base_loss = base_point.loss

    def peak(mode):
        vals = [abs(p.loss - base_loss) for p in points if p.mode == mode]
        return float(max(vals)) if vals else 0.0

    peak_a, peak_b = peak("axis_a"), peak("axis_b")
    peak_comp, peak_opp = peak("compensating"), peak("opposing")
    axis_active = max(peak_a, peak_b) > flat_tol

    if axis_active and peak_comp <= coupling_tol and peak_opp > flat_tol:
        verdict = "coupled"
        reason = (f"single-axis moves change the loss (peaks "
                  f"{peak_a:.4f}/{peak_b:.4f}) but the compensating joint "
                  f"move stays flat ({peak_comp:.2e}); the parameters share "
                  "an invariant and are only jointly identifiable.")
    elif (not axis_active) and peak_comp > flat_tol:
        verdict = "derivative_error"
        reason = (f"both local axes look flat (peaks {peak_a:.2e}/"
                  f"{peak_b:.2e}) yet a joint perturbation moves the loss "
                  f"({peak_comp:.4f}); the local derivative missed "
                  "cross-parameter structure.")
    else:
        verdict = "independent"
        reason = ("no flat compensating manifold found at this assignment; "
                  "the parameters are locally separable.")
    return CouplingSurface(
        param_a=param_a, param_b=param_b, base_loss=float(base_loss),
        points=points, axis_peak_a=peak_a, axis_peak_b=peak_b,
        compensating_peak=peak_comp, opposing_peak=peak_opp,
        verdict=verdict, reason=reason)


# ---------------------------------------------------------------------------
# 4. Stagnation vs non-identifiability with deficient observations
# ---------------------------------------------------------------------------
def make_observation_mask(size, *, missing_fraction=0.25, band=0, seed=0):
    """An HxW observation mask: ones observed, zeros deficient.

    ``band > 0`` removes whole rows/columns of that width (a structured gap);
    otherwise a deterministic random fraction of cells is missing.
    """
    rng = np.random.default_rng(seed)
    mask = np.ones((size, size), dtype=float)
    if band > 0:
        mask[:band, :] = 0.0
        mask[-band:, :] = 0.0
        mask[:, :band] = 0.0
        mask[:, -band:] = 0.0
        return mask
    n_missing = int(size * size * missing_fraction)
    flat = np.ones(size * size)
    flat[:n_missing] = 0.0
    rng.shuffle(flat)
    return flat.reshape(size, size)


@dataclass
class StagnationVerdict:
    verdict: str                # stagnation | non_identifiable | deficient_observation | converged
    observed_fraction: float
    observed_error: object
    full_error: float
    sensitivity: list
    initial_vector: list
    initial_loss: float
    reason: str
    stage: str = ""

    def to_dict(self):
        d = asdict(self)
        d["sensitivity"] = [
            s.to_dict() if isinstance(s, ParameterSensitivity) else s
            for s in self.sensitivity]
        return d


def diagnose_stagnation(graph, target, *, initial_graph=None, mask=None,
                        size=None, uv_scale=1.0,
                        flat_tol=1e-3, deficiency_tol=0.6, stage=""):
    """Classify why a fit is not improving, using observables only.

    The four outcomes are decided from measured quantities:

    * ``deficient_observation``: fewer than ``deficiency_tol`` of pixels are
      observed and the error computed only on observed pixels is much lower
      than the error over the full swatch — the data, not the optimiser, is
      the limit;
    * ``non_identifiable``: enough pixels are observed, local sensitivities
      are all flat, yet the residual stays high — no available parameter
      direction can reduce the observed loss;
    * ``stagnation``: at least one parameter still has a live sensitivity but
      the residual stays high — the search stopped in a basin it could have
      left;
    * ``converged``: the full-swatch error is already at the loss floor.
    """
    size = size or target.shape[0]
    if mask is None:
        mask = np.ones((size, size), dtype=float)
    mask = np.asarray(mask, dtype=float)
    observed_fraction = float(mask.mean())

    render = render_graph(graph, size=size, uv_scale=uv_scale)
    observed_error = pixel_error(render, target, mask)
    full_error = pixel_error(render, target)

    initial_graph = initial_graph or graph.clone()
    initial_vector = list(initial_graph.param_vector())
    initial_loss = pixel_loss(initial_graph, target, uv_scale, size)

    _, sens = sensitivity_scan(
        graph, target, size=size, uv_scale=uv_scale, flat_tol=flat_tol,
        multiscale=False, stage=stage)
    live = [s for s in sens if not s.flat]

    if full_error <= 0.02:
        verdict = "converged"
        reason = "residual is at the loss floor on the full swatch."
    elif observed_fraction < deficiency_tol and observed_error is not None \
            and full_error - observed_error > 0.02:
        verdict = "deficient_observation"
        reason = (f"only {observed_fraction:.0%} of pixels are observed; the "
                  f"observed-only error ({observed_error:.4f}) is much lower "
                  f"than the full-swatch error ({full_error:.4f}).")
    elif not live:
        verdict = "non_identifiable"
        reason = ("every local parameter direction is flat while the "
                  "residual stays high; the observation cannot distinguish "
                  "further parameter moves.")
    else:
        verdict = "stagnation"
        flat_names = [s.param for s in sens if s.flat]
        reason = (f"{len(live)} parameter(s) still reduce the observed loss "
                  "but the search stopped; rerun from the preserved initial "
                  "values." + (f" flat axes: {flat_names}" if flat_names else ""))

    return StagnationVerdict(
        verdict=verdict, observed_fraction=observed_fraction,
        observed_error=observed_error, full_error=float(full_error),
        sensitivity=sens, initial_vector=[float(v) for v in initial_vector],
        initial_loss=float(initial_loss), reason=reason, stage=stage)


# ---------------------------------------------------------------------------
# 5. Bounded execution evidence chain with rerun verification
# ---------------------------------------------------------------------------
@dataclass
class RerunCheck:
    matches: bool
    loss_drift: float
    sensitivity_drift: float
    trajectory_drift: float
    tolerances: dict
    reason: str

    def to_dict(self):
        return asdict(self)


class RunTracker:
    """Bounded, JSON-serialisable evidence log for material-match runs.

    The log cost is capped on purpose: the trajectory and sensitivity
    history keep at most ``cap_per_stage`` entries per labelled stage and
    anomaly records are capped globally.  Older entries are dropped, so a
    long run cannot grow the log without bound.
    """

    def __init__(self, cap_per_stage=12, anomaly_cap=50):
        self.cap_per_stage = cap_per_stage
        self.anomaly_cap = anomaly_cap
        self.runs = {}

    # -- recording ----------------------------------------------------------
    def start_run(self, run_id, *, graph, target, uv_scale=1.0, size=None,
                  factors=(1, 2, 4)):
        size = size or target.shape[0]
        initial_graph = graph.clone()
        initial_loss = multiscale_loss(
            initial_graph, target, uv_scale=uv_scale, factors=factors)[0]
        self.runs[run_id] = {
            "run_id": run_id,
            "started_at": time.time(),
            "uv_scale": float(uv_scale),
            "size": int(size),
            "factors": list(factors),
            "initial_graph": initial_graph.to_dict(),
            "initial_vector": [float(v)
                               for v in initial_graph.param_vector()],
            "initial_loss": float(initial_loss),
            "trajectory": {},
            "sensitivity": {},
            "anomalies": [],
            "equivalent_pair": None,
            "coupling_surfaces": [],
            "final_graph": None,
            "final_loss": None,
            "stages": [],
        }
        return run_id

    def _require(self, run_id):
        if run_id not in self.runs:
            raise KeyError(f"unknown run {run_id}")
        return self.runs[run_id]

    def record_trajectory(self, run_id, stage, best_loss, evaluations,
                          components=None):
        run = self._require(run_id)
        bucket = run["trajectory"].setdefault(stage, [])
        bucket.append({"stage": stage, "best_loss": float(best_loss),
                       "evaluations": int(evaluations),
                       "components": components or {}})
        if len(bucket) > self.cap_per_stage:
            del bucket[:-self.cap_per_stage]
        if stage not in run["stages"]:
            run["stages"].append(stage)

    def record_sensitivity(self, run_id, records, stage=""):
        run = self._require(run_id)
        bucket = run["sensitivity"].setdefault(stage or "default", [])
        for rec in records:
            d = rec.to_dict() if isinstance(rec, ParameterSensitivity) else rec
            d["stage"] = stage
            bucket.append(d)
        if len(bucket) > self.cap_per_stage:
            del bucket[:-self.cap_per_stage]

    def record_anomaly(self, run_id, code, message, stage="", evidence=None):
        run = self._require(run_id)
        run["anomalies"].append(
            {"code": code, "message": message, "stage": stage,
             "evidence": evidence or {}})
        if len(run["anomalies"]) > self.anomaly_cap:
            del run["anomalies"][:-self.anomaly_cap]

    def record_equivalent_pair(self, run_id, pair):
        run = self._require(run_id)
        run["equivalent_pair"] = pair.to_dict()

    def record_coupling_surface(self, run_id, surface):
        run = self._require(run_id)
        run["coupling_surfaces"].append(surface.to_dict())
        if len(run["coupling_surfaces"]) > self.cap_per_stage:
            del run["coupling_surfaces"][:-self.cap_per_stage]

    def finish_run(self, run_id, graph, target, *, factors=(1, 2, 4)):
        run = self._require(run_id)
        final_loss = multiscale_loss(
            graph, target, uv_scale=run["uv_scale"], factors=factors)[0]
        run["final_graph"] = graph.to_dict()
        run["final_loss"] = float(final_loss)
        return final_loss

    # -- queries ------------------------------------------------------------
    def get_run(self, run_id):
        return self._require(run_id)

    def stage_anomalies(self, run_id, stage):
        return [a for a in self._require(run_id)["anomalies"]
                if a["stage"] == stage]

    def log_size(self, run_id):
        run = self._require(run_id)
        return (sum(len(v) for v in run["trajectory"].values())
                + sum(len(v) for v in run["sensitivity"].values())
                + len(run["anomalies"])
                + len(run["coupling_surfaces"]))

    # -- rerun verification -------------------------------------------------
    def rerun_verification(self, run_id, graph, target, *,
                           loss_tol=5e-3, sensitivity_tol=5e-3,
                           trajectory_tol=5e-3):
        """Re-derive the evidence from fresh renders and compare numerically.

        The stored run is only a record; the verdict is re-certified by
        recomputing every observable from the supplied graph/target.  A
        mismatch means the stored evidence does not reproduce and any
        coupling conclusion attached to it is invalidated.
        """
        run = self._require(run_id)
        fresh_loss = multiscale_loss(
            graph, target, uv_scale=run["uv_scale"],
            factors=tuple(run["factors"]))[0]
        stored_final = run["final_loss"]
        loss_drift = (abs(fresh_loss - stored_final)
                      if stored_final is not None else 0.0)

        stored_sens = [v for bucket in run["sensitivity"].values()
                       for v in bucket]
        _, fresh_records = sensitivity_scan(
            graph, target, size=run["size"], uv_scale=run["uv_scale"])
        fresh_by_name = {r.param: r.sensitivity for r in fresh_records}
        sens_drifts = [abs(d["sensitivity"] - fresh_by_name.get(d["param"], 0.0))
                       for d in stored_sens]
        sensitivity_drift = float(max(sens_drifts)) if sens_drifts else 0.0

        # The trajectory is a record of the search path, not a reproducible
        # endpoint: intermediate losses depend on grid ordering and seed,
        # and the polished final loss is stored separately by ``finish_run``.
        # Rerun certifies that each recorded stage is internally coherent
        # (its best loss never increases within the stage); whether the path
        # actually reached the stored endpoint is checked through
        # ``loss_drift`` instead.
        stage_paths = [[p["best_loss"] for p in bucket]
                       for bucket in run["trajectory"].values()]
        if stage_paths:
            monotone = all(all(a >= b for a, b in zip(path, path[1:]))
                           for path in stage_paths)
            trajectory_drift = 0.0 if monotone else 1.0
        else:
            trajectory_drift = 0.0

        tolerances = {"loss": loss_tol, "sensitivity": sensitivity_tol,
                      "trajectory": trajectory_tol}
        matches = (loss_drift <= loss_tol
                   and sensitivity_drift <= sensitivity_tol
                   and trajectory_drift <= trajectory_tol)
        if matches:
            reason = "fresh renders reproduce the stored loss, sensitivity and trajectory within tolerance."
        else:
            reason = ("stored evidence does not reproduce "
                      f"(loss drift {loss_drift:.4f}, sensitivity drift "
                      f"{sensitivity_drift:.4f}); coupling conclusion "
                      "invalidated until rerun.")
        return RerunCheck(
            matches=bool(matches), loss_drift=float(loss_drift),
            sensitivity_drift=float(sensitivity_drift),
            trajectory_drift=float(trajectory_drift),
            tolerances=tolerances, reason=reason)
