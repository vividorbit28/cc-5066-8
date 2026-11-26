"""Inverse parameter search over procedural material graphs.

Search design
-------------
The space mixes continuous parameters with discrete choices and integer
seeds, and the objective is multimodal: several parameter assignments can
produce the same local statistics.  Two design choices make the problem
tractable without pretending a unique answer exists:

1. **Mixed discrete/continuous operators.** An outer layer enumerates the
   discrete configurations (noise kind, octaves, blend mode) and sweeps
   random seeds; inside each fixed configuration, a dense full-range
   axis-wise grid search plus coordinate descent handles the continuous
   coordinates.
2. **Grouped, staged optimisation.** The continuous parameters are split
   into small groups (pattern geometry, threshold shaping, colour ramp).
   A discrete prescreen first ranks configurations against global
   statistics (histogram, radial spectrum, gradient distribution) on a
   small render, which tolerates phase misalignment; survivors are then
   fitted to the pixel-level loss at full resolution.  Small groups keep
   the dense grid tractable instead of spreading it over a 20+ dimensional
   box.

All assignments within a tolerance of the best loss are retained and
clustered into *equivalent solutions*, because appearance alone cannot
certify which cluster generated the target.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from .graph import MaterialGraph, spec_index
from .render import multiscale_loss


# The narrowest, strongest-coupled basin is the (frequency, rotation,
# threshold band) quartet; keeping this group small is what makes the
# shrinking axis-grid search reliable.  Secondary fractal and colour
# parameters are released afterwards, once the structure is locked in.
CORE_SUFFIXES = ("scale", "rotation", "low", "high")
SHAPING_SUFFIXES = ("softness", "lacunarity", "gain")
COLOR_SUFFIXES = ("stop0_r", "stop0_g", "stop0_b", "stop1_r", "stop1_g",
                  "stop1_b", "stop2_r", "stop2_g", "stop2_b", "roughness")
PHASE_SUFFIXES = ("scale_x", "scale_y", "offset_x", "offset_y")


@dataclass
class SearchSpace:
    specs: list
    is_integer: list
    low: np.ndarray
    high: np.ndarray
    names: list

    @staticmethod
    def from_graph(graph: MaterialGraph):
        specs = spec_index(graph)
        low, high, is_int, names = [], [], [], []
        for nid, spec in specs:
            names.append(f"{nid}.{spec.name}")
            if spec.kind == "discrete":
                low.append(0)
                high.append(len(spec.choices) - 1)
                is_int.append(True)
            elif spec.kind == "seed":
                low.append(spec.low)
                high.append(spec.high)
                is_int.append(True)
            else:
                low.append(spec.low)
                high.append(spec.high)
                is_int.append(False)
        return SearchSpace(specs, is_int,
                           np.array(low, float), np.array(high, float), names)

    @property
    def dim(self):
        return len(self.specs)

    def group_dims(self, suffixes):
        dims = []
        for j, (nid, spec) in enumerate(self.specs):
            if not self.is_integer[j] and spec.name in suffixes:
                dims.append(j)
        return np.array(dims, int)

    def random_vector(self, rng):
        vec = rng.uniform(self.low, self.high)
        for j, is_int in enumerate(self.is_integer):
            if is_int:
                vec[j] = rng.integers(self.low[j], self.high[j] + 1)
        return vec

    def repair(self, vec):
        v = np.clip(vec, self.low, self.high)
        for j, is_int in enumerate(self.is_integer):
            if is_int:
                v[j] = int(round(v[j]))
        return v

    def apply(self, graph, vec):
        for (nid, spec), raw in zip(self.specs, vec):
            if spec.kind == "discrete":
                value = spec.choices[int(round(raw))]
            else:
                value = spec.clamp(raw)
            graph.nodes[nid].params[spec.name] = value

    def describe(self, vec):
        out = {}
        for (nid, spec), raw in zip(self.specs, vec):
            if spec.kind == "discrete":
                out[f"{nid}.{spec.name}"] = spec.choices[int(round(raw))]
            else:
                out[f"{nid}.{spec.name}"] = float(raw)
        return out


@dataclass
class TrajectoryPoint:
    iteration: int
    evaluations: int
    best_loss: float
    mean_loss: float
    best_params: dict
    components: dict
    elapsed: float
    stage: str = ""


@dataclass
class EquivalentSolution:
    params: dict
    loss: float
    components: dict
    cluster_id: int


@dataclass
class SearchResult:
    best_params: dict
    best_loss: float
    best_components: dict
    graph: MaterialGraph
    trajectory: list
    equivalents: list
    evaluations: int
    warnings: list
    success: bool


def _parameter_distance(space, a, b):
    d = 0.0
    for j, (nid, spec) in enumerate(space.specs):
        if spec.kind in ("discrete", "seed"):
            d += 0.0 if int(round(a[j])) == int(round(b[j])) else 1.0
        else:
            span = max(spec.high - spec.low, 1e-9)
            d += abs(a[j] - b[j]) / span
    return d / len(space.specs)


def _cluster(space, vectors, losses, radius=0.12):
    clusters = []
    for vec, loss in zip(vectors, losses):
        for c in clusters:
            if _parameter_distance(space, vec, c["center"]) <= radius:
                c["members"].append((vec, loss))
                if loss < c["best_loss"]:
                    c["center"] = vec.copy()
                    c["best_loss"] = loss
                break
        else:
            clusters.append({"center": vec.copy(), "best_loss": loss,
                             "members": [(vec, loss)]})
    clusters.sort(key=lambda c: c["best_loss"])
    for i, c in enumerate(clusters):
        c["id"] = i
    return clusters


class _Runner:
    def __init__(self, space, work, target, factors, uv_scale, callback,
                 seed=0, screen_target=None, screen_factors=None):
        self.space = space
        self.work = work
        self.target = target
        self.factors = factors
        self.uv_scale = uv_scale
        self.callback = callback
        # A smaller render of the target used only for the discrete
        # prescreen, where per-evaluation speed matters more than precision.
        self.screen_target = screen_target
        self.screen_factors = screen_factors or factors
        self.rng = np.random.default_rng(seed)
        self.evals = 0
        self.warn = False
        self.history = []
        self.t0 = time.time()
        self.archive_vec, self.archive_loss, self.archive_comp = [], [], []

    def objective(self, vec, profile):
        self.space.apply(self.work, vec)
        if profile == "screen" and self.screen_target is not None:
            target, factors = self.screen_target, self.screen_factors
        else:
            target, factors = self.target, self.factors
        loss, comp, _, mean_warn = multiscale_loss(
            self.work, target, uv_scale=self.uv_scale,
            factors=factors, profile="fine" if profile == "screen"
            else profile)
        self.evals += 1
        self.warn = self.warn or mean_warn
        if self.callback:
            self.callback("eval", {"evaluations": self.evals, "loss": loss})
        return loss, comp

    def record(self, stage, vecs, fits, comps, iteration):
        best_i = int(np.argmin(fits))
        point = TrajectoryPoint(
            iteration=iteration, evaluations=self.evals,
            best_loss=float(fits[best_i]),
            mean_loss=float(np.mean(fits)),
            best_params=self.space.describe(vecs[best_i]),
            components=comps[best_i],
            elapsed=time.time() - self.t0, stage=stage)
        self.history.append(point)
        if self.callback:
            self.callback("iteration", point.__dict__)

    def coordinate_descent(self, center, dims, profile, steps):
        best_vec = center.copy()
        best_loss, best_comp = self.objective(best_vec, profile)
        step = (self.space.high - self.space.low) * 0.05
        for _ in range(steps):
            improved = False
            for j in dims:
                for direction in (-1.0, 1.0):
                    trial = best_vec.copy()
                    trial[j] += direction * step[j]
                    trial = self.space.repair(trial)
                    trial_loss, trial_comp = self.objective(trial, profile)
                    if trial_loss < best_loss:
                        best_vec, best_loss, best_comp = \
                            trial, trial_loss, trial_comp
                        improved = True
            step *= 0.6
            if not improved and step[dims].max() < 1e-3:
                break
        return best_vec, best_loss, best_comp

    def axis_grid_search(self, center, dims, profile, sweeps=5,
                         grid_points=21, shrink=0.0):
        """Dense full-range axis-wise grid search.

        The appearance loss has narrow peaks (exact rotation / phase) on
        broad flat plateaus, and its coupled parameters only improve
        jointly: a first sweep typically ends at a wrong-but-plausible
        frequency, then a second full sweep jumps to the true basin once
        the sibling axes (rotation, threshold) have moved.  Window
        contraction breaks that mechanism by removing the true basin from
        later grids, so this search keeps a uniform dense grid over the
        *full* range on every sweep.  ``shrink > 0`` restricts the window
        for the secondary groups that already start inside the core basin.
        """
        best_vec = center.copy()
        best_loss, best_comp = self.objective(best_vec, profile)
        lo = self.space.low.copy()
        hi = self.space.high.copy()
        for sweep in range(sweeps):
            for j in dims:
                if shrink > 0 and sweep > 0:
                    grid = np.linspace(lo[j], hi[j], grid_points)
                else:
                    grid = np.linspace(self.space.low[j],
                                       self.space.high[j], grid_points)
                axis_best_loss, axis_best_x = best_loss, best_vec[j]
                for x in grid:
                    trial = best_vec.copy()
                    trial[j] = x
                    trial = self.space.repair(trial)
                    trial_loss, trial_comp = self.objective(trial, profile)
                    if trial_loss < axis_best_loss:
                        axis_best_loss = trial_loss
                        axis_best_x = x
                        best_comp = trial_comp
                best_vec[j] = axis_best_x
                best_loss = min(best_loss, axis_best_loss)
                if shrink > 0:
                    span = hi[j] - lo[j]
                    lo[j] = max(self.space.low[j],
                                axis_best_x - shrink * span)
                    hi[j] = min(self.space.high[j],
                                axis_best_x + shrink * span)
        return best_vec, best_loss, best_comp


def inverse_search(graph: MaterialGraph, target, *,
                   max_configs=4, seed_trials=2,
                   equivalent_tol=0.05, cluster_radius=0.12,
                   refine_steps=3, seed=0, callback=None,
                   factors=(1, 2, 4), uv_scale=1.0, screen_size=32):
    rng = np.random.default_rng(seed)
    space = SearchSpace.from_graph(graph)
    work = graph.clone()

    full_size = target.shape[0]
    screen_target = target
    screen_factors = factors
    if screen_size and screen_size < full_size:
        from .render import _downscale
        # Integer downscale for the shallow prescreen render.  Deep
        # prescreen and all refinement always use the full-resolution
        # target so discrete choices stay separable.
        factor = max(1, full_size // screen_size)
        screen_target = _downscale(target, factor)
        screen_factors = tuple(f for f in factors
                               if f <= max(screen_target.shape[0] // 8, 1))
        screen_factors = screen_factors or (1,)
    runner = _Runner(space, work, target, factors, uv_scale, callback,
                     seed=seed, screen_target=screen_target,
                     screen_factors=screen_factors)

    int_dims_idx = np.where(np.array(space.is_integer))[0]
    non_seed_int = [j for j in int_dims_idx
                    if space.specs[j][1].kind == "discrete"]
    seed_dims = [j for j in int_dims_idx
                 if space.specs[j][1].kind == "seed"]
    grid_axes = [range(int(space.low[j]), int(space.high[j]) + 1)
                 for j in non_seed_int]
    grid_combos = list(itertools.product(*grid_axes)) if grid_axes else [()]
    # Seeds are integer-valued and span a huge range, so an unconstrained
    # random sweep almost never redraws the graph's own seed and explodes the
    # configuration count.  Anchor the candidate seeds on the graph's current
    # value (a fitted graph typically starts near the right generator) and
    # add a small deterministic spread of neighbours instead.
    seed_choices = []
    for j in seed_dims:
        current = int(round(work.param_vector()[j]))
        lo = int(space.low[j])
        hi = int(space.high[j])
        candidates = {current}
        spread = max(1, (hi - lo) // 200)
        for k in range(1, seed_trials):
            candidates.add(int(np.clip(
                current + rng.choice([-1, 1]) * k * spread, lo, hi)))
        while len(candidates) < min(seed_trials, hi - lo + 1):
            candidates.add(int(rng.integers(lo, hi + 1)))
        seed_choices.append((j, sorted(candidates)))
    seed_combos = list(itertools.product(
        *[c for _, c in seed_choices])) if seed_choices else [()]

    # Every configuration starts from the graph's current continuous values
    # (the engineering defaults).  Randomising continuous coordinates here
    # makes the deterministic axis-grid search path-dependent and reliably
    # traps it in local minima; only discrete choices and seeds vary between
    # configurations, while the staged grid search explores the continuous
    # space itself.
    base_continuous = work.param_vector()
    configurations = []
    for combo in grid_combos:
        for seed_combo in seed_combos:
            vec = base_continuous.copy()
            for j, v in zip(non_seed_int, combo):
                vec[j] = v
            for (j, _), v in zip(seed_choices, seed_combo):
                vec[j] = v
            configurations.append(vec)

    # Discrete prescreen: the full staged pipeline is expensive, but a
    # short core-group grid fit cheaply reveals which discrete choices
    # (noise kind / octaves / blend mode / seed) can even reach the target
    # structure.  Rank configurations on that score and only fully optimise
    # the survivors, so a bad octave count cannot win by default.
    core_dims = space.group_dims(CORE_SUFFIXES)
    neutral = {sfx: (1.0 if sfx.startswith("scale_") else 0.0)
               for sfx in PHASE_SUFFIXES}

    def neutralize(vec):
        out = vec.copy()
        for j, (nid, spec) in enumerate(space.specs):
            if spec.name in neutral and not space.is_integer[j]:
                out[j] = neutral[spec.name]
        return out

    # Two-tier discrete prescreen.  A shallow, small-render grid fit ranks
    # every configuration cheaply; the handful of survivors then get the
    # reliable dense 25-point core fit at full resolution.  The shallow tier
    # mis-ranks near-degenerate choices (e.g. octave counts), so it keeps a
    # generous survivor pool and the deep tier does the actual discrimination
    # on full-resolution renders.
    shallow_pool = min(max_configs * 5, len(configurations))
    shallow = []
    for cfg in configurations:
        _, shallow_loss, _ = runner.axis_grid_search(
            neutralize(cfg), core_dims, "screen",
            sweeps=2, grid_points=11)
        shallow.append((shallow_loss, cfg))
    shallow.sort(key=lambda x: x[0])
    survivors = [cfg for _, cfg in shallow[:shallow_pool]]

    screened = []
    for cfg in survivors:
        # Deep tier runs on the full-resolution target under the fine
        # objective, which is what separates near-degenerate octave counts.
        probe_vec, screen_loss, screen_comp = runner.axis_grid_search(
            neutralize(cfg), core_dims, "fine",
            sweeps=4, grid_points=25)
        screened.append((screen_loss, cfg, probe_vec))
    screened.sort(key=lambda x: x[0])
    configurations = [(cfg, vec) for _, cfg, vec in screened[:max_configs]]

    groups = [
        ("core", space.group_dims(CORE_SUFFIXES)),
        ("shaping", space.group_dims(SHAPING_SUFFIXES)),
        ("color", space.group_dims(COLOR_SUFFIXES)),
        ("phase", space.group_dims(PHASE_SUFFIXES)),
    ]
    groups = [(name, dims) for name, dims in groups if dims.size]
    all_cont = np.where(~np.array(space.is_integer))[0]

    global_best = {"vec": None, "loss": np.inf, "comp": None}
    iteration = 0

    for cfg_i, (base_vec, core_vec) in enumerate(configurations):
        center = core_vec.copy()
        for stage, (gname, dims) in enumerate(groups):
            if gname == "core":
                # Reuse the prescreen core fit (same 25-point search); one
                # extra sweep lets the winner tighten within its basin.
                center, loss, comp = runner.axis_grid_search(
                    center, dims, "fine",
                    sweeps=2, grid_points=25)
            elif gname == "phase":
                # Translation offsets on a finite swatch are intrinsically
                # ambiguous; only polish them locally.
                center, loss, comp = runner.coordinate_descent(
                    center, dims, "fine", max(2, refine_steps // 2))
            else:
                # Secondary groups start inside the already-locked core
                # basin, so the cheaper full-range axis grid suffices.
                center, loss, comp = runner.axis_grid_search(
                    center, dims, "fine",
                    sweeps=max(3, refine_steps), grid_points=13)
            iteration += 1
            runner.record(f"cfg {cfg_i + 1} {gname}", [center], [loss],
                          [comp], iteration)

        # Joint polish over every continuous dimension under the fine loss.
        center, loss, comp = runner.axis_grid_search(
            center, all_cont, "fine", sweeps=max(2, refine_steps),
            grid_points=9)
        runner.record(f"cfg {cfg_i + 1} joint-grid", [center], [loss],
                      [comp], iteration)
        # Final small-step polish breaks through the grid quantisation limit
        # (the axis grid can only land on its sampled coordinates).
        center, loss, comp = runner.coordinate_descent(
            center, all_cont, "fine", max(4, refine_steps * 2))
        iteration += 1
        runner.record(f"cfg {cfg_i + 1} joint-polish", [center], [loss],
                      [comp], iteration)

        runner.archive_vec.append(center.copy())
        runner.archive_loss.append(loss)
        runner.archive_comp.append(comp)
        if loss < global_best["loss"]:
            global_best = {"vec": center.copy(), "loss": loss, "comp": comp}

    best_vec = global_best["vec"]
    best_loss = global_best["loss"]
    best_comp = global_best["comp"]
    space.apply(graph, best_vec)

    archive_vec = np.array(runner.archive_vec)
    archive_loss = np.array(runner.archive_loss)
    tol = max(equivalent_tol, best_loss * 0.2)
    keep = archive_loss <= best_loss + tol
    clusters = _cluster(space, archive_vec[keep], archive_loss[keep],
                        radius=cluster_radius)
    equivalents = []
    for c in clusters:
        vec = c["center"]
        space.apply(work, vec)
        _, ccomp, _, _ = multiscale_loss(work, target, uv_scale=uv_scale,
                                         factors=factors)
        equivalents.append(EquivalentSolution(
            params=space.describe(vec), loss=float(c["best_loss"]),
            components=ccomp, cluster_id=c["id"]))
    equivalents.sort(key=lambda e: e.loss)

    warnings = []
    if runner.warn:
        warnings.append(
            "mean_colour_collapse: one or more candidates matched the "
            "average colour while losing texture contrast; penalised.")
    if len(equivalents) > 1:
        warnings.append(
            f"{len(equivalents)} distinct parameter clusters fit within "
            f"tolerance {tol:.4f}; appearance is ambiguous from this "
            "single observation and all clusters are retained.")

    return SearchResult(
        best_params=space.describe(best_vec),
        best_loss=float(best_loss),
        best_components=best_comp,
        graph=graph,
        trajectory=runner.history,
        equivalents=equivalents,
        evaluations=runner.evals,
        warnings=warnings,
        success=best_loss < 0.12)
