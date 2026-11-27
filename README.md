# Procedural Material Matcher

Inverse-fit an **editable procedural material graph** to a target texture,
with an explicit parameter-coupling diagnostic chain. Appearance on a single
swatch never certifies a unique parameter set, so the tool records why
parameters are ambiguous, which branches predict an independent view, and
which conclusions are reproducible — instead of silently writing a preset.

## Entry points

### Primary: the web authoring UI (graphical)

The browser page is the main material-matching surface:

```bash
python3 -m material_matcher
# or, equivalently
python3 -m material_matcher.web_app --port 8000
```

Then open <http://127.0.0.1:8000>. The UI provides, in one screen:

1. **Target appearance** — load a procedural (known-truth) or
   inexpressible target, choose template/coordinate space/render size.
2. **Photometric edits** — brightness/contrast/hue change only the pixels
   the optimiser sees; the original image file is never handed to the
   solver.
3. **Inverse search** — live run state, evaluation count, per-stage loss
   trajectory and the fitted graph render next to the target and an
   amplified difference image.
4. **Deficient-observation mask** — hide a border band or a random fraction
   of pixels so stagnation can be separated from non-identifiability.
5. **Recovered material graph** — every node parameter with its spec,
   downloadable as editable JSON (never a baked texture).
6. **Equivalent solutions** — all parameter clusters within tolerance of
   the best loss.
7. **Anomalies** — named, evidence-backed flags linking parameters to the
   render difference.
8. **Witness decal** — object- vs. world-space sampling after a mesh move.
9. **Parameter coupling diagnostics** — the panel described below.

### Auxiliary: the command line

The CLI is a convenience for scripting and headless checks only; it does
not replace the graphical workflow.

```bash
python3 -m material_matcher.cli demo --outdir workdir/demo
python3 -m material_matcher.cli fit path/to/target.png --outdir workdir/fit
python3 -m material_matcher.cli render fitted_graph.json out.png
python3 -m material_matcher.cli coupling --size 48 --mask-band 8 \
    --outdir workdir/coupling
```

The `coupling` subcommand writes `coupling.json` with the equivalent-pair
attribution, perturbation-surface verdicts and stagnation classification.

## The parameter-coupling diagnostic chain

Every probe recomputes values from rendered pixels. Ground-truth generator
parameters are never read by a diagnostic.

### 1. Per-parameter sensitivity

`sensitivity_scan` sweeps each continuous parameter independently around
the fitted assignment and records the loss at every delta, the maximum
loss movement (sensitivity) and whether the axis is locally **flat**. These
records feed both the perturbation surface and the stagnation diagnosis.

### 2. Equivalent pair and second-view attribution

Two fitted graphs are built to look identical at the front view:

* **branch A** changes the *surface frequency* (`noise.scale`) while the
  sampling transform is locked at neutral scale;
* **branch B** changes the *sampling scale* (isotropic
  `transform.scale_x`/`scale_y`) while the surface frequency is locked.

The `locked` list states exactly which parameters the editor holds fixed.
The camera distance then changes from the front `uv` to a second `uv`, and
the **second-view pixel error** decides attribution:

* both branches still match → a degenerate **parameter family**: it is
  retained, `preset_eligible = none`, and nothing is written to a material
  preset;
* one branch predicts the independent view within tolerance and strictly
  better than its sibling → only that branch is preset-eligible.

### 3. Two-parameter perturbation surface

`perturbation_surface` sweeps two parameters along their physical
compensation direction. For the frequency/sampling pair the invariant is
the ratio S/T, so `transform.scale_y` follows `scale_x` to keep sampling
isotropic. Four arms are recorded: each axis alone, the *compensating*
joint move (invariant preserved) and the *opposing* joint move (control).

* single-axis moves change the loss while the compensating move stays flat
  → **coupled** — the parameters are only jointly identifiable;
* both local axes look flat but a joint move changes the loss →
  **derivative_error** — the local first-order probe missed cross-parameter
  structure;
* otherwise **independent**.

### 4. Stagnation vs. non-identifiability

`diagnose_stagnation` classifies an unimproving fit from measured
quantities, and preserves the rerunnable initial parameter vector:

* `converged` — full-swatch error is at the loss floor;
* `deficient_observation` — too few pixels are observed and the
  observed-only error is much lower than the full-swatch error;
* `non_identifiable` — enough pixels are observed but every local parameter
  direction is flat while the residual stays high;
* `stagnation` — at least one parameter still reduces the observed loss but
  the search stopped; rerun from the preserved initial values.

### 5. Bounded evidence log and rerun verification

`RunTracker` keeps the loss trajectory, sensitivity records, equivalent
pair, perturbation surfaces and stage-localised anomalies. Log cost is
capped: each stage keeps at most `cap_per_stage` entries and anomalies are
capped globally, so long runs cannot grow the log without bound.

`rerun_verification` re-derives the loss and sensitivities from fresh
renders instead of trusting stored numbers. A drift beyond tolerance
invalidates the coupling conclusion attached to that run.

## Tests

```bash
python3 -m unittest discover -s tests
```

The coupling tests (`tests/test_coupling.py`) drive the probes directly on
controlled graphs so they stay fast; the search-driven tests exercise the
full pipeline and take longer.
