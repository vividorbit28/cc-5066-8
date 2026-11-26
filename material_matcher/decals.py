"""Witness decal layout: object-space vs world-space equivalence tests.

Two fitting-equivalent graphs are handed to the decal compositor: one
samples its pattern in object coordinates, the other in world coordinates.
On a single static swatch the two are indistinguishable, because object and
world coordinates coincide for an unmoved mesh.

After the decalled mesh is moved, the two behaviours diverge:

* object space  -> the pattern is locked to the mesh and slides with it;
* world space   -> the pattern stays grounded in the world and appears to
  flow across the decal boundary.

The witness decal stores a border ring of pixels; after the move we measure
the colour discontinuity across the decal boundary and compare it with the
unmoved baseline.  Parameters whose predictions stay consistent under the
move (and, when periodic aliasing makes the first scale ambiguous, under a
second observation scale) remain accepted as same-solution equivalents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .graph import MaterialGraph, spec_index
from .render import multiscale_loss


def shifted_grid(size, uv_scale, shift_x=0.0, shift_y=0.0):
    """Coordinate grid shifted by a world-space translation."""
    coords = (np.arange(size) + 0.5) / size * uv_scale - uv_scale / 2.0
    x, y = np.meshgrid(coords, coords)
    return x + shift_x, y + shift_y


def render_in_space(graph, size, uv_scale, space, shift_x=0.0, shift_y=0.0):
    """Render a graph as the decal compositor would after a mesh move.

    Object-space sampling is locked to the mesh, so its coordinates are
    unchanged by the move.  World-space sampling stays grounded, so its
    coordinates follow the world translation.
    """
    if space == "world":
        x, y = shifted_grid(size, uv_scale, shift_x, shift_y)
    else:
        x, y = shifted_grid(size, uv_scale, 0.0, 0.0)
    return graph.evaluate(x, y, space_override=space)


def boundary_discontinuity(img, border=6):
    """Mean absolute colour jump across the inner/outer border ring."""
    inner = img[border:-border, border:-border]
    outer = img.copy()
    outer[border:-border, border:-border] = inner
    diff = np.abs(img - outer)
    ring = np.ones(img.shape[:2], bool)
    ring[border:-border, border:-border] = False
    return float(diff[ring].mean()) if ring.any() else 0.0


def _normalise_param(value, spec):
    if spec.kind in ("discrete", "seed"):
        return int(round(value))
    span = max(spec.high - spec.low, 1e-9)
    return float((value - spec.low) / span)


def _param_distance(graph_a, graph_b):
    out = {}
    for nid, spec in spec_index(graph_a):
        va = graph_a.get_param(nid, spec.name)
        vb = graph_b.get_param(nid, spec.name)
        name = f"{nid}.{spec.name}"
        if spec.kind in ("discrete", "seed"):
            if spec.choices and not isinstance(
                    spec.choices[0], (int, float)):
                out[name] = 0.0 if va == vb else 1.0
            else:
                out[name] = 0.0 if int(round(va)) == int(round(vb)) else 1.0
        else:
            out[name] = abs(_normalise_param(va, spec)
                            - _normalise_param(vb, spec))
    return out


@dataclass
class WitnessReport:
    shift: tuple
    baseline_boundary_error: float
    object_boundary_error: float
    world_boundary_error: float
    object_motion_residual: float
    world_motion_residual: float
    second_scale_motion_residual: float
    second_scale_uv: float
    alias_warning: bool
    resolved_space: str
    accepted_params: list
    rejected_params: list
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "shift": list(self.shift),
            "baseline_boundary_error": self.baseline_boundary_error,
            "object_boundary_error": self.object_boundary_error,
            "world_boundary_error": self.world_boundary_error,
            "object_motion_residual": self.object_motion_residual,
            "world_motion_residual": self.world_motion_residual,
            "second_scale_motion_residual":
                self.second_scale_motion_residual,
            "second_scale_uv": self.second_scale_uv,
            "alias_warning": self.alias_warning,
            "resolved_space": self.resolved_space,
            "accepted_params": self.accepted_params,
            "rejected_params": self.rejected_params,
            "detail": self.detail,
        }


def run_witness(graph_object: MaterialGraph, graph_world: MaterialGraph,
                target, shift=(0.35, 0.2), size=128, uv_scale=1.0,
                accept_threshold=0.15, second_uv_scale=None):
    """Run the full witness-decal experiment on two equivalent graphs."""
    sx, sy = shift

    base_obj = render_in_space(graph_object, size, uv_scale, "object")
    base_wld = render_in_space(graph_world, size, uv_scale, "world")
    moved_obj = render_in_space(graph_object, size, uv_scale, "object",
                                sx, sy)
    moved_wld = render_in_space(graph_world, size, uv_scale, "world",
                                sx, sy)

    base_border = 0.5 * (boundary_discontinuity(base_obj)
                         + boundary_discontinuity(base_wld))
    border_obj = boundary_discontinuity(moved_obj)
    border_wld = boundary_discontinuity(moved_wld)

    # Motion residual: how much the observed decal changes after the move.
    obj_res = float(np.abs(moved_obj - base_obj).mean())
    wld_res = float(np.abs(moved_wld - base_wld).mean())

    # A periodic pattern can make the world-space residual spuriously small.
    # Re-observe on a second decal scale whose period no longer aligns with
    # the translation, breaking the alias.
    if second_uv_scale is None:
        second_uv_scale = uv_scale * 1.63
    base2 = render_in_space(graph_world, size, second_uv_scale, "world")
    moved2 = render_in_space(graph_world, size, second_uv_scale, "world",
                             sx, sy)
    second_res = float(np.abs(moved2 - base2).mean())

    alias = wld_res < 0.25 * obj_res and second_res > 0.6 * obj_res
    effective_world = max(wld_res, second_res * 0.8) if alias else wld_res
    resolved = "object" if obj_res <= effective_world else "world"

    # Parameter-level acceptance: compare the two graphs and reject
    # parameters that differ strongly while their decals stay ambiguous.
    distances = _param_distance(graph_object, graph_world)
    accepted, rejected = [], []
    for name, d in distances.items():
        entry = {"param": name, "normalised_distance": round(d, 4)}
        (accepted if d <= accept_threshold else rejected).append(entry)

    detail = {
        "target_loss_object": multiscale_loss(
            graph_object, target, uv_scale=uv_scale)[0],
        "target_loss_world": multiscale_loss(
            graph_world, target, uv_scale=uv_scale)[0],
        "param_distances": {k: round(v, 4) for k, v in distances.items()},
    }
    return WitnessReport(
        shift=(float(sx), float(sy)),
        baseline_boundary_error=base_border,
        object_boundary_error=border_obj,
        world_boundary_error=border_wld,
        object_motion_residual=obj_res,
        world_motion_residual=wld_res,
        second_scale_motion_residual=second_res,
        second_scale_uv=float(second_uv_scale),
        alias_warning=bool(alias),
        resolved_space=resolved,
        accepted_params=accepted,
        rejected_params=rejected,
        detail=detail)
