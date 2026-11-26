"""Canonical graph templates and synthetic target generators.

Targets come in two flavours:

* ``procedural_target`` renders a graph whose true parameters are known,
  so parameter recovery can be scored against ground truth.
* ``inexpressible_target`` builds patterns the node vocabulary cannot
  represent (hard checkerboard, fine grid, blobs with sharp edges).  The
  optimiser is expected to leave a structured residual rather than fit them.
"""
from __future__ import annotations

import numpy as np

from .graph import MaterialGraph
from .render import render_graph


def build_chain(seed=1, space="object", threshold=True, **overrides):
    """Noise -> Transform -> [Threshold] -> ColorMap -> Output chain."""
    g = MaterialGraph()
    g.add("noise", "noise", {"seed": seed})
    g.add("transform", "transform", inputs=("noise",))
    last = "transform"
    if threshold:
        g.add("threshold", "threshold", inputs=("transform",))
        last = "threshold"
    g.add("colormap", "colormap", inputs=(last,))
    g.add("output", "output", {"space": space}, inputs=("colormap",))
    for key, val in overrides.items():
        node_id, param = key.split(".", 1)
        g.set_param(node_id, param, val)
    g.validate()
    return g


def build_blend_graph(seed_a=3, seed_b=11, space="object"):
    """Two transformed noise sources combined through a blend node."""
    g = MaterialGraph()
    g.add("noise", "noise_a", {"seed": seed_a})
    g.add("transform", "xform_a", inputs=("noise_a",))
    g.add("noise", "noise_b", {"seed": seed_b, "kind": "cellular"})
    g.add("transform", "xform_b", inputs=("noise_b",))
    g.add("blend", "blend", inputs=("xform_a", "xform_b"))
    g.add("colormap", "colormap", inputs=("blend",))
    g.add("output", "output", {"space": space}, inputs=("colormap",))
    g.validate()
    return g


def procedural_target(params=None, size=128, seed=7, space="object",
                      threshold=True):
    """Render a target with known ground-truth parameters.

    Returns (image, graph) so tests can compare against the true values.
    """
    g = build_chain(seed=seed, space=space, threshold=threshold)
    if params:
        for key, val in params.items():
            node_id, param = key.split(".", 1)
            g.set_param(node_id, param, val)
    return render_graph(g, size=size), g


# ---------------------------------------------------------------------------
# Inexpressible patterns (no node in the vocabulary can generate these)
# ---------------------------------------------------------------------------
def _checkerboard(size, cells=8):
    idx = np.arange(size) * cells // size
    grid = (idx[:, None] + idx[None, :]) % 2
    base = np.empty((size, size, 3))
    base[grid == 0] = [0.12, 0.14, 0.22]
    base[grid == 1] = [0.92, 0.85, 0.62]
    return base


def _grid_lines(size, cells=10):
    img = np.full((size, size, 3), [0.85, 0.86, 0.82])
    step = size / cells
    for k in range(cells + 1):
        c = int(k * step)
        if 0 <= c < size:
            img[c, :] = [0.1, 0.1, 0.15]
            img[:, c] = [0.1, 0.1, 0.15]
    return img


def _sharp_blobs(size, n_blobs=12, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.full((size, size, 3), [0.9, 0.88, 0.8])
    palette = np.array([[0.8, 0.2, 0.2], [0.2, 0.5, 0.8],
                        [0.25, 0.65, 0.3], [0.7, 0.5, 0.15]])
    for _ in range(n_blobs):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(size // 12, size // 6)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        # Hard edge: smooth noise cannot reproduce the discontinuity.
        img[mask] = palette[rng.integers(0, len(palette))]
    return img


INEXPRESSIBLE_KINDS = {
    "checkerboard": _checkerboard,
    "grid": _grid_lines,
    "blobs": _sharp_blobs,
}


def inexpressible_target(kind="checkerboard", size=128):
    return np.clip(INEXPRESSIBLE_KINDS[kind](size), 0.0, 1.0)
