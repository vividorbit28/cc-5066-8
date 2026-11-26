"""Procedural material graph: node registry, pull-based evaluator, JSON I/O.

A graph is a finite directed graph of typed nodes.  Evaluation is pull based:
a transform node replaces the sampling coordinates before requesting its
upstream node, which is what makes scale/rotation/offset observable in the
rendered texture.  A graph never references an external image file, so the
fitted result stays an editable procedural material rather than a baked
texture.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import noise as nmod


# ---------------------------------------------------------------------------
# Parameter descriptors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str                 # "continuous" | "discrete" | "seed"
    low: float = 0.0
    high: float = 1.0
    choices: tuple = ()       # for discrete
    default: Any = None

    def clamp(self, value):
        if self.kind == "continuous":
            return float(min(max(value, self.low), self.high))
        if self.kind == "discrete":
            # String-valued choices (noise kind, coordinate space) are kept
            # as strings; a numeric value is interpreted as a choice index so
            # vector-encoded and hand-written assignments both behave.
            if self.choices and not isinstance(
                    self.choices[0], (int, float)):
                if isinstance(value, str):
                    return value if value in self.choices else self.default
                idx = int(round(value))
                if 0 <= idx < len(self.choices):
                    return self.choices[idx]
                return self.default
            return int(round(value))
        if self.kind == "seed":
            return int(round(value))
        return value


@dataclass
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)  # upstream node ids

    def to_dict(self):
        return {"id": self.id, "type": self.type,
                "params": dict(self.params), "inputs": list(self.inputs)}

    @staticmethod
    def from_dict(d):
        return Node(d["id"], d["type"], dict(d.get("params", {})),
                    list(d.get("inputs", [])))


# ---------------------------------------------------------------------------
# Node registry
# ---------------------------------------------------------------------------
def _spec_rotation():
    return ParamSpec("rotation", "continuous", -math.pi, math.pi, default=0.0)


REGISTRY = {
    "noise": [
        ParamSpec("kind", "discrete", choices=("gradient", "value", "cellular"),
                  default="gradient"),
        ParamSpec("seed", "seed", low=0, high=9999, default=1),
        ParamSpec("scale", "continuous", 1.0, 12.0, default=4.0),
        ParamSpec("octaves", "discrete", choices=tuple(range(1, 7)), default=4),
        ParamSpec("lacunarity", "continuous", 1.5, 3.0, default=2.0),
        ParamSpec("gain", "continuous", 0.25, 0.75, default=0.5),
    ],
    "transform": [
        ParamSpec("scale_x", "continuous", 0.25, 4.0, default=1.0),
        ParamSpec("scale_y", "continuous", 0.25, 4.0, default=1.0),
        ParamSpec("rotation", "continuous", -math.pi, math.pi, default=0.0),
        ParamSpec("offset_x", "continuous", -2.0, 2.0, default=0.0),
        ParamSpec("offset_y", "continuous", -2.0, 2.0, default=0.0),
    ],
    "threshold": [
        ParamSpec("low", "continuous", 0.0, 1.0, default=0.35),
        ParamSpec("high", "continuous", 0.0, 1.0, default=0.65),
        ParamSpec("softness", "continuous", 0.005, 0.25, default=0.05),
    ],
    "colormap": [
        ParamSpec("stop0_r", "continuous", 0.0, 1.0, default=0.08),
        ParamSpec("stop0_g", "continuous", 0.0, 1.0, default=0.10),
        ParamSpec("stop0_b", "continuous", 0.0, 1.0, default=0.16),
        ParamSpec("stop1_r", "continuous", 0.0, 1.0, default=0.75),
        ParamSpec("stop1_g", "continuous", 0.0, 1.0, default=0.55),
        ParamSpec("stop1_b", "continuous", 0.0, 1.0, default=0.30),
        ParamSpec("stop2_r", "continuous", 0.0, 1.0, default=0.95),
        ParamSpec("stop2_g", "continuous", 0.0, 1.0, default=0.93),
        ParamSpec("stop2_b", "continuous", 0.0, 1.0, default=0.85),
    ],
    "blend": [
        ParamSpec("mode", "discrete", choices=("mix", "multiply", "add"),
                  default="mix"),
        ParamSpec("factor", "continuous", 0.0, 1.0, default=0.5),
    ],
    "output": [
        ParamSpec("space", "discrete", choices=("object", "world"),
                  default="object"),
        ParamSpec("roughness", "continuous", 0.1, 0.95, default=0.6),
    ],
}

NODE_LABELS = {
    "noise": "Noise",
    "transform": "Transform",
    "threshold": "Threshold",
    "colormap": "Color Map",
    "blend": "Blend",
    "output": "Output",
}


def spec_index(graph):
    """Return (node_id, ParamSpec) pairs for every searchable parameter."""
    out = []
    for node in graph.nodes.values():
        for spec in REGISTRY[node.type]:
            out.append((node.id, spec))
    return out


# ---------------------------------------------------------------------------
# Graph container
# ---------------------------------------------------------------------------
class MaterialGraph:
    def __init__(self, nodes=None, output_id=None):
        self.nodes = {n.id: n for n in (nodes or [])}
        self.output_id = output_id

    # -- construction / I/O -------------------------------------------------
    def add(self, node_type, node_id, params=None, inputs=()):
        spec_names = {s.name for s in REGISTRY[node_type]}
        defaults = {s.name: s.default for s in REGISTRY[node_type]}
        defaults.update(params or {})
        unknown = set(defaults) - spec_names
        if unknown:
            raise ValueError(f"unknown params for {node_type}: {sorted(unknown)}")
        node = Node(node_id, node_type, defaults, list(inputs))
        self.nodes[node_id] = node
        if node_type == "output":
            self.output_id = node_id
        return node

    def to_dict(self):
        return {
            "output_id": self.output_id,
            "nodes": [self.nodes[k].to_dict()
                      for k in self.topo_order()],
        }

    @staticmethod
    def from_dict(d):
        g = MaterialGraph()
        for nd in d["nodes"]:
            g.nodes[nd["id"]] = Node.from_dict(nd)
        g.output_id = d["output_id"]
        g.validate()
        return g

    def clone(self):
        return MaterialGraph.from_dict(self.to_dict())

    # -- structure ----------------------------------------------------------
    def topo_order(self):
        order, visited = [], set()

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for parent in self.nodes[nid].inputs:
                visit(parent)
            order.append(nid)

        roots = [self.output_id] if self.output_id else list(self.nodes)
        for r in roots:
            visit(r)
        return order

    def validate(self):
        if not self.nodes:
            raise ValueError("empty graph")
        if self.output_id not in self.nodes:
            raise ValueError("output node missing")
        if self.nodes[self.output_id].type != "output":
            raise ValueError("output_id does not point at an output node")
        seen = set()

        def walk(nid, stack):
            if nid in stack:
                raise ValueError(f"cycle detected at {nid}")
            if nid in seen:
                return
            seen.add(nid)
            node = self.nodes[nid]
            if node.type not in REGISTRY:
                raise ValueError(f"unknown node type {node.type}")
            arity = _INPUT_ARITY[node.type]
            if len(node.inputs) != arity:
                raise ValueError(
                    f"{node.type} node {nid} needs {arity} inputs, "
                    f"got {len(node.inputs)}")
            for parent in node.inputs:
                if parent not in self.nodes:
                    raise ValueError(f"node {nid} references missing {parent}")
                walk(parent, stack | {nid})

        walk(self.output_id, set())
        for nid, node in self.nodes.items():
            if nid not in seen:
                raise ValueError(f"node {nid} is not connected to output")
        return True

    # -- parameter access ---------------------------------------------------
    def get_param(self, node_id, name):
        return self.nodes[node_id].params[name]

    def set_param(self, node_id, name, value):
        spec = next(s for s in REGISTRY[self.nodes[node_id].type]
                    if s.name == name)
        self.nodes[node_id].params[name] = spec.clamp(value)

    def param_vector(self):
        """Flat numeric vector in spec_index order.

        Discrete choice parameters are encoded by their choice index so the
        whole vector stays numeric for the optimiser.
        """
        values = []
        for nid, spec in spec_index(self):
            raw = self.nodes[nid].params[spec.name]
            if spec.kind == "discrete":
                values.append(float(spec.choices.index(raw)))
            else:
                values.append(float(raw))
        return np.array(values, dtype=float)

    def set_param_vector(self, vector):
        for (nid, spec), value in zip(spec_index(self), vector):
            self.set_param(nid, spec.name, value)

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, x, y, space_override=None):
        """Evaluate output for coordinate grids; returns HxWx3 float array."""
        evaluator = _Evaluator(self)
        return evaluator.run(self.output_id, x, y, space_override)


_INPUT_ARITY = {
    "noise": 0,
    "transform": 1,
    "threshold": 1,
    "colormap": 1,
    "blend": 2,
    "output": 1,
}


def _as_rgb(a):
    if a.ndim == 3:
        return a
    return np.broadcast_to(a[..., None], a.shape + (3,)).copy()


def _smoothstep(edge0, edge1, v):
    if edge1 <= edge0:
        return np.where(v >= edge1, 1.0, 0.0)
    t = np.clip((v - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _Evaluator:
    def __init__(self, graph):
        self.graph = graph
        self._memo = {}

    def run(self, root_id, x, y, space_override):
        rgb = _as_rgb(self._eval(root_id, x, y))
        out_node = self.graph.nodes[root_id]
        # Space is metadata consumed by the rendering harness.
        out_node.params["_resolved_space"] = (
            space_override or out_node.params["space"])
        return np.clip(rgb, 0.0, 1.0)

    def _eval(self, nid, x, y):
        key = (nid, id(x), id(y))
        if key in self._memo:
            return self._memo[key]
        node = self.graph.nodes[nid]
        p = node.params
        t = node.type
        result = None

        if t == "noise":
            s = p["scale"]
            result = nmod.fbm(x * s, y * s, p["seed"], kind=p["kind"],
                              octaves=p["octaves"],
                              lacunarity=p["lacunarity"], gain=p["gain"])
        elif t == "transform":
            c, s = math.cos(p["rotation"]), math.sin(p["rotation"])
            x1 = x / max(p["scale_x"], 1e-4)
            y1 = y / max(p["scale_y"], 1e-4)
            xr = c * x1 - s * y1 + p["offset_x"]
            yr = s * x1 + c * y1 + p["offset_y"]
            result = self._eval(node.inputs[0], xr, yr)
        elif t == "threshold":
            src = _as_rgb(self._eval(node.inputs[0], x, y))
            soft = max(p["softness"], 1e-4)
            low = min(p["low"], p["high"] - soft)
            high = max(p["high"], low + soft)
            mask = (_smoothstep(low - soft, low + soft, src)
                    * (1.0 - _smoothstep(high - soft, high + soft, src)))
            result = src * mask
        elif t == "colormap":
            src = _as_rgb(self._eval(node.inputs[0], x, y))[..., 0]
            t0 = np.clip(src * 2.0, 0.0, 1.0)[..., None]
            t1 = np.clip(src * 2.0 - 1.0, 0.0, 1.0)[..., None]
            c0 = np.array([p["stop0_r"], p["stop0_g"], p["stop0_b"]])
            c1 = np.array([p["stop1_r"], p["stop1_g"], p["stop1_b"]])
            c2 = np.array([p["stop2_r"], p["stop2_g"], p["stop2_b"]])
            lower = c0 + (c1 - c0) * t0
            upper = c1 + (c2 - c1) * t1
            result = np.where((src[..., None] < 0.5), lower, upper)
        elif t == "blend":
            a = _as_rgb(self._eval(node.inputs[0], x, y))
            b = _as_rgb(self._eval(node.inputs[1], x, y))
            f = p["factor"]
            if p["mode"] == "mix":
                result = a * (1.0 - f) + b * f
            elif p["mode"] == "multiply":
                result = a * (1.0 - f) + (a * b) * f
            else:  # add
                result = np.clip(a * (1.0 - f)
                                 + np.clip(a + b, 0, 1) * f, 0.0, 1.0)
        elif t == "output":
            result = self._eval(node.inputs[0], x, y)
        else:
            raise ValueError(f"unsupported node type {t}")

        self._memo[key] = result
        return result
