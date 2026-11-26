"""Deterministic 2D noise primitives implemented with numpy only.

Everything is seeded by an integer seed so any recovered graph can be
regenerated exactly.  All primitives operate in an abstract texture
coordinate frame; the graph's output node decides whether those coordinates
are object space or world space.
"""
from __future__ import annotations

import numpy as np

I32 = np.int32
MAX_I32 = np.int32(0x7FFFFFFF)


def _hash2(ix, iy, seed):
    """Deterministic per-lattice-cell hash in [0, 1).

    int32 overflow wraps naturally, matching shader-style hash behaviour.
    """
    ix = ix.astype(np.int32)
    iy = iy.astype(np.int32)
    seed_term = np.int32((int(seed) * 1442695041 + 974711) % 2147483648)
    h = (ix * I32(374761393) + iy * I32(668265263) + seed_term)
    h = (h ^ (h >> I32(13))) * I32(1274126177)
    h = h ^ (h >> I32(16))
    return (h.astype(np.uint32).astype(np.float64) / 4294967296.0)


def _grad(ix, iy, seed):
    """Pseudo-random unit gradient vectors per lattice cell."""
    h = _hash2(ix, iy, seed)
    ang = h * (2.0 * np.pi)
    return np.cos(ang), np.sin(ang)


def _smooth(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a, b, t):
    return a + (b - a) * t


def gradient_noise(x, y, seed=0):
    """Perlin-style value-gradient lattice noise, output roughly in [-1, 1]."""
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    u = _smooth(fx)
    v = _smooth(fy)
    x1 = x0 + I32(1)
    y1 = y0 + I32(1)

    g00x, g00y = _grad(x0, y0, seed)
    g10x, g10y = _grad(x1, y0, seed)
    g01x, g01y = _grad(x0, y1, seed)
    g11x, g11y = _grad(x1, y1, seed)

    d00 = g00x * fx + g00y * fy
    d10 = g10x * (fx - 1.0) + g10y * fy
    d01 = g01x * fx + g01y * (fy - 1.0)
    d11 = g11x * (fx - 1.0) + g11y * (fy - 1.0)

    n0 = _lerp(d00, d10, u)
    n1 = _lerp(d01, d11, u)
    return _lerp(n0, n1, v) * 1.5


def value_noise(x, y, seed=0):
    """Smooth interpolated lattice value noise in [0, 1]."""
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    u = _smooth(fx)
    v = _smooth(fy)
    x1 = x0 + I32(1)
    y1 = y0 + I32(1)

    h00 = _hash2(x0, y0, seed)
    h10 = _hash2(x1, y0, seed)
    h01 = _hash2(x0, y1, seed)
    h11 = _hash2(x1, y1, seed)
    return _lerp(_lerp(h00, h10, u), _lerp(h01, h11, u), v)


def cellular_noise(x, y, seed=0):
    """Worley F1 distance normalised to roughly [0, 1]."""
    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    f1 = np.full_like(x, 2.0)
    for j in range(-1, 2):
        for i in range(-1, 2):
            cx = i + _hash2(ix + I32(i), iy + I32(j), seed)
            cy = j + _hash2(ix + I32(i), iy + I32(j), seed + 777)
            d2 = (fx - cx) ** 2 + (fy - cy) ** 2
            f1 = np.minimum(f1, d2)
    return np.clip(np.sqrt(f1) * 1.5, 0.0, 1.0)


def fbm(x, y, seed, kind="gradient", octaves=4, lacunarity=2.0, gain=0.5):
    """Fractal Brownian motion built from one of the base primitives.

    Returned values are rescaled to [0, 1] for uniform downstream handling.
    """
    if kind == "value":
        base = value_noise
        lo, hi = 0.0, 1.0
    elif kind == "cellular":
        base = cellular_noise
        lo, hi = 0.0, 1.0
    else:
        base = gradient_noise
        lo, hi = -1.0, 1.0

    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x)
    norm = 0.0
    for o in range(int(octaves)):
        total += amp * base(x * freq, y * freq, seed + o * 101)
        norm += amp
        amp *= gain
        freq *= lacunarity
    total /= norm
    return np.clip((total - lo) / (hi - lo), 0.0, 1.0)
