"""Rendering harness and multi-scale appearance loss.

The loss deliberately combines local pixel structure with global statistics
(histograms, gradient distribution, radial power spectrum) at several
resolutions.  Matching mean colour alone can never drive the loss to zero,
which prevents the optimiser from collapsing onto a flat average colour.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def coordinate_grid(size, uv_scale=1.0):
    """Square coordinate grid centred on the origin."""
    coords = (np.arange(size) + 0.5) / size * uv_scale - uv_scale / 2.0
    x, y = np.meshgrid(coords, coords)
    return x, y


def render_graph(graph, size=128, uv_scale=1.0, space_override=None):
    x, y = coordinate_grid(size, uv_scale)
    rgb = graph.evaluate(x, y, space_override=space_override)
    return np.clip(rgb, 0.0, 1.0)


def load_target(path, size=128):
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def encode_png(arr):
    arr8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    import io
    buf = io.BytesIO()
    Image.fromarray(arr8, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def save_png(arr, path):
    with open(path, "wb") as fh:
        fh.write(encode_png(arr))


def _downscale(arr, factor):
    if factor == 1:
        return arr
    h = arr.shape[0] // factor
    crop = arr[:h * factor, :h * factor]
    return crop.reshape(h, factor, h, factor, -1).mean(axis=(1, 3))


def _histograms(arr, bins=16):
    feats = []
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=bins, range=(0, 1),
                            density=True)
        feats.append(h / bins)  # integrates to ~1
    # luminance distribution captures tone mapping independent of hue
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    h, _ = np.histogram(lum, bins=bins, range=(0, 1), density=True)
    feats.append(h / bins)
    return np.concatenate(feats)


def _gradient_stats(arr):
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    gx = np.diff(lum, axis=1)
    gy = np.diff(lum, axis=0)
    mag = np.concatenate([np.abs(gx).ravel(), np.abs(gy).ravel()])
    qs = np.quantile(mag, [0.25, 0.5, 0.75, 0.95]) if mag.size else np.zeros(4)
    # All entries are clipped to [0, 1] so the component stays comparable
    # with the other L1 loss terms.
    return np.clip(np.concatenate([[mag.mean(), mag.std()], qs]), 0, 1)


def _radial_spectrum(arr):
    """Log-magnitude radial spectrum, standardised to a stable range."""
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    lum = lum - lum.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(lum))) ** 2
    h, w = lum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    n_bins = min(16, min(h, w) // 2)
    out = np.zeros(n_bins)
    for r in range(n_bins):
        mask = radius == r
        if mask.any():
            out[r] = spec[mask].mean()
    log_spec = np.log1p(out)
    norm = np.linalg.norm(log_spec) + 1e-12
    return log_spec / norm


def feature_vector(arr):
    return np.concatenate([
        _histograms(arr),
        _gradient_stats(arr),
        _radial_spectrum(arr),
    ])


# Coarse fitting relies on global statistics (histogram / spectrum /
# gradient distribution), which steer the optimiser toward the right
# structure even when pattern phase is misaligned.  Fine fitting then
# restores pixel-level weight for exact appearance recovery.
WEIGHT_PROFILES = {
    "coarse": {"pixel": 0.3, "histogram": 1.0, "gradient": 0.8,
               "spectrum": 1.2},
    "fine": {"pixel": 1.0, "histogram": 0.5, "gradient": 0.5,
             "spectrum": 0.8},
}
W_PIXEL = 1.0
W_HIST = 0.5
W_GRAD = 0.5
W_SPEC = 0.8


def loss_terms(render, target, profile="fine"):
    """Return (total_loss, component_dict, mean_colour_warning)."""
    w = WEIGHT_PROFILES[profile]
    pixel = np.abs(render - target).mean()
    hist = np.abs(_histograms(render) - _histograms(target)).mean()
    grad = np.abs(_gradient_stats(render) - _gradient_stats(target)).mean()
    # Spectrum vectors are L2-normalised, so compare them with L2 distance.
    spec = float(np.linalg.norm(
        _radial_spectrum(render) - _radial_spectrum(target)))
    total = (w["pixel"] * pixel + w["histogram"] * hist
             + w["gradient"] * grad + w["spectrum"] * spec)

    tgt_flat = target.reshape(-1, 3)
    tgt_std = tgt_flat.std(axis=0).mean()
    ren_flat = render.reshape(-1, 3)
    ren_std = ren_flat.std(axis=0).mean()
    mean_only = bool(tgt_std > 0.05 and ren_std < 0.35 * tgt_std)
    if mean_only:
        total += 0.15  # explicit penalty for collapsing onto average colour
    return total, {
        "pixel": float(pixel),
        "histogram": float(hist),
        "gradient": float(grad),
        "spectrum": float(spec),
        "mean_colour_collapse": mean_only,
    }


def multiscale_loss(graph, target, uv_scale=1.0, factors=(1, 2, 4),
                    space_override=None, profile="fine"):
    """Evaluate loss at several scales; coarser scales emphasise global stats."""
    base = render_graph(graph, target.shape[0], uv_scale, space_override)
    total = 0.0
    detail = {}
    mean_warn = False
    for f in factors:
        r = _downscale(base, f)
        t = _downscale(target, f)
        l, comp = loss_terms(r, t, profile=profile)
        weight = 1.0 / f
        total += weight * l
        detail[f"scale_{f}"] = comp
        mean_warn = mean_warn or comp["mean_colour_collapse"]
    return float(total), detail, base, mean_warn


def appearance_summary(arr):
    flat = arr.reshape(-1, 3)
    lum = arr @ np.array([0.2126, 0.7152, 0.0722])
    return {
        "mean_rgb": [float(v) for v in flat.mean(axis=0)],
        "std_rgb": [float(v) for v in flat.std(axis=0)],
        "luma_mean": float(lum.mean()),
        "luma_std": float(lum.std()),
    }
