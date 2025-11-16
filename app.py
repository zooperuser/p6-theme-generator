import os
import re
import numpy as np
import gradio as gr
from PIL import Image
from datetime import datetime, UTC

class Config:
    """Minimal configuration for the app."""
    # Reserved for future settings if needed
    pass

# --- Workaround for h11 LocalProtocolError with brotli compression ---
# Some environments with certain clients trigger a Content-Length mismatch when
# Gradio's brotli middleware compresses responses. We strip 'br' from
# Accept-Encoding so the brotli middleware won't engage.
class StripBrotliMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        try:
            if scope.get("type") == "http":
                headers = scope.get("headers") or []
                new_headers = []
                for k, v in headers:
                    try:
                        if k.lower() == b"accept-encoding":
                            # Remove 'br' from Accept-Encoding
                            val = v.decode("latin-1")
                            encs = [e.strip() for e in val.split(",") if e.strip()]
                            encs = [e for e in encs if e.lower() != "br"]
                            if encs:
                                new_val = ", ".join(encs).encode("latin-1")
                                new_headers.append((k, new_val))
                            # else: drop header entirely
                        else:
                            new_headers.append((k, v))
                    except Exception:
                        # If anything goes wrong decoding, drop the header to be safe
                        if k.lower() != b"accept-encoding":
                            new_headers.append((k, v))
                # Rebuild scope with adjusted headers
                scope = dict(scope)
                scope["headers"] = new_headers
        except Exception:
            # Fallback: do nothing if unexpected structure
            pass
        return await self.app(scope, receive, send)

class ImagePaletteGenerator:
    def _format_chaotic_palette(
        self,
        hex_colors: list[str],
        title: str = "Chaotic Color Spiral",
        rng: np.random.Generator | None = None,
    ) -> tuple[str, list[str]]:
        # Generate a chaotic-but-balanced ordering so the bar keeps surprising the eye
        n = len(hex_colors)
        if n == 0:
            return "", []

        if rng is None:
            rng = np.random.default_rng()

        # Convert to LAB for perceptual distance calculations; fall back if parsing fails
        rgb_rows: list[list[int]] = []
        for hx in hex_colors:
            h = hx.lstrip("#")
            if len(h) != 6:
                rgb_rows = []
                break
            try:
                rgb_rows.append([int(h[i : i + 2], 16) for i in (0, 2, 4)])
            except Exception:
                rgb_rows = []
                break

        if len(rgb_rows) != n:
            # Safety fallback to a zig-zag if parsing fails
            indices = []
            left, right = 0, n - 1
            flip = True
            while left <= right:
                if flip:
                    indices.append(left)
                    left += 1
                else:
                    indices.append(right)
                    right -= 1
                flip = not flip
            ordered = [hex_colors[i] for i in indices]
            html = self._format_custom_palette(ordered, title=title)
            return html, ordered

        lab = self._rgb_to_lab_np(np.array(rgb_rows, dtype=np.float32))
        centres = lab.mean(axis=0)
        chroma = np.linalg.norm(lab[:, 1:], axis=1)
        start_base = chroma + np.linalg.norm(lab - centres, axis=1) * 0.4
        start_base = np.maximum(start_base, 0.0)
        start_probs = start_base + 1e-6
        start_probs = start_probs / start_probs.sum()
        top_k = min(6, n)
        candidate_indices = np.argsort(-start_base)[:top_k]
        candidate_weights = start_probs[candidate_indices]
        candidate_weights = candidate_weights / candidate_weights.sum()
        start_idx = int(rng.choice(candidate_indices, p=candidate_weights))

        positions: list[int | None] = [None] * n
        available = list(range(n))
        left, right = 0, n - 1

        available.remove(start_idx)
        positions[left] = start_idx
        left += 1
        last_vec = lab[start_idx]

        if left <= right and available:
            partner_candidates = lab[available] - lab[start_idx]
            partner_dist = np.linalg.norm(partner_candidates, axis=1)
            partner_dist = partner_dist ** (1.0 + rng.random() * 0.8)
            if rng.random() < 0.35:
                partner_dist = 1.0 / (partner_dist + 1e-6)
            partner_dist += rng.random(len(partner_dist)) * 0.06
            partner_dist = np.maximum(partner_dist, 0.0)
            partner_probs = partner_dist / partner_dist.sum()
            partner_choice = int(rng.choice(len(available), p=partner_probs))
            partner_idx = available.pop(partner_choice)
            positions[right] = partner_idx
            right -= 1

        while available and left <= right:
            if left == right:
                # Center slot: pick the tone closest to the running average for a soft anchor
                center_vec = lab[list(filter(lambda v: v is not None, positions))].mean(axis=0) if any(p is not None for p in positions) else centres
                offsets = np.linalg.norm(lab[available] - center_vec, axis=1)
                offsets = offsets + rng.random(len(offsets)) * 0.08
                offsets = np.maximum(offsets, 0.0)
                inv = 1.0 / (offsets + 1e-6)
                inv = inv / inv.sum()
                center_choice = int(rng.choice(len(available), p=inv))
                positions[left] = available.pop(center_choice)
                break

            candidate_lab = lab[available]
            distance_last = np.linalg.norm(candidate_lab - last_vec, axis=1)
            distance_mean = np.linalg.norm(candidate_lab - centres, axis=1)
            composite = 0.6 * distance_last + 0.4 * distance_mean
            composite = (composite + rng.random(len(composite)) * 0.12) ** (1.0 + rng.random() * 0.6)
            composite = np.maximum(composite, 0.0)
            composite = composite / composite.sum()
            next_choice = int(rng.choice(len(available), p=composite))
            next_idx = available.pop(next_choice)
            positions[left] = next_idx
            left += 1
            last_vec = lab[next_idx]

            if not available or left > right:
                break

            partner_lab = lab[available] - lab[next_idx]
            partner_dist = np.linalg.norm(partner_lab, axis=1)
            partner_dist = partner_dist ** (1.0 + rng.random() * 0.9)
            if rng.random() < 0.45:
                partner_dist = 1.0 / (partner_dist + 1e-6)
            partner_dist += rng.random(len(partner_dist)) * 0.05
            partner_dist = np.maximum(partner_dist, 0.0)
            partner_probs = partner_dist / partner_dist.sum()
            partner_choice = int(rng.choice(len(available), p=partner_probs))
            partner_idx = available.pop(partner_choice)
            positions[right] = partner_idx
            right -= 1

        # Fill any leftover slots in the unlikely event something remained unassigned
        for idx in range(n):
            if positions[idx] is None and available:
                positions[idx] = available.pop(0)

        ordered = [hex_colors[i] for i in positions if i is not None]
        html = self._format_custom_palette(ordered, title=title)
        return html, ordered
    def __init__(self, config: Config):
        self.config = config

    def _get_text_color_for_bg(self, hex_color: str) -> str:
        hex_color = hex_color.lstrip('#')
        try:
            r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            luminance = (0.299 * r + 0.587 * g + 0.114 * b)
            return '#000000' if luminance > 150 else '#FFFFFF'
        except Exception:
            return '#000000'

    # --- Custom palette formatting (for image-based random pixels) ---
    def _format_custom_palette(self, hex_colors: list[str], title: str = "Random Image Pixels") -> str:
        if not hex_colors:
            return "<div class='palette-empty'>No colors found from the image.</div>"

        segments = []
        for hx in hex_colors:
            hx_norm = (hx or "").strip()
            if not hx_norm.startswith("#") and len(hx_norm) == 6:
                hx_norm = "#" + hx_norm
            text_color = self._get_text_color_for_bg(hx_norm)
            segments.append(
                f"<div class='palette-segment' style='background:{hx_norm};color:{text_color};' "
                f"title='{hx_norm}'><span>{hx_norm}</span></div>"
            )

        title_html = f"<div class='palette-title'>{title}</div>" if title else ""
        return (
            "<div class='palette-progress-wrapper'>"
            + title_html
            + "<div class='palette-progress-bar'>"
            + "".join(segments)
            + "</div>"
            + "</div>"
        )

    # -------- Advanced image palette generation (perceptual & spatial weighting) --------
    def _resize_preserve(self, img: Image.Image, max_side: int = 512) -> Image.Image:
        try:
            w, h = img.size
            s = max(w, h)
            if s <= max_side:
                return img
            scale = max_side / float(s)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", getattr(Image, "BILINEAR", 2)))
            return img.resize(new_size, resample)
        except Exception:
            return img

    def _rgb_to_hsv_np(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # arr: (N,3) in [0,255]
        a = arr.astype(np.float32) / 255.0
        r, g, b = a[:, 0], a[:, 1], a[:, 2]
        cmax = np.max(a, axis=1)
        cmin = np.min(a, axis=1)
        delta = cmax - cmin + 1e-8
        # Hue
        h = np.zeros_like(cmax)
        mask = delta > 1e-7
        # Cases
        rc = ((g - b) / delta) % 6
        gc = ((b - r) / delta) + 2
        bc = ((r - g) / delta) + 4
        choice = np.argmax(a == cmax[:, None], axis=1)
        # 0=r,1=g,2=b
        h[mask & (choice == 0)] = rc[mask & (choice == 0)]
        h[mask & (choice == 1)] = gc[mask & (choice == 1)]
        h[mask & (choice == 2)] = bc[mask & (choice == 2)]
        h = (h / 6.0) % 1.0
        s = np.where(cmax <= 1e-7, 0.0, delta / (cmax + 1e-8))
        v = cmax
        return h, s, v

    def _rgb_to_lab_np(self, arr: np.ndarray) -> np.ndarray:
        # arr: (N,3) in [0,255] sRGB to LAB (D65)
        a = arr.astype(np.float32) / 255.0
        # sRGB -> linear
        def inv_gamma(u):
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
        lin = inv_gamma(a)
        # linear RGB -> XYZ (D65)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32)
        xyz = lin @ M.T
        # Normalize by white point D65
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        x = xyz[:, 0] / Xn
        y = xyz[:, 1] / Yn
        z = xyz[:, 2] / Zn
        eps = 216/24389
        kappa = 24389/27
        def f(t):
            return np.where(t > eps, np.cbrt(t), (kappa * t + 16) / 116)
        fx, fy, fz = f(x), f(y), f(z)
        L = 116 * fy - 16
        a_ = 500 * (fx - fy)
        b_ = 200 * (fy - fz)
        return np.stack([L, a_, b_], axis=1).astype(np.float32)

    def _sobel_magnitude(self, gray: np.ndarray) -> np.ndarray:
        # gray: (H,W) float32 0..1
        H, W = gray.shape
        pad = np.pad(gray, 1, mode='edge')
        # Sobel kernels
        gx = (
            -1*pad[0:H,   0:W] + 0*pad[0:H,   1:W+1] + 1*pad[0:H,   2:W+2] +
            -2*pad[1:H+1, 0:W] + 0*pad[1:H+1, 1:W+1] + 2*pad[1:H+1, 2:W+2] +
            -1*pad[2:H+2, 0:W] + 0*pad[2:H+2, 1:W+1] + 1*pad[2:H+2, 2:W+2]
        )
        gy = (
             1*pad[0:H,   0:W] + 2*pad[0:H,   1:W+1] + 1*pad[0:H,   2:W+2] +
             0*pad[1:H+1, 0:W] + 0*pad[1:H+1, 1:W+1] + 0*pad[1:H+1, 2:W+2] +
            -1*pad[2:H+2, 0:W] + -2*pad[2:H+2, 1:W+1] + -1*pad[2:H+2, 2:W+2]
        )
        mag = np.sqrt(gx * gx + gy * gy)
        # Normalize
        mmin, mmax = float(mag.min()), float(mag.max())
        if mmax > mmin:
            mag = (mag - mmin) / (mmax - mmin)
        else:
            mag = np.zeros_like(mag)
        return mag.astype(np.float32)

    def _compute_sampling_weights(self, rgb: np.ndarray, H: int, W: int) -> np.ndarray:
        # rgb flat: (N,3) 0..255
        h, s, v = self._rgb_to_hsv_np(rgb)
        # luminance for edges
        lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).astype(np.float32) / 255.0
        gray = lum.reshape(H, W)
        edges = self._sobel_magnitude(gray).reshape(-1)
        # center bias
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dist2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        dist2 = dist2 / (dist2.max() + 1e-8)
        center = np.exp(-2.5 * dist2).reshape(-1)  # sharper center
        # saturation emphasis and value contrast
        sat = s.astype(np.float32)
        val = v.astype(np.float32)
        # combine weights
        w = 0.40 * edges + 0.25 * center + 0.25 * sat + 0.10 * (val * (1 - val))  # mid-values get some weight
        w = np.clip(w, 0, None)
        # sanitize and normalize
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            w = np.ones_like(w, dtype=np.float64) / w.size
        else:
            w = (w.astype(np.float64) / total)
            # ensure exact sum 1.0 to satisfy RNG choice checks
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                w = np.ones_like(w, dtype=np.float64) / w.size
            else:
                # adjust last element by residual
                residual = 1.0 - s
                w[-1] = max(0.0, w[-1] + residual)
                # renormalize lightly in case of clamp
                s2 = w.sum()
                if s2 > 0:
                    w /= s2
                else:
                    w = np.ones_like(w, dtype=np.float64) / w.size
        return w

    def _kmeans_lab(self, lab: np.ndarray, k: int, weights: np.ndarray | None = None, iters: int = 12, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        # lab: (N,3), return (centers(k,3), labels(N))
        N = lab.shape[0]
        if rng is None:
            rng = np.random.default_rng()
        k = max(1, min(k, max(1, N)))
        # k-means++ init with (optional) weights
        centers = np.empty((k, 3), dtype=np.float32)
        # choose first center (ensure probabilities sum to 1)
        if weights is not None:
            w0 = np.array(weights, dtype=np.float64).reshape(-1)
            if w0.shape[0] != N:
                w0 = None
            else:
                # Sanitize, clip negatives, normalize to sum==1
                w0 = np.nan_to_num(w0, nan=0.0, posinf=0.0, neginf=0.0)
                w0 = np.clip(w0, 0.0, None)
                s0 = float(w0.sum())
                if s0 > 0 and np.isfinite(s0):
                    w0 = w0 / s0
                    # adjust last by residual to hit exactly 1.0
                    resid = 1.0 - float(w0.sum())
                    w0[-1] = max(0.0, w0[-1] + resid)
                    s1 = float(w0.sum())
                    if s1 > 0:
                        w0 /= s1
                else:
                    w0 = None
            if w0 is not None:
                try:
                    idx0 = rng.choice(N, p=w0)
                except Exception:
                    idx0 = rng.choice(N)
            else:
                idx0 = rng.choice(N)
        else:
            idx0 = rng.integers(0, N)
        centers[0] = lab[idx0]
        # select remaining
        dist2 = np.full(N, np.inf, dtype=np.float64)
        for c in range(1, k):
            # update dist2 to nearest center
            d = np.linalg.norm(lab - centers[c-1], axis=1)
            dist2 = np.minimum(dist2, d.astype(np.float64) ** 2)
            probs = dist2 * (weights if weights is not None else 1.0)
            s = float(probs.sum())
            if s <= 0 or not np.isfinite(s):
                idx = rng.integers(0, N)
            else:
                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0) / s
                try:
                    idx = rng.choice(N, p=probs)
                except Exception as e:
                    print(f"[WARN] rng.choice failed for kmeans++ (p=probs): {e}; falling back to uniform.")
                    idx = rng.choice(N)
            centers[c] = lab[idx]
        labels = np.zeros(N, dtype=np.int32)
        for _ in range(iters):
            # assign
            # compute distance to centers
            dists = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # update centers (weighted)
            for c in range(k):
                mask = labels == c
                if not np.any(mask):
                    # re-seed empty cluster
                    centers[c] = lab[rng.integers(0, N)]
                    continue
                if weights is not None:
                    w = weights[mask].astype(np.float64)
                    s = float(w.sum())
                    if s <= 0:
                        centers[c] = lab[mask].mean(axis=0)
                    else:
                        centers[c] = (lab[mask] * w[:, None]).sum(axis=0) / s
                else:
                    centers[c] = lab[mask].mean(axis=0)
        return centers, labels

    def generate_random_palette_from_image(self, image: Image.Image | None, n_colors: int = 15, seed: int | None = None) -> tuple[str, str, str, str, str, str]:
        """Generate an image palette using a perceptually-aware, weighted, clustered pipeline.

        Steps:
        - Preprocess: RGB conversion, resize (<=512px max side)
        - Spatial/Perceptual Weights: edges (Sobel), center bias, saturation, mid-value contrast
        - Sampling: probabilistic sampling by weights (grid-free)
        - Clustering: k-means in CIE LAB (k = n_colors), weighted by sampling probabilities
        - Output: Hex colors of clusters sorted by total cluster weight
        """
        try:
            n_colors = int(max(1, min(60, n_colors)))
        except Exception:
            n_colors = 15
        if image is None:
            return "<div class='palette-empty'>Please upload an image to sample colors.</div>", "", "", "", "", "", "", ""

        try:
            # Preprocess
            img = image.convert("RGB")
            img = self._resize_preserve(img, 512)
            arr = np.array(img)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return "<div class='palette-empty'>Unsupported image format for sampling.</div>", "", "", "", "", "", "", ""
            H, W, _ = arr.shape
            flat_rgb = arr.reshape(-1, 3)

            # Weights
            weights = self._compute_sampling_weights(flat_rgb, H, W)
            # final guard: exact normalization
            if weights.ndim != 1 or weights.shape[0] != flat_rgb.shape[0]:
                weights = np.ones((flat_rgb.shape[0],), dtype=np.float64) / flat_rgb.shape[0]
            else:
                s = weights.sum()
                if not np.isfinite(s) or s <= 0:
                    weights = np.ones((flat_rgb.shape[0],), dtype=np.float64) / flat_rgb.shape[0]
                else:
                    weights = weights / s

            # RNG
            if seed is None:
                try:
                    seed = int(datetime.now(UTC).timestamp() * 1e9) & 0xFFFFFFFF
                except Exception:
                    seed = None
            rng = np.random.default_rng(seed)

            # Sample points
            total = flat_rgb.shape[0]
            sample_n = int(min(max(n_colors * 400, 2000), 15000))
            replace = sample_n > total
            try:
                idx = rng.choice(total, size=sample_n, replace=replace, p=weights)
            except Exception as e:
                print(f"[WARN] rng.choice failed for random pixel sampling (p=weights): {e}; falling back to uniform.")
                idx = rng.choice(total, size=sample_n, replace=replace)
            samp_rgb = flat_rgb[idx]
            samp_w = weights[idx]
            samp_lab = self._rgb_to_lab_np(samp_rgb)

            # Cluster
            centers_lab, labels = self._kmeans_lab(samp_lab, n_colors, weights=samp_w, rng=rng)

            # For output hex, compute centroid in RGB space for each cluster (weighted)
            hex_colors: list[str] = []
            cluster_strengths = []
            for c in range(centers_lab.shape[0]):
                mask = labels == c
                if not np.any(mask):
                    continue
                w = samp_w[mask]
                rgb_pts = samp_rgb[mask].astype(np.float32)
                s = float(w.sum())
                if s <= 0:
                    mean_rgb = rgb_pts.mean(axis=0)
                    strength = rgb_pts.shape[0]
                else:
                    mean_rgb = (rgb_pts * w[:, None]).sum(axis=0) / s
                    strength = s
                r, g, b = [int(np.clip(v, 0, 255)) for v in mean_rgb]
                hex_colors.append(f"#{r:02X}{g:02X}{b:02X}")
                cluster_strengths.append(float(strength))

            # Sort by strength and ensure uniqueness
            order = np.argsort(-np.array(cluster_strengths)) if cluster_strengths else []
            sorted_hex = []
            seen = set()
            for i in order:
                hx = hex_colors[i]
                if hx not in seen:
                    sorted_hex.append(hx)
                    seen.add(hx)
                if len(sorted_hex) >= n_colors:
                    break

            # Fallback: if fewer than requested, pad by distinct frequent pixels
            if len(sorted_hex) < n_colors:
                # simple popularity padding from weighted sample
                packed = (samp_rgb[:, 0].astype(np.uint32) << 16) | (samp_rgb[:, 1].astype(np.uint32) << 8) | samp_rgb[:, 2].astype(np.uint32)
                # weight-aware counts: approximate by rounding weights
                order2 = np.argsort(-samp_w)
                for j in order2:
                    v = int(packed[j])
                    hx = f"#{(v >> 16) & 0xFF:02X}{(v >> 8) & 0xFF:02X}{v & 0xFF:02X}"
                    if hx not in seen:
                        sorted_hex.append(hx)
                        seen.add(hx)
                        if len(sorted_hex) >= n_colors:
                            break

            take = sorted_hex[:n_colors]
            # Suppress titles to maximize vertical space so all 15 cards fit
            html = self._format_custom_palette(take, title="Color Progression")
            chaotic_html, chaotic_order = self._format_chaotic_palette(take, title="AWESOME PALETTE", rng=rng)
            hex_csv = ", ".join(take)
            hex_csv_reversed = ", ".join(reversed(take))
            fonts = [self._get_text_color_for_bg(hx) for hx in take]
            font_csv = ", ".join(fonts)
            font_csv_reversed = ", ".join(reversed(fonts))
            chaotic_hex_csv = ", ".join(chaotic_order)
            chaotic_font_csv = ", ".join([self._get_text_color_for_bg(hx) for hx in chaotic_order])
            return html, hex_csv, hex_csv_reversed, font_csv, font_csv_reversed, chaotic_html, chaotic_hex_csv, chaotic_font_csv
        except Exception as e:
            return f"<div class='palette-empty'>Failed to analyze image colors: {e}</div>", "", "", "", "", "", "", ""

    # Semantic search removed

    # Mood generation removed

    # Clear removed

    # Logging helpers removed (mood features removed)


# -------------------------------------------------
# 5. UI
# -------------------------------------------------

def create_ui(generator: ImagePaletteGenerator):
    # Helper to shorten prompt labels for dropdown (kept for future use)
    def _short_label(text, maxlen=75):
        t = (text or "").replace("\n", " ").strip()
        return t if len(t) <= maxlen else t[:maxlen-3] + "..."

    def _noop():
        return None

    with gr.Blocks() as demo:
        gr.HTML(
            """
            <style>
            @keyframes twinkle {
                0%, 100% { opacity: 0.3; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.2); }
            }
            
            @keyframes shootingStar1 {
                0% {
                    transform: translateX(-100px) translateY(0);
                    opacity: 0;
                }
                5% { opacity: 1; }
                95% { opacity: 1; }
                100% {
                    transform: translateX(calc(100vw + 100px)) translateY(0);
                    opacity: 0;
                }
            }
            
            @keyframes shootingStar2 {
                0% {
                    transform: translateX(calc(100vw + 100px)) translateY(0);
                    opacity: 0;
                }
                5% { opacity: 1; }
                95% { opacity: 1; }
                100% {
                    transform: translateX(-100px) translateY(0);
                    opacity: 0;
                }
            }
            
            @keyframes shootingStar3 {
                0% {
                    transform: translateX(-100px) translateY(0);
                    opacity: 0;
                }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% {
                    transform: translateX(calc(100vw + 100px)) translateY(0);
                    opacity: 0;
                }
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            
            @keyframes pulse-glow {
                0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.4); }
                50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.7); }
            }
            
            @keyframes shimmer {
                0% { background-position: -100% 0; }
                100% { background-position: 200% 0; }
            }
            
            :root {
                --palette-bar-height: 56px;
                --palette-count: 15;
                --gold-1: #FFD700;
                --gold-2: #FFA500;
                --gold-3: #DAA520;
                --gold-4: #B8860B;
            }
            
            body {
                background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
                position: relative;
                overflow-x: hidden;
            }
            
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                background-image:
                    radial-gradient(2px 2px at 20% 30%, rgba(255, 215, 0, 0.8), transparent),
                    radial-gradient(2px 2px at 60% 70%, rgba(255, 215, 0, 0.6), transparent),
                    radial-gradient(1px 1px at 50% 50%, rgba(255, 215, 0, 0.7), transparent),
                    radial-gradient(1px 1px at 80% 10%, rgba(255, 215, 0, 0.5), transparent),
                    radial-gradient(2px 2px at 90% 60%, rgba(255, 215, 0, 0.8), transparent),
                    radial-gradient(1px 1px at 33% 80%, rgba(255, 215, 0, 0.6), transparent),
                    radial-gradient(2px 2px at 15% 90%, rgba(255, 215, 0, 0.7), transparent);
                background-size: 200% 200%, 300% 300%, 250% 250%, 280% 280%, 220% 220%, 260% 260%, 240% 240%;
                animation: twinkle 3s ease-in-out infinite;
                z-index: 1;
            }
            
            body::after {
                content: '';
                position: fixed;
                top: 15%;
                left: 0;
                width: 80px;
                height: 2px;
                background: linear-gradient(to right, transparent, rgba(255, 215, 0, 0.9), rgba(255, 215, 0, 0.6), transparent);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.9), 0 0 30px rgba(255, 215, 0, 0.7);
                border-radius: 50%;
                animation: shootingStar1 4s ease-in-out infinite;
                z-index: 1;
            }
            
            .gradio-container::before {
                content: '';
                position: fixed;
                top: 45%;
                right: 0;
                width: 100px;
                height: 2px;
                background: linear-gradient(to left, transparent, rgba(255, 215, 0, 0.9), rgba(255, 215, 0, 0.6), transparent);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.9), 0 0 30px rgba(255, 215, 0, 0.7);
                border-radius: 50%;
                animation: shootingStar2 5s ease-in-out 1s infinite;
                z-index: 1;
                pointer-events: none;
            }
            
            .gradio-container::after {
                content: '';
                position: fixed;
                top: 75%;
                left: 0;
                width: 90px;
                height: 2px;
                background: linear-gradient(to right, transparent, rgba(255, 215, 0, 0.85), rgba(255, 215, 0, 0.55), transparent);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.85), 0 0 30px rgba(255, 215, 0, 0.65);
                border-radius: 50%;
                animation: shootingStar3 3.5s ease-in-out 2s infinite;
                z-index: 1;
                pointer-events: none;
            }
            
            .gradio-container {
                position: relative;
                z-index: 2;
            }

            #main_layout {
                gap: 28px;
                margin: 18px 28px 32px;
                align-items: flex-start;
                position: relative;
            }

            #visual_panel {
                gap: 18px;
            }

            #image_palette_group {
                position: relative;
                overflow: hidden;
                border-radius: 22px;
                background: linear-gradient(145deg, rgba(27, 39, 53, 0.9), rgba(15, 20, 30, 0.85));
                box-shadow: 0 30px 70px rgba(0, 0, 0, 0.5), 0 0 80px rgba(255, 215, 0, 0.3), inset 0 0 60px rgba(255, 215, 0, 0.1);
                transition: all 0.3s ease;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.3);
            }
            
            #image_palette_group:hover {
                transform: translateY(-4px);
                box-shadow: 0 40px 90px rgba(0, 0, 0, 0.6), 0 0 100px rgba(255, 215, 0, 0.5), inset 0 0 80px rgba(255, 215, 0, 0.15);
            }

            #image_palette_group .gr-image {
                min-height: 420px;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 0;
            }

            #image_palette_group .gr-image img {
                width: 100% !important;
                height: 100% !important;
                object-fit: contain !important;
                background: rgba(17, 19, 25, 0.32);
            }

            #random_palette_html {
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                pointer-events: none;
            }

            #random_palette_html .palette-progress-wrapper {
                padding: 16px 22px 22px;
                background: linear-gradient(180deg, rgba(15, 18, 28, 0.05) 0%, rgba(15, 18, 28, 0.75) 55%, rgba(12, 16, 24, 0.92) 100%);
                backdrop-filter: blur(12px);
            }

            #random_palette_html .palette-progress-bar,
            #random_palette_html .palette-segment {
                pointer-events: auto;
            }

            .palette-progress-wrapper {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }

            .palette-title {
                font-family: "Inter", "Segoe UI", sans-serif;
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: rgba(255, 215, 0, 0.9);
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
            }

            #random_palette_html .palette-title {
                color: rgba(255, 215, 0, 1);
                text-shadow: 0 0 15px rgba(255, 215, 0, 0.8), 0 1px 3px rgba(0, 0, 0, 0.55);
            }

            .palette-progress-bar {
                display: flex;
                gap: 0;
                width: 100%;
                height: var(--palette-bar-height);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.26);
            }

            .palette-segment {
                flex: 1 1 calc(100% / var(--palette-count));
                min-width: calc(100% / var(--palette-count));
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: "Inter", "Segoe UI", sans-serif;
                font-size: 0.85rem;
                font-weight: 600;
                padding: 0 6px;
                text-align: center;
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.45);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border-right: 1px solid rgba(255, 255, 255, 0.25);
                position: relative;
                overflow: hidden;
            }
            
            .palette-segment::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                transition: left 0.5s ease;
            }
            
            .palette-segment:hover::before {
                left: 100%;
            }

            .palette-segment:last-child {
                border-right: none;
            }

            .palette-segment span {
                opacity: 0;
                transition: opacity 0.18s ease;
            }

            .palette-segment:hover {
                transform: translateY(-4px) scale(1.05);
                filter: brightness(1.15) saturate(1.2);
                z-index: 10;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            }

            .palette-segment:hover span {
                opacity: 1;
            }

            .palette-empty {
                padding: 14px 0;
                text-align: center;
                color: #7a7a7a;
                font-size: 0.92rem;
            }
            #chaotic_palette_container .palette-progress-wrapper {
                background: linear-gradient(135deg, rgba(30, 30, 30, 0.95), rgba(20, 20, 20, 0.9));
                border-radius: 22px;
                padding: 22px 24px 26px;
                box-shadow: 0 24px 40px rgba(0, 0, 0, 0.4), 0 0 60px rgba(255, 215, 0, 0.25), inset 0 0 40px rgba(255, 215, 0, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 215, 0, 0.4);
                transition: all 0.3s ease;
            }
            
            #chaotic_palette_container .palette-progress-wrapper:hover {
                box-shadow: 0 28px 50px rgba(0, 0, 0, 0.5), 0 0 80px rgba(255, 215, 0, 0.4), inset 0 0 60px rgba(255, 215, 0, 0.15);
                transform: translateY(-2px);
            }

            @media (prefers-color-scheme: dark) {
                #chaotic_palette_container .palette-progress-wrapper {
                    background: linear-gradient(135deg, rgba(30, 30, 30, 0.95), rgba(15, 15, 15, 0.9));
                    box-shadow: 0 26px 44px rgba(0, 0, 0, 0.6), 0 0 60px rgba(255, 215, 0, 0.3);
                    border: 1px solid rgba(255, 215, 0, 0.4);
                }
                
                #chaotic_palette_container .palette-progress-wrapper:hover {
                    box-shadow: 0 30px 54px rgba(0, 0, 0, 0.7), 0 0 80px rgba(255, 215, 0, 0.45);
                }

                .palette-title {
                    color: rgba(255, 215, 0, 1);
                    text-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
                }
            }

            #chaotic_palette_container .palette-progress-bar {
                height: 48px;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.18);
            }

            #chaotic_palette_container .palette-title {
                letter-spacing: 0.04em;
                text-transform: none;
                font-size: 1rem;
            }

            #chaotic_palette_container .palette-segment {
                border-right: 1px solid rgba(15, 23, 42, 0.18);
            }

            #chaotic_palette_container .palette-segment:last-child {
                border-right: none;
            }

            #control_panel {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }

            .control-card {
                background: linear-gradient(135deg, rgba(30, 30, 30, 0.92), rgba(20, 20, 20, 0.88));
                border-radius: 22px;
                padding: 22px 24px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 40px rgba(255, 215, 0, 0.2), inset 0 0 30px rgba(255, 215, 0, 0.05);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.3);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .control-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.1), transparent);
                animation: shimmer 3s infinite;
            }
            
            .control-card:hover {
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5), 0 0 60px rgba(255, 215, 0, 0.35), inset 0 0 40px rgba(255, 215, 0, 0.08);
                transform: translateY(-2px);
            }

            @media (prefers-color-scheme: dark) {
                .control-card {
                    background: linear-gradient(135deg, rgba(25, 25, 25, 0.95), rgba(15, 15, 15, 0.92));
                    box-shadow: 0 26px 54px rgba(0, 0, 0, 0.6), 0 0 50px rgba(255, 215, 0, 0.25);
                    border: 1px solid rgba(255, 215, 0, 0.35);
                }
                
                .control-card:hover {
                    box-shadow: 0 30px 64px rgba(0, 0, 0, 0.7), 0 0 70px rgba(255, 215, 0, 0.4);
                }
            }

            .control-row {
                gap: 12px !important;
            }

            .control-row button,
            .control-row .gr-button {
                min-height: 46px;
                border-radius: 14px !important;
                font-weight: 600;
                letter-spacing: 0.03em;
                white-space: normal !important;
                padding-inline: 14px;
                background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #DAA520 100%) !important;
                color: #000 !important;
                border: none !important;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
                position: relative;
                overflow: hidden;
                text-shadow: 0 1px 2px rgba(255, 255, 255, 0.3);
            }
            
            .control-row button::before,
            .control-row .gr-button::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 0;
                height: 0;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.4);
                transform: translate(-50%, -50%);
                transition: width 0.6s, height 0.6s;
            }
            
            .control-row button:hover::before,
            .control-row .gr-button:hover::before {
                width: 300px;
                height: 300px;
            }
            
            .control-row button:hover,
            .control-row .gr-button:hover {
                transform: translateY(-2px) scale(1.02);
                box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5), 0 0 40px rgba(255, 215, 0, 0.3) !important;
                background: linear-gradient(135deg, #FFA500 0%, #FFD700 50%, #FFA500 100%) !important;
            }
            
            .control-row button:active,
            .control-row .gr-button:active {
                transform: translateY(0) scale(0.98);
            }

            .segment-button-row {
                display: flex !important;
                flex-wrap: wrap;
                gap: 0 !important;
                border-radius: 14px;
                overflow: hidden;
                box-shadow: 0 16px 34px rgba(0, 0, 0, 0.4), 0 0 30px rgba(255, 215, 0, 0.25);
                background: linear-gradient(135deg, rgba(40, 40, 40, 0.95), rgba(25, 25, 25, 0.9));
                border: 1px solid rgba(255, 215, 0, 0.3);
            }

            .segment-button-row > div {
                flex: 1 1 0;
                display: flex;
                min-width: 0;
            }

            .segment-button-row button {
                flex: 1 1 auto;
                border-radius: 0 !important;
                border: none !important;
                min-height: 46px;
                white-space: normal !important;
                padding-inline: 14px;
                background: transparent !important;
                transition: all 0.3s ease !important;
                position: relative;
                color: rgba(255, 215, 0, 0.9) !important;
            }
            
            .segment-button-row button:hover {
                background: linear-gradient(135deg, rgba(255, 215, 0, 0.2), rgba(218, 165, 32, 0.2)) !important;
                transform: scale(1.05);
                z-index: 2;
                color: rgba(255, 215, 0, 1) !important;
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
            }
            
            .segment-button-row button:active {
                transform: scale(0.95);
            }

            .segment-button-row > div:not(:last-child) button {
                border-right: 1px solid rgba(15, 23, 42, 0.12) !important;
            }

            @media (prefers-color-scheme: dark) {
                .segment-button-row {
                    background: linear-gradient(135deg, rgba(30, 30, 30, 0.95), rgba(15, 15, 15, 0.9));
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), 0 0 40px rgba(255, 215, 0, 0.3);
                    border: 1px solid rgba(255, 215, 0, 0.4);
                }
                
                .segment-button-row button:hover {
                    background: linear-gradient(135deg, rgba(255, 215, 0, 0.3), rgba(218, 165, 32, 0.3)) !important;
                }

                .segment-button-row > div:not(:last-child) button {
                    border-right: 1px solid rgba(255, 215, 0, 0.2) !important;
                }
            }

            .hidden-textstore {
                position: absolute;
                width: 1px !important;
                height: 1px !important;
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
                clip: rect(0 0 0 0) !important;
                clip-path: inset(50%) !important;
                white-space: nowrap !important;
            }

            .hidden-textstore textarea,
            .hidden-textstore label {
                position: absolute !important;
                left: -9999px !important;
                opacity: 0 !important;
                pointer-events: none !important;
            }

            #rand_count_slider label {
                font-size: 0.95rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                background: linear-gradient(135deg, #FFD700, #FFA500, #DAA520);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                filter: drop-shadow(0 0 5px rgba(255, 215, 0, 0.3));
            }

            #rand_count_slider .gr-slider {
                padding-top: 8px;
            }

            #palette_data_section .gr-textbox textarea {
                height: 70px !important;
                font-family: "JetBrains Mono", Consolas, monospace;
                border-radius: 12px !important;
                border: 2px solid rgba(255, 215, 0, 0.3) !important;
                transition: all 0.3s ease !important;
                background: rgba(20, 20, 20, 0.5) !important;
                color: rgba(255, 215, 0, 0.9) !important;
            }
            
            #palette_data_section .gr-textbox textarea:focus {
                border-color: rgba(255, 215, 0, 0.6) !important;
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.3), inset 0 0 10px rgba(255, 215, 0, 0.1) !important;
            }

            #palette_data_section button {
                width: 100%;
            }

            #raw_hex_row_input textarea {
                font-family: "JetBrains Mono", Consolas, monospace;
            }

            @media (max-width: 1100px) {
                #main_layout {
                    flex-direction: column;
                }

                #image_palette_group .gr-image {
                    min-height: 320px;
                }

                #control_panel {
                    width: 100%;
                }
            }
            </style>
            """
        )

        # Visual canvas and control panels
        with gr.Row(elem_id="main_layout"):
            with gr.Column(scale=7, elem_id="visual_panel"):
                with gr.Group(elem_id="image_palette_group"):
                    image_upload = gr.Image(type="pil", label="", elem_id="image_upload")
                    random_palette = gr.HTML(elem_id="random_palette_html")
                with gr.Group(elem_id="chaotic_palette_container"):
                    chaotic_palette = gr.HTML(elem_id="chaotic_palette_html")

            with gr.Column(scale=5, elem_id="control_panel"):
                with gr.Group(elem_classes=["control-card"], elem_id="primary_controls_card"):
                    rand_count = gr.Slider(minimum=3, maximum=30, step=1, value=15, label="Number of Colors", elem_id="rand_count_slider")
                    with gr.Row(elem_classes=["segment-button-row"], elem_id="action_buttons_row"):
                        rand_btn = gr.Button("Generate", elem_id="rand_btn")
                        change_image_btn = gr.Button("Change Image", elem_id="change_image_btn")
                        clear_image_btn = gr.Button("Clear Image", elem_id="clear_image_btn")

                with gr.Group(elem_classes=["control-card"], elem_id="palette_data_section"):
                    gr.Markdown("**Palette Data & Export**")
                    rand_hex_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="rand_hex_codes_box",
                        elem_classes=["hidden-textstore"],
                    )
                    rand_hex_reversed_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="rand_hex_reversed_box",
                        elem_classes=["hidden-textstore"],
                    )
                    chaotic_hex_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="chaotic_hex_codes_box",
                        elem_classes=["hidden-textstore"],
                    )

                    rand_font_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="rand_font_codes_box",
                        elem_classes=["hidden-textstore"],
                    )
                    chaotic_font_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="chaotic_font_codes_box",
                        elem_classes=["hidden-textstore"],
                    )

                    rand_font_reversed_box = gr.Textbox(
                        label="",
                        show_label=False,
                        interactive=False,
                        elem_id="rand_font_reversed_box",
                        elem_classes=["hidden-textstore"],
                    )

                    with gr.Row(elem_classes=["segment-button-row"], elem_id="progression_copy_row"):
                        copy_rand_hex_btn = gr.Button("Copy Color Progression", elem_id="copy_rand_hex_btn")
                        copy_rand_font_btn = gr.Button("Copy Progression Text", elem_id="copy_rand_font_btn")
                    
                    with gr.Row(elem_classes=["segment-button-row"], elem_id="awesome_copy_row"):
                        copy_chaotic_hex_btn = gr.Button("Copy Awesome Palette", elem_id="copy_chaotic_hex_btn")
                        copy_chaotic_font_btn = gr.Button("Copy Awesome Text", elem_id="copy_chaotic_font_btn")

                with gr.Group(elem_classes=["control-card"], elem_id="raw_hex_section"):
                    gr.Markdown("**Manual Palette Builder**")
                    with gr.Row(elem_classes=["control-row"], elem_id="raw_hex_row_input"):
                        raw_hex_input = gr.Textbox(
                            label="Raw Hex Codes (no #, any separators)",
                            lines=4,
                            placeholder="e.g. FF0000 00FF00 0000FF ...",
                            elem_id="raw_hex_input_box",
                        )
                    with gr.Row(elem_classes=["control-row"], elem_id="raw_hex_row_outputs"):
                        with gr.Column(scale=1):
                            raw_hex_output_hex = gr.Textbox(
                                label="Normalized Hex (#RRGGBB)",
                                interactive=False,
                                elem_id="raw_hex_output_hex_box",
                            )
                        with gr.Column(scale=1):
                            raw_hex_output_font = gr.Textbox(
                                label="Font Colors (#000/#FFF)",
                                interactive=False,
                                elem_id="raw_hex_output_font_box",
                            )
                    with gr.Row(elem_classes=["segment-button-row"], elem_id="raw_hex_row_copy"):
                        copy_raw_hex_btn = gr.Button("Copy Hex", elem_id="copy_raw_hex_btn")
                        copy_raw_font_btn = gr.Button("Copy Font", elem_id="copy_raw_font_btn")

        # --- End of component definitions ---

        # Minimal UI for image-based palette generation

        def random_from_image(image: Image.Image | None, n: int):
            html, hex_csv, hex_csv_reversed, font_csv, font_csv_reversed, chaotic_html, chaotic_hex_csv, chaotic_font_csv = generator.generate_random_palette_from_image(
                image, int(n) if n is not None else 15
            )
            return html, hex_csv, hex_csv_reversed, font_csv, font_csv_reversed, chaotic_html, chaotic_hex_csv, chaotic_font_csv

        def clear_image():
            # Reset image and clear all outputs
            return None, "", "", "", "", "", "", "", ""

        # Random image palette wiring (button and auto on image change)
        rand_btn.click(
            fn=random_from_image,
            inputs=[image_upload, rand_count],
            outputs=[random_palette, rand_hex_box, rand_hex_reversed_box, rand_font_box, rand_font_reversed_box, chaotic_palette, chaotic_hex_box, chaotic_font_box],
        )
        image_upload.change(
            fn=random_from_image,
            inputs=[image_upload, rand_count],
            outputs=[random_palette, rand_hex_box, rand_hex_reversed_box, rand_font_box, rand_font_reversed_box, chaotic_palette, chaotic_hex_box, chaotic_font_box],
        )

        # Keep the canvas height in sync with the number of cards selected (JS-only to avoid preprocess issues)
        rand_count.change(
            fn=_noop,
            inputs=None,
            outputs=None,
            js="""
            () => {
                const root = document.documentElement;
                const host = document.getElementById('rand_count_slider');
                let n = null;
                if (host) {
                    const range = host.querySelector('input[type="range"]');
                    if (range) n = parseInt(range.value, 10);
                }
                if (root) {
                    root.style.setProperty('--palette-count', String(Number.isFinite(n) ? n : 15));
                }
            }
            """,
        )

        # Change Image opens the file chooser of the image component
        change_image_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js="""
            () => {
              const root = document.getElementById('image_upload');
              if (!root) return;
              const input = root.querySelector('input[type=file]');
              if (input) input.click();
            }
            """,
        )

        # Clear image and palettes
        clear_image_btn.click(
            fn=clear_image,
            inputs=None,
            outputs=[image_upload, random_palette, rand_hex_box, rand_hex_reversed_box, rand_font_box, rand_font_reversed_box, chaotic_palette, chaotic_hex_box, chaotic_font_box],
        )

        # Copy Chaotic Hex Codes functionality
        copy_chaotic_hex_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js=
            """
            () => {
               const host = document.getElementById('chaotic_hex_codes_box');
               const field = host ? host.querySelector('textarea, input') : null;
               if(field){
                   navigator.clipboard.writeText(field.value||'');
                   const btn = document.getElementById('copy_chaotic_hex_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }else{
                   alert('No chaotic palette available yet. Generate a palette first.');
               }
            }
            """,
        )

        # Copy Chaotic Font Colors functionality
        copy_chaotic_font_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js=
            """
            () => {
               const host = document.getElementById('chaotic_font_codes_box');
               const field = host ? host.querySelector('textarea, input') : null;
               if(field){
                   navigator.clipboard.writeText(field.value||'');
                   const btn = document.getElementById('copy_chaotic_font_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }else{
                   alert('No chaotic text colors available yet. Generate a palette first.');
               }
            }
            """,
        )

        # Copy Random Hex Codes functionality
        copy_rand_hex_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js=
            """
            () => {
               const host = document.getElementById('rand_hex_codes_box');
               const field = host ? host.querySelector('textarea, input') : null;
               if(field){
                   navigator.clipboard.writeText(field.value||'');
                   const btn = document.getElementById('copy_rand_hex_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }else{
                   alert('No palette generated yet. Upload an image and click Generate.');
               }
            }
            """,
        )

        # Copy Random Font Colors functionality
        copy_rand_font_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js=
            """
            () => {
               const host = document.getElementById('rand_font_codes_box');
               const field = host ? host.querySelector('textarea, input') : null;
               if(field){
                   navigator.clipboard.writeText(field.value||'');
                   const btn = document.getElementById('copy_rand_font_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }else{
                   alert('No text color data available yet. Generate a palette first.');
               }
            }
            """,
        )

        def _process_raw_hex(raw: str):
            """Process manual palette input supporting hex codes and RGB(r,g,b) entries.

            Accepts:
            - Hex codes: #RRGGBB or RRGGBB (case-insensitive)
            - RGB tuples: RGB(r, g, b) with 0-255 components (case-insensitive, optional whitespace)
            Preserves first-seen order and removes duplicates.
            Returns two comma-separated strings: hex codes and recommended font colors.
            """
            if not raw:
                return "", ""

            pattern = re.compile(r"(?i)#?([0-9a-f]{6})|rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)")
            seen: set[str] = set()
            ordered: list[str] = []

            for m in pattern.finditer(raw):
                hex_part = m.group(1)
                if hex_part:
                    code = hex_part.upper()
                    full = f"#{code}"
                    if full not in seen:
                        seen.add(full)
                        ordered.append(full)
                    continue

                r_str, g_str, b_str = m.group(2), m.group(3), m.group(4)
                if r_str is not None and g_str is not None and b_str is not None:
                    try:
                        r, g, b = int(r_str), int(g_str), int(b_str)
                        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                            continue
                        full = f"#{r:02X}{g:02X}{b:02X}"
                        if full not in seen:
                            seen.add(full)
                            ordered.append(full)
                    except Exception:
                        continue

            if not ordered:
                return "", ""

            fonts = [generator._get_text_color_for_bg(h) for h in ordered]
            return ", ".join(ordered), ", ".join(fonts)

        raw_hex_input.change(
            fn=_process_raw_hex,
            inputs=[raw_hex_input],
            outputs=[raw_hex_output_hex, raw_hex_output_font],
        )

        # Copy buttons for raw hex processing
        copy_raw_hex_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js="""
            () => {
               const ta = document.querySelector('#raw_hex_output_hex_box textarea');
               if(ta){
                   navigator.clipboard.writeText(ta.value||'');
                   const btn = document.getElementById('copy_raw_hex_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }
            }
            """,
        )
        copy_raw_font_btn.click(
            fn=_noop,
            inputs=None,
            outputs=None,
            js="""
            () => {
               const ta = document.querySelector('#raw_hex_output_font_box textarea');
               if(ta){
                   navigator.clipboard.writeText(ta.value||'');
                   const btn = document.getElementById('copy_raw_font_btn');
                   if(btn){
                       const old = btn.textContent;
                       btn.textContent = 'Copied!';
                       setTimeout(()=>btn.textContent=old,1200);
                   }
               }
            }
            """,
        )

    return demo


if __name__ == "__main__":
    cfg = Config()
    gen = ImagePaletteGenerator(cfg)
    ui = create_ui(gen)
    try:
        # Attach middleware to underlying FastAPI app before launch
        if hasattr(ui, "app") and ui.app is not None:
            ui.app.add_middleware(StripBrotliMiddleware)
    except Exception:
        pass

    # Electron integration: if ELECTRON_WRAPPER env var is set, don't auto-open a browser.
    electron_mode = os.environ.get("ELECTRON_WRAPPER") == "1"
    # Allow external override of port for coordination with Electron wrapper
    port_env = os.environ.get("PORT")
    server_port = None
    if port_env:
        try:
            server_port = int(port_env)
        except Exception:
            server_port = None

    if electron_mode:
        # Run headless (no browser), bind to localhost only for security, suppress share links.
        ui.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            inbrowser=False,
            share=False,
            show_api=False,
            prevent_thread_lock=False  # Keep blocking so process lifetime matches Electron expectation
        )
    else:
        # Enable a public Gradio share link when running locally
        ui.launch(share=True)
