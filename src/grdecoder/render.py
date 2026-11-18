"""
render.py - from a stretch of signal and a timebase to a picture.

Steps, in order:

1. Droop correction (optional). The recording chain was AC-coupled: after
   every large level change the signal decays back toward zero (visible as
   dark-to-light shading down each picture line). This is the behaviour of
   a first-order high-pass filter with time constant tau. Its exact inverse
   is   x(t) = y(t) + (1/tau) * integral(y)   which we apply with a very
   slow leak on the integrator so that DC offsets cannot make it drift.
   tau can be estimated automatically (see `estimate_droop_tau`).

2. Anti-alias low-pass. A line holds ~2600 samples of picture but we draw
   it with a few hundred pixels, so the signal is band-limited to the pixel
   grid before sampling.

3. Sampling on the timebase. Pixel (row j, column k) is read at file time
   t(k + u_j), where u_j runs from `top` to `bottom` (fractions of a line,
   measured from the destaggered sync).

4. Tone mapping is done separately (`tone_map`) so that it can be changed
   in the GUI without re-sampling.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .timebase import Timebase


# --------------------------------------------------------------------------
# droop (low-frequency loss) correction
# --------------------------------------------------------------------------
def droop_correct(x: np.ndarray, fs: float, tau_s: float, leak_s: float = 0.05) -> np.ndarray:
    """Undo a first-order high-pass of time constant `tau_s` (seconds).

    y = x + g * I,   I[n] = x[n] + lam * I[n-1],   g = 1 / (tau * fs)

    `leak_s` (default 50 ms, ~6 lines) bounds how long any DC offset can
    accumulate in the integrator; it is far longer than tau (~1 ms) so the
    correction itself is unaffected.
    """
    if tau_s is None or tau_s <= 0:
        return x
    g = 1.0 / (tau_s * fs)
    lam = np.exp(-1.0 / (leak_s * fs))
    integ = ss.lfilter([1.0], [1.0, -lam], x - np.mean(x))
    return x + g * integ


def estimate_droop_tau(x: np.ndarray, a: int, plateaus: list[tuple[float, float]], fs: float) -> float | None:
    """Estimate the droop time constant from flat reference plateaus.

    Every second sync pulse is followed by a long flat pulse top (~0.4 ms).
    In the signal as recorded that top slopes downward; after correct droop
    compensation it is flat. Because the correction is linear in g = 1/tau,
    the slope after correction is  slope(y) + g * slope(I)  and the g that
    makes it zero follows in closed form (least squares over all plateaus).

    x         signal stretch, starting at absolute sample `a`
    plateaus  list of absolute (start, stop) sample positions of plateaus
    Returns tau in seconds, or None if no usable estimate.
    """
    lam = np.exp(-1.0 / (0.05 * fs))
    integ = ss.lfilter([1.0], [1.0, -lam], x - np.mean(x))
    sy, si = [], []
    for p0, p1 in plateaus:
        i0, i1 = int(p0) - a, int(p1) - a
        if i0 < 0 or i1 > len(x) or i1 - i0 < 8:
            continue
        n = np.arange(i1 - i0)
        sy.append(np.polyfit(n, x[i0:i1], 1)[0])
        si.append(np.polyfit(n, integ[i0:i1], 1)[0])
    if len(sy) < 10:
        return None
    sy, si = np.array(sy), np.array(si)
    g = -np.dot(sy, si) / np.dot(si, si)
    if not np.isfinite(g) or g <= 0:
        return None
    return 1.0 / (g * fs)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def sample_picture(
    x: np.ndarray,
    a: int,
    tb: Timebase,
    first_line: int,
    n_lines: int,
    top: float,
    bottom: float,
    height: int,
    period: float,
    field_offset: float = 0.0,
    clamp: tuple[float, float] | None = None,
) -> np.ndarray:
    """Read the picture off the signal.

    x            signal stretch starting at absolute sample index `a`
    tb           timebase of the run (line coordinate -> absolute sample)
    first_line   index (in run lines) of the picture's first line
    top, bottom  active window inside a line, as fractions of a line
    height       number of output rows
    field_offset extra shift (samples) applied with opposite sign to even
                 and odd lines, to fine-align the two interlaced fields
    clamp        (start, stop) fractions of a line used as black-level
                 reference ("back porch", just after the sync pulse). Its
                 median is subtracted from every line, which removes
                 line-to-line level jumps that show up as vertical streaks.
    Returns a float image of shape (height, n_lines).
    """
    # band-limit to the row spacing (Butterworth, zero phase)
    samples_per_row = (bottom - top) * period / height
    if samples_per_row > 1.5:
        wn = 0.9 / samples_per_row
        sos = ss.butter(6, wn, output="sos")
        x = ss.sosfiltfilt(sos, x)

    u = top + (bottom - top) * (np.arange(height) + 0.5) / height  # (H,)
    k = first_line + np.arange(n_lines)  # (W,)
    q = k[None, :] + u[:, None]  # (H, W)
    t = tb(q.ravel()).reshape(q.shape)
    # Parity is counted along the picture's own columns, the same way
    # measure_field_offset counts it (a picture can start on an odd run line).
    fshift = np.where(np.arange(n_lines) % 2 == 0, 0.5, -0.5)[None, :] * field_offset
    img = np.interp(t + fshift - a, np.arange(len(x)), x)
    if clamp is not None:
        ur = np.linspace(clamp[0], clamp[1], 48)
        tr = tb((k[None, :] + ur[:, None]).ravel()).reshape(len(ur), n_lines)
        ref = np.median(np.interp(tr - a, np.arange(len(x)), x), axis=0)
        img = img - ref[None, :]
    return img


def measure_field_offset(img: np.ndarray, max_shift: float = 4.0) -> float:
    """Vertical misalignment between even and odd columns, in rows.

    Compares each even column with the average of its two odd neighbours
    and finds the sub-row shift that matches them best (cross-correlation
    of the column derivatives, parabolic peak). Positive = odd columns sit
    lower than even ones.
    """
    d = np.diff(img, axis=0)
    ev = d[:, 2:-1:2]
    od = 0.5 * (d[:, 1:-2:2] + d[:, 3::2])
    n = min(ev.shape[1], od.shape[1])
    ev, od = ev[:, :n], od[:, :n]
    shifts = np.arange(-int(max_shift), int(max_shift) + 1)
    c = []
    for s in shifts:
        c.append(np.sum(ev * np.roll(od, -s, axis=0)))
    c = np.array(c)
    i = int(np.argmax(c))
    frac = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den < 0:
            frac = 0.5 * (c[i - 1] - c[i + 1]) / den
    return float(shifts[i] + frac)


def equalise_fields(img: np.ndarray, smooth_rows: int = 31) -> tuple[np.ndarray, float]:
    """Remove the brightness difference between even and odd lines.

    Even and odd lines start with differently shaped sync pulses, so after
    the AC-coupled recording chain their levels differ slightly (3-6 % of
    full scale on the master tape) - visible as fine vertical stripes. For
    every row we take the median difference between even columns and the
    average of their odd neighbours, smooth it down the picture, and split
    it evenly between the two fields. Returns (image, mean difference).
    """
    from scipy.ndimage import uniform_filter1d

    out = img.copy()
    ev = img[:, 2:-1:2]
    od = 0.5 * (img[:, 1:-2:2] + img[:, 3::2])
    d = uniform_filter1d(np.median(ev - od, axis=1), smooth_rows, mode="nearest")
    out[:, 0::2] -= 0.5 * d[:, None]
    out[:, 1::2] += 0.5 * d[:, None]
    return out, float(np.mean(d))


# --------------------------------------------------------------------------
# tone mapping and output
# --------------------------------------------------------------------------
def auto_levels(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    """Black and white points from percentiles of the picture."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-9
    return float(lo), float(hi)


def tone_map(img: np.ndarray, black: float, white: float, gamma: float = 1.0, invert: bool = False) -> np.ndarray:
    """Map signal values to 0..1 display values."""
    v = (img - black) / (white - black)
    v = np.clip(v, 0.0, 1.0)
    if invert:
        v = 1.0 - v
    if gamma != 1.0:
        v = v ** (1.0 / gamma)
    return v


def resize_columns(img: np.ndarray, width: int) -> np.ndarray:
    """Stretch the 512 picture lines to `width` columns (linear interpolation).

    Used to give exported images square pixels at the chosen aspect ratio.
    Works for grey (H, W) and colour (H, W, 3) images.
    """
    h, w = img.shape[:2]
    if width == w:
        return img
    xs = (np.arange(width) + 0.5) * w / width - 0.5
    xs = np.clip(xs, 0, w - 1)
    i0 = np.floor(xs).astype(int)
    i1 = np.minimum(i0 + 1, w - 1)
    f = (xs - i0)
    if img.ndim == 3:
        f = f[None, :, None]
    else:
        f = f[None, :]
    return img[:, i0] * (1 - f) + img[:, i1] * f


def combine_colour(r: np.ndarray, g: np.ndarray, b: np.ndarray, max_shift: int = 12) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Stack three separations into one RGB image.

    The three frames of a colour picture are decoded independently and can
    be offset by a few lines/rows (each frame's start is found separately),
    so G and B are first registered to R: the integer shift (within
    +-`max_shift`) that maximises the cross-correlation of their edge images,
    computed with FFTs. Returns (rgb float image, [(dy, dx) applied to g, b]).
    """
    # frames can differ by a few lines in length: crop to the common size
    h = min(x.shape[0] for x in (r, g, b))
    w = min(x.shape[1] for x in (r, g, b))
    r, g, b = r[:h, :w], g[:h, :w], b[:h, :w]
    er = _edges(r)
    out, shifts = [r], []
    for ch in (g, b):
        ec = _edges(ch)
        xc = np.real(np.fft.ifft2(np.fft.fft2(er) * np.conj(np.fft.fft2(ec))))
        dys = np.r_[0 : max_shift + 1, -max_shift:0]
        dxs = np.r_[0 : max_shift + 1, -max_shift:0]
        sub = xc[np.ix_(dys % xc.shape[0], dxs % xc.shape[1])]
        i, j = np.unravel_index(np.argmax(sub), sub.shape)
        dy, dx = int(dys[i]), int(dxs[j])
        out.append(np.roll(np.roll(ch, dy, 0), dx, 1))
        shifts.append((dy, dx))
    return np.dstack(out), shifts


def _edges(img: np.ndarray) -> np.ndarray:
    """Gradient magnitude, mean removed (for registration)."""
    gy, gx = np.gradient(img.astype(np.float64))
    e = np.hypot(gx, gy)
    return e - e.mean()


def measure_circle_aspect(img: np.ndarray) -> tuple[float, dict]:
    """Display aspect (width/height) that makes the calibration circle round.

    Picture 1 on the record is a circle, reproduced on the cover so that the
    recipient can check the decoding. Here it calibrates the picture
    geometry. The ring is found as the strongest pixels away from the
    picture's edges (the framed area has bright and dark borders that must
    not be mistaken for the ring), an axis-aligned ellipse is fitted to them
    by least squares, with outliers (streaks, specks) removed in a few
    passes, and the width/height ratio (for 512 lines) at which the ring's
    horizontal and vertical diameters are equal on screen is returned.
    `img` is the un-tone-mapped picture (rows, lines).
    """
    from scipy.ndimage import median_filter

    h, w = img.shape
    d = img - median_filter(img, size=(1, 31), mode="nearest")  # remove vertical streaks
    if abs(np.percentile(d, 0.2)) > abs(np.percentile(d, 99.8)):
        d = -d  # ring darker than background
    my, mx = int(0.04 * h), int(0.05 * w)
    inner = d[my : h - my, mx : w - mx]
    thr = np.percentile(inner, 98.5)
    ys, xs = np.nonzero(inner > thr)
    if len(xs) < 50:
        raise ValueError("No ring found.")
    ys = ys.astype(float) + my
    xs = xs.astype(float) + mx
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        M = np.c_[xs[keep] ** 2, ys[keep] ** 2, xs[keep], ys[keep]]
        A, B, C, D = np.linalg.lstsq(M, np.ones(keep.sum()), rcond=None)[0]
        cx, cy = -C / (2 * A), -D / (2 * B)
        k = 1 + C * C / (4 * A) + D * D / (4 * B)
        if A * B <= 0 or k / A <= 0:  # not an ellipse
            raise ValueError("No ring found.")
        rx, ry = np.sqrt(k / A), np.sqrt(k / B)
        r = np.hypot((xs - cx) / rx, (ys - cy) / ry)
        keep = np.abs(r - 1) < 0.04
        if keep.sum() < 50:
            raise ValueError("No ring found.")
    if not (0.15 * w < rx < 0.6 * w and 0.15 * h < ry < 0.6 * h):
        raise ValueError("No ring found.")
    # a real ring: most bright pixels lie on it, all the way round
    ang = np.arctan2((ys[keep] - cy) / ry, (xs[keep] - cx) / rx)
    coverage = len(np.unique(((ang + np.pi) / (2 * np.pi) * 36).astype(int))) / 36
    if keep.mean() < 0.6 or coverage < 0.6:
        raise ValueError("This picture is not the calibration circle.")
    aspect = (ry / h) / (rx / 512)  # aspect is defined for a width of 512 lines
    return float(aspect), {"rows": (int(cy - ry), int(cy + ry)), "lines": (int(cx - rx), int(cx + rx))}
