"""
pipeline.py - the decoder as one object: analyse a file, then render pictures.

    dec = Decoder("384kHzStereo.wav")
    dec.analyse()                      # find all pictures on both channels
    pic = dec.pictures[0]
    out = dec.render(pic, DecodeSettings())
    out.image                          # float array (rows, 512)

Everything the GUI and the command-line tool do goes through this class.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.signal as ss

from . import sync
from .audio import AudioSource
from .pictures import LINES_PER_PICTURE, find_start_marker, hatch_score, place_picture, find_picture_blocks, line_profiles, picture_mask
from .render import (auto_levels, droop_correct, equalise_fields, estimate_droop_tau, measure_field_offset,
                     resize_columns,
                     sample_picture, tone_map)
from .timebase import Timebase, build_timebase, destagger

# Nominal line period. The record cover gives "about 8 milliseconds"; the
# measured period on the master tape is 8.32-8.33 ms, i.e. a line rate of
# 120 Hz = twice the 60 Hz US mains / TV field rate the 1977 equipment was
# almost certainly locked to. We use 1/120 s as the speed reference.
NOMINAL_LINE_S = 1.0 / 120.0

# Lines kept per picture: from the start marker to the end of the framed area
# (measured 530-541 on the master tape, a few outliers up to 555).
MAX_PICTURE_LINES = 540

# Part of every line used as black-level reference: the flat stretch after
# the sync pulse and before the picture ("back porch"), as fractions of a line.
BACK_PORCH = (0.055, 0.100)


# --------------------------------------------------------------------------
# data classes
# --------------------------------------------------------------------------
@dataclass
class DecodeSettings:
    """All user-adjustable decoding parameters (defaults = recommended)."""

    # timebase
    mode: str = "smooth"            # "fixed" | "line" | "smooth"  (see timebase.py)
    smooth_hz: float = 40.0         # correction bandwidth in mode "smooth"
    destagger: bool = True          # move staggered sync pulses onto an even grid
    speed_trim: float = 1.0         # mode "fixed" only: playback-speed fine trim
    # Off by default: after destaggering, the picture content of the two fields
    # already lines up (checked on the calibration circle). The automatic
    # measurement is fooled by a fine periodic texture whose phase differs
    # between fields, so turning this on makes edges jagged on the master tape.
    field_align: bool = False       # experimental: auto-align the two interlaced fields
    field_level: bool = True        # equalise the brightness of the two fields
    # signal
    droop: bool = True              # undo the low-frequency loss (AC coupling)
    droop_tau_ms: float | None = None  # None = estimate automatically
    clamp: bool = True              # per-line black-level clamp on the back porch
    # geometry
    top: float = 0.115              # active picture window inside a line (fractions)
    bottom: float = 0.915
    height: int = 768               # output rows
    margin_lines: int = 0           # extra lines drawn left/right of the 512 picture lines
    aspect: float = 1.485           # display width/height of 512 lines; 1.485 makes picture 1 (the circle) round
    # tone
    invert: bool = True             # higher signal = darker on this recording (sun, planets come out right)
    gamma: float = 1.0
    black_pct: float = 0.3
    white_pct: float = 99.7


@dataclass
class SyncSettings:
    """Fine-tuning of the sync-pulse detection (analysis stage).

    The defaults work on the 384 kHz master transfer and on 44.1/48 kHz or
    band-limited copies of it. Change them only if pictures are missed or
    the timing plot shows many predicted (red) pulses.
    """

    template_width: float = 0.04    # length of the sync template after the rise, fraction of a line
    min_score: float = 0.6          # minimum match (0..1) for a pulse to count
    edge_scale_us: float = 20.0     # time over which a rising edge is measured (µs)


@dataclass
class Picture:
    """One picture found in the file."""

    channel: int                  # 0 = left, 1 = right
    number: int                   # 1-based count within its channel
    run_id: int                   # index into Decoder.runs[channel]
    first_line: int               # first picture line inside the run
    n_lines: int                  # normally 512
    t_start: float                # seconds
    t_end: float
    sync_found: float             # fraction of sync pulses actually measured
    label: str = ""

    @property
    def name(self) -> str:
        return f"{'LR'[self.channel]}{self.number:03d}"


@dataclass
class Rendered:
    """Result of rendering one picture."""

    picture: Picture
    settings: DecodeSettings
    image: np.ndarray             # float, (rows, n_lines), signal units
    timebase: Timebase
    lines: slice                  # picture lines inside the timebase arrays
    stagger: float                # samples
    droop_tau_ms: float | None
    field_offset_rows: float
    period_samples: float
    fs: int

    # ---- convenience views for the diagnostics panel -------------------
    @property
    def speed_vs_nominal_ppm(self) -> float:
        """Mean playback speed relative to the 120 Hz line-rate reference."""
        nominal = NOMINAL_LINE_S * self.fs
        return (nominal / self.period_samples - 1.0) * 1e6

    def timing_error_us(self, which: str = "raw") -> np.ndarray:
        e = self.timebase.error_raw if which == "raw" else self.timebase.error_used
        return e[self.lines] / self.fs * 1e6

    def display(self, black: float | None = None, white: float | None = None) -> np.ndarray:
        """Tone-mapped 0..1 image with the settings' levels and gamma."""
        if black is None or white is None:
            black, white = auto_levels(self.image, self.settings.black_pct, self.settings.white_pct)
        return tone_map(self.image, black, white, self.settings.gamma, self.settings.invert)


# --------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------
class Decoder:
    """Analyse a Voyager image-data recording and render its pictures."""

    def __init__(self, path: str | Path, speed_guess: float = 1.0, sync_settings: SyncSettings | None = None):
        self.source = AudioSource(path)
        self.sync_settings = sync_settings or SyncSettings()
        self.fs = self.source.fs
        # expected line period in this file; refined during analysis
        self.period_guess = NOMINAL_LINE_S * self.fs / speed_guess
        self.period: dict[int, float] = {}
        self.templates: dict[int, sync.SyncTemplate] = {}
        self.template_rate: dict[int, float] = {}  # sample rate the template was made at
        self.runs: dict[int, list[sync.LineTrack]] = {}
        self.pictures: list[Picture] = []
        self._tau_cache: dict[tuple[int, int], float | None] = {}

    # ---------------------------------------------------------------- analyse
    def analyse(self, channels=(0, 1), progress: Callable[[float, str], None] | None = None) -> list[Picture]:
        """Find sync pulses, line runs and pictures on the given channels."""
        self.pictures = []
        nch = len(channels)
        for ci, ch in enumerate(channels):
            def prog(f, msg, ci=ci):
                if progress:
                    progress((ci + f) / nch, f"{'LR'[ch]}: {msg}")
            self._analyse_channel(ch, prog)
        self.pictures.sort(key=lambda p: (p.channel, p.t_start))
        return self.pictures

    def _analyse_channel(self, ch: int, prog) -> None:
        fs = self.fs
        S = self.sync_settings
        P = self.period_guess
        block = int(20 * fs)
        overlap = int(0.2 * fs)
        nblocks = int(np.ceil(self.source.info.frames / (block - overlap)))
        # below ~150 kHz the sync pulse spans only a few samples: work on a
        # 4x (or more) upsampled copy so the template has enough samples
        U = 1 if fs >= 150000 else int(np.ceil(176400 / fs))

        def up(x):
            return ss.resample_poly(x, U, 1) if U > 1 else x

        # pass 1: rising edges everywhere -> rough line period
        cands = []
        for bi, (start, x) in enumerate(self.source.blocks(ch, block, overlap)):
            prog(0.15 * bi / nblocks, f"scanning {100 * bi / nblocks:.0f} %")
            if len(x) > 4 * P:
                cands.append(sync.find_candidates(x, P, fs, S.edge_scale_us) + start)
        try:
            P = sync.estimate_period(np.unique(np.concatenate(cands)), P)
        except (ValueError, IndexError):
            self.runs[ch] = []
            return

        # pass 2: aligned folds at a few places per block; the sharpest one that
        # shows the staggered sync pair gives the template
        best_key, best = None, None
        for bi, (start, x) in enumerate(self.source.blocks(ch, block, overlap)):
            prog(0.15 + 0.15 * bi / nblocks, f"learning the sync pulse {100 * bi / nblocks:.0f} %")
            xu = up(x)
            for f in (0.15, 0.5, 0.85):
                prof = sync.aligned_fold(xu, int(f * len(xu)), P * U)
                if prof is None:
                    continue
                info = sync.inspect_fold(prof, P * U, fs * U, S.edge_scale_us)
                key = (info.i2 is not None, info.sharpness)
                if best_key is None or key > best_key:
                    best_key, best = key, info
        if best is None:
            self.runs[ch] = []
            return
        tpl = sync.template_from_fold(best, P * U, S.template_width, fs * U)

        # pass 3: matched-filter detection of every sync pulse
        pos_all, score_all = [], []
        for bi, (start, x) in enumerate(self.source.blocks(ch, block, overlap)):
            prog(0.3 + 0.5 * bi / nblocks, f"measuring sync pulses {100 * bi / nblocks:.0f} %")
            pos, score = sync.detect(up(x), tpl, P * U, S.min_score)
            pos = pos / U
            # keep each pulse from exactly one block (centre part of the overlap)
            lo = 0 if start == 0 else overlap // 2
            hi = len(x) - overlap // 2
            keep = (pos >= lo) & (pos < hi)
            pos_all.append(pos[keep] + start)
            score_all.append(score[keep])
        pos = np.concatenate(pos_all) if pos_all else np.empty(0)
        score = np.concatenate(score_all) if score_all else np.empty(0)
        try:
            P = sync.estimate_period(pos[score > max(0.8, S.min_score)], P)
        except ValueError:
            pass
        self.period[ch] = P
        self.templates[ch] = tpl
        self.template_rate[ch] = fs * U

        prog(0.82, "tracking lines")
        runs = []
        for t, m, s in sync.merge_runs(sync.track_lines(pos, score, P, min_score=S.min_score), P):
            runs.append(sync.LineTrack(ch, t, m, s, float(np.median(np.diff(t)))))
        self.runs[ch] = runs

        n = 0
        for ri, run in enumerate(runs):
            prog(0.85 + 0.15 * ri / max(1, len(runs)), "locating pictures")
            ds = destagger(run.times)
            a = int(ds.times[0] - P)
            x = self.source.read(ch, a, int(ds.times[-1] + 2 * P))
            prof = line_profiles(x, a, ds.times, P)
            sim = picture_mask(prof)
            hatch = hatch_score(prof)
            run.meta["similarity"] = sim
            run.meta["hatch"] = hatch
            run.meta["stagger"] = ds.stagger
            # picture lines always carry the staggered sync (~1.6 % of a line);
            # runs without it are lead-in / test signals, not pictures
            if abs(ds.stagger) < 0.005 * P:
                continue
            for blk in find_picture_blocks(sim):
                first, nl = place_picture(blk, sim)
                # precise anchor: the picture starts right after the hatched start
                # marker. The framed area then runs ~535 lines (the cover's 512
                # plus a border); keep all of it up to the end of the block.
                end = find_start_marker(hatch, blk[0])
                if end is not None and end + 1 + LINES_PER_PICTURE // 2 <= len(run):
                    first = end + 1
                    nl = int(np.clip(blk[1] - first, LINES_PER_PICTURE, MAX_PICTURE_LINES))
                    nl = min(nl, len(run) - first)
                n += 1
                m = run.measured[first : first + nl]
                self.pictures.append(Picture(
                    channel=ch, number=n, run_id=ri, first_line=first, n_lines=nl,
                    t_start=run.times[first] / fs, t_end=run.times[min(first + nl, len(run) - 1)] / fs,
                    sync_found=float(np.mean(m)),
                ))

    # ----------------------------------------------------------------- render
    def render(self, pic: Picture, st: DecodeSettings) -> Rendered:
        """Decode one picture with the given settings."""
        run = self.runs[pic.channel][pic.run_id]
        P = self.period[pic.channel]
        if st.destagger:
            ds = destagger(run.times)
            t_sync, stagger = ds.times, ds.stagger
        else:
            t_sync, stagger = run.times, 0.0
        tb = build_timebase(t_sync, run.measured, st.mode, st.smooth_hz, self.fs, st.speed_trim)

        first = max(0, pic.first_line - st.margin_lines)
        n_lines = min(len(run) - first, pic.n_lines + 2 * st.margin_lines)

        # signal stretch covering the picture plus one line either side
        q0, q1 = first - 1, first + n_lines + 1
        a = int(min(tb(np.array([q0]))[0], run.times[max(q0, 0)])) - int(P)
        b = int(max(tb(np.array([q1]))[0], run.times[min(q1, len(run) - 1)])) + int(P)
        x = self.source.read(pic.channel, a, b)

        tau = None
        if st.droop:
            tau = st.droop_tau_ms / 1e3 if st.droop_tau_ms else self._auto_tau(pic, x, a, P)
            if tau:
                x = droop_correct(x, self.fs, tau)

        clamp = BACK_PORCH if st.clamp else None
        args = (x, a, tb, first, n_lines, st.top, st.bottom, st.height, tb.period)
        img = sample_picture(*args, clamp=clamp)
        f_rows = 0.0
        if st.field_align:
            f_rows = measure_field_offset(img)
            if abs(f_rows) > 0.05:
                spr = (st.bottom - st.top) * tb.period / st.height  # samples per row
                img = sample_picture(*args, field_offset=-f_rows * spr, clamp=clamp)
        if st.field_level:
            img, _ = equalise_fields(img)

        return Rendered(
            picture=pic, settings=st, image=img, timebase=tb,
            lines=slice(first, first + n_lines),
            stagger=stagger, droop_tau_ms=tau * 1e3 if tau else None,
            field_offset_rows=f_rows, period_samples=tb.period, fs=self.fs,
        )

    def _auto_tau(self, pic: Picture, x: np.ndarray, a: int, P: float) -> float | None:
        """Droop time constant from the flat tops of the long sync pulses (cached)."""
        key = (pic.channel, pic.run_id)
        if key in self._tau_cache:
            return self._tau_cache[key]
        t = self.runs[pic.channel][pic.run_id].times
        t = t[(t - a > 0.01 * P) & (t - a < len(x) - 0.1 * P)]
        # which parity carries the long flat top? the one that is higher 3 % of a line after the spike
        lvl = np.interp(t - a + 0.03 * P, np.arange(len(x)), x) - np.interp(t - a - 0.005 * P, np.arange(len(x)), x)
        par = 0 if np.mean(lvl[0::2]) > np.mean(lvl[1::2]) else 1
        tops = [(tk + 0.008 * P, tk + 0.040 * P) for tk in t[par::2]]
        tau = estimate_droop_tau(x, a, tops, self.fs)
        if tau is not None and not (0.05e-3 < tau < 50e-3):
            tau = None
        self._tau_cache[key] = tau
        return tau

    # ----------------------------------------------------------------- colour
    def colour_triples(self, threshold: float = 0.6) -> list[tuple[Picture, Picture, Picture]]:
        """Suggest colour pictures: three consecutive frames of the same scene.

        Colour pictures were recorded as three separations one after another.
        Neighbouring pictures are compared through small, fast renders of
        their *edge maps*: separations share edges but not brightness, while
        different pictures with a similar layout (two pages of numbers) share
        brightness but not edges. The edge maps are blurred and the best
        match over small shifts is taken, because each frame's start line is
        found independently and can be off by a few lines.

        Pictures whose separations differ in content (the solar spectrum is
        one) are not found this way; they can still be combined by hand.
        """
        from scipy.ndimage import gaussian_filter

        quick = DecodeSettings(height=128, field_align=False, margin_lines=0)
        edges = {}
        for p in self.pictures:
            img = gaussian_filter(self.render(p, quick).image, 1.0)
            gy, gx = np.gradient(img)
            e = gaussian_filter(np.hypot(gx, gy), 2.0)
            edges[id(p)] = (e - e.mean()) / (e.std() + 1e-12)

        def similar(p, q):
            a, b = edges[id(p)], edges[id(q)]
            if p.channel != q.channel or min(a.shape[1], b.shape[1]) < 400:
                return False
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])  # lengths differ by a few lines
            a, b = a[:h, :w], b[:h, :w]
            xc = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))) / a.size
            dy, dx = np.r_[0:5, -4:0], np.r_[0:13, -12:0]
            return xc[np.ix_(dy % a.shape[0], dx % a.shape[1])].max() > threshold

        out, ps, i = [], self.pictures, 0
        while i + 2 < len(ps):
            if similar(ps[i], ps[i + 1]) and similar(ps[i + 1], ps[i + 2]):
                out.append((ps[i], ps[i + 1], ps[i + 2]))
                i += 3
            else:
                i += 1
        return out

    # ---------------------------------------------------------------- export
    def export_png(self, r: Rendered, path: str | Path, bits: int = 16) -> Path:
        """Save a rendered picture with square pixels at the settings' aspect."""
        img = r.display()
        return save_png(resize_columns(img, display_width(r.settings, img.shape[1])), path, bits)

    def export_timing_csv(self, r: Rendered, path: str | Path) -> Path:
        """Per-line timing data of a rendered picture (for analysis elsewhere)."""
        path = Path(path)
        tb = r.timebase
        lines = range(r.lines.start, r.lines.stop)
        speed = np.r_[tb.speed_ppm(), np.nan]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["picture_line", "sync_time_s", "sync_measured", "timing_error_raw_us",
                        "timing_error_applied_us", "speed_deviation_ppm"])
            run = self.runs[r.picture.channel][r.picture.run_id]
            for i, k in enumerate(lines):
                w.writerow([i, f"{tb.t_sync[k] / r.fs:.7f}", int(run.measured[k]),
                            f"{tb.error_raw[k] / r.fs * 1e6:.3f}", f"{tb.error_used[k] / r.fs * 1e6:.3f}",
                            f"{speed[k]:.1f}"])
        return path


def display_width(st: DecodeSettings, n_columns: int) -> int:
    """Pixel width that gives square pixels: `aspect` is defined for 512 lines."""
    return int(round(st.height * st.aspect * n_columns / LINES_PER_PICTURE))


def save_png(img: np.ndarray, path: str | Path, bits: int = 16) -> Path:
    """Write a 0..1 grey or RGB image as an 8- or 16-bit PNG (Pillow)."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img, 0, 1)
    if bits == 16 and img.ndim == 2:
        Image.fromarray((img * 65535 + 0.5).astype(np.uint16)).save(path)
    else:
        # Pillow cannot write 16-bit RGB PNGs; colour is saved as 8-bit
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(path)
    return path
