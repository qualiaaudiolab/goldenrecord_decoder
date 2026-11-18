"""
End-to-end tests on a synthetic recording with known faults (see synthetic.py).

Run with:   python -m pytest -q
"""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from grdecoder.pipeline import DecodeSettings, Decoder
from grdecoder.timebase import build_timebase, destagger
from synthetic import make_recording


@pytest.fixture(scope="module")
def decoded(tmp_path_factory):
    y, truth = make_recording()
    path = tmp_path_factory.mktemp("rec") / "synthetic.wav"
    sf.write(path, y, truth["fs"], subtype="FLOAT")
    dec = Decoder(path)
    dec.analyse(channels=(0,))
    return dec, truth


def test_finds_one_picture_of_512_lines(decoded):
    dec, truth = decoded
    assert len(dec.pictures) == 1
    p = dec.pictures[0]
    # the cover's 512 lines; the decoder keeps the whole framed area, which the
    # finder locates to within a few lines of its end
    assert 512 <= p.n_lines <= 516
    # the picture starts exactly where we put it (right after the start marker)
    run = dec.runs[0][p.run_id]
    first_found = int(round((run.times[p.first_line] / truth["fs"]) * 120))
    assert first_found == truth["lines_before"]


def test_sync_times_match_truth(decoded):
    """Sync positions, after removing the stagger, follow the injected warp to < 0.5 us rms."""
    dec, truth = decoded
    run = dec.runs[0][dec.pictures[0].run_id]
    t = destagger(run.times).times / truth["fs"]
    q = np.arange(len(t))
    inj = np.interp(t, truth["warp_t"], truth["warp"])
    res = (t - np.polyval(np.polyfit(q, t, 1), q)) - (inj - np.polyval(np.polyfit(q, inj, 1), q))
    assert res.std() < 0.5e-6, res.std()


def test_line_period_and_stagger(decoded):
    dec, truth = decoded
    P = dec.period[0]
    assert abs(P / (truth["fs"] / 120) - 1) < 1e-3
    ds = destagger(dec.runs[0][dec.pictures[0].run_id].times)
    assert abs(abs(ds.stagger) - truth["stagger"] * P) < 1.0  # within one sample


def test_timing_error_matches_injected_wow(decoded):
    dec, truth = decoded
    run = dec.runs[0][dec.pictures[0].run_id]
    tb = build_timebase(destagger(run.times).times, run.measured, "line", fs=truth["fs"])
    t_sync = run.times / truth["fs"]
    injected = np.interp(t_sync, truth["warp_t"], truth["warp"])
    injected = injected - np.polyval(np.polyfit(t_sync, injected, 1), t_sync)
    measured = tb.error_raw / truth["fs"]
    # measured error is tape position behind real time -> same sign as the warp
    r = np.corrcoef(measured, injected)[0, 1]
    assert r > 0.98, r


def test_droop_time_constant_estimate(decoded):
    dec, truth = decoded
    r = dec.render(dec.pictures[0], DecodeSettings(height=256))
    assert r.droop_tau_ms is not None
    assert abs(r.droop_tau_ms / (truth["tau"] * 1e3) - 1) < 0.25


def _column_spread(img):
    """How much the columns differ (truth: all columns identical).

    The outer 12 columns are left out: the picture finder may include a
    line or two of blank border there.
    """
    img = img[:, 12:-12]
    return float(np.std(img - img.mean(axis=1, keepdims=True)) / np.std(img))


def test_sync_locked_beats_fixed_speed(decoded):
    dec, _ = decoded
    p = dec.pictures[0]
    st = DecodeSettings(height=512, margin_lines=0, invert=False)
    smooth = _column_spread(dec.render(p, st).image)
    fixed = _column_spread(dec.render(p, replace(st, mode="fixed")).image)
    assert smooth < 0.5 * fixed, (smooth, fixed)
    assert smooth < 0.15


def test_render_shapes(decoded):
    dec, _ = decoded
    st = DecodeSettings(height=300, margin_lines=4)
    r = dec.render(dec.pictures[0], st)
    assert r.image.shape == (300, dec.pictures[0].n_lines + 8)
    disp = r.display()
    assert disp.min() >= 0 and disp.max() <= 1


@pytest.mark.parametrize("variant", ["44.1 kHz", "lowpass 15 kHz"])
def test_other_copies(tmp_path, variant):
    """The decoder must also work on a 44.1 kHz copy and on a band-limited copy."""
    import scipy.signal as ss

    y, truth = make_recording()
    fs = truth["fs"]
    if variant == "44.1 kHz":
        y = ss.resample_poly(y, 147, 320).astype(np.float32)
        fs = 44100
    else:
        y = ss.sosfiltfilt(ss.butter(8, 15000, fs=fs, output="sos"), y).astype(np.float32)
    path = tmp_path / "copy.wav"
    sf.write(path, y, fs, subtype="FLOAT")
    dec = Decoder(path)
    dec.analyse(channels=(0,))
    assert len(dec.pictures) == 1
    p = dec.pictures[0]
    assert p.sync_found > 0.98
    run = dec.runs[0][p.run_id]
    t = destagger(run.times).times / fs
    q = np.arange(len(t))
    inj = np.interp(t, truth["warp_t"], truth["warp"])
    res = (t - np.polyval(np.polyfit(q, t, 1), q)) - (inj - np.polyval(np.polyfit(q, inj, 1), q))
    assert res.std() < 2e-6, res.std()


@pytest.mark.parametrize("start", [0, 1])
def test_field_alignment(tmp_path, start):
    """Interlaced fields offset by ~2 rows are measured and brought into line,
    whether the picture starts on an even or an odd line of its run."""
    from grdecoder.render import measure_field_offset

    shift = 0.002  # fraction of a line; ~1.9 rows at 768 rows over 0.8 of a line
    y, truth = make_recording(field_shift=shift, square=True, wow=(), tau=None)
    path = tmp_path / "fields.wav"
    sf.write(path, y, truth["fs"], subtype="FLOAT")
    dec = Decoder(path)
    dec.analyse(channels=(0,))
    p = dec.pictures[0]
    # move the picture start by one line to test both parities
    p = replace(p, first_line=p.first_line + start, n_lines=p.n_lines - start)
    off = dec.render(p, DecodeSettings(field_align=False, field_level=False, droop=False))
    on = dec.render(p, DecodeSettings(field_align=True, field_level=False, droop=False))
    assert abs(measure_field_offset(off.image)) > 1.4
    assert abs(measure_field_offset(on.image)) < 0.3
