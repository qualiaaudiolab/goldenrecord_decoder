"""
cli.py - batch decoding from the command line.

    python -m grdecoder decode recording.wav --out pictures/

writes every picture as a 16-bit PNG, colour pictures as RGB PNGs, a
per-picture timing CSV, and a summary report.csv. Run with --help for all
options; every option maps to a field of pipeline.DecodeSettings.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from .pipeline import DecodeSettings, Decoder, SyncSettings, display_width, save_png
from .render import auto_levels, combine_colour, resize_columns, tone_map


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="grdecoder", description="Voyager Golden Record picture decoder")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("gui", help="start the graphical interface (default)")

    t = sub.add_parser("selftest", help="check that all components load (writes a JSON report)")
    t.add_argument("--out", type=Path, default=Path("grdecoder_selftest.json"))

    d = sub.add_parser("decode", help="decode all pictures of a recording to PNG files")
    d.add_argument("wav", type=Path, help="recording of the picture section (any rate, mono or stereo)")
    d.add_argument("--out", type=Path, default=Path("decoded"), help="output folder")
    d.add_argument("--channels", type=int, nargs="+", default=None, help="channels to decode (0=left, 1=right)")
    d.add_argument("--no-colour", action="store_true", help="do not combine colour separations")
    d.add_argument("--timing-csv", action="store_true", help="also write per-line timing CSV per picture")
    for f in fields(SyncSettings):  # sync detection fine-tuning: --sync-min-score etc.
        v = getattr(SyncSettings(), f.name)
        d.add_argument("--sync-" + f.name.replace("_", "-"), dest="sync_" + f.name, type=type(v), default=v,
                       help=f"sync detection (default {v})")
    defaults = DecodeSettings()
    for f in fields(DecodeSettings):
        v = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(v, bool):
            d.add_argument(flag, type=_bool, default=v, metavar="yes|no", help=f"(default {'yes' if v else 'no'})")
        elif v is None:
            d.add_argument(flag, type=float, default=None, help="(default: automatic)")
        else:
            d.add_argument(flag, type=type(v), default=v, help=f"(default {v})")
    return p


def _bool(s: str) -> bool:
    return s.lower() in ("1", "yes", "y", "true", "on")


def decode(args) -> int:
    st = DecodeSettings(**{f.name: getattr(args, f.name) for f in fields(DecodeSettings)})
    sync_st = SyncSettings(**{f.name: getattr(args, "sync_" + f.name) for f in fields(SyncSettings)})
    dec = Decoder(args.wav, sync_settings=sync_st)
    info = dec.source.info
    print(f"{info.path.name}: {info.samplerate} Hz, {info.channels} ch, {info.duration:.1f} s, {info.subtype}")
    channels = args.channels if args.channels is not None else list(range(min(2, info.channels)))

    t0 = time.time()
    last = [0.0]

    def progress(f, msg):
        if time.time() - last[0] > 0.5:
            print(f"\r  analysing: {f * 100:5.1f} %  {msg:<40}", end="", flush=True)
            last[0] = time.time()

    pics = dec.analyse(channels, progress)
    print(f"\r  found {len(pics)} pictures in {time.time() - t0:.1f} s" + " " * 40)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pics:
        r = dec.render(p, st)
        path = dec.export_png(r, out / f"{p.name}.png")
        if args.timing_csv:
            dec.export_timing_csv(r, out / f"{p.name}_timing.csv")
        e = r.timing_error_us("raw")
        rows.append([p.name, "LR"[p.channel], p.number, f"{p.t_start:.3f}", p.n_lines, f"{p.sync_found:.4f}",
                     f"{r.speed_vs_nominal_ppm:.0f}", f"{np.std(e):.2f}", f"{np.ptp(e):.1f}",
                     f"{r.droop_tau_ms:.3f}" if r.droop_tau_ms else "", f"{r.field_offset_rows:.2f}", path.name])
        print(f"  {p.name}  t={p.t_start:7.2f}s  speed {r.speed_vs_nominal_ppm:+6.0f} ppm  "
              f"timing err {np.std(e):5.1f} us rms  -> {path.name}")

    if not args.no_colour:
        for a, b, c in dec.colour_triples():
            imgs = [dec.render(p, st).image for p in (a, b, c)]
            rgb, shifts = combine_colour(*imgs)
            lo, hi = auto_levels(rgb, st.black_pct, st.white_pct)
            rgb = tone_map(rgb, lo, hi, st.gamma, st.invert)
            width = display_width(st, rgb.shape[1])
            name = f"{a.name}-{c.name}_colour.png"
            save_png(resize_columns(rgb, width), out / name, bits=8)
            print(f"  colour: {a.name}+{b.name}+{c.name} (R, G, B) -> {name}")

    with open(out / "report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["picture", "channel", "number", "start_s", "lines", "syncs_measured", "speed_vs_120Hz_ppm",
                    "timing_error_rms_us", "timing_error_pp_us", "droop_tau_ms", "field_offset_rows", "file"])
        w.writerows(rows)
    print(f"done in {time.time() - t0:.1f} s -> {out.resolve()}")
    return 0


def selftest(args) -> int:
    """Import every component and record what works (used by the Windows build check)."""
    import json
    import platform

    report = {"python": platform.python_version(), "platform": platform.platform()}

    def check(name, fn):
        try:
            report[name] = fn()
        except Exception as ex:  # record, do not stop
            report[name] = f"ERROR: {ex!r}"

    check("numpy", lambda: __import__("numpy").__version__)
    check("scipy", lambda: __import__("scipy").__version__)
    check("soundfile", lambda: __import__("soundfile").__libsndfile_version__)
    check("qt", lambda: __import__("PySide6.QtWidgets", fromlist=["x"]) and __import__("PySide6").__version__)
    check("pyqtgraph", lambda: __import__("pyqtgraph").__version__)
    check("sounddevice", lambda: __import__("sounddevice").get_portaudio_version()[1])
    check("audio_outputs", lambda: sum(1 for d in __import__("sounddevice").query_devices()
                                       if d["max_output_channels"] > 0))
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if not any(str(v).startswith("ERROR") for k, v in report.items() if k != "audio_outputs") else 1


def main(argv=None) -> int:
    # a windowed .exe has no console: give print() somewhere harmless to write
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    argv = sys.argv[1:] if argv is None else list(argv)
    # anything that is not a sub-command (e.g. a file dropped onto the app) -> GUI
    if argv and argv[0] not in ("gui", "decode", "selftest", "-h", "--help"):
        from .gui.app import run

        return run([sys.argv[0], *argv])
    args = build_parser().parse_args(argv)
    if args.command == "decode":
        return decode(args)
    if args.command == "selftest":
        return selftest(args)
    from .gui.app import run  # imported lazily: the CLI must work without Qt

    return run([sys.argv[0]])


if __name__ == "__main__":
    sys.exit(main())
