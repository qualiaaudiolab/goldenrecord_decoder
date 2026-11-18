## New in 1.1

* Sync detection works at any sample rate and on band-limited copies (44.1 kHz, 20 kHz low-pass): pulse templates are learned from the recording.
* New **Sync detection** settings (template length, minimum match, edge scale) with a **Re-analyse** button, and a **Sync template** diagnostics tab.

## Install (Windows)

1. Download **GoldenRecordDecoder-…-windows.zip** below (under *Assets*).
2. Right-click the zip → **Extract All…** → choose a permanent folder (e.g. in Documents).
3. Open the folder and double-click **GoldenRecordDecoder.exe**. Keep the `_internal` folder next to it.
4. If Windows shows "Windows protected your PC" (the program is not digitally signed): **More info → Run anyway**.

## Use

**Open…** (or drag a WAV/FLAC onto the window) → the pictures appear on the left → click one to decode it.
Defaults are the recommended settings; **A/B: uncorrected** shows the plain decode for comparison.

See the README and `docs/HOW_IT_WORKS.md` in the repository for details.
