## Golden Record Decoder 1.2

Decodes the pictures of the Voyager Golden Record from a recording of its picture section, with the tape's timing errors corrected line by line.

* Finds every picture on both channels and locks to every sync pulse, learning the pulse shape from the recording itself, so 384 kHz, 44.1 kHz and band-limited copies all decode.
* Corrects wow and flutter, the staggered sync of the interlaced fields, low-frequency loss ("droop") and line-to-line level jumps.
* Calibrates the aspect ratio on the calibration circle and combines colour separations.
* Plays the recording while the picture is drawn line by line in step with the sound.
* Shows its working: timing error, speed, line scope, picture finder and sync template.

The recording is not included. See the README and `docs/HOW_IT_WORKS.md` for details.

## Install (Windows)

1. Download **GoldenRecordDecoder-…-windows.zip** below (under *Assets*).
2. Right-click the zip → **Extract All…** → choose a permanent folder (e.g. in Documents).
3. Open the folder and double-click **GoldenRecordDecoder.exe**. Keep the `_internal` folder next to it.
4. If Windows shows "Windows protected your PC" (the program is not digitally signed): **More info → Run anyway**.

## Use

**Open…** (or drag a WAV/FLAC onto the window) → the pictures appear on the left → click one to decode it.
Defaults are the recommended settings; **A/B: uncorrected** shows the plain decode for comparison.
