"""
grdecoder - decode the pictures of the Voyager Golden Record from a
recording of its image-data section.

Modules, in processing order:
    audio      random-access reading of large WAV files
    sync       sync-pulse detection, template matching, line tracking
    pictures   locating the 512-line pictures inside runs of lines
    timebase   wow/flutter correction from the measured sync times
    render     droop correction, sampling, tone mapping, colour
    pipeline   Decoder: ties it all together
    cli        command-line batch decoding
    gui        graphical interface (PySide6 + pyqtgraph)
"""

__version__ = "1.1.1"
