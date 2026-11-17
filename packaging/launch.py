"""Entry script for the PyInstaller build (the .exe starts the GUI)."""

import sys

from grdecoder.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
