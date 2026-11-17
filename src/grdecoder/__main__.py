"""Entry point:  python -m grdecoder  (GUI)  or  python -m grdecoder decode file.wav"""

import sys

from .cli import main

sys.exit(main())
