import sys
from pathlib import Path

# make `grdecoder` (in src/) and the synthetic-signal helper importable without installing
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
