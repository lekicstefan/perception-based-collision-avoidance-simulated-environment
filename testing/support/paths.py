"""Paths shared by tests and tools. Nothing here depends on where a test file sits."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]     # testing/support/paths.py -> repository root
TESTING_DIR = REPO_ROOT / "testing"
DATA_DIR = TESTING_DIR / "data"                     # recorded and generated data used by tests and tools
RECORDINGS_DIR = DATA_DIR / "recordings"            # development recordings (phase 3)
RUNS_DIR = REPO_ROOT / "runs"                       # run folders written by Unity and the processor