import sys
from pathlib import Path

# Add the repo root to sys.path so that 'src' is importable from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))