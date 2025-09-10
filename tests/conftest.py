# tests/conftest.py
# Ensures imports like `from src.baselines import ...` work when running pytest
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
