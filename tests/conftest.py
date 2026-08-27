import sys
from pathlib import Path

# The package is imported from the source tree so the suite runs without a colcon build.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
