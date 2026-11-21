import sys
from pathlib import Path

# Add project root to sys.path so 'src' is importable in all tests
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
