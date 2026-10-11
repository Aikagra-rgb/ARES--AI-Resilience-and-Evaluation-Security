import os
import sys
from pathlib import Path

# Test-suite environment. These must be set BEFORE any `ares`/`app` module is imported, because both
# settings objects are built at import time. Production-grade defaults (auth on, no header-only
# identity) are exercised explicitly in test_security.py by patching the live settings objects.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("ARES_AUTH_ENABLED", "false")
os.environ.setdefault("ALLOW_DEV_IDENTITY", "true")

# Add backend directory to sys.path so 'ares' can be imported
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
