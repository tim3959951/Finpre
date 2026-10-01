"""Fintech Agent: multi-agent financial analysis with TimesFM forecasting."""
import os as _os
import sys as _sys

if _sys.platform == "darwin":
    # On macOS torch, LightGBM (Homebrew) and scikit-learn each load their own libomp. Mixing their thread
    # pools deadlocks once the agents run in worker threads, so default to one OpenMP thread (matrix
    # multiplies still use Accelerate). Must run before torch / lightgbm are imported; override via env.
    _os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.0.0"
