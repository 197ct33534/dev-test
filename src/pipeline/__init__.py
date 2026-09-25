"""Quant Engine pipeline package (Worker 5).

Also re-exports legacy ``src/pipeline.py`` helpers so existing imports like
``from src.pipeline import refresh_data`` keep working after this package
shadows the old module file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from src.data_loader import load_league_data
from src.pipeline.run_pipeline import run_quant_pipeline

__all__ = [
    "run_quant_pipeline",
    "refresh_data",
    "sync_to_global_db",
    "load_league_data",
]


def _load_legacy_pipeline() -> Any:
    """Load sibling ``src/pipeline.py`` under a private module name."""
    legacy_path = Path(__file__).resolve().parent.parent / "pipeline.py"
    mod_name = "src._pipeline_legacy"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    if not legacy_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(mod_name, legacy_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_pipeline()

if _legacy is not None:
    sync_to_global_db: Callable[..., Any] = _legacy.sync_to_global_db

    def refresh_data(*args: Any, **kwargs: Any) -> Any:
        """Delegate to legacy refresh, resolving deps via this package.

        Ensures ``monkeypatch.setattr("src.pipeline.load_league_data", ...)``
        and ``src.pipeline.sync_to_global_db`` still affect the call path after
        the package shadowed ``src/pipeline.py``.
        """
        # Late import avoids circular binding during package init.
        import src.pipeline as pkg

        _legacy.load_league_data = pkg.load_league_data
        _legacy.sync_to_global_db = pkg.sync_to_global_db
        return _legacy.refresh_data(*args, **kwargs)

else:  # pragma: no cover — legacy file always present in this repo

    def refresh_data(*_a: Any, **_k: Any) -> Any:
        raise ImportError("legacy src/pipeline.py not found")

    def sync_to_global_db(*_a: Any, **_k: Any) -> Any:
        raise ImportError("legacy src/pipeline.py not found")
