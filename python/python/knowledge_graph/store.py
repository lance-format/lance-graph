"""Persistent storage helpers built on Lance datasets."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

    import pyarrow as pa

    from .config import KnowledgeGraphConfig


LOGGER = logging.getLogger(__name__)


class LanceGraphStore:
    """Manage Lance-backed tables that feed the query engine."""

    def __init__(self, config: "KnowledgeGraphConfig"):
        self._config = config
        self._root: "Path" = config.storage_path
        self._lance: Optional[ModuleType] = None
        self._lance_attempted = False

    @property
    def config(self) -> "KnowledgeGraphConfig":
        """Return the configuration backing this store."""
        return self._config

    @property
    def root(self) -> "Path":
        """Return the root path for persisted datasets."""
        return self._root

    def ensure_layout(self) -> None:
        """Create the storage layout if it does not already exist."""
        self._root.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> Dict[str, "Path"]:
        """Enumerate known Lance datasets."""
        datasets: Dict[str, Path] = {}
        if not self._root.exists():
            return datasets
        for child in self._root.iterdir():
            if child.is_dir() and child.suffix == ".lance":
                datasets[child.stem] = child
        return datasets

    def _dataset_path(self, name: str) -> "Path":
        """Create the canonical path for a dataset."""
        safe_name = name.replace("/", "_")
        return self._root / f"{safe_name}.lance"

    def _get_lance(self) -> ModuleType:
        if not self._lance_attempted:
            self._lance_attempted = True
            try:
                module = import_module("lance")
            except ImportError as e:
                raise ImportError(
                    "Lance module is required but not installed. "
                    "Install it with: pip install pylance"
                ) from e

            has_loader = hasattr(module, "dataset")
            if not (has_loader):
                raise ImportError(
                    "Installed `lance` package is missing required dataset APIs."
                )
            self._lance = module
        if self._lance is None:
            raise ImportError("Lance module failed to load")
        return self._lance

    def load_tables(
        self,
        names: Optional[Iterable[str]] = None,
    ) -> Mapping[str, "pa.Table"]:
        """Load Lance datasets as PyArrow tables."""
        lance = self._get_lance()

        self.ensure_layout()
        available = self.list_datasets()
        requested = list(names) if names is not None else list(available.keys())

        tables: Dict[str, "pa.Table"] = {}
        for name in requested:
            path = available.get(name, self._dataset_path(name))
            if not path.exists():
                raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
            dataset = lance.dataset(str(path))
            table = dataset.scanner().to_table()
            tables[name] = table
        return tables

    def write_tables(self, tables: Mapping[str, "pa.Table"]) -> None:
        """Persist PyArrow tables as Lance datasets."""
        lance = self._get_lance()
        import pyarrow as pa  # Local import; optional dependency

        self.ensure_layout()
        for name, table in tables.items():
            if not isinstance(table, pa.Table):
                raise TypeError(
                    f"Dataset '{name}' must be a pyarrow.Table (got {type(table)!r})"
                )
            path = self._dataset_path(name)
            mode = "overwrite" if path.exists() else "create"
            lance.write_dataset(table, str(path), mode=mode)
