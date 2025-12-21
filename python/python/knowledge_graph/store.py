"""Persistent storage helpers built on Lance datasets."""

from __future__ import annotations

import logging
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional

import fsspec
import pyarrow as pa

if TYPE_CHECKING:
    from types import ModuleType

    from .config import KnowledgeGraphConfig


LOGGER = logging.getLogger(__name__)


class LanceGraphStore:
    """Manage Lance-backed tables that feed the query engine."""

    def __init__(self, config: "KnowledgeGraphConfig"):
        self._config = config
        self._root: Path | str = config.storage_path
        self._lance: Optional[ModuleType] = None
        self._lance_attempted = False

        # Initialize filesystem interface
        # We convert to string to ensure compatibility with fsspec, but we'll
        # use self._root (the original type) when reconstructing return values.
        try:
            self._fs, self._fs_path = fsspec.core.url_to_fs(
                str(self._root), **(self.config.storage_options or {})
            )
        except ImportError:
            # Re-raise explicit ImportError if protocol driver (e.g. gcsfs, s3fs)
            # is missing
            raise

    @property
    def config(self) -> "KnowledgeGraphConfig":
        """Return the configuration backing this store."""
        return self._config

    @property
    def root(self) -> Path | str:
        """Return the root path for persisted datasets."""
        return self._root

    def ensure_layout(self) -> None:
        """Create the storage layout if it does not already exist."""
        try:
            self._fs.makedirs(self._fs_path, exist_ok=True)
        except Exception:
            # S3/GCS might not support directory creation or it might be implicit.
            # We treat failure here as non-fatal if the path is actually accessible
            # later,
            # but usually makedirs is safe on object stores (no-op).
            pass

    def list_datasets(self) -> Dict[str, Path | str]:
        """Enumerate known Lance datasets."""
        datasets: Dict[str, Path | str] = {}

        try:
            if not self._fs.exists(self._fs_path):
                return datasets
            infos = self._fs.ls(self._fs_path, detail=True)
        except Exception as e:
            # We want to swallow "not found" errors but raise others (like Auth errors)
            if isinstance(e, FileNotFoundError):
                return datasets

            msg = str(e).lower()
            if "not found" in msg or "no such file" in msg or "does not exist" in msg:
                return datasets
            raise

        root_str = str(self._root)
        for info in infos:
            name = info["name"].rstrip("/")
            base_name = name.split("/")[-1]
            if info["type"] == "directory" and base_name.endswith(".lance"):
                dataset_name = base_name[:-6]
                full_path = f"{root_str.rstrip('/')}/{base_name}"
                if isinstance(self._root, Path):
                    datasets[dataset_name] = Path(full_path)
                else:
                    datasets[dataset_name] = full_path
        return datasets

    def _dataset_path(self, name: str) -> Path | str:
        """Create the canonical path for a dataset."""
        safe_name = name.replace("/", "_")
        if isinstance(self._root, Path):
            return self._root / f"{safe_name}.lance"
        return f"{self._root.rstrip('/')}/{safe_name}.lance"

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

    def _path_exists(self, path: Path | str) -> bool:
        if isinstance(path, Path):
            return path.exists()
        try:
            fs, p = fsspec.core.url_to_fs(path)
        except Exception:
            # If we cannot resolve the filesystem (e.g. missing gcsfs), we should raise
            # rather than assuming the path does not exist.
            raise
        try:
            return fs.exists(p)
        except Exception:
            return False

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
            if not self._path_exists(path):
                raise FileNotFoundError(f"Dataset '{name}' not found at {path}")
            dataset = lance.dataset(
                str(path), storage_options=self.config.storage_options
            )
            table = dataset.scanner().to_table()
            tables[name] = table
        return tables

    def write_tables(self, tables: Mapping[str, "pa.Table"]) -> None:
        """Persist PyArrow tables as Lance datasets."""
        lance = self._get_lance()

        self.ensure_layout()
        for name, table in tables.items():
            if not isinstance(table, pa.Table):
                raise TypeError(
                    f"Dataset '{name}' must be a pyarrow.Table (got {type(table)!r})"
                )
            path = self._dataset_path(name)
            mode = "overwrite" if self._path_exists(path) else "create"
            lance.write_dataset(
                table, str(path), mode=mode, storage_options=self.config.storage_options
            )
