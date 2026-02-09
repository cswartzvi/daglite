"""High-level store that handles serialization via Datasets."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, TypeVar, cast

from daglite.datasets.base import AbstractDataset

if TYPE_CHECKING:
    from daglite.drivers.base import Driver

T = TypeVar("T")


class DatasetStore:
    """
    High-level store that handles serialization via Datasets.

    This wraps a Driver (like FileDriver) and adds automatic
    serialization/deserialization using the Dataset registry.

    Format is inferred from the file extension in the key.

    This is the user-facing API - it accepts Python objects and handles
    all serialization internally.

    Examples:
        >>> from daglite.datasets import DatasetStore
        >>> store = DatasetStore("/tmp/outputs")  # doctest: +SKIP
        >>> store.save("data.pkl", {"data": [1, 2, 3]})  # doctest: +SKIP
        >>> store.load("data.pkl", dict)  # doctest: +SKIP
        {'data': [1, 2, 3]}
    """

    def __init__(self, driver: Driver | str) -> None:
        """
        Initialize with a driver.

        Args:
            driver: A Driver instance or string path (creates FileDriver).
        """
        if isinstance(driver, str):
            from daglite.drivers import FileDriver

            self._driver = FileDriver(driver)
        else:
            self._driver = driver

    @property
    def base_path(self) -> str:
        """Get base path (for FileDriver compatibility)."""
        return getattr(self._driver, "base_path", "")

    @property
    def is_local(self) -> bool:
        """Whether the underlying driver accesses local storage."""
        return getattr(self._driver, "is_local", True)

    def save(
        self,
        key: str,
        value: Any,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a value using Dataset serialization.

        Args:
            key: Storage key/path. Format hint from driver (e.g., extension).
            value: Value to serialize and save.
            format: Serialization format. If None, inferred from type and/or driver hint.
            options: Additional options passed to the Dataset's save method.

        Returns:
            The actual path where data was stored.
        """
        value_type = type(value)

        if format is None:
            hint = self._driver.get_format_hint(key)
            format = AbstractDataset.infer_format(value_type, hint)

        dataset_cls = AbstractDataset.get(value_type, format)
        options = options or {}
        dataset = dataset_cls(**options)
        data = dataset.serialize(value)
        return self._driver.save(key, data)

    def load(
        self,
        key: str,
        return_type: type[T] | None = None,
        format: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> T:
        """
        Load a value using Dataset deserialization.

        Args:
            key: Storage key/path. Format hint from driver (e.g., extension).
            return_type: Expected return type. If None, uses pickle format.
            format: Serialization format. If None, inferred from type and/or driver hint.
            options: Additional options passed to the Dataset's load method.

        Returns:
            The deserialized value.

        Raises:
            KeyError: If key not found.
        """
        data = self._driver.load(key)

        if return_type is None and format is None:
            return pickle.loads(data)  # No type hint or format - assume pickle

        hint = self._driver.get_format_hint(key)
        assert return_type is not None
        if format is None:
            format = AbstractDataset.infer_format(return_type, hint)
        dataset_cls = AbstractDataset.get(return_type, format)
        options = options or {}
        dataset = dataset_cls(**options)
        return cast(T, dataset.deserialize(data))

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self._driver.exists(key)

    def delete(self, key: str) -> None:
        """Delete stored data."""
        self._driver.delete(key)

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        return self._driver.list_keys()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize for pickling."""
        return {"driver": self._driver}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Reconstruct from pickled state."""
        self._driver = state["driver"]
