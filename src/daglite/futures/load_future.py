"""Contains representations of dataset load futures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from typing_extensions import override

from daglite._validation import check_key_placeholders
from daglite._validation import check_key_template
from daglite.datasets.store import DatasetStore
from daglite.futures._shared import build_node_inputs
from daglite.futures._shared import build_output_configs
from daglite.futures.task_future import TaskFuture
from daglite.graph.nodes import DatasetNode
from daglite.utils import build_repr

R = TypeVar("R")


def load_dataset(
    key: str,
    *,
    load_type: type[R] | None = None,
    load_format: str | None = None,
    load_store: DatasetStore | str | None = None,
    load_options: dict[str, Any] | None = None,
    **extras: Any,
) -> DatasetFuture[R]:
    """
    Creates a lazy dataset load that will be resolved during graph evaluation.

    The returned `DatasetFuture` participates in the DAG just like a `TaskFuture` — it can be
    chained with with all other methods of the fluent API, including.

    Args:
        key: Storage key (or template with `{param}` placeholders).
        load_type: Expected Python return type for deserialization dispatch.
        load_format: Explicit serialization format hint (e.g. `"pickle"`).
        load_store: `DatasetStore` or path string.  Falls back to global settings.
        load_options: Extra options forwarded to the `Dataset` constructor.
        **extras: Values (or futures) for key-template formatting.

    Returns:
        A `DatasetFuture` that produces the loaded value when evaluated.

    Examples:
        >>> from daglite import load_dataset, evaluate, task
        >>> @task
        ... def process(data: dict) -> str:
        ...     return str(data)
        >>> future = load_dataset("input.pkl").then(process)
        >>> evaluate(future)  # doctest: +SKIP
    """
    from daglite.tasks import Task as _Task

    check_key_template(key)

    if isinstance(load_store, str):
        resolved_store: DatasetStore | None = DatasetStore(load_store)
    else:
        resolved_store = load_store

    stub_task = _Task(
        func=_dataset_load_stub,
        name=f"load({key})",
        description="Dataset load stub",
        backend_name=None,
    )

    return DatasetFuture(
        task=stub_task,
        kwargs=dict(extras),
        backend_name=None,
        load_key=key,
        load_store=resolved_store,
        load_type=load_type,
        load_format=load_format,
        load_options=load_options or {},
    )


@dataclass(frozen=True)
class DatasetFuture(TaskFuture[R]):
    """Represents a lazy dataset load that will produce a value of type R when evaluated."""

    load_key: str
    """Storage key template (may contain `{param}` placeholders)."""

    load_store: DatasetStore | None
    """Explicit store override.  Falls back to global settings when `None`."""

    load_type: type[R] | None
    """Expected Python type for deserialization dispatch."""

    load_format: str | None
    """Explicit serialization format hint."""

    load_options: dict[str, Any]
    """Additional options forwarded to the `Dataset` constructor."""

    def __repr__(self) -> str:
        return build_repr("DatasetFuture", f"key={self.load_key!r}", kwargs=self.kwargs)

    @override
    def build_node(self) -> DatasetNode:  # type: ignore[override]
        available_names = set(self.kwargs.keys())
        check_key_placeholders(self.load_key, available_names)

        kwargs = build_node_inputs(self.kwargs)
        output_configs = build_output_configs(self._output_futures, available_names)
        return DatasetNode(
            id=self.id,
            name=f"load({self.load_key})",
            store=self.load_store or self._resolve_default_store(),
            load_key=self.load_key,
            return_type=self.load_type,
            load_format=self.load_format,
            load_options=self.load_options or {},
            kwargs=kwargs,
            output_configs=tuple(output_configs),
        )

    @staticmethod
    def _resolve_default_store() -> DatasetStore:
        """Fall back to the global settings datastore."""
        from daglite.settings import get_global_settings

        store_or_path = get_global_settings().datastore_store
        if isinstance(store_or_path, str):
            return DatasetStore(store_or_path)
        return store_or_path


def _dataset_load_stub() -> Any:
    """Placeholder performs the actual load at runtime."""
    raise AssertionError("_dataset_load_stub should never be called directly")
