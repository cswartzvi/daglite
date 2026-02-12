"""Base class for futures, representing unevaluated task invocations and their configurations."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Generic, ParamSpec, TypeVar
from uuid import UUID
from uuid import uuid4

from typing_extensions import Self, override

from daglite._validation import check_key_template
from daglite.datasets.store import DatasetStore
from daglite.graph.builder import GraphBuilder

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

S1 = TypeVar("S1")
S2 = TypeVar("S2")
S3 = TypeVar("S3")
S4 = TypeVar("S4")
S5 = TypeVar("S5")
S6 = TypeVar("S6")


# region Future Types


@dataclass(frozen=True)
class BaseTaskFuture(abc.ABC, GraphBuilder, Generic[R]):
    """Base class for all task futures, representing unevaluated task invocations."""

    # Internal unique ID for this future
    _id: UUID = field(init=False, repr=False)

    # Future outputs to be saved after execution, accumulated via save() calls.
    _output_futures: tuple[OutputFuture, ...] = field(init=False, repr=False, default=())

    task_store: DatasetStore | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Generate unique ID at creation time."""
        object.__setattr__(self, "_id", uuid4())
        object.__setattr__(self, "_output_futures", ())

    @property
    @override
    def id(self) -> UUID:
        return self._id

    def save(
        self,
        key: str,
        *,
        save_format: str | None = None,
        save_checkpoint: str | bool | None = None,
        save_store: DatasetStore | str | None = None,
        save_options: dict[str, Any] | None = None,
        **extras: Any,
    ) -> Self:
        """
        Save task output for inspection (non-blocking side effect).

        The output will be stored with the given key after task execution completes.
        Can be called multiple times to save to multiple keys.

        Args:
            key: Storage key for this output. Can use {param} format strings which
                will be auto-resolved from task parameters.
            save_checkpoint: If provided, marks this save as a resumption point for
                evaluate(from_=name). Can be:
                - None: No checkpoint (default, just saves output)
                - True: Use the key as the checkpoint name
                - str: Explicit checkpoint name
            save_format: Optional serialization format hint (e.g. "pickle"). If not provided,
                inferred from value type and/or driver hints.
            save_store: Dataset store override for this specific save. If not provided, uses the
                task's default store, then falls back to global settings.
            save_options: Additional options passed to the Dataset's save method.
            **extras: Extra values for key formatting or storage metadata (can include TaskFutures).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no store is configured at any level (explicit, task, or global).

        Examples:
            >>> from daglite import task
            >>> @task
            ... def process(data_id: str) -> Result: ...
            >>> @task
            ... def get_version() -> str: ...

            Simple save (no checkpoint):
            >>> process(data_id="abc123").save(
            ...     "processed_{data_id}_{extra}", extra=get_version()
            ... )  # doctest: +ELLIPSIS
            TaskFuture(...)

            Multiple saves:
            >>> future = process(data_id="abc")
            >>> future.save("v1_{data_id}").save("v2_{data_id}")  # doctest: +ELLIPSIS
            TaskFuture(...)

            Save with checkpoint (using key as name):
            >>> process(data_id="abc").save("output", save_checkpoint=True)  # doctest: +ELLIPSIS
            TaskFuture(...)

            Save with explicit checkpoint name:
            >>> @task
            ... def train_model(model_type: str) -> Model: ...
            >>> train_model(model_type="linear").save(
            ...     "model_{model_type}", save_checkpoint="trained_model"
            ... )  # doctest: +ELLIPSIS
            TaskFuture(...)
        """
        check_key_template(key)

        if isinstance(save_store, str):
            resolved_store = DatasetStore(save_store)
        else:
            resolved_store = save_store or self.task_store

        # Determine checkpoint name
        if save_checkpoint is True:
            checkpoint_name = key
        elif save_checkpoint:
            checkpoint_name = save_checkpoint
        else:
            checkpoint_name = None

        config = OutputFuture(
            key=key,
            name=checkpoint_name,
            format=save_format,
            store=resolved_store,
            options=save_options or {},
            extras=dict(extras),
        )
        new_configs = self._output_futures + (config,)

        # Create a new future with the same ID and updated future outputs
        new_future = replace(self)
        object.__setattr__(new_future, "_id", self._id)
        object.__setattr__(new_future, "_output_futures", new_configs)  # Must match attribute name

        return new_future

    # NOTE: The following methods are to prevent accidental usage of unevaluated nodes.

    def __bool__(self) -> bool:
        raise TypeError(
            "TaskFutures cannot be used in boolean context. Did you mean to call evaluate() first?"
        )

    def __len__(self) -> int:
        raise TypeError("TaskFutures do not support len(). Did you mean to call evaluate() first?")

    def __repr__(self) -> str:  # pragma : no cover
        return f"<Lazy {id(self):#x}>"


@dataclass(frozen=True)
class OutputFuture:
    """Represents a pending output configuration for a task future, to be saved after execution."""

    key: str
    name: str | None
    format: str | None
    store: DatasetStore | None
    options: dict[str, Any] | None
    extras: dict[str, Any]  # Raw values - can be scalars or TaskFutures
