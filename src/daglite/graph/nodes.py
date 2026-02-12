"""Defines graph nodes for the daglite Intermediate Representation (IR)."""

import inspect
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Callable, TypeVar
from uuid import UUID

from typing_extensions import override

from daglite.backends.context import get_dataset_reporter
from daglite.backends.context import get_event_reporter
from daglite.backends.context import get_plugin_manager
from daglite.backends.context import reset_current_task
from daglite.backends.context import set_current_task
from daglite.datasets.reporters import DirectDatasetReporter
from daglite.datasets.store import DatasetStore
from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.base import BaseGraphNode
from daglite.graph.base import NodeInput
from daglite.graph.base import NodeMetadata
from daglite.graph.base import NodeOutputConfig

_DIRECT_REPORTER = DirectDatasetReporter()

T_co = TypeVar("T_co", covariant=True)

# region Standard Nodes


@dataclass(frozen=True)
class TaskNode(BaseGraphNode):
    """Basic function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for this task node."""

    kwargs: Mapping[str, NodeInput]
    """Keyword parameters from the task function mapped to node inputs."""

    retries: int = 0
    """Number of times to retry the task on failure."""

    cache: bool = False
    """Whether to enable hash-based caching for this task."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds. None means no expiration."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # This is unlikely to happen given retries is checked at task level, but just in case
        assert self.retries >= 0, "Retries must be non-negative"

    @override
    def dependencies(self) -> set[UUID]:
        deps = {p.reference for p in self.kwargs.values() if p.reference is not None}
        for config in self.output_configs:
            for param in config.dependencies.values():
                if param.reference is not None:
                    deps.add(param.reference)
        return deps

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        inputs = {name: param.resolve(completed_nodes) for name, param in self.kwargs.items()}
        return inputs

    @override
    def to_metadata(self) -> NodeMetadata:
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind="task",
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
        )

    @override
    async def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        metadata = replace(self.to_metadata(), key=self.name)
        resolved_output_extras = kwargs.get("resolved_output_deps", [])

        return await _run_implementation(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            output_config=self.output_configs,
            output_deps=resolved_output_extras,
            retries=self.retries,
            cache_enabled=self.cache,
            cache_ttl=self.cache_ttl,
            key_extras={},
        )


@dataclass(frozen=True)
class MapTaskNode(BaseGraphNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: str
    """Mapping mode: 'extend' for Cartesian product, 'zip' for parallel iteration."""

    fixed_kwargs: Mapping[str, NodeInput]
    """Fixed keyword parameters of the task function to node inputs."""

    mapped_kwargs: Mapping[str, NodeInput]
    """Mapped keyword parameters of the task function to node inputs."""

    retries: int = 0
    """Number of times to retry the task on failure."""

    cache: bool = False
    """Whether to enable hash-based caching for this task."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds. None means no expiration."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # This is unlikely to happen given retries is checked at task level, but just in case
        assert self.retries >= 0, "Retries must be non-negative"

    @override
    def dependencies(self) -> set[UUID]:
        deps = set()
        for param in self.fixed_kwargs.values():
            if param.reference is not None:
                deps.add(param.reference)
        for param in self.mapped_kwargs.values():
            if param.reference is not None:
                deps.add(param.reference)
        for config in self.output_configs:
            for param in config.dependencies.values():
                if param.reference is not None:
                    deps.add(param.reference)
        return deps

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        fixed_inputs = {nm: pm.resolve(completed_nodes) for nm, pm in self.fixed_kwargs.items()}
        mapped_inputs = {nm: pm.resolve(completed_nodes) for nm, pm in self.mapped_kwargs.items()}
        inputs = {**fixed_inputs, **mapped_inputs}
        return inputs

    def build_iteration_calls(self, resolved_inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build the list of input dictionaries for each iteration of the mapped node.

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
        """

        from itertools import product

        fixed = {k: v for k, v in resolved_inputs.items() if k in self.fixed_kwargs}
        mapped = {k: v for k, v in resolved_inputs.items() if k in self.mapped_kwargs}

        calls: list[dict[str, Any]] = []

        if self.mode == "product":
            items = list(mapped.items())
            names, lists = zip(*items) if items else ([], [])
            for combo in product(*lists):
                kw = dict(fixed)
                for name, val in zip(names, combo):
                    kw[name] = val
                calls.append(kw)
        elif self.mode == "zip":
            lengths = {len(v) for v in mapped.values()}
            if len(lengths) > 1:
                length_details = {name: len(vals) for name, vals in mapped.items()}
                raise ParameterError(
                    f"Map task '{self.name}' in 'zip' mode requires all sequences to have the "
                    f"same length. Got mismatched lengths: {length_details}. "
                    f"Consider using 'product' mode if you want a Cartesian product instead."
                )
            n = lengths.pop() if lengths else 0
            for i in range(n):
                kw = dict(fixed)
                for name, vs in mapped.items():
                    kw[name] = vs[i]
                calls.append(kw)
        else:
            raise ExecutionError(
                f"Unknown map mode '{self.mode}'. Expected 'product' or 'zip'. "
                f"This indicates an internal error in graph construction."
            )

        return calls

    @override
    def to_metadata(self) -> NodeMetadata:
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind="map",
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
        )

    @override
    async def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace

        iteration_index = kwargs["iteration_index"]
        resolved_output_deps = kwargs.get("resolved_output_deps", [])

        node_key = f"{self.name}[{iteration_index}]"
        metadata = replace(self.to_metadata(), key=node_key)

        return await _run_implementation(
            func=self.func,
            metadata=metadata,
            resolved_inputs=resolved_inputs,
            output_config=self.output_configs,
            output_deps=resolved_output_deps,
            retries=self.retries,
            cache_enabled=self.cache,
            cache_ttl=self.cache_ttl,
            key_extras={"iteration_index": iteration_index},
        )


# region Special Nodes


@dataclass(frozen=True)
class DatasetNode(BaseGraphNode):
    """
    Dataset load node representation within the graph IR.

    Unlike task nodes, this node does not execute a user function.  Instead it
    loads a previously-saved dataset from a :class:`DatasetStore` and returns
    the deserialized value.

    The storage *key* may contain ``{placeholder}`` templates that are resolved
    from dependency values at runtime (exactly like output-save keys).
    """

    store: DatasetStore
    """The dataset store to load from."""

    load_key: str
    """Storage key template (may contain ``{param}`` placeholders)."""

    return_type: type | None = None
    """Expected Python type for deserialization dispatch."""

    load_format: str | None = None
    """Explicit serialization format hint (e.g. ``'pickle'``, ``'pandas/csv'``)."""

    load_options: dict[str, Any] = field(default_factory=dict)
    """Additional options forwarded to the ``Dataset`` constructor."""

    kwargs: Mapping[str, NodeInput] = field(default_factory=dict)
    """Keyword parameters used for key-template formatting."""

    @override
    def dependencies(self) -> set[UUID]:
        deps = {p.reference for p in self.kwargs.values() if p.reference is not None}
        for config in self.output_configs:
            for param in config.dependencies.values():
                if param.reference is not None:
                    deps.add(param.reference)
        return deps

    @override
    def resolve_inputs(self, completed_nodes: Mapping[UUID, Any]) -> dict[str, Any]:
        return {name: param.resolve(completed_nodes) for name, param in self.kwargs.items()}

    @override
    def to_metadata(self) -> NodeMetadata:
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind="dataset",
            description=self.description,
            backend_name=self.backend_name,
            key=self.key,
        )

    @override
    async def run(self, resolved_inputs: dict[str, Any], **kwargs: Any) -> Any:
        from dataclasses import replace as dc_replace

        # Format the key template with resolved dependency values
        try:
            key = self.load_key.format(**resolved_inputs)
        except KeyError as e:
            available = sorted(resolved_inputs.keys())
            raise ValueError(
                f"Dataset key template '{self.load_key}' references {e} "
                f"which is not available.\n"
                f"Available variables: {', '.join(available)}"
            ) from e

        metadata = dc_replace(self.to_metadata(), key=self.name)
        token = set_current_task(metadata)

        try:
            hook = get_plugin_manager().hook
            hook.before_dataset_load(
                key=key,
                return_type=self.return_type,
                format=self.load_format,
                options=self.load_options or None,
            )

            start_time = time.time()
            result = self.store.load(
                key,
                return_type=self.return_type,
                format=self.load_format,
                options=self.load_options or None,
            )
            duration = time.time() - start_time

            hook.after_dataset_load(
                key=key,
                return_type=self.return_type,
                format=self.load_format,
                options=self.load_options or None,
                result=result,
                duration=duration,
            )

            # Save outputs if chained via .save()
            resolved_output_deps = kwargs.get("resolved_output_deps", [])
            _save_outputs(
                result=result,
                resolved_inputs=resolved_inputs,
                output_config=self.output_configs,
                output_deps=resolved_output_deps,
            )

            return result
        finally:
            reset_current_task(token)


# region Common Run


async def _run_implementation(
    func: Callable[..., Any],
    metadata: NodeMetadata,
    resolved_inputs: dict[str, Any],
    output_config: tuple[NodeOutputConfig, ...],
    output_deps: list[dict[str, Any]],
    retries: int = 0,
    cache_enabled: bool = False,
    cache_ttl: int | None = None,
    key_extras: dict[str, Any] | None = None,
) -> Any:
    """
    Private implementation for running a node with context setup and retries.

    Args:
        func: Async function to execute.
        metadata: Metadata for the node being executed.
        resolved_inputs: Pre-resolved parameter inputs for this node.
        output_config: Output configuration tuple for this node.
        output_deps: List of resolved output dependencies for each output config.
        retries: Number of times to retry on failure.
        cache_enabled: Whether caching is enabled for this node.
        cache_ttl: Time-to-live for cache in seconds, if None no expiration.
        key_extras: Additional variables for key formatting.

    Returns:
        Result of the function execution.
    """

    token = set_current_task(metadata)
    hook = get_plugin_manager().hook
    reporter = get_event_reporter()

    # Check cache before execution and return cached result if available
    cached_result = hook.check_cache(
        func=func,
        metadata=metadata,
        inputs=resolved_inputs,
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
    )
    if cached_result is not None:
        result = (
            cached_result["value"]
            if isinstance(cached_result, dict) and "value" in cached_result
            else cached_result
        )
        hook.on_cache_hit(
            func=func,
            metadata=metadata,
            inputs=resolved_inputs,
            result=result,
            reporter=reporter,
        )
        reset_current_task(token)
        return result

    hook.before_node_execute(metadata=metadata, inputs=resolved_inputs, reporter=reporter)

    last_error: Exception | None = None
    attempt, max_attempts = 0, retries + 1
    start_time = time.time()

    try:
        # Main execution loop with retry handling
        while attempt < max_attempts:  # pragma: no branch
            attempt += 1
            try:
                if attempt > 1:
                    assert last_error is not None
                    hook.before_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        reporter=reporter,
                        attempt=attempt,
                        last_error=last_error,
                    )

                # Synchronous/asynchronous function handling
                if inspect.iscoroutinefunction(func):
                    result = await func(**resolved_inputs)
                else:
                    result = func(**resolved_inputs)

                duration = time.time() - start_time

                if attempt > 1:
                    hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        reporter=reporter,
                        attempt=attempt,
                        succeeded=True,
                    )
                hook.after_node_execute(
                    metadata=metadata,
                    inputs=resolved_inputs,
                    result=result,
                    duration=duration,
                    reporter=reporter,
                )
                hook.update_cache(
                    func=func,
                    metadata=metadata,
                    inputs=resolved_inputs,
                    result=result,
                    cache_enabled=cache_enabled,
                    cache_ttl=cache_ttl,
                )

                # Save outputs via dataset reporter if configured
                _save_outputs(
                    result=result,
                    resolved_inputs=resolved_inputs,
                    output_config=output_config,
                    output_deps=output_deps,
                    key_extras=key_extras or {},
                )

                return result

            except Exception as error:
                # Determine if a retry is available
                last_error = error

                if attempt > 1:
                    hook.after_node_retry(
                        metadata=metadata,
                        inputs=resolved_inputs,
                        reporter=reporter,
                        attempt=attempt,
                        succeeded=False,
                    )

                if attempt >= max_attempts:
                    break  # No more retries left

        # All attempts exhausted
        duration = time.time() - start_time
        assert last_error is not None
        hook.on_node_error(
            metadata=metadata,
            inputs=resolved_inputs,
            reporter=reporter,
            error=last_error,
            duration=duration,
        )
        raise last_error

    finally:
        reset_current_task(token)


def _save_outputs(
    result: Any,
    resolved_inputs: dict[str, Any],
    output_config: tuple[NodeOutputConfig, ...],
    output_deps: list[dict[str, Any]],
    key_extras: dict[str, Any] | None = None,
) -> None:
    """
    Save task outputs via the dataset reporter or directly.

    For each output config the store is resolved (explicit store on the config,
    or the settings-level default).  Routing then depends on the driver's
    locality:

    * **Local drivers** (filesystem, SQLite) – save through the
      ``DatasetReporter`` so that the coordinator process performs the write.
    * **Remote drivers** (S3, GCS, …) – save directly from the worker since
      the remote store is accessible everywhere.

    Exceptions are **not** caught – a failed save fails the task.

    Args:
        result: The task execution result to save.
        resolved_inputs: Resolved inputs for key formatting.
        output_config: Output configuration tuple for this node.
        output_deps: List of resolved output dependencies for each output config.
        key_extras: Additional variables for key formatting.
    """
    from daglite.graph.base import NodeOutputConfig
    from daglite.settings import get_global_settings

    if not output_config:
        return

    dataset_reporter = get_dataset_reporter()

    for idx, config in enumerate(output_config):
        if not isinstance(config, NodeOutputConfig):  # pragma: no cover
            continue

        # Resolve the target store: explicit config → settings default
        store = config.store
        if store is None:
            settings = get_global_settings()
            store_or_path = settings.datastore_store
            if isinstance(store_or_path, str):
                from daglite.datasets.store import DatasetStore

                store = DatasetStore(store_or_path)
            else:
                store = store_or_path

        # Build the storage key by formatting with extras and metadata
        key_extras = key_extras or {}
        format_vars = {**resolved_inputs, **output_deps[idx], **key_extras}
        try:
            key = config.key.format(**format_vars)
        except KeyError as e:
            available = sorted(format_vars.keys())
            raise ValueError(
                f"Output key template '{config.key}' references {e} which is not available.\n"
                f"Available variables: {', '.join(available)}\n"
                f"These come from: task inputs, extra dependencies passed to .save(), "
                f"and internal key extras."
            ) from e

        # Replace the reporter if the store is remote since the worker can write directly
        reporter = dataset_reporter if dataset_reporter and store.is_local else _DIRECT_REPORTER
        reporter.save(key, result, store, format=config.format, options=config.options)
