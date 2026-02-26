"""
Worker-side execution functions for graph nodes.

These functions run on the backend worker (inline, thread, or process) and handle context setup,
hook calls, retries, caching, generator materialization, and output saving.
"""

# NOTE Functions should be defined at module level to be picklable for process backends.

from __future__ import annotations

import inspect
import time
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Generator
from collections.abc import Iterator
from typing import Any, Callable

from daglite.backends.context import get_cache_store
from daglite.backends.context import get_dataset_reporter
from daglite.backends.context import get_event_reporter
from daglite.backends.context import get_plugin_manager
from daglite.backends.context import reset_current_task
from daglite.backends.context import set_current_task
from daglite.datasets.reporters import DirectDatasetReporter
from daglite.datasets.store import DatasetStore
from daglite.graph.nodes.base import NodeMetadata
from daglite.graph.nodes.base import NodeOutputConfig

_DIRECT_REPORTER = DirectDatasetReporter()


async def run_task_worker(
    func: Callable[..., Any],
    metadata: NodeMetadata,
    inputs: dict[str, Any],
    output_configs: tuple[NodeOutputConfig, ...],
    output_parameters: list[dict[str, Any]],
    retries: int = 0,
    cache_enabled: bool = False,
    cache_ttl: int | None = None,
    cache_hash_fn: Callable[..., str] | None = None,
    key_extras: dict[str, Any] | None = None,
    iteration_index: int | None = None,
) -> Any:
    """
    Execute a task function on a backend worker and persist outputs.

    Args:
        func: The task function to execute (sync or async).
        metadata: Metadata for the node being executed.
        inputs: Dictionary of resolved input values for the function execution.
        output_configs: Output configuration tuple for this node.
        output_parameters: List of resolved output parameters for each output config.
        retries: Number of times to retry on failure.
        cache_enabled: Whether caching is enabled for this node.
        cache_ttl: Time-to-live for cache in seconds, if None no expiration.
        cache_hash_fn: Custom hash function ``(func, inputs) -> str``. If None,
            the built-in ``default_cache_hash`` is used.
        key_extras: Additional variables for key formatting.
        iteration_index: Optional index for map iterations, used for metadata key formatting.

    Returns:
        Result of the function execution.
    """
    # Build effective key_extras once, including iteration_index when present,
    # so both _run_task_func and _save_outputs see the same variables.
    effective_extras = dict(key_extras) if key_extras else {}
    if iteration_index is not None:
        effective_extras["iteration_index"] = iteration_index

    result = await _run_task_func(
        func=func,
        metadata=metadata,
        inputs=inputs,
        retries=retries,
        cache_enabled=cache_enabled,
        cache_ttl=cache_ttl,
        cache_hash_fn=cache_hash_fn,
        key_extras=effective_extras,
        iteration_index=iteration_index,
    )
    _save_outputs(
        result=result,
        resolved_inputs=inputs,
        output_config=output_configs,
        output_deps=output_parameters,
        key_extras=effective_extras,
    )
    return result


async def load_dataset_worker(
    store: DatasetStore,
    load_key: str,
    return_type: type | None,
    load_format: str | None,
    load_options: dict[str, Any],
    metadata: NodeMetadata,
    resolved_inputs: dict[str, Any],
    output_configs: tuple[NodeOutputConfig, ...],
    output_parameters: list[dict[str, Any]],
) -> Any:
    """
    Load a dataset from a store on the backend worker with lifecycle hooks.

    Args:
        store: The dataset store to load from.
        load_key: Storage key template (may contain ``{param}`` placeholders).
        return_type: Expected Python type for deserialization dispatch.
        load_format: Explicit serialization format hint.
        load_options: Additional options forwarded to the Dataset constructor.
        metadata: Metadata for the node being executed.
        resolved_inputs: Resolved inputs for key-template formatting.
        output_configs: Output configuration tuple for this node.
        output_parameters: Resolved output parameters for each output config.

    Returns:
        The deserialized dataset value.
    """
    from dataclasses import replace as dc_replace

    # Format the key template with resolved dependency values
    try:
        key = load_key.format(**resolved_inputs)
    except KeyError as e:
        available = sorted(resolved_inputs.keys())
        raise ValueError(
            f"Dataset key template '{load_key}' references {e} "
            f"which is not available.\n"
            f"Available variables: {', '.join(available)}"
        ) from e

    node_metadata = dc_replace(metadata, key=metadata.name)
    token = set_current_task(node_metadata)

    try:
        hook = get_plugin_manager().hook
        hook.before_dataset_load(
            key=key,
            return_type=return_type,
            format=load_format,
            options=load_options or None,
        )

        start_time = time.time()
        result = store.load(
            key,
            return_type=return_type,
            format=load_format,
            options=load_options or None,
        )
        duration = time.time() - start_time

        hook.after_dataset_load(
            key=key,
            return_type=return_type,
            format=load_format,
            options=load_options or None,
            result=result,
            duration=duration,
        )

        # Save outputs if chained via .save()
        _save_outputs(
            result=result,
            resolved_inputs=resolved_inputs,
            output_config=output_configs,
            output_deps=output_parameters,
        )

        return result
    finally:
        reset_current_task(token)


async def _run_task_func(
    func: Callable[..., Any],
    metadata: NodeMetadata,
    inputs: dict[str, Any],
    retries: int = 0,
    cache_enabled: bool = False,
    cache_ttl: int | None = None,
    cache_hash_fn: Callable[..., str] | None = None,
    key_extras: dict[str, Any] | None = None,
    iteration_index: int | None = None,
) -> Any:
    """
    Execute a task function on a backend worker with full lifecycle support.

    Args:
        func: The task function to execute (sync or async).
        metadata: Metadata for the node being executed.
        inputs: Dictionary of resolved input values for the function execution.
        retries: Number of times to retry on failure.
        cache_enabled: Whether caching is enabled for this node.
        cache_ttl: Time-to-live for cache in seconds, if None no expiration.
        cache_hash_fn: Custom hash function ``(func, inputs) -> str``. If None,
            the built-in ``default_cache_hash`` is used.
        key_extras: Additional variables for key formatting.
        iteration_index: Optional index for map iterations, used for metadata key formatting.

    Returns:
        Result of the function execution.
    """
    from dataclasses import replace

    # Set metadata key: "name[idx]" for map iterations, "name" for regular tasks
    if iteration_index is not None:
        metadata = replace(metadata, key=f"{metadata.name}[{iteration_index}]")
        key_extras = {**(key_extras or {}), "iteration_index": iteration_index}
    else:
        metadata = replace(metadata, key=metadata.name)
    node_key = metadata.key or metadata.name

    token = set_current_task(metadata)
    hook = get_plugin_manager().hook
    reporter = get_event_reporter()

    # Built-in cache check before execution
    cache_store = get_cache_store() if cache_enabled else None
    cache_key: str | None = None
    if cache_enabled and cache_store is not None:
        from daglite.cache.core import CACHE_MISS

        cache_key, cached = _cache_get(cache_store, cache_hash_fn, func, inputs, node_key)
        if cached is not CACHE_MISS:
            hook.on_cache_hit(
                func=func, metadata=metadata, inputs=inputs, result=cached, reporter=reporter
            )
            reset_current_task(token)
            return cached

    hook.before_node_execute(metadata=metadata, inputs=inputs, reporter=reporter)

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
                        inputs=inputs,
                        reporter=reporter,
                        attempt=attempt,
                        last_error=last_error,
                    )

                # Synchronous/asynchronous function handling
                if inspect.iscoroutinefunction(func):
                    result = await func(**inputs)
                elif inspect.isasyncgenfunction(func):
                    result = [item async for item in func(**inputs)]
                else:
                    result = func(**inputs)

                # Materialize async generators/iterators to lists
                if isinstance(result, (AsyncGenerator, AsyncIterator)):
                    result = [item async for item in result]

                # Materialize sync generators/iterators to lists
                elif isinstance(result, (Generator, Iterator)) and not isinstance(
                    result, (str, bytes)
                ):
                    result = list(result)

                duration = time.time() - start_time

                if attempt > 1:
                    hook.after_node_retry(
                        metadata=metadata,
                        inputs=inputs,
                        reporter=reporter,
                        attempt=attempt,
                        succeeded=True,
                    )
                hook.after_node_execute(
                    metadata=metadata,
                    inputs=inputs,
                    result=result,
                    duration=duration,
                    reporter=reporter,
                )

                # Built-in cache update after successful execution
                if cache_enabled and cache_store is not None and cache_key is not None:
                    _cache_put(cache_store, cache_key, result, cache_ttl, node_key)

                return result

            except Exception as error:
                # Determine if a retry is available
                last_error = error

                if attempt > 1:
                    hook.after_node_retry(
                        metadata=metadata,
                        inputs=inputs,
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
            inputs=inputs,
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

    For each output config the store is resolved (explicit store on the config, or the
    settings-level default). Routing then depends on the driver's locality:

    * **Local drivers** (filesystem, SQLite) - save through the
      `DatasetReporter` so that the coordinator process performs the write.
    * **Remote drivers** (S3, GCS, …) - save directly from the worker since
      the remote store is accessible everywhere.

    Exceptions are **not** caught - a failed save fails the task.

    Args:
        result: The task execution result to save.
        resolved_inputs: Resolved inputs for key formatting.
        output_config: Output configuration tuple for this node.
        output_deps: List of resolved output dependencies for each output config.
        key_extras: Additional variables for key formatting.
    """
    from daglite.graph.nodes.base import NodeOutputConfig
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
        except KeyError as e:  # pragma: no cover – check_key_placeholders at build time
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


def _cache_get(
    cache_store: Any,
    cache_hash_fn: Callable[..., str] | None,
    func: Callable[..., Any],
    inputs: dict[str, Any],
    node_key: str,
) -> tuple[str, Any]:
    """
    Compute the cache key and attempt a cache read.

    Args:
        cache_store: The active cache store
        cache_hash_fn: Custom hash function or ``None` to use `default_cache_hash`.
        func: The task function being executed.
        inputs: Resolved input values.
        node_key: Metadata key used in warning messages.

    Returns:
        `(cache_key, cached_value)` — `cached_value` is `CACHE_MISS` on a miss or when the read
        raises an unexpected exception.
    """
    from daglite.cache.core import CACHE_MISS
    from daglite.cache.core import default_cache_hash

    _hash_fn = cache_hash_fn if cache_hash_fn is not None else default_cache_hash
    cache_key = _hash_fn(func, inputs)
    try:
        return cache_key, cache_store.get(cache_key)
    except (KeyError, FileNotFoundError, TypeError):  # pragma: no cover
        from daglite.plugins.builtin.logging import get_logger

        get_logger().warning(
            f"Failed to read cache for key {cache_key} on node {node_key}. "
            "This will not affect task execution.",
            exc_info=True,
        )
        return cache_key, CACHE_MISS


def _cache_put(
    cache_store: Any,
    cache_key: str,
    result: Any,
    cache_ttl: int | None,
    node_key: str,
) -> None:
    """
    Attempt to write a result to the cache, logging a warning on failure.

    Args:
        cache_store: The active cache store.
        cache_key: The cache key to write to.
        result: The task result to cache.
        cache_ttl: Time-to-live in seconds, or `None` for no expiration.
        node_key: Metadata key used in warning messages.
    """
    try:
        cache_store.put(cache_key, result, ttl=cache_ttl)
    except Exception:  # pragma: no cover
        from daglite.plugins.builtin.logging import get_logger

        get_logger().warning(
            f"Failed to write cache for key {cache_key} on node {node_key}. "
            "This will not affect task success.",
            exc_info=True,
        )
