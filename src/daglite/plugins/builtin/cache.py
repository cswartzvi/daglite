"""Cache plugin for automatic task result caching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from daglite.plugins.hooks.markers import hook_impl

if TYPE_CHECKING:
    from daglite.cache.store import CacheStore
    from daglite.graph.nodes.base import NodeMetadata


class CachePlugin:
    """
    Plugin for automatic caching of task results.

    Caches task results based on a hash of the function source and input parameters.

    Examples:
        >>> from daglite.cache.store import FileCacheStore
        >>> from daglite.plugins.builtin.cache import CachePlugin
        >>> store = FileCacheStore("/tmp/cache")
        >>> plugin = CachePlugin(store=store)
        >>> from daglite import task
        >>> @task(cache=True)
        ... def expensive_computation(x: int) -> int:
        ...     return x * 2
        >>> result = expensive_computation(x=5).run(plugins=[plugin])
    """

    def __init__(self, store: CacheStore) -> None:
        """
        Initialize cache plugin.

        Args:
            store: Cache store implementation for reading/writing cached values.
        """
        self.store = store

    @hook_impl
    def check_cache(
        self,
        func: Any,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        cache_enabled: bool,
        cache_ttl: int | None,
    ) -> Any | None:
        """Check for cached result before node execution."""
        if not cache_enabled:
            return None

        from daglite.cache.core import default_cache_hash

        cache_key = default_cache_hash(func, inputs)

        try:
            # Wrapper distinguishes cache miss (None) from cached None value ({"value": None})
            cached_wrapper = self.store.get(cache_key, return_type=dict)  # type: ignore[type-var]
            if (
                cached_wrapper is not None
                and isinstance(cached_wrapper, dict)
                and "value" in cached_wrapper
            ):
                return cached_wrapper
            return None
        except (KeyError, FileNotFoundError, TypeError):  # pragma: no cover
            return None

    @hook_impl
    def update_cache(
        self,
        func: Any,
        metadata: NodeMetadata,
        inputs: dict[str, Any],
        result: Any,
        cache_enabled: bool,
        cache_ttl: int | None,
    ) -> None:
        """Store result in cache after successful execution."""
        if not cache_enabled:
            return

        from daglite.cache.core import default_cache_hash

        cache_key = default_cache_hash(func, inputs)
        wrapped_result = {"value": result}
        self.store.put(cache_key, wrapped_result, ttl=cache_ttl)


__all__ = ["CachePlugin"]
