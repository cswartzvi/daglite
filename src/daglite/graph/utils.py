"""Utility functions for graph execution."""

import asyncio
import inspect
from collections.abc import AsyncGenerator
from collections.abc import AsyncIterator
from collections.abc import Generator
from collections.abc import Iterator
from typing import Any

from typing_extensions import TypeIs

from daglite.graph.base import CompositeGraphNode
from daglite.graph.base import GraphNode


def is_composite_node(node: GraphNode) -> TypeIs[CompositeGraphNode]:
    """Check if a node is a composite node."""
    from daglite.graph.composite import CompositeMapTaskNode
    from daglite.graph.composite import CompositeTaskNode

    return isinstance(node, (CompositeTaskNode, CompositeMapTaskNode))


def materialize_sync(result: Any) -> Any:
    """
    Materialize coroutines and generators in synchronous execution context.

    Converts async results to their materialized form by running them in a new event loop.

    Args:
        result: The result to materialize (may be coroutine, generator, iterator, or value)

    Returns:
        Materialized result (coroutines awaited, generators/iterators converted to lists)
    """
    if inspect.iscoroutine(result):
        result = asyncio.run(result)

    if isinstance(result, (AsyncGenerator, AsyncIterator)):

        async def _collect():
            items = []
            async for item in result:
                items.append(item)
            return items

        return asyncio.run(_collect())

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)

    return result


async def materialize_async(result: Any) -> Any:
    """
    Materialize coroutines and generators in asynchronous execution context.

    Converts async results to their materialized form within the current event loop.

    Args:
        result: The result to materialize (may be coroutine, generator, iterator, or value)

    Returns:
        Materialized result (coroutines awaited, generators/iterators converted to lists)
    """
    if inspect.iscoroutine(result):
        result = await result

    if isinstance(result, (AsyncGenerator, AsyncIterator)):
        items = []
        async for item in result:
            items.append(item)
        return items

    if isinstance(result, (Generator, Iterator)) and not isinstance(result, (str, bytes)):
        return list(result)

    return result
