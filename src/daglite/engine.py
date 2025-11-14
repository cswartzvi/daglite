from collections.abc import MutableMapping
from typing import Any, TypeVar

from daglite.nodes import CallNode
from daglite.nodes import Node

T = TypeVar("T")


def evaluate(expr: Node[T]) -> T:
    """
    Evaluate a Node[T] expression to a concrete T.

    This is a simple single-process evaluator with memoization. It:
      - walks the graph recursively,
      - evaluates dependencies first,
      - calls underlying Task functions with concrete values,
      - caches each node's result so it is only computed once.
    """

    memo: MutableMapping[int, Any] = {}

    def _evaluate(node: Any) -> Any:
        """Internal recursive evaluator with memoization."""

        # Non-node values are returned as-is.
        if not isinstance(node, Node):
            return node

        node_id = id(node)
        if node_id in memo:
            return memo[node_id]

        if isinstance(node, CallNode):
            resolved: dict[str, Any] = {}
            for name, value in node.kwargs.items():
                if isinstance(value, Node):
                    resolved[name] = _evaluate(value)  # node dependency
                else:
                    resolved[name] = value  # literal value, leave as-is
            result = node.task._fn(**resolved)

        # elif isinstance(node, MapNode):
        #     # Resolve the sequence to map over.
        #     if isinstance(node.values, Lazy):
        #         seq = _eval(node.values)
        #     else:
        #         seq = node.values

        #     # We expect 'seq' to be a concrete sequence here.
        #     concrete_list: List[Any] = list(seq)

        #     results: List[Any] = []
        #     for item in concrete_list:
        #         # Each item might itself be Lazy (nested mapping), so evaluate it.
        #         arg_val = _eval(item) if isinstance(item, Lazy) else item
        #         # Build kwargs for this call: just the mapped param.
        #         call_kwargs = {node.param: arg_val}
        #         results.append(node.task._fn(**call_kwargs))

        #     result = results

        else:
            raise TypeError(f"Unknown node type: {type(node)!r}")

        memo[node_id] = result
        return result

    return _evaluate(expr)
