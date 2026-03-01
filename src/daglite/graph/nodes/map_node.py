"""Map task node representation within the graph IR."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from itertools import product
from typing import Any, Callable
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import override

from daglite._typing import MapMode
from daglite._typing import NodeKind
from daglite._typing import Submission
from daglite.backends.base import Backend
from daglite.exceptions import ExecutionError
from daglite.exceptions import ParameterError
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_node_changes
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes._shared import resolve_output_parameters
from daglite.graph.nodes._workers import run_task_worker
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import PrepareCollectNode


@dataclass(frozen=True)
class MapTaskNode(PrepareCollectNode):
    """Map function task node representation within the graph IR."""

    func: Callable
    """Function to be executed for each map iteration."""

    mode: MapMode
    """Mapping mode: 'product' for Cartesian product, 'zip' for parallel iteration."""

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

    cache_hash_fn: Callable[..., str] | None = None
    """Custom hash function ``(func, inputs) -> str`` for the cache key."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # This is unlikely to happen given retries is checked at task level, but just in case
        assert self.retries >= 0, "Retries must be non-negative"

    @property
    @override
    def kind(self) -> NodeKind:
        return "map"

    @override
    def get_dependencies(self) -> set[UUID]:
        kwargs = {**self.fixed_kwargs, **self.mapped_kwargs}
        return collect_dependencies(kwargs, self.output_configs)

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> MapTaskNode:
        changes = remap_node_changes(
            id_mapping,
            self.output_configs,
            fixed_kwargs=self.fixed_kwargs,
            mapped_kwargs=self.mapped_kwargs,
        )
        return replace(self, **changes) if changes else self

    @override
    def _prepare(self, completed_nodes: Mapping[UUID, Any]) -> list[Submission]:
        inputs = resolve_inputs({**self.fixed_kwargs, **self.mapped_kwargs}, completed_nodes)
        iteration_calls = self.build_iteration_calls(inputs)
        output_parameters = resolve_output_parameters(self.output_configs, completed_nodes)
        submissions: list[Submission] = []
        for idx, iteration_call in enumerate(iteration_calls):
            submission = functools.partial(
                run_task_worker,
                func=self.func,
                metadata=self.metadata,
                inputs=iteration_call,
                output_configs=self.output_configs,
                output_parameters=output_parameters,
                retries=self.retries,
                cache_enabled=self.cache,
                cache_ttl=self.cache_ttl,
                cache_hash_fn=self.cache_hash_fn,
                iteration_index=idx,
            )
            submissions.append(submission)
        return submissions

    @override
    def _collect(self, results: list[Any]) -> Any:
        return list(results)

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        submissions = self._prepare(completed_nodes)
        iteration_count = len(submissions)

        start_time = time.perf_counter()
        hooks.before_mapped_node_execute(metadata=self.metadata, iteration_count=iteration_count)

        futures = [backend.submit(fn, timeout=self.timeout) for fn in submissions]
        results = await asyncio.gather(*futures)

        duration = time.perf_counter() - start_time
        hooks.after_mapped_node_execute(
            metadata=self.metadata,
            iteration_count=iteration_count,
            duration=duration,
        )

        return self._collect(results)

    def build_iteration_calls(self, resolved_inputs: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Build the list of input dictionaries for each iteration of the mapped node.

        Args:
            resolved_inputs: Pre-resolved parameter inputs for this node.
        """
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
