"""
Composite node representations within the graph IR.

Composite nodes fold linear sequences of nodes into single execution units, reducing backend
submission overhead. They are created by the graph optimizer and are transparent to the user — the
same worker function handles per-node hooks, caching, retries, and output saving.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections import deque
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Awaitable, Callable
from uuid import UUID

from pluggy import HookRelay
from typing_extensions import Literal, override

from daglite._typing import NodeKind
from daglite._typing import Submission
from daglite.backends.base import Backend
from daglite.graph.nodes._shared import collect_dependencies
from daglite.graph.nodes._shared import remap_node_inputs
from daglite.graph.nodes._shared import remap_output_configs
from daglite.graph.nodes._shared import resolve_inputs
from daglite.graph.nodes._shared import resolve_output_parameters
from daglite.graph.nodes._workers import _save_outputs
from daglite.graph.nodes._workers import run_task_worker
from daglite.graph.nodes.base import BaseGraphNode
from daglite.graph.nodes.base import NodeInput
from daglite.graph.nodes.base import NodeMetadata
from daglite.graph.nodes.base import NodeOutputConfig
from daglite.graph.nodes.iter_node import IterNode
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.reduce_node import ReduceConfig
from daglite.graph.nodes.task_node import TaskNode
from daglite.settings import get_global_settings

TerminalKind = Literal["collect", "join", "reduce"]


# region Node Definitions


@dataclass(frozen=True)
class CompositeTaskNode(BaseGraphNode):
    """A linear sequence of task nodes folded into a single node."""

    steps: tuple[CompositeStep, ...]
    """Ordered sequence of steps to execute."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "composite_task"

    @override
    def get_dependencies(self) -> set[UUID]:
        internal_ids = {step.id for step in self.steps}
        all_deps: set[UUID] = set()
        for step in self.steps:
            all_deps |= collect_dependencies(step.external_params, step.output_configs)
        return all_deps - internal_ids

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> CompositeTaskNode:
        new_steps = _remap_composite_steps(self.steps, id_mapping)
        new_oc = remap_output_configs(self.output_configs, id_mapping)
        if new_steps is not None or new_oc is not None:
            return replace(
                self,
                steps=new_steps if new_steps is not None else self.steps,
                output_configs=new_oc if new_oc is not None else self.output_configs,
            )
        return self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        steps = self.steps
        completed_results = dict(completed_nodes)
        runner = functools.partial(
            _run_composite_steps, steps=steps, completed_results=completed_results
        )

        async with _composite_hook_scope(hooks, self.metadata, len(steps)):
            return await backend.submit(runner, timeout=self.timeout)


@dataclass(frozen=True)
class CompositeMapTaskNode(BaseGraphNode):
    """
    A map task node followed by linear sequence of task nodes folded into a single node.

    Supports three terminal modes:
    * **collect** — `asyncio.gather` all iteration results into a list.
    * **join** — collect all, then run a final reducer `TaskNode` once.
    * **reduce** — streaming fold via `ReduceConfig` with O(1) memory.
    """

    source_map: MapTaskNode
    """The originating map node that generates iterations."""

    steps: tuple[CompositeStep, ...] = field(default=())
    """Ordered `.then()` steps to run per iteration after the source map."""

    terminal: TerminalKind = "collect"
    """How iteration results are aggregated."""

    join_step: CompositeStep | None = field(default=None)
    """If `terminal='join'`, the step that receives the full list."""

    reduce_config: ReduceConfig | None = field(default=None)
    """If `terminal='reduce'`, configuration for the streaming fold."""

    initial_accumulator: NodeInput | None = field(default=None)
    """If `terminal='reduce'`, the initial accumulator value (may reference another node)."""

    iter_source: IterSourceConfig | None = field(default=None)
    """Iterator source config (enables lazy generation on coordinator if present)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.terminal == "reduce":
            assert self.reduce_config is not None, "reduce terminal requires reduce_config"
            assert self.initial_accumulator is not None, (
                "reduce terminal requires initial_accumulator"
            )
        elif self.terminal == "join":
            assert self.join_step is not None, "join terminal requires join_step"

    @property
    @override
    def kind(self) -> NodeKind:
        return "composite_map"

    @property
    def _internal_ids(self) -> set[UUID]:
        """Node IDs internal to this composite (excluded from graph-level dependencies)."""
        ids = {self.source_map.id} | {step.id for step in self.steps}
        if self.join_step is not None:
            ids.add(self.join_step.id)
        if self.iter_source is not None:
            ids.add(self.iter_source.id)
        return ids

    @override
    def get_dependencies(self) -> set[UUID]:
        all_deps = self.source_map.get_dependencies()

        # Include deps from iter_source kwargs and output_configs
        if self.iter_source is not None:
            all_deps |= collect_dependencies(
                self.iter_source.kwargs, self.iter_source.output_configs
            )

        # Include deps from steps
        for step in self.steps:
            all_deps |= collect_dependencies(step.external_params, step.output_configs)

        # Include deps from join_step if present
        if self.join_step is not None:
            all_deps |= collect_dependencies(
                self.join_step.external_params, self.join_step.output_configs
            )

        # Include initial_accumulator ref if present
        if self.initial_accumulator is not None and self.initial_accumulator.reference is not None:
            all_deps.add(self.initial_accumulator.reference)

        return all_deps - self._internal_ids

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> CompositeMapTaskNode:
        oc_mapping = {k: v for k, v in id_mapping.items() if k not in self._internal_ids}
        changed = False

        # Build mapping for source_map: exclude iter_source.id since that ref is internal to the
        # composite (the generator runs on the coordinator).
        source_map_mapping = id_mapping
        if self.iter_source is not None:
            source_map_mapping = {k: v for k, v in id_mapping.items() if k != self.iter_source.id}

        # Remap steps
        new_steps = _remap_composite_steps(self.steps, id_mapping)
        if new_steps is not None:
            changed = True
        else:
            new_steps = self.steps

        # Remap source_map
        new_source_map = self.source_map.remap_references(source_map_mapping)
        if new_source_map is not self.source_map:  # pragma: no cover — cross-composite remap
            changed = True

        # Remap join_step if present
        new_join_step = self.join_step
        if self.join_step is not None:
            new_join_step = _remap_composite_step(self.join_step, id_mapping, oc_mapping)
            if new_join_step is not self.join_step:
                changed = True

        # Remap initial_accumulator ref if present
        new_initial_accumulator = self.initial_accumulator
        if (
            self.initial_accumulator is not None
            and self.initial_accumulator.reference is not None
            and self.initial_accumulator.reference in id_mapping
        ):  # pragma: no cover
            new_initial_accumulator = NodeInput(
                _kind=self.initial_accumulator._kind,
                value=self.initial_accumulator.value,
                reference=id_mapping[self.initial_accumulator.reference],
            )
            changed = True

        # Remap iter_source kwargs and output_configs
        new_iter_source = self.iter_source
        if self.iter_source is not None:
            new_iter_kwargs = remap_node_inputs(self.iter_source.kwargs, id_mapping)
            new_iter_oc = remap_output_configs(self.iter_source.output_configs, oc_mapping)
            if new_iter_kwargs is not self.iter_source.kwargs or new_iter_oc is not None:
                new_iter_source = replace(
                    self.iter_source,
                    kwargs=new_iter_kwargs,
                    output_configs=(
                        new_iter_oc if new_iter_oc is not None else self.iter_source.output_configs
                    ),
                )
                changed = True

        # Remap output_configs
        new_output_configs = remap_output_configs(self.output_configs, id_mapping)
        if new_output_configs is not None:
            changed = True

        if not changed:
            return self
        return replace(
            self,
            steps=new_steps,
            source_map=new_source_map,
            join_step=new_join_step,
            initial_accumulator=new_initial_accumulator,
            iter_source=new_iter_source,
            output_configs=(
                new_output_configs if new_output_configs is not None else self.output_configs
            ),
        )

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        steps = self.steps
        completed_results = dict(completed_nodes)

        # Builds a runner function that executes the entire composite sequence for one iteration
        def _make_steps_runner(source_fn: Submission, iteration_index: int) -> Submission:
            return functools.partial(
                _run_composite_map_steps,
                source_fn=source_fn,
                steps=steps,
                completed_results=completed_results,
                iteration_index=iteration_index,
            )

        submissions: Iterable[tuple[int, Submission]]
        if self.iter_source:
            submissions = enumerate(self._iter_submissions(completed_results))  # lazy submissions
        else:
            submissions = enumerate(self.source_map._prepare(completed_nodes))  # eager submissions

        async with _composite_hook_scope(hooks, self.metadata, len(steps) + 1):
            if self.terminal == "reduce":
                return await self._execute_with_reduce(
                    backend, submissions, _make_steps_runner, hooks, completed_nodes
                )

            futures = [
                backend.submit(_make_steps_runner(fn, idx), timeout=self.timeout)
                for idx, fn in submissions
            ]

            if self.terminal == "join":
                return await self._execute_with_join(backend, futures, hooks, completed_results)

            return await self._execute_with_collect(futures, hooks)

    def _iter_submissions(self, completed_results: dict[UUID, Any]) -> Iterator[Submission]:
        """Lazily yield source submissions from a generator."""
        assert self.iter_source is not None

        # Resolve generator inputs and call on coordinator
        gen_inputs = resolve_inputs(self.iter_source.kwargs, completed_results)
        gen_result = self.iter_source.func(**gen_inputs)

        # Identify flow parameter (which mapped kwarg receives each item)
        (flow_param,) = self.source_map.mapped_kwargs.keys()

        # Resolve fixed kwargs for the map function
        map_fixed = resolve_inputs(dict(self.source_map.fixed_kwargs), completed_results)

        # Resolve output configs for iter source saves (if any) and map step saves
        iter_oc = self.iter_source.output_configs
        iter_oc_params = resolve_output_parameters(iter_oc, completed_results) if iter_oc else []
        map_oc_params = resolve_output_parameters(self.source_map.output_configs, completed_results)

        for idx, item in enumerate(gen_result):
            # Save each yielded item if the iter source has output_configs
            if iter_oc:
                _save_outputs(
                    result=item,
                    resolved_inputs=gen_inputs,
                    output_config=iter_oc,
                    output_deps=iter_oc_params,
                    key_extras={"iteration_index": idx},
                )

            map_inputs = dict(map_fixed)
            map_inputs[flow_param] = item
            yield functools.partial(
                run_task_worker,
                func=self.source_map.func,
                metadata=self.source_map.metadata,
                inputs=map_inputs,
                output_configs=self.source_map.output_configs,
                output_parameters=map_oc_params,
                retries=self.source_map.retries,
                cache_enabled=self.source_map.cache,
                cache_ttl=self.source_map.cache_ttl,
                iteration_index=idx,
            )

    async def _execute_with_collect(
        self, futures: list[Awaitable[Any]], hooks: HookRelay
    ) -> list[Any]:
        """Gather all iteration results into a list."""
        iteration_count = len(futures)
        hooks.before_mapped_node_execute(
            metadata=self.source_map.metadata, iteration_count=iteration_count
        )
        map_start = time.perf_counter()
        results = await asyncio.gather(*futures)
        hooks.after_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
            duration=time.perf_counter() - map_start,
        )
        return results

    async def _execute_with_join(
        self,
        backend: Backend,
        futures: list[Awaitable[Any]],
        hooks: HookRelay,
        completed_results: dict[UUID, Any],
    ) -> Any:
        """Gather all iteration results, then run the join reducer once."""
        results = await self._execute_with_collect(futures, hooks)

        step = self.join_step
        assert step is not None  # guaranteed by __post_init__
        resolved = resolve_inputs(step.external_params, completed_results)
        if step.flow_param is not None:  # pragma: no branch
            resolved[step.flow_param] = results

        output_parameters = resolve_output_parameters(step.output_configs, completed_results)

        join_runner = functools.partial(
            run_task_worker,
            func=step.func,
            metadata=step.metadata,
            inputs=resolved,
            output_configs=step.output_configs,
            output_parameters=output_parameters,
            retries=step.retries,
            cache_enabled=step.cache,
            cache_ttl=step.cache_ttl,
        )
        return await backend.submit(join_runner, timeout=self.timeout)

    async def _execute_with_reduce(
        self,
        backend: Backend,
        submissions: Iterable[tuple[int, Submission]],
        make_runner: Callable[[Submission, int], Submission],
        hooks: HookRelay,
        completed_nodes: Mapping[UUID, Any],
    ) -> Any:
        """Streaming fold with bounded concurrency, dispatching by reduce mode."""
        assert self.reduce_config is not None
        assert self.initial_accumulator is not None
        cfg = self.reduce_config
        accumulator = self.initial_accumulator.resolve(completed_nodes)

        hooks.before_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=0,  # unknown upfront for streaming
        )
        reduce_start = time.perf_counter()

        if cfg.mode == "ordered":
            accumulator, iteration_count = await self._execute_with_reduce_ordered(
                backend, submissions, make_runner, cfg, accumulator
            )
        else:
            accumulator, iteration_count = await self._execute_with_reduce_unordered(
                backend, submissions, make_runner, cfg, accumulator
            )

        hooks.after_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
            duration=time.perf_counter() - reduce_start,
        )

        # Save the final accumulator if the reduce terminal has output_configs
        if self.output_configs:
            output_parameters = resolve_output_parameters(self.output_configs, completed_nodes)
            _save_outputs(
                result=accumulator,
                resolved_inputs={},
                output_config=self.output_configs,
                output_deps=output_parameters,
            )

        return accumulator

    async def _execute_with_reduce_ordered(
        self,
        backend: Backend,
        submissions: Iterable[tuple[int, Submission]],
        make_runner: Callable[[Submission, int], Submission],
        cfg: ReduceConfig,
        accumulator: Any,
    ) -> tuple[Any, int]:
        """
        Ordered streaming reduce using a sliding-window deque.

        The oldest future is awaited and folded before a new one is submitted, preserving
        submission order.
        """
        settings = get_global_settings()
        concurrency = settings.iterator_back_pressure

        iteration_count = 0
        pending: deque[Awaitable[Any]] = deque()

        for idx, source_fn in submissions:
            runner = make_runner(source_fn, idx)
            pending.append(backend.submit(runner, timeout=self.timeout))
            if len(pending) >= concurrency:
                item = await pending.popleft()
                accumulator = await _apply_reduce(cfg, accumulator, item)
                iteration_count += 1

        while pending:
            item = await pending.popleft()
            accumulator = await _apply_reduce(cfg, accumulator, item)
            iteration_count += 1

        return accumulator, iteration_count

    async def _execute_with_reduce_unordered(
        self,
        backend: Backend,
        submissions: Iterable[tuple[int, Submission]],
        make_runner: Callable[[Submission, int], Submission],
        cfg: ReduceConfig,
        accumulator: Any,
    ) -> tuple[Any, int]:
        """
        Unordered streaming reduce using a semaphore with relay tasks.

        Relay tasks await each backend future and push the completed result into a queue so the
        fastest-completing result is reduced first.
        """
        settings = get_global_settings()
        concurrency = settings.iterator_back_pressure
        semaphore = asyncio.Semaphore(concurrency)

        iteration_count = 0
        _SENTINEL = object()
        result_queue: asyncio.Queue[Any] = asyncio.Queue()
        producer_error: list[Exception] = []

        # Producer task: submit all iterations, launching a relay task for each to push results
        # into the queue as they complete. If any submission raises an exception, capture it and
        # re-raise after all results have been processed.
        async def _producer() -> None:
            relay_tasks: list[asyncio.Task[None]] = []
            try:
                for idx, source_fn in submissions:
                    await semaphore.acquire()
                    runner = make_runner(source_fn, idx)
                    future = backend.submit(runner, timeout=self.timeout)
                    relay_tasks.append(asyncio.ensure_future(_relay(future)))
            except Exception as exc:
                producer_error.append(exc)
            if relay_tasks:  # pragma: no branch
                await asyncio.gather(*relay_tasks, return_exceptions=True)
            await result_queue.put(_SENTINEL)

        async def _relay(future: Awaitable[Any]) -> None:
            try:
                r = await future
                await result_queue.put(r)
            except Exception as exc:
                await result_queue.put(_WorkerError(exc))
            finally:
                semaphore.release()

        # Start producer task and consume results as they arrive
        producer_task = asyncio.ensure_future(_producer())
        while True:
            item = await result_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, _WorkerError):
                raise item.exc
            accumulator = await _apply_reduce(cfg, accumulator, item)
            iteration_count += 1
        await producer_task
        if producer_error:
            raise producer_error[0]

        return accumulator, iteration_count


# region Helper Types


@dataclass(frozen=True)
class CompositeStep:
    """
    One step in a composite task or composite map task node.

    Captures everything `run_task_worker` needs to execute one step, plus the
    `flow_param` that receives the previous step's result.
    """

    id: UUID
    """Original node ID for this step (used for result aliasing)."""

    name: str
    """Human-readable name of the original node."""

    description: str | None
    """Optional description of the original node."""

    func: Callable[..., Any]
    """The task function to execute for this step."""

    flow_param: str | None
    """Parameter name receiving the previous step's result or None for first step in a composite."""

    external_params: Mapping[str, NodeInput]
    """Parameters sourced from outside the sequence (literals or refs to completed nodes)."""

    output_configs: tuple[NodeOutputConfig, ...]
    """Output save/checkpoint configurations for this step."""

    step_kind: NodeKind = "task"
    """Kind of the original node this step was built from."""

    retries: int = 0
    """Number of times to retry this step on failure."""

    cache: bool = False
    """Whether hash-based caching is enabled for this step."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds."""

    timeout: float | None = None
    """Per-step timeout in seconds (preserved from the original node for aggregation)."""

    @property
    def metadata(self) -> NodeMetadata:
        """Build a node metadata instance for this step."""
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind=self.step_kind,
            description=self.description,
        )

    @classmethod
    def from_node(cls, node: BaseGraphNode, *, flow_param: str | None) -> CompositeStep:
        """Build a composite step instance from a task or map node."""
        if isinstance(node, MapTaskNode):
            external_params: Mapping[str, NodeInput] = dict(node.fixed_kwargs)
        elif isinstance(node, TaskNode):
            external_params = {k: v for k, v in node.kwargs.items() if k != flow_param}
        else:  # pragma: no cover
            raise ValueError(f"Unsupported node type for composite step: {type(node)}")
        return cls(
            id=node.id,
            name=node.name,
            description=node.description,
            func=node.func,
            flow_param=flow_param,
            external_params=external_params,
            output_configs=node.output_configs,
            retries=node.retries,
            cache=node.cache,
            cache_ttl=node.cache_ttl,
            timeout=node.timeout,
            step_kind=node.kind,
        )


@dataclass(frozen=True)
class IterSourceConfig:
    """Configuration for a lazy iterator source folded into a composite."""

    id: UUID
    """Original node ID for this iter source (used for result aliasing)."""

    func: Callable[..., Any]
    """Generator/iterator-returning function."""

    kwargs: Mapping[str, NodeInput]
    """Keyword parameters mapped to node inputs."""

    output_configs: tuple[NodeOutputConfig, ...] = ()
    """Output save/checkpoint configurations from the original iter node (if present)."""

    retries: int = 0
    cache: bool = False
    cache_ttl: int | None = None

    @classmethod
    def from_iter_node(cls, node: IterNode) -> IterSourceConfig:
        """Builds an iter source config instance from an iter node."""
        assert node.kind == "iter", "Expected IterNode for IterSourceConfig"
        return cls(
            id=node.id,
            func=node.func,
            kwargs=node.kwargs,
            output_configs=node.output_configs,
            retries=node.retries,
            cache=node.cache,
            cache_ttl=node.cache_ttl,
        )


class _WorkerError:
    """Wrapper to distinguish worker exceptions from normal results in a queue."""

    __slots__ = ("exc",)

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


# region Helper Functions


@asynccontextmanager
async def _composite_hook_scope(hooks: HookRelay, metadata: NodeMetadata, num_steps: int):
    """Context manager that wraps a composite execution with before/after/error hooks."""
    start_time = time.perf_counter()
    hooks.before_composite_execute(metadata=metadata, num_steps=num_steps)
    try:
        yield
    except Exception as e:
        hooks.on_composite_error(
            metadata=metadata,
            num_steps=num_steps,
            error=e,
            duration=time.perf_counter() - start_time,
        )
        raise
    hooks.after_composite_execute(
        metadata=metadata, num_steps=num_steps, duration=time.perf_counter() - start_time
    )


async def _apply_reduce(cfg: ReduceConfig, acc: Any, item: Any) -> Any:
    """Apply the reduce function (sync or async) with retries."""
    kwargs = {cfg.accumulator_param: acc, cfg.item_param: item}
    last_error: Exception | None = None
    for attempt in range(1 + cfg.retries):
        try:
            if inspect.iscoroutinefunction(cfg.func):
                return await cfg.func(**kwargs)
            return cfg.func(**kwargs)
        except Exception as e:
            last_error = e
            if (
                attempt < cfg.retries
            ):  # pragma: no branch — coverage.py misses nested-async branches
                continue
    raise last_error  # type: ignore[misc]  # pragma: no cover


def _remap_composite_step(
    step: CompositeStep,
    id_mapping: Mapping[UUID, UUID],
    oc_mapping: Mapping[UUID, UUID] | None = None,
) -> CompositeStep:
    """Remaps a single composite step, returning the original if no changes were made."""
    new_params = remap_node_inputs(step.external_params, id_mapping)
    new_oc = remap_output_configs(
        step.output_configs, oc_mapping if oc_mapping is not None else id_mapping
    )
    if new_params is not step.external_params or new_oc is not None:
        return replace(
            step,
            external_params=new_params,
            **(dict(output_configs=new_oc) if new_oc is not None else {}),
        )
    return step


def _remap_composite_steps(
    steps: tuple[CompositeStep, ...], id_mapping: Mapping[UUID, UUID]
) -> tuple[CompositeStep, ...] | None:
    """Remaps an entire sequence of steps, returning None if nothing changed."""
    internal_ids = {step.id for step in steps}
    oc_mapping = {k: v for k, v in id_mapping.items() if k not in internal_ids}
    new_steps: list[CompositeStep] = []
    changed = False
    for step in steps:
        new_step = _remap_composite_step(step, id_mapping, oc_mapping)
        if new_step is not step:
            changed = True
        new_steps.append(new_step)
    return tuple(new_steps) if changed else None


async def _run_composite_steps(
    steps: tuple[CompositeStep, ...], completed_results: dict[UUID, Any]
) -> Any:
    """Execute a sequence of composite steps in a composite task node."""
    previous_output: Any = None
    for step in steps:
        previous_output = await _run_step(step, previous_output, completed_results)
        completed_results[step.id] = previous_output
    return previous_output


async def _run_composite_map_steps(
    source_fn: Submission,
    steps: tuple[CompositeStep, ...],
    completed_results: dict[UUID, Any],
    iteration_index: int,
) -> Any:
    """Execute one iteration of the composite map steps."""
    previous_output = await source_fn()
    for step in steps:
        previous_output = await _run_step(
            step, previous_output, completed_results, iteration_index=iteration_index
        )
    return previous_output


async def _run_step(
    step: CompositeStep,
    previous_output: Any,
    completed_results: dict[UUID, Any],
    *,
    iteration_index: int | None = None,
) -> Any:
    """Executes a single composite step."""
    resolved = resolve_inputs(step.external_params, completed_results)

    # Inject previous output into the flow parameter
    if step.flow_param is not None:
        resolved[step.flow_param] = previous_output

    output_parameters = resolve_output_parameters(step.output_configs, completed_results)
    return await run_task_worker(
        func=step.func,
        metadata=step.metadata,
        inputs=resolved,
        output_configs=step.output_configs,
        output_parameters=output_parameters,
        retries=step.retries,
        cache_enabled=step.cache,
        cache_ttl=step.cache_ttl,
        iteration_index=iteration_index,
    )
