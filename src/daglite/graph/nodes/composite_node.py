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
from collections.abc import Iterator
from collections.abc import Mapping
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
from daglite.graph.nodes.map_node import MapTaskNode
from daglite.graph.nodes.reduce_node import ReduceConfig

TerminalKind = Literal["collect", "join", "reduce"]


# region Node Definitions


@dataclass(frozen=True)
class CompositeTaskNode(BaseGraphNode):
    """A linear sequence of `TaskNode`s folded into a single node."""

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
        new_steps = _remap_steps(self.steps, id_mapping)
        new_output_configs = remap_output_configs(self.output_configs, id_mapping)
        if new_steps is not None or new_output_configs is not None:
            changes: dict[str, Any] = {}
            if new_steps is not None:
                changes["steps"] = new_steps
            if new_output_configs is not None:
                changes["output_configs"] = new_output_configs
            return replace(self, **changes)
        return self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        steps = self.steps
        snapshot = dict(completed_nodes)

        runner = functools.partial(_run_composite_steps, steps=steps, snapshot=snapshot)

        start_time = time.perf_counter()
        hooks.before_composite_execute(metadata=self.metadata, num_steps=len(steps))

        try:
            future = backend.submit(runner, timeout=self.timeout)
            result = await future
        except Exception as e:
            duration = time.perf_counter() - start_time
            hooks.on_composite_error(
                metadata=self.metadata, num_steps=len(steps), error=e, duration=duration
            )
            raise

        duration = time.perf_counter() - start_time
        hooks.after_composite_execute(
            metadata=self.metadata, num_steps=len(steps), duration=duration
        )
        return result


@dataclass(frozen=True)
class CompositeMapTaskNode(BaseGraphNode):
    """
    A linear sequence of `MapTaskNode` followed by parallel `TaskNode`s folded into a single node.

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

    initial_input: NodeInput | None = field(default=None)
    """If ``terminal='reduce'``, the initial accumulator value (may reference another node)."""

    iter_source: IterSourceConfig | None = field(default=None)
    """Iterator source config (enables lazy generation on coordinator if present)."""

    @property
    @override
    def kind(self) -> NodeKind:
        return "composite_map"

    @override
    def get_dependencies(self) -> set[UUID]:
        # Collect internal IDs to exclude from dependencies since they're not real graph nodes
        internal_ids = {self.source_map.id} | {step.id for step in self.steps}
        if self.join_step is not None:
            internal_ids.add(self.join_step.id)
        if self.iter_source is not None:
            internal_ids.add(self.iter_source.id)

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

        # Include initial_input ref if present
        if self.initial_input is not None and self.initial_input.reference is not None:
            all_deps.add(self.initial_input.reference)

        return all_deps - internal_ids

    @override
    def remap_references(self, id_mapping: Mapping[UUID, UUID]) -> CompositeMapTaskNode:
        changes: dict[str, Any] = {}

        # Build filtered mapping for output_configs: exclude internal IDs
        internal_ids = {self.source_map.id} | {link.id for link in self.steps}
        if self.join_step is not None:
            internal_ids.add(self.join_step.id)
        if self.iter_source is not None:
            internal_ids.add(self.iter_source.id)
        oc_mapping = {k: v for k, v in id_mapping.items() if k not in internal_ids}

        # Build mapping for source_map: exclude iter_source.id since that ref is internal to the
        # composite (the generator runs on the coordinator).
        source_map_mapping = id_mapping
        if self.iter_source is not None:
            source_map_mapping = {k: v for k, v in id_mapping.items() if k != self.iter_source.id}

        # Remap steps
        new_steps = _remap_steps(self.steps, id_mapping)
        if new_steps is not None:
            changes["steps"] = new_steps

        # Remap source_map
        new_source = self.source_map.remap_references(source_map_mapping)
        if new_source is not self.source_map:  # pragma: no cover — cross-composite remap
            changes["source_map"] = new_source

        # Remap join_step if present
        if self.join_step is not None:
            new_join = _remap_composite_step(self.join_step, id_mapping, oc_mapping)
            if new_join is not self.join_step:
                changes["join_step"] = new_join

        # Remap initial_input ref if present
        if (
            self.initial_input is not None and self.initial_input.reference is not None
        ):  # pragma: no branch
            if self.initial_input.reference in id_mapping:  # pragma: no cover
                changes["initial_input"] = NodeInput(
                    _kind=self.initial_input._kind,
                    value=self.initial_input.value,
                    reference=id_mapping[self.initial_input.reference],
                )

        # Remap iter_source kwargs and output_configs
        if self.iter_source is not None:
            new_iter_kwargs = remap_node_inputs(self.iter_source.kwargs, id_mapping)
            new_iter_oc = remap_output_configs(self.iter_source.output_configs, oc_mapping)
            if new_iter_kwargs is not self.iter_source.kwargs or new_iter_oc is not None:
                iter_changes: dict[str, Any] = {}
                if new_iter_kwargs is not self.iter_source.kwargs:
                    iter_changes["kwargs"] = new_iter_kwargs
                if new_iter_oc is not None:
                    iter_changes["output_configs"] = new_iter_oc
                changes["iter_source"] = replace(self.iter_source, **iter_changes)

        # Remap output_configs
        new_oc = remap_output_configs(self.output_configs, id_mapping)
        if new_oc is not None:
            changes["output_configs"] = new_oc

        if changes:
            return replace(self, **changes)
        return self

    @override
    async def execute(
        self, backend: Backend, completed_nodes: Mapping[UUID, Any], hooks: HookRelay
    ) -> Any:
        steps = self.steps
        snapshot = dict(completed_nodes)

        start_time = time.perf_counter()
        hooks.before_composite_execute(
            metadata=self.metadata,
            num_steps=len(steps) + 1,  # +1 for source map
        )

        try:

            def _make_steps_runner(source_fn: Submission, iteration_index: int) -> Submission:
                return functools.partial(
                    _run_composite_map_step,
                    source_fn=source_fn,
                    steps=steps,
                    snapshot=snapshot,
                    iteration_index=iteration_index,
                )

            # Iter + reduce: true streaming with backpressure.
            # Items are generated, submitted, awaited, and reduced
            # incrementally — the generator is never fully consumed upfront.
            if (
                self.iter_source is not None
                and self.terminal == "reduce"
                and self.reduce_config is not None
            ):
                result = await self._execute_iter_streaming_reduce(
                    backend,
                    snapshot,
                    _make_steps_runner,
                    hooks,
                    completed_nodes,
                )
            else:
                # Submit iterations to the backend.
                # For iter sources the generator is consumed lazily so each item
                # is dispatched to a worker as it yields (enabling interleaving).
                # For regular map sources all submissions are prepared upfront.
                if self.iter_source is not None:
                    futures: list[Awaitable[Any]] = []
                    for idx, source_fn in enumerate(self._iter_submissions(snapshot)):
                        runner = _make_steps_runner(source_fn, idx)
                        futures.append(backend.submit(runner, timeout=self.timeout))
                else:
                    source_submissions = self.source_map._prepare(completed_nodes)
                    runners = [
                        _make_steps_runner(fn, idx) for idx, fn in enumerate(source_submissions)
                    ]
                    futures = [backend.submit(runner, timeout=self.timeout) for runner in runners]

                iteration_count = len(futures)

                if self.terminal == "reduce" and self.reduce_config is not None:
                    result = await self._execute_reduce(
                        futures, hooks, completed_nodes, iteration_count
                    )
                else:
                    result = await self._execute_batch(
                        backend,
                        futures,
                        hooks,
                        iteration_count,
                        snapshot,
                    )
        except Exception as e:
            duration = time.perf_counter() - start_time
            hooks.on_composite_error(
                metadata=self.metadata,
                num_steps=len(steps) + 1,
                error=e,
                duration=duration,
            )
            raise

        duration = time.perf_counter() - start_time
        hooks.after_composite_execute(
            metadata=self.metadata,
            num_steps=len(steps) + 1,
            duration=duration,
        )
        return result

    def _iter_submissions(self, snapshot: dict[UUID, Any]) -> Iterator[Submission]:
        """
        Lazily yield source submissions from a generator.

        Iterates the generator on the coordinator, saves each yielded item when
        ``output_configs`` are present, and yields ``Submission`` callables one
        at a time so the caller can dispatch each to a backend worker as it
        arrives — enabling true interleaving of generation and processing.
        """
        assert self.iter_source is not None

        # Resolve generator inputs and call on coordinator
        gen_inputs = resolve_inputs(self.iter_source.kwargs, snapshot)
        gen_result = self.iter_source.func(**gen_inputs)

        # Identify flow parameter (which mapped kwarg receives each item)
        ((flow_param, _),) = self.source_map.mapped_kwargs.items()

        # Resolve fixed kwargs for the map function
        map_fixed = resolve_inputs(dict(self.source_map.fixed_kwargs), snapshot)

        # Resolve output configs for iter source saves (once, outside loop)
        iter_oc = self.iter_source.output_configs
        iter_oc_params = resolve_output_parameters(iter_oc, snapshot) if iter_oc else []

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
            output_parameters = resolve_output_parameters(self.source_map.output_configs, snapshot)
            yield functools.partial(
                run_task_worker,
                func=self.source_map.func,
                metadata=self.source_map.metadata,
                inputs=map_inputs,
                output_configs=self.source_map.output_configs,
                output_parameters=output_parameters,
                retries=self.source_map.retries,
                cache_enabled=self.source_map.cache,
                cache_ttl=self.source_map.cache_ttl,
                iteration_index=idx,
            )

    async def _execute_batch(
        self,
        backend: Backend,
        futures: list[Awaitable[Any]],
        hooks: HookRelay,
        iteration_count: int,
        snapshot: dict[UUID, Any],
    ) -> Any:
        """Gather all iteration results, optionally run join reducer."""
        hooks.before_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
        )

        map_start = time.perf_counter()
        results = await asyncio.gather(*futures)

        hooks.after_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
            duration=time.perf_counter() - map_start,
        )

        collected = list(results)

        # Terminal: join — run the reducer once on the full list
        if self.terminal == "join" and self.join_step is not None:
            step = self.join_step
            resolved = resolve_inputs(step.external_params, snapshot)
            if step.flow_param is not None:  # pragma: no branch
                resolved[step.flow_param] = collected

            output_parameters = resolve_output_parameters(step.output_configs, snapshot)

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

            future = backend.submit(join_runner, timeout=self.timeout)
            return await future

        return collected

    async def _execute_iter_streaming_reduce(
        self,
        backend: Backend,
        snapshot: dict[UUID, Any],
        make_runner: Callable[[Submission, int], Submission],
        hooks: HookRelay,
        completed_nodes: Mapping[UUID, Any],
    ) -> Any:
        """
        True streaming reduce for iter sources using producer-consumer.

        A **producer** task generates items and submits them to the backend,
        placing the resulting futures into a bounded ``asyncio.Queue``.  A
        **consumer** task pulls futures from the queue, awaits results, and
        folds them into the accumulator.  Both tasks run concurrently on the
        event loop so generation, worker execution, and reduction overlap.

        The queue's ``maxsize`` provides natural backpressure — when it is full
        the producer pauses until the consumer drains a slot.

        For **ordered** mode the consumer awaits futures in FIFO order.
        For **unordered** mode completed results are relayed to the consumer
        via a separate result queue so reduction overlaps with generation.
        """
        assert self.reduce_config is not None
        assert self.initial_input is not None
        cfg = self.reduce_config
        accumulator = self.initial_input.resolve(completed_nodes)

        hooks.before_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=0,  # unknown upfront for streaming
        )

        reduce_start = time.perf_counter()

        _SENTINEL = object()
        concurrency = 8  # max in-flight futures before backpressure kicks in
        iteration_count = 0

        if cfg.mode == "ordered":
            # Producer → bounded queue of futures → consumer awaits in FIFO order.
            queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=concurrency)
            producer_error: list[Exception] = []

            async def _ordered_producer() -> None:
                try:
                    for idx, source_fn in enumerate(self._iter_submissions(snapshot)):
                        runner = make_runner(source_fn, idx)
                        future = backend.submit(runner, timeout=self.timeout)
                        await queue.put(future)  # blocks when queue is full
                except Exception as exc:
                    producer_error.append(exc)
                finally:
                    await queue.put(_SENTINEL)

            producer_task = asyncio.ensure_future(_ordered_producer())
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                result = await item
                accumulator = await _apply_reduce(cfg, accumulator, result)
                iteration_count += 1
            await producer_task
            if producer_error:
                raise producer_error[0]
        else:
            # Unordered: producer submits to backend and spawns relay tasks
            # that await each future and push completed results into a result
            # queue.  A semaphore limits in-flight work.  The producer waits for
            # all relay tasks to finish before sending the sentinel, guaranteeing
            # the consumer sees every result without polling.
            semaphore = asyncio.Semaphore(concurrency)
            result_queue: asyncio.Queue[Any] = asyncio.Queue()
            producer_error_u: list[Exception] = []

            async def _relay(future: Any) -> None:
                """Await a backend future and push the result (or error) to the queue."""
                try:
                    r = await future
                    await result_queue.put(r)
                except Exception as exc:
                    await result_queue.put(_WorkerError(exc))
                finally:
                    semaphore.release()

            async def _unordered_producer() -> None:
                relay_tasks: list[asyncio.Task[None]] = []
                try:
                    for idx, source_fn in enumerate(self._iter_submissions(snapshot)):
                        await semaphore.acquire()  # backpressure
                        runner = make_runner(source_fn, idx)
                        future = backend.submit(runner, timeout=self.timeout)
                        relay_tasks.append(asyncio.ensure_future(_relay(future)))
                except Exception as exc:
                    producer_error_u.append(exc)
                # Wait for all in-flight relay tasks before signalling completion.
                if relay_tasks:
                    await asyncio.gather(*relay_tasks, return_exceptions=True)
                await result_queue.put(_SENTINEL)

            producer_task = asyncio.ensure_future(_unordered_producer())
            while True:
                item = await result_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, _WorkerError):
                    raise item.exc
                accumulator = await _apply_reduce(cfg, accumulator, item)
                iteration_count += 1
            await producer_task
            if producer_error_u:
                raise producer_error_u[0]

        hooks.after_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
            duration=time.perf_counter() - reduce_start,
        )

        return accumulator

    async def _execute_reduce(
        self,
        futures: list[Awaitable[Any]],
        hooks: HookRelay,
        completed_nodes: Mapping[UUID, Any],
        iteration_count: int,
    ) -> Any:
        """
        Streaming fold over pre-submitted futures (non-iter path).

        The reduce function runs on the **coordinator** (not on the backend) because it needs to
        see results as they stream in from the backend.  Map iterations are submitted to the
        backend as usual.  If you need a heavy post-processing step, use ``.join()`` instead,
        which submits a single reducer to the backend after all iterations complete.
        """
        assert self.reduce_config is not None
        assert self.initial_input is not None
        cfg = self.reduce_config
        accumulator = self.initial_input.resolve(completed_nodes)

        hooks.before_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
        )

        reduce_start = time.perf_counter()

        if cfg.mode == "ordered":
            for future in futures:
                item = await future
                accumulator = await _apply_reduce(cfg, accumulator, item)
        else:
            for coro in asyncio.as_completed(futures):
                item = await coro
                accumulator = await _apply_reduce(cfg, accumulator, item)

        hooks.after_mapped_node_execute(
            metadata=self.source_map.metadata,
            iteration_count=iteration_count,
            duration=time.perf_counter() - reduce_start,
        )

        return accumulator


# region Helper Types


@dataclass(frozen=True)
class CompositeStep:
    """
    One step in a `CompositeTaskNode` or `CompositeMapTaskNode`.

    Captures everything ``run_task_worker`` needs to execute one step, plus the
    ``flow_param`` that receives the previous step's result.
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
    """
    Parameter name receiving the previous step's result or None for first step in a composite.
    """

    external_params: Mapping[str, NodeInput]
    """Parameters sourced from outside the sequence (literals or refs to completed nodes)."""

    output_configs: tuple[NodeOutputConfig, ...]
    """Output save/checkpoint configurations for this step."""

    retries: int = 0
    """Number of times to retry this step on failure."""

    cache: bool = False
    """Whether hash-based caching is enabled for this step."""

    cache_ttl: int | None = None
    """Time-to-live for cached results in seconds."""

    timeout: float | None = None
    """Per-step timeout in seconds (preserved from the original node for aggregation)."""

    step_kind: NodeKind = "task"
    """Kind of the original node this step was built from."""

    @property
    def metadata(self) -> NodeMetadata:
        """Build ``NodeMetadata`` for this step."""
        return NodeMetadata(
            id=self.id,
            name=self.name,
            kind=self.step_kind,
            description=self.description,
        )


class _WorkerError:
    """Wrapper to distinguish worker exceptions from normal results in a queue."""

    __slots__ = ("exc",)

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


@dataclass(frozen=True)
class IterSourceConfig:
    """Configuration for a lazy iterator source folded into a composite."""

    id: UUID
    """Original ``IterNode`` ID (used for dependency exclusion)."""

    func: Callable[..., Any]
    """Generator/iterator-returning function."""

    kwargs: Mapping[str, NodeInput]
    """Keyword parameters mapped to node inputs."""

    output_configs: tuple[NodeOutputConfig, ...] = ()
    """Output save/checkpoint configurations from the original ``IterNode``."""

    retries: int = 0
    cache: bool = False
    cache_ttl: int | None = None


# region Helper Functions


async def _apply_reduce(cfg: ReduceConfig, acc: Any, item: Any) -> Any:
    """
    Apply the reduce function (sync or async) with retries.

    Extracted to module level so it can be shared by ``_execute_reduce`` and
    ``_execute_iter_reduce``.
    """
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
    """
    Remaps external_params and output_configs references in a CompositeStep.

    ``oc_mapping`` is used for output_configs remapping.  When ``None``,
    ``id_mapping`` is used.  Callers that know which IDs are step-internal
    should pass a filtered mapping that excludes them so that internal
    references (resolved within the composite snapshot) are left untouched.
    """
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


def _remap_steps(
    steps: tuple[CompositeStep, ...], id_mapping: Mapping[UUID, UUID]
) -> tuple[CompositeStep, ...] | None:
    """
    Remaps an entire sequence of steps, returning None if nothing changed.

    Internal step IDs are excluded from the output_configs remapping so that
    references resolved within the composite's own snapshot are not broken.
    """
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


async def _run_composite_steps(steps: tuple[CompositeStep, ...], snapshot: dict[UUID, Any]) -> Any:
    """
    Execute a `CompositeTaskNode` steps as a single sequential unit.

    Defined at module level so it is picklable by `ProcessPoolExecutor`.
    """
    result: Any = None

    for i, step in enumerate(steps):
        resolved = resolve_inputs(step.external_params, snapshot)
        if i > 0 and step.flow_param is not None:
            resolved[step.flow_param] = result

        output_parameters = resolve_output_parameters(step.output_configs, snapshot)

        result = await run_task_worker(
            func=step.func,
            metadata=step.metadata,
            inputs=resolved,
            output_configs=step.output_configs,
            output_parameters=output_parameters,
            retries=step.retries,
            cache_enabled=step.cache,
            cache_ttl=step.cache_ttl,
        )

        # Make result available for subsequent steps' external_params
        snapshot[step.id] = result

    return result


async def _run_composite_map_step(
    source_fn: Submission,
    steps: tuple[CompositeStep, ...],
    snapshot: dict[UUID, Any],
    iteration_index: int,
) -> Any:
    """
    Executes one iteration of the composite map steps.

    Defined at module level so it is picklable by `ProcessPoolExecutor`.
    """
    result = await source_fn()

    for step in steps:
        resolved = resolve_inputs(step.external_params, snapshot)
        if step.flow_param is not None:  # pragma: no branch – map steps must have flow_param
            resolved[step.flow_param] = result

        output_parameters = resolve_output_parameters(step.output_configs, snapshot)

        result = await run_task_worker(
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
    return result
