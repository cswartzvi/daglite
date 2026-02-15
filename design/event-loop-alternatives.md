# Event Loop Performance Analysis

## Current Implementation

The current `EventProcessor` polls event sources every 1ms when idle:

```python
# src/daglite/plugins/events.py
def _process_loop(self) -> None:
    while self._running:
        has_events = False
        for source in self._sources.values():
            event = self._get_event(source)
            if event:
                self._registry.dispatch(event)
                has_events = True
        if not has_events:
            time.sleep(0.001)  # 1ms sleep when idle
```

### Performance Impact

**CPU Usage**: ~0.1% for idle event loop (1ms sleep = 1000 iterations/second)
- **Low-frequency events** (e.g., 1 event/second): 999 wasted iterations
- **High-frequency events** (e.g., 100 events/second): 900 wasted iterations
- **Burst events**: Good responsiveness (1ms latency)

**When is this a problem?**
- ❌ Long-running graphs (hours) with infrequent events → wastes CPU over time
- ❌ Battery-powered devices → drains battery
- ✅ Short-running graphs (seconds-minutes) → negligible overhead
- ✅ High event frequency → minimal waste

**Recommendation**: For most daglite use cases (data pipelines, batch jobs), the current implementation is **perfectly fine**. The overhead is small and the simplicity is valuable.

---

## Alternative 1: Adaptive Backoff (Simple)

**Idea**: Start with 1ms, increase to 10ms → 100ms when idle

```python
def _process_loop(self) -> None:
    sleep_time = 0.001  # Start at 1ms
    max_sleep = 0.1     # Cap at 100ms

    while self._running:
        has_events = False
        for source in self._sources.values():
            event = self._get_event(source)
            if event:
                self._registry.dispatch(event)
                has_events = True

        if has_events:
            sleep_time = 0.001  # Reset to 1ms when events occur
        else:
            sleep_time = min(sleep_time * 2, max_sleep)  # Exponential backoff

        time.sleep(sleep_time)
```

**Pros**:
- ✅ Reduces CPU usage during idle periods (0.1% → ~0.01%)
- ✅ Still responsive for bursts (1ms initial latency)
- ✅ Simple to implement (5 lines)

**Cons**:
- ⚠️ Latency increases during low-frequency events (up to 100ms)
- ⚠️ Slightly more complex than current implementation

**When to use**: If users report CPU usage concerns on long-running jobs

---

## Alternative 2: Async Event Loop (Your Idea)

**Idea**: Use `asyncio.Queue` instead of polling

```python
import asyncio
from asyncio import Queue as AsyncQueue

class AsyncEventProcessor:
    """Async event processor using asyncio.Queue."""

    def __init__(self, registry: EventRegistry):
        self._registry = registry
        self._queue: AsyncQueue[dict[str, Any]] = AsyncQueue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start async event processing."""
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop async event processing."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self) -> None:
        """Process events from queue - no polling!"""
        while True:
            event = await self._queue.get()  # Blocks until event available
            self._registry.dispatch(event)
            self._queue.task_done()

    async def dispatch(self, event: dict[str, Any]) -> None:
        """Dispatch event immediately (non-blocking)."""
        await self._queue.put(event)
```

**Integration with ThreadReporter**:
```python
class AsyncThreadReporter:
    def __init__(self, queue: AsyncQueue[Any]):
        self._queue = queue

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        """Send event from worker thread to async queue."""
        event = {"type": event_type, **data}
        # Need to use asyncio.run_coroutine_threadsafe()
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(self._queue.put(event), loop)
```

**Pros**:
- ✅ **Zero CPU usage** when idle (no polling!)
- ✅ Perfect for async-first architectures
- ✅ Scales to thousands of concurrent events

**Cons**:
- ❌ **Much more complex** - requires async everywhere
- ❌ **Breaking change** - Engine must be async
- ❌ **Thread safety issues** - workers need `run_coroutine_threadsafe()`
- ❌ **Overkill** for current daglite architecture

**When to use**: If daglite moves to async-first architecture (big refactor)

---

## Alternative 3: Threading.Event (Middle Ground)

**Idea**: Use `threading.Event` to wake up thread when events arrive

```python
import threading

class EventDrivenProcessor:
    """Event processor with explicit wake-up mechanism."""

    def __init__(self, registry: EventRegistry):
        self._registry = registry
        self._sources: dict[UUID, Any] = {}
        self._running = False
        self._thread: Thread | None = None
        self._wake_event = threading.Event()

    def notify_event(self) -> None:
        """Wake up processor thread (called by reporters)."""
        self._wake_event.set()

    def _process_loop(self) -> None:
        while self._running:
            # Wait for wake-up signal with timeout
            self._wake_event.wait(timeout=0.1)  # 100ms max sleep
            self._wake_event.clear()

            # Process all available events
            for source in self._sources.values():
                while True:  # Drain queue
                    event = self._get_event_nonblocking(source)
                    if event is None:
                        break
                    self._registry.dispatch(event)
```

**Updated Reporter**:
```python
class ThreadReporterWithWakeup:
    def __init__(self, queue: Queue[Any], wake_callback: Callable[[], None]):
        self._queue = queue
        self._wake_callback = wake_callback

    def report(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, **data}
        self._queue.put(event)
        self._wake_callback()  # Wake up processor
```

**Pros**:
- ✅ Near-zero CPU when idle (only wakes on events)
- ✅ Simple to understand (just adds wake-up mechanism)
- ✅ Compatible with existing threading architecture

**Cons**:
- ⚠️ Requires reporters to notify processor (tight coupling)
- ⚠️ Still uses timeout for robustness (100ms max latency)

**When to use**: Good middle ground if CPU usage becomes a concern

---

## Recommendation

**For now**: Keep current implementation (1ms polling)
- Simple, predictable, works well for typical use cases
- CPU overhead is negligible for batch/pipeline workloads
- Easy to debug and maintain

**If CPU becomes an issue**: Implement Alternative 1 (Adaptive Backoff)
- Minimal code change (5 lines)
- Preserves simplicity
- Reduces CPU during idle periods

**For future async refactor**: Consider Alternative 2 (Async Event Loop)
- Only if daglite moves to async-first architecture
- Requires Engine to be fully async
- Best performance but high complexity cost

---

## Measurement

To decide if optimization is needed, add telemetry:

```python
import time

class EventProcessor:
    def _process_loop(self) -> None:
        iterations = 0
        events_processed = 0
        start_time = time.time()

        while self._running:
            iterations += 1
            has_events = False

            for source in self._sources.values():
                event = self._get_event(source)
                if event:
                    self._registry.dispatch(event)
                    has_events = True
                    events_processed += 1

            if not has_events:
                time.sleep(0.001)

        duration = time.time() - start_time
        efficiency = events_processed / iterations if iterations > 0 else 0
        logger.info(
            f"EventProcessor stats: {iterations} iterations, "
            f"{events_processed} events ({efficiency:.1%} efficiency), "
            f"{duration:.1f}s runtime"
        )
```

**Interpretation**:
- `efficiency < 1%`: Consider adaptive backoff
- `efficiency > 10%`: Current implementation is fine
- `runtime > 1 hour` + `efficiency < 1%`: Definitely optimize

