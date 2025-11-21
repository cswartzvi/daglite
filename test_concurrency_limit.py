"""Test concurrency limiting in ThreadBackend."""

import threading
import time
from collections import defaultdict

from daglite import evaluate, task
from daglite.backends.local import ThreadBackend
from daglite.settings import DagliteSettings, set_global_settings

# Track concurrent executions
concurrent_count = 0
max_concurrent = 0
lock = threading.Lock()


@task(backend="threading")
def track_concurrency(x: int, delay: float = 0.1) -> int:
    """Track how many tasks run concurrently."""
    global concurrent_count, max_concurrent

    with lock:
        concurrent_count += 1
        if concurrent_count > max_concurrent:
            max_concurrent = concurrent_count
        current = concurrent_count

    time.sleep(delay)

    with lock:
        concurrent_count -= 1

    return x * 2


def test_no_limit():
    """Test with no concurrency limit."""
    global max_concurrent, concurrent_count
    max_concurrent = 0
    concurrent_count = 0

    print("=== Test: No Concurrency Limit ===")

    # Set global pool to 10 threads
    set_global_settings(DagliteSettings(max_backend_threads=10))

    # Submit 10 tasks directly to backend
    backend = ThreadBackend(max_workers=None)
    calls = [{"x": i, "delay": 0.1} for i in range(10)]

    start = time.time()
    results = backend.run_many(track_concurrency.fn, calls)
    elapsed = time.time() - start

    print(f"Processed {len(results)} tasks in {elapsed:.2f}s")
    print(f"Max concurrent tasks: {max_concurrent}")
    print("Expected: ~10 concurrent (all at once)")
    print()

    assert max_concurrent >= 8, f"Expected ~10 concurrent, got {max_concurrent}"


def test_with_limit():
    """Test with concurrency limit of 3."""
    global max_concurrent, concurrent_count
    max_concurrent = 0
    concurrent_count = 0

    print("=== Test: Concurrency Limit = 3 ===")

    # Set global pool to 10 threads
    set_global_settings(DagliteSettings(max_backend_threads=10))

    # Create backend with max_workers=3
    backend = ThreadBackend(max_workers=3)

    # Submit 10 tasks directly to backend
    calls = [{"x": i, "delay": 0.1} for i in range(10)]

    start = time.time()
    results = backend.run_many(track_concurrency.fn, calls)
    elapsed = time.time() - start

    print(f"Processed {len(results)} tasks in {elapsed:.2f}s")
    print(f"Max concurrent tasks: {max_concurrent}")
    print(f"Expected: 3 concurrent (limited by max_workers)")
    print()

    # With limit of 3, max concurrent should be close to 3
    assert max_concurrent <= 4, f"Expected max 3-4 concurrent, got {max_concurrent}"
    assert max_concurrent >= 2, f"Expected at least 2-3 concurrent, got {max_concurrent}"


def test_timing_difference():
    """Test that limiting concurrency increases execution time."""
    global max_concurrent, concurrent_count

    print("=== Test: Timing Comparison ===")
    set_global_settings(DagliteSettings(max_backend_threads=10))

    # Test 1: No limit (should be fast)
    max_concurrent = 0
    concurrent_count = 0
    backend_unlimited = ThreadBackend(max_workers=None)
    calls = [{"x": i, "delay": 0.1} for i in range(10)]

    start = time.time()
    backend_unlimited.run_many(track_concurrency.fn, calls)
    time_unlimited = time.time() - start
    print(f"Unlimited: {time_unlimited:.2f}s (max concurrent: {max_concurrent})")

    # Test 2: Limited to 2 workers (should be slower)
    max_concurrent = 0
    concurrent_count = 0
    backend_limited = ThreadBackend(max_workers=2)

    start = time.time()
    backend_limited.run_many(track_concurrency.fn, calls)
    time_limited = time.time() - start
    print(f"Limited to 2: {time_limited:.2f}s (max concurrent: {max_concurrent})")

    print(f"\nWith 10 tasks at 0.1s each:")
    print(f"  - Unlimited should take ~0.1s (all parallel)")
    print(f"  - Limited to 2 should take ~0.5s (10 tasks / 2 workers * 0.1s)")
    print()

    # Limited should take significantly longer
    assert time_limited > time_unlimited * 2, "Limited should be slower than unlimited"


if __name__ == "__main__":
    test_no_limit()
    test_with_limit()
    test_timing_difference()
    print("=== All Concurrency Limit Tests Passed! ===")
