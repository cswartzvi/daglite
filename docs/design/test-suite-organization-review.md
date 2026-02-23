# Test Suite Organization Review

This document reviews the current pytest layout and proposes a clearer structure for collocation, naming, and class organization.

## Why this review

The suite already has strong coverage, but some tests are currently grouped by *how they are invoked* (for example, `.run()` / `.run_async()`) instead of by the API capability they validate. That creates overlap between top-level `tests/` and `tests/integration/`, and makes it harder to find the canonical place for behavior tests.

## Current structure observations

### 1) `tests/integration/` is doing two different jobs

`tests/integration/` currently contains both:

- true cross-component tests (plugins + backends + datasets + retries + logging), and
- many API behavior tests for task-future execution and fluent chaining.

This makes the folder conceptually broad: "integration" often means environment or system boundaries, while many of these tests are behavior-level for a single API surface.

### 2) Test location guidance is currently inconsistent

`tests/test_tasks_futures.py` contains this guidance:

> "Tests in this file should NOT focus on evaluation. Evaluation tests are in tests/evaluation/."

But there is no `tests/evaluation/` package today; evaluation behavior now lives in `tests/behavior/test_run_sync.py` and `tests/behavior/test_run_async.py`.

This mismatch is a concrete signal that the test taxonomy evolved but naming did not.

### 3) Similar domains are split across multiple directories

Examples:

- Fluent API execution tests are in `tests/behavior/test_fluent_api.py`, while related node/future behavior is in `tests/test_iter_future.py`, `tests/test_reduce_future.py`, and others.
- Iteration behavior appears both as future-level tests (`tests/test_iter_future.py`) and execution-level tests (`tests/behavior/test_iter_execution.py`), which is valid but not obvious without conventions.

### 4) Pytest test classes are used heavily (good), but with uneven granularity

The suite uses class-based grouping consistently, which improves readability. However:

- some files contain very large classes spanning many concerns,
- class names occasionally encode implementation details rather than behavior slices,
- and there are small naming collisions (`TestIterValidation` appears in multiple files), which increases navigation friction.

## Recommended target model

Use a **capability-first + level-second** organization.

### Proposed directory taxonomy

- `tests/unit/`
  - Pure object/model/API-contract tests without runtime evaluation orchestration.
  - Example topics: task definition validation, node remapping, config resolution logic.

- `tests/behavior/`
  - Behavior-level execution tests through public APIs (what you currently called some "integration" tests but are really API behavior tests).
  - Example topics: `TaskFuture.run()`, `TaskFuture.run_async()`, fluent chaining semantics, map/reduce/iter end-to-end behavior.

- `tests/integration/`
  - True integration boundaries: process backend serialization, plugin lifecycle with execution engine, dataset store + backend interactions, logging plugin with reporter stack, CLI command surface.

- `tests/contracts/` (optional)
  - Stable behavior guarantees (error shapes/messages, public type invariants) when you want explicit public API compatibility tests.
  - **Typing tests are a strong fit here** when they validate the published typing contract (`TaskFuture[T]`, fluent chaining inference, map/reduce generic propagation) rather than implementation details.

### Where to put typing tests

Short answer: **yes, `tests/contracts` can contain typing tests**.

A practical split:

- `tests/contracts/typing/`
  - Tests that lock public type behavior for users and should remain stable across refactors.
  - Examples: expected inference at API boundaries, regressions in generic return types, checker parity assertions.

- `tests/unit/typing/` (or keep helper assets under `tests/typing/`)
  - Internal typing helpers/fixtures or checker-specific utilities that support contract tests.
  - Files used only as scaffolding for tools (mypy/pyright/pyrefly/ty) can remain colocated with contract tests or in a small helpers package.

Given the current repository, a low-risk first step would be:

- the typing contract tests now live in `tests/contracts/typing/`,
- keep checker helpers and contract files colocated in that package,
- add helper-only fixtures under `tests/unit/typing/` only when needed.

If you prefer fewer directories, use `tests/core/` + `tests/integration/`; the key is to stop overloading "integration" for all evaluated graphs.

### Migration mapping (minimal churn)

- ✅ `tests/integration/test_tasks_sync.py` -> `tests/behavior/test_run_sync.py`
- ✅ `tests/integration/test_tasks_async.py` -> `tests/behavior/test_run_async.py`
- ✅ `tests/integration/test_fluent_api.py` -> `tests/behavior/test_fluent_api.py`
- ✅ `tests/integration/test_iter.py` -> `tests/behavior/test_iter_execution.py`
- Keep in `tests/integration/`: `test_backends.py`, `test_dataset_backends.py`, `test_logging.py`, `test_plugins.py`, and any file that validates multiple subsystems together.

This gives immediate collocation wins without changing assertions.

## Guidance for pytest class organization

You can keep classes as the primary structure, but use a simple convention:

1. **Class = one behavior slice**
   - Good: `TestRunSyncSinglePath`, `TestRunSyncParallelSiblings`, `TestRunSyncFailurePropagation`
   - Avoid: one class covering all sync execution behavior across unrelated concerns.

2. **3–10 tests per class target**
   - If a class grows beyond ~15 tests, split by behavior axis (success paths, errors, edge cases, backend-specific behavior).

3. **Name classes by outcome, not mechanism**
   - Prefer `TestRetryPolicy` over `TestRetriesBranchCoverage`.
   - Coverage intent belongs in comments/PR notes, not class names.

4. **Mirror sync/async structure where possible**
   - Parallel class trees improve discoverability:
     - `TestRunSyncSinglePath` / `TestRunAsyncSinglePath`
     - `TestRunSyncErrorPropagation` / `TestRunAsyncErrorPropagation`

5. **Keep backend-specific execution in dedicated classes**
   - e.g., `TestRunAsyncProcessBackendPickleConstraints`.

6. **Use shared helper fixtures instead of mega-classes**
   - Keep fixture intent in `conftest.py` or local file fixtures.
   - Avoid classes that exist mainly to share setup state unless state is behavior-relevant.

## Low-risk rollout plan

1. **Phase 1: taxonomy + renames** ✅
   - Files moved to `tests/behavior/` and `tests/contracts/typing/`.
   - Stale comments/docstrings updated away from `tests/evaluation/`.

2. **Phase 2: class shape cleanup** ✅ (incremental)
   - Class names were normalized toward behavior intent and sync/async mirroring in run behavior suites.
   - Follow-up splits can still be done for very large classes if desired.

3. **Phase 3: markers** ✅
   - Added `behavior`, `integration`, `contracts`, and `typing_contract` markers.
   - Marker assignment is automatic during collection based on test path, enabling CI partitioning without per-test decorator churn.

## Suggested acceptance criteria for the refactor

- A contributor can find all `run()/run_async()` behavior tests under a single non-integration location.
- `tests/integration/` only contains cross-subsystem tests.
- No stale references to non-existent test directories.
- Class names are unique within their file and descriptive of behavior intent.

## Optional next step

If helpful, the next PR can be a pure test-layout refactor with no assertion changes, so risk stays low and git history is easy to review.
