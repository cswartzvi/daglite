# Typing Tests

These tests validate Daglite's **public typing contract** (not just internal implementation details).

## Scope

This directory covers expectations that users rely on when using type checkers with Daglite APIs, such as:

- generic propagation across `TaskFuture[T]` flows,
- fluent API inference (`then`, `map`, `join`, `reduce`),
- checker parity and regressions across supported tools.

## Why this matters

Even if runtime behavior is unchanged, typing regressions can break user workflows and IDE assistance.
Treat these tests as compatibility/contract coverage for the public API surface.

## Layout

These files now live in `tests/contracts/typing/` to make their contract role explicit.
Use this directory as the canonical location for typing contract tests.
