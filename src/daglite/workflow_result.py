"""WorkflowResult: container for multi-sink workflow outputs."""

from __future__ import annotations

from collections.abc import ItemsView
from collections.abc import Mapping
from collections.abc import ValuesView
from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

from daglite.exceptions import AmbiguousResultError


@dataclass(frozen=True)
class WorkflowResult(Mapping[str, Any]):
    """
    Holds the evaluated outputs of a ``@workflow``, indexable by task name or UUID.

    Access results by name (``result["task_name"]``) or by UUID
    (``result[uuid]``).  If two sink nodes share the same name, name-based
    lookup raises ``AmbiguousResultError``; UUID-based lookup always works.
    """

    _results: dict[UUID, Any]
    """Primary store: UUID → result value."""

    _by_name: dict[str, list[UUID]]
    """Secondary index: task name → list of UUIDs (typically one, but may be many)."""

    @classmethod
    def build(cls, results: dict[UUID, Any], name_for: dict[UUID, str]) -> WorkflowResult:
        """Build a WorkflowResult from raw results and a uuid→name mapping."""
        by_name: dict[str, list[UUID]] = {}
        for uid, name in name_for.items():
            by_name.setdefault(name, []).append(uid)
        return cls(_results=results, _by_name=by_name)

    def __getitem__(self, key: str | UUID) -> Any:
        if isinstance(key, UUID):
            try:
                return self._results[key]
            except KeyError:
                raise KeyError(f"No workflow output with UUID {key!r}")
        uuids = self._by_name.get(key)
        if not uuids:
            raise KeyError(f"No workflow output named {key!r}")
        if len(uuids) > 1:
            raise AmbiguousResultError(
                f"Multiple sink nodes named {key!r}: {uuids}. "
                f"Use a UUID to disambiguate: result[uuid]"
            )
        return self._results[uuids[0]]

    def single(self, name: str) -> Any:
        """
        Return the result for ``name``, asserting there is exactly one.

        Equivalent to ``result[name]`` — raises ``AmbiguousResultError`` if
        multiple sinks share this name, or ``KeyError`` if none do.  The method
        name makes the intent explicit when reading code.

        Args:
            name: Task name or alias to look up.

        Returns:
            The evaluated output of the single matching sink node.

        Raises:
            KeyError: If no sink with this name exists.
            AmbiguousResultError: If multiple sinks share this name.
        """
        return self[name]

    def all(self, name: str) -> list[Any]:
        """
        Return all results for ``name`` as a list.

        Unlike ``result[name]``, this never raises ``AmbiguousResultError``.
        Useful for fan-out patterns where multiple sink nodes intentionally
        share the same name and you want all of their values.

        Args:
            name: Task name or alias to look up.

        Returns:
            A list of evaluated outputs for all matching sink nodes.
            Returns an empty list if no sink with this name exists.
        """
        return [self._results[uid] for uid in self._by_name.get(name, [])]

    def values(self) -> ValuesView[Any]:
        """Iterate over all results in name-index order, expanding duplicate-named sinks."""
        return _WorkflowValuesView(self)

    def items(self) -> ItemsView[str, Any]:
        """
        Iterate over (name, value) pairs, expanding duplicate-named sinks.

        Unlike ``result[name]``, this never raises ``AmbiguousResultError``.
        Duplicate-named sinks each appear as a separate ``(name, value)`` pair.
        """
        return _WorkflowItemsView(self)

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __repr__(self) -> str:
        names = list(self._by_name)
        return f"WorkflowResult({names})"


class _WorkflowValuesView(ValuesView[Any]):
    """ValuesView that expands duplicate-named sinks instead of raising."""

    _mapping: WorkflowResult  # type: ignore[assignment]

    def __iter__(self) -> Iterator[Any]:
        for uuids in self._mapping._by_name.values():
            for uid in uuids:
                yield self._mapping._results[uid]

    def __len__(self) -> int:
        return len(self._mapping._results)


class _WorkflowItemsView(ItemsView[str, Any]):
    """ItemsView that expands duplicate-named sinks instead of raising."""

    _mapping: WorkflowResult  # type: ignore[assignment]

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        for name, uuids in self._mapping._by_name.items():
            for uid in uuids:
                yield name, self._mapping._results[uid]

    def __len__(self) -> int:
        return len(self._mapping._results)
