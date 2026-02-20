"""WorkflowResult: container for multi-sink workflow outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
from uuid import UUID

from daglite.exceptions import AmbiguousResultError


@dataclass(frozen=True)
class WorkflowResult:
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
    def _build(cls, results: dict[UUID, Any], name_for: dict[UUID, str]) -> WorkflowResult:
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

    def keys(self) -> Iterator[str]:
        """Iterate over sink node names."""
        return iter(self._by_name)

    def values(self) -> Iterator[Any]:
        """Iterate over results in name-index order."""
        for uuids in self._by_name.values():
            for uid in uuids:
                yield self._results[uid]

    def items(self) -> Iterator[tuple[str, Any]]:
        """
        Iterate over (name, value) pairs.

        Raises:
            AmbiguousResultError: If two sink nodes share the same name.
        """
        for name in self._by_name:
            yield name, self[name]

    def __repr__(self) -> str:
        names = list(self._by_name)
        return f"WorkflowResult({names})"
