"""Typed event container for the plugin event system."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class Event:
    """
    Typed event for worker → coordinator communication.

    Attributes:
        type: Event type identifier (e.g. ``"daglite-logging-node-start"``).
        data: Arbitrary payload data.  Plugin-specific — the exact keys depend
            on the event type.

    Examples:
        >>> event = Event(type="progress", data={"percent": 100})
        >>> event.type
        'progress'
        >>> event.data["percent"]
        100
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
