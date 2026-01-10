"""Output plugin for automatic save/checkpoint handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from daglite.plugins.hooks.markers import hook_impl

if TYPE_CHECKING:
    from daglite.graph.base import GraphMetadata
    from daglite.graph.base import OutputConfig
    from daglite.outputs.base import OutputStore
    from daglite.plugins.reporters import EventReporter


class OutputPlugin:
    """
    Automatically saves outputs based on OutputConfig.

    Intercepts after_node_execute and saves outputs for .save() and .checkpoint() calls.
    """

    def __init__(self, store: OutputStore | None = None) -> None:
        """Initialize with optional default store."""
        self.store = store

    @hook_impl
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        output_config: tuple[OutputConfig, ...],
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """Save outputs after successful node execution."""
        if not output_config:
            return

        for config in output_config:
            try:
                formatted_key = config.key.format(**inputs)
            except KeyError as e:
                raise ValueError(
                    f"Output key '{config.key}' references parameter {e} which is not in inputs"
                ) from e

            # Determine which store to use
            store = config.store or self.store
            if store is None:
                raise ValueError(
                    f"No output store configured for node '{metadata.name}'. "
                    "Provide store via OutputPlugin(store=...), @task(store=...), "
                    "or .save(store=...)"
                )

            # TODO: Support extra refs (requires access to completed_nodes)
            extras = {}
            for extra_name, param in config.extras.items():
                if not param.is_ref:
                    extras[extra_name] = param.value

            store.save(key=formatted_key, value=result)

            if reporter:
                reporter.report(
                    "output_saved",
                    {
                        "key": formatted_key,
                        "checkpoint_name": config.name,
                        "node_id": metadata.id,
                        "node_name": metadata.name,
                    },
                )


__all__ = ["OutputPlugin"]
