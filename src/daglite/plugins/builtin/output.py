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

    Intercepts after_node_execute and saves outputs for `.save()` calls.
    """

    def __init__(self, store: OutputStore | str | None = None) -> None:
        """
        Initialize with optional default store.

        Args:
            store: Default output store. Can be an OutputStore instance or a string path
                (which will be converted to FileOutputStore). If None, stores must be
                provided at task or explicit level.

        Examples:
            >>> # String shortcut
            >>> plugin = OutputPlugin(store="/tmp/outputs")  # doctest: +SKIP
            >>>
            >>> # Explicit OutputStore
            >>> from daglite.outputs.store import FileOutputStore
            >>> plugin = OutputPlugin(store=FileOutputStore("/tmp/outputs"))  # doctest: +SKIP
        """
        if isinstance(store, str):
            from daglite.outputs.store import FileOutputStore

            self.store = FileOutputStore(store)
        else:
            self.store = store

    @hook_impl
    def after_node_execute(
        self,
        metadata: GraphMetadata,
        inputs: dict[str, Any],
        result: Any,
        output_config: tuple[OutputConfig, ...],
        output_extras: list[dict[str, Any]],
        duration: float,
        reporter: EventReporter | None,
    ) -> None:
        """Save outputs after successful node execution."""
        if not output_config:
            return

        for config, extras in zip(output_config, output_extras):
            # Format key with inputs
            try:
                formatted_key = config.key.format(**inputs)
            except KeyError as e:
                raise ValueError(
                    f"Output key '{config.key}' references parameter {e} which is not in inputs"
                ) from e

            # Determine which store to use (config store takes precedence over plugin default)
            store = config.store or self.store
            if store is None:
                raise ValueError(
                    f"No output store configured for saving '{formatted_key}' "
                    f"in task '{metadata.name}'. "
                    f"\nFix by adding a store at one of these levels:"
                    f"\n  1. OutputPlugin: plugins=[OutputPlugin(store='/path/to/outputs')]"
                    f"\n  2. Task decorator: @task(store='/path/to/outputs')"
                    f"\n  3. Explicit save: .save('{config.key}', store='/path/to/outputs')"
                )

            # Save the result (extras are already resolved and available if needed by store)
            store.save(key=formatted_key, value=result)

            if reporter:
                reporter.report(
                    "daglite-output-saved",
                    {
                        "key": formatted_key,
                        "checkpoint_name": config.name,
                        "node_id": metadata.id,
                        "node_name": metadata.name,
                    },
                )


__all__ = ["OutputPlugin"]
