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

            # Resolve extras from ParamInputs
            extras = {}
            for extra_name, param in config.extras.items():
                if param.kind == "value":
                    # Literal value
                    extras[extra_name] = param.value
                elif param.kind == "ref":
                    # Reference - check if the extra name matches an input name
                    if extra_name in inputs:
                        # The extra references a task parameter - use its resolved value
                        extras[extra_name] = inputs[extra_name]
                    else:
                        # The extra is a ref but not a task parameter
                        raise ValueError(
                            f"Cannot resolve extra parameter '{extra_name}' "
                            f"for output '{formatted_key}'. "
                            f"Extra '{extra_name}' references a task output but is not "
                            f"a parameter of '{metadata.name}'. "
                            f"To use task outputs in extras, pass them as parameters to the task."
                        )
                else:
                    raise ValueError(
                        f"Unsupported ParamInput kind '{param.kind}' for extra '{extra_name}'"
                    )

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
