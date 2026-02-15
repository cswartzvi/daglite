# Rich Output Plugin

!!! note "Coming Soon"
    This page is under construction. A detailed guide is planned covering the topics outlined below.

The `daglite-rich` plugin provides enhanced terminal output using the [Rich](https://rich.readthedocs.io/) library, including formatted logging and progress bars for DAG execution.

---

## Installation

```bash
uv pip install daglite-rich
```

Or install with the core library:

```bash
uv pip install daglite[rich]
```

---

## Planned Content

### Rich Logging

- Enhanced log formatting with colors and styling
- Integration with the centralized logging plugin
- Configuration options

### Progress Bars

- Real-time progress tracking during DAG evaluation
- Task-level progress indicators
- Customizing progress display

### Usage with Pipelines

- Automatic progress display for CLI pipelines
- Combining with other plugins

---

## See Also

- [Built-in Plugins](built-in.md) - Centralized logging
- [Creating Plugins](creating.md) - Build custom plugins
- [Rich documentation](https://rich.readthedocs.io/) - Rich library reference
