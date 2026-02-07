# Plugins

Daglite's plugin system allows you to extend functionality without adding mandatory dependencies to the core library.

## Plugin Types

### Built-in Plugins

Plugins shipped with the core `daglite` package that require no additional installation.

**Centralized Logging** - Cross-process/thread logging that works seamlessly with threading, multiprocessing, and distributed execution.

[→ Learn more about built-in plugins](built-in.md)

### Extension Plugins

Optional packages that extend Daglite with additional capabilities.

#### daglite-cli

Command-line interface for running pipelines from the terminal.

- Run pipelines with `daglite run`
- Pass parameters via command line
- Choose execution backends (Inline, threading, multiprocessing)
- Enable async execution for parallel task execution

[→ Learn more about daglite-cli](cli.md)

#### daglite-serialization

Serialization and hashing support for scientific Python libraries.

- Fast hashing for NumPy arrays (sample-based for large arrays)
- Efficient hashing for Pandas DataFrames (schema + sample)
- Image hashing with Pillow (thumbnail-based)
- Custom hash strategies

[→ Learn more about daglite-serialization](serialization.md)

---

## Installation

### Built-in Plugins

Built-in plugins are available immediately after installing `daglite`:

```bash
uv pip install daglite
```

No additional installation needed!

### Extension Plugins

Install extension plugins as needed:

#### Install All Extension Plugins

```bash
uv pip install daglite[cli] daglite-serialization[all]
```

#### Install Specific Extension Plugins

```bash
# CLI only
uv pip install daglite[cli]

# Serialization with specific libraries
uv pip install daglite-serialization[numpy]
uv pip install daglite-serialization[pandas]
uv pip install daglite-serialization[pillow]
```

---

## Plugin Philosophy

Daglite's core library has minimal dependencies to ensure it runs anywhere Python runs. Plugins extend this core with optional functionality:

- **No mandatory overhead** - Only install what you need
- **Seamless integration** - Plugins work automatically once installed
- **Type-safe** - Full type checking support across plugins
- **Modular** - Mix and match plugins based on your requirements

---

## Creating Custom Plugins

Want to create your own Daglite plugin? See the [Creating Plugins](creating.md) guide to learn how to extend Daglite using the pluggy-based hook system.

---

## Future Plugins

We're planning additional plugins for:

- Distributed execution (Dask, Ray)
- Cloud storage backends (S3, GCS, Azure)
- Workflow visualization
- Advanced caching strategies

Have an idea for a plugin? [Open a discussion](https://github.com/cswartzvi/daglite/discussions) or contribute!
