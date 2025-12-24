# Plugins

Daglite's plugin system allows you to extend functionality without adding mandatory dependencies to the core library. Plugins are optional packages that integrate seamlessly with Daglite's architecture.

## Available Plugins

### daglite-cli

Command-line interface for running pipelines from the terminal.

- Run pipelines with `daglite run`
- Pass parameters via command line
- Choose execution backends (sequential, threading, multiprocessing)
- Enable async execution for parallel task execution

[→ Learn more about daglite-cli](cli.md)

### daglite-serialization

Serialization and hashing support for scientific Python libraries.

- Fast hashing for NumPy arrays (sample-based for large arrays)
- Efficient hashing for Pandas DataFrames (schema + sample)
- Image hashing with Pillow (thumbnail-based)
- Custom hash strategies

[→ Learn more about daglite-serialization](serialization.md)

---

## Installation

### Install All Plugins

```bash
pip install daglite[cli] daglite-serialization[all]
```

### Install Specific Plugins

```bash
# CLI only
pip install daglite[cli]

# Serialization with specific libraries
pip install daglite-serialization[numpy]
pip install daglite-serialization[pandas]
pip install daglite-serialization[pillow]
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
