# daglite-serialization

Serialization plugin for [daglite](https://github.com/cswartzvi/daglite) with support for popular scientific Python libraries.

## Features

- **NumPy**: Fast sample-based hashing for large arrays (<100ms for 800MB)
- **Pandas**: Schema + sample hashing for DataFrames and Series
- **Pillow**: Thumbnail-based hashing for images

## Installation

```bash
# Install with all plugins
pip install daglite-serialization[all]

# Or install specific plugins
pip install daglite-serialization[numpy]
pip install daglite-serialization[pandas]
pip install daglite-serialization[pillow]
```

## Usage

### Automatic Registration

Import and register all available handlers:

```python
from daglite_serialization import register_all

# Register all available plugins (numpy, pandas, pillow)
register_all()
```

### Selective Registration

Register only specific plugins:

```python
from daglite_serialization.numpy import register_numpy_handlers
from daglite_serialization.pandas import register_pandas_handlers

register_numpy_handlers()
register_pandas_handlers()
```

### With Caching

Once registered, types work automatically with daglite caching:

```python
from daglite import task
from daglite_serialization import register_all
import numpy as np

register_all()

@task(cache=True)
def process_array(arr: np.ndarray) -> np.ndarray:
    return arr * 2

# Cache key includes fast hash of array
result = process_array(np.random.rand(10000, 10000)).run()
```

## Hash Strategies

### NumPy Arrays

- **Small arrays (<10k elements)**: Full hash for correctness
- **Large arrays**: Sample first/middle/last portions (~0.2% of data)
- **Performance**: <100ms for 800MB arrays (vs ~2000ms for full hash)

### Pandas DataFrames

- **Small DataFrames (<1k rows)**: Full hash using `pd.util.hash_pandas_object`
- **Large DataFrames**: Hash schema + first/last 500 rows
- **Performance**: ~10ms for 1M rows (vs ~5000ms for full hash)

### Pillow Images

- **Strategy**: Downsample to 32x32 thumbnail + metadata
- **Performance**: Fast for any image size, catches all visible changes

## Custom Hash Strategies

You can override the default strategies:

```python
from daglite.serialization import default_registry
import hashlib

# Full hash instead of sampling
def hash_numpy_array_full(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()

default_registry.register_hash_strategy(
    np.ndarray,
    hash_numpy_array_full,
    "Full hash of numpy array"
)
```

## Development

```bash
# Install in development mode
pip install -e ".[dev,test,all]"

# Run tests
pytest

# Format code
ruff format src tests

# Lint
ruff check src tests
```

## License

MIT
