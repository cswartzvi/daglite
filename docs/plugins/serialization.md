# Serialization Plugin

The `daglite-serialization` plugin provides fast, efficient hashing and serialization support for popular scientific Python libraries. This is essential for caching workflows that use NumPy arrays, Pandas DataFrames, or Pillow images.

## Installation

```bash
# Install with all supported libraries
pip install daglite-serialization[all]

# Or install specific libraries
pip install daglite-serialization[numpy]
pip install daglite-serialization[pandas]
pip install daglite-serialization[pillow]
```

---

## Quick Start

### Automatic Registration

Register all available handlers at once:

```python
from daglite import task
from daglite_serialization import register_all
import numpy as np

# Register all plugins (numpy, pandas, pillow)
register_all()

@task(cache=True)
def process_array(arr: np.ndarray) -> np.ndarray:
    """Cache key automatically includes fast hash of array."""
    return arr * 2

# Subsequent calls with same array use cached result
result = process_array(np.random.rand(10000, 10000)).run()
```

### Selective Registration

Register only the handlers you need:

```python
from daglite_serialization.numpy import register_handlers as register_numpy
from daglite_serialization.pandas import register_handlers as register_pandas

register_numpy()
register_pandas()
```

---

## Supported Libraries

### NumPy

Fast hashing for NumPy arrays using sampling strategies.

**Hash Strategies:**

- **Small arrays** (<10k elements): Full hash for complete accuracy
- **Large arrays** (≥10k elements): Sample-based hash (~0.2% of data)
  - Samples from beginning, middle, and end
  - Includes shape and dtype metadata
  - Performance: <100ms for 800MB arrays (vs ~2000ms for full hash)

**Example:**

```python
import numpy as np
from daglite import task
from daglite_serialization import register_all

register_all()

@task(cache=True)
def expensive_computation(matrix: np.ndarray) -> float:
    """Cached computation on large arrays."""
    return np.linalg.det(matrix)

# First call computes and caches
result1 = expensive_computation(np.random.rand(1000, 1000)).run()

# Second call with same array uses cache
result2 = expensive_computation(np.random.rand(1000, 1000)).run()
```

---

### Pandas

Efficient hashing for DataFrames and Series using schema and sampling.

**Hash Strategies:**

- **Small DataFrames** (<1k rows): Full hash using `pd.util.hash_pandas_object`
- **Large DataFrames** (≥1k rows): Hash schema + first/last 500 rows
  - Includes column names, dtypes, and index
  - Performance: ~10ms for 1M rows (vs ~5000ms for full hash)

**Example:**

```python
import pandas as pd
from daglite import task
from daglite_serialization import register_all

register_all()

@task(cache=True)
def aggregate_data(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Cached aggregation operation."""
    return df.groupby(group_by).agg({
        'value': ['mean', 'std', 'count']
    })

# Works with large DataFrames
df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'value': np.random.randn(1000000)
})

result = aggregate_data(df, group_by='category').run()
```

**Supported Types:**

- `pd.DataFrame`
- `pd.Series`
- `pd.Index`

---

### Pillow (PIL)

Image hashing using thumbnail downsampling.

**Hash Strategy:**

- Downsample to 32×32 thumbnail
- Include image metadata (mode, size)
- Fast for any image size
- Catches all visually significant changes

**Example:**

```python
from PIL import Image
from daglite import task
from daglite_serialization import register_all

register_all()

@task(cache=True)
def apply_filter(img: Image.Image, blur_radius: int) -> Image.Image:
    """Cached image filtering."""
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

img = Image.open('photo.jpg')
filtered = apply_filter(img, blur_radius=5).run()
```

---

## Custom Hash Strategies

Override default strategies for specific types:

```python
from daglite.serialization import default_registry
import hashlib
import numpy as np

def hash_numpy_array_full(arr: np.ndarray) -> str:
    """Full hash instead of sampling (slower but exact)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()

# Register custom strategy
default_registry.register_hash_strategy(
    np.ndarray,
    hash_numpy_array_full,
    "Full hash of numpy array"
)
```

### Custom Type Registration

Register hash strategies for your own types:

```python
from dataclasses import dataclass
import hashlib

@dataclass
class CustomData:
    matrix: np.ndarray
    metadata: dict

def hash_custom_data(data: CustomData) -> str:
    """Hash custom data type."""
    # Combine hashes from components
    matrix_hash = default_registry.hash(data.matrix)
    metadata_hash = hashlib.sha256(
        str(sorted(data.metadata.items())).encode()
    ).hexdigest()
    return hashlib.sha256(
        f"{matrix_hash}:{metadata_hash}".encode()
    ).hexdigest()

default_registry.register_hash_strategy(
    CustomData,
    hash_custom_data,
    "Hash CustomData by components"
)
```

---

## Performance Characteristics

### NumPy Arrays

| Array Size | Elements | Full Hash Time | Sample Hash Time | Speedup |
|------------|----------|----------------|------------------|---------|
| 10 MB      | 1M       | ~20ms          | ~5ms             | 4×      |
| 100 MB     | 10M      | ~200ms         | ~8ms             | 25×     |
| 800 MB     | 100M     | ~2000ms        | ~100ms           | 20×     |

### Pandas DataFrames

| Rows    | Columns | Full Hash Time | Sample Hash Time | Speedup |
|---------|---------|----------------|------------------|---------|
| 1K      | 10      | ~5ms           | ~5ms             | 1×      |
| 100K    | 10      | ~500ms         | ~8ms             | 62×     |
| 1M      | 10      | ~5000ms        | ~10ms            | 500×    |

### Pillow Images

| Image Size  | Pixels   | Hash Time |
|-------------|----------|-----------|
| 640×480     | 307K     | ~5ms      |
| 1920×1080   | 2M       | ~10ms     |
| 4096×4096   | 16M      | ~15ms     |

All benchmarks on modern CPU. Actual performance varies by hardware.

---

## When to Use Sampling

### Safe to Use Sampling:

- Large datasets where small changes are unlikely
- Intermediate results in data pipelines
- Arrays/DataFrames that rarely change
- Development and testing workflows

### Use Full Hashing When:

- Exact cache invalidation is critical
- Working with adversarial data
- Financial or scientific precision required
- Small datasets where performance doesn't matter

**Trade-off:** Sampling provides 10-500× speedup but has a small risk of hash collisions if only sampled regions differ.

---

## Best Practices

### 1. Register Early

Register handlers at the start of your script or in a setup module:

```python
# myproject/setup.py
from daglite_serialization import register_all

register_all()
```

```python
# myproject/pipeline.py
from myproject.setup import *  # Handlers registered
from daglite import task

@task(cache=True)
def process(data):
    ...
```

### 2. Use Type Annotations

Help type checkers understand your code:

```python
import numpy as np
from numpy.typing import NDArray

@task(cache=True)
def process(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return arr * 2
```

### 3. Document Hash Strategies

When using custom strategies, document your choice:

```python
# Using full hash for financial data - exact cache invalidation required
default_registry.register_hash_strategy(
    FinancialData,
    hash_financial_data_full,
    "Full hash for financial accuracy"
)
```

### 4. Test Cache Behavior

Verify caching works as expected:

```python
import numpy as np
from daglite import task
from daglite_serialization import register_all

register_all()

@task(cache=True)
def slow_operation(arr: np.ndarray) -> float:
    import time
    time.sleep(1)  # Simulate slow operation
    return arr.sum()

# First call takes ~1 second
arr = np.ones(1000)
result1 = slow_operation(arr).run()

# Second call with same array returns immediately
result2 = slow_operation(arr).run()
assert result1 == result2
```

---

## Troubleshooting

### "No hash strategy registered" Error

Register the appropriate handler:

```python
from daglite_serialization.numpy import register_handlers

register_handlers()  # Now np.ndarray hashing works
```

### Hash Collisions

If you suspect hash collisions with sampling:

```python
# Switch to full hashing for specific type
from daglite.serialization import default_registry
import hashlib
import numpy as np

def full_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()

default_registry.register_hash_strategy(np.ndarray, full_hash, "Full hash")
```

### Performance Issues

If hashing is slow:

1. Check array/DataFrame sizes
2. Consider if caching is beneficial
3. Profile to confirm hashing is the bottleneck
4. Adjust sampling strategy if needed

---

## See Also

- [Tasks User Guide](../user-guide/tasks.md) - Learn about task caching
- [Creating Plugins](creating.md) - Create custom serialization plugins
- [Serialization Source Code](https://github.com/cswartzvi/daglite/tree/main/extras/serialization) - Implementation details
