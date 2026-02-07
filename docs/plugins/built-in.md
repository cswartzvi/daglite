# Built-in Plugins

Daglite ships with default plugins that are part of the core library. These plugins provide essential functionality without requiring additional dependencies.

## Centralized Logging

The `CentralizedLoggingPlugin` enables logging that works seamlessly across different execution backends—threading, multiprocessing, and eventually distributed execution.

### Why Centralized Logging?

When running tasks across multiple processes or threads, standard Python logging can be problematic:

- **Multiprocessing**: Logs from worker processes don't reach the main process
- **Threading**: Race conditions can garble log output
- **Distributed**: Logs scattered across machines are hard to correlate

Daglite's centralized logging solves this by routing all logs through the event reporter system to a single coordinator process.

---

## Quick Start

### Basic Usage

```python
from daglite import task, evaluate
from daglite.plugins import CentralizedLoggingPlugin, get_logger
import logging

# Configure Python logging (optional)
logging.basicConfig(
    format="%(daglite_task_name)s [%(levelname)s] %(message)s",
    level=logging.INFO
)

@task
def process_data(x: int) -> int:
    logger = get_logger(__name__)
    logger.info(f"Processing {x}")
    result = x * 2
    logger.info(f"Result: {result}")
    return result

# Add plugin to enable centralized logging
plugin = CentralizedLoggingPlugin(level=logging.INFO)

result = evaluate(
    process_data.product(x=[1, 2, 3]),
    plugins=[plugin]
)
# Output:
# process_data [INFO] Processing 1
# process_data [INFO] Result: 2
# process_data [INFO] Processing 2
# process_data [INFO] Result: 4
# process_data [INFO] Processing 3
# process_data [INFO] Result: 6
```

---

## API Reference

### `get_logger(name=None)`

Get a logger instance that works across process/thread/machine boundaries.

**Parameters:**
- `name` (str, optional) - Logger name for code organization. If None, uses `"daglite.tasks"`. Typically use `__name__` for module-based naming.

**Returns:**
- `logging.LoggerAdapter` - Logger with automatic task context injection

**Features:**
- Automatically injects task context into all log records:
  - `daglite_task_name` - Name of the task being executed
  - `daglite_task_id` - Unique ID for the task instance
  - `daglite_task_key` - Node key in the DAG
- Routes logs through the reporter system when available
- Falls back to standard Python logging for Inline execution

**Example:**

```python
@task
def my_task(x: int):
    logger = get_logger(__name__)  # Use module name
    logger.info(f"Starting with {x}")

    try:
        result = expensive_computation(x)
        logger.info(f"Completed successfully: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise
```

### `CentralizedLoggingPlugin(level=logging.WARNING)`

Plugin that centralizes logs from all workers to the coordinator.

**Parameters:**
- `level` (int) - Minimum log level to handle on coordinator side. Default is `logging.WARNING`.

**How It Works:**

1. **Worker Side**: The `get_logger()` function automatically adds a handler that sends log records via the event reporter
2. **Coordinator Side**: The plugin receives log events and emits them through Python's logging system
3. **Task Context**: Task name, ID, and node key are automatically included in every log record

**Example:**

```python
import logging
from daglite import task, evaluate
from daglite.plugins import CentralizedLoggingPlugin, get_logger

# Configure logging format with task context
logging.basicConfig(
    format="[%(levelname)s] %(daglite_task_name)s: %(message)s",
    level=logging.DEBUG
)

@task(backend_name="multiprocessing")
def cpu_intensive_task(n: int) -> int:
    logger = get_logger(__name__)
    logger.debug(f"Starting computation for {n}")

    result = sum(i ** 2 for i in range(n))

    logger.info(f"Completed: {result}")
    return result

# Use plugin with INFO level
plugin = CentralizedLoggingPlugin(level=logging.INFO)

result = evaluate(
    cpu_intensive_task.product(n=[1000, 2000, 3000]),
    plugins=[plugin],
    use_async=True  # Runs in multiprocessing
)
```

---

## Advanced Usage

### Custom Log Formatting

Use standard Python logging configuration with task context fields:

```python
import logging

# Configure with task context in format string
logging.basicConfig(
    format="%(asctime)s | %(daglite_task_name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

@task
def process(item: dict):
    logger = get_logger(__name__)
    logger.info(f"Processing item {item['id']}")
    # Output: 2025-01-15 10:30:45 | process | INFO | Processing item 123
    return item
```

Available task context fields:
- `%(daglite_task_name)s` - Task function name
- `%(daglite_task_id)s` - Unique task instance ID (UUID)
- `%(daglite_task_key)s` - Node key in the execution graph

### File Logging

Write logs to a file:

```python
import logging

# Configure file handler
logging.basicConfig(
    filename="pipeline.log",
    format="%(asctime)s [%(levelname)s] %(daglite_task_name)s - %(message)s",
    level=logging.INFO
)

plugin = CentralizedLoggingPlugin(level=logging.INFO)

result = evaluate(my_dag, plugins=[plugin])
```

### Module-Based Logger Names

Organize logs by module:

```python
# myproject/tasks/etl.py
from daglite import task
from daglite.plugins import get_logger

@task
def extract(source: str):
    logger = get_logger(__name__)  # Uses "myproject.tasks.etl"
    logger.info(f"Extracting from {source}")
    return data

# myproject/tasks/ml.py
from daglite import task
from daglite.plugins import get_logger

@task
def train_model(data):
    logger = get_logger(__name__)  # Uses "myproject.tasks.ml"
    logger.info("Training model")
    return model
```

Then filter logs by module:

```python
# Only log from ML tasks
logging.getLogger("myproject.tasks.ml").setLevel(logging.DEBUG)
logging.getLogger("myproject.tasks.etl").setLevel(logging.WARNING)
```

### Structured Logging

Add custom fields to log records:

```python
@task
def process_batch(batch_id: int, items: list):
    logger = get_logger(__name__)

    # Add custom fields via 'extra'
    logger.info(
        f"Processing batch {batch_id}",
        extra={
            "batch_id": batch_id,
            "item_count": len(items),
            "batch_size": len(items)
        }
    )

    return results
```

### Different Log Levels

Control verbosity per execution:

```python
# Development - verbose logging
dev_plugin = CentralizedLoggingPlugin(level=logging.DEBUG)
evaluate(my_dag, plugins=[dev_plugin])

# Production - only warnings and errors
prod_plugin = CentralizedLoggingPlugin(level=logging.WARNING)
evaluate(my_dag, plugins=[prod_plugin])
```

### Logging Exceptions

Log exceptions with full tracebacks:

```python
@task
def risky_operation(data):
    logger = get_logger(__name__)

    try:
        result = process(data)
        logger.info("Success")
        return result
    except ValueError as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        raise
```

---

## Performance Considerations

### Log Level Filtering

Filtering happens on the **coordinator side** to minimize data transfer:

```python
# Workers send all log records
# Coordinator filters based on plugin level
plugin = CentralizedLoggingPlugin(level=logging.WARNING)

@task
def task_with_debug_logs():
    logger = get_logger()
    logger.debug("This is sent to coordinator")  # But filtered out
    logger.warning("This appears in output")     # This is shown
```

### Overhead

- **Inline execution**: Negligible overhead (standard Python logging)
- **Threading**: Minimal overhead from thread-safe queue
- **Multiprocessing**: Small serialization overhead per log record
- **Distributed**: Network latency for log events

For high-throughput scenarios, consider:
- Using higher log levels (WARNING instead of INFO)
- Logging less frequently
- Batching log messages

---

## Working with Async Execution

The plugin works seamlessly with async execution:

```python
@task(backend_name="threading")
def io_task(url: str):
    logger = get_logger(__name__)
    logger.info(f"Fetching {url}")
    data = requests.get(url).json()
    logger.info(f"Received {len(data)} items")
    return data

plugin = CentralizedLoggingPlugin(level=logging.INFO)

# Logs from all threads appear in order
result = evaluate(
    io_task.product(url=urls),
    plugins=[plugin],
    use_async=True
)
```

---

## Integration with External Loggers

### Using with Third-Party Logging

The plugin integrates with standard Python logging, so it works with:

- **Loguru**: Configure Loguru to intercept standard logging
- **structlog**: Use structlog processors with the standard logging
- **Cloud Logging**: Send to Google Cloud Logging, AWS CloudWatch, etc.

**Example with Cloud Logging:**

```python
import logging
import google.cloud.logging

# Set up Google Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()

# Daglite logs will be sent to Cloud Logging
plugin = CentralizedLoggingPlugin(level=logging.INFO)
evaluate(my_dag, plugins=[plugin])
```

### JSON Logging

Use `python-json-logger` for structured JSON logs:

```python
from pythonjsonlogger import jsonlogger
import logging

# Configure JSON formatter
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    "%(asctime)s %(name)s %(levelname)s %(daglite_task_name)s %(message)s"
)
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Logs will be JSON formatted
plugin = CentralizedLoggingPlugin(level=logging.INFO)
evaluate(my_dag, plugins=[plugin])
# Output: {"asctime": "2025-01-15 10:30:45", "name": "daglite.tasks",
#          "levelname": "INFO", "daglite_task_name": "process", "message": "Processing"}
```

---

## Best Practices

### 1. Use Module Names

```python
# Good - organized by module
logger = get_logger(__name__)

# Less useful - all logs from same logger
logger = get_logger()
```

### 2. Log at Appropriate Levels

```python
@task
def process(item):
    logger = get_logger(__name__)

    logger.debug(f"Raw item: {item}")           # Detailed debugging
    logger.info(f"Processing {item['id']}")     # Progress information
    logger.warning("Using fallback method")     # Potential issues
    logger.error("Failed to process")           # Errors
    logger.critical("System unstable")          # Critical failures
```

### 3. Include Context in Messages

```python
# Good - provides context
logger.info(f"Processed {len(items)} items in {duration:.2f}s")

# Less helpful
logger.info("Done")
```

### 4. Don't Log Sensitive Data

```python
# Bad - logs credentials
logger.info(f"Connecting to {username}:{password}@host")

# Good - masks sensitive info
logger.info(f"Connecting to {username}:***@host")
```

### 5. Configure Logging Once

```python
# At application startup
import logging

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(daglite_task_name)s - %(message)s",
    level=logging.INFO
)

# Then use throughout application
plugin = CentralizedLoggingPlugin(level=logging.INFO)
```

---

## Troubleshooting

### Logs Not Appearing

**Problem**: No logs appear when using async execution.

**Solution**: Make sure you're using the plugin:

```python
# Without plugin - logs may not appear in async mode
evaluate(my_dag, use_async=True)

# With plugin - logs work correctly
plugin = CentralizedLoggingPlugin(level=logging.INFO)
evaluate(my_dag, plugins=[plugin], use_async=True)
```

### Duplicate Logs

**Problem**: Each log message appears multiple times.

**Solution**: Check for multiple logging configurations or handlers:

```python
# Bad - adds handler every time
logging.basicConfig(...)  # Called multiple times

# Good - configure once at startup
if not logging.getLogger().handlers:
    logging.basicConfig(...)
```

### Missing Task Context

**Problem**: Task name doesn't appear in logs.

**Solution**: Use the format fields correctly:

```python
# Include task context fields in format
logging.basicConfig(
    format="%(daglite_task_name)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
```

---

## See Also

- [Creating Plugins](creating.md) - Build custom plugins
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html) - Standard library logging
- [Settings](../api-reference/settings.md) - Configure plugin tracing with `enable_plugin_tracing`
