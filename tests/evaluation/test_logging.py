"""Integration tests for centralized logging.

Tests verify that logs from worker tasks are properly routed to the coordinator
across different execution backends using the CentralizedLoggingPlugin.
"""

import logging

import pytest

from daglite import evaluate
from daglite import task
from daglite.plugins import CentralizedLoggingPlugin
from daglite.plugins import get_logger


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests to prevent handler accumulation."""
    # Remove all handlers from all loggers before test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)

    logging.root.handlers.clear()

    yield

    # Clean up after test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)

    logging.root.handlers.clear()


# Module-level tasks required for multiprocessing backend (pickle compatibility)


@task
def worker_task(x: int) -> int:
    """Task that logs during execution."""
    logger = get_logger(__name__)
    logger.info(f"Worker processing {x}")
    return x * 2


@task
def metadata_task(x: int) -> int:
    """Task for testing metadata preservation."""
    logger = get_logger()
    logger.info(f"Metadata test {x}")
    return x


@task
def exception_task(x: int) -> int:
    """Task that logs an exception."""
    logger = get_logger(__name__)
    try:
        raise ValueError(f"Worker error for {x}")
    except ValueError:
        logger.exception("Exception in worker")
    return x


@task
def map_worker(x: int) -> int:
    """Task for testing mapped execution."""
    logger = get_logger()
    logger.info(f"Map processing {x}")
    return x * 2


@task
def multi_level_task(x: int) -> int:
    """Task that logs at different levels."""
    logger = get_logger(__name__)
    logger.debug(f"Debug: {x}")
    logger.info(f"Info: {x}")
    logger.warning(f"Warning: {x}")
    logger.error(f"Error: {x}")
    return x


@task
def multi_logger_task(x: int) -> int:
    """Task that uses multiple loggers."""
    logger1 = get_logger("custom.logger.one")
    logger2 = get_logger("custom.logger.two")
    logger1.info(f"Logger one: {x}")
    logger2.info(f"Logger two: {x}")
    return x


@task
def upstream_task(x: int) -> int:
    """Upstream task in a pipeline."""
    logger = get_logger()
    logger.info(f"Upstream processing {x}")
    return x * 2


@task
def downstream_task(x: int) -> int:
    """Downstream task in a pipeline."""
    logger = get_logger()
    logger.info(f"Downstream processing {x}")
    return x + 10


@task
def double(x: int) -> int:
    """Task that doubles a value."""
    logger = get_logger()
    logger.info(f"Doubling {x}")
    return x * 2


class TestCentralizedLoggingIntegration:
    """Integration tests for centralized logging with different backends."""

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_logging_with_reporter_backends(self, backend_name, caplog):
        """Test that logging works correctly with backends that use reporters."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        # Configure task with backend
        task_with_backend = worker_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 84

        # Verify log was captured and routed through coordinator
        assert "Worker processing 42" in caplog.text
        assert len(caplog.records) > 0

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_task_metadata_preserved(self, backend_name, caplog):
        """Test that daglite_* metadata fields are preserved from workers."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = metadata_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=99), plugins=[plugin])
            assert result == 99

        # Verify task metadata fields exist in log records
        assert len(caplog.records) > 0
        record = caplog.records[0]
        assert hasattr(record, "daglite_task_name")
        assert hasattr(record, "daglite_task_id")
        assert hasattr(record, "daglite_node_key")
        assert record.daglite_task_name == "metadata_task"

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_exception_logging(self, backend_name, caplog):
        """Test that exception tracebacks are transmitted from workers."""
        plugin = CentralizedLoggingPlugin(level=logging.ERROR)

        task_with_backend = exception_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.ERROR):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Verify exception details are in logs
        assert "Exception in worker" in caplog.text
        assert "ValueError" in caplog.text
        assert "Worker error for 42" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_mapped_task_logging(self, backend_name, caplog):
        """Test logging from mapped tasks with node keys."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = map_worker.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            future = task_with_backend.product(x=[1, 2, 3])
            results = evaluate(future, plugins=[plugin])
            assert results == [2, 4, 6]

        # Verify all mapped tasks logged
        log_text = caplog.text
        assert "Map processing 1" in log_text
        assert "Map processing 2" in log_text
        assert "Map processing 3" in log_text

        # Verify node keys are present (map_worker[0], map_worker[1], etc.)
        records = caplog.records
        assert len(records) >= 3
        for record in records:
            assert hasattr(record, "daglite_node_key")
            # Node keys should be like "map_worker[0]", "map_worker[1]", etc.
            assert record.daglite_node_key.startswith("map_worker")

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_log_level_filtering(self, backend_name, caplog):
        """Test that log level filtering works correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.WARNING)

        task_with_backend = multi_level_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.WARNING):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Only WARNING and ERROR should be present
        assert "Debug: 42" not in caplog.text
        assert "Info: 42" not in caplog.text
        assert "Warning: 42" in caplog.text
        assert "Error: 42" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_multiple_loggers(self, backend_name, caplog):
        """Test that multiple logger names work correctly."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        task_with_backend = multi_logger_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            result = evaluate(task_with_backend(x=42), plugins=[plugin])
            assert result == 42

        # Both logger messages should appear
        assert "Logger one: 42" in caplog.text
        assert "Logger two: 42" in caplog.text


class TestCentralizedLoggingWithPipelines:
    """Test centralized logging with complex pipeline configurations."""

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_pipeline_with_dependencies(self, backend_name, caplog):
        """Test logging in pipeline with task dependencies."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        upstream_with_backend = upstream_task.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            future = upstream_with_backend(x=5).then(downstream_task)
            result = evaluate(future, plugins=[plugin])
            assert result == 20  # (5 * 2) + 10

        # Both tasks should have logged
        assert "Upstream processing 5" in caplog.text
        assert "Downstream processing 10" in caplog.text

    @pytest.mark.parametrize("backend_name", ["threads", "processes"])
    def test_product_mapping(self, backend_name, caplog):
        """Test logging in product() pattern with mapped tasks."""
        plugin = CentralizedLoggingPlugin(level=logging.INFO)

        double_with_backend = double.with_options(backend_name=backend_name)

        with caplog.at_level(logging.INFO):
            future = double_with_backend.product(x=[1, 2, 3, 4, 5])
            results = evaluate(future, plugins=[plugin])
            assert results == [2, 4, 6, 8, 10]

        # All double operations should be logged
        for i in range(1, 6):
            assert f"Doubling {i}" in caplog.text
