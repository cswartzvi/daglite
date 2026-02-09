"""Unit tests for _save_outputs in graph/nodes.py."""

import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from daglite.datasets.store import DatasetStore
from daglite.graph.base import OutputConfig
from daglite.graph.nodes import _save_outputs


class TestSaveOutputsNoOp:
    """Tests for _save_outputs early returns."""

    def test_empty_output_config(self):
        """No-op when output_config is empty."""
        _save_outputs(
            result="value",
            resolved_inputs={},
            output_config=(),
            output_deps=[],
        )


class TestSaveOutputsKeyFormatting:
    """Tests for key formatting in _save_outputs."""

    def test_simple_key_no_placeholders(self):
        """Literal key with no placeholders is used as-is."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="output.pkl", store=store)

            with patch("daglite.graph.nodes.get_dataset_reporter") as mock_reporter:
                mock_reporter.return_value = MagicMock(is_direct=True)
                _save_outputs(
                    result={"data": 1},
                    resolved_inputs={},
                    output_config=(config,),
                    output_deps=[{}],
                )
                mock_reporter.return_value.save.assert_called_once()
                call_args = mock_reporter.return_value.save.call_args
                assert call_args[0][0] == "output.pkl"

    def test_key_formatted_from_inputs(self):
        """Key placeholders are resolved from resolved_inputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="output_{data_id}.pkl", store=store)

            with patch("daglite.graph.nodes.get_dataset_reporter") as mock_reporter:
                mock_reporter.return_value = MagicMock(is_direct=True)
                _save_outputs(
                    result="value",
                    resolved_inputs={"data_id": "abc123"},
                    output_config=(config,),
                    output_deps=[{}],
                )
                call_args = mock_reporter.return_value.save.call_args
                assert call_args[0][0] == "output_abc123.pkl"

    def test_key_formatted_from_extras(self):
        """Key placeholders can come from output_deps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="output_{version}.pkl", store=store)

            with patch("daglite.graph.nodes.get_dataset_reporter") as mock_reporter:
                mock_reporter.return_value = MagicMock(is_direct=True)
                _save_outputs(
                    result="value",
                    resolved_inputs={},
                    output_config=(config,),
                    output_deps=[{"version": "v2"}],
                )
                call_args = mock_reporter.return_value.save.call_args
                assert call_args[0][0] == "output_v2.pkl"

    def test_key_formatted_from_key_extras(self):
        """Key placeholders can come from key_extras."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="output_{iteration_index}.pkl", store=store)

            with patch("daglite.graph.nodes.get_dataset_reporter") as mock_reporter:
                mock_reporter.return_value = MagicMock(is_direct=True)
                _save_outputs(
                    result="value",
                    resolved_inputs={},
                    output_config=(config,),
                    output_deps=[{}],
                    key_extras={"iteration_index": 3},
                )
                call_args = mock_reporter.return_value.save.call_args
                assert call_args[0][0] == "output_3.pkl"

    def test_missing_placeholder_raises_value_error(self):
        """Missing placeholder raises ValueError with helpful message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="output_{missing}.pkl", store=store)

            with patch("daglite.graph.nodes.get_dataset_reporter") as mock_reporter:
                mock_reporter.return_value = MagicMock(is_direct=True)
                with pytest.raises(ValueError, match="not available"):
                    _save_outputs(
                        result="value",
                        resolved_inputs={"data_id": "abc"},
                        output_config=(config,),
                        output_deps=[{}],
                    )


class TestSaveOutputsRouting:
    """Tests for local vs. remote routing in _save_outputs."""

    def test_local_store_routes_through_reporter(self):
        """Local store saves are routed via the dataset reporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="out.pkl", store=store, format="pickle")

            mock_reporter = MagicMock()

            with patch("daglite.graph.nodes.get_dataset_reporter", return_value=mock_reporter):
                _save_outputs(
                    result={"data": 1},
                    resolved_inputs={},
                    output_config=(config,),
                    output_deps=[{}],
                )

            mock_reporter.save.assert_called_once_with(
                "out.pkl", {"data": 1}, store, format="pickle", options={}
            )

    def test_remote_store_saves_directly(self):
        """Remote (non-local) store saves directly, bypassing reporter."""
        mock_store = MagicMock()
        mock_store.is_local = False
        config = OutputConfig(key="out.pkl", store=mock_store, format="pickle")

        with patch("daglite.graph.nodes.get_dataset_reporter") as mock_get_reporter:
            mock_get_reporter.return_value = MagicMock()
            _save_outputs(
                result={"data": 1},
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )

        mock_store.save.assert_called_once_with("out.pkl", {"data": 1}, format="pickle", options={})

    def test_no_reporter_saves_directly(self):
        """When no dataset reporter is set, saves directly to store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DatasetStore(tmpdir)
            config = OutputConfig(key="out.txt", store=store, format="text")

            with patch("daglite.graph.nodes.get_dataset_reporter", return_value=None):
                _save_outputs(
                    result="hello",
                    resolved_inputs={},
                    output_config=(config,),
                    output_deps=[{}],
                )

            assert store.exists("out.txt")
            assert store.load("out.txt", return_type=str) == "hello"


class TestSaveOutputsStoreResolution:
    """Tests for store resolution in _save_outputs."""

    def test_explicit_store_used(self):
        """Store from OutputConfig is used when present."""
        mock_store = MagicMock()
        mock_store.is_local = False
        config = OutputConfig(key="out.pkl", store=mock_store)

        with patch("daglite.graph.nodes.get_dataset_reporter", return_value=None):
            _save_outputs(
                result="value",
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )

        mock_store.save.assert_called_once()

    def test_settings_fallback_string(self):
        """When config.store is None, falls back to settings.datastore_store (string)."""
        config = OutputConfig(key="out.pkl", store=None)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("daglite.graph.nodes.get_dataset_reporter", return_value=None),
            patch("daglite.settings.get_global_settings") as mock_settings,
        ):
            mock_settings.return_value.datastore_store = tmpdir
            _save_outputs(
                result={"a": 1},
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )
            # Should have created a DatasetStore from the path and saved

    def test_settings_fallback_store_instance(self):
        """When config.store is None, falls back to settings.datastore_store (DatasetStore)."""
        config = OutputConfig(key="out.pkl", store=None)
        mock_store = MagicMock()
        mock_store.is_local = False

        with (
            patch("daglite.graph.nodes.get_dataset_reporter", return_value=None),
            patch("daglite.settings.get_global_settings") as mock_settings,
        ):
            mock_settings.return_value.datastore_store = mock_store
            _save_outputs(
                result="value",
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )

        mock_store.save.assert_called_once()


class TestSaveOutputsFormatAndOptions:
    """Tests for format and options pass-through."""

    def test_format_passed_to_reporter(self):
        """Format from OutputConfig is passed through to reporter.save()."""
        mock_store = MagicMock()
        mock_store.is_local = False
        config = OutputConfig(key="data.txt", store=mock_store, format="text")

        with patch("daglite.graph.nodes.get_dataset_reporter", return_value=None):
            _save_outputs(
                result="hello",
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )

        _, kwargs = mock_store.save.call_args
        assert kwargs["format"] == "text"

    def test_options_passed_to_reporter(self):
        """Options from OutputConfig are passed through."""
        mock_store = MagicMock()
        mock_store.is_local = False
        config = OutputConfig(key="data.pkl", store=mock_store, options={"protocol": 5})

        with patch("daglite.graph.nodes.get_dataset_reporter", return_value=None):
            _save_outputs(
                result="value",
                resolved_inputs={},
                output_config=(config,),
                output_deps=[{}],
            )

        _, kwargs = mock_store.save.call_args
        assert kwargs["options"] == {"protocol": 5}

    def test_multiple_configs(self):
        """_save_outputs handles multiple output configs."""
        mock_store = MagicMock()
        mock_store.is_local = False
        config1 = OutputConfig(key="out1.pkl", store=mock_store)
        config2 = OutputConfig(key="out2.pkl", store=mock_store)

        with patch("daglite.graph.nodes.get_dataset_reporter", return_value=None):
            _save_outputs(
                result="value",
                resolved_inputs={},
                output_config=(config1, config2),
                output_deps=[{}, {}],
            )

        assert mock_store.save.call_count == 2
