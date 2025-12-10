"""Tests for daglite_serialization plugin."""

import time

import pytest

from daglite.serialization import SerializationRegistry


class TestNumpyPlugin:
    """Tests for numpy serialization plugin."""

    def test_numpy_array_hash_performance(self):
        """Test that numpy array hashing is fast for large arrays."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        # Create a large array (800MB)
        large_array = np.random.rand(10000, 10000)

        # Time the hash
        start = time.time()
        hash_value = hash_numpy_array(large_array)
        elapsed = time.time() - start

        # Should be very fast (<100ms as per requirements)
        assert elapsed < 0.1, f"Hash took {elapsed:.3f}s, should be < 0.1s"

        # Should be deterministic
        hash_value2 = hash_numpy_array(large_array)
        assert hash_value == hash_value2

    def test_numpy_small_array_hash(self):
        """Test that small numpy arrays are fully hashed."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3, 4, 5])
        arr3 = np.array([1, 2, 3, 4, 6])

        # Same arrays should have same hash
        assert hash_numpy_array(arr1) == hash_numpy_array(arr2)

        # Different arrays should have different hash
        assert hash_numpy_array(arr1) != hash_numpy_array(arr3)

    def test_numpy_middle_sampling(self):
        """Test that middle sampling catches changes."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        # Create large identical arrays
        arr1 = np.ones((10000, 10000))
        arr2 = np.ones((10000, 10000))

        # Modify middle of arr2
        mid = arr2.shape[0] // 2
        arr2[mid : mid + 10] = 2.0

        # Should detect the change
        assert hash_numpy_array(arr1) != hash_numpy_array(arr2)

    def test_register_numpy_handlers(self):
        """Test registering numpy handlers."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import register_handlers

        # Create fresh registry
        registry = SerializationRegistry()

        # Should raise before registration
        arr = np.array([1, 2, 3])
        with pytest.raises(TypeError, match="No hash strategy registered"):
            registry.hash_value(arr)

        # Register handlers
        register_handlers()

        # Should work now with default registry
        from daglite.serialization import default_registry

        hash_value = default_registry.hash_value(arr)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


class TestPandasPlugin:
    """Tests for pandas serialization plugin."""

    def test_dataframe_hash_performance(self):
        """Test that DataFrame hashing is fast for large DataFrames."""
        pytest.importorskip("pandas")
        import numpy as np
        import pandas as pd
        from daglite_serialization.pandas import hash_pandas_dataframe

        # Create a large DataFrame (1M rows)
        df = pd.DataFrame(
            {
                "a": np.random.rand(1_000_000),
                "b": np.random.rand(1_000_000),
                "c": np.random.randint(0, 100, 1_000_000),
            }
        )

        # Time the hash
        start = time.time()
        hash_value = hash_pandas_dataframe(df)
        elapsed = time.time() - start

        # Should be reasonably fast (<1s)
        assert elapsed < 1.0, f"Hash took {elapsed:.3f}s, should be < 1.0s"

        # Should be deterministic
        hash_value2 = hash_pandas_dataframe(df)
        assert hash_value == hash_value2

    def test_dataframe_small_hash(self):
        """Test that small DataFrames are fully hashed."""
        pytest.importorskip("pandas")
        import pandas as pd
        from daglite_serialization.pandas import hash_pandas_dataframe

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df3 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})

        # Same DataFrames should have same hash
        assert hash_pandas_dataframe(df1) == hash_pandas_dataframe(df2)

        # Different DataFrames should have different hash
        assert hash_pandas_dataframe(df1) != hash_pandas_dataframe(df3)

    def test_series_hash(self):
        """Test pandas Series hashing."""
        pytest.importorskip("pandas")
        import pandas as pd
        from daglite_serialization.pandas import hash_pandas_series

        series1 = pd.Series([1, 2, 3, 4, 5], name="data")
        series2 = pd.Series([1, 2, 3, 4, 5], name="data")
        series3 = pd.Series([1, 2, 3, 4, 6], name="data")

        # Same series should have same hash
        assert hash_pandas_series(series1) == hash_pandas_series(series2)

        # Different series should have different hash
        assert hash_pandas_series(series1) != hash_pandas_series(series3)

    def test_register_pandas_handlers(self):
        """Test registering pandas handlers."""
        pytest.importorskip("pandas")
        import pandas as pd
        from daglite_serialization.pandas import register_handlers

        # Register handlers
        register_handlers()

        # Should work with default registry
        from daglite.serialization import default_registry

        df = pd.DataFrame({"a": [1, 2, 3]})
        hash_value = default_registry.hash_value(df)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

        series = pd.Series([1, 2, 3])
        hash_value = default_registry.hash_value(series)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


class TestPillowPlugin:
    """Tests for Pillow (PIL) serialization plugin."""

    def test_image_hash(self):
        """Test PIL Image hashing."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.pillow import hash_pil_image
        from PIL import Image

        # Create test images
        img1 = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))
        img2 = img1.copy()
        img3 = Image.fromarray(np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8))

        # Same image should have same hash
        assert hash_pil_image(img1) == hash_pil_image(img2)

        # Different images should (likely) have different hashes
        # Note: There's a tiny chance of collision with downsampling
        assert hash_pil_image(img1) != hash_pil_image(img3)

    def test_register_pillow_handlers(self):
        """Test registering pillow handlers."""
        pytest.importorskip("PIL")
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.pillow import register_handlers
        from PIL import Image

        # Register handlers
        register_handlers()

        # Should work with default registry
        from daglite.serialization import default_registry

        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        hash_value = default_registry.hash_value(img)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64


class TestRegisterAll:
    """Tests for register_all() function."""

    def test_register_all(self):
        """Test register_all() registers available plugins."""
        pytest.importorskip("numpy")

        from daglite_serialization import register_all

        # Register all
        register_all()

        # Should work with numpy at minimum
        import numpy as np

        from daglite.serialization import default_registry

        arr = np.array([1, 2, 3])
        hash_value = default_registry.hash_value(arr)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64

    def test_register_all_missing_deps(self):
        """Test register_all() gracefully handles missing dependencies."""
        from daglite_serialization import register_all

        # Should not raise even if some dependencies are missing
        register_all()


class TestNumpyPluginExtended:
    """Extended tests for numpy plugin edge cases."""

    def test_numpy_2d_array_sampling(self):
        """Test 2D array sampling logic."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        # Create large 2D array (trigger 2D sampling path)
        arr = np.ones((5000, 5000))  # > 10000 elements, 2D shape

        # Should hash successfully
        hash1 = hash_numpy_array(arr)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Modify middle rows - should detect change
        arr[2500:2510, :] = 2.0
        hash2 = hash_numpy_array(arr)
        assert hash1 != hash2

    def test_numpy_3d_array_sampling(self):
        """Test multi-dimensional (3D+) array sampling logic."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        # Create 3D array (trigger flattening path)
        arr1 = np.ones((100, 100, 100))  # > 10000 elements, 3D shape
        arr2 = np.ones((100, 100, 100))

        # Should hash successfully
        hash1 = hash_numpy_array(arr1)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Modify beginning (which will be sampled) - should detect change
        arr2[0:10, 0:10, 0:10] = 2.0
        hash2 = hash_numpy_array(arr2)
        assert hash1 != hash2

    def test_numpy_1d_large_array_sampling(self):
        """Test 1D array sampling for large arrays."""
        pytest.importorskip("numpy")
        import numpy as np
        from daglite_serialization.numpy import hash_numpy_array

        # Create large 1D array (trigger 1D sampling path)
        arr = np.ones(100000)  # > 10000 elements, 1D shape

        # Should hash successfully
        hash1 = hash_numpy_array(arr)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Modify middle - should detect change
        arr[50000:51000] = 2.0
        hash2 = hash_numpy_array(arr)
        assert hash1 != hash2


class TestPandasPluginExtended:
    """Extended tests for pandas plugin edge cases."""

    def test_large_series_sampling(self):
        """Test Series sampling for large series."""
        pytest.importorskip("pandas")
        import pandas as pd
        from daglite_serialization.pandas import hash_pandas_series

        # Create large Series (trigger sampling path)
        series = pd.Series(range(10000), name="large_series")

        # Should hash successfully
        hash1 = hash_pandas_series(series)
        assert isinstance(hash1, str)
        assert len(hash1) == 64

        # Create different large series
        series2 = pd.Series(range(1, 10001), name="large_series")
        hash2 = hash_pandas_series(series2)
        assert hash1 != hash2
