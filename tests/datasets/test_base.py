"""Unit tests for AbstractDataset registry, lookup, format inference, and plugin discovery."""

import pytest

from daglite.datasets.base import AbstractDataset


class _Marker:
    """Unique type that is never registered by built-in datasets."""

    pass


class _MarkerChild(_Marker):
    """Subclass of _Marker for testing subclass lookups."""

    pass


class TestInitSubclassRegistration:
    """Tests for the auto-registration via __init_subclass__."""

    def test_register_with_class_params(self):
        """Subclass with class params is registered."""

        class MarkerDS(
            AbstractDataset,
            format="marker_fmt",
            types=_Marker,
            extensions="mrk",
        ):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return _Marker()

        try:
            assert (_Marker, "marker_fmt") in AbstractDataset._registry
            assert AbstractDataset._registry[(_Marker, "marker_fmt")] is MarkerDS
        finally:
            AbstractDataset._registry.pop((_Marker, "marker_fmt"), None)
            AbstractDataset._extension_hints.pop("mrk", None)

    def test_register_with_class_variables(self):
        """Subclass using class-variable style is registered."""

        class MarkerDS2(AbstractDataset):
            format = "marker_cv"
            types = (_Marker,)
            extensions = ("mrkv",)

            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return _Marker()

        try:
            assert (_Marker, "marker_cv") in AbstractDataset._registry
        finally:
            AbstractDataset._registry.pop((_Marker, "marker_cv"), None)
            AbstractDataset._extension_hints.pop("mrkv", None)

    def test_abstract_intermediate_skips_registration(self):
        """Subclass without format acts as abstract intermediate (not registered)."""
        registry_before = dict(AbstractDataset._registry)

        class Intermediate(AbstractDataset):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return None

        assert AbstractDataset._registry == registry_before

    def test_multiple_types_registered(self):
        """A single dataset registering multiple types creates one entry per type."""

        class MultiDS(
            AbstractDataset,
            format="multi_fmt",
            types=(int, float),
            extensions="mlt",
        ):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return 0

        try:
            assert (int, "multi_fmt") in AbstractDataset._registry
            assert (float, "multi_fmt") in AbstractDataset._registry
        finally:
            AbstractDataset._registry.pop((int, "multi_fmt"), None)
            AbstractDataset._registry.pop((float, "multi_fmt"), None)
            AbstractDataset._extension_hints.pop("mlt", None)

    def test_extension_hints_populated(self):
        """Extension hints are populated when extensions are provided."""

        class ExtDS(
            AbstractDataset,
            format="ext_fmt",
            types=_Marker,
            extensions=("ext1", "ext2"),
        ):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return _Marker()

        try:
            assert (_Marker, "ext_fmt") in AbstractDataset._extension_hints.get("ext1", [])
            assert (_Marker, "ext_fmt") in AbstractDataset._extension_hints.get("ext2", [])
        finally:
            AbstractDataset._registry.pop((_Marker, "ext_fmt"), None)
            AbstractDataset._extension_hints.pop("ext1", None)
            AbstractDataset._extension_hints.pop("ext2", None)


class TestGet:
    """Tests for AbstractDataset.get() lookup."""

    def test_exact_match(self):
        """Exact (type, format) match returns correct dataset."""
        ds = AbstractDataset.get(str, "text")
        assert ds.serialize("hello") == b"hello"

    def test_subclass_match(self):
        """Subclass of a registered type uses parent's dataset."""

        class MyStr(str):
            pass

        ds = AbstractDataset.get(MyStr, "text")
        assert ds.deserialize(b"hi") == "hi"

    def test_object_fallback_for_pickle(self):
        """Pickle format with object registration covers arbitrary types."""
        ds = AbstractDataset.get(object, "pickle")
        data = ds.serialize({"a": 1})
        assert ds.deserialize(data) == {"a": 1}

    def test_not_found_with_available_formats(self):
        """Raises ValueError listing available formats when type is known."""
        with pytest.raises(ValueError, match="Available formats"):
            AbstractDataset.get(str, "nonexistent_format_xyz")

    def test_not_found_no_formats(self):
        """Raises ValueError with pip install hint when type is truly unknown."""
        # _Marker inherits from object which has pickle; use a type that
        # bypasses the object fallback by temporarily removing it.
        original = AbstractDataset._registry.pop((object, "pickle"), None)
        try:
            with pytest.raises(ValueError, match="pip install"):
                AbstractDataset.get(_Marker, "nonexistent_format_xyz")
        finally:
            if original is not None:
                AbstractDataset._registry[(object, "pickle")] = original

    def test_subclass_match_with_issubclass_type_error(self):
        """get() handles non-class types gracefully without crashing."""
        # Registering with a non-class type that makes issubclass raise TypeError
        # This exercises the `except TypeError: continue` branch
        original = dict(AbstractDataset._registry)
        # Add a key with a non-class to trigger TypeError in issubclass
        AbstractDataset._registry[("not_a_class", "fake_fmt")] = type(  # type: ignore[assignment]
            "FakeDS",
            (AbstractDataset,),
            {
                "format": "fake_fmt",
                "supported_types": (),
                "file_extensions": (),
                "serialize": lambda self, v, **o: b"",
                "deserialize": lambda self, d, **o: None,
            },
        )
        try:
            with pytest.raises(ValueError):
                AbstractDataset.get(_Marker, "fake_fmt")
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original)


class TestInferFormat:
    """Tests for AbstractDataset.infer_format() format inference."""

    def test_infer_from_extension_exact(self):
        """Extension hint yields correct format for exact type match."""
        assert AbstractDataset.infer_format(str, "txt") == "text"

    def test_infer_from_extension_subclass(self):
        """Extension hint works for subclasses."""

        class MyDict(dict):
            pass

        fmt = AbstractDataset.infer_format(MyDict, "pkl")
        assert fmt == "pickle"

    def test_infer_from_type_no_extension(self):
        """Without extension, first registered format for the type wins."""
        fmt = AbstractDataset.infer_format(str)
        assert fmt == "text"

    def test_infer_subclass_no_extension(self):
        """Subclass without extension falls back to parent's registration."""

        class MyBytes(bytes):
            pass

        fmt = AbstractDataset.infer_format(MyBytes)
        # bytes is registered with "raw"
        assert fmt == "raw"

    def test_infer_unknown_type_raises(self):
        """Raises ValueError for a completely unknown type."""
        # Remove the object→pickle fallback so _Marker has no format
        original = AbstractDataset._registry.pop((object, "pickle"), None)
        try:
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_Marker)
        finally:
            if original is not None:
                AbstractDataset._registry[(object, "pickle")] = original

    def test_infer_with_issubclass_type_error(self):
        """infer_format handles non-class types in extension hints gracefully."""
        original_hints = dict(AbstractDataset._extension_hints)
        original_registry = dict(AbstractDataset._registry)
        AbstractDataset._extension_hints["badext"] = [("not_a_class", "badfmt")]  # type: ignore[list-item]
        # Remove object→pickle so _Marker truly has no format
        AbstractDataset._registry.pop((object, "pickle"), None)
        try:
            # Should not crash, just skip the bad entry and raise
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_Marker, "badext")
        finally:
            AbstractDataset._extension_hints.clear()
            AbstractDataset._extension_hints.update(original_hints)
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)


class TestGetFormatsForType:
    """Tests for get_formats_for_type()."""

    def test_exact_type(self):
        """Returns all formats for an exact type."""
        formats = AbstractDataset.get_formats_for_type(str)
        assert "text" in formats

    def test_subclass_inherits_parent_formats(self):
        """Subclass includes parent's formats via issubclass."""

        class MyDict(dict):
            pass

        formats = AbstractDataset.get_formats_for_type(MyDict)
        assert "pickle" in formats

    def test_unknown_type_inherits_object_formats(self):
        """Unknown type inherits object's formats via subclass matching."""
        formats = AbstractDataset.get_formats_for_type(_Marker)
        # _Marker is a subclass of object, so it should find pickle via object
        assert "pickle" in formats

    def test_truly_unknown_type_empty(self):
        """Returns empty set when object fallback is removed."""
        original = dict(AbstractDataset._registry)
        AbstractDataset._registry.pop((object, "pickle"), None)
        try:
            formats = AbstractDataset.get_formats_for_type(_Marker)
            assert len(formats) == 0
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original)

    def test_non_class_type_in_registry(self):
        """Handles TypeError from issubclass gracefully."""
        original = dict(AbstractDataset._registry)
        AbstractDataset._registry[("not_a_class", "bad_fmt")] = type(  # type: ignore[assignment]
            "FakeDS2",
            (AbstractDataset,),
            {
                "format": "bad_fmt",
                "supported_types": (),
                "file_extensions": (),
                "serialize": lambda self, v, **o: b"",
                "deserialize": lambda self, d, **o: None,
            },
        )
        try:
            # Should not raise, just skip the bad entry
            formats = AbstractDataset.get_formats_for_type(_Marker)
            assert isinstance(formats, set)
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original)


class TestGetExtension:
    """Tests for get_extension()."""

    def test_known_type_format(self):
        """Returns first extension for a known (type, format)."""
        ext = AbstractDataset.get_extension(str, "text")
        assert ext == "txt"

    def test_unknown_type_format(self):
        """Returns None for an unknown (type, format)."""
        ext = AbstractDataset.get_extension(_Marker, "nonexistent_xyz")
        assert ext is None

    def test_no_extensions_defined(self):
        """Returns None when dataset has no file extensions."""

        class NoExtDS(
            AbstractDataset,
            format="noext_fmt",
            types=_Marker,
        ):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):  # pragma: no cover
                return _Marker()

        try:
            ext = AbstractDataset.get_extension(_Marker, "noext_fmt")
            assert ext is None
        finally:
            AbstractDataset._registry.pop((_Marker, "noext_fmt"), None)


class TestLoadPlugins:
    """Tests for load_plugins() and _discover_for_type()."""

    def test_load_plugins_sets_auto_discover(self):
        """load_plugins(auto_discover=...) updates the class flag."""
        original = AbstractDataset._auto_discover
        try:
            AbstractDataset.load_plugins(auto_discover=False)
            assert AbstractDataset._auto_discover is False
            AbstractDataset.load_plugins(auto_discover=True)
            assert AbstractDataset._auto_discover is True
        finally:
            AbstractDataset._auto_discover = original

    def test_load_plugins_no_names_loads_all(self):
        """Calling load_plugins() without names doesn't crash."""
        AbstractDataset.load_plugins()

    def test_load_plugins_specific_name(self):
        """Calling load_plugins('nonexistent') doesn't crash."""
        AbstractDataset.load_plugins("nonexistent_plugin_xyz123")

    def test_discover_for_type_no_op_when_already_discovered(self):
        """Once a module is discovered, _discover_for_type is a no-op."""
        original = set(AbstractDataset._discovered)
        try:
            AbstractDataset._discovered.add("builtins")
            # Should be a no-op on second call
            AbstractDataset._discover_for_type(int)
        finally:
            AbstractDataset._discovered = original

    def test_auto_discover_on_get(self):
        """Auto-discover is triggered on get() for unknown types."""
        # This just exercises the auto-discover code path without expecting
        # it to find anything for _Marker
        with pytest.raises(ValueError):
            AbstractDataset.get(_Marker, "nonexistent_123")

    def test_auto_discover_on_infer_format(self):
        """Auto-discover is triggered on infer_format() for unknown types."""
        original_discovered = set(AbstractDataset._discovered)
        original_registry = dict(AbstractDataset._registry)
        # Remove object→pickle so _Marker truly has no format
        AbstractDataset._registry.pop((object, "pickle"), None)
        try:
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_Marker)
        finally:
            AbstractDataset._discovered = original_discovered
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)

    def test_infer_format_after_discovery_finds_type(self):
        """After _discover_for_type adds a registration, infer_format returns it."""
        original_registry = dict(AbstractDataset._registry)
        original_discovered = set(AbstractDataset._discovered)
        try:
            # Pre-register _Marker during the discovery path
            AbstractDataset._registry[(_Marker, "discovered_fmt")] = type(
                "DiscoveredDS",
                (AbstractDataset,),
                {
                    "format": "discovered_fmt",
                    "supported_types": (_Marker,),
                    "file_extensions": (),
                    "serialize": lambda self, v, **o: b"",
                    "deserialize": lambda self, d, **o: _Marker(),
                },
            )
            # Clear discovered so infer_format falls through to the "after discovery" path
            AbstractDataset._discovered.discard(_Marker.__module__.split(".")[0])
            fmt = AbstractDataset.infer_format(_Marker)
            assert fmt == "discovered_fmt"
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            AbstractDataset._discovered = original_discovered

    def test_get_after_discovery_via_subclass(self):
        """get() finds a subclass match after auto-discovery."""
        original_registry = dict(AbstractDataset._registry)
        original_discovered = set(AbstractDataset._discovered)

        class DiscoveredDS(AbstractDataset, format="disc_sub", types=_Marker, extensions="dsc"):
            def serialize(self, value, **opts):  # pragma: no cover
                return b""

            def deserialize(self, data, **opts):
                return _MarkerChild()

        try:
            # Clear discovered so get() triggers auto-discovery path
            AbstractDataset._discovered.discard(_MarkerChild.__module__.split(".")[0])
            ds = AbstractDataset.get(_MarkerChild, "disc_sub")
            result = ds.deserialize(b"")
            assert isinstance(result, _MarkerChild)
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            AbstractDataset._extension_hints.pop("dsc", None)
            AbstractDataset._discovered = original_discovered

    def test_infer_format_subclass_no_extension_fallback(self):
        """infer_format falls back to subclass check in registry when no extension."""

        class ChildDict(dict):
            pass

        # dict has pickle registered; ChildDict should find it
        fmt = AbstractDataset.infer_format(ChildDict)
        assert fmt == "pickle"

    def test_infer_format_extension_hint_subclass_no_match(self):
        """infer_format tries extension hint subclass check, finds no match, falls through."""
        original_hints = dict(AbstractDataset._extension_hints)

        class _Unrelated:
            pass

        # Add a hint for an unrelated type — issubclass(str, _Unrelated) is False
        AbstractDataset._extension_hints.setdefault("txt", []).append((_Unrelated, "nope"))

        try:
            # str has an exact hint for "txt" → "text" which matches first,
            # but let's use a type that won't match the exact check
            # We need a type where the extension hints only have the subclass path
            # Use a fresh extension
            AbstractDataset._extension_hints["xtest"] = [(_Unrelated, "nope")]

            # query for int with extension "xtest" — issubclass(int, _Unrelated) is False
            # Falls through to registry check, int→pickle via object subclass
            fmt = AbstractDataset.infer_format(int, "xtest")
            assert fmt == "pickle"
        finally:
            AbstractDataset._extension_hints.clear()
            AbstractDataset._extension_hints.update(original_hints)


class TestGetAutoDiscoverSlowPath:
    """Tests for get() slow path that runs after _discover_for_type."""

    def test_exact_key_match_after_auto_discover(self):
        """get() finds exact key match after auto-discover registers a dataset."""
        original_registry = dict(AbstractDataset._registry)
        original_discovered = AbstractDataset._discovered.copy()

        class _DiscType:
            pass

        class _DiscDataset(AbstractDataset, types=_DiscType, format="disc"):
            def serialize(self, value, **options):
                return b"disc"

            def deserialize(self, data, **options):
                return _DiscType()

        # Remove the registration so the fast path won't find it
        key = (_DiscType, "disc")
        saved_cls = AbstractDataset._registry.pop(key)

        # Mock _discover_for_type to re-register on discovery
        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        try:
            AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
            AbstractDataset._discovered.discard(_DiscType.__module__.split(".")[0])

            ds = AbstractDataset.get(_DiscType, "disc")
            assert ds.serialize(None) == b"disc"
        finally:
            AbstractDataset._discover_for_type = orig_discover
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            AbstractDataset._discovered = original_discovered

    def test_subclass_match_after_auto_discover(self):
        """get() finds subclass match after auto-discover registers a parent dataset."""

        class _DiscParent:
            pass

        class _DiscChild(_DiscParent):
            pass

        original_registry = dict(AbstractDataset._registry)
        original_discovered = AbstractDataset._discovered.copy()

        class _DiscParentDataset(AbstractDataset, types=_DiscParent, format="discfmt"):
            def serialize(self, value, **options):
                return b"parent"

            def deserialize(self, data, **options):
                return _DiscParent()

        # Remove the registration so subclass check pre-discover won't find it
        key = (_DiscParent, "discfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        try:
            AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
            AbstractDataset._discovered.discard(_DiscChild.__module__.split(".")[0])

            ds = AbstractDataset.get(_DiscChild, "discfmt")
            assert ds.serialize(None) == b"parent"
        finally:
            AbstractDataset._discover_for_type = orig_discover
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            AbstractDataset._discovered = original_discovered


class TestInferFormatAutoDiscoverSlowPath:
    """Tests for infer_format() slow path that runs after _discover_for_type."""

    def test_infer_format_exact_match_after_auto_discover(self):
        """infer_format finds exact type match after auto-discover registers it."""

        class _InferType:
            pass

        original_registry = dict(AbstractDataset._registry)
        original_discovered = AbstractDataset._discovered.copy()

        class _InferDataset(AbstractDataset, types=_InferType, format="inferfmt"):
            def serialize(self, value, **options):
                return b""

            def deserialize(self, data, **options):
                return _InferType()

        # Remove registration so pre-discover path won't find it
        key = (_InferType, "inferfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        pickle_cls, pickle_key = None, None
        try:
            AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
            AbstractDataset._discovered.discard(_InferType.__module__.split(".")[0])

            # Remove (object, "pickle") so subclass fallback doesn't match first
            pickle_key = (object, "pickle")
            pickle_cls = AbstractDataset._registry.pop(pickle_key, None)

            fmt = AbstractDataset.infer_format(_InferType)
            assert fmt == "inferfmt"
        finally:
            AbstractDataset._discover_for_type = orig_discover
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            if pickle_cls is not None and pickle_key is not None:
                AbstractDataset._registry[pickle_key] = pickle_cls
            AbstractDataset._discovered = original_discovered


class TestTypeErrorBranches:
    """Tests for issubclass TypeError handling in various code paths."""

    def test_get_subclass_type_error_in_post_discover_loop(self):
        """get() handles TypeError from issubclass in post-discover slow path."""
        original_registry = dict(AbstractDataset._registry)
        original_discovered = AbstractDataset._discovered.copy()

        class _BadMeta(type):
            def __subclasscheck__(cls, subclass):
                raise TypeError("intentional")

        class _BadType(metaclass=_BadMeta):
            pass

        # Register a dataset with format "badfmt" for _BadType parent
        class _BadParent:
            pass

        class _BadDataset(AbstractDataset, types=_BadParent, format="badfmt"):
            def serialize(self, value, **options):
                return b""

            def deserialize(self, data, **options):
                return None

        # Remove the registration so fast path and normal subclass fail
        key = (_BadParent, "badfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            # Re-add it so the post-discover loop finds it
            cls._registry[key] = saved_cls

        try:
            AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
            AbstractDataset._discovered.discard("tests")

            # This should trigger the TypeError in issubclass in the post-discover loop
            with pytest.raises(ValueError, match="No dataset registered"):
                AbstractDataset.get(_BadType, "badfmt")
        finally:
            AbstractDataset._discover_for_type = orig_discover
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            AbstractDataset._discovered = original_discovered

    def test_infer_format_subclass_type_error(self):
        """infer_format handles TypeError from issubclass in the subclass loop."""
        original_registry = dict(AbstractDataset._registry)

        class _BadMeta(type):
            def __subclasscheck__(cls, subclass):
                raise TypeError("intentional")

        class _BadType2(metaclass=_BadMeta):
            pass

        # Remove (object, "pickle") to prevent it catching all types
        pickle_key = (object, "pickle")
        pickle_cls = AbstractDataset._registry.pop(pickle_key, None)

        try:
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_BadType2)
        finally:
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            if pickle_cls is not None:
                AbstractDataset._registry[pickle_key] = pickle_cls

    def test_infer_format_extension_subclass_type_error(self):
        """infer_format handles TypeError from issubclass in the extension hints loop."""
        original_hints = dict(AbstractDataset._extension_hints)
        original_registry = dict(AbstractDataset._registry)

        class _HintParent:
            pass

        class _BadMeta(type):
            """Metaclass that raises TypeError on issubclass."""

            def __instancecheck__(cls, instance):
                raise TypeError("intentional")

            def __subclasscheck__(cls, subclass):
                raise TypeError("intentional")

        class _BadHintType(metaclass=_BadMeta):
            pass

        # _BadHintType as hint type, but query type is _HintParent.
        # issubclass(_HintParent, _BadHintType) calls _BadMeta.__subclasscheck__  → TypeError
        AbstractDataset._extension_hints["badext"] = [(_BadHintType, "badfmt")]

        # Remove (object, "pickle") so fallback doesn't match
        pickle_key = (object, "pickle")
        pickle_cls = AbstractDataset._registry.pop(pickle_key, None)

        try:
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_HintParent, "badext")
        finally:
            AbstractDataset._extension_hints.clear()
            AbstractDataset._extension_hints.update(original_hints)
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            if pickle_cls is not None:
                AbstractDataset._registry[pickle_key] = pickle_cls


class TestInferFormatAutoDiscoverDisabled:
    """Tests for infer_format when auto-discover is disabled."""

    def test_infer_format_no_auto_discover_raises(self):
        """When auto_discover=False, infer_format doesn't try discovery."""
        original_auto = AbstractDataset._auto_discover
        original_registry = dict(AbstractDataset._registry)

        class _NoDiscType:
            pass

        # Remove (object, "pickle") so no fallback
        pickle_key = (object, "pickle")
        pickle_cls = AbstractDataset._registry.pop(pickle_key, None)

        try:
            AbstractDataset._auto_discover = False
            with pytest.raises(ValueError, match="No default format"):
                AbstractDataset.infer_format(_NoDiscType)
        finally:
            AbstractDataset._auto_discover = original_auto
            AbstractDataset._registry.clear()
            AbstractDataset._registry.update(original_registry)
            if pickle_cls is not None:
                AbstractDataset._registry[pickle_key] = pickle_cls
