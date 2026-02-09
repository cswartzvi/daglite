"""Unit tests for AbstractDataset registry, lookup, format inference, and plugin discovery."""

import pytest

from daglite.datasets.base import AbstractDataset
from daglite.exceptions import DatasetError


class _Marker:
    """Unique type that is never registered by built-in datasets."""

    pass


class _MarkerChild(_Marker):
    """Subclass of _Marker for testing subclass lookups."""

    pass


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the AbstractDataset class-level state around every test."""
    saved_registry = dict(AbstractDataset._registry)
    saved_hints = {k: list(v) for k, v in AbstractDataset._extension_hints.items()}
    saved_discovered = set(AbstractDataset._discovered)
    saved_auto = AbstractDataset._auto_discover

    yield

    AbstractDataset._registry.clear()
    AbstractDataset._registry.update(saved_registry)
    AbstractDataset._extension_hints.clear()
    AbstractDataset._extension_hints.update(saved_hints)
    AbstractDataset._discovered = saved_discovered
    AbstractDataset._auto_discover = saved_auto


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
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        assert (_Marker, "marker_fmt") in AbstractDataset._registry
        assert AbstractDataset._registry[(_Marker, "marker_fmt")] is MarkerDS

    def test_register_with_class_variables(self):
        """Subclass using class-variable style is registered."""

        class MarkerDS2(AbstractDataset):
            format = "marker_cv"
            types = (_Marker,)
            extensions = ("mrkv",)

            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        assert (_Marker, "marker_cv") in AbstractDataset._registry

    def test_abstract_intermediate_skips_registration(self):
        """Subclass without format acts as abstract intermediate (not registered)."""
        registry_before = dict(AbstractDataset._registry)

        class Intermediate(AbstractDataset):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
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
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return 0

        assert (int, "multi_fmt") in AbstractDataset._registry
        assert (float, "multi_fmt") in AbstractDataset._registry

    def test_extension_hints_populated(self):
        """Extension hints are populated when extensions are provided."""

        class ExtDS(
            AbstractDataset,
            format="ext_fmt",
            types=_Marker,
            extensions=("ext1", "ext2"),
        ):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        assert (_Marker, "ext_fmt") in AbstractDataset._extension_hints.get("ext1", [])
        assert (_Marker, "ext_fmt") in AbstractDataset._extension_hints.get("ext2", [])

    def test_extension_hints_shared_extension(self):
        """Two datasets sharing the same extension both appear in hints."""

        class MarkerDS1(
            AbstractDataset,
            format="shared1",
            types=_Marker,
            extensions="shx",
        ):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        class MarkerDS2(
            AbstractDataset,
            format="shared2",
            types=_Marker,
            extensions="shx",
        ):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        hints = AbstractDataset._extension_hints["shx"]
        assert (_Marker, "shared1") in hints
        assert (_Marker, "shared2") in hints

    def test_error_on_missing_types(self):
        """Raises DatasetError if neither class variable nor parameter provides types."""

        with pytest.raises(DatasetError):

            class BadDS(AbstractDataset, format="bad_fmt"):
                def serialize(self, obj):  # pragma: no cover
                    return b""

                def deserialize(self, data):  # pragma: no cover
                    return None

    def test_error_on_unrecognized_init_args(self):
        """Raises ValueError if unexpected args/kwargs are passed to constructor."""

        class NoArgsDS(AbstractDataset, format="noargs_fmt", types=_Marker):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        with pytest.raises(ValueError, match="does not accept parameters"):
            NoArgsDS("unexpected_arg")

        with pytest.raises(ValueError, match="does not accept parameters"):
            NoArgsDS(unexpected_kwarg=123)


class TestGet:
    """Tests for AbstractDataset.get() lookup."""

    def test_exact_match(self):
        """Exact (type, format) match returns correct dataset."""
        ds = AbstractDataset.get(str, "text")()
        assert ds.serialize("hello") == b"hello"

    def test_subclass_match(self):
        """Subclass of a registered type uses parent's dataset."""

        class MyStr(str):
            pass

        ds = AbstractDataset.get(MyStr, "text")()
        assert ds.deserialize(b"hi") == "hi"

    def test_object_fallback_for_pickle(self):
        """Pickle format with object registration covers arbitrary types."""
        ds = AbstractDataset.get(object, "pickle")()
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
        AbstractDataset._registry.pop((object, "pickle"), None)

        with pytest.raises(ValueError, match="No dataset registered"):
            AbstractDataset.get(_Marker, "nonexistent_format_xyz")

    def test_subclass_match_skips_non_matching_type(self):
        """get() skips format-matched entries where issubclass is False."""

        class _TypeA:
            pass

        class _TypeB:
            pass

        class _TypeBChild(_TypeB):
            pass

        class DS_A(AbstractDataset, format="shared_fmt", types=_TypeA):
            def serialize(self, obj):  # pragma: no cover
                return b"a"

            def deserialize(self, data):  # pragma: no cover
                return _TypeA()

        class DS_B(AbstractDataset, format="shared_fmt", types=_TypeB):
            def serialize(self, obj):
                return b"b"

            def deserialize(self, data):
                return _TypeB()

        # _TypeBChild is not _TypeA, so the loop skips DS_A, then finds DS_B
        ds = AbstractDataset.get(_TypeBChild, "shared_fmt")()
        assert ds.serialize(None) == b"b"

    def test_subclass_match_with_issubclass_type_error(self):
        """get() handles non-class types gracefully without crashing."""
        # Inject a non-class key directly to trigger TypeError in issubclass.
        # Remove (object, "pickle") so _Marker can't match via object fallback,
        # forcing the loop to iterate all entries and hit the bad one,
        # exercising the `except TypeError: continue` branch.
        AbstractDataset._registry.pop((object, "pickle"), None)
        AbstractDataset._registry[("not_a_class", "fake_fmt")] = AbstractDataset  # type: ignore[assignment]

        with pytest.raises(ValueError):
            AbstractDataset.get(_Marker, "fake_fmt")


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
        # Remove the object->pickle fallback so _Marker has no format
        AbstractDataset._registry.pop((object, "pickle"), None)

        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_Marker)

    def test_infer_with_issubclass_type_error(self):
        """infer_format handles non-class types in extension hints gracefully."""
        AbstractDataset._extension_hints["badext"] = [("not_a_class", "badfmt")]  # type: ignore[list-item]
        # Remove object->pickle so _Marker truly has no format
        AbstractDataset._registry.pop((object, "pickle"), None)

        # Should not crash, just skip the bad entry and raise
        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_Marker, "badext")


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
        AbstractDataset._registry.pop((object, "pickle"), None)

        formats = AbstractDataset.get_formats_for_type(_Marker)
        assert len(formats) == 0

    def test_non_class_type_in_registry(self):
        """Handles TypeError from issubclass gracefully."""
        # Inject a non-class key directly to trigger TypeError in issubclass
        AbstractDataset._registry[("not_a_class", "bad_fmt")] = AbstractDataset  # type: ignore[assignment]

        # Should not raise, just skip the bad entry
        formats = AbstractDataset.get_formats_for_type(_Marker)
        assert isinstance(formats, set)


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
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):  # pragma: no cover
                return _Marker()

        ext = AbstractDataset.get_extension(_Marker, "noext_fmt")
        assert ext is None


class TestLoadPlugins:
    """Tests for load_plugins() and _discover_for_type()."""

    def test_load_plugins_sets_auto_discover(self):
        """load_plugins(auto_discover=...) updates the class flag."""
        AbstractDataset.load_plugins(auto_discover=False)
        assert AbstractDataset._auto_discover is False
        AbstractDataset.load_plugins(auto_discover=True)
        assert AbstractDataset._auto_discover is True

    def test_load_plugins_no_names_loads_all(self):
        """Calling load_plugins() without names doesn't crash."""
        AbstractDataset.load_plugins()

    def test_load_plugins_specific_name(self):
        """Calling load_plugins('nonexistent') doesn't crash."""
        AbstractDataset.load_plugins("nonexistent_plugin_xyz123")

    def test_discover_for_type_no_op_when_already_discovered(self):
        """Once a module is discovered, _discover_for_type is a no-op."""
        AbstractDataset._discovered.add("builtins")
        # Should be a no-op on second call
        AbstractDataset._discover_for_type(int)

    def test_auto_discover_on_get(self):
        """Auto-discover is triggered on get() for unknown types."""
        # This just exercises the auto-discover code path without expecting
        # it to find anything for _Marker
        with pytest.raises(ValueError):
            AbstractDataset.get(_Marker, "nonexistent_123")

    def test_auto_discover_on_infer_format(self):
        """Auto-discover is triggered on infer_format() for unknown types."""
        # Remove object->pickle so _Marker truly has no format
        AbstractDataset._registry.pop((object, "pickle"), None)

        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_Marker)

    def test_infer_format_after_discovery_finds_type(self):
        """After _discover_for_type adds a registration, infer_format returns it."""
        # Pre-register _Marker during the discovery path
        AbstractDataset._registry[(_Marker, "discovered_fmt")] = type(
            "DiscoveredDS",
            (AbstractDataset,),
            {
                "format": "discovered_fmt",
                "types": (_Marker,),
                "extensions": (),
                "serialize": lambda self, v: b"",
                "deserialize": lambda self, d: _Marker(),
            },
        )
        # Clear discovered so infer_format falls through to the "after discovery" path
        AbstractDataset._discovered.discard(_Marker.__module__.split(".")[0])
        fmt = AbstractDataset.infer_format(_Marker)
        assert fmt == "discovered_fmt"

    def test_get_after_discovery_via_subclass(self):
        """get() finds a subclass match after auto-discovery."""

        class DiscoveredDS(AbstractDataset, format="disc_sub", types=_Marker, extensions="dsc"):
            def serialize(self, obj):  # pragma: no cover
                return b""

            def deserialize(self, data):
                return _MarkerChild()

        # Clear discovered so get() triggers auto-discovery path
        AbstractDataset._discovered.discard(_MarkerChild.__module__.split(".")[0])
        ds = AbstractDataset.get(_MarkerChild, "disc_sub")()
        result = ds.deserialize(b"")
        assert isinstance(result, _MarkerChild)

    def test_infer_format_subclass_no_extension_fallback(self):
        """infer_format falls back to subclass check in registry when no extension."""

        class ChildDict(dict):
            pass

        # dict has pickle registered; ChildDict should find it
        fmt = AbstractDataset.infer_format(ChildDict)
        assert fmt == "pickle"

    def test_infer_format_extension_hint_subclass_no_match(self):
        """infer_format tries extension hint subclass check, finds no match, falls through."""

        class _Unrelated:
            pass

        # Use a fresh extension with only an unrelated type
        AbstractDataset._extension_hints["xtest"] = [(_Unrelated, "nope")]

        # query for int with extension "xtest" -- issubclass(int, _Unrelated) is False
        # Falls through to registry check, int->pickle via object subclass
        fmt = AbstractDataset.infer_format(int, "xtest")
        assert fmt == "pickle"


class TestGetAutoDiscoverSlowPath:
    """Tests for get() slow path that runs after _discover_for_type."""

    def test_exact_key_match_after_auto_discover(self):
        """get() finds exact key match after auto-discover registers a dataset."""

        class _DiscType:
            pass

        class _DiscDataset(AbstractDataset, types=_DiscType, format="disc"):
            def serialize(self, obj):
                return b"disc"

            def deserialize(self, data):
                return _DiscType()

        # Remove the registration so the fast path won't find it
        key = (_DiscType, "disc")
        saved_cls = AbstractDataset._registry.pop(key)

        # Mock _discover_for_type to re-register on discovery
        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
        AbstractDataset._discovered.discard(_DiscType.__module__.split(".")[0])

        ds = AbstractDataset.get(_DiscType, "disc")()
        assert ds.serialize(None) == b"disc"

        # Restore original discover method (fixture handles registry/discovered)
        AbstractDataset._discover_for_type = orig_discover

    def test_subclass_match_after_auto_discover(self):
        """get() finds subclass match after auto-discover registers a parent dataset."""

        class _DiscParent:
            pass

        class _DiscChild(_DiscParent):
            pass

        class _DiscParentDataset(AbstractDataset, types=_DiscParent, format="discfmt"):
            def serialize(self, obj):
                return b"parent"

            def deserialize(self, data):
                return _DiscParent()

        # Remove the registration so subclass check pre-discover won't find it
        key = (_DiscParent, "discfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
        AbstractDataset._discovered.discard(_DiscChild.__module__.split(".")[0])

        ds = AbstractDataset.get(_DiscChild, "discfmt")()
        assert ds.serialize(None) == b"parent"

        AbstractDataset._discover_for_type = orig_discover


class TestInferFormatAutoDiscoverSlowPath:
    """Tests for infer_format() slow path that runs after _discover_for_type."""

    def test_infer_format_exact_match_after_auto_discover(self):
        """infer_format finds exact type match after auto-discover registers it."""

        class _InferType:
            pass

        class _InferDataset(AbstractDataset, types=_InferType, format="inferfmt"):
            def serialize(self, obj):
                return b""

            def deserialize(self, data):
                return _InferType()

        # Remove registration so pre-discover path won't find it
        key = (_InferType, "inferfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            cls._registry[key] = saved_cls

        AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
        AbstractDataset._discovered.discard(_InferType.__module__.split(".")[0])

        # Remove (object, "pickle") so subclass fallback doesn't match first
        AbstractDataset._registry.pop((object, "pickle"), None)

        fmt = AbstractDataset.infer_format(_InferType)
        assert fmt == "inferfmt"

        AbstractDataset._discover_for_type = orig_discover


class TestTypeErrorBranches:
    """Tests for issubclass TypeError handling in various code paths."""

    def test_get_subclass_type_error_in_post_discover_loop(self):
        """get() handles TypeError from issubclass in post-discover slow path."""

        class _BadMeta(type):
            def __subclasscheck__(cls, subclass):
                raise TypeError("intentional")

        class _BadType(metaclass=_BadMeta):
            pass

        # Register a dataset with format "badfmt" for _BadType parent
        class _BadParent:
            pass

        class _BadDataset(AbstractDataset, types=_BadParent, format="badfmt"):
            def serialize(self, obj):
                return b""

            def deserialize(self, data):
                return None

        # Remove the registration so fast path and normal subclass fail
        key = (_BadParent, "badfmt")
        saved_cls = AbstractDataset._registry.pop(key)

        orig_discover = AbstractDataset._discover_for_type

        @classmethod
        def fake_discover(cls, type_):
            # Re-add it so the post-discover loop finds it
            cls._registry[key] = saved_cls

        AbstractDataset._discover_for_type = fake_discover  # type: ignore[method-assign]
        AbstractDataset._discovered.discard("tests")

        # This should trigger the TypeError in issubclass in the post-discover loop
        with pytest.raises(ValueError, match="No dataset registered"):
            AbstractDataset.get(_BadType, "badfmt")

        AbstractDataset._discover_for_type = orig_discover

    def test_infer_format_subclass_type_error(self):
        """infer_format handles TypeError from issubclass in the subclass loop."""

        class _BadMeta(type):
            def __subclasscheck__(cls, subclass):
                raise TypeError("intentional")

        class _BadType2(metaclass=_BadMeta):
            pass

        # Remove (object, "pickle") to prevent it catching all types
        AbstractDataset._registry.pop((object, "pickle"), None)

        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_BadType2)

    def test_infer_format_extension_subclass_type_error(self):
        """infer_format handles TypeError from issubclass in the extension hints loop."""

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
        # issubclass(_HintParent, _BadHintType) calls _BadMeta.__subclasscheck__ -> TypeError
        AbstractDataset._extension_hints["badext"] = [(_BadHintType, "badfmt")]

        # Remove (object, "pickle") so fallback doesn't match
        AbstractDataset._registry.pop((object, "pickle"), None)

        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_HintParent, "badext")


class TestInferFormatAutoDiscoverDisabled:
    """Tests for infer_format when auto-discover is disabled."""

    def test_infer_format_no_auto_discover_raises(self):
        """When auto_discover=False, infer_format doesn't try discovery."""

        class _NoDiscType:
            pass

        # Remove (object, "pickle") so no fallback
        AbstractDataset._registry.pop((object, "pickle"), None)
        AbstractDataset._auto_discover = False

        with pytest.raises(ValueError, match="No default format"):
            AbstractDataset.infer_format(_NoDiscType)
