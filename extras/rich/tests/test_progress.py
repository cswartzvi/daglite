import daglite_rich


def test_import_daglite_rich() -> None:
    assert daglite_rich.__version__  # type: ignore[attr-defined]
