from daglite.cli._shared import discover_workflows


def list_workflows(root: str) -> None:
    """List discovered workflows under a module or package."""
    try:
        workflows = discover_workflows(root)
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(2) from e

    if not workflows:
        print(f"No workflows found under {root!r}.")
        return

    for import_path, meta in workflows:
        print(f"{import_path}   (name={meta.name!r})")
