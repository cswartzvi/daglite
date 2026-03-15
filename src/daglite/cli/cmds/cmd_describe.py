from daglite.cli._shared import build_workflow_app
from daglite.cli._shared import discover_workflows
from daglite.cli._shared import get_workflow
from daglite.cli._shared import normalize_tokens
from daglite.cli._shared import validate_workflow_args


def describe_workflow(target: str, *rest: str):
    """Describe a workflow target.

    Targets can be specified as:
      - module.sub_module:workflow_func
      - module.sub_module.workflow_func
      - path/to/module.py:workflow_func

    You can also pass workflow arguments to preview how they are parsed:
      - daglite describe module:workflow -- --param1 value1 --param2 value2
    """  # noqa: D213
    try:
        _print_describe_help(target)
    except Exception as e:
        _print_workflow_suggestions(target)
        if ":" not in target and "." not in target:
            print(f"Tip: use 'daglite list {target}' to discover workflows.")
        else:
            print(f"Error: {e}")
        raise SystemExit(2) from e

    tokens = normalize_tokens(rest)
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]

    if tokens:
        try:
            bound = validate_workflow_args(target, tokens)
        except Exception as e:
            print("\nArgument validation failed:")
            print(f"  {e}")
            raise SystemExit(2) from e

        print("\nParsed values (validated & converted):")
        for name, value in bound.arguments.items():
            print(f"  {name} = {value!r} ({type(value).__name__})")


def _print_describe_help(target: str) -> None:
    workflow_obj = get_workflow(target)
    wf_app = build_workflow_app(workflow_obj, target, show_meta_flags=False)
    wf_app.help_print()


def _print_workflow_suggestions(root: str) -> None:
    try:
        workflows = discover_workflows(root)
    except ModuleNotFoundError:
        workflows = []

    if not workflows:
        return

    print(f"No workflow target was found for {root!r}.")
    print("Did you mean one of these?")
    for import_path, meta in workflows:
        print(f"  {import_path}   (name={meta.name!r})")
