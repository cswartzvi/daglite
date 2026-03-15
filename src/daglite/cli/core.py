from __future__ import annotations

import sys

from cyclopts import App


def main(argv: list[str] | None = None) -> None:
    """Entry point for the Daglite CLI."""

    import daglite
    from daglite.cli._shared import HELP_FLAGS
    from daglite.cli._shared import print_run_error
    from daglite.cli.cmds.cmd_describe import describe_workflow
    from daglite.cli.cmds.cmd_list import list_workflows
    from daglite.cli.cmds.cmd_run import run_help
    from daglite.cli.cmds.cmd_run import run_workflow

    app = App(
        name="daglite",
        help="Daglite - Lightweight Python framework for building static DAGs.",
        help_flags=["-h", "--help"],
        version=daglite.__version__,
        version_flags=["--version"],
    )

    app.command(list_workflows, name="list")
    app.command(describe_workflow, name="describe")
    app.command(run_help, name="run")

    argv = list(sys.argv[1:] if argv is None else argv)

    # Special handling for 'run': the first positional arg is the workflow
    # target and everything after it becomes workflow-specific arguments
    # parsed by a per-workflow cyclopts App (see build_workflow_app).
    if argv and argv[0] == "run":
        if len(argv) == 1 or argv[1] in HELP_FLAGS:
            app(["run", "--help"])
            return

        target = argv[1]
        wf_args = argv[2:]
        try:
            run_workflow(target, wf_args)
        except Exception as e:
            print_run_error(e)
            raise SystemExit(2)
        return

    app(argv)


if __name__ == "__main__":
    main()
